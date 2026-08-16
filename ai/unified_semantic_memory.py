"""
الذاكرة الدلالية الموحدة للوكلاء (Unified Semantic Memory)
==========================================================
طبقة توحيد واحدة تمنح كل وكلاء NSM وصولًا مشتركًا إلى الذاكرة الدلالية،
بدل التشتت بين ثلاث وحدات منفصلة كان لكل منها ناقلها الخاص:

  • QdrantSemanticMemory (ai/qdrant_semantic_memory.py) — محادثات المحادثة
    (مجموعة nsm_conversations)
  • ReflectionMemory (ai/reflection_memory.py) — دروس الأخطاء والنجاحات
  • SharedKnowledge (ai/shared_knowledge.py) — معارف المهام المشتركة
    (مجموعة nsm_skb)

الوحدة الجديدة تقدم واجهة موحدة واحدة تستخدمها جميع الوكلاء (agent_loop،
terminal، notebook، موحدات المهام):

  • add_finding(...)      — حفظ معلومة/نتيجة/درس دلالي (تنقله لكل الطبقات)
  • search(...)           — بحث دلالي موحّد يجمع نتائج المحادثات + الدروس
                            + المعارف المشتركة مرتّبة بالدرجة
  • agent_recall(...)     — تذكّر سياق خاص بوكيل (حفظ + بحث معاً)
  • summary(...)          — إحصاء الذاكرة الموحدة للعرض في اللوحات

القواعد:
  - لا تستدعي أي API خارجي إلزامي: Qdrant (bge-m3 عربي) يُفعَّل فقط إن وُجدت
    المفاتيح (QDRANT_URL/QDRANT_API_KEY) والمكتبة؛ وإلا يُستخدم البحث المحلي
    (TF عربي في SQLite) دون أي أثر على بقية النظام.
  - كل فشل جزئي معزول: وحدة واحدة معطلة لا تسقِط الباقي.
  - لا تغيّر أي سلوك قائم: الوحدات القديمة تُستدعى كما هي.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("UnifiedSemanticMemory")

_ROOT = Path(__file__).resolve().parent.parent
_LOCAL_DB = _ROOT / "data" / "unified_semantic_memory.db"

_INIT_SQL = """
CREATE TABLE IF NOT EXISTS findings (
    id        TEXT PRIMARY KEY,
    agent_id  TEXT NOT NULL,
    kind      TEXT NOT NULL,          -- finding | reflection | conversation
    text      TEXT NOT NULL,
    payload   TEXT NOT NULL,
    score     REAL NOT NULL DEFAULT 0,
    ts        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usm_ts ON findings(ts);
"""

_LOCAL_LIMIT = 25     # حد النتائج المحلية قبل دمجها مع الطبقات الخارجية
_UNIFIED_LIMIT = 5    # عدد النتائج النهائي الموحد


def _normalize(text: str) -> str:
    """تطبيع عربي للبحث المحلي (متوافق مع qdrant_semantic_memory._normalize)."""
    import re
    if not text:
        return ""
    t = str(text).lower().strip()
    t = re.sub(r"[أإآا]", "ا", t)
    t = t.replace("ة", "ه").replace("ى", "ي")
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = re.sub(r"[^\u0600-\u06FFa-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _local_score(query: str, text: str) -> float:
    """درجة ترتيب TF بسيطة للنص العربي (بدون أي مكتبات خارجية)."""
    qw = [w for w in _normalize(query).split() if w]
    tw = [w for w in _normalize(text).split() if w]
    if not qw or not tw:
        return 0.0
    tc = {}
    for w in tw:
        tc[w] = tc.get(w, 0) + 1
    hits = sum(tc.get(w, 0) for w in qw)
    if not hits:
        return 0.0
    return hits / max(len(tw), 1)


class UnifiedSemanticMemory:
    """واجهة الذاكرة الدلالية الموحدة — Qdrant أولًا ثم SQLite محلي دائمًا."""

    def __init__(self, local_db: str = str(_LOCAL_DB)) -> None:
        self._path = local_db
        self._db_conn = None
        self._qdrant_conv = None  # ai.qdrant_semantic_memory.QdrantSemanticMemory (كسول)
        self._qdrant_conv_tried = False
        self._sk = None           # ai.shared_knowledge.SharedKnowledgeBase (كسول)
        self._sk_tried = False

    # ── طبقة محلية ──────────────────────────────────────────────────────────

    def _get_conn(self):
        if self._db_conn is None:
            os.makedirs(os.path.dirname(self._path), exist_ok=True)
            import sqlite3
            self._db_conn = sqlite3.connect(self._path)
            for stmt in _INIT_SQL.split(";"):
                if stmt.strip():
                    self._db_conn.execute(stmt)
            self._db_conn.commit()
        return self._db_conn

    def _id_for(self, agent_id: str, kind: str, text: str) -> str:
        h = hashlib.sha1(f"{agent_id}|{kind}|{text}".encode("utf-8")).hexdigest()[:16]
        return f"usm_{kind}_{h}"

    # ── الطبقات الخارجية (كسولة ومعزولة عن الأخطاء) ────────────────────────

    def _conv_mem(self):
        """QdrantSemanticMemory — طبقة محادثات Qdrant (اختيارية)."""
        if self._qdrant_conv_tried:
            return self._qdrant_conv
        self._qdrant_conv_tried = True
        try:
            from ai.qdrant_semantic_memory import QdrantSemanticMemory
            self._qdrant_conv = QdrantSemanticMemory()
        except Exception as exc:
            logger.debug("unified: qdrant_conv unavailable: %s", exc)
            self._qdrant_conv = None
        return self._qdrant_conv

    def _shared_kb(self):
        """SharedKnowledgeStore — طبقة معارف المهام المشتركة (اختيارية)."""
        if self._sk_tried:
            return self._sk
        self._sk_tried = True
        try:
            from ai.shared_knowledge import get_skb
            self._sk = get_skb()
        except Exception as exc:
            logger.debug("unified: shared_kb unavailable: %s", exc)
            self._sk = None
        return self._sk

    # ── الحفظ ───────────────────────────────────────────────────────────────

    def add_finding(self, agent_id: str, kind: str = "finding",
                    text: str = "", tool: str = "", source: str = "",
                    payload: Optional[Dict[str, Any]] = None) -> bool:
        """حفظ معلومة دلالية في الذاكرة الموحدة (كل الطبقات المتاحة).

        kind: finding (نتيجة مهمة) | reflection (درس خطأ/نجاح) |
              conversation (محادثة).
        """
        if not text:
            return False
        fid = self._id_for(agent_id, kind, text)
        stored_any = False
        # 1) المحلية دائمًا
        try:
            self._get_conn().execute(
                "INSERT OR REPLACE INTO findings VALUES (?,?,?,?,?,?,?)",
                (fid, agent_id or "main", kind, text[:2000],
                 json.dumps(payload or {}, ensure_ascii=False, default=str),
                 _local_score(text, text), time.time()),
            )
            self._get_conn().commit()
            stored_any = True
        except Exception as exc:
            logger.warning("unified local store failed: %s", exc)
        # 2) Qdrant إن كانت مفعّلة
        cm = self._conv_mem()
        if cm is not None and bool(getattr(cm, "active", False)):
            try:
                # واجهة QdrantSemanticMemory.add_conversation(convo_id, user_text, assistant_text, relevance)
                cm.add_conversation(
                    fid, text[:500],
                    (payload.get("extra") or "")[:500],
                    float(payload.get("relevance") or 0.0))
                stored_any = True
            except Exception as exc:
                logger.debug("unified qdrant add failed: %s", exc)
        # 3) معرف المهام المشتركة (SharedKnowledgeBase.share_finding)
        sk = self._shared_kb()
        if sk is not None:
            try:
                res = sk.share_finding(
                    (payload or {}).get("task_id") or "",
                    (payload or {}).get("role") or (agent_id or "main"),
                    text[:2000], tool=tool[:60], source=source[:120])
                if res and res.get("ok"):
                    stored_any = True
            except Exception as exc:
                logger.debug("unified skb add failed: %s", exc)
        return stored_any

    # ── البحث الموحد ────────────────────────────────────────────────────────

    def search(self, query: str, agent_id: str = "",
               limit: int = _UNIFIED_LIMIT) -> List[Tuple[float, Dict[str, Any]]]:
        """بحث دلالي موحّد عبر كل الطبقات — يعيد (درجة، {id, kind, text, payload})."""
        results: List[Tuple[float, Dict[str, Any]]] = []

        # محلية (SQLite TF عربي)
        try:
            c = self._get_conn()
            rows = c.execute(
                "SELECT id, agent_id, kind, text, payload FROM findings "
                "ORDER BY ts DESC LIMIT ?", (_LOCAL_LIMIT,)).fetchall()
            for row in rows:
                sc = _local_score(query, row[3])
                if sc > 0:
                    try:
                        payload = json.loads(row[4]) if row[4] else {}
                    except Exception:
                        payload = {}
                    results.append((sc, {
                        "id": row[0], "agent_id": row[1], "kind": row[2],
                        "text": row[3], "payload": payload, "backend": "local",
                    }))
        except Exception as exc:
            logger.debug("unified local search failed: %s", exc)

        # طبقة المحادثات Qdrant إن توفرت
        cm = self._conv_mem()
        cm_active = cm is not None and bool(getattr(cm, "active", False))
        if cm_active:
            try:
                for sc, pl in cm.search_conversations(query, limit=3):
                    results.append((sc * 0.95, {
                        "id": pl.get("id", ""), "agent_id": pl.get("agent_id", "main"),
                        "kind": "conversation",
                        "text": (pl.get("user_text") or "") + " — " + (pl.get("assistant_text") or ""),
                        "payload": pl, "backend": "qdrant_conv",
                    }))
            except Exception as exc:
                logger.debug("unified qdrant search failed: %s", exc)

        # طبقة معرف المهام المشتركة إن توفرت
        sk = self._shared_kb()
        if sk is not None:
            try:
                for pl in (sk.query_knowledge(query, k=3) or []):
                    results.append((float(pl.get("score") or 0) * 0.9, {
                        "id": str(pl.get("fid") or ""),
                        "agent_id": str(pl.get("role", "")),
                        "kind": "finding",
                        "text": pl.get("text") or "",
                        "payload": pl, "backend": "shared_kb",
                    }))
            except Exception as exc:
                logger.debug("unified skb search failed: %s", exc)

        results.sort(key=lambda r: r[0], reverse=True)
        return results[:limit]

    # ── تذكّر الوكيل ────────────────────────────────────────────────────────

    def agent_recall(self, agent_id: str, query: str,
                     extra: Optional[str] = None) -> str:
        """حفظ + استحضار — الدالة التي يستخدمها الوكيل مباشرة قبل مهمة جديدة.
        يعيد نصًا عربيًا جاهزًا للإدراج في سياق الاستدعاء."""
        if extra:
            self.add_finding(agent_id, "finding", extra)
        hits = self.search(query, agent_id=agent_id)
        if not hits:
            return ""
        lines = ["## 🧠 تذكّر من الذاكرة الدلالية الموحدة:"]
        for sc, pl in hits[:3]:
            lines.append(f"- [{pl.get('kind')}] ({sc:.2f}) {pl.get('text', '')[:200]}")
        return "\n".join(lines)

    # ── إحصاءات ─────────────────────────────────────────────────────────────

    def summary(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {"finding": 0, "reflection": 0, "conversation": 0}
        try:
            for row in self._get_conn().execute("SELECT kind, COUNT(*) FROM findings GROUP BY kind"):
                counts[row[0]] = int(row[1])
        except Exception:
            pass
        _cm = self._conv_mem()
        _cm_active = bool(_cm) and bool(getattr(_cm, "active", False))
        return {
            "total_findings": sum(counts.values()),
            "counts": counts,
            "qdrant_active": _cm_active,
        }


# ── Singleton ─────────────────────────────────────────────────────────────────

_instance: Optional[UnifiedSemanticMemory] = None


def get_unified_memory() -> UnifiedSemanticMemory:
    global _instance
    if _instance is None:
        _instance = UnifiedSemanticMemory()
    return _instance
