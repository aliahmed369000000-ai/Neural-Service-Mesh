"""
Shared Knowledge Base (SKB) — ناقل معرفة مشترك للوكلاء المتعاونين
==================================================================
يُمكّن أدوار فريق المهمة التعاونية من مشاركة نتائجها واستحضار ما عرفه
الزملاء لحظةً بلحظة، عبر قناتين متكاملتين:

  1) Qdrant Cloud (الطبقة الغنية): بحث دلالي حقيقي بالعربية عبر
     Cloudflare bge-m3 — الأدوار تجد معرفة الزملاء حتى لو صيغت بعبارات
     مختلفة عن سؤالها (مثل «بصريات ابن الهيثم» ↔ «المناظر»).
  2) SQLite محلي (fallback آمن): إن غابت المكتبة أو المفاتيح أو الاتصال
     يعمل الناقل محليًا بحثًا حرفيًا صامتًا دون أي انقطاع.

يُستدعى تلقائيًا داخل دورة دور التعاون: كل نتيجة بحث/جلب ناجحة تُشارك
في الناقل (`share_finding`)، وقبل كل بحث يستحضر الدور ما وجده زملاؤه
(`query_knowledge`). كما يُثري اجتماع التجميع بقسم «معارف الفريق».

تدهور آمن كامل: كل مسار wrapped في try/except — أي فشل Qdrant يتحول
للطبقة المحلية بصمت، وفشل الطبقة المحلية يُبتلَع هو الآخر.
لا يتطلب هذا النظام تثبيت أي شيء جديد في بيئة الإنتاج (qdrant-client
موجود أصلًا في requirements.txt منذ إضافة vector_backend).
"""
from __future__ import annotations

import contextlib
import datetime
import json
import logging
import os
import re
import sqlite3
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("SharedKnowledge")

# ── إعدادات Qdrant ────────────────────────────────────────────────────────
_SKB_COLLECTION = "nsm_shared_knowledge"   # مجموعة مخصصة للناقل المشترك
_EMBED_DIM = 1024                          # أبعاد Cloudflare bge-m3
_EMBED_MODEL = "@cf/baai/bge-m3"
_MAX_FINDING_CHARS = 2500                  # حد حجم كل معرفة مشاركة
_MAX_PER_TASK = 80                         # سقف معارف لكل مهمة تعاونية
_SEARCH_LIMIT_DEFAULT = 8                  # كم نتيجة يستحضر الدور قبل بحثه
_FRESHNESS_DAYS = 7                        # عمر المعرفة بعد توهينها
_LOCAL_DB = "data/shared_knowledge.db"


# ══════════════════════════════════════════════════════════════════
# المغلّف المتجهي (نفس نمط vector_backend — reuse لا تكرار)
# ══════════════════════════════════════════════════════════════════

class _Embedder:
    """توليد متجهات للعربية عبر Cloudflare bge-m3 مع فشل صامت."""

    def __init__(self):
        self._account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
        self._token = os.getenv("CF_API_TOKEN", "").strip()
        self._tried = False
        self._ok = False

    def available(self) -> bool:
        if self._tried:
            return self._ok
        self._tried = True
        if not self._account_id or not self._token:
            return False
        from ai.llm_fallback import _post_json
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{self._account_id}/ai/run/{_EMBED_MODEL}")
        try:
            data = _post_json(url, {"text": ["اختبار"]},
                              {"Authorization": f"Bearer {self._token}",
                               "Content-Type": "application/json"}, 12)
            vecs = ((data or {}).get("result", {}) or {}).get("data", [])
            self._ok = bool(vecs and len(vecs[0]) >= 512)
            if self._ok:
                logger.info("SKB: التغليف المتجهي العربي مفعّل (bge-m3)")
        except Exception as e:
            logger.debug(f"SKB embedder unavailable: {e}")
            self._ok = False
        return self._ok

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.available() or not (text or "").strip():
            return None
        from ai.llm_fallback import _post_json
        url = (f"https://api.cloudflare.com/client/v4/accounts/"
               f"{self._account_id}/ai/run/{_EMBED_MODEL}")
        try:
            data = _post_json(url, {"text": [(text or "")[:2000]]},
                              {"Authorization": f"Bearer {self._token}",
                               "Content-Type": "application/json"}, 15)
            vecs = ((data or {}).get("result", {}) or {}).get("data", [])
            return vecs[0] if vecs else None
        except Exception as e:
            logger.debug(f"SKB embed error: {e}")
            return None


# ══════════════════════════════════════════════════════════════════
# الطبقة المحلية (SQLite) — البحث الحرفي الآمن
# ══════════════════════════════════════════════════════════════════

_LOCAL_INIT = """
CREATE TABLE IF NOT EXISTS skb_findings (
    id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    role TEXT NOT NULL,
    text TEXT NOT NULL,
    tool TEXT NOT NULL,
    source TEXT,
    ts REAL NOT NULL,
    searchable TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_skb_task ON skb_findings(task_id);
CREATE INDEX IF NOT EXISTS idx_skb_ts ON skb_findings(ts);
"""


class _LocalStore:
    """بحث حرفي محلي يعمل دائمًا كطبقة أخيرة."""

    def __init__(self, db_path: str = _LOCAL_DB):
        self._db = db_path
        self._lock = threading.Lock()
        self._ensure_init()

    def _ensure_init(self):
        try:
            os.makedirs(os.path.dirname(self._db) or ".", exist_ok=True)
            with sqlite3.connect(self._db, timeout=30) as conn:
                conn.executescript(_LOCAL_INIT)
        except Exception as e:
            logger.debug(f"SKB local init error: {e}")

    def add(self, fid: str, task_id: str, role: str, text: str,
            tool: str, source: str, ts: float) -> bool:
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                exists = conn.execute(
                    "SELECT 1 FROM skb_findings WHERE id=?", (fid,)
                ).fetchone() is not None
                if exists:
                    return False
                conn.execute(
                    "INSERT OR REPLACE INTO skb_findings "
                    "(id, task_id, role, text, tool, source, ts, searchable) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (fid, task_id, role, text[:_MAX_FINDING_CHARS],
                     tool, source, ts,
                     _normalize(text[:_MAX_FINDING_CHARS])),
                )
            return True
        except Exception as e:
            logger.debug(f"SKB local add error: {e}")
            return False

    def search(self, query: str, task_id: Optional[str] = None,
               k: int = _SEARCH_LIMIT_DEFAULT) -> List[Tuple[float, Dict[str, Any]]]:
        """بحث حرفي محلي خالص: لا يعتمد أبدًا على التضمين أو Qdrant."""
        try:
            if not (query or "").strip():
                return []
            q_words = [w for w in _normalize(query).split() if len(w) >= 2]
            if not q_words:
                return []
            # مطابقة مرنة: القالب المرتّب OR كل كلمة منفردة (يغطي
            # الاستعلامات العربية الطويلة غير المرتّبة في النص)
            conds = ["searchable LIKE ?"] + [
                "searchable LIKE ?" for _ in q_words]
            params = ["%" + _normalize(query)[:200].replace(" ", "%") + "%"] + [
                "%" + w + "%" for w in q_words]
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                rows = conn.execute(
                    "SELECT id, task_id, role, text, tool, source, ts "
                    "FROM skb_findings WHERE (" + " OR ".join(conds) + ") "
                    "ORDER BY ts DESC LIMIT ?",
                    params + [k * 3]).fetchall()
        except Exception as e:
            logger.debug(f"SKB local search error: {e}")
            return []
        scored: List[Tuple[float, Dict[str, Any]]] = []
        for fid, task_id_, role, text, tool, source, ts in rows:
            if task_id and task_id_ != task_id:
                continue
            t = _normalize(text)
            hits = sum(1 for w in q_words if w in t)
            score = hits / len(q_words)
            if score > 0:
                scored.append((score, {
                    "id": fid, "task_id": task_id_, "role": role,
                    "text": text, "tool": tool, "source": source,
                    "ts": ts, "backend": "local",
                }))
        scored.sort(key=lambda x: (-x[0], -(x[1].get("ts") or 0)))
        return scored[:k]

    def prune(self, max_age_days: int = _FRESHNESS_DAYS) -> int:
        try:
            cutoff = _now_ts() - max_age_days * 86400
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                cur = conn.execute("DELETE FROM skb_findings WHERE ts < ?",
                                   (cutoff,))
                conn.commit()
            return cur.rowcount
        except Exception:
            return 0

    def recent(self, k: int = 10) -> List[Tuple[float, Dict[str, Any]]]:
        """أحدث المعارف دون استعلام (للوحة المراقبة)."""
        try:
            with self._lock, sqlite3.connect(self._db, timeout=30) as conn:
                rows = conn.execute(
                    "SELECT task_id, role, text, tool, source, ts "
                    "FROM skb_findings ORDER BY ts DESC LIMIT ?", (k,)
                ).fetchall()
        except Exception:
            return []
        return [(1.0, {
            "task_id": r[0], "role": r[1], "text": r[2],
            "tool": r[3], "source": r[4], "ts": r[5], "backend": "local",
        }) for r in rows]

    def count(self, task_id: Optional[str] = None) -> int:
        try:
            with sqlite3.connect(self._db, timeout=30) as conn:
                if task_id:
                    return conn.execute(
                        "SELECT COUNT(*) FROM skb_findings WHERE task_id=?",
                        (task_id,)).fetchone()[0]
                return conn.execute(
                    "SELECT COUNT(*) FROM skb_findings").fetchone()[0]
        except Exception:
            return 0


# ══════════════════════════════════════════════════════════════════
# طبقة Qdrant (الطبقة الغنية — اختيارية بالكامل)
# ══════════════════════════════════════════════════════════════════

class _QdrantLayer:
    """تُفعَّل فقط إن: مكتبة مثبتة + مفاتيح + اتصال ناجح. أي فشل → محلية."""

    def __init__(self):
        self._client = None
        self._embedder = _Embedder()
        self._tried = False
        self._ok = False

    def _ensure(self) -> bool:
        if self._tried:
            return self._ok
        self._tried = True
        url = os.getenv("QDRANT_URL", "").strip()
        key = os.getenv("QDRANT_API_KEY", "").strip()
        if not url or not key:
            return False
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            client = QdrantClient(url=url, api_key=key, timeout=8,
                                  check_compatibility=False)
            existing = [c.name for c in client.get_collections().collections]
            if _SKB_COLLECTION not in existing:
                client.create_collection(
                    collection_name=_SKB_COLLECTION,
                    vectors_config=VectorParams(
                        size=_EMBED_DIM, distance=Distance.COSINE),
                )
            self._client = client
            self._ok = True
            logger.info(
                "SKB: Qdrant متصل — الناقل المتجهي المشترك مفعّل "
                "(bge-m3 عربي)")
        except Exception as e:
            logger.debug(f"SKB Qdrant unavailable: {e}")
            self._ok = False
        return self._ok

    def available(self) -> bool:
        return self._ensure()

    def add(self, fid: str, task_id: str, role: str, text: str,
            tool: str, source: str, ts: float) -> bool:
        if not self.available():
            return False
        vec = self._embedder.embed(text)
        if not vec:
            return False
        try:
            from qdrant_client.models import PointStruct
            self._client.upsert(
                collection_name=_SKB_COLLECTION,
                points=[PointStruct(
                    id=fid,
                    vector=vec,
                    payload={
                        "task_id": task_id, "role": role,
                        "text": text[:_MAX_FINDING_CHARS],
                        "tool": tool, "source": source, "ts": ts,
                        "backend": "qdrant",
                    },
                )],
            )
            return True
        except Exception as e:
            logger.debug(f"SKB qdrant add error: {e}")
            return False

    def search(self, query: str, task_id: Optional[str] = None,
               k: int = _SEARCH_LIMIT_DEFAULT) -> List[Tuple[float, Dict[str, Any]]]:
        if not self.available():
            return []
        vec = self._embedder.embed(query)
        if not vec:
            return []
        try:
            from qdrant_client.models import (
                FieldCondition, Filter, MatchValue, Range)
            conditions = [
                FieldCondition(key="ts",
                               range=Range(gte=_now_ts() -
                                           _FRESHNESS_DAYS * 86400)),
            ]
            if task_id:
                conditions.append(
                    FieldCondition(key="task_id",
                                   match=MatchValue(value=task_id)))
            qfilter = Filter(must=conditions)
            if hasattr(self._client, "query_points"):
                resp = self._client.query_points(
                    collection_name=_SKB_COLLECTION, query=vec,
                    limit=k, query_filter=qfilter,
                )
                hits = resp.points
            else:
                hits = self._client.search(
                    collection_name=_SKB_COLLECTION, query_vector=vec,
                    limit=k, query_filter=qfilter,
                )
            return [(float(h.score), h.payload or {}) for h in hits]
        except Exception as e:
            logger.debug(f"SKB qdrant search error: {e}")
            return []

    def stats(self) -> Dict[str, Any]:
        if not self.available():
            return {"active": False}
        try:
            info = self._client.get_collection(_SKB_COLLECTION)
            return {
                "active": True,
                "points_count": int(getattr(info, "points_count", None) or 0),
            }
        except Exception:
            return {"active": True, "points_count": None}


# ══════════════════════════════════════════════════════════════════
# الواجهة العامة
# ══════════════════════════════════════════════════════════════════

def _now_ts() -> float:
    return datetime.datetime.now().timestamp()


def _fid(task_id: str, role: str, index: int) -> str:
    """معرّف ثابت لكل معرفة (نفس المدخل = نفس النقطة تُحدَّث)."""
    key = f"skb:{task_id}:{role}:{index}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _normalize(text: str) -> str:
    """تطبيع النص للبحث الحرفي المحلي."""
    t = re.sub(r"[\u064B-\u0652\u0640]", "", (text or ""))
    t = re.sub(r"[إأآا]", "ا", t)
    t = re.sub(r"[ة]", "ه", t)
    t = re.sub(r"[ى]", "ي", t)
    return re.sub(r"[^\w\u0600-\u06FF]+", " ", t).strip()


class SharedKnowledgeBase:
    """ناقل معرفة مشترك: Qdrant (دلالة عربية) + SQLite محلي (fallback)."""

    def __init__(self) -> None:
        self._local = _LocalStore()
        self._qdrant = _QdrantLayer()

    # ── مشاركة ───────────────────────────────────────────────────────
    def share_finding(self, task_id: str, role: str, finding: str,
                      tool: str = "web_search", source: str = "",
                      index: int = 0) -> Dict[str, Any]:
        """يشارك الدور نتيجة بحثه/اكتشافه مع فريق المهمة.

        يرجع dict {ok, backend} — backend في ('qdrant', 'local', 'none')."""
        if not task_id or not role or not finding or not finding.strip():
            return {"ok": False, "backend": "none"}
        fid = _fid(task_id, role, index)
        text = finding[:_MAX_FINDING_CHARS].strip()
        ts = _now_ts()
        ok = self._qdrant.add(fid, task_id, role, text, tool, source, ts)
        backend = "qdrant" if ok else ("local" if self._local.add(
            fid, task_id, role, text, tool, source, ts) else "none")
        if not ok:
            logger.debug(
                f"SKB share via qdrant failed for {role} → local fallback")
        return {"ok": bool(backend != "none"), "backend": backend}

    # ── استحضار ───────────────────────────────────────────────────────
    def query_knowledge(self, query: str,
                        task_id: Optional[str] = None,
                        k: int = _SEARCH_LIMIT_DEFAULT) -> List[Dict[str, Any]]:
        """يستحضر ما شاركه زملاء الدور (دلالة إن توفّر Qdrant)."""
        if not (query or "").strip():
            return []
        results = self._qdrant.search(query, task_id=task_id, k=k)
        if not results:
            results = self._local.search(query, task_id=task_id, k=k)
        # تجنب تكرار نفس النص
        seen: List[str] = []
        out: List[Dict[str, Any]] = []
        for score, payload in results:
            txt = payload.get("text", "")
            if txt in seen:
                continue
            seen.append(txt)
            out.append({**payload, "score": round(score, 3)})
        return out[:k]

    # ── إدارة ─────────────────────────────────────────────────────────
    def prune(self) -> int:
        """حذف المعارف القديمة (فوق سقف العمر)."""
        n = self._local.prune(_FRESHNESS_DAYS)
        return n

    def stats(self) -> Dict[str, Any]:
        q = self._qdrant.stats()
        return {
            "qdrant_active": q.get("active", False),
            "qdrant_points": q.get("points_count"),
            "local_count": self._local.count(),
            "embedder_available": self._qdrant._embedder.available(),
        }


# ══════════════════════════════════════════════════════════════════
# Singleton + helpers للتكامل داخل التعاون
# ══════════════════════════════════════════════════════════════════

_SKB: Optional[SharedKnowledgeBase] = None
_SKB_LOCK = threading.Lock()


def get_skb() -> SharedKnowledgeBase:
    global _SKB
    if _SKB is None:
        with _SKB_LOCK:
            if _SKB is None:
                _SKB = SharedKnowledgeBase()
    return _SKB


def share_task_finding(task_id: str, role: str, finding: str,
                       tool: str = "web_search", source: str = "",
                       index: int = 0) -> Dict[str, Any]:
    """اختصار مشاركة معرفة ضمن مهمة تعاونية."""
    try:
        return get_skb().share_finding(
            task_id, role, finding, tool, source, index)
    except Exception as e:
        logger.debug(f"SKB share error (swallowed): {e}")
        return {"ok": False, "backend": "none"}


def query_task_knowledge(query: str, task_id: Optional[str] = None,
                         k: int = _SEARCH_LIMIT_DEFAULT) -> List[Dict[str, Any]]:
    """اختصار استحضار معرفة لفريق المهمة."""
    try:
        return get_skb().query_knowledge(query, task_id, k)
    except Exception as e:
        logger.debug(f"SKB query error (swallowed): {e}")
        return []


def skb_stats() -> Dict[str, Any]:
    try:
        return get_skb().stats()
    except Exception:
        return {"qdrant_active": False, "local_count": 0,
                "embedder_available": False}


def skb_latest(limit: int = 10) -> List[Dict[str, Any]]:
    """أحدث المعارف المشتركة (للوحة المراقبة) — قراءة مباشرة من المحلي."""
    try:
        _skb = get_skb()
        return [x[1] for x in _skb._local.recent(limit)]
    except Exception:
        return []
