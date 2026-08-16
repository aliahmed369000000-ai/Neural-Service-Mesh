"""
ai/qdrant_semantic_memory.py
=============================
🆕 الحزمة 4: الذاكرة الدلالية للمحادثات — QdrantSemanticMemory

يحفظ كل محادثة (سؤال/جواب) في مجموعة Qdrant مخصصة «nsm_conversations»
بمتجهات bge-m3 عربية (1024 بعد، عبر ai.shared_knowledge._Embedder المعاد
استخدامه — لا تكرار)، ويسترجع عند كل سؤال جديد أقرب 3 محادثات دلاليًا
(جيب التمام على Cloudflare bge-m3).

لماذا منفصل عن shared_knowledge؟
- skb_findings مخصصة لمخرجات *الوكلاء* (tool findings) وتوهين 7 أيام،
  بينما المحادثات الدردشية تُحفظ بلا توهين زمني (الذاكرة تراكمية).
- مجموعة مستقلة «nsm_conversations» = فصل مسؤولية واضح.

التدرّج الآمن الكامل — ثلاث طبقات، كلها صامتة عند أي فشل:
  1. Qdrant + bge-m3 (إن توفّرت المكتبة والمفاتيح والاتصال)
  2. SQLite محلي (data/qdrant_semantic_memory.db) بحقل searchable + بحث
     ترتيب TF مع أوزان عربية — يعمل دائمًا بلا إنترنت
  3. أي استثناء → [] / False بدون أي أثر على المحادثة الحيّة

الاستخدام:
    from ai.qdrant_semantic_memory import QdrantSemanticMemory
    mem = QdrantSemanticMemory()
    mem.add_conversation("ma_hu_altwhid", "user", "ما معنى التوحيد؟", "assistant", "التوحيد هو إفراد الله...")
    for score, payload in mem.search_conversations("التوحيد والشرك"):
        print(score, payload["user_text"], payload["assistant_text"])
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("QdrantSemanticMemory")

_CONVO_COLLECTION = "nsm_conversations"      # مجموعة Qdrant للمحادثات
_LOCAL_DB = "data/qdrant_semantic_memory.db"
_SEARCH_LIMIT = 3                           # أقرب 3 محادثات دلاليًا
_INIT_SQL = """
CREATE TABLE IF NOT EXISTS nsm_conversations (
    id       TEXT PRIMARY KEY,
    user_text      TEXT NOT NULL,
    assistant_text TEXT NOT NULL,
    score      REAL NOT NULL,
    ts         REAL NOT NULL,
    searchable TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_convo_ts ON nsm_conversations(ts);
"""
# كلمات عربية بلا محتوى دلالي تُستثنى من الترتيب المحلي
_AR_STOP = {
    "ما", "هو", "هي", "هو", "في", "من", "عن", "الى", "إلى", "على", "أن", "أن",
    "هذا", "هذه", "ذلك", "التي", "الذي", "و", "أو", "لا", "بل", "ثم", "قد",
    "هل", "اذا", "إذا", "كما", "حيث", "كل", "أي", "مع", "عند", "بين", "بعد",
    "قبل", "الان", "الآن", "غير", "فقط", "جدا", "أيضا", "أيضاً", "ايضا",
    "هو", "هي", "هم", "هن", "انا", "أنا", "انت", "أنت", "كان", "كانت", "يكون",
    "لا", "ليس", "ليست", "انه", "أنه", "انها", "أنها", "هذا", "هذه", "ذلك",
}


def _normalize(text: str) -> str:
    """تطبيع نص عربي لبحث محلي (مثل shared_knowledge._normalize)."""
    if not text:
        return ""
    t = text.lower().strip()
    t = re.sub(r"[أإآا]", "ا", t)
    t = t.replace("ة", "ه").replace("ى", "ي")
    t = re.sub(r"[ًٌٍَُِّْ]", "", t)
    t = re.sub(r"[^\u0600-\u06FFa-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class QdrantSemanticMemory:
    """ذاكرة دلالية للمحادثات — Qdrant أولًا ثم SQLite محلي كاحتياط دائم."""

    def __init__(self) -> None:
        self._client = None
        self._embed_ok = False
        self._tried = False
        self._embedder = None  # ai.shared_knowledge._Embedder (يُحمَّل كسولًا)
        self._dim = 1024
        self._local_db_conn = None

    # ──────────────────────────────────────────────────────────────────
    # الطبقة الأولى: Qdrant
    # ──────────────────────────────────────────────────────────────────
    def _try_qdrant(self) -> bool:
        """حاول الاتصال بـ Qdrant مرة واحدة (يُحفظ النتاج)."""
        if self._tried:
            return self._embed_ok
        self._tried = True
        url = os.getenv("QDRANT_URL", "").strip()
        key = os.getenv("QDRANT_API_KEY", "").strip()
        if not url or not key:
            return False
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            from ai.shared_knowledge import _Embedder, _EMBED_DIM
            self._client = QdrantClient(url=url, api_key=key, timeout=8,
                                        check_compatibility=False)
            existing = [c.name for c in self._client.get_collections().collections]
            if _CONVO_COLLECTION not in existing:
                self._client.create_collection(
                    collection_name=_CONVO_COLLECTION,
                    vectors_config=VectorParams(size=_EMBED_DIM,
                                                distance=Distance.COSINE),
                )
            self._embedder = _Embedder()
            self._dim = _EMBED_DIM
            self._embed_ok = bool(self._embedder.available())
            if self._embed_ok:
                logger.info(
                    "QdrantSemanticMemory: Qdrant + bge-m3 مفعّل للمحادثات"
                )
        except Exception as exc:
            logger.debug(f"QdrantSemanticMemory unavailable: {exc}")
            self._client, self._embedder, self._embed_ok = None, None, False
        return self._embed_ok

    # ──────────────────────────────────────────────────────────────────
    # الطبقة الثانية: SQLite محلي دائم (يعمل بلا إنترنت)
    # ──────────────────────────────────────────────────────────────────
    def _local_conn(self) -> Optional[sqlite3.Connection]:
        if self._local_db_conn is None:
            try:
                db_path = os.path.join(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    _LOCAL_DB,
                )
                os.makedirs(os.path.dirname(db_path), exist_ok=True)
                conn = sqlite3.connect(db_path, check_same_thread=False)
                conn.execute("PRAGMA journal_mode=WAL")
                conn.executescript(_INIT_SQL)
                conn.commit()
                self._local_db_conn = conn
            except Exception as exc:
                logger.debug(f"QdrantSemanticMemory local db unavailable: {exc}")
                return None
        return self._local_db_conn

    def _local_score(self, query: str, user_text: str, assistant_text: str) -> float:
        """ترتيب تقريبي TF موزون عربي للنص المحلي (0..1)."""
        q_tokens = set(_normalize(query).split()) - _AR_STOP
        if not q_tokens:
            return 0.0
        u_tokens = set(_normalize(user_text).split())
        a_tokens = set(_normalize(assistant_text).split())
        hit_user = len(q_tokens & u_tokens)
        hit_all = hit_user + len(q_tokens & a_tokens)
        return min(1.0, (hit_user * 2 + (hit_all - hit_user)) / max(1, len(q_tokens) * 3))

    def _local_search(self, query: str, limit: int) -> List[Tuple[float, Dict[str, Any]]]:
        conn = self._local_conn()
        if not conn:
            return []
        try:
            rows = conn.execute(
                "SELECT id, user_text, assistant_text, ts FROM nsm_conversations "
                "ORDER BY ts DESC LIMIT ?", (min(limit * 40, 2000),)
            ).fetchall()
        except Exception:
            return []
        scored = []
        for _rid, _u, _a, _ts in rows:
            s = self._local_score(query, _u, _a)
            if s > 0.0:
                scored.append((s, {
                    "id": _rid, "user_text": _u, "assistant_text": _a,
                    "ts": _ts, "backend": "sqlite",
                }))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:limit]

    # ──────────────────────────────────────────────────────────────────
    # الواجهة العامة
    # ──────────────────────────────────────────────────────────────────
    @property
    def active(self) -> bool:
        """هل طبقة Qdrant مفعّلة؟"""
        return self._try_qdrant()

    def stats(self) -> Dict[str, Any]:
        """حالة الذاكرة + عدد النقاط."""
        info: Dict[str, Any] = {"qdrant_active": self._try_qdrant()}
        if self._client and info["qdrant_active"]:
            try:
                col = self._client.get_collection(_CONVO_COLLECTION)
                info["qdrant_points"] = int(getattr(col, "points_count", None) or 0)
            except Exception:
                info["qdrant_points"] = None
        conn = self._local_conn()
        if conn:
            try:
                info["sqlite_points"] = int(conn.execute(
                    "SELECT COUNT(*) FROM nsm_conversations").fetchone()[0])
            except Exception:
                info["sqlite_points"] = None
        return info

    def add_conversation(
        self,
        convo_id: str,
        user_text: str,
        assistant_text: str,
        relevance: float = 0.0,
    ) -> bool:
        """
        يحفظ محادثة في الذاكرة الدلالية (Qdrant أولًا، وعند تعذّره SQLite محلي).
        relevance: درجة جودة الرد (0..1) من التقييم الذاتي — محفوظة كميتا.
        """
        saved_any = False
        if (user_text or "").strip() and (assistant_text or "").strip():
            # ── Qdrant ────────────────────────────────────────────
            if self._try_qdrant():
                try:
                    vec = self._embedder.embed(f"{user_text} {assistant_text}")
                    if vec:
                        from qdrant_client.models import PointStruct
                        self._client.upsert(
                            collection_name=_CONVO_COLLECTION,
                            points=[PointStruct(
                                id=convo_id,
                                vector=vec,
                                payload={
                                    "user_text": user_text[:4000],
                                    "assistant_text": assistant_text[:8000],
                                    "relevance": float(relevance or 0.0),
                                    "ts": __import__("time").time(),
                                    "backend": "qdrant",
                                },
                            )],
                        )
                        saved_any = True
                except Exception as exc:
                    logger.debug(f"Qdrant save error: {exc}")
            # ── SQLite محلي (دائمًا — حتى لو نجح Qdrant كنسخة احتياطية) ──
            conn = self._local_conn()
            if conn:
                try:
                    import time as _time
                    searchable = _normalize(f"{user_text} {assistant_text}")
                    conn.execute(
                        "INSERT OR REPLACE INTO nsm_conversations "
                        "(id, user_text, assistant_text, score, ts, searchable) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (convo_id, user_text[:2000], assistant_text[:4000],
                         float(relevance or 0.0), _time.time(), searchable),
                    )
                    conn.commit()
                    saved_any = True
                except Exception as exc:
                    logger.debug(f"Local save error: {exc}")
        return saved_any

    def search_conversations(
        self,
        query: str,
        limit: int = _SEARCH_LIMIT,
        skip_id: Optional[str] = None,
    ) -> List[Tuple[float, Dict[str, Any]]]:
        """
        يسترجع أقرب محادثات دلاليًا لسياق السؤال الجديد.
        Qdrant (bge-m3 عربي) أولًا، وعند تعذّره الترتيب المحلي TF العربي.
        """
        if not (query or "").strip():
            return []
        # ── Qdrant ────────────────────────────────────────────────
        if self._try_qdrant():
            try:
                vec = self._embedder.embed(query)
                if vec:
                    from qdrant_client.models import Filter, FieldCondition, Range
                    qfilter = Filter(
                        must=[FieldCondition(
                            key="ts", range=Range(gte=__import__("time").time()
                                                  - 180 * 86400))]
                    )
                    if hasattr(self._client, "query_points"):
                        resp = self._client.query_points(
                            collection_name=_CONVO_COLLECTION,
                            query=vec, limit=limit + 2,
                            query_filter=qfilter,
                        )
                        hits = resp.points
                    else:
                        hits = self._client.search(
                            collection_name=_CONVO_COLLECTION,
                            query_vector=vec, limit=limit + 2,
                            query_filter=qfilter,
                        )
                    results = [
                        (float(h.score), dict(h.payload or {})) for h in hits
                    ]
                    if skip_id:
                        results = [(s, p) for s, p in results
                                   if str(p.get("id", "")) != skip_id]
                    if results:
                        return results[:limit]
            except Exception as exc:
                logger.debug(f"Qdrant search error: {exc}")
        # ── SQLite محلي ────────────────────────────────────────────
        local = self._local_search(query, limit + 2)
        if skip_id:
            local = [(s, p) for s, p in local
                     if str(p.get("id", "")) != skip_id]
        return local[:limit]
