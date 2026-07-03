"""
Vector Backend — طبقة ذاكرة متجهية حقيقية اختيارية (Qdrant Cloud)
=====================================================================
تُضيف طبقة "قوية" فوق الذاكرة الدلالية الحالية (TF-IDF)، بنفس فلسفة
ai/llm_fallback.py تماماً: سلسلة تراجع (fallback chain) صامتة —
إن فشل أي جزء (لا مفاتيح، لا مكتبة مثبتة، لا شبكة، انتهت حصة الخادم
المجاني...) يتراجع النظام فوراً للطبقة الأدنى بدون أي انقطاع بالخدمة.

الترتيب الكامل للذاكرة الدلالية بعد هذه الإضافة:
    1) Qdrant Cloud (embeddings حقيقية عبر Cloudflare bge-m3، دقة عالية)
    2) TF-IDF محلي (يعمل دائماً — صفر اعتمادية خارجية)

الإعداد اختياري بالكامل — لا شيء يتعطل إن لم تُضِف شيئاً:
    QDRANT_URL       — رابط الـ cluster من Qdrant Cloud (باقة مجانية دائمة)
    QDRANT_API_KEY   — مفتاح API من Qdrant Cloud
    (تُستخدم CF_ACCOUNT_ID / CF_API_TOKEN الموجودة أصلاً في مشروعك — نفس
     المفاتيح المستخدمة لـ LLM Fallback — لتوليد embeddings متعددة اللغات
     تدعم العربية عبر نموذج @cf/baai/bge-m3، فلا حاجة لأي حساب إضافي
     للـ embeddings نفسها.)
"""
from __future__ import annotations

import logging
import os
import uuid
from typing import Dict, List, Optional, Tuple

from ai.llm_fallback import _post_json  # نفس دالة الطلب المستخدمة لكل مزودي LLM

logger = logging.getLogger("VectorBackend")

_CF_EMBED_MODEL = "@cf/baai/bge-m3"   # متعدد اللغات — يدعم العربية
_EMBED_DIM = 1024                      # أبعاد bge-m3
_COLLECTION = "nsm_memory"


def _point_id(key: str) -> str:
    """UUID ثابت من نص المفتاح — يضمن أن نفس المفتاح دائماً يُحدِّث نفس النقطة
    بدل إنشاء نسخ مكررة (يطابق منطق upsert_fact في nsm_memory.py)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


class CloudflareEmbedder:
    """يولّد embeddings حقيقية عبر Cloudflare Workers AI (bge-m3)."""

    def __init__(self):
        self._account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
        self._token = os.getenv("CF_API_TOKEN", "").strip()

    def available(self) -> bool:
        return bool(self._account_id and self._token)

    def embed(self, text: str) -> Optional[List[float]]:
        if not self.available() or not text.strip():
            return None
        url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{self._account_id}/ai/run/{_CF_EMBED_MODEL}"
        )
        try:
            data = _post_json(
                url, {"text": [text[:2000]]},
                {"Authorization": f"Bearer {self._token}",
                 "Content-Type": "application/json"},
                12,
            )
            vecs = (data.get("result", {}) or {}).get("data", [])
            return vecs[0] if vecs else None
        except Exception as e:
            logger.debug(f"CloudflareEmbedder error: {e}")
            return None


class QdrantBackend:
    """
    طبقة ذاكرة متجهية حقيقية (اختيارية بالكامل). تُستخدم فقط إن:
      1) مكتبة qdrant-client مثبتة
      2) QDRANT_URL و QDRANT_API_KEY موجودة
      3) الاتصال بالخادم ناجح فعلياً (يُفحص مرة واحدة فقط عند أول استخدام)
    أي فشل في أي شرط → available() ترجع False تلقائياً والنظام يتراجع
    فوراً وبصمت لـ TF-IDF المحلي في nsm_memory.py.
    """

    def __init__(self):
        self._client = None
        self._embedder = CloudflareEmbedder()
        self._checked = False
        self._ok = False

    def _ensure_client(self) -> bool:
        if self._checked:
            return self._ok
        self._checked = True
        url = os.getenv("QDRANT_URL", "").strip()
        api_key = os.getenv("QDRANT_API_KEY", "").strip()
        if not url or not api_key or not self._embedder.available():
            return False
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            client = QdrantClient(url=url, api_key=api_key, timeout=8)
            existing = [c.name for c in client.get_collections().collections]
            if _COLLECTION not in existing:
                client.create_collection(
                    collection_name=_COLLECTION,
                    vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
                )
            self._client = client
            self._ok = True
            logger.info("QdrantBackend: متصل بنجاح — الذاكرة المتجهية الحقيقية مفعّلة")
        except Exception as e:
            logger.debug(f"QdrantBackend unavailable, falling back to TF-IDF: {e}")
            self._ok = False
        return self._ok

    def available(self) -> bool:
        return self._ensure_client()

    def upsert(self, key: str, text: str, payload: dict) -> bool:
        """key: مفتاح فريد ثابت (نفس المفتاح = نفس النقطة تُحدَّث لا تتكرر)."""
        if not self.available():
            return False
        try:
            vec = self._embedder.embed(text)
            if not vec:
                return False
            from qdrant_client.models import PointStruct
            self._client.upsert(
                collection_name=_COLLECTION,
                points=[PointStruct(id=_point_id(key), vector=vec, payload=payload)],
            )
            return True
        except Exception as e:
            logger.debug(f"QdrantBackend upsert error: {e}")
            return False

    def search(self, query: str, k: int = 5,
               filter_payload: Optional[Dict[str, str]] = None) -> List[Tuple[float, dict]]:
        if not self.available():
            return []
        try:
            vec = self._embedder.embed(query)
            if not vec:
                return []
            qfilter = None
            if filter_payload:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                qfilter = Filter(must=[
                    FieldCondition(key=fk, match=MatchValue(value=fv))
                    for fk, fv in filter_payload.items()
                ])
            # قد تختلف الواجهة بين إصدارات qdrant-client (query_points الأحدث
            # مقابل search القديمة) — نجرّب الأحدث ثم نتراجع للقديمة تلقائياً.
            if hasattr(self._client, "query_points"):
                resp = self._client.query_points(
                    collection_name=_COLLECTION, query=vec,
                    limit=k, query_filter=qfilter,
                )
                hits = resp.points
            else:
                hits = self._client.search(
                    collection_name=_COLLECTION, query_vector=vec,
                    limit=k, query_filter=qfilter,
                )
            return [(float(h.score), h.payload or {}) for h in hits]
        except Exception as e:
            logger.debug(f"QdrantBackend search error: {e}")
            return []


_singleton: Optional[QdrantBackend] = None


def get_vector_backend() -> QdrantBackend:
    """نسخة وحيدة مشتركة (singleton) — نفس نمط LLMFallback في المشروع."""
    global _singleton
    if _singleton is None:
        _singleton = QdrantBackend()
    return _singleton
