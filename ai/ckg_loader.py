# -*- coding: utf-8 -*-
"""
ai/ckg_loader.py — تحميل كسول ومؤشرات محوسبة لقاعدة CKG المعرفية.

المشكلة التي يحلها هذا الملف:
  cognitive_graph.json حجمه ~39MB. المنطق القديم كان يقرأ الملف ويحلّل
  JSON كاملًا عند كل فقدان للكاش، ثم يبحث بمطابقة عادية مع normaliza_arabic
  داخل حلقات على 7,300+ مفهوم و19,000+ علاقة لكل استعلام.

الحل:
  1. `get_ckg_data()` — يحمّل ويحلّل مرة واحدة فقط في عمر العملية
     (Module-level singleton) مع حماية LFS pointer والفشل الناعم.
     لا يعتمد على Streamlit إطلاقًا: يعمل في أي سياق بايثون.
  2. `build_indices()` — يبني مرة واحدة فهارس lookup باسم مطبَّع:
     - concepts_by_normalized: {الاسم المطبَّع: قائمة المفاهيم المطابقة}
     - relations_index: {الاسم المطبَّع: قائمة العلاقات التي طرفها هذا الاسم}
     فيتحول البحث من O(N×M) normalize إلى O(1) dict lookup.
  3. `search_ckg_query(q_norm)` — البحث الكامل (مفهوم مباشر + العلاقات)
     عبر الفهارس.

الاستخدام في app_core (اختياري ومتسامح مع الغياب):

    try:
        from ai.ckg_loader import get_indices
        concept, related, rels = get_indices().search(q_norm)
    except ImportError:
        # السلوك القديم
        ...
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# نحاول استيراد normalize_arabic الحقيقية من app_core (نفس الدالة التي
# يستخدمها search_knowledge في app_core.py) لضمان تطابق الفهارس مع نتائج
# البحث الأصلي؛ إن غابت نستخدم التطبيع المحلي المبسط أدناه.
_APP_CORE_NORMALIZE = None
try:
    import app_core as _ac  # noqa: E402
    _APP_CORE_NORMALIZE = getattr(_ac, "normalize_arabic", None)
except Exception:
    _APP_CORE_NORMALIZE = None

# ═══════════════════════════════════════════════════════════════════════════
# الثوابت
# ═══════════════════════════════════════════════════════════════════════════
DEFAULT_CKG_PATH = Path("knowledge") / "cognitive_graph.json"

# المفتاح الوحيد المطلوب في بنية الملف.
_CKG_MANDATORY = ("concepts", "relations")

# دالة التطبيع الافتراضية إن لم يُمرَّر تطبيع خارجي (نسخة مبسطة من
# normalize_arabic الموجودة في app_core — إزالة التشكيل والهمزات والألفات).
_NORM_TABLE = str.maketrans("أإآؤئى", "اااااا")


def _simple_normalize_arabic(text: str) -> str:
    """تطبيع خفيف لا يعتمد على app_core — يطابق normalize_arabic الحقيقية
    في app_core: إزالة التشكيل (064B-065F) والألف الخنجرية والتنوين
    والأحرف غير المرئية وتوحيد المسافات."""
    text = _strip_tashkeel(text)
    for ch in ("أ", "إ", "آ", "ٱ"):
        text = text.replace(ch, "ا")
    text = text.replace("\ufeff", "")
    text = " ".join(text.split())
    return text.strip()


def _strip_tashkeel(text: str) -> str:
    """إزالة التشكيل وعلامات الألف الخنجرية والتنوين."""
    return "".join(
        ch for ch in text
        if not ("\u0610" <= ch <= "\u061A") and not ("\u064B" <= ch <= "\u065F")
    )


def normalize_for_index(text: str) -> str:
    """التطبيع المعتمد للفهارس — نسخة محلية مطابقة لسلوك normalize_arabic
    الحقيقية في app_core (لا تعتمد على app_core عند الاستيراد المنعزل)."""
    text = _strip_tashkeel(text)
    for ch in ("أ", "إ", "آ", "ٱ"):
        text = text.replace(ch, "ا")
    text = text.replace("\ufeff", "")
    text = " ".join(text.split())
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# التحميل الكسول لمرة واحدة
# ═══════════════════════════════════════════════════════════════════════════
_empty = {"concepts": {}, "relations": {}, "meta": {}, "_meta": {}}

# Singleton داخل العملية: (المسار, المود-تايم, البيانات).
_cache: Dict[str, Any] = {"path": None, "mtime": None, "data": None}


def get_ckg_data(path: Optional[Path] = None) -> Dict[str, Any]:
    """يحمّل الـ CKG مرة واحدة في عمر العملية ويعيد نفس المرجع بعدها.

    - يتعرف على Git LFS pointer ويعيد البنية الفارغة.
    - يعيد تحميل البيانات فقط إذا تغير mtime للملف (تطوير حار).
    - لا يعتمد على Streamlit إطلاقًا.
    """
    path = path or DEFAULT_CKG_PATH
    try:
        stat = path.stat()
    except OSError:
        return _empty
    mtime = stat.st_mtime
    if _cache["path"] == str(path) and _cache["mtime"] == mtime and _cache["data"] is not None:
        return _cache["data"]
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content or content.startswith("version https://git-lfs"):
            _cache.update(path=str(path), mtime=mtime, data=_empty)
            return _empty
        data = json.loads(content)
        if not isinstance(data, dict):
            _cache.update(path=str(path), mtime=mtime, data=_empty)
            return _empty
        for key in _CKG_MANDATORY:
            if key not in data or not isinstance(data[key], dict):
                data[key] = {}
        data.setdefault("meta", {})
        data.setdefault("_meta", {})
        _cache.update(path=str(path), mtime=mtime, data=data)
        return data
    except Exception:
        return _empty


def reset_ckg_cache() -> None:
    """لمسح المرجع المحفوظ — للاختبارات فقط."""
    _cache.update(path=None, mtime=None, data=None)


# ═══════════════════════════════════════════════════════════════════════════
# الفهارس المحوسبة
# ═══════════════════════════════════════════════════════════════════════════
class CkgIndices:
    """فهارس lookup محوسبة مرة واحدة: الاسم المطبَّع → مفاهيم / علاقات."""

    __slots__ = ("concepts_by_normalized", "relations_index", "normalized_keys",
                 "data", "built_at", "normalize")

    def __init__(self, data: Dict[str, Any], normalize: Any = None) -> None:
        self.data = data
        self.normalize = normalize if callable(normalize) else normalize_for_index
        fn = self.normalize
        concepts = data.get("concepts", {}) or {}
        relations = data.get("relations", {}) or {}

        concepts_by_normalized: Dict[str, List[Tuple[str, Dict]]] = {}
        for cname, cdata in concepts.items():
            if not isinstance(cdata, dict):
                continue
            key = fn(cname)
            concepts_by_normalized.setdefault(key, []).append((cname, cdata))

        # relations_index: الاسم المطبَّع → [(source, target, relation_type, weight)]
        relations_index: Dict[str, List[Dict[str, Any]]] = {}
        for _rel_key, rel_data in relations.items():
            if not isinstance(rel_data, dict):
                continue
            src = str(rel_data.get("source", ""))
            tgt = str(rel_data.get("target", ""))
            entry = {
                "source": src,
                "target": tgt,
                "relation_type": rel_data.get("relation_type", ""),
                "weight": rel_data.get("weight", 0),
            }
            for name in (src, tgt):
                key = fn(name)
                relations_index.setdefault(key, []).append(entry)

        self.concepts_by_normalized = concepts_by_normalized
        self.relations_index = relations_index
        self.normalized_keys = concepts_by_normalized.keys()
        self.built_at = time.time()

    # ── البحث ─────────────────────────────────────────────────────────────
    def find_concept(self, query: str) -> Optional[Dict[str, Any]]:
        """بحث مباشر عن مفهوم: يطبِّع المدخل أولًا ثم مطابقة كاملة ثم جزئية.

        يقبل نصًا خامًا (بالتشكيل) أو اسمًا مطبَّعًا مسبقًا — التطبيع
        داخل الدالة نفسه متطابق مع التطبيع المستخدم في بناء الفهارس.
        """
        q_norm = self.normalize(query)
        hits = self.concepts_by_normalized.get(q_norm)
        if hits:
            cname, cdata = hits[0]
            return {"name": cname, **cdata}
        # مطابقة جزئية: الاسم المطبَّع للاستعلام داخل أي مفهوم مطبَّع.
        for key, hits in self.concepts_by_normalized.items():
            if q_norm and q_norm in key:
                cname, cdata = hits[0]
                return {"name": cname, **cdata}
        return None

    def related_relations(self, query: str) -> Tuple[List[str], List[Dict]]:
        """العلاقات التي طرفها الاسم، مع العكس المزدوج — يطبِّع المدخل أولًا."""
        q_norm = self.normalize(query)
        rels = self.relations_index.get(q_norm, [])
        related: List[str] = []
        relations: List[Dict] = []
        seen = set()
        for entry in rels:
            # entry قد تظهر مرتين (مرة من source ومرة من target).
            tag = (entry["source"], entry["target"], entry["relation_type"])
            if tag in seen:
                continue
            seen.add(tag)
            if normalize_for_index(entry["source"]) == q_norm:
                other = entry["target"]
            else:
                other = entry["source"]
            related.append(other)
            relations.append({
                "target": other,
                "type": entry["relation_type"],
                "weight": entry["weight"],
            })
        return related, relations

    def summary(self) -> Dict[str, int]:
        """أعداد سريعة من meta/المفاهيم — أسهل وأخف من المرور الكامل."""
        meta = self.data.get("meta") or {}
        _meta = self.data.get("_meta") or {}
        clusters = meta.get("clusters")
        if not clusters:
            clusters = {
                cdata.get("cluster")
                for cdata in self.data.get("concepts", {}).values()
                if isinstance(cdata, dict) and cdata.get("cluster")
            }
        try:
            return {
                "concepts": int(_meta.get("total_concepts") or meta.get("total_concepts") or len(self.data.get("concepts", {}))),
                "relations": int(_meta.get("total_relations") or meta.get("total_relations") or len(self.data.get("relations", {}))),
                "roots": int(meta.get("arabic_roots") or len(self.data.get("arabic_roots") or {})),
                "clusters": int(len(clusters)),
            }
        except Exception:
            return {}

    @property
    def index_size(self) -> int:
        """عدد المفاتيح المطبَّعة في فهرس المفاهيم."""
        return len(self.concepts_by_normalized)


def get_indices(path: Optional[Path] = None, normalize: Any = None) -> CkgIndices:
    """يحمّل CKG مرة واحدة ويبني الفهارس مرة واحدة (module-level).

    `normalize` الافتراضي هو normalize_arabic الحقيقية من app_core
    (نفس الدالة المستخدمة في search_knowledge) لضمان تطابق تام مع نتائج
    البحث الأصلي؛ إن تعذر استيرادها يُستخدم التطبيع المحلي.
    """
    global _index_cache
    key = str(path or DEFAULT_CKG_PATH)
    entry = _index_cache.get(key)
    if entry is not None:
        return entry
    data = get_ckg_data(path)
    fn = normalize if callable(normalize) else (_APP_CORE_NORMALIZE or normalize_for_index)
    entry = CkgIndices(data, normalize=fn)
    _index_cache[key] = entry
    return entry


_index_cache: Dict[str, CkgIndices] = {}


def reset_indices_cache() -> None:
    """لمسح الفهارس المحوسبة — للاختبارات فقط."""
    global _index_cache
    _index_cache.clear()
    reset_ckg_cache()


def search_ckg_query(q_norm: str, path: Optional[Path] = None) -> Dict[str, Any]:
    """البحث الكامل عبر الفهارس: مفهوم مباشر + العلاقات المزدوجة."""
    indices = get_indices(path)
    concept = indices.find_concept(q_norm)
    related, relations = indices.related_relations(q_norm) if concept else ([], [])
    return {
        "concept_data": concept,
        "ckg_related": related,
        "ckg_relations": relations,
        "found": bool(concept or related),
    }
