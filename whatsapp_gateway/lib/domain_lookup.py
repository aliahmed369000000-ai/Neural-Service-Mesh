"""
lib/domain_lookup.py — بحث عن مفاهيم دراسية داخل بوابة واتساب المعزولة.

نسخة مستقلة عن knowledge_sources/domain_lookup.py بالمستودع الرئيسي (لا
يمكن استيرادها مباشرة لأن Root Directory على Vercel = whatsapp_gateway/
فقط). تقرأ نفس المنطق (تطابق كلمات) لكن من ملف JSON محلي مُصدَّر مسبقاً
(knowledge/domains.json) بدل استيراد وحدة بايثون من الجذر.

لو احتجت تحديث محتوى المواد مستقبلاً: عدّل knowledge_sources/domain_sources.py
بالمستودع الرئيسي، ثم أعد تصدير JSON وانسخه هنا (لا تُعدّل هذا الملف
كمصدر حقيقة أساسي — هو نسخة مُصدَّرة فقط).
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_PATH = Path(__file__).resolve().parent.parent / "knowledge" / "domains.json"
_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# كلمات وظيفية شائعة تُستبعد من المطابقة — مطابق لـ
# knowledge_sources/domain_lookup.py بالمستودع الرئيسي.
_STOPWORDS_AR = {
    "ما", "لا", "لم", "لن", "هل", "هو", "هي", "قد", "في", "من", "الى",
    "إلى", "على", "عن", "أن", "ان", "إن", "كل", "أو", "او", "ثم", "لك",
    "بل", "كان", "كانت", "هذا", "هذه", "ذلك", "تلك", "كيف", "متى", "أين",
    "اين", "لماذا", "ايضا", "أيضاً", "مع", "بين", "عند", "بعد", "قبل",
    "معنى", "معني", "اشرح", "شرح", "وضح",
}

_RAW_CACHE: Optional[Dict[str, Any]] = None
_INDEX_CACHE: Optional[List[Dict[str, Any]]] = None


def _tokenize(text: str) -> set:
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if len(w) > 1 and w.lower() not in _STOPWORDS_AR
    }


def _load_raw() -> Dict[str, Any]:
    global _RAW_CACHE
    if _RAW_CACHE is None:
        with open(_DATA_PATH, "r", encoding="utf-8") as f:
            _RAW_CACHE = json.load(f)
    return _RAW_CACHE


def _build_index() -> List[Dict[str, Any]]:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    raw = _load_raw()
    labels = raw.get("labels_ar", {})
    index: List[Dict[str, Any]] = []
    for domain, items in raw.get("domains", {}).items():
        for item in items:
            haystack = f"{item['concept']} {item['text']}"
            index.append(
                {
                    "domain": domain,
                    "domain_ar": labels.get(domain, domain),
                    "concept": item["concept"],
                    "text": item["text"],
                    "importance": item.get("importance", 0.5),
                    "tokens": _tokenize(haystack),
                }
            )
    _INDEX_CACHE = index
    return index


def list_domains() -> Dict[str, str]:
    """يُرجع {domain_key: الاسم بالعربي} لكل التخصصات المتاحة."""
    return dict(_load_raw().get("labels_ar", {}))


def search_domain_concepts(
    query: str, limit: int = 3, domain: Optional[str] = None, min_overlap: int = 2
) -> List[Dict[str, str]]:
    """بحث بتطابق الكلمات، مطابق منطقياً لـknowledge_sources/domain_lookup.py.

    min_overlap: أقل عدد كلمات مميزة يجب أن تتشارك بين السؤال والمفهوم."""
    q_tokens = _tokenize(query)
    if not q_tokens:
        return []
    scored: List[Dict[str, Any]] = []
    for entry in _build_index():
        if domain and entry["domain"] != domain:
            continue
        overlap = q_tokens & entry["tokens"]
        if len(overlap) < min_overlap:
            continue
        score = len(overlap) + entry["importance"] * 0.1
        scored.append({**entry, "score": score})
    scored.sort(key=lambda r: r["score"], reverse=True)
    return [
        {"domain": r["domain"], "domain_ar": r["domain_ar"],
         "concept": r["concept"], "text": r["text"]}
        for r in scored[:limit]
    ]


def get_concepts_by_domain(domain: str, limit: int = 30) -> List[Dict[str, str]]:
    """يُرجع قائمة أسماء المفاهيم (بدون النص الكامل) لتخصص معيّن — تُستخدم
    لعرض قائمة اختيار للمستخدم قبل أن يطلب شرح مفهوم بعينه."""
    raw = _load_raw()
    items = raw.get("domains", {}).get(domain, [])
    return [{"concept": it["concept"]} for it in items[:limit]]
