"""
Domain Lookup — بحث بسيط عن مفاهيم في مصادر التخصصات (domain_sources.py)
==========================================================================
بحث خفيف بدون اعتماديات ثقيلة (بدون sklearn)، حتى يعمل بشكل موثوق في أي
سياق يستدعيه: التطبيق الرئيسي، وai/social_agent.py (الذي يمر عبره رد كل
منصة اجتماعية مُفعَّلة — تيليجرام، انستقرام، فيسبوك، ديسكورد... إلخ).

الطريقة: تطابق كلمات (word overlap) بين نص الاستعلام واسم المفهوم/نصه،
بعد استبعاد الكلمات الوظيفية الشائعة (مثل "ما"، "في"، "معنى") التي لا
تدل فعلياً على صلة الموضوع، واشتراط تطابق كلمتين مميزتين فأكثر — لتفادي
نتائج زائفة (مثال واقعي اكتُشف أثناء الاختبار: سؤال قرآني بحت كان
يُرفَق خطأً بمرجع نحوي غير ذي صلة لمجرد تشارك كلمة عامة واحدة مثل
"معنى"). ليست بحثاً دلالياً متقدماً، لكنها كافية لإرجاع مفاهيم ذات صلة
فعلية بدل نتائج عشوائية، وبدون أي اعتماديات إضافية قد تفشل بالنشر.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from knowledge_sources.domain_sources import get_all_domain_items

DOMAIN_LABELS_AR: Dict[str, str] = {
    "physics": "فيزياء",
    "math": "رياضيات",
    "history": "تاريخ",
    "biology": "أحياء",
    "civilizations": "حضارات",
    "chemistry": "كيمياء",
    "arabic_grammar": "نحو عربي",
    "english": "إنجليزي",
    "geography": "جغرافيا",
    "computer_science": "علوم حاسوب",
}

# كلمات وظيفية شائعة تُستبعد من المطابقة — تشارُكها وحدها بين السؤال
# ومفهوم ما لا يدل على صلة حقيقية بالموضوع.
_STOPWORDS_AR = {
    "ما", "لا", "لم", "لن", "هل", "هو", "هي", "قد", "في", "من", "الى",
    "إلى", "على", "عن", "أن", "ان", "إن", "كل", "أو", "او", "ثم", "لك",
    "بل", "كان", "كانت", "هذا", "هذه", "ذلك", "تلك", "كيف", "متى", "أين",
    "اين", "لماذا", "ايضا", "أيضاً", "مع", "بين", "عند", "بعد", "قبل",
    "معنى", "معني", "اشرح", "شرح", "وضح",
}

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def _tokenize(text: str) -> set:
    return {
        w.lower() for w in _WORD_RE.findall(text or "")
        if len(w) > 1 and w.lower() not in _STOPWORDS_AR
    }


_INDEX_CACHE: Optional[List[Dict[str, Any]]] = None


def _build_index() -> List[Dict[str, Any]]:
    """يبني فهرساً مُخزَّناً (module-level cache) مرة واحدة لكل عمليات البحث
    التالية — البيانات ثابتة (مصادر مضمَّنة بالكود)، فلا داعي لإعادة البناء."""
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    index: List[Dict[str, Any]] = []
    for domain, items in get_all_domain_items().items():
        for item in items:
            haystack = f"{item['concept']} {item['text']}"
            index.append(
                {
                    "domain": domain,
                    "domain_ar": DOMAIN_LABELS_AR.get(domain, domain),
                    "concept": item["concept"],
                    "text": item["text"],
                    "importance": item.get("importance", 0.5),
                    "tokens": _tokenize(haystack),
                }
            )
    _INDEX_CACHE = index
    return index


def search_domain_concepts(
    query: str, limit: int = 3, domain: Optional[str] = None, min_overlap: int = 2
) -> List[Dict[str, str]]:
    """يبحث عن أقرب المفاهيم لنص `query` عبر كل التخصصات (أو تخصص محدد فقط).

    min_overlap: أقل عدد كلمات مميزة (بعد استبعاد الكلمات الوظيفية) يجب
    أن تتشارك بين السؤال والمفهوم لقبوله — يمنع تفعيل نتيجة على مجرد
    كلمة عامة واحدة قد تكون مصادفة لفظية لا صلة موضوعية حقيقية.

    يُرجع قائمة عناصر {domain, domain_ar, concept, text} مرتبة تنازلياً حسب
    قوة التطابق، أو قائمة فارغة إن لم يوجد تطابق كافٍ (لا نتائج مزيَّفة عند
    غياب تطابق حقيقي)."""
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
        {
            "domain": r["domain"],
            "domain_ar": r["domain_ar"],
            "concept": r["concept"],
            "text": r["text"],
        }
        for r in scored[:limit]
    ]


def list_domains() -> Dict[str, str]:
    """يُرجع {domain_key: الاسم بالعربي} لكل التخصصات المتاحة حالياً."""
    return dict(DOMAIN_LABELS_AR)
