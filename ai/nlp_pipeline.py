"""
خط أنابيب NLP موحّد لـ NSM — أفضل مسار عملي بدون كسر الاعتماديات.

المراحل:
  1) تنظيف عربي (arabic_text_clean)
  2) تطبيع اختياري CAMeL
  3) كشف لهجة هجين: معجم يمني محلي + CAMeL SAN إن وُجد
  4) توسيع استعلام + RAG يمني عند الحاجة
  5) حزمة جاهزة لـ QA / Chat / Agents

الاستخدام:
    from ai.nlp_pipeline import process_query
    info = process_query("كيفك ياخوي ايش الاخبار")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def process_query(
    text: str,
    *,
    mode: str = "auto",
    use_camel: bool = True,
    use_rag: bool = True,
    top_k_rag: int = 3,
    yemen_threshold: float = 0.25,
) -> Dict[str, Any]:
    """
    mode: auto | search | display | dialect
    """
    raw = (text or "").strip()
    out: Dict[str, Any] = {
        "raw": raw,
        "cleaned": raw,
        "normalized": raw,
        "dialect_score": 0.0,
        "is_yemeni": False,
        "camel": {},
        "hybrid_score": 0.0,
        "expanded_queries": [raw] if raw else [],
        "expansion_terms": [],
        "rag_context": "",
        "rag_hits": [],
        "pipeline": [],
    }
    if not raw:
        return out

    # 1) تنظيف
    try:
        from ai.arabic_text_clean import clean_arabic
        m = "dialect" if mode in ("auto", "dialect") else (mode if mode in ("search", "display") else "display")
        out["cleaned"] = clean_arabic(raw, mode=m)
        out["pipeline"].append("arabic_text_clean")
    except Exception:
        out["cleaned"] = raw

    working = out["cleaned"] or raw

    # 2) CAMeL normalize (اختياري)
    if use_camel:
        try:
            from ai.camel_optional import camel_available, camel_normalize, identify_dialect
            if camel_available():
                out["normalized"] = camel_normalize(working) or working
                out["pipeline"].append("camel_normalize")
                did = identify_dialect(working, level="city")
                out["camel"] = did
                out["pipeline"].append("camel_did")
            else:
                out["normalized"] = working
        except Exception:
            out["normalized"] = working
    else:
        out["normalized"] = working

    # 3) كشف لهجة محلي
    local_score = 0.0
    try:
        from ai.yemeni_dialect import detect_yemeni_score
        local_score = float(detect_yemeni_score(out["normalized"] or working))
        out["pipeline"].append("yemeni_lexicon_score")
    except Exception:
        pass
    out["dialect_score"] = local_score

    # 4) درجة هجينة
    camel_y = float((out.get("camel") or {}).get("yemen_score") or 0.0)
    camel_flag = bool((out.get("camel") or {}).get("is_yemeni_san"))
    # دمج: وزن أعلى للمعجم المحلي + دعم CAMeL
    hybrid = min(1.0, local_score * 0.7 + camel_y * 0.5 + (0.15 if camel_flag else 0.0))
    if local_score >= yemen_threshold or camel_flag or camel_y >= 0.15:
        hybrid = max(hybrid, local_score, camel_y * 0.9)
    out["hybrid_score"] = float(hybrid)
    out["is_yemeni"] = hybrid >= yemen_threshold or camel_flag

    # 5) توسيع + RAG عند اللهجة
    if out["is_yemeni"] or use_rag:
        try:
            from ai.query_expand import expand_query, expansion_terms
            out["expanded_queries"] = expand_query(out["normalized"] or raw)
            out["expansion_terms"] = expansion_terms(out["normalized"] or raw)
            out["pipeline"].append("query_expand")
        except Exception:
            pass

        if use_rag and out["is_yemeni"]:
            try:
                from ai.yemeni_rag import get_yemeni_index, retrieve_yemeni_context
                hits = get_yemeni_index().search(out["normalized"] or raw, top_k=top_k_rag)
                out["rag_hits"] = [{"text": h.text, "score": h.score} for h in hits]
                out["rag_context"] = retrieve_yemeni_context(out["normalized"] or raw, top_k=top_k_rag)
                out["pipeline"].append("yemeni_rag")
            except Exception:
                pass

    return out


def preprocess_for_ckg(text: str) -> str:
    """نص موحّد لاستخراج مفاهيم CKG."""
    try:
        from ai.arabic_text_clean import normalize_for_search
        return normalize_for_search(text)
    except Exception:
        return (text or "").strip()


def is_yemeni_query(text: str) -> bool:
    return bool(process_query(text, use_rag=False).get("is_yemeni"))
