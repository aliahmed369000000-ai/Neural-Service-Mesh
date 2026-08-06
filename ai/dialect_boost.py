"""
طبقة تعزيز لهجي موحّدة للاستخدام من QA / Chat / Agents.

تجمع:
  - كشف اللهجة اليمنية
  - توسيع الاستعلام
  - استرجاع أمثلة RAG
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def analyze_and_boost(query: str, top_k_rag: int = 3) -> Dict[str, Any]:
    """
    يُرجع:
      dialect_score, is_yemeni, expanded_queries, expansion_terms, rag_context, rag_hits
      + hybrid_score / camel عند توفر خط الأنابيب الموحّد
    """
    out: Dict[str, Any] = {
        "query": query,
        "dialect_score": 0.0,
        "is_yemeni": False,
        "expanded_queries": [query] if query else [],
        "expansion_terms": [],
        "rag_context": "",
        "rag_hits": [],
    }
    if not query or not str(query).strip():
        return out

    # المسار المفضّل: خط الأنابيب الموحّد (تنظيف + هجين + RAG)
    try:
        from ai.nlp_pipeline import process_query
        info = process_query(query, use_rag=True, top_k_rag=top_k_rag)
        out.update({
            "dialect_score": float(info.get("hybrid_score") or info.get("dialect_score") or 0.0),
            "is_yemeni": bool(info.get("is_yemeni")),
            "normalized": info.get("normalized") or info.get("cleaned") or query,
            "expanded_queries": info.get("expanded_queries") or [query],
            "expansion_terms": info.get("expansion_terms") or [],
            "rag_context": info.get("rag_context") or "",
            "rag_hits": info.get("rag_hits") or [],
            "hybrid_score": float(info.get("hybrid_score") or 0.0),
            "camel": info.get("camel") or {},
            "pipeline": info.get("pipeline") or [],
        })
        return out
    except Exception:
        pass

    try:
        from ai.yemeni_dialect import detect_yemeni_score, normalize_yemeni
        score = float(detect_yemeni_score(query))
        out["dialect_score"] = score
        out["is_yemeni"] = score >= 0.25
        out["normalized"] = normalize_yemeni(query)
    except Exception:
        pass

    try:
        from ai.query_expand import expand_query, expansion_terms
        out["expanded_queries"] = expand_query(query)
        out["expansion_terms"] = expansion_terms(query)
    except Exception:
        pass

    try:
        from ai.yemeni_rag import get_yemeni_index, retrieve_yemeni_context
        hits = get_yemeni_index().search(query, top_k=top_k_rag)
        out["rag_hits"] = [{"text": h.text, "score": h.score} for h in hits]
        out["rag_context"] = retrieve_yemeni_context(query, top_k=top_k_rag)
    except Exception:
        pass

    return out


def best_search_queries(query: str, limit: int = 5) -> List[str]:
    """قائمة استعلامات للبحث المتوازي على CKG."""
    info = analyze_and_boost(query, top_k_rag=0)
    qs = list(info.get("expanded_queries") or [query])
    return qs[:limit]
