"""
MCP Internal Gateway — نواة موحدة للوعي المعرفي
================================================
واجهة واحدة يستدعيها الوكلاء بدل تكرار كود CKG/Pipeline.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def search_ckg(query: str, limit: int = 5) -> Dict[str, Any]:
    query = (query or "").strip()
    limit = max(1, min(20, int(limit)))
    # Prefer encoder v2 semantic if available
    try:
        from ai.ckg_text_encoder_v2 import encode_query, search as ckg_search
        hits = ckg_search(query, limit=limit) if callable(ckg_search) else []
        return {"ok": True, "query": query, "hits": hits, "via": "ckg_text_encoder_v2"}
    except Exception:
        pass
    try:
        from knowledge.cognitive_graph import CognitiveKnowledgeGraph
        ckg = CognitiveKnowledgeGraph()
        names = list(getattr(ckg, "_concepts", {}) or {})
        hits = [n for n in names if query in n][:limit]
        return {"ok": True, "query": query, "hits": hits, "via": "cognitive_graph"}
    except Exception as e:
        return {"ok": False, "error": str(e), "query": query}


def reason(question: str, train_on_query: bool = False) -> Dict[str, Any]:
    try:
        from ai.reasoning_pipeline import ReasoningPipeline
        pipe = ReasoningPipeline(train_on_query=bool(train_on_query), use_deep_routing=True)
        result = pipe.answer(question or "")
        return {
            "ok": True,
            "answer": getattr(result, "answer_text", str(result)),
            "weights": getattr(result, "decision_weights", {}),
            "ranked": (getattr(result, "ranked_concepts", None) or [])[:8],
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def unified_query(question: str) -> Dict[str, Any]:
    return {
        "ckg": search_ckg(question, limit=8),
        "reasoning": reason(question, train_on_query=False),
    }


def handle_gateway_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(بواب[ةه]\s*mcp|mcp\s*gateway|نواة\s*موحد|استعلام\s*موحد)", text, re.I):
        m = re.search(r"(?:سؤال|query)[:\s]+(.+)$", text, re.I)
        q = m.group(1).strip() if m else "الأمانة"
        r = unified_query(q)
        return "## 🌐 بوابة MCP داخلية\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2, default=str)[:3500] + "\n```"
    return None
