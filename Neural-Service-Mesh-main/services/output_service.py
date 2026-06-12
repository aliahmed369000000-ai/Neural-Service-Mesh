
from __future__ import annotations
from typing import Any, Dict
from datetime import datetime
from core.node import BaseNode, NodeSchema


class OutputNode(BaseNode):
    def __init__(self, name: str = "OutputNode", output_format: str = "json"):
        super().__init__(name=name, description="Terminal node: formats final result",
                         tags=["output", "terminal"])
        self.output_format = output_format

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"text": "str", "analysis": "dict", "keywords": "list"},
                          required=["text"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "Any", "format": "str",
                                  "rendered_at": "str", "success": "bool"},
                          required=["result", "format", "success"])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.output_format == "text":
            result = self._as_text(data)
        elif self.output_format == "summary":
            result = self._as_summary(data)
        else:
            result = dict(data)
        return {"result": result, "format": self.output_format,
                "rendered_at": datetime.utcnow().isoformat(), "success": True}

    def _as_text(self, d):
        a = d.get("analysis", {})
        lines = [
            "=== Neural Service Mesh Output ===",
            f"Text     : {d.get('text','')}",
            f"Words    : {d.get('word_count','N/A')}",
            f"Sentences: {a.get('sentence_count','N/A')}",
            f"Language : {a.get('language','N/A')}",
            f"Keywords : {', '.join(d.get('keywords',[])[:5])}",
            "==================================",
        ]
        return "\n".join(lines)

    def _as_summary(self, d):
        a = d.get("analysis", {})
        return {
            "text_preview": str(d.get("text", ""))[:100],
            "word_count": d.get("word_count"),
            "sentence_count": a.get("sentence_count"),
            "top_keywords": d.get("keywords", [])[:5],
            "language": a.get("language"),
        }
