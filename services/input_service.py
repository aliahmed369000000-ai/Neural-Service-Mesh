from __future__ import annotations
from typing import Any, Dict
from core.node import BaseNode, NodeSchema


class InputNode(BaseNode):
    def __init__(self, name: str = "InputNode"):
        super().__init__(name=name, description="Entry point: accepts raw text",
                         tags=["input", "entry"])

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"text": "str", "source": "str", "metadata": "dict"},
                          required=["text"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"text": "str", "source": "str",
                                  "char_count": "int", "word_count": "int"},
                          required=["text", "char_count", "word_count"])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = str(data.get("text", ""))
        return {
            "text": text,
            "source": data.get("source", "unknown"),
            "char_count": len(text),
            "word_count": len(text.split()) if text.strip() else 0,
            "metadata": data.get("metadata", {}),
        }
