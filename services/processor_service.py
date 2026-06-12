
from __future__ import annotations
import re
from typing import Any, Dict
from core.node import BaseNode, NodeSchema


class ProcessorNode(BaseNode):
    def __init__(self, name: str = "ProcessorNode"):
        super().__init__(name=name, description="Analyzes and enriches text",
                         tags=["processor", "analysis"])

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={"text": "str", "word_count": "int", "char_count": "int"},
                          required=["text"])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(
            fields={"text": "str", "processed_text": "str", "word_count": "int",
                    "char_count": "int", "sentences": "list", "keywords": "list", "analysis": "dict"},
            required=["text", "processed_text", "analysis"])

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        text = str(data.get("text", ""))
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
        keywords = self._keywords(text)
        words = text.split()
        analysis = {
            "sentence_count": len(sentences),
            "unique_word_count": len(set(text.lower().split())),
            "avg_word_length": round(sum(len(w) for w in words) / len(words), 2) if words else 0.0,
            "has_numbers": any(c.isdigit() for c in text),
            "language": "arabic" if sum(1 for c in text if "\u0600" <= c <= "\u06FF") / max(len(text), 1) > 0.3 else "latin",
        }
        return {
            "text": text,
            "processed_text": text.strip().lower(),
            "word_count": data.get("word_count") or len(words),
            "char_count": data.get("char_count") or len(text),
            "sentences": sentences,
            "keywords": keywords,
            "analysis": analysis,
        }

    def _keywords(self, text: str, top: int = 10) -> list:
        stop = {"the","a","an","is","it","in","on","at","to","for","of","and","or","but","not",
                "with","this","that","are","was","be","as","from","by","have","has","had"}
        freq: Dict[str, int] = {}
        for w in text.split():
            w = w.strip(".,!?;:\"'()[]{}").lower()
            if len(w) > 3 and w not in stop:
                freq[w] = freq.get(w, 0) + 1
        return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:top]]
