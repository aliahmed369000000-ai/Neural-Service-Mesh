"""
RAG خفيف على جمل اللهجة اليمنية (BM25 تقريبي بدون مكتبات).

يستخدم data/yemeni/sentences.jsonl بعد:
  python3 scripts/prepare_yemeni_lisan.py
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

from ai.query_expand import expansion_terms

_WORD = re.compile(r"[\u0600-\u06FF0-9]+")
_DEFAULT_PATH = "data/yemeni/sentences.jsonl"


def _tok(text: str) -> List[str]:
    return [w for w in _WORD.findall((text or "").lower()) if len(w) > 1]


@dataclass
class RAGHit:
    text: str
    score: float
    index: int


class YemeniSentenceIndex:
    def __init__(self, path: str = _DEFAULT_PATH, max_docs: int = 30000):
        self.path = path
        self.docs: List[str] = []
        self.doc_len: List[int] = []
        self.df: Dict[str, int] = defaultdict(int)
        self.tf: List[Counter] = []
        self.avgdl = 0.0
        self._load(max_docs)

    def _load(self, max_docs: int) -> None:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if len(self.docs) >= max_docs:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    t = (obj.get("text") or "").strip()
                except Exception:
                    t = line
                if len(t) < 8:
                    continue
                terms = _tok(t)
                if not terms:
                    continue
                c = Counter(terms)
                self.docs.append(t)
                self.tf.append(c)
                self.doc_len.append(len(terms))
                for term in c:
                    self.df[term] += 1
        n = len(self.docs) or 1
        self.avgdl = sum(self.doc_len) / n

    def __len__(self) -> int:
        return len(self.docs)

    def search(self, query: str, top_k: int = 5, k1: float = 1.5, b: float = 0.75) -> List[RAGHit]:
        if not self.docs:
            return []
        q_terms = _tok(query)
        # توسيع بمفردات MSA/لهجة
        for t in expansion_terms(query):
            q_terms.extend(_tok(t))
        if not q_terms:
            return []
        N = len(self.docs)
        scores = []
        q_set = set(q_terms)
        for i, c in enumerate(self.tf):
            score = 0.0
            dl = self.doc_len[i]
            for term in q_set:
                if term not in c:
                    continue
                df = self.df.get(term, 0) or 1
                idf = math.log(1 + (N - df + 0.5) / (df + 0.5))
                tf = c[term]
                denom = tf + k1 * (1 - b + b * dl / max(self.avgdl, 1e-6))
                score += idf * (tf * (k1 + 1)) / max(denom, 1e-6)
            if score > 0:
                scores.append((score, i))
        scores.sort(reverse=True)
        hits = []
        for sc, i in scores[:top_k]:
            hits.append(RAGHit(text=self.docs[i], score=float(sc), index=i))
        return hits


@lru_cache(maxsize=1)
def get_yemeni_index(max_docs: int = 30000) -> YemeniSentenceIndex:
    return YemeniSentenceIndex(max_docs=max_docs)


def retrieve_yemeni_context(query: str, top_k: int = 3) -> str:
    """نص سياق جاهز للحقن في المطالبات."""
    hits = get_yemeni_index().search(query, top_k=top_k)
    if not hits:
        return ""
    parts = [f"- {h.text}" for h in hits]
    return "أمثلة لهجية ذات صلة:\n" + "\n".join(parts)
