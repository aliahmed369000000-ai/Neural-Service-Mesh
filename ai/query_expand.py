"""
توسيع الاستعلام (Query Expansion) للفصحى ↔ اللهجة اليمنية.

يزيد استدعاء CKG/البحث عبر إضافة مرادفات لهجية أو فصيحة من:
  data/yemeni/msa_dialect_pairs.jsonl
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Set

_WORD = re.compile(r"[\u0600-\u06FF]+")


@lru_cache(maxsize=1)
def _load_maps():
    path = "data/yemeni/msa_dialect_pairs.jsonl"
    msa_to_dia: Dict[str, Set[str]] = {}
    dia_to_msa: Dict[str, Set[str]] = {}
    if not os.path.exists(path):
        return msa_to_dia, dia_to_msa
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                d = (obj.get("dialect_norm") or obj.get("dialect") or "").strip()
                m = (obj.get("msa_norm") or obj.get("msa") or "").strip()
                if not d or not m:
                    continue
                msa_to_dia.setdefault(m, set()).add(d)
                dia_to_msa.setdefault(d, set()).add(m)
    except Exception:
        pass
    return msa_to_dia, dia_to_msa


def expand_query(text: str, max_extra: int = 8) -> List[str]:
    """
    يُرجع قائمة صيغ: الأصلية + توسيعات لهجية/فصيحة.
    """
    if not text or not text.strip():
        return []
    variants = [text.strip()]
    msa_to_dia, dia_to_msa = _load_maps()
    words = _WORD.findall(text)
    extra: List[str] = []
    for w in words:
        wn = w
        # لهجة → فصيح
        for m in list(dia_to_msa.get(wn, []))[:2]:
            extra.append(text.replace(w, m))
        # فصيح → لهجة
        for d in list(msa_to_dia.get(wn, []))[:2]:
            extra.append(text.replace(w, d))
        if len(extra) >= max_extra:
            break
    for e in extra:
        if e not in variants:
            variants.append(e)
        if len(variants) >= max_extra + 1:
            break
    return variants


def expansion_terms(text: str, max_terms: int = 12) -> List[str]:
    """مفردات إضافية فقط (للبحث/BM25)."""
    msa_to_dia, dia_to_msa = _load_maps()
    terms: List[str] = []
    seen = set()
    for w in _WORD.findall(text or ""):
        for x in list(dia_to_msa.get(w, [])) + list(msa_to_dia.get(w, [])):
            if x not in seen and x != w:
                seen.add(x)
                terms.append(x)
            if len(terms) >= max_terms:
                return terms
    return terms
