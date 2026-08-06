"""
أدوات تحسين دقة اللهجة اليمنية — خفيفة وبدون نماذج ثقيلة.

  - detect_yemeni_score(text): 0..1 مدى اللهجة اليمنية
  - normalize_yemeni(text): تطبيع إملائي للهجة
  - expand_tokenizer_from_lexicon(...)
  - load_msa_dialect_map()
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_WORD = re.compile(r"[\u0600-\u06FF]+")

# مؤشرات لهجية يمنية شائعة (وزن بسيط)
_YEMENI_MARKERS = {
    "ايش", "ليش", "وين", "فين", "مره", "ياخي", "ياخوي", "اخوي", "قلي",
    "ابشر", "أبشر", "سدا", "زول", "جهال", "قات", "بن", "شوه", "كذا",
    "هيا", "تعالي", "اشترك", "مافي", "مافش", "عنديش", "عندناش",
    "صنعاء", "عدن", "تعز", "حضرموت", "المكلا", "مارب", "مأرب",
    "حنش", "دحباش", "شلح", "قعد", "اقعد", "هاته", "هات",
    "والله", "واللهي", "امال", "امّال", "بس", "يعني",
}

_YEMENI_SUFFIXES = ("ش", "وش", "يش", "ناش", "كمش", "هش")
_YEMENI_PREFIXES = ("ه", "ها", "هال", "فال")


def normalize_yemeni(text: str) -> str:
    """تطبيع إملائي يحافظ على طابع اللهجة قدر الإمكان."""
    if not text:
        return ""
    t = _TASHKEEL.sub("", text)
    t = re.sub(r"[أإآٱ]", "ا", t)
    t = re.sub(r"[ىئ]", "ي", t)
    # توحيد تكرارات الحروف المفرطة: خاااااصة → خاصه (مع إبقاء حرفين كحد)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    # مسافات
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokenize_ar(text: str) -> List[str]:
    return _WORD.findall(normalize_yemeni(text))


def detect_yemeni_score(text: str) -> float:
    """
    تقدير 0..1 لحضور سمات يمنية.
    ليس مصنّفاً لغوياً كاملاً — إشارة عملية للمسارات (routing / boost).
    """
    words = tokenize_ar(text)
    if not words:
        return 0.0
    hits = 0.0
    for w in words:
        wl = w
        if wl in _YEMENI_MARKERS or normalize_yemeni(wl) in {normalize_yemeni(x) for x in _YEMENI_MARKERS}:
            hits += 1.5
            continue
        for s in _YEMENI_SUFFIXES:
            if wl.endswith(s) and len(wl) > len(s) + 1:
                hits += 0.6
                break
        for p in _YEMENI_PREFIXES:
            if wl.startswith(p) and len(wl) > len(p) + 1:
                hits += 0.4
                break
    # كلمات معجم Lisan إن وُجد
    lex = _lexicon_set()
    if lex:
        for w in words:
            if normalize_yemeni(w) in lex:
                hits += 0.35
    score = hits / max(len(words), 1)
    return float(max(0.0, min(1.0, score)))


@lru_cache(maxsize=1)
def _lexicon_set():
    path = "data/yemeni/dialect_lexicon.json"
    if not os.path.exists(path):
        return frozenset()
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        entries = data.get("entries") or []
        # أعلى 8k تكراراً
        return frozenset(
            e.get("norm") or normalize_yemeni(e.get("raw") or "")
            for e in entries[:8000]
            if e.get("norm") or e.get("raw")
        )
    except Exception:
        return frozenset()


@lru_cache(maxsize=1)
def load_msa_dialect_map() -> Dict[str, str]:
    """msa_norm → مثال لهجي شائع."""
    path = "data/yemeni/msa_dialect_pairs.jsonl"
    out: Dict[str, str] = {}
    if not os.path.exists(path):
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                p = json.loads(line)
                msa = p.get("msa_norm") or ""
                dia = p.get("dialect") or ""
                if msa and dia and msa not in out:
                    out[msa] = dia
    except Exception:
        pass
    return out


def expand_yemeni_tokenizer_from_lexicon(
    tokenizer,
    lexicon_path: str = "data/yemeni/dialect_lexicon.json",
    max_words: int = 4000,
) -> int:
    """
    يوسّع YemeniTokenizer بأكثر المفردات تكراراً من Lisan-Yemeni.
    يُرجع عدد الكلمات المُضافة.
    """
    if not os.path.exists(lexicon_path):
        return 0
    with open(lexicon_path, encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries") or []
    added = 0
    # دعم YemeniTokenizer._add_word أو word-level add
    for e in entries[:max_words]:
        w = e.get("raw") or e.get("token") or e.get("norm")
        if not w:
            continue
        w = str(w).strip()
        if len(w) < 2:
            continue
        before = getattr(tokenizer, "vocab_size", None)
        if hasattr(tokenizer, "_add_word"):
            tokenizer._add_word(normalize_yemeni(w))
            # أيضاً الشكل الخام إن اختلف
            if normalize_yemeni(w) != w:
                tokenizer._add_word(w)
            added += 1
        elif hasattr(tokenizer, "word_to_id") and callable(getattr(tokenizer, "word_to_id")):
            # property vs method
            pass
    return added


def load_yemeni_sentences(path: str = "data/yemeni/sentences.jsonl", limit: int = 0) -> List[str]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                t = obj.get("text") or ""
                if t:
                    out.append(t)
            except Exception:
                continue
            if limit and len(out) >= limit:
                break
    return out
