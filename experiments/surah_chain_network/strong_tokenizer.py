"""
Tokenizer قوي لـ SurahChain LM:
  - رموز خاصة قياسية (PAD/UNK/BOS/EOS/…)
  - كلمات عربية بعد تطبيع
  - وحدات حرفية كاحتياط (character fallback)
  - دمج ثنائيات شائعة (bigram merges) بأسلوب BPE مبسّط من بيانات التدريب
  - encode / decode ثنائي الاتجاه + حفظ/تحميل JSON
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_AR_DIAC = re.compile(r"[\u064B-\u065F\u0670\u0640]")
_TOKEN_RE = re.compile(r"[\u0600-\u06FFa-zA-Z0-9]+|[^\s]", re.UNICODE)


class StrongTokenizer:
    PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
    OFFSET = 6
    SPECIAL = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>")

    def __init__(self, vocab_size: int = 8192, merges: int = 500):
        self.vocab_size = int(vocab_size)
        self.max_merges = int(merges)
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.merges: List[Tuple[str, str]] = []  # (a,b) -> a+b order of learned merges
        self.merge_set = set()
        for i, tok in enumerate(self.SPECIAL):
            self.word_to_id[tok] = i
            self.id_to_word[i] = tok

    def normalize(self, text: str) -> str:
        text = _AR_DIAC.sub("", text or "")
        text = (
            text.replace("أ", "ا")
            .replace("إ", "ا")
            .replace("آ", "ا")
            .replace("ى", "ي")
            .replace("ة", "ه")
        )
        return text.strip()

    def _add(self, token: str) -> int:
        if not token:
            return self.UNK
        if token in self.word_to_id:
            return self.word_to_id[token]
        if len(self.word_to_id) >= self.vocab_size:
            return self.UNK
        idx = len(self.word_to_id)
        self.word_to_id[token] = idx
        self.id_to_word[idx] = token
        return idx

    def _pre_tokenize(self, text: str) -> List[str]:
        text = self.normalize(text)
        return _TOKEN_RE.findall(text)

    def build_from_texts(self, texts: Sequence[str], max_vocab: Optional[int] = None) -> int:
        """يبني مفردات كلمات + حروف + دمج ثنائيات متكررة."""
        cap = max_vocab or self.vocab_size
        word_counts: Counter = Counter()
        char_counts: Counter = Counter()
        for t in texts:
            toks = self._pre_tokenize(t)
            word_counts.update(toks)
            for w in toks:
                char_counts.update(list(w))

        self.word_to_id = {tok: i for i, tok in enumerate(self.SPECIAL)}
        self.id_to_word = {i: tok for i, tok in enumerate(self.SPECIAL)}
        self.merges = []
        self.merge_set = set()

        # 1) أحرف شائعة أولاً (لدعم الكلمات النادرة)
        for ch, _ in char_counts.most_common(200):
            if len(self.word_to_id) >= cap:
                break
            self._add(ch)

        # 2) كلمات كاملة
        for w, _ in word_counts.most_common(cap):
            if len(self.word_to_id) >= cap:
                break
            self._add(w)

        # 3) BPE-lite: دمج أزواج الرموز داخل الكلمات الأكثر تكراراً
        # نمثّل كل كلمة كقائمة رموز أحرف في البداية ثم ندمج
        word_freq = Counter({w: c for w, c in word_counts.items() if len(w) >= 2})
        splits = {w: list(w) for w in word_freq}
        for _ in range(self.max_merges):
            if len(self.word_to_id) >= cap:
                break
            pair_counts: Counter = Counter()
            for w, freq in word_freq.items():
                syms = splits[w]
                for i in range(len(syms) - 1):
                    pair_counts[(syms[i], syms[i + 1])] += freq
            if not pair_counts:
                break
            (a, b), _ = pair_counts.most_common(1)[0]
            merged = a + b
            self.merges.append((a, b))
            self.merge_set.add((a, b))
            self._add(merged)
            # طبّق الدمج على splits
            new_splits = {}
            for w, syms in splits.items():
                out = []
                i = 0
                while i < len(syms):
                    if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                new_splits[w] = out
            splits = new_splits

        self.vocab_size = max(self.vocab_size, len(self.word_to_id))
        return len(self.word_to_id)

    def _apply_merges(self, chars: List[str]) -> List[str]:
        syms = list(chars)
        for a, b in self.merges:
            out = []
            i = 0
            merged = a + b
            while i < len(syms):
                if i < len(syms) - 1 and syms[i] == a and syms[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
        return syms

    def _tokenize_word(self, word: str) -> List[str]:
        if word in self.word_to_id:
            return [word]
        # BPE على مستوى الأحرف
        pieces = self._apply_merges(list(word))
        # إن بقي جزء غير معروف انقسم إلى أحرف معروفة
        final = []
        for p in pieces:
            if p in self.word_to_id:
                final.append(p)
            else:
                for ch in p:
                    final.append(ch if ch in self.word_to_id else None)
        return [x for x in final if x is not None] or [None]

    def encode(self, text: str, max_len: int = 256) -> np.ndarray:
        words = self._pre_tokenize(text)
        ids = [self.BOS]
        for w in words:
            for piece in self._tokenize_word(w):
                if piece is None:
                    ids.append(self.UNK)
                else:
                    ids.append(self.word_to_id.get(piece, self.UNK))
        ids.append(self.EOS)
        return np.array(ids[:max_len], dtype=np.int64)

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        special = {self.PAD, self.UNK, self.BOS, self.EOS, self.SEP, self.MASK}
        parts: List[str] = []
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            tok = self.id_to_word.get(i, "")
            if not tok:
                continue
            # لصق الحروف/القطع الفرعية بدون مسافة إن كانت قصيرة جداً
            if parts and len(tok) == 1 and len(parts[-1]) >= 1 and not parts[-1].endswith(" "):
                # إن كان السابق قطعة فرعية أيضاً
                if len(parts[-1]) < 4 or len(tok) == 1:
                    parts[-1] = parts[-1] + tok
                    continue
            parts.append(tok)
        # مسافات بين وحدات تبدو ككلمات
        out = []
        buf = ""
        for p in parts:
            if len(p) == 1:
                buf += p
            else:
                if buf:
                    out.append(buf)
                    buf = ""
                out.append(p)
        if buf:
            out.append(buf)
        return " ".join(out)

    def save(self, path: str) -> str:
        path = str(path)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": "strong-tokenizer-v1",
            "vocab_size": self.vocab_size,
            "word_to_id": self.word_to_id,
            "merges": self.merges,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        Path(tmp).replace(path)
        return path

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.word_to_id = {str(k): int(v) for k, v in data.get("word_to_id", {}).items()}
        self.id_to_word = {int(v): str(k) for k, v in self.word_to_id.items()}
        self.merges = [tuple(x) for x in data.get("merges", [])]
        self.merge_set = set(self.merges)
        self.vocab_size = int(data.get("vocab_size", max(self.vocab_size, len(self.word_to_id))))
        for i, tok in enumerate(self.SPECIAL):
            self.word_to_id[tok] = i
            self.id_to_word[i] = tok
