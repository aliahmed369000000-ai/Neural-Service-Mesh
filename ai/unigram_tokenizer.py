"""
Unigram Language Model Tokenizer (أسلوب SentencePiece-Unigram).

الفكرة:
  - كل مقطع فرعي له احتمال (أو درجة log-prob)
  - تقسيم النص = اختيار التجزئة ذات أعلى احتمال إجمالي (Viterbi)
  - التدريب: ابدأ بقاموس كبير من المقاطع المرشحة ثم احذف الأقل فائدة تدريجياً

هذا تنفيذ تعليمي/عملي خفيف (بدون مكتبة sentencepiece).
الواجهة متوافقة مع باقي tokenizers في NSM.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>")
SPIECE_UNDERLINE = "\u2581"
DEFAULT_PATH = "models/unigram_tokenizer.json"
MAX_SEQ_LEN = 128


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_sp_string(text: str) -> str:
    """نص → سلسلة بـ ▁ قبل كل كلمة (بدون مسافات عادية)."""
    text = normalize_arabic(text)
    if not text:
        return ""
    parts = []
    for w in text.split(" "):
        if w:
            parts.append(SPIECE_UNDERLINE + w)
    return "".join(parts)


class UnigramTokenizer:
    PAD, UNK, BOS, EOS, SEP, MASK = PAD, UNK, BOS, EOS, SEP, MASK
    DEFAULT_VOCAB_PATH = DEFAULT_PATH

    def __init__(self, vocab_size: int = 8192, vocab_path: Optional[str] = None):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        # log probabilities (أعلى = أفضل)
        self.log_probs: Dict[str, float] = {}
        self._max_piece_len = 1

        path = vocab_path or DEFAULT_PATH
        if path and os.path.exists(path):
            self.load(path)

    def train(
        self,
        texts: Iterable[str],
        num_iters: int = 3,
        seed_max_len: int = 8,
    ) -> int:
        """
        1) اجمع كل المقاطع المرشحة (محارف + n-grams حتى seed_max_len) مع تكرارها
        2) خصص احتمالات أولية بالتكرار
        3) كرّر: قيّم فائدة كل مقطع عبر EM مبسّط، واحذف الأضعف حتى vocab_size
        """
        corpus_strings: List[str] = []
        for t in texts:
            s = to_sp_string(str(t))
            if s:
                corpus_strings.append(s)

        # مرشحون: كل محرف + كل substring حتى seed_max_len
        counts: Counter = Counter()
        for s in corpus_strings:
            n = len(s)
            for i in range(n):
                for L in range(1, min(seed_max_len, n - i) + 1):
                    counts[s[i : i + L]] += 1

        # ابدأ بأكبر المرشحين (مع ضمان كل المحارف)
        chars = {c for s in corpus_strings for c in s}
        ranked = [p for p, _ in counts.most_common()]
        # المحارف أولاً دائماً
        pieces = list(chars)
        for p in ranked:
            if p not in chars:
                pieces.append(p)
            if len(pieces) >= max(self.vocab_size * 3, self.vocab_size + 500):
                break

        total = sum(counts[p] for p in pieces) or 1
        log_probs = {
            p: math.log(max(counts[p], 1) / total) for p in pieces
        }

        target = max(self.vocab_size - len(SPECIAL_TOKENS), 32)

        for _it in range(max(1, num_iters)):
            if len(log_probs) <= target:
                break
            # EM مبسّط: أعد تقدير التكرار عبر أفضل تجزئة Viterbi
            expected: Counter = Counter()
            for s in corpus_strings:
                seg = self._viterbi(s, log_probs)
                expected.update(seg)
            if not expected:
                break
            tot = sum(expected.values()) or 1
            for p in list(log_probs.keys()):
                if p in expected:
                    log_probs[p] = math.log(expected[p] / tot)
                else:
                    # مقاطع لم تُستخدم → عقوبة
                    log_probs[p] = log_probs.get(p, -20.0) - 2.0

            # احذف الأسوأ مع الإبقاء على المحارف
            if len(log_probs) > target:
                removable = [
                    p for p in log_probs.keys() if len(p) > 1
                ]
                removable.sort(key=lambda p: log_probs[p])
                drop_n = max(1, (len(log_probs) - target) // max(1, num_iters - _it))
                for p in removable[:drop_n]:
                    del log_probs[p]

        # بناء القاموس النهائي: الأفضل أولاً
        ordered = sorted(log_probs.keys(), key=lambda p: -log_probs[p])
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        self.log_probs = {}
        for p in ordered:
            if len(self.token_to_id) >= self.vocab_size:
                break
            if p in self.token_to_id:
                continue
            idx = len(self.token_to_id)
            self.token_to_id[p] = idx
            self.id_to_token[idx] = p
            self.log_probs[p] = log_probs[p]
        # ضمان المحارف
        for c in chars:
            if c not in self.token_to_id and len(self.token_to_id) < self.vocab_size:
                idx = len(self.token_to_id)
                self.token_to_id[c] = idx
                self.id_to_token[idx] = c
                self.log_probs[c] = log_probs.get(c, -10.0)

        self.vocab_size = max(self.vocab_size, len(self.token_to_id))
        self._max_piece_len = max((len(p) for p in self.log_probs), default=1)
        return len(self.token_to_id)

    def _viterbi(self, s: str, log_probs: Dict[str, float]) -> List[str]:
        """أفضل تجزئة بأعلى مجموع log-prob."""
        n = len(s)
        if n == 0:
            return []
        NEG = -1e18
        best = [NEG] * (n + 1)
        back: List[Optional[int]] = [None] * (n + 1)
        best[0] = 0.0
        max_l = max((len(p) for p in log_probs), default=1)
        unk_pen = -15.0

        for i in range(n):
            if best[i] <= NEG / 2:
                continue
            upper = min(n, i + max_l)
            found = False
            for j in range(i + 1, upper + 1):
                piece = s[i:j]
                if piece in log_probs:
                    score = best[i] + log_probs[piece]
                    if score > best[j]:
                        best[j] = score
                        back[j] = i
                    found = True
            # مسار حرف واحد دائماً متاح
            j = i + 1
            piece = s[i:j]
            lp = log_probs.get(piece, unk_pen)
            score = best[i] + lp
            if score > best[j]:
                best[j] = score
                back[j] = i

        # استرجاع
        if best[n] <= NEG / 2:
            return list(s)  # fallback محارف
        pieces: List[str] = []
        idx = n
        while idx > 0:
            prev = back[idx]
            if prev is None:
                pieces.append(s[idx - 1 : idx])
                idx -= 1
            else:
                pieces.append(s[prev:idx])
                idx = prev
        pieces.reverse()
        return pieces

    def _segment(self, text: str) -> List[str]:
        s = to_sp_string(text)
        if not s:
            return []
        if not self.log_probs:
            # بدون تدريب: محارف
            return list(s)
        return self._viterbi(s, self.log_probs)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        pieces = self._segment(text)
        ids = [self.BOS]
        for p in pieces:
            ids.append(self.token_to_id.get(p, self.UNK))
        ids.append(self.EOS)
        return np.array(ids[:max_len], dtype=np.int64)

    def content_ids(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        pieces = self._segment(text)
        return np.array(
            [self.token_to_id.get(p, self.UNK) for p in pieces][:max_len],
            dtype=np.int64,
        )

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        special = {self.PAD, self.UNK, self.BOS, self.EOS, self.SEP, self.MASK}
        buf = []
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            tok = self.id_to_token.get(i, SPECIAL_TOKENS[self.UNK])
            if tok in SPECIAL_TOKENS:
                continue
            buf.append(tok)
        text = "".join(buf).replace(SPIECE_UNDERLINE, " ").strip()
        return re.sub(r"\s+", " ", text)

    def vocab_id(self) -> int:
        return max(self.vocab_size, len(self.token_to_id))

    @property
    def word_to_id(self) -> Dict[str, int]:
        return self.token_to_id

    @property
    def id_to_word(self) -> Dict[int, str]:
        return self.id_to_token

    def save(self, path: Optional[str] = None) -> str:
        path = path or DEFAULT_PATH
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "version": "unigram-v1",
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "log_probs": self.log_probs,
            "underline": SPIECE_UNDERLINE,
        }
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp, path)
        return path

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        self.token_to_id = {str(k): int(v) for k, v in data.get("token_to_id", {}).items()}
        self.id_to_token = {int(v): str(k) for k, v in self.token_to_id.items()}
        self.log_probs = {str(k): float(v) for k, v in data.get("log_probs", {}).items()}
        self.vocab_size = int(data.get("vocab_size", max(self.vocab_size, len(self.token_to_id))))
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok
        self._max_piece_len = max((len(p) for p in self.log_probs), default=1)


if __name__ == "__main__":
    texts = [
        "الصبر مفتاح الفرج",
        "التقوى من الايمان",
        "العلم نور والجهل ظلام",
        "الرحمه وسعت كل شيء",
    ]
    tok = UnigramTokenizer(vocab_size=200)
    n = tok.train(texts, num_iters=2, seed_max_len=6)
    s = "الصبر مفتاح الفرج"
    ids = tok.encode(s)
    print("vocab", n)
    print("ids", ids.tolist())
    print("decode", repr(tok.decode(ids)))
