"""
BPE Tokenizer (Byte-Pair Encoding) خفيف — بدون مكتبات خارجية.

مناسب للعربية داخل NSM:
  - يتعامل مع كلمات نادرة عبر تقسيمها إلى وحدات فرعية (subwords)
  - encode() / decode() ثنائيا الاتجاه
  - يُدرَّب على نصوص CKG/القرآن ثم يُحفظ كـ JSON

الخوارزمية:
  1) تطبيع عربي بسيط + تقسيم إلى كلمات
  2) كل كلمة = تسلسل محارف + رمز نهاية كلمة </w>
  3) دمج أزواج الرموز الأكثر تكراراً حتى الوصول لـ num_merges
  4) عند الترميز: تطبيق عمليات الدمج بالترتيب نفسه
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# رموز خاصة — نفس ترتيب WordTokenizer/HashTokenizer للتوافق مع النموذج
PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>")
OFFSET = 6
END_WORD = "</w>"
DEFAULT_BPE_PATH = "models/bpe_tokenizer.json"
MAX_SEQ_LEN = 128


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def split_words(text: str) -> List[str]:
    return re.findall(r"[\u0600-\u06FF]+|\d+", normalize_arabic(text))


class BPETokenizer:
    """
    tokenizer=BPE ثنائي الاتجاه.

    الملفات المحفوظة (JSON):
      - vocab: token → id
      - merges: قائمة أزواج الدمج بالترتيب
      - version / vocab_size
    """

    PAD, UNK, BOS, EOS, SEP, MASK = PAD, UNK, BOS, EOS, SEP, MASK
    OFFSET = OFFSET
    DEFAULT_VOCAB_PATH = DEFAULT_BPE_PATH

    def __init__(
        self,
        vocab_size: int = 8192,
        vocab_path: Optional[str] = None,
    ):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        self.merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}

        path = vocab_path or DEFAULT_BPE_PATH
        if path and os.path.exists(path):
            self.load(path)

    # ── تدريب BPE ──────────────────────────────────────────────────────────
    def train(self, texts: Iterable[str], num_merges: Optional[int] = None) -> int:
        """
        يدرّب BPE من نصوص خام.
        num_merges: عدد عمليات الدمج (افتراضي ≈ vocab_size - OFFSET - alphabet).
        """
        word_freq: Counter = Counter()
        for text in texts:
            for w in split_words(text):
                if w:
                    word_freq[w] += 1

        # تمثيل كل كلمة كتسلسل محارف + </w>
        corpus: Dict[Tuple[str, ...], int] = {}
        alphabet: Counter = Counter()
        for word, freq in word_freq.items():
            chars = tuple(list(word) + [END_WORD])
            corpus[chars] = corpus.get(chars, 0) + freq
            for ch in chars:
                alphabet[ch] += freq

        # قاموس أولي: خاص + محارف
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        for ch, _ in alphabet.most_common():
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                if idx >= self.vocab_size:
                    break
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        target_merges = num_merges
        if target_merges is None:
            target_merges = max(0, self.vocab_size - len(self.token_to_id))

        self.merges = []
        self._merge_ranks = {}

        for _ in range(target_merges):
            if len(self.token_to_id) >= self.vocab_size:
                break
            pairs = self._count_pairs(corpus)
            if not pairs:
                break
            best, _freq = pairs.most_common(1)[0]
            a, b = best
            new_tok = a + b
            if new_tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[new_tok] = idx
                self.id_to_token[idx] = new_tok
            self.merges.append((a, b))
            self._merge_ranks[(a, b)] = len(self.merges) - 1
            corpus = self._merge_corpus(corpus, a, b)

        self.vocab_size = max(self.vocab_size, len(self.token_to_id))
        return len(self.token_to_id)

    @staticmethod
    def _count_pairs(corpus: Dict[Tuple[str, ...], int]) -> Counter:
        pairs: Counter = Counter()
        for word, freq in corpus.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    @staticmethod
    def _merge_corpus(
        corpus: Dict[Tuple[str, ...], int], a: str, b: str
    ) -> Dict[Tuple[str, ...], int]:
        new_corpus: Dict[Tuple[str, ...], int] = {}
        bigram = (a, b)
        for word, freq in corpus.items():
            merged: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            key = tuple(merged)
            new_corpus[key] = new_corpus.get(key, 0) + freq
        return new_corpus

    def _bpe_word(self, word: str) -> List[str]:
        """تطبيق سلسلة الدمج على كلمة واحدة."""
        if not word:
            return []
        symbols = list(word) + [END_WORD]
        if not self.merges:
            return symbols

        while True:
            if len(symbols) < 2:
                break
            # أزواج موجودة مع أقل rank (أقدم دمج)
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [
                (self._merge_ranks[p], i, p)
                for i, p in enumerate(pairs)
                if p in self._merge_ranks
            ]
            if not ranked:
                break
            ranked.sort()
            _rank, idx, (a, b) = ranked[0]
            new_symbols: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_symbols.append(a + b)
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols
        return symbols

    # ── واجهة NSM ──────────────────────────────────────────────────────────
    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        ids = [self.BOS]
        for word in split_words(text):
            for piece in self._bpe_word(word):
                ids.append(self.token_to_id.get(piece, self.UNK))
        ids.append(self.EOS)
        return np.array(ids[:max_len], dtype=np.int64)

    def content_ids(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        ids: List[int] = []
        for word in split_words(text):
            for piece in self._bpe_word(word):
                ids.append(self.token_to_id.get(piece, self.UNK))
        return np.array(ids[:max_len], dtype=np.int64)

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        special = {self.PAD, self.UNK, self.BOS, self.EOS, self.SEP, self.MASK}
        parts: List[str] = []
        buf = ""
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            tok = self.id_to_token.get(i, SPECIAL_TOKENS[self.UNK])
            if tok in SPECIAL_TOKENS:
                continue
            if tok.endswith(END_WORD):
                buf += tok[: -len(END_WORD)]
                if buf:
                    parts.append(buf)
                buf = ""
            else:
                buf += tok
        if buf:
            parts.append(buf)
        return " ".join(parts)

    def vocab_id(self) -> int:
        return max(self.vocab_size, len(self.token_to_id))

    # توافق أسماء مع WordTokenizer
    @property
    def word_to_id(self) -> Dict[str, int]:
        return self.token_to_id

    @property
    def id_to_word(self) -> Dict[int, str]:
        return self.id_to_token

    def save(self, path: Optional[str] = None) -> str:
        path = path or DEFAULT_BPE_PATH
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "version": "bpe-v1",
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": [list(m) for m in self.merges],
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
        self.merges = [tuple(m) for m in data.get("merges", [])]
        self._merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.vocab_size = int(data.get("vocab_size", max(self.vocab_size, len(self.token_to_id))))
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok


def demo_roundtrip() -> None:
    texts = [
        "الصبر مفتاح الفرج",
        "التقوى من الايمان",
        "العلم نور والجهل ظلام",
        "الرحمه وسعت كل شيء",
    ]
    tok = BPETokenizer(vocab_size=200)
    n = tok.train(texts, num_merges=80)
    s = "الصبر مفتاح الفرج"
    ids = tok.encode(s)
    print("vocab", n, "ids", ids.tolist(), "decode", tok.decode(ids))


if __name__ == "__main__":
    demo_roundtrip()
