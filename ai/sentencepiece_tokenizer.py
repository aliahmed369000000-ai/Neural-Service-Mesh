"""
SentencePiece-style Tokenizer (تنفيذ خفيف — بدون مكتبة sentencepiece).

أفكار SentencePiece الأصلية (Google):
  1) لا يعتمد على قاطع كلمات خاص بلغة — يعامل النص كسلسلة محارف Unicode
  2) المسافة تُرمَّز برمز صريح ▁ (U+2581) بدل حذفها
  3) التدريب عادة BPE أو Unigram LM على هذه السلسلة
  4) مناسب للعربية والخلط مع أرقام/لاتيني لأن ما فيش pre-tokenizer لغوي

هذا الملف ينفّذ مسار **BPE على مستوى المحارف + ▁** بواجهة NSM الموحدة:
  encode / decode / content_ids / train / save / load
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>")
OFFSET = 6
SPIECE_UNDERLINE = "\u2581"  # ▁ — رمز المسافة في SentencePiece
DEFAULT_SP_PATH = "models/sentencepiece_tokenizer.json"
MAX_SEQ_LEN = 128


def normalize_arabic(text: str) -> str:
    """تطبيع خفيف — بدون حذف المسافات (جوهر SentencePiece)."""
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    # توحيد المسافات المتعددة
    text = re.sub(r"\s+", " ", text).strip()
    return text


def to_sp_chars(text: str) -> List[str]:
    """
    تحويل النص إلى تسلسل SentencePiece:
    كل كلمة تُسبق بـ ▁ ثم محارفها: 'مرحبا بك' → ▁ م ر ح ب ا ▁ ب ك
    """
    text = normalize_arabic(text)
    if not text:
        return []
    chars: List[str] = []
    for word in text.split(" "):
        if not word:
            continue
        chars.append(SPIECE_UNDERLINE)
        chars.extend(list(word))
    return chars


class SentencePieceTokenizer:
    """
    SentencePiece تقريبي (BPE على محارف + ▁).
    """

    PAD, UNK, BOS, EOS, SEP, MASK = PAD, UNK, BOS, EOS, SEP, MASK
    OFFSET = OFFSET
    DEFAULT_VOCAB_PATH = DEFAULT_SP_PATH

    def __init__(self, vocab_size: int = 8192, vocab_path: Optional[str] = None):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        self.merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}

        path = vocab_path or DEFAULT_SP_PATH
        if path and os.path.exists(path):
            self.load(path)

    def train(self, texts: Iterable[str], num_merges: Optional[int] = None) -> int:
        # كل «مستند/جملة» كتسلسل محارف SP مع تكرار
        corpus: Dict[Tuple[str, ...], int] = Counter()
        alphabet: Counter = Counter()
        for text in texts:
            chs = to_sp_chars(str(text))
            if not chs:
                continue
            key = tuple(chs)
            corpus[key] += 1
            for ch in chs:
                alphabet[ch] += 1

        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        for ch, _ in alphabet.most_common():
            if len(self.token_to_id) >= self.vocab_size:
                break
            if ch not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[ch] = idx
                self.id_to_token[idx] = ch

        target = num_merges if num_merges is not None else max(0, self.vocab_size - len(self.token_to_id))
        self.merges = []
        self._merge_ranks = {}

        # للعمل على Counter قابل للتعديل
        corpus_map: Dict[Tuple[str, ...], int] = dict(corpus)

        for _ in range(target):
            if len(self.token_to_id) >= self.vocab_size:
                break
            pairs = Counter()
            for word, freq in corpus_map.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            (a, b), _f = pairs.most_common(1)[0]
            new_tok = a + b
            if new_tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[new_tok] = idx
                self.id_to_token[idx] = new_tok
            self.merges.append((a, b))
            self._merge_ranks[(a, b)] = len(self.merges) - 1
            corpus_map = self._merge_corpus(corpus_map, a, b)

        self.vocab_size = max(self.vocab_size, len(self.token_to_id))
        return len(self.token_to_id)

    @staticmethod
    def _merge_corpus(
        corpus: Dict[Tuple[str, ...], int], a: str, b: str
    ) -> Dict[Tuple[str, ...], int]:
        new_c: Dict[Tuple[str, ...], int] = {}
        for word, freq in corpus.items():
            out: List[str] = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(word[i])
                    i += 1
            key = tuple(out)
            new_c[key] = new_c.get(key, 0) + freq
        return new_c

    def _bpe_segment(self, chars: List[str]) -> List[str]:
        symbols = list(chars)
        if not self.merges or len(symbols) < 2:
            return symbols
        while True:
            if len(symbols) < 2:
                break
            pairs = [(symbols[i], symbols[i + 1]) for i in range(len(symbols) - 1)]
            ranked = [
                (self._merge_ranks[p], i, p)
                for i, p in enumerate(pairs)
                if p in self._merge_ranks
            ]
            if not ranked:
                break
            ranked.sort()
            _r, _i, (a, b) = ranked[0]
            new_sym: List[str] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_sym.append(a + b)
                    i += 2
                else:
                    new_sym.append(symbols[i])
                    i += 1
            symbols = new_sym
        return symbols

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        chars = to_sp_chars(text)
        pieces = self._bpe_segment(chars)
        ids = [self.BOS]
        for p in pieces:
            ids.append(self.token_to_id.get(p, self.UNK))
        ids.append(self.EOS)
        return np.array(ids[:max_len], dtype=np.int64)

    def content_ids(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        chars = to_sp_chars(text)
        pieces = self._bpe_segment(chars)
        ids = [self.token_to_id.get(p, self.UNK) for p in pieces]
        return np.array(ids[:max_len], dtype=np.int64)

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
        text = "".join(buf)
        # ▁ → مسافة
        text = text.replace(SPIECE_UNDERLINE, " ").strip()
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
        path = path or DEFAULT_SP_PATH
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "version": "sentencepiece-bpe-v1",
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
            "merges": [list(m) for m in self.merges],
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
        self.merges = [tuple(m) for m in data.get("merges", [])]
        self._merge_ranks = {tuple(m): i for i, m in enumerate(self.merges)}
        self.vocab_size = int(data.get("vocab_size", max(self.vocab_size, len(self.token_to_id))))
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok


if __name__ == "__main__":
    texts = [
        "الصبر مفتاح الفرج",
        "التقوى من الايمان",
        "العلم نور",
        "الرحمه وسعت كل شيء",
    ]
    tok = SentencePieceTokenizer(vocab_size=250)
    n = tok.train(texts, num_merges=100)
    s = "الصبر مفتاح الفرج"
    ids = tok.encode(s)
    print("vocab", n, "ids", ids.tolist())
    print("decode", repr(tok.decode(ids)))
