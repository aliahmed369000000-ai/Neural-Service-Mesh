"""
Byte-level BPE (BBPE) — أسلوب GPT-2.

الفكرة:
  1) النص → بايتات UTF-8 (0–255) — لا UNK على أي يونيكود صالح
  2) BPE على تسلسل البايتات
  3) decode: بايتات → UTF-8

مناسب للعربية والخلط مع أي سكربت دون قاطع لغوي.
"""
from __future__ import annotations

import json
import os
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>")
OFFSET = 6
DEFAULT_PATH = "models/byte_bpe_tokenizer.json"
MAX_SEQ_LEN = 128


def text_to_bytes(text: str) -> List[int]:
    return list(text.encode("utf-8"))


def bytes_to_text(data: List[int]) -> str:
    try:
        return bytes(data).decode("utf-8", errors="replace")
    except Exception:
        return ""


class ByteBPETokenizer:
    """
    BBPE: الرموز الأساسية = بايتات 0–255 (بعد OFFSET للرموز الخاصة).
    """

    PAD, UNK, BOS, EOS, SEP, MASK = PAD, UNK, BOS, EOS, SEP, MASK
    DEFAULT_VOCAB_PATH = DEFAULT_PATH

    def __init__(self, vocab_size: int = 8192, vocab_path: Optional[str] = None):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        self.merges: List[Tuple[str, str]] = []
        self._merge_ranks: Dict[Tuple[str, str], int] = {}
        # بايتات أساسية: مفتاح "b0".."b255"
        for b in range(256):
            key = f"b{b}"
            if key not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[key] = idx
                self.id_to_token[idx] = key
        path = vocab_path or DEFAULT_PATH
        if path and os.path.exists(path):
            self.load(path)

    def _byte_token(self, b: int) -> str:
        return f"b{b}"

    def train(self, texts: Iterable[str], num_merges: Optional[int] = None) -> int:
        corpus: Dict[Tuple[str, ...], int] = Counter()
        for t in texts:
            toks = tuple(self._byte_token(b) for b in text_to_bytes(str(t)))
            if toks:
                corpus[toks] += 1

        target = num_merges if num_merges is not None else max(0, self.vocab_size - len(self.token_to_id))
        self.merges = []
        self._merge_ranks = {}
        corpus_map = dict(corpus)

        for _ in range(target):
            if len(self.token_to_id) >= self.vocab_size:
                break
            pairs: Counter = Counter()
            for word, freq in corpus_map.items():
                for i in range(len(word) - 1):
                    pairs[(word[i], word[i + 1])] += freq
            if not pairs:
                break
            (a, b), _f = pairs.most_common(1)[0]
            new_tok = a + "_" + b  # رمز دمج
            if new_tok not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[new_tok] = idx
                self.id_to_token[idx] = new_tok
            self.merges.append((a, b))
            self._merge_ranks[(a, b)] = len(self.merges) - 1
            # merge corpus
            new_c: Dict[Tuple[str, ...], int] = {}
            for word, freq in corpus_map.items():
                out: List[str] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                        out.append(new_tok)
                        i += 2
                    else:
                        out.append(word[i])
                        i += 1
                key = tuple(out)
                new_c[key] = new_c.get(key, 0) + freq
            corpus_map = new_c

        self.vocab_size = max(self.vocab_size, len(self.token_to_id))
        return len(self.token_to_id)

    def _bpe(self, byte_toks: List[str]) -> List[str]:
        symbols = list(byte_toks)
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
            merged = a + "_" + b
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                    new_sym.append(merged)
                    i += 2
                else:
                    new_sym.append(symbols[i])
                    i += 1
            symbols = new_sym
        return symbols

    def _token_to_bytes(self, tok: str) -> List[int]:
        """فك رمز إلى بايتات (يدعم المدموج b1_b2_...)."""
        if tok.startswith("b") and "_" not in tok:
            try:
                return [int(tok[1:])]
            except ValueError:
                return []
        parts = tok.split("_")
        out: List[int] = []
        for p in parts:
            if p.startswith("b"):
                try:
                    out.append(int(p[1:]))
                except ValueError:
                    pass
        return out

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        raw = [self._byte_token(b) for b in text_to_bytes(text)]
        pieces = self._bpe(raw)
        ids = [self.BOS]
        for p in pieces:
            ids.append(self.token_to_id.get(p, self.UNK))
        ids.append(self.EOS)
        return np.array(ids[:max_len], dtype=np.int64)

    def content_ids(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        raw = [self._byte_token(b) for b in text_to_bytes(text)]
        pieces = self._bpe(raw)
        return np.array(
            [self.token_to_id.get(p, self.UNK) for p in pieces][:max_len],
            dtype=np.int64,
        )

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        special = {self.PAD, self.UNK, self.BOS, self.EOS, self.SEP, self.MASK}
        data: List[int] = []
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            tok = self.id_to_token.get(i)
            if not tok or tok in SPECIAL_TOKENS:
                continue
            data.extend(self._token_to_bytes(tok))
        return bytes_to_text(data)

    def vocab_id(self) -> int:
        return max(self.vocab_size, len(self.token_to_id))

    @property
    def word_to_id(self):
        return self.token_to_id

    @property
    def id_to_word(self):
        return self.id_to_token

    def save(self, path: Optional[str] = None) -> str:
        path = path or DEFAULT_PATH
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "version": "byte-bpe-v1",
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
        # ضمان البايتات
        for b in range(256):
            key = f"b{b}"
            if key not in self.token_to_id:
                idx = len(self.token_to_id)
                self.token_to_id[key] = idx
                self.id_to_token[idx] = key


if __name__ == "__main__":
    tok = ByteBPETokenizer(vocab_size=400)
    n = tok.train(["الصبر مفتاح الفرج", "العلم نور"], num_merges=80)
    s = "الصبر"
    ids = tok.encode(s)
    print("vocab", n, "ids", ids.tolist(), "decode", repr(tok.decode(ids)))
