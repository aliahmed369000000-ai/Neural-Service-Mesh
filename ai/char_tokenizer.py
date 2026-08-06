"""
Character-level Tokenizer للعربية.

كل محرف Unicode = رمز مستقل.
مزايا: صفر UNK تقريباً، بسيط جداً.
عيوب: تسلسلات أطول → أبطأ على نفس max_seq.
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

import numpy as np

PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>", "<MASK>")
DEFAULT_PATH = "models/char_tokenizer.json"
MAX_SEQ_LEN = 128


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


class CharTokenizer:
    PAD, UNK, BOS, EOS, SEP, MASK = PAD, UNK, BOS, EOS, SEP, MASK
    DEFAULT_VOCAB_PATH = DEFAULT_PATH

    def __init__(self, vocab_size: int = 512, vocab_path: Optional[str] = None):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        path = vocab_path or DEFAULT_PATH
        if path and os.path.exists(path):
            self.load(path)
        else:
            self._seed_arabic()

    def _seed_arabic(self) -> None:
        seed = (
            " \n\t"
            + "0123456789"
            + "ءآأؤإئابةتثجحخدذرزسشصضطظعغفقكلمنهويىة"
            + "،؛؟!.-:،"
        )
        for ch in seed:
            self._add(ch)

    def _add(self, ch: str) -> int:
        if ch in self.token_to_id:
            return self.token_to_id[ch]
        if len(self.token_to_id) >= self.vocab_size:
            return self.UNK
        idx = len(self.token_to_id)
        self.token_to_id[ch] = idx
        self.id_to_token[idx] = ch
        return idx

    def train(self, texts: Iterable[str], max_vocab: Optional[int] = None) -> int:
        cap = max_vocab or self.vocab_size
        counts: Counter = Counter()
        for t in texts:
            counts.update(normalize_arabic(str(t)))
        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        for ch, _ in counts.most_common(max(0, cap - len(SPECIAL_TOKENS))):
            self._add(ch)
        self.vocab_size = max(self.vocab_size, len(self.token_to_id))
        return len(self.token_to_id)

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        text = normalize_arabic(text)
        ids = [self.BOS] + [self.token_to_id.get(ch, self.UNK) for ch in text] + [self.EOS]
        return np.array(ids[:max_len], dtype=np.int64)

    def content_ids(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        text = normalize_arabic(text)
        return np.array(
            [self.token_to_id.get(ch, self.UNK) for ch in text][:max_len],
            dtype=np.int64,
        )

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        special = {self.PAD, self.UNK, self.BOS, self.EOS, self.SEP, self.MASK}
        out = []
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            tok = self.id_to_token.get(i, "")
            if tok in SPECIAL_TOKENS:
                continue
            out.append(tok)
        return "".join(out)

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
            "version": "char-v1",
            "vocab_size": self.vocab_size,
            "token_to_id": self.token_to_id,
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
        self.vocab_size = int(data.get("vocab_size", max(self.vocab_size, len(self.token_to_id))))
        for i, tok in enumerate(SPECIAL_TOKENS):
            self.token_to_id[tok] = i
            self.id_to_token[i] = tok


if __name__ == "__main__":
    tok = CharTokenizer(512)
    tok.train(["الصبر مفتاح الفرج", "العلم نور"])
    ids = tok.encode("الصبر")
    print(ids.tolist(), tok.decode(ids))
