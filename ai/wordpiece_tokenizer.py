"""
WordPiece Tokenizer للعربية — تنفيذ خفيف بدون مكتبات خارجية.

الفرق الجوهري عن BPE:
  - الترميز: Longest-Match من اليسار لليمين (أطول مقطع موجود في القاموس)
  - المقاطع غير الابتدائية تُسبَق بـ "##" (مثل BERT)
  - التدريب هنا تقريبي بالتكرار (شائع عملياً) وليس likelihood الكامل لـ Google

الواجهة متوافقة مع WordTokenizer / BPETokenizer في NSM:
  encode / decode / content_ids / save / load / vocab_id
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
CONT_PREFIX = "##"
DEFAULT_WP_PATH = "models/wordpiece_tokenizer.json"
MAX_SEQ_LEN = 128


def normalize_arabic(text: str) -> str:
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ة", "ه")
    return text


def split_words(text: str) -> List[str]:
    return re.findall(r"[\u0600-\u06FF]+|\d+", normalize_arabic(text))


class WordPieceTokenizer:
    """
    WordPiece ثنائي الاتجاه للعربية.
    """

    PAD, UNK, BOS, EOS, SEP, MASK = PAD, UNK, BOS, EOS, SEP, MASK
    OFFSET = OFFSET
    DEFAULT_VOCAB_PATH = DEFAULT_WP_PATH

    def __init__(self, vocab_size: int = 8192, vocab_path: Optional[str] = None):
        self.vocab_size = int(vocab_size)
        self.token_to_id: Dict[str, int] = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token: Dict[int, str] = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        # أطوال الرموز مرتبة تنازلياً لتسريع longest-match
        self._max_token_len = 1

        path = vocab_path or DEFAULT_WP_PATH
        if path and os.path.exists(path):
            self.load(path)

    def _add_token(self, tok: str) -> int:
        if tok in self.token_to_id:
            return self.token_to_id[tok]
        if len(self.token_to_id) >= self.vocab_size:
            return self.UNK
        idx = len(self.token_to_id)
        self.token_to_id[tok] = idx
        self.id_to_token[idx] = tok
        self._max_token_len = max(self._max_token_len, len(tok.replace(CONT_PREFIX, "")))
        return idx

    def train(self, texts: Iterable[str], num_merges: Optional[int] = None) -> int:
        """
        تدريب تقريبي بأسلوب دمج الأزواج (شبيه BPE) مع وسم ## للمقاطع الداخلية.

        1) كل كلمة → محارف منفصلة؛ الأول بدون ## والباقي ##c
        2) دمج أكثر الأزواج تكراراً حتى امتلاء القاموس
        """
        word_freq: Counter = Counter()
        for text in texts:
            for w in split_words(text):
                if w:
                    word_freq[w] += 1

        # تمثيل WordPiece الأولي
        corpus: Dict[Tuple[str, ...], int] = {}
        for word, freq in word_freq.items():
            if not word:
                continue
            pieces = [word[0]] + [CONT_PREFIX + ch for ch in word[1:]]
            corpus[tuple(pieces)] = corpus.get(tuple(pieces), 0) + freq

        self.token_to_id = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = {i: t for i, t in enumerate(SPECIAL_TOKENS)}
        # أضف كل الرموز الأحادية الظاهرة
        singles: Counter = Counter()
        for pieces, freq in corpus.items():
            for p in pieces:
                singles[p] += freq
        for tok, _ in singles.most_common():
            self._add_token(tok)

        target = num_merges if num_merges is not None else max(0, self.vocab_size - len(self.token_to_id))
        for _ in range(target):
            if len(self.token_to_id) >= self.vocab_size:
                break
            pairs = Counter()
            for pieces, freq in corpus.items():
                for i in range(len(pieces) - 1):
                    pairs[(pieces[i], pieces[i + 1])] += freq
            if not pairs:
                break
            (a, b), _f = pairs.most_common(1)[0]
            # الدمج: الأول يحتفظ بشكله؛ الثاني إن كان ## يُزال بادئته عند اللصق ثم يُعاد ## إن لزم
            if b.startswith(CONT_PREFIX):
                merged = a + b[len(CONT_PREFIX) :]
            else:
                merged = a + b
            # إذا لم يكن المقطع في بداية كلمة (a يبدأ بـ ## أو الدمج استمرار) أبقِ ##
            if a.startswith(CONT_PREFIX) and not merged.startswith(CONT_PREFIX):
                merged = CONT_PREFIX + merged
            self._add_token(merged)
            corpus = self._apply_merge(corpus, a, b, merged)

        self.vocab_size = max(self.vocab_size, len(self.token_to_id))
        self._refresh_max_len()
        return len(self.token_to_id)

    @staticmethod
    def _apply_merge(
        corpus: Dict[Tuple[str, ...], int], a: str, b: str, merged: str
    ) -> Dict[Tuple[str, ...], int]:
        new_c: Dict[Tuple[str, ...], int] = {}
        for pieces, freq in corpus.items():
            out: List[str] = []
            i = 0
            while i < len(pieces):
                if i < len(pieces) - 1 and pieces[i] == a and pieces[i + 1] == b:
                    out.append(merged)
                    i += 2
                else:
                    out.append(pieces[i])
                    i += 1
            key = tuple(out)
            new_c[key] = new_c.get(key, 0) + freq
        return new_c

    def _refresh_max_len(self) -> None:
        m = 1
        for tok in self.token_to_id:
            if tok in SPECIAL_TOKENS:
                continue
            core = tok[len(CONT_PREFIX) :] if tok.startswith(CONT_PREFIX) else tok
            m = max(m, len(core))
        self._max_token_len = m

    def _tokenize_word(self, word: str) -> List[str]:
        """Longest-match WordPiece على كلمة واحدة."""
        if not word:
            return []
        if word in self.token_to_id:
            return [word]

        tokens: List[str] = []
        start = 0
        n = len(word)
        while start < n:
            end = min(n, start + self._max_token_len)
            matched = None
            while end > start:
                piece = word[start:end]
                cand = piece if start == 0 else CONT_PREFIX + piece
                if cand in self.token_to_id:
                    matched = cand
                    break
                # جرّب أيضاً بدون ## في البداية إن فشل
                if start == 0 and piece in self.token_to_id:
                    matched = piece
                    break
                end -= 1
            if matched is None:
                # حرف واحد كـ UNK مسار: إن وُجد الحرف في القاموس استخدمه
                ch = word[start]
                cand = ch if start == 0 else CONT_PREFIX + ch
                if cand in self.token_to_id:
                    tokens.append(cand)
                elif ch in self.token_to_id:
                    tokens.append(ch)
                else:
                    tokens.append(SPECIAL_TOKENS[self.UNK])
                start += 1
            else:
                tokens.append(matched)
                start = end
        return tokens

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        ids = [self.BOS]
        for word in split_words(text):
            for piece in self._tokenize_word(word):
                if piece == SPECIAL_TOKENS[self.UNK]:
                    ids.append(self.UNK)
                else:
                    ids.append(self.token_to_id.get(piece, self.UNK))
        ids.append(self.EOS)
        return np.array(ids[:max_len], dtype=np.int64)

    def content_ids(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        ids: List[int] = []
        for word in split_words(text):
            for piece in self._tokenize_word(word):
                if piece == SPECIAL_TOKENS[self.UNK]:
                    ids.append(self.UNK)
                else:
                    ids.append(self.token_to_id.get(piece, self.UNK))
        return np.array(ids[:max_len], dtype=np.int64)

    def decode(self, ids, skip_special: bool = True) -> str:
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        special = {self.PAD, self.UNK, self.BOS, self.EOS, self.SEP, self.MASK}
        words: List[str] = []
        buf = ""
        for i in ids:
            i = int(i)
            if skip_special and i in special:
                continue
            tok = self.id_to_token.get(i, SPECIAL_TOKENS[self.UNK])
            if tok in SPECIAL_TOKENS:
                continue
            if tok.startswith(CONT_PREFIX):
                buf += tok[len(CONT_PREFIX) :]
            else:
                if buf:
                    words.append(buf)
                buf = tok
        if buf:
            words.append(buf)
        return " ".join(words)

    def vocab_id(self) -> int:
        return max(self.vocab_size, len(self.token_to_id))

    @property
    def word_to_id(self) -> Dict[str, int]:
        return self.token_to_id

    @property
    def id_to_word(self) -> Dict[int, str]:
        return self.id_to_token

    def save(self, path: Optional[str] = None) -> str:
        path = path or DEFAULT_WP_PATH
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)
        payload = {
            "version": "wordpiece-v1",
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
        self._refresh_max_len()


if __name__ == "__main__":
    texts = [
        "الصبر مفتاح الفرج",
        "التقوى من الايمان",
        "العلم نور والجهل ظلام",
        "الرحمه وسعت كل شيء",
        "الصابرون يوفون اجرهم بغير حساب",
    ]
    tok = WordPieceTokenizer(vocab_size=300)
    n = tok.train(texts, num_merges=120)
    s = "الصبر مفتاح الفرج"
    ids = tok.encode(s)
    print("vocab", n)
    print("ids", ids.tolist())
    print("pieces", [tok.id_to_token.get(int(i), "?") for i in ids])
    print("decode", tok.decode(ids))
