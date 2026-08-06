"""
مصنع موحّد لكل تقنيات الـtokenizer في NSM.

  from ai.tokenizer_factory import get_tokenizer, list_tokenizers
  tok = get_tokenizer("unigram", vocab_size=8192)
  tok = get_tokenizer("sentencepiece", vocab_path="models/...")
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# الاسم الرسمي → أسماء بديلة
ALIASES = {
    "word": "word",
    "wordlevel": "word",
    "bpe": "bpe",
    "wordpiece": "wordpiece",
    "wp": "wordpiece",
    "sentencepiece": "sentencepiece",
    "spm": "sentencepiece",
    "sp": "sentencepiece",
    "unigram": "unigram",
    "unigramlm": "unigram",
    "char": "char",
    "character": "char",
    "bytebpe": "byte_bpe",
    "byte_bpe": "byte_bpe",
    "bbpe": "byte_bpe",
    "hash": "hash",
}

DEFAULT_PATHS = {
    "word": "models/tokenizer_vocab.json",
    "bpe": "models/bpe_tokenizer.json",
    "wordpiece": "models/wordpiece_tokenizer.json",
    "sentencepiece": "models/sentencepiece_tokenizer.json",
    "unigram": "models/unigram_tokenizer.json",
    "char": "models/char_tokenizer.json",
    "byte_bpe": "models/byte_bpe_tokenizer.json",
}


def list_tokenizers() -> List[Dict[str, str]]:
    return [
        {"id": "word", "name": "Word-level", "desc": "قاموس كلمات كاملة"},
        {"id": "bpe", "name": "BPE", "desc": "Byte-Pair Encoding على الكلمات"},
        {"id": "wordpiece", "name": "WordPiece", "desc": "Longest-match مع ##"},
        {"id": "sentencepiece", "name": "SentencePiece-BPE", "desc": "BPE على محارف + ▁"},
        {"id": "unigram", "name": "Unigram LM", "desc": "تجزئة Viterbi بالاحتمالات"},
        {"id": "char", "name": "Character", "desc": "محرف = رمز"},
        {"id": "byte_bpe", "name": "Byte-level BPE", "desc": "BPE على بايتات UTF-8 (أسلوب GPT-2)"},
        {"id": "hash", "name": "Hash (متقادم)", "desc": "FNV بدون decode حقيقي"},
    ]


def normalize_type(name: str) -> str:
    key = (name or "word").strip().lower().replace("-", "").replace("_", "")
    return ALIASES.get(key, ALIASES.get(name.strip().lower(), "word"))


def get_tokenizer(
    tokenizer_type: str = "word",
    vocab_size: int = 8192,
    vocab_path: Optional[str] = None,
    **kwargs: Any,
):
    """يُنشئ tokenizer حسب النوع."""
    t = normalize_type(tokenizer_type)

    if t == "hash":
        from ai.arabic_transformer import HashTokenizer
        return HashTokenizer(vocab_size)

    if t == "bpe":
        from ai.bpe_tokenizer import BPETokenizer
        path = vocab_path or DEFAULT_PATHS["bpe"]
        return BPETokenizer(vocab_size, vocab_path=path)

    if t == "wordpiece":
        from ai.wordpiece_tokenizer import WordPieceTokenizer
        path = vocab_path or DEFAULT_PATHS["wordpiece"]
        return WordPieceTokenizer(vocab_size, vocab_path=path)

    if t == "sentencepiece":
        from ai.sentencepiece_tokenizer import SentencePieceTokenizer
        path = vocab_path or DEFAULT_PATHS["sentencepiece"]
        return SentencePieceTokenizer(vocab_size, vocab_path=path)

    if t == "unigram":
        from ai.unigram_tokenizer import UnigramTokenizer
        path = vocab_path or DEFAULT_PATHS["unigram"]
        return UnigramTokenizer(vocab_size, vocab_path=path)

    if t == "char":
        from ai.char_tokenizer import CharTokenizer
        path = vocab_path or DEFAULT_PATHS["char"]
        return CharTokenizer(vocab_size, vocab_path=path)

    if t == "byte_bpe":
        from ai.byte_bpe_tokenizer import ByteBPETokenizer
        path = vocab_path or DEFAULT_PATHS["byte_bpe"]
        return ByteBPETokenizer(vocab_size, vocab_path=path)

    # word
    from ai.arabic_transformer import WordTokenizer
    path = vocab_path or DEFAULT_PATHS["word"]
    return WordTokenizer(vocab_size, vocab_path=path)


def tokenizer_version_tag(tokenizer_type: str) -> str:
    t = normalize_type(tokenizer_type)
    return {
        "word": "word-v1",
        "bpe": "bpe-v1",
        "wordpiece": "wordpiece-v1",
        "sentencepiece": "sentencepiece-v1",
        "unigram": "unigram-v1",
        "char": "char-v1",
        "byte_bpe": "byte-bpe-v1",
        "hash": "hash-v1",
    }.get(t, "word-v1")


def tokenizer_save_name(tokenizer_obj) -> str:
    name = type(tokenizer_obj).__name__
    return {
        "BPETokenizer": "bpe_tokenizer.json",
        "WordPieceTokenizer": "wordpiece_tokenizer.json",
        "SentencePieceTokenizer": "sentencepiece_tokenizer.json",
        "UnigramTokenizer": "unigram_tokenizer.json",
        "CharTokenizer": "char_tokenizer.json",
        "ByteBPETokenizer": "byte_bpe_tokenizer.json",
        "WordTokenizer": "tokenizer_vocab.json",
    }.get(name, "tokenizer_vocab.json")
