"""
اختبارات ai/yemeni_tokenizer.py — سقف MAX_VOCAB_SIZE=8192 وتجزئة subword
الاحتياطية عند امتلاء القاموس (أُضيفت يوليو 2026 لمنع IndexError عند
الربط بطبقة إخراج ثابتة الحجم، ولتقليل انفجار القاموس مع اللهجات).
"""
import random

import pytest

from ai.yemeni_tokenizer import (
    YemeniTokenizer, MAX_VOCAB_SIZE, _SUBWORD_HASH_START,
)


@pytest.fixture
def tokenizer():
    return YemeniTokenizer(grow_vocab=True)


class TestVocabCap:
    def test_ids_never_exceed_max_vocab_size(self, tokenizer):
        """
        اختبار الانحدار الأهم في هذا الملف: قبل هذا الإصلاح، ضخ كلمات كافية
        كان يُنتج IDs تتجاوز حجم طبقة الإخراج الثابتة → IndexError عند
        الاستدلال. هذا الاختبار يضخ 9000 كلمة عشوائية (أكثر بكثير من السقف)
        ويتحقق أن كل ID صادر يبقى ضمن النطاق دائماً.
        """
        random.seed(42)
        letters = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
        words = ["".join(random.choices(letters, k=6)) for _ in range(9000)]

        max_id_seen = -1
        for w in words:
            ids = tokenizer._word_or_subwords_to_ids(w)
            assert all(0 <= i < MAX_VOCAB_SIZE for i in ids), (
                f"ID خارج النطاق المسموح لكلمة '{w}': {ids}"
            )
            max_id_seen = max(max_id_seen, *ids)

        assert max_id_seen < MAX_VOCAB_SIZE

    def test_vocab_size_never_exceeds_subword_hash_start(self, tokenizer):
        """
        منطقة الكلمات الكاملة (word2id) يجب ألا تتجاوز _SUBWORD_HASH_START
        أبداً — الباقي محجوز لتجزئة subword الاحتياطية.
        """
        random.seed(1)
        letters = "ابتثجحخدذرزسشصضطظعغفقكلمنهوي"
        for _ in range(5000):
            w = "".join(random.choices(letters, k=7))
            tokenizer._word_or_subwords_to_ids(w)
        assert tokenizer.vocab_size <= _SUBWORD_HASH_START


class TestRoundTrip:
    def test_known_words_decode_perfectly(self, tokenizer):
        """كلمات ضمن سعة القاموس العادية يجب أن تُفَك بدقة 100%."""
        text = "الحمد لله رب العالمين"
        ids = tokenizer.encode(text)
        assert tokenizer.decode(ids) == text

    def test_total_id_space_matches_max_vocab_size(self, tokenizer):
        assert tokenizer.total_id_space == MAX_VOCAB_SIZE
