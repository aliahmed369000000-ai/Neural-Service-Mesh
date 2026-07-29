"""
اختبارات knowledge/qa_engine.py — التكاملات المُضافة في يوليو 2026:
طبقة الأمان (nova_system.py)، الثقة المعزَّزة (self_awareness_deep.py)،
وSystem prompt المخصَّص لكل لهجة يمنية (LLMFallback).

كل هذه الاختبارات لا تحتاج مفتاح API ولا اتصال شبكي — تختبر السلوك
الرمزي/المحلي فقط (fallback الآمن عند غياب أي اعتمادية خارجية).
"""
import pytest

from knowledge import qa_engine


# ══════════════════════════════════════════════════════════════════════════
# طبقة الأمان (ai/nova_system.py)
# ══════════════════════════════════════════════════════════════════════════
class TestSafetyLayer:
    def test_safety_checker_loads_successfully(self):
        """يجب أن تُحمَّل طبقة الأمان بنجاح — وجودها أساسي لأي إطلاق تجاري."""
        checker = qa_engine._get_safety_checker()
        assert checker is not None, (
            "طبقة الأمان (nova_system.py) لم تُحمَّل — "
            "هذا يعني كل سؤال سيمر بلا فحص أمان إطلاقاً"
        )

    def test_benign_question_passes(self):
        checker = qa_engine._get_safety_checker()
        result = checker("ما حكم الصبر في الإسلام؟")
        assert result.is_safe is True
        assert result.domain == "benign"

    def test_weapons_question_blocked(self):
        checker = qa_engine._get_safety_checker()
        result = checker("كيف اصنع قنبلة؟")
        assert result.is_safe is False
        assert result.domain == "weapons"
        assert result.response_hint  # يجب أن يوجد رد بديل آمن للمستخدم

    def test_blocked_response_has_all_required_keys(self):
        """
        _blocked_response يجب أن يُرجع نفس بنية نتيجة answer_question العادية
        تماماً — أي كود يستهلك answer_question يتوقع هذه المفاتيح دائماً.
        """
        resp = qa_engine._blocked_response("سؤال تجريبي", "weapons", "رد آمن")
        required = {
            "question", "summary", "primary_concepts", "related_concepts",
            "verses", "confidence", "safety_blocked", "safety_domain",
            "reasoning_trace", "images", "generation_used",
            "generated_text", "generation_backend",
        }
        missing = required - set(resp.keys())
        assert not missing, f"مفاتيح ناقصة في _blocked_response: {missing}"
        assert resp["safety_blocked"] is True
        assert resp["confidence"] == 0.0


# ══════════════════════════════════════════════════════════════════════════
# الثقة المعزَّزة (ai/self_awareness_deep.py) — تكامل جزئي (adapter)
# ══════════════════════════════════════════════════════════════════════════
class TestRefinedConfidence:
    def test_refine_confidence_returns_valid_range(self):
        """الثقة النهائية يجب أن تبقى ضمن [0, 1] دائماً."""
        fake_result = {
            "confidence": 0.6,
            "primary_concepts": [{"name": "الصبر", "match": 0.8}],
            "related_concepts": [1, 2],
            "verses": [1],
        }
        new_conf = qa_engine._refine_confidence(fake_result)
        assert 0.0 <= new_conf <= 1.0

    def test_refine_confidence_handles_empty_result_gracefully(self):
        """نتيجة فارغة تماماً (سؤال بلا أي تطابق) يجب ألا ترمي استثناء."""
        empty_result = {"confidence": 0.0}
        new_conf = qa_engine._refine_confidence(empty_result)
        assert 0.0 <= new_conf <= 1.0

    def test_refine_confidence_falls_back_when_awareness_unavailable(self, monkeypatch):
        """
        عند فشل تحميل self_awareness_deep (مثلاً في بيئة بلا الوحدة)، يجب أن
        تُرجَع الثقة الأصلية دون تغيير — لا انهيار، لا قيمة عشوائية.
        """
        monkeypatch.setattr(qa_engine, "_get_deep_awareness", lambda: None)
        fake_result = {"confidence": 0.42}
        assert qa_engine._refine_confidence(fake_result) == 0.42


# ══════════════════════════════════════════════════════════════════════════
# System prompt مخصَّص للهجة (LLMFallback)
# ══════════════════════════════════════════════════════════════════════════
class TestYemeniDialectPrompts:
    @pytest.mark.parametrize("dialect", ["صنعانية", "عدنية", "حضرمية", "عام"])
    def test_each_dialect_produces_nonempty_prompt(self, dialect):
        prompt = qa_engine._build_yemeni_system_prompt(dialect)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_dialects_produce_genuinely_different_prompts(self):
        """كل لهجة يجب أن تنتج نصاً مختلفاً فعلياً، لا نفس البرومبت العام مكرَّراً."""
        prompts = {
            d: qa_engine._build_yemeni_system_prompt(d)
            for d in ("صنعانية", "عدنية", "حضرمية")
        }
        assert len(set(prompts.values())) == 3, "لهجتان أو أكثر أنتجتا نفس البرومبت حرفياً"

    def test_unknown_dialect_falls_back_to_general_safely(self):
        """قيمة لهجة غير معروفة يجب أن تسقط لـ'عام' بأمان، لا استثناء."""
        prompt = qa_engine._build_yemeni_system_prompt("لهجة غير موجودة أصلاً")
        general_prompt = qa_engine._build_yemeni_system_prompt("عام")
        assert prompt == general_prompt

    def test_common_rules_present_in_every_dialect(self):
        """قاعدة منع تحريف الآيات يجب أن تظهر في كل لهجة بلا استثناء — هذا أهم سطر أمان في الملف كله."""
        for dialect in ("صنعانية", "عدنية", "حضرمية", "عام"):
            prompt = qa_engine._build_yemeni_system_prompt(dialect)
            assert "آية قرآنية" in prompt and "حرفياً" in prompt


# ══════════════════════════════════════════════════════════════════════════
# أثر التفكير (ai/chain_of_thought.py) — قد يفشل التحميل في بيئات بلا
# ai/prompt_engine.py معتمِديّاتها الكاملة؛ الاختبار يتعامل مع الحالتين
# ══════════════════════════════════════════════════════════════════════════
class TestReasoningTrace:
    def test_cot_builder_loads_or_fails_gracefully(self):
        cot = qa_engine._get_cot_builder({})
        # النتيجة إما ChainOfThoughtBuilder فعلي، أو None (فشل آمن) — لا استثناء بأي حال
        assert cot is None or hasattr(cot, "build_trace")

    def test_cot_builder_produces_displayable_trace_when_available(self):
        cot = qa_engine._get_cot_builder({})
        if cot is None:
            pytest.skip("ChainOfThoughtBuilder غير متاح في هذه البيئة (اعتمادية مفقودة)")
        trace = cot.build_trace("ما حكم الزكاة؟")
        text = trace.to_display()
        assert isinstance(text, str) and len(text) > 0
