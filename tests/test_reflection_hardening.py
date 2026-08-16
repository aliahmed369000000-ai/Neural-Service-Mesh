"""اختبارات تقوية نظام التقييم الذاتي — لا تستدعي أي API خارجي."""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from ai.agent_reflection import (  # noqa: E402
    ReflectionContext,
    ReflectionFailure,
    ReflectionPolicy,
    reflection_summary,
)


def test_new_arabic_classes():
    """الأصناف الجديدة تصنّف الأخطاء العربية والإنجليزية بدقة."""
    cases = [
        ("انقطع الاتصال بالشبكة", "network_error"),
        ("خطأ في الاتصال: socket timeout", "network_error"),
        ("نفدت حصة الاستخدام لهذا الشهر", "quota_exceeded"),
        ("DNS resolve error", "network_error"),
        ("invalid api key unauthorized", "auth_issue"),
        ("JSON parse failed", "format_error"),
        ("context window exceeded", "context_overflow"),
        ("انقطاع مزود لا يوجد مزوّد", "provider_unavailable"),
        ("لا استجابة من الخادم", "timeout"),
    ]
    for msg, expected in cases:
        f = ReflectionFailure(msg, 1)
        assert f.reason == expected, f"{msg!r} → {f.reason!r} expected {expected!r}"


def test_decay_stops_repetition():
    """FAILURE_DECAY يوقف التكرار العقيم: سببان متكرران → لا إعادة محاولة."""
    ctx = ReflectionContext()
    ctx.record_failure("انقطع الاتصال بالشبكة", is_empty=False)
    ctx.record_failure("انقطع الاتصال بالشبكة", is_empty=False)
    # تكرار نفس السبب مرتين → should_retry يجب أن يعود False
    assert not ctx.should_retry(), "التكرار العقيم يجب أن يوقف إعادة المحاولة"


def test_different_reasons_keep_retrying():
    """أسباب مختلفة لا تفعّل decay."""
    ctx = ReflectionContext()
    ctx.record_failure("rate limit 429", is_empty=False)
    ctx.record_failure("socket timeout", is_empty=False)
    assert ctx.should_retry(), "الأسباب المختلفة لا تفعّل decay"


def test_max_rounds_four_attempts():
    """MAX_REFLECT_ROUNDS=3 → 4 محاولات إجمالية كحد أقصى."""
    ctx = ReflectionContext()
    for i in range(4):
        ctx.record_failure("transient error", is_empty=False)
    assert not ctx.should_retry()


def test_backoff_cap():
    """الانتظار مسقوف بـ MAX_BACKOFF_SECONDS."""
    assert ReflectionPolicy.backoff(100) == ReflectionPolicy.MAX_BACKOFF_SECONDS
    assert ReflectionPolicy.backoff(1) == ReflectionPolicy.BACKOFF_SECONDS


def test_summary_format():
    """reflection_summary يعيد ملخصًا عربيًا صالحًا للوحات."""
    ctx = ReflectionContext()
    ctx.attempt_no = 3
    ctx.record_failure("rate limit 429", is_empty=False)
    ctx.record_failure("rate limit 429", is_empty=False)
    s = reflection_summary(ctx)
    assert s["rounds"] == 2
    assert s["repeated_reason"] is True
    assert s["patterns"].get("rate_limit") == 2


def test_empty_context_summary():
    ctx = ReflectionContext()
    assert reflection_summary(ctx)["rounds"] == 0


def test_py_compile():
    import py_compile
    py_compile.compile(str(HERE / "ai/agent_reflection.py"), doraise=True)
