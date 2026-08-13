"""
نظام التقييم الذاتي (Self-Reflection) للوكلاء.

يراجع الوكيل أخطاءه تلقائياً ويصنّف سبب كل فشل ويقرر ما إذا كان قابلاً
للاستعادة، ثم يعيد المحاولة باستراتيجية مصححة دون تدخل المستخدم.

- لا يستدعي أي API خارجي: التحليل محلي (أنماط نصية) والاستراتيجية
  تصحيحية محلية (إعادة المحاولة مع انتظار قصير، تبديل المسار، تبسيط المدخل).
- يبث أحداثاً جديدة إلى ناقل الأحداث حتى تظهر دورات التصحيح في لوحة
  المراقبة: reflect_started، reflect_analysis، reflect_retry،
  reflect_resolved، reflect_gave_up.

التكامل الوحيد المطلوب مع الواجهة: تغليف استدعاء كل وكيل بدالة
`reflecting_call` من هذا الملف.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── أحداث جديدة في ناقل الأحداث (تُضاف دون تغيير الأنواع القديمة) ────────
EVENT_REFLECT_STARTED = "reflect_started"
EVENT_REFLECT_ANALYSIS = "reflect_analysis"
EVENT_REFLECT_RETRY = "reflect_retry"
EVENT_REFLECT_RESOLVED = "reflect_resolved"
EVENT_REFLECT_GAVE_UP = "reflect_gave_up"


class ReflectionPolicy:
    """سياسة التقييم الذاتي — ثوابت قابلة للضبط."""
    MAX_REFLECT_ROUNDS = 2          # دورتا مراجعة إضافيتان بعد الفشل الأول (3 محاولات إجمالاً)
    BACKOFF_SECONDS = 1.5           # انتظار قصير قبل إعادة المحاولة
    RECOVERABLE_PATTERNS = (
        # أنماط أخطاء عابرة قابلة للاستعادة
        r"rate.?limit", r"429", r"timeout", r"overloaded",
        r"service unavailable", r"5\d\d", r"temporary",
        r"refused", r"unavailable", r"busy", r"retry",
        r"empty", r"فارغ", r"تعذّر", r"فشل", r"خطأ",
        r"no provider", r"لا يوجد مزوّد",
    )
    # الأنماط التي تعني رفض الخدمة ولا يجب إعادة المحاولة عليها فوراً
    UNRECOVERABLE_HINTS = (
        r"content policy", r"violation", r"banned", r"account",
        r"authentication required", r"401", r"403",
    )

    @classmethod
    def backoff(cls, attempt_no: int) -> float:
        return cls.BACKOFF_SECONDS * min(attempt_no, 3)


class ReflectionFailure:
    """تشخيص فشل واحد: سبب مصنف + قرار قابلية الاستعادة + استراتيجية التصحيح."""

    def __init__(self, error_message: str, attempt_no: int, is_empty: bool = False):
        self.error_message = error_message
        self.attempt_no = attempt_no
        self.is_empty = is_empty
        self.reason = self._classify()
        self.recoverable = self._is_recoverable()
        self.strategy = self._choose_strategy()

    # ── تصنيف السبب ───────────────────────────────────────────────────────

    def _classify(self) -> str:
        text = f"{self.error_message}".lower()
        if self.is_empty:
            return "empty_response"
        if re.search(r"rate.?limit|429|overloaded|busy", text):
            return "rate_limit"
        if re.search(r"timeout|timed out", text):
            return "timeout"
        if re.search(r"no provider|لا يوجد مزوّد|unavailable|unreachable", text):
            return "provider_unavailable"
        if re.search(r"refused|4\d\d|5\d\d|http", text):
            return "transient_http"
        return "unknown"

    def _is_recoverable(self) -> bool:
        text = f"{self.error_message}".lower()
        for hint in ReflectionPolicy.UNRECOVERABLE_HINTS:
            if re.search(hint, text):
                return False
        return (
            self.is_empty
            or self.reason in {"rate_limit", "timeout", "provider_unavailable", "transient_http", "unknown"}
        )

    def _choose_strategy(self) -> str:
        if self.reason == "rate_limit":
            return "retry_with_backoff"
        if self.reason == "timeout":
            return "retry_with_backoff"
        if self.reason == "provider_unavailable":
            return "switch_provider_hint"
        if self.is_empty:
            return "simplify_prompt"
        return "retry_with_backoff"

    def to_event_metadata(self) -> Dict[str, Any]:
        return {
            "attempt_no": self.attempt_no,
            "reason": self.reason,
            "recoverable": self.recoverable,
            "strategy": self.strategy,
        }


class ReflectionContext:
    """سجل محاولات التقييم الذاتي لكل وكيل خلال المهمة الواحدة."""

    def __init__(self):
        self.attempt_no = 0
        self.reflections: List[ReflectionFailure] = []
    def record_failure(self, error_message: str, is_empty: bool = False) -> ReflectionFailure:
        # attempt_no يُحدَّث من reflecting_call قبل كل استدعاء؛ هنا نسجّل الفشل فقط
        failure = ReflectionFailure(error_message, self.attempt_no, is_empty)
        self.reflections.append(failure)
        return failure
    def should_retry(self) -> bool:
        # نسمح بدورات مراجعة إضافية تصل إلى MAX_REFLECT_ROUNDS بعد أول فشل.
        # عدد دورات الاستنفاد المتبقية = MAX_REFLECT_ROUNDS - عدد دورات المراجعة المستنفدة.
        if len(self.reflections) >= ReflectionPolicy.MAX_REFLECT_ROUNDS + 1:
            return False
        last = self.reflections[-1] if self.reflections else None
        return last is not None and last.recoverable

    def reset(self) -> None:
        self.attempt_no = 0
        self.reflections = []


def _emit(event_type: str, agent_id: str, title: str, status: str, detail: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    try:
        from ai.agent_event_bus import emit_event
        emit_event(event_type, agent_id=agent_id, title=title, status=status, detail=detail, metadata=metadata or {})
    except Exception as exc:
        logger.warning("Reflection event emission skipped: %s", exc)


def _is_bad_response(response: Optional[str]) -> bool:
    """يحدد ما إذا كان الرد الناجح ظاهرياً ردّاً فاشلاً يحتاج تقييماً ذاتياً."""
    if not response:
        return True
    text = response.strip()
    if not text:
        return True
    # استجابة لا تحمل محتوى حقيقياً أو تعلن فشلاً صريحاً
    if len(text) < 4 and not re.search(r"[a-zA-Z\u0600-\u06FF]", text):
        return True
    failure_decl = re.compile(
        r"^\s*(عذراً|عذرا|أسف|آسف|تعذر|تعذّر|فشل|لم أتمكن|لم أستطع|لم يتمكن|sorry|failed|could not)",
        re.IGNORECASE,
    )
    if failure_decl.match(text) and len(text) < 300:
        return True
    return False


def reflecting_call(
    agent_id: str,
    title: str,
    call_fn: Callable[[], str],
    context: ReflectionContext,
    on_retry: Optional[Callable[[int, Dict[str, Any]], None]] = None,
) -> str:
    """ينفذ استدعاء الوكيل مع حلقة تقييم ذاتي كاملة.

    - call_fn: دالة تنفيذ الوكيل الفعلي التي تعيد النص أو ترفع استثناء.
    - context: سجل المحاولات الحالي لهذه المهمة.
    - on_retry: خطاف اختياري يُستدعى قبل كل إعادة محاولة (لإعادة بث agent_started
      أو تعديل إعدادات الاستدعاء). تستقبل (attempt_no, strategy).

    النتيجة: نص الرد الناجح، أو إعادة رفع آخر استثناء بعد استنفاد الدورات
    مع بث reflect_gave_up.
    """
    last_error: Optional[str] = None
    while True:
        context.attempt_no += 1
        try:
            response = call_fn()
        except Exception as exc:
            last_error = str(exc)
            response = None
        if response is None or _is_bad_response(response):
            failure = context.record_failure(last_error or "", is_empty=(response is None))
            _emit(
                EVENT_REFLECT_STARTED,
                agent_id=agent_id, title=title, status="running",
                detail=f"مراجعة ذاتية بعد الفشل (محاولة {context.attempt_no})",
                metadata=failure.to_event_metadata(),
            )
            _emit(
                EVENT_REFLECT_ANALYSIS,
                agent_id=agent_id, title=title, status="running",
                detail=f"السبب المصنّف: {failure.reason} · الاستراتيجية: {failure.strategy}",
                metadata=failure.to_event_metadata(),
            )
            if not context.should_retry():
                _emit(
                    EVENT_REFLECT_GAVE_UP,
                    agent_id=agent_id, title=title, status="error",
                    detail=f"استُنفدت دورات التقييم الذاتي دون نجاح بعد {context.attempt_no} محاولة",
                    metadata=failure.to_event_metadata(),
                )
                if response is None and last_error:
                    raise RuntimeError(last_error) from None
                # ردّ فاشل ظاهرياً: نعيد رميه كخطأ قابل للاكتشاف
                raise RuntimeError(f"فشل الوكيل رغم التقييم الذاتي: {failure.reason}")
            _emit(
                EVENT_REFLECT_RETRY,
                agent_id=agent_id, title=title, status="running",
                detail=f"إعادة المحاولة بعد {ReflectionPolicy.backoff(context.attempt_no):.1f}ث — {failure.strategy}",
                metadata=failure.to_event_metadata(),
            )
            if on_retry is not None:
                on_retry(context.attempt_no, {"strategy": failure.strategy, "reason": failure.reason})
            time.sleep(ReflectionPolicy.backoff(context.attempt_no))
            continue

        # نجاح — بث نتيجة المراجعة إن سبقت المحاولة تقييمات
        if context.reflections:
            _emit(
                EVENT_REFLECT_RESOLVED,
                agent_id=agent_id, title=title, status="done",
                detail=f"تصحيح تلقائي ناجح بعد {context.attempt_no} محاولة",
                metadata={"attempts": context.attempt_no, "final_strategy": context.reflections[-1].strategy},
            )
        return response
