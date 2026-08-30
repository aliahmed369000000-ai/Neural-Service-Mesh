"""طبقة تنسيق خفيفة بين الوكيل المركزي والوكلاء المتخصصين.

تحدد هذه الوحدة متى تكون المهمة مركبة بما يكفي للتفويض، وتترك التنفيذ
الفِعلي للـ UnifiedAgentChat الموجود في المشروع. لا تستدعي أي مزود خارجي
ولا تنفذ أدوات من تلقاء نفسها؛ لذلك يمكن اختبارها محلياً دون مفاتيح API.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple


_COMPLEX_HINTS = (
    "حلل", "حلّل", "قارن", "راجع", "قيّم", "تقرير", "استراتيجية",
    "خطة", "شامل", "متكامل", "من جميع الجوانب", "ابحث ثم", "ثم اكتب",
    "عدة زوايا", "وكلاء", "agents", "analyze", "compare", "review",
    "strategy", "comprehensive", "report", "research and",
)
_CODING_HINTS = (
    "كود", "كوداً", "برمج", "برمجية", "ملف", "مستودع", "github", "git",
    "streamlit", "python", "عدّل", "عدل", "أنشئ ملف", "أنشئ تطبيق",
    "اكتب دالة", "اختبر الكود", "commit", "push", "code", "repository",
)
_DIRECT_HINTS = (
    "حالة المهام", "قائمة المهام", "مهامي", "حالة الحوكمة",
    "حالة النظام الذاتي", "حالة الأمان الذاتي",
)


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    lowered = (text or "").strip().lower()
    return any(h.lower() in lowered for h in hints)


def should_delegate_request(text: str) -> bool:
    """يقرر هل يستحق الطلب فريقاً من الوكلاء المتخصصين.

    المهام البرمجية والأوامر المباشرة تبقى في مسار NSMAgent الحالي لأنه
    يملك أدوات القراءة/التعديل/الاختبار والـ self-healing. أما الطلبات
    التحليلية المركبة فتذهب إلى UnifiedAgentChat الذي يختار حتى ثلاثة
    متخصصين ثم يولّف إجابة نهائية واحدة.
    """
    value = re.sub(r"\s+", " ", (text or "").strip())
    if not value or _contains_any(value, _DIRECT_HINTS):
        return False
    if _contains_any(value, _CODING_HINTS):
        return False
    return len(value) >= 40 and (
        _contains_any(value, _COMPLEX_HINTS)
        or value.count(" و") >= 2
        or value.count(" ثم ") >= 1
    )


def delegate_to_unified_chat(
    text: str,
    orchestrator: Optional[Any] = None,
) -> Tuple[str, Dict[str, Any]]:
    """يفوض الطلب إلى المدير الموحّد مع إنشاء نسخة كسولة عند الحاجة."""
    if orchestrator is None:
        from ai.agent_categories import UnifiedAgentChat
        orchestrator = UnifiedAgentChat()
    response, meta = orchestrator.chat(text)
    return str(response or "").strip(), dict(meta or {})


def classify_request(text: str) -> Dict[str, Any]:
    """يصنف الطلب محلياً قبل استدعاء النموذج لتقليل التخمين والتكلفة."""
    value = re.sub(r"\s+", " ", (text or "").strip())
    if not value:
        return {"route": "direct", "confidence": 1.0, "reasons": ["empty"]}
    if _contains_any(value, _CODING_HINTS):
        route, reasons = "coding", ["coding_hint"]
    elif _contains_any(value, _DIRECT_HINTS):
        route, reasons = "direct", ["direct_command"]
    elif _contains_any(value, _COMPLEX_HINTS) or value.count(" و") >= 2 or " ثم " in value:
        route, reasons = "orchestrated", ["complexity_hint"]
    else:
        route, reasons = "conversation", ["default"]
    confidence = 0.92 if reasons[0] != "default" else 0.62
    return {"route": route, "confidence": confidence, "reasons": reasons, "language": "ar" if re.search(r"[\u0600-\u06ff]", value) else "en"}


__all__ = ["should_delegate_request", "delegate_to_unified_chat", "classify_request"]
