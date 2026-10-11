# -*- coding: utf-8 -*-
"""ai/task_timeouts.py — مهل المهام الطويلة (نموذج محلي على CPU) في مكان واحد.

نموذج 7B على CPU يولّد بضعة رموز في الثانية، وأول طلب يحمّل الأوزان في الذاكرة
(30–60ث وحدها)، فمهلة 60ث السابقة كانت تقطع الإجابات الطويلة وتُسقط العقدة إلى
Groq/HF دون داعٍ. الافتراضي الآن 180ث (NSM_LOCAL_TIMEOUT_SEC، من 5ث إلى 1800ث).

ورفع مهلة النموذج وحده لا يكفي: كل مهمة استدلال كانت تُنفَّذ **متزامنة داخل
حلقة أحداث العقدة** (async)، فاستدلال طويل يجمّد /status وكل الاتصالات (ويقتل
المشرف العقدة كمتجمّدة)؛ لذلك تُنقَل الآن إلى خيط (asyncio.to_thread)، وتُرفع
مهل انتظار الطرف الطالب بما يتناسب (remote_task_timeout) كي لا يستسلم قبل العامل.
"""
from __future__ import annotations

import os

DEFAULT_LOCAL_TIMEOUT = 180.0
MIN_LOCAL_TIMEOUT = 5.0
MAX_LOCAL_TIMEOUT = 1800.0

# أنواع المهام التي قد تُجيب منها العقدة بنموذج لغوي (ثابت mesh_task_protocol.KIND_INFERENCE)
_LLM_KINDS = ("inference_request",)


def local_timeout_sec() -> float:
    try:
        v = float(os.getenv("NSM_LOCAL_TIMEOUT_SEC", "") or DEFAULT_LOCAL_TIMEOUT)
    except ValueError:
        v = DEFAULT_LOCAL_TIMEOUT
    return min(max(v, MIN_LOCAL_TIMEOUT), MAX_LOCAL_TIMEOUT)


def local_llm_configured() -> bool:
    return bool(os.getenv("NSM_LOCAL_LLM_URL", "").strip())


def remote_task_timeout(kind: str, base: float) -> float:
    """مهلة انتظار نتيجة مهمة من عقدة أخرى. تُرفع لمهام النماذج اللغوية عند ضبط
    نموذج محلي (مهلة النموذج + 20ث هامش)؛ NSM_TASK_TIMEOUT_SEC تفرض قيمة صريحة."""
    explicit = os.getenv("NSM_TASK_TIMEOUT_SEC", "").strip()
    if explicit:
        try:
            return max(float(explicit), 1.0)
        except ValueError:
            pass
    if kind in _LLM_KINDS and local_llm_configured():
        return max(base, local_timeout_sec() + 20.0)
    return base
