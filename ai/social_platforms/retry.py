"""
طبقة الموثوقية الموحّدة — retry + exponential backoff + احترام rate limits.
==========================================================================
تُطبَّق كـ decorator على دوال المحولات (publish/fetch_new_items/reply) التي
تستدعي شبكات خارجية عبر requests. الهدف: تحمّل الأعطال المؤقتة (شبكة، 5xx،
429) دون أن يوقف ذلك المراقبة كاملةً، مع عدم إعادة المحاولة على أخطاء دائمة
(400/401/403/404) لأنها لن تُحَل بالتكرار.

القواعد:
- NotConfiguredError (بيانات اعتماد ناقصة) لا تُعاد محاولتها أبداً — تُرفع فوراً.
- 429 (Too Many Requests): تُحترم قيمة Retry-After من الخادم إن وُجدت، وإلا
  نستخدم backoff أُسّي + jitter.
- 5xx (خطأ خادم مؤقت) و أخطاء الشبكة (Timeout/ConnectionError): backoff أُسّي + jitter.
- 4xx غير 429 (خطأ دائم في الطلب/الصلاحيات): تُرفع فوراً بلا إعادة محاولة.
- بعد استنفاد كل المحاولات: يُرفع الاستثناء الأخير كما هو (لا يُبتلع الخطأ).
"""

from __future__ import annotations

import functools
import logging
import random
import time
from typing import Callable, Optional, Tuple, Type

import requests

from .base import NotConfiguredError

logger = logging.getLogger("nsm.social.retry")

# الاستثناءات التي تستحق إعادة محاولة (مشاكل شبكة عابرة، ليست أخطاء منطقية)
_RETRYABLE_NETWORK_EXC: Tuple[Type[BaseException], ...] = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)


def _retry_after_seconds(response: Optional["requests.Response"]) -> Optional[float]:
    """يقرأ رأس Retry-After من استجابة 429/503 إن وُجد (ثوانٍ أو تاريخ HTTP)."""
    if response is None:
        return None
    header = response.headers.get("Retry-After")
    if not header:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None  # صيغة تاريخ HTTP غير مدعومة هنا — نسقط لـ backoff الافتراضي


def _is_retryable_http_error(exc: requests.exceptions.HTTPError) -> bool:
    resp = exc.response
    if resp is None:
        return True  # لا يمكن تحديد الحالة — الأسلم إعادة المحاولة
    status = resp.status_code
    if status == 429:
        return True
    if 500 <= status < 600:
        return True
    return False  # 4xx دائم (401/403/404/400...) — لا فائدة من التكرار


def with_retry(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 20.0,
    jitter: float = 0.3,
):
    """Decorator لدوال المحولات التي تنفّذ طلبات HTTP.

    max_retries: عدد محاولات إعادة إضافية بعد المحاولة الأولى (فالمجموع = max_retries+1).
    base_delay/max_delay: حدود backoff الأُسّي بالثواني.
    jitter: نسبة عشوائية (0-1) تُضاف/تُطرح من التأخير لتفادي "thundering herd".
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc: Optional[BaseException] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except NotConfiguredError:
                    raise  # لا تُعاد المحاولة أبداً — ليست مشكلة شبكة
                except requests.exceptions.HTTPError as exc:
                    if not _is_retryable_http_error(exc) or attempt == max_retries:
                        raise
                    last_exc = exc
                    delay = _retry_after_seconds(exc.response)
                    if delay is None:
                        delay = min(max_delay, base_delay * (2 ** attempt))
                        delay += random.uniform(-jitter, jitter) * delay
                        delay = max(0.0, delay)
                    logger.warning(
                        "retry %s/%s لـ %s بعد %.2fs — %s",
                        attempt + 1, max_retries, getattr(func, "__qualname__", func), delay, exc,
                    )
                    time.sleep(delay)
                except _RETRYABLE_NETWORK_EXC as exc:
                    if attempt == max_retries:
                        raise
                    last_exc = exc
                    delay = min(max_delay, base_delay * (2 ** attempt))
                    delay += random.uniform(-jitter, jitter) * delay
                    delay = max(0.0, delay)
                    logger.warning(
                        "retry %s/%s لـ %s بعد %.2fs — خطأ شبكة: %s",
                        attempt + 1, max_retries, getattr(func, "__qualname__", func), delay, exc,
                    )
                    time.sleep(delay)
            if last_exc:  # pragma: no cover — احتياطي دفاعي، لا يجب الوصول له فعلياً
                raise last_exc
        return wrapper

    return decorator
