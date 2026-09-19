"""
NSM Tier Gate — core/tier.py
=============================
نقطة تمديد بسيطة تفصل بين ميزات النسخة المجانية (Core) وميزات مستقبلية
مدفوعة (Pro)، دون التأثير على أي سلوك حالي.

الاستخدام:
    from core.tier import is_pro_enabled, require_pro

    if is_pro_enabled():
        ...  # تفعيل ميزة Pro

    @require_pro("multi_agent_dashboard")
    def some_pro_only_function(...):
        ...

يتم التفعيل حالياً فقط عبر متغير البيئة NSM_PRO_LICENSE (فارغ أو غير
موجود = وضع Core المجاني الافتراضي). لا يوجد أي تحقق عن بُعد أو اتصال
شبكي هنا — هذا مجرد "مفتاح" محلي جاهز للتوسعة لاحقاً.
"""
from __future__ import annotations

import functools
import os
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def is_pro_enabled() -> bool:
    """
    يعيد True إذا كانت النسخة الحالية مفعّلة كـ Pro (عبر متغير بيئة محلي).
    الافتراضي: False (النسخة المجانية Core).
    """
    return bool(os.environ.get("NSM_PRO_LICENSE", "").strip())


class ProFeatureLocked(Exception):
    """تُرفع عند محاولة استخدام ميزة Pro في نسخة Core."""

    def __init__(self, feature_name: str):
        self.feature_name = feature_name
        super().__init__(
            f"الميزة '{feature_name}' متاحة فقط في نسخة NSM Pro. "
            "فعّل NSM_PRO_LICENSE أو راجع خطط الترقية."
        )


def require_pro(feature_name: str) -> Callable[[F], F]:
    """
    ديكوريتور لتغليف أي دالة تمثل ميزة Pro مستقبلية.
    لا يغيّر أي دالة حالية — يُستخدم فقط عند إضافة ميزات جديدة مدفوعة.
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if not is_pro_enabled():
                raise ProFeatureLocked(feature_name)
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
