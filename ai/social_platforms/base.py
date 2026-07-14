"""
الواجهة الأساسية لكل محولات المنصات — عقد ثابت يضمن أن الوكيل الموحد
يتعامل مع كل منصة بنفس الطريقة (نشر / جلب جديد / رد).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional


class NotConfiguredError(RuntimeError):
    """تُرفع عندما تكون بيانات اعتماد المنصة غير متوفرة — لا نلفّق نتائج بديلة أبداً."""


class PlatformCapabilityError(RuntimeError):
    """تُرفع عندما تكون العملية غير مدعومة إطلاقاً بواجهة برمجة المنصة
    الرسمية (وليست مسألة بيانات اعتماد ناقصة) — مثال: Pinterest API v5
    لا يوفّر أي endpoint لقراءة أو الرد على التعليقات مهما كانت الصلاحيات
    الممنوحة. تُستخدم بدل NotImplementedError الصامت لتوضيح أن القيد من
    المنصة نفسها، وبدل NotConfiguredError لأن المشكلة ليست بيانات اعتماد."""


@dataclass
class SocialItem:
    """عنصر وارد من منصة (منشن، تعليق، رسالة خاصة) بصيغة موحّدة."""
    platform: str
    external_id: str          # معرف فريد على المنصة نفسها (لتفادي التكرار)
    kind: str                 # "mention" | "comment" | "dm" | "reply_target"
    author: str
    text: str
    thread_id: Optional[str] = None   # للرد ضمن نفس الخيط/المحادثة إن وُجد
    url: Optional[str] = None
    raw: dict = field(default_factory=dict)


class PlatformAdapter:
    """
    عقد أساسي لكل محول منصة. المحولات الفعلية (Discord/Telegram/...)
    تُطبّق هذه الدوال. أي دالة غير مطبَّقة أو بدون بيانات اعتماد ترفع
    NotConfiguredError — لا نتائج مزيّفة، لا صمت عن الفشل.
    """
    platform_id: str = "base"
    #: أسماء متغيرات البيئة/الأسرار المطلوبة لتفعيل هذا المحول
    required_env: List[str] = []
    #: هل تدعم هذه المنصة webhook حقيقي (استقبال أحداث بدلاً من polling)؟
    #: True فقط للمنصات التي توفّر فعلياً HTTP push API موثّق للأحداث
    #: الواردة (وليس فقط webhooks صادرة للنشر). راجع WEBHOOKS.md لتفصيل
    #: كل منصة وسبب True/False قبل تغيير هذه القيمة.
    supports_webhook: bool = False
    #: هل تدعم هذه المنصة مراقبة (fetch_new_items) ورداً (reply) أصلاً عبر
    #: API عام موثّق؟ False لمنصات تسمح بالنشر فقط (مثل Pinterest API v5
    #: الذي لا يوفّر أي endpoint للتعليقات) — عندها fetch_new_items/reply
    #: يرفعان PlatformCapabilityError بدل محاولة استدعاء endpoint غير موجود.
    #: دورة الاستطلاع بـsocial_agent.py تتخطى استدعاء fetch_new_items
    #: أصلاً لهذه المنصات (لا رفع أخطاء متكررة بلا فائدة بالسجل).
    supports_monitoring: bool = True

    def is_configured(self) -> bool:
        return all(os.environ.get(k) for k in self.required_env)

    def missing_env(self) -> List[str]:
        return [k for k in self.required_env if not os.environ.get(k)]

    def _require_configured(self):
        if not self.is_configured():
            raise NotConfiguredError(
                f"{self.platform_id}: بيانات الاعتماد المفقودة — {', '.join(self.missing_env())}"
            )

    # ── الواجهة التي يجب على كل محول تطبيقها ────────────────────────────
    def publish(self, text: str) -> str:
        """ينشر منشوراً/رسالة جديدة. يعيد المعرف الخارجي للمنشور."""
        raise NotImplementedError

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        """يجلب العناصر الجديدة (منشن/تعليقات/رسائل) غير الموجودة في since_ids."""
        raise NotImplementedError

    def reply(self, item: SocialItem, text: str) -> str:
        """يرد على عنصر محدد. يعيد المعرف الخارجي للرد."""
        raise NotImplementedError
