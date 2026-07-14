"""
محول Pinterest — عبر Pinterest API v5 الرسمي.
يتطلب: PINTEREST_ACCESS_TOKEN، PINTEREST_BOARD_ID (اللوحة الافتراضية
للنشر — كل Pin على Pinterest يجب أن ينتمي للوحة، لا يوجد "نشر عام" بلا
لوحة)، PINTEREST_DEFAULT_IMAGE_URL.

⚠️ قيدان حقيقيان من المنصة نفسها (موثّقان هنا بدل التحايل عليهما):

1. Pinterest لا يدعم "منشور نصي" إطلاقاً — كل Pin عبارة عن صورة أو فيديو
   إلزامياً (`media_source`)، والنص (عنوان/وصف) عنصر مصاحب فقط وليس
   المحتوى الأساسي. توحيد publish(text) مع بقية المنصات (التي تنشر نصاً
   خالصاً) يتطلب صورة افتراضية — لذلك PINTEREST_DEFAULT_IMAGE_URL مطلوب
   إلزامياً. بدونه لا يوجد "نشر نص فقط" ممكن تقنياً على Pinterest، ونرفض
   بوضوح بدل تلفيق صورة أو تجاهل القيد.

2. Pinterest API v5 **لا يوفّر أي endpoint** لقراءة أو الرد على تعليقات
   الـPins (تأكيد موثّق من وثائق Pinterest ومصادر مطوّرين مستقلة عدّة —
   "Reading or replying to comments is not offered by Pinterest's API").
   لذلك: `supports_monitoring = False`، وfetch_new_items/reply يرفعان
   PlatformCapabilityError بدل استدعاء endpoint غير موجود أو إرجاع نتائج
   فارغة صامتة قد تُفهم خطأً كـ"لا تعليقات جديدة".
"""

from __future__ import annotations

import os
from typing import List

import requests

from .base import PlatformAdapter, SocialItem, PlatformCapabilityError
from .retry import with_retry

API_BASE = "https://api.pinterest.com/v5"


class PinterestAdapter(PlatformAdapter):
    platform_id = "pinterest"
    required_env = ["PINTEREST_ACCESS_TOKEN", "PINTEREST_BOARD_ID", "PINTEREST_DEFAULT_IMAGE_URL"]
    supports_monitoring = False  # Pinterest API v5: لا comments endpoint إطلاقاً

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {os.environ['PINTEREST_ACCESS_TOKEN']}",
            "Content-Type": "application/json",
        }

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        title = (text[:97] + "...") if len(text) > 100 else text
        r = requests.post(
            f"{API_BASE}/pins",
            headers=self._headers(),
            json={
                "board_id": os.environ["PINTEREST_BOARD_ID"],
                "title": title,
                "description": text,
                "media_source": {
                    "source_type": "image_url",
                    "url": os.environ["PINTEREST_DEFAULT_IMAGE_URL"],
                },
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        raise PlatformCapabilityError(
            "pinterest: Pinterest API v5 لا يوفّر أي endpoint لقراءة تعليقات "
            "الـPins — لا مراقبة ممكنة لهذه المنصة، القيد من المنصة نفسها."
        )

    def reply(self, item: SocialItem, text: str) -> str:
        raise PlatformCapabilityError(
            "pinterest: Pinterest API v5 لا يوفّر أي endpoint للرد على "
            "تعليقات الـPins — القيد من المنصة نفسها."
        )
