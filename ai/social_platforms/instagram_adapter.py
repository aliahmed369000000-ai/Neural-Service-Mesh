"""
محول Instagram — عبر Meta Graph API لحساب Instagram Business/Creator
المرتبط بصفحة فيسبوك.

يتطلب: INSTAGRAM_ACCESS_TOKEN (Page Access Token طويل الأمد)،
INSTAGRAM_BUSINESS_ACCOUNT_ID.

المراقبة: يجلب التعليقات على آخر 10 وسائط للحساب، والردود عليها.
الإشارات (mentions) تتطلب صلاحية instagram_manage_insights الإضافية.

ملاحظة: Instagram Graph API لا يدعم نشر نص بلا صورة/فيديو — قيد من المنصة.
"""

from __future__ import annotations

import os
from typing import List

import requests

from .base import PlatformAdapter, SocialItem, NotConfiguredError
from .retry import with_retry

GRAPH = "https://graph.facebook.com/v19.0"


class InstagramAdapter(PlatformAdapter):
    platform_id = "instagram"
    required_env = ["INSTAGRAM_ACCESS_TOKEN", "INSTAGRAM_BUSINESS_ACCOUNT_ID"]

    @with_retry()
    def publish(self, text: str) -> str:
        # Instagram Graph API لا يسمح بمنشور نصي بلا صورة/فيديو — هذا قيد
        # من المنصة نفسها وليس نقصاً في الكود.
        raise NotConfiguredError(
            "Instagram لا يدعم نشر نص فقط عبر Graph API — يلزم رابط صورة/فيديو عام."
        )

    @with_retry()
    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        acc = os.environ["INSTAGRAM_BUSINESS_ACCOUNT_ID"]
        token = os.environ["INSTAGRAM_ACCESS_TOKEN"]

        # ① جلب آخر 10 وسائط للحساب مع تعليقاتها
        r = requests.get(
            f"{GRAPH}/{acc}/media",
            params={
                "fields": "id,timestamp,comments{id,text,username,timestamp}",
                "limit": 10,
                "access_token": token,
            },
            timeout=30,
        )
        r.raise_for_status()
        items: List[SocialItem] = []
        for media in r.json().get("data", []):
            for c in media.get("comments", {}).get("data", []):
                cid = c["id"]
                if cid in since_ids:
                    continue
                items.append(SocialItem(
                    platform="instagram", external_id=cid, kind="comment",
                    author=c.get("username", "unknown"), text=c.get("text", ""),
                    thread_id=media.get("id"), raw=c,
                ))

        # ② جلب الإشارات (mentions) — يتطلب صلاحية instagram_manage_insights
        # إن لم تكن الصلاحية مفعّلة، تفشل المكالمة بـ 403 ونتجاهلها بصمت
        try:
            rm = requests.get(
                f"{GRAPH}/{acc}/tags",
                params={
                    "fields": "id,caption,media_type,timestamp",
                    "access_token": token,
                },
                timeout=30,
            )
            if rm.ok:
                for m in rm.json().get("data", []):
                    mid = m["id"]
                    if mid in since_ids:
                        continue
                    items.append(SocialItem(
                        platform="instagram", external_id=mid, kind="mention",
                        author="mention", text=m.get("caption", ""),
                        raw=m,
                    ))
        except Exception:  # noqa: BLE001
            pass  # صلاحية mentions اختيارية — لا تعطّل باقي المراقبة

        return items

    @with_retry()
    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        token = os.environ["INSTAGRAM_ACCESS_TOKEN"]
        r = requests.post(
            f"{GRAPH}/{item.external_id}/replies",
            data={"message": text, "access_token": token}, timeout=30,
        )
        r.raise_for_status()
        return r.json().get("id", "")
