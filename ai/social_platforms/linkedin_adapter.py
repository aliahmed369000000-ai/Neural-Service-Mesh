"""
محول LinkedIn — عبر Posts API الرسمي الحالي (/rest/posts)، الذي حلّ محل
UGC Posts API وShares API المتوقفين. يتطلب: LINKEDIN_ACCESS_TOKEN
(OAuth 2.0 بصلاحية w_member_social للحساب الشخصي أو w_organization_social
لصفحة شركة)، LINKEDIN_AUTHOR_URN (مثال: urn:li:person:XXXX أو
urn:li:organization:XXXX). LINKEDIN_API_VERSION اختياري (افتراضي أدناه —
LinkedIn يُصدر إصداراً جديداً كل شهر بصيغة YYYYMM، يُفضّل تحديثه دورياً).

ملاحظة مهمة (لا تلفيق): النشر (publish) موثّق رسمياً ويعمل بثبات. أما
قراءة التعليقات/الإشارات (fetch_new_items/reply) فتعتمد على Social Actions
API القديم الذي يتطلب غالباً موافقة LinkedIn Marketing Developer Platform
(شراكة معتمدة) — إن رجع 403 فهذا يعني الحساب لم يُفعَّل لهذا المستوى من
الوصول، وليس خطأً في الكود.
"""

from __future__ import annotations

import os
from typing import List
from urllib.parse import quote

import requests

from .base import PlatformAdapter, SocialItem
from .retry import with_retry

API_BASE = "https://api.linkedin.com"
DEFAULT_API_VERSION = "202501"  # حدّثها دورياً حسب إصدارات LinkedIn الشهرية


class LinkedInAdapter(PlatformAdapter):
    platform_id = "linkedin"
    required_env = ["LINKEDIN_ACCESS_TOKEN", "LINKEDIN_AUTHOR_URN"]

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {os.environ['LINKEDIN_ACCESS_TOKEN']}",
            "Content-Type": "application/json",
            "X-Restli-Protocol-Version": "2.0.0",
            "LinkedIn-Version": os.environ.get("LINKEDIN_API_VERSION", DEFAULT_API_VERSION),
        }

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        payload = {
            "author": os.environ["LINKEDIN_AUTHOR_URN"],
            "commentary": text,
            "visibility": "PUBLIC",
            "distribution": {
                "feedDistribution": "MAIN_FEED",
                "targetEntities": [], "thirdPartyDistributionChannels": [],
            },
            "lifecycleState": "PUBLISHED",
            "isReshareDisabledByAuthor": False,
        }
        r = requests.post(f"{API_BASE}/rest/posts", headers=self._headers(),
                           json=payload, timeout=30)
        r.raise_for_status()
        return r.headers.get("x-restli-id", "")

    @with_retry()
    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        author = os.environ["LINKEDIN_AUTHOR_URN"]
        r = requests.get(
            f"{API_BASE}/rest/posts",
            headers=self._headers(),
            params={"author": author, "q": "author", "count": 5, "sortBy": "LAST_MODIFIED"},
            timeout=30,
        )
        r.raise_for_status()
        items: List[SocialItem] = []
        for post in r.json().get("elements", []):
            post_urn = post.get("id", "")
            if not post_urn:
                continue
            # تعليقات المنشور — يتطلب صلاحية Marketing Developer Platform
            c = requests.get(
                f"{API_BASE}/v2/socialActions/{quote(post_urn, safe='')}/comments",
                headers=self._headers(), timeout=30,
            )
            if not c.ok:
                continue  # لا صلاحية وصول كافية لهذا المنشور — تجاهل بصمت وجرّب التالي
            for cm in c.json().get("elements", []):
                cid = cm.get("$URN") or cm.get("id", "")
                if not cid or cid in since_ids:
                    continue
                items.append(SocialItem(
                    platform="linkedin", external_id=cid, kind="comment",
                    author=cm.get("actor", "unknown"),
                    text=cm.get("message", {}).get("text", ""),
                    thread_id=post_urn, raw=cm,
                ))
        return items

    @with_retry()
    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        parent = item.thread_id or item.external_id
        r = requests.post(
            f"{API_BASE}/v2/socialActions/{quote(parent, safe='')}/comments",
            headers=self._headers(),
            json={
                "actor": os.environ["LINKEDIN_AUTHOR_URN"],
                "object": parent,
                "message": {"text": text},
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("$URN") or data.get("id", "")
