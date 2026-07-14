"""
محول Reddit — عبر Reddit OAuth2 API (نمط تطبيق "script" الرسمي، بدون praw).
يتطلب: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USERNAME,
REDDIT_PASSWORD, REDDIT_USER_AGENT (Reddit يرفض الطلبات بدون User-Agent
وصفي فريد)، REDDIT_SUBREDDIT (السابريديت الافتراضي للنشر — منشورات Reddit
النصية تتطلب subreddit محدداً، على عكس بقية المنصات).

المراقبة تعتمد على صندوق الوارد (/message/unread) الذي يضم إشارات
اسم المستخدم (username mentions) والردود على تعليقاتك — وهو المسار
الرسمي الوحيد لرصد التفاعل دون معرفة كل منشوراتك مسبقاً.
"""

from __future__ import annotations

import os
import time
from typing import List, Optional

import requests

from .base import PlatformAdapter, SocialItem
from .retry import with_retry

TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"


class RedditAdapter(PlatformAdapter):
    platform_id = "reddit"
    required_env = [
        "REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET",
        "REDDIT_USERNAME", "REDDIT_PASSWORD",
        "REDDIT_USER_AGENT", "REDDIT_SUBREDDIT",
    ]

    _token_cache: Optional[str] = None
    _token_expiry: float = 0.0

    def _access_token(self) -> str:
        self._require_configured()
        # إعادة استخدام التوكن إن كان لا يزال صالحاً (صلاحيته ساعة تقريباً)
        if self._token_cache and time.time() < self._token_expiry - 30:
            return self._token_cache
        r = requests.post(
            TOKEN_URL,
            auth=(os.environ["REDDIT_CLIENT_ID"], os.environ["REDDIT_CLIENT_SECRET"]),
            data={
                "grant_type": "password",
                "username": os.environ["REDDIT_USERNAME"],
                "password": os.environ["REDDIT_PASSWORD"],
            },
            headers={"User-Agent": os.environ["REDDIT_USER_AGENT"]},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        self._token_cache = data["access_token"]
        self._token_expiry = time.time() + data.get("expires_in", 3600)
        return self._token_cache

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._access_token()}",
            "User-Agent": os.environ["REDDIT_USER_AGENT"],
        }

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        title = (text[:97] + "...") if len(text) > 100 else text
        r = requests.post(
            f"{API_BASE}/api/submit",
            headers=self._headers(),
            data={
                "sr": os.environ["REDDIT_SUBREDDIT"],
                "kind": "self",
                "title": title,
                "text": text,
                "api_type": "json",
            },
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        errors = data.get("json", {}).get("errors", [])
        if errors:
            raise RuntimeError(f"reddit: فشل النشر — {errors}")
        return data["json"]["data"]["name"]

    @with_retry()
    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        r = requests.get(
            f"{API_BASE}/message/unread", headers=self._headers(),
            params={"limit": 25}, timeout=30,
        )
        r.raise_for_status()
        items: List[SocialItem] = []
        for child in r.json().get("data", {}).get("children", []):
            d = child.get("data", {})
            fullname = d.get("name", "")
            ext_id = d.get("id", fullname)
            if ext_id in since_ids or not fullname.startswith("t1_"):
                continue  # نتجاهل الرسائل الخاصة (t4) — نراقب الإشارات/الردود فقط
            items.append(SocialItem(
                platform="reddit", external_id=ext_id, kind="mention",
                author=d.get("author", "unknown"), text=d.get("body", ""),
                thread_id=d.get("context", fullname),
                url=f"https://reddit.com{d.get('context', '')}" if d.get("context") else None,
                raw={"name": fullname, **d},
            ))
        return items

    @with_retry()
    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        thing_id = item.raw.get("name") or (
            item.external_id if item.external_id.startswith("t1_") else f"t1_{item.external_id}"
        )
        r = requests.post(
            f"{API_BASE}/api/comment", headers=self._headers(),
            data={"thing_id": thing_id, "text": text, "api_type": "json"},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json()
        errors = data.get("json", {}).get("errors", [])
        if errors:
            raise RuntimeError(f"reddit: فشل الرد — {errors}")
        things = data["json"]["data"]["things"]
        return things[0]["data"]["name"] if things else ""
