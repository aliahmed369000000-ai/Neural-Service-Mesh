"""
محول X (Twitter) — عبر X API v2 (OAuth 1.0a user-context للنشر/الرد).
يتطلب: TWITTER_API_KEY, TWITTER_API_SECRET, TWITTER_ACCESS_TOKEN,
TWITTER_ACCESS_TOKEN_SECRET.
ملاحظة: النشر والرد على X API v2 يتطلبان خطة مدفوعة من X — إن رجع الخادم
402/403 فهذا يعني أن مستوى API الحالي لا يسمح بذلك، وليس خطأ في الكود.
"""

from __future__ import annotations

import os
from typing import List

import requests
from requests_oauthlib import OAuth1

from .base import PlatformAdapter, SocialItem
from .retry import with_retry

API_BASE = "https://api.twitter.com/2"


class TwitterAdapter(PlatformAdapter):
    platform_id = "twitter"
    required_env = [
        "TWITTER_API_KEY", "TWITTER_API_SECRET",
        "TWITTER_ACCESS_TOKEN", "TWITTER_ACCESS_TOKEN_SECRET",
    ]

    def _auth(self):
        return OAuth1(
            os.environ["TWITTER_API_KEY"], os.environ["TWITTER_API_SECRET"],
            os.environ["TWITTER_ACCESS_TOKEN"], os.environ["TWITTER_ACCESS_TOKEN_SECRET"],
        )

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        r = requests.post(f"{API_BASE}/tweets", auth=self._auth(),
                           json={"text": text}, timeout=30)
        r.raise_for_status()
        return r.json()["data"]["id"]

    @with_retry()
    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        me = requests.get(f"{API_BASE}/users/me", auth=self._auth(), timeout=30)
        me.raise_for_status()
        username = me.json()["data"]["username"]
        r = requests.get(
            f"{API_BASE}/tweets/search/recent",
            auth=self._auth(),
            params={"query": f"@{username} -from:{username}", "max_results": 20,
                    "tweet.fields": "author_id,conversation_id"},
            timeout=30,
        )
        r.raise_for_status()
        items = []
        for t in r.json().get("data", []):
            tid = t["id"]
            if tid in since_ids:
                continue
            items.append(SocialItem(
                platform="twitter", external_id=tid, kind="mention",
                author=str(t.get("author_id", "unknown")), text=t.get("text", ""),
                thread_id=str(t.get("conversation_id", tid)),
                url=f"https://twitter.com/i/web/status/{tid}", raw=t,
            ))
        return items

    @with_retry()
    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        r = requests.post(
            f"{API_BASE}/tweets", auth=self._auth(),
            json={"text": text, "reply": {"in_reply_to_tweet_id": item.external_id}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["data"]["id"]
