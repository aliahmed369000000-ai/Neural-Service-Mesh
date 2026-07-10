"""
محول Facebook — عبر Meta Graph API لصفحة فيسبوك.
يتطلب: FACEBOOK_PAGE_ACCESS_TOKEN، FACEBOOK_PAGE_ID.
"""

from __future__ import annotations

import os
from typing import List

import requests

from .base import PlatformAdapter, SocialItem

GRAPH = "https://graph.facebook.com/v19.0"


class FacebookAdapter(PlatformAdapter):
    platform_id = "facebook"
    required_env = ["FACEBOOK_PAGE_ACCESS_TOKEN", "FACEBOOK_PAGE_ID"]

    def publish(self, text: str) -> str:
        self._require_configured()
        page = os.environ["FACEBOOK_PAGE_ID"]
        token = os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"]
        r = requests.post(
            f"{GRAPH}/{page}/feed",
            data={"message": text, "access_token": token}, timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        page = os.environ["FACEBOOK_PAGE_ID"]
        token = os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"]
        r = requests.get(
            f"{GRAPH}/{page}/feed",
            params={"fields": "comments{id,message,from,created_time}",
                    "access_token": token, "limit": 10},
            timeout=30,
        )
        r.raise_for_status()
        items = []
        for post in r.json().get("data", []):
            for c in post.get("comments", {}).get("data", []):
                cid = c["id"]
                if cid in since_ids:
                    continue
                items.append(SocialItem(
                    platform="facebook", external_id=cid, kind="comment",
                    author=c.get("from", {}).get("name", "unknown"),
                    text=c.get("message", ""), thread_id=post.get("id"), raw=c,
                ))
        return items

    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        token = os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"]
        r = requests.post(
            f"{GRAPH}/{item.external_id}/comments",
            data={"message": text, "access_token": token}, timeout=30,
        )
        r.raise_for_status()
        return r.json().get("id", "")
