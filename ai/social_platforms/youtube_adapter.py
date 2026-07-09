"""
محول YouTube — عبر YouTube Data API v3.
النشر (community posts) غير متاح عبر API عام؛ يدعم هذا المحول:
  - المراقبة والرد على تعليقات الفيديوهات (commentThreads.insert / comments.insert)
  - "النشر" هنا يُفعَّل كتعليق ثابت (pinned-style) على أحدث فيديو، لأن
    منشورات المجتمع (Community Posts) لا تدعمها Data API رسمياً.
يتطلب: YOUTUBE_API_KEY (قراءة)، YOUTUBE_OAUTH_ACCESS_TOKEN (كتابة/رد —
يُجدَّد عبر YOUTUBE_CLIENT_ID/YOUTUBE_CLIENT_SECRET/YOUTUBE_OAUTH_REFRESH_TOKEN)،
YOUTUBE_CHANNEL_ID.
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests

from .base import PlatformAdapter, SocialItem, NotConfiguredError

API_BASE = "https://www.googleapis.com/youtube/v3"
TOKEN_URL = "https://oauth2.googleapis.com/token"


class YouTubeAdapter(PlatformAdapter):
    platform_id = "youtube"
    required_env = ["YOUTUBE_API_KEY", "YOUTUBE_CHANNEL_ID"]
    #: مطلوبة إضافياً فقط للرد/النشر (كتابة) — القراءة تعمل بـ API key وحده
    write_env = ["YOUTUBE_CLIENT_ID", "YOUTUBE_CLIENT_SECRET", "YOUTUBE_OAUTH_REFRESH_TOKEN"]

    def _can_write(self) -> bool:
        return all(os.environ.get(k) for k in self.write_env)

    def missing_env(self) -> list:
        """يعرض الفرق بين المتطلبات الكاملة (قراءة+كتابة) والمتاحة."""
        base_missing = super().missing_env()
        write_missing = [k for k in self.write_env if not os.environ.get(k)]
        # نوضّح في الرسالة أن الكتابة/الرد تحتاج متطلبات إضافية
        if base_missing:
            return base_missing
        if write_missing:
            return [f"{k} (للكتابة/الرد فقط)" for k in write_missing]
        return []

    def _access_token(self) -> str:
        if not self._can_write():
            raise NotConfiguredError(
                "youtube: الرد/النشر يتطلبان أيضاً "
                + ", ".join(self.write_env)
            )
        r = requests.post(TOKEN_URL, data={
            "client_id": os.environ["YOUTUBE_CLIENT_ID"],
            "client_secret": os.environ["YOUTUBE_CLIENT_SECRET"],
            "refresh_token": os.environ["YOUTUBE_OAUTH_REFRESH_TOKEN"],
            "grant_type": "refresh_token",
        }, timeout=30)
        r.raise_for_status()
        return r.json()["access_token"]

    def _latest_video_id(self) -> Optional[str]:
        r = requests.get(f"{API_BASE}/search", params={
            "key": os.environ["YOUTUBE_API_KEY"],
            "channelId": os.environ["YOUTUBE_CHANNEL_ID"],
            "order": "date", "part": "id", "maxResults": 1, "type": "video",
        }, timeout=30)
        r.raise_for_status()
        items = r.json().get("items", [])
        return items[0]["id"]["videoId"] if items else None

    def publish(self, text: str) -> str:
        vid = self._latest_video_id()
        if not vid:
            raise NotConfiguredError("youtube: لا يوجد فيديو حالي للنشر عليه كتعليق.")
        token = self._access_token()
        r = requests.post(
            f"{API_BASE}/commentThreads", params={"part": "snippet"},
            headers={"Authorization": f"Bearer {token}"},
            json={"snippet": {"videoId": vid,
                               "topLevelComment": {"snippet": {"textOriginal": text}}}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        vid = self._latest_video_id()
        if not vid:
            return []
        r = requests.get(f"{API_BASE}/commentThreads", params={
            "key": os.environ["YOUTUBE_API_KEY"],
            "videoId": vid, "part": "snippet", "order": "time", "maxResults": 20,
        }, timeout=30)
        r.raise_for_status()
        items = []
        for c in r.json().get("items", []):
            cid = c["id"]
            if cid in since_ids:
                continue
            sn = c["snippet"]["topLevelComment"]["snippet"]
            items.append(SocialItem(
                platform="youtube", external_id=cid, kind="comment",
                author=sn.get("authorDisplayName", "unknown"),
                text=sn.get("textOriginal", ""), thread_id=cid, raw=c,
            ))
        return items

    def reply(self, item: SocialItem, text: str) -> str:
        token = self._access_token()
        r = requests.post(
            f"{API_BASE}/comments", params={"part": "snippet"},
            headers={"Authorization": f"Bearer {token}"},
            json={"snippet": {"parentId": item.external_id, "textOriginal": text}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]
