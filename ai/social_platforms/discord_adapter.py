"""
محول Discord — عبر Discord Bot API (رمز بوت من Developer Portal).
يتطلب: DISCORD_BOT_TOKEN و DISCORD_CHANNEL_ID (القناة الافتراضية للنشر/المراقبة).
"""

from __future__ import annotations

import os
from typing import List

import requests

from .base import PlatformAdapter, SocialItem, NotConfiguredError

API_BASE = "https://discord.com/api/v10"


class DiscordAdapter(PlatformAdapter):
    platform_id = "discord"
    required_env = ["DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"]

    def _headers(self):
        return {
            "Authorization": f"Bot {os.environ['DISCORD_BOT_TOKEN']}",
            "Content-Type": "application/json",
        }

    def publish(self, text: str) -> str:
        self._require_configured()
        chan = os.environ["DISCORD_CHANNEL_ID"]
        r = requests.post(
            f"{API_BASE}/channels/{chan}/messages",
            headers=self._headers(), json={"content": text}, timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        chan = os.environ["DISCORD_CHANNEL_ID"]
        r = requests.get(
            f"{API_BASE}/channels/{chan}/messages",
            headers=self._headers(), params={"limit": 50}, timeout=30,
        )
        r.raise_for_status()
        items = []
        me_id = os.environ.get("DISCORD_BOT_USER_ID", "")
        for m in r.json():
            mid = m["id"]
            if mid in since_ids:
                continue
            if me_id and m.get("author", {}).get("id") == me_id:
                continue  # تجاهل رسائل البوت نفسه
            items.append(SocialItem(
                platform="discord", external_id=mid, kind="mention",
                author=m.get("author", {}).get("username", "unknown"),
                text=m.get("content", ""), thread_id=chan,
                url=f"https://discord.com/channels/@me/{chan}/{mid}",
                raw=m,
            ))
        return items

    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        chan = item.thread_id or os.environ["DISCORD_CHANNEL_ID"]
        r = requests.post(
            f"{API_BASE}/channels/{chan}/messages",
            headers=self._headers(),
            json={"content": text, "message_reference": {"message_id": item.external_id}},
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["id"]
