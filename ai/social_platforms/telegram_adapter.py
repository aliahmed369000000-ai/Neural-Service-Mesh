"""
محول Telegram — عبر Telegram Bot API.
يتطلب: TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID (الدردشة/القناة الافتراضية).
المراقبة تستخدم getUpdates (polling) مع تتبع آخر update_id.
"""

from __future__ import annotations

import os
from typing import List

import requests

from .base import PlatformAdapter, SocialItem

API_BASE = "https://api.telegram.org/bot{token}"


class TelegramAdapter(PlatformAdapter):
    platform_id = "telegram"
    required_env = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]

    def _base(self):
        return API_BASE.format(token=os.environ["TELEGRAM_BOT_TOKEN"])

    def publish(self, text: str) -> str:
        self._require_configured()
        r = requests.post(
            f"{self._base()}/sendMessage",
            json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text},
            timeout=30,
        )
        r.raise_for_status()
        return str(r.json()["result"]["message_id"])

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        offset = None
        if since_ids:
            try:
                offset = max(int(x) for x in since_ids) + 1
            except ValueError:
                offset = None
        params = {"timeout": 0, "limit": 50}
        if offset is not None:
            params["offset"] = offset
        r = requests.get(f"{self._base()}/getUpdates", params=params, timeout=30)
        r.raise_for_status()
        items = []
        for upd in r.json().get("result", []):
            uid = str(upd["update_id"])
            if uid in since_ids:
                continue
            msg = upd.get("message") or upd.get("channel_post")
            if not msg or "text" not in msg:
                continue
            items.append(SocialItem(
                platform="telegram", external_id=uid, kind="dm",
                author=msg.get("from", {}).get("username", "unknown"),
                text=msg["text"], thread_id=str(msg["chat"]["id"]),
                raw=upd,
            ))
        return items

    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        r = requests.post(
            f"{self._base()}/sendMessage",
            json={
                "chat_id": item.thread_id or os.environ["TELEGRAM_CHAT_ID"],
                "text": text,
                "reply_to_message_id": item.raw.get("message", {}).get("message_id"),
            },
            timeout=30,
        )
        r.raise_for_status()
        return str(r.json()["result"]["message_id"])
