"""
محول Telegram — عبر Telegram Bot API.
يتطلب: TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID (الدردشة/القناة الافتراضية).

يدعم وضعين للمراقبة (قابلان للتبديل بدون تغيير أي منطق آخر في الوكيل):

1. Polling (الافتراضي، fetch_new_items أدناه) — عبر getUpdates، مناسب
   لأي بيئة استضافة بلا endpoint عام ثابت (مثل Streamlit Community Cloud).
2. Webhook (اختياري، عبر set_webhook/delete_webhook أدناه) — تيليجرام
   يدعم setWebhook رسمياً فيدفع التحديثات فوراً بدل الاستطلاع الدوري.
   يتطلب endpoint HTTPS عام ثابت (مثال: FastAPI في api_server.py على
   منفذ منفصل عن Streamlit). عند تفعيله يجب استدعاء delete_webhook()
   قبل العودة لوضع polling، لأن تيليجرام يمنع استخدام getUpdates وwebhook
   معاً على نفس البوت في نفس الوقت (تُرجع 409 Conflict خلاف ذلك).

آلية تحليل التحديث الوارد (سواء عبر polling أو webhook) موحّدة في
_parse_update لتفادي ازدواج المنطق وضمان تطابق الشكل الناتج (SocialItem).
"""

from __future__ import annotations

import os
from typing import List, Optional

import requests

from .base import PlatformAdapter, SocialItem
from .retry import with_retry

API_BASE = "https://api.telegram.org/bot{token}"


class TelegramAdapter(PlatformAdapter):
    platform_id = "telegram"
    required_env = ["TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"]
    supports_webhook = True  # تيليجرام يوفّر setWebhook رسمياً — راجع WEBHOOKS.md

    def _base(self):
        return API_BASE.format(token=os.environ["TELEGRAM_BOT_TOKEN"])

    # ── تحليل موحّد لتحديث تيليجرام (يُستخدم من polling و webhook معاً) ──
    @staticmethod
    def _parse_update(upd: dict) -> Optional[SocialItem]:
        uid = upd.get("update_id")
        if uid is None:
            return None
        msg = upd.get("message") or upd.get("channel_post")
        if not msg or "text" not in msg:
            return None
        return SocialItem(
            platform="telegram", external_id=str(uid), kind="dm",
            author=msg.get("from", {}).get("username", "unknown"),
            text=msg["text"], thread_id=str(msg["chat"]["id"]),
            raw=upd,
        )

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        r = requests.post(
            f"{self._base()}/sendMessage",
            json={"chat_id": os.environ["TELEGRAM_CHAT_ID"], "text": text},
            timeout=30,
        )
        r.raise_for_status()
        return str(r.json()["result"]["message_id"])

    @with_retry()
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
            item = self._parse_update(upd)
            if item is not None:
                items.append(item)
        return items

    @with_retry()
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

    # ── إدارة وضع Webhook (اختياري، بديل عن getUpdates) ─────────────────
    @with_retry()
    def set_webhook(self, url: str, secret_token: Optional[str] = None) -> dict:
        """يفعّل webhook على عنوان url (يجب أن يكون HTTPS عاماً).
        secret_token اختياري لكن يُنصح به بشدة: تيليجرام يرسله في رأس
        X-Telegram-Bot-Api-Secret-Token مع كل طلب، ليتحقق الـendpoint أن
        الطلب فعلاً من تيليجرام قبل معالجته."""
        self._require_configured()
        payload = {"url": url, "allowed_updates": ["message", "channel_post"]}
        if secret_token:
            payload["secret_token"] = secret_token
        r = requests.post(f"{self._base()}/setWebhook", json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        if not data.get("ok"):
            raise RuntimeError(f"فشل تفعيل webhook تيليجرام: {data}")
        return data

    @with_retry()
    def delete_webhook(self) -> dict:
        """يلغي webhook ويعيد البوت لوضع getUpdates (polling)."""
        self._require_configured()
        r = requests.post(f"{self._base()}/deleteWebhook", timeout=30)
        r.raise_for_status()
        return r.json()

    @with_retry()
    def webhook_info(self) -> dict:
        """يعيد حالة webhook الحالية من تيليجرام (getWebhookInfo) — مفيد
        لعرضها بواجهة الإعدادات والتأكد من عدم وجود أخطاء تسليم متراكمة."""
        self._require_configured()
        r = requests.get(f"{self._base()}/getWebhookInfo", timeout=30)
        r.raise_for_status()
        return r.json().get("result", {})
