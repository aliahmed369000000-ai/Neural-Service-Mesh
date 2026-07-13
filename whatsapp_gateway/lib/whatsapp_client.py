"""
whatsapp_client.py — طبقة رقيقة فوق WhatsApp Cloud API الرسمي.

⚠️ نسخة طبق الأصل من ai/whatsapp/whatsapp_client.py بالمستودع الرئيسي.
"""
from __future__ import annotations

import os
from typing import Optional

_GRAPH_API_VERSION = "v21.0"
_HTTP_TIMEOUT = 10


class WhatsAppSendError(RuntimeError):
    pass


def verify_webhook_challenge(
    mode: Optional[str], token: Optional[str], challenge: Optional[str]
) -> Optional[str]:
    expected_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
    if not expected_token:
        return None
    if mode == "subscribe" and token == expected_token:
        return challenge
    return None


def send_text_message(to_phone: str, text: str) -> None:
    import requests

    access_token = os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()
    phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()
    if not access_token or not phone_number_id:
        raise WhatsAppSendError(
            "WHATSAPP_ACCESS_TOKEN أو WHATSAPP_PHONE_NUMBER_ID غير مضبوطين بالبيئة"
        )

    url = f"https://graph.facebook.com/{_GRAPH_API_VERSION}/{phone_number_id}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_phone,
        "type": "text",
        "text": {"body": text},
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=_HTTP_TIMEOUT)
    except Exception as exc:
        raise WhatsAppSendError(f"فشل الاتصال بـWhatsApp Cloud API: {exc}") from exc

    if not resp.ok:
        raise WhatsAppSendError(
            f"WhatsApp Cloud API رفض الطلب ({resp.status_code}): {resp.text[:300]}"
        )


def extract_incoming_message(payload: dict) -> Optional[tuple[str, str]]:
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages")
        if not messages:
            return None
        message = messages[0]
        if message.get("type") != "text":
            return None
        sender = message.get("from")
        text = message.get("text", {}).get("body")
        if not sender or text is None:
            return None
        return sender, text
    except (KeyError, IndexError, TypeError):
        return None
