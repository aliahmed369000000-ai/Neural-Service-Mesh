"""
whatsapp_client.py — طبقة رقيقة فوق WhatsApp Cloud API الرسمي (مباشر
من Meta، بدون أي BSP وسيط مثل Twilio/Gupshup — راجع القرار السابق:
هذا هو المسار المجاني الوحيد بدون markup إضافي).

مسؤوليتان فقط:
  1. verify_webhook_challenge(): التحقق أثناء ربط الـwebhook لأول مرة
     (Meta يرسل GET بمعامِلات hub.mode/hub.verify_token/hub.challenge).
  2. send_text_message(): إرسال رد نصي لمستخدم عبر POST لنقطة /messages.

متغيرات البيئة المطلوبة (تُضبط بـVercel لاحقاً، غير مطلوبة للاختبار):
    WHATSAPP_VERIFY_TOKEN     — نص عشوائي تختاره أنت، يُدخَل بلوحة Meta أيضاً
    WHATSAPP_ACCESS_TOKEN     — من Meta Business/App
    WHATSAPP_PHONE_NUMBER_ID  — معرّف رقم الهاتف المسجَّل بـCloud API
"""
from __future__ import annotations

import os
from typing import Optional

_GRAPH_API_VERSION = "v21.0"
_HTTP_TIMEOUT = 10


class WhatsAppSendError(RuntimeError):
    """فشل إرسال رسالة عبر Cloud API (رفض من Meta، شبكة، أو إعداد ناقص)."""


def verify_webhook_challenge(
    mode: Optional[str], token: Optional[str], challenge: Optional[str]
) -> Optional[str]:
    """يُستدعى من معالج GET بـwebhook.py أثناء ربط الرقم بلوحة Meta.
    يعيد قيمة challenge (يجب إرجاعها كنص خام بالاستجابة) لو التحقق نجح،
    أو None لو فشل (رمز تحقق خاطئ أو mode ليس 'subscribe') — عندها
    المستدعي يجب أن يرجّع HTTP 403 بدل تمرير القيمة."""
    expected_token = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
    if not expected_token:
        return None  # لا رمز تحقق مضبوط بالبيئة أصلاً — ما نقدر نتحقق، نرفض بأمان
    if mode == "subscribe" and token == expected_token:
        return challenge
    return None


def send_text_message(to_phone: str, text: str) -> None:
    """يرسل رسالة نصية عبر WhatsApp Cloud API. يرفع WhatsAppSendError
    برسالة عربية واضحة لو فشل الإرسال (بدلاً من فشل صامت) — عكس
    state_store حيث فضّلنا التدهور الصامت، هنا فشل الإرسال يعني
    المستخدم لن يستلم رداً إطلاقاً فيستحق تسجيلاً واضحاً بالخطأ."""
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
    """يستخرج (رقم_المرسل, نص_الرسالة) من جسم POST الوارد من Meta،
    أو None لو الحدث ليس رسالة نصية واردة (مثل: تأكيد تسليم، رسالة
    وسائط، أو حدث فارغ عند فحص الاتصال الأولي) — تلك تُتجاهَل بصمت،
    ليست خطأ."""
    try:
        entry = payload.get("entry", [])[0]
        change = entry.get("changes", [])[0]
        value = change.get("value", {})
        messages = value.get("messages")
        if not messages:
            return None  # قد يكون هذا حدث "statuses" (تأكيد تسليم) لا رسالة واردة
        message = messages[0]
        if message.get("type") != "text":
            return None  # نطاقنا نصوص فقط (لا صوت/صورة/موقع)
        sender = message.get("from")
        text = message.get("text", {}).get("body")
        if not sender or text is None:
            return None
        return sender, text
    except (KeyError, IndexError, TypeError):
        return None
