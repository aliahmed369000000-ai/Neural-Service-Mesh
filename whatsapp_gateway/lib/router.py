"""
router.py — آلة الحالة (state machine) لمعالجة رسائل واتساب الواردة.

⚠️ منطقياً مطابق لـai/whatsapp/router.py بالمستودع الرئيسي، بفارق واحد
متعمَّد: يستدعي accounts_lookup.get_user_by_phone() (عبر Upstash) بدل
ai.accounts.get_user_by_phone() (عبر SQLite غير المتاح من هنا).
"""
from __future__ import annotations

from typing import Optional, Tuple

from .quran_lookup import get_ayah, parse_ayah_reference, AyahNotFound
from .accounts_lookup import get_user_by_phone

DEFAULT_STATE = "menu"

_WELCOME = (
    "مرحباً 👋 هذا رد آلي لمشروع Neural Service Mesh (ليس محادثة مع شخص).\n\n"
    "اختر رقماً:\n"
    "1️⃣ بحث نص آية بالرقم\n"
    "2️⃣ حالة حسابي\n"
    "0️⃣ التحدث مع شخص"
)

_ASK_AYAH_FORMAT = "أرسل رقم السورة والآية بصيغة سورة:آية (مثال: 2:255)"

_HUMAN_HANDOFF = "تم تسجيل طلبك للتحدث مع شخص، سيتواصل معك فريق الدعم قريباً 🙏"


def handle_incoming_message(
    phone: str, text: str, state: Optional[str]
) -> Tuple[str, str]:
    text = (text or "").strip()
    state = state or DEFAULT_STATE

    if text == "0":
        return _HUMAN_HANDOFF, DEFAULT_STATE

    if state == "awaiting_ayah":
        return _handle_awaiting_ayah(text)

    if text == "1":
        return _ASK_AYAH_FORMAT, "awaiting_ayah"
    if text == "2":
        return _handle_account_status(phone), DEFAULT_STATE

    return _WELCOME, DEFAULT_STATE


def _handle_awaiting_ayah(text: str) -> Tuple[str, str]:
    ref = parse_ayah_reference(text)
    if ref is None:
        return (
            "صيغة غير صحيحة. " + _ASK_AYAH_FORMAT + "\n(أرسل 0 للتحدث مع شخص)",
            "awaiting_ayah",
        )
    surah, ayah = ref
    try:
        result = get_ayah(surah, ayah)
    except AyahNotFound as exc:
        return str(exc) + "\n\n" + _ASK_AYAH_FORMAT, "awaiting_ayah"

    reply = f"📖 {result['surah']}:{result['ayah']}\n\n{result['text']}\n\n" + _WELCOME
    return reply, DEFAULT_STATE


def _handle_account_status(phone: str) -> str:
    user = get_user_by_phone(phone)
    if user is None:
        return (
            "هذا الرقم غير مرتبط بأي حساب NSM.\n"
            "اربط رقمك من الموقع أولاً (تسجيل دخول ← إضافة رقم هاتف)، "
            "ثم أرسل 2 مرة أخرى."
        )
    return (
        f"✅ حسابك: {user['username']}\n"
        f"مسجّل منذ: {user['created_at'][:10]}\n\n"
        + _WELCOME
    )
