"""
router.py — آلة الحالة (state machine) لمعالجة رسائل واتساب الواردة.

مصمَّم خصيصاً ليكون منطقاً صرفاً (pure logic): يستقبل (رقم الهاتف، نص
الرسالة، الحالة الحالية) ويعيد (نص الرد، الحالة الجديدة) — بدون أي
اتصال شبكي مباشر (لا واتساب، لا Upstash، لا LLM). هذا يسمح باختبار كل
سيناريوهات القائمة محلياً وبشكل حتمي قبل ربطه بأي بنية تحتية حقيقية.

الحالة (state) نص بسيط يُخزَّن خارجياً (Upstash Redis بالإنتاج، أو أي
قاموس بالاختبار) — القيم الممكنة: "menu" (افتراضي)، "awaiting_ayah".

نطاق متعمَّد وضيّق (يطابق سياسة Meta لبوتات WhatsApp Business API
2026 — بدون محادثة مفتوحة): بحث نص آية بالرقم فقط للجمهور العام،
واستعلام حالة حساب مباشر من قاعدة البيانات (بدون LLM) للمسجّلين.
أي نص خارج هاتين المهمتين يُعاد توجيهه للقائمة، لا يُفسَّر بحرية.
"""
from __future__ import annotations

from typing import Optional, Tuple

from .quran_lookup import get_ayah, parse_ayah_reference, AyahNotFound

try:
    from ..accounts import get_user_by_phone
except Exception:  # pragma: no cover — يسمح باختبار هذا الملف منعزلاً لو فشل استيراد accounts
    get_user_by_phone = None  # type: ignore

DEFAULT_STATE = "menu"

_WELCOME = (
    "مرحباً 👋 هذا رد آلي لمشروع Neural Service Mesh (ليس محادثة مع شخص).\n\n"
    "اختر رقماً:\n"
    "1️⃣ بحث نص آية بالرقم\n"
    "2️⃣ حالة حسابي\n"
    "0️⃣ التحدث مع شخص"
)

_ASK_AYAH_FORMAT = "أرسل رقم السورة والآية بصيغة سورة:آية (مثال: 2:255)"

_FALLBACK = "لم أفهم هذا الخيار. " + _WELCOME

_HUMAN_HANDOFF = "تم تسجيل طلبك للتحدث مع شخص، سيتواصل معك فريق الدعم قريباً 🙏"


def handle_incoming_message(
    phone: str, text: str, state: Optional[str]
) -> Tuple[str, str]:
    """نقطة الدخول الرئيسية. تعيد (نص_الرد, الحالة_الجديدة)."""
    text = (text or "").strip()
    state = state or DEFAULT_STATE

    if text == "0":
        return _HUMAN_HANDOFF, DEFAULT_STATE

    if state == "awaiting_ayah":
        return _handle_awaiting_ayah(text)

    # state == "menu" (أو أي حالة غير معروفة تُعامَل كقائمة رئيسية)
    if text == "1":
        return _ASK_AYAH_FORMAT, "awaiting_ayah"
    if text == "2":
        return _handle_account_status(phone), DEFAULT_STATE

    # أول تواصل (لا نص مطابق لأي خيار) — نعرض الترحيب بدل رفض فوري
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
    if get_user_by_phone is None:
        return "تعذّر الوصول لنظام الحسابات حالياً. حاول لاحقاً."

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
