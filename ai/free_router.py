"""
free_router.py — طبقة توجيه موحّدة لاستبدال الاعتماد الحصري على OpenRouter
============================================================================
يعيد استخدام نفس منطق الاتصال المباشر بـ Groq / Gemini / Cloudflare
الموجود أصلاً في ai/ultraplinian.py (مصدر وحيد للحقيقة — لا تكرار كود)،
ويوفّر واجهة بسيطة واحدة تستخدمها أي نقطة بالمشروع كانت تعتمد على
OPENROUTER_API_KEY حصراً وتتوقف بدونه:

    chat_free(messages, temperature=0.7, max_tokens=2048) -> (text, model_used)

سلوك التوجيه: يجرّب كل نموذج من FREE_DIRECT_MODELS بالترتيب
(Groq → Gemini → Cloudflare) حتى ينجح أحدها. لو فشلت كل المحاولات
(غياب كل المفاتيح أو أخطاء شبكة) يرفع NoProviderAvailable برسالة عربية
واضحة بدل الانهيار الصامت أو رمي استثناء HTTP خام.

هذا الملف لا يستبدل OpenRouter عند وجوده — فقط يوفّر مساراً بديلاً يُستدعى
كـ fallback من المتصلين (streamlit_app.py، ai/social_agent.py) عند غياب
OPENROUTER_API_KEY أو فشل الاتصال به.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .ultraplinian import (
    FREE_DIRECT_MODELS,
    _call_cloudflare_model,
    _call_gemini_model,
    _call_groq_model,
    _parse_model_id,
)


class NoProviderAvailable(RuntimeError):
    """لا يوجد أي نموذج مجاني مباشر متاح (لا مفاتيح صالحة ولا استجابة ناجحة)."""


def chat_free(
    messages: List[Dict],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Tuple[str, str]:
    """
    يجرّب كل نموذج من FREE_DIRECT_MODELS بالترتيب (Groq → Gemini → Cloudflare)
    حتى ينجح أحدها. يعيد (النص المولّد, معرّف النموذج الذي نجح).
    يرفع NoProviderAvailable لو فشلت كل المحاولات.
    """
    errors: List[str] = []
    for model_id in FREE_DIRECT_MODELS:
        provider, real_model = _parse_model_id(model_id)
        try:
            if provider == "groq":
                text = _call_groq_model(real_model, messages, temperature, max_tokens)
            elif provider == "gemini":
                system_prompt = next(
                    (m.get("content", "") for m in messages if m.get("role") == "system"),
                    "",
                )
                user_content = next(
                    (m.get("content", "") for m in reversed(messages) if m.get("role") == "user"),
                    "",
                )
                text = _call_gemini_model(
                    real_model, user_content, system_prompt, temperature, max_tokens
                )
            elif provider == "cloudflare":
                text = _call_cloudflare_model(real_model, messages, max_tokens)
            else:
                continue
            if text and text.strip():
                return text.strip(), model_id
        except Exception as exc:  # أي خطأ (مفتاح ناقص، شبكة، حصة منتهية...) → جرّب التالي
            errors.append(f"{model_id}: {str(exc)[:150]}")
            continue

    raise NoProviderAvailable(
        "تعذّر الاتصال بأي نموذج مجاني مباشر (Groq/Gemini/Cloudflare) ولا "
        "يوجد مفتاح OpenRouter صالح.\nالتفاصيل: " + " | ".join(errors[:3])
    )


def has_any_free_key() -> bool:
    """فحص سريع (بدون استدعاء شبكي) هل يوجد على الأقل مفتاح واحد من مفاتيح
    النماذج المجانية المباشرة مضبوطاً في البيئة — مفيد لعرض حالة بالواجهة."""
    import os

    return bool(
        os.getenv("GROQ_API_KEY", "").strip()
        or os.getenv("GOOGLE_API_KEY", "").strip()
        or (os.getenv("CF_API_TOKEN", "").strip() and os.getenv("CF_ACCOUNT_ID", "").strip())
    )
