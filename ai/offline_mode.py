"""
ai/offline_mode.py — نقطة فحص واحدة لوضع النشر المغلق (بدون إنترنت خارجي)
===========================================================================
مصدر وحيد للحقيقة يُستخدم من كل أداة تعتمد على اتصال إنترنت خارجي (بحث
الويب، مصادر الصور، الأصوات السحابية...) لتفادي تكرار قراءة
NSM_OFFLINE_MODE في كل ملف على حدة، وضمان رسالة عربية موحّدة للمستخدم
بدل محاولة الاتصال والفشل ببطء أو بصمت.

نفس متغيّر البيئة المستخدم في ai/llm_fallback.py لتفعيل النموذج المحلي.

الاستخدام:
    from ai.offline_mode import is_offline, offline_message

    def some_tool(...):
        if is_offline():
            return offline_message("بحث الويب")
        ...  # المسار المتصل بالإنترنت كالمعتاد
"""
from __future__ import annotations

import os


def is_offline() -> bool:
    """True إذا كان NSM يعمل في وضع النشر المغلق (بدون إنترنت خارجي)."""
    return os.getenv("NSM_OFFLINE_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def offline_message(feature_name: str) -> str:
    """رسالة عربية موحّدة وواضحة عند تعطيل ميزة تعتمد على الإنترنت
    الخارجي بسبب وضع النشر المغلق."""
    return (
        f"🔒 ميزة «{feature_name}» تحتاج اتصالاً بالإنترنت الخارجي، وهي "
        f"معطّلة حالياً لأن NSM يعمل في وضع النشر المغلق داخل شبكة الجهة "
        f"(NSM_OFFLINE_MODE=1)."
    )


def offline_status() -> dict:
    """ملخص حالة الوضع المغلق للاستخدام في واجهة الصحة/النظام.
    لا يرمي استثناءات — كل فشل يُسجَّل كـ False/رسالة.
    """
    offline = is_offline()
    status = {
        "offline_mode": offline,
        "ollama_reachable": None,
        "ollama_url": None,
        "local_model": None,
        "message": "",
    }
    if not offline:
        status["message"] = "الوضع المتصل بالإنترنت (NSM_OFFLINE_MODE غير مفعّل)."
        return status

    import os
    url = os.getenv("NSM_LOCAL_LLM_URL", "http://localhost:11434").rstrip("/")
    model = os.getenv("NSM_LOCAL_MODEL", "qwen2.5:7b-instruct-q4_K_M")
    status["ollama_url"] = url
    status["local_model"] = model

    try:
        import urllib.request
        req = urllib.request.Request(f"{url}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            status["ollama_reachable"] = resp.status == 200
    except Exception:
        status["ollama_reachable"] = False

    if status["ollama_reachable"]:
        status["message"] = (
            f"وضع مغلق نشط — Ollama متاح ({url})، النموذج الافتراضي: {model}."
        )
    else:
        status["message"] = (
            f"وضع مغلق نشط — تعذّر الوصول لـ Ollama على {url}. "
            "سيُستخدم CKG Synthesis كاحتياطي محلي بالكامل."
        )
    return status


def disabled_online_features() -> list:
    """قائمة الميزات التي تُعطَّل صراحة في الوضع المغلق (للتوثيق/الواجهة)."""
    return [
        "بحث الويب",
        "بحث الصور",
        "تحويل نص↔صوت السحابي",
        "مزوّدو LLM السحابيون (Anthropic/Gemini/Groq/...)",
        "وكلاء يعتمدون على بحث ويب حي",
    ]

