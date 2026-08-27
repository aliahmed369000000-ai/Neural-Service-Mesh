"""
Free GPU Providers Registry — مزوّدو GPU مجاني / رصيد مجاني + API Key
====================================================================
كتالوج موحّد للمشاريع التدريبية في NSM Notebook.

ملاحظات صادقة:
  • «مجاني» غالباً = حصة أسبوعية/شهرية أو رصيد ترحيبي، وليس GPU بلا حدود.
  • Colab لا يوفّر REST API رسمياً لتأجير GPU؛ الاعتماد على دفاتر/جسر.
  • Kaggle API مناسب لرفع kernels ومهام تدريب.
  • Modal / Lightning / HF يدعمون API أو CLI بمفتاح.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

# كتالوج ثابت — يُحدَّث يدوياً عند تغيّر العروض
FREE_GPU_CATALOG: List[Dict[str, Any]] = [
    {
        "id": "kaggle",
        "name": "Kaggle Notebooks",
        "tier": "free_quota",
        "gpu": "P100 / T4 (أحياناً T4×2)",
        "quota_ar": "~30 ساعة GPU أسبوعياً (+ TPU منفصل)",
        "session_limit_ar": "حتى ~9–12 ساعة للجلسة",
        "api_key_env": ["KAGGLE_USERNAME", "KAGGLE_KEY"],
        "signup_url": "https://www.kaggle.com/",
        "api_docs": "https://github.com/Kaggle/kaggle-api",
        "best_for": "تدريب دفاتر، استمرارية أفضل من Colab المجاني",
        "nsm_module": "ai.kaggle_provider",
        "supports_api_submit": True,
        "priority": 1,
    },
    {
        "id": "modal",
        "name": "Modal",
        "tier": "free_credits",
        "gpu": "T4 / L4 / A10 / A100 (حسب الرصيد)",
        "quota_ar": "رصيد مجاني شهري تقريبي (يُحدَّث من Modal؛ غالباً عشرات $)",
        "session_limit_ar": "Serverless — بلا انقطاع جلسة متصفح",
        "api_key_env": ["MODAL_TOKEN_ID", "MODAL_TOKEN_SECRET"],
        "signup_url": "https://modal.com/",
        "api_docs": "https://modal.com/docs",
        "best_for": "تدريب ودوال GPU عبر API/CLI بدون إدارة سيرفر",
        "nsm_module": "ai.free_gpu_providers",
        "supports_api_submit": True,
        "priority": 2,
    },
    {
        "id": "lightning",
        "name": "Lightning AI",
        "tier": "free_credits",
        "gpu": "T4 / A10G (حسب الخطة)",
        "quota_ar": "رصيد شهري مجاني محدود (~ساعات GPU)",
        "session_limit_ar": "وظائف سحابية — أقل عرضة لقطع Colab",
        "api_key_env": ["LIGHTNING_API_KEY", "LIGHTNING_USER_ID"],
        "signup_url": "https://lightning.ai/",
        "api_docs": "https://lightning.ai/docs",
        "best_for": "PyTorch Lightning / Studios",
        "nsm_module": "ai.free_gpu_providers",
        "supports_api_submit": True,
        "priority": 3,
    },
    {
        "id": "huggingface",
        "name": "Hugging Face Jobs / Spaces GPU",
        "tier": "free_limited",
        "gpu": "T4 وغيرها حسب المنتج",
        "quota_ar": "حصص مجانية محدودة + ترقية مدفوعة",
        "session_limit_ar": "Jobs/Spaces حسب الإعداد",
        "api_key_env": ["HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"],
        "signup_url": "https://huggingface.co/",
        "api_docs": "https://huggingface.co/docs/huggingface_hub",
        "best_for": "تشغيل مهام مرتبطة بنماذج HF",
        "nsm_module": "ai.free_gpu_providers",
        "supports_api_submit": True,
        "priority": 4,
    },
    {
        "id": "colab",
        "name": "Google Colab",
        "tier": "free_quota",
        "gpu": "غالباً T4",
        "quota_ar": "ديناميكي ~12–30 ساعة/أسبوع (غير مضمون وقت الذروة)",
        "session_limit_ar": "قطع متكرر — غير مثالي للتدريب الطويل",
        "api_key_env": [],  # لا API GPU رسمي
        "signup_url": "https://colab.research.google.com/",
        "api_docs": "https://colab.research.google.com/",
        "best_for": "تجربة سريعة؛ استخدم دفاتر notebooks/*Colab*",
        "nsm_module": "scripts.colab_bootstrap",
        "supports_api_submit": False,
        "priority": 5,
    },
    {
        "id": "nvidia_nim",
        "name": "NVIDIA NIM / build.nvidia.com",
        "tier": "free_inference",
        "gpu": "استضافة استدلال (ليس تدريب حر كامل)",
        "quota_ar": "طبقة مجانية للاستدلال عبر API",
        "session_limit_ar": "حسب الحصة",
        "api_key_env": ["NVIDIA_API_KEY"],
        "signup_url": "https://build.nvidia.com/",
        "api_docs": "https://docs.nvidia.com/nim/",
        "best_for": "استدلال نماذج وليس استبدال تدريب كامل",
        "nsm_module": None,
        "supports_api_submit": True,
        "priority": 8,
    },
    {
        "id": "groq",
        "name": "Groq",
        "tier": "free_inference",
        "gpu": "LPU استدلال سريع",
        "quota_ar": "طبقة مجانية بمفتاح API",
        "session_limit_ar": "حدود rate limit",
        "api_key_env": ["GROQ_API_KEY"],
        "signup_url": "https://console.groq.com/",
        "api_docs": "https://console.groq.com/docs",
        "best_for": "استدلال LLM وليس تدريب من الصفر",
        "nsm_module": None,
        "supports_api_submit": True,
        "priority": 9,
    },
    {
        "id": "openrouter_free",
        "name": "OpenRouter (:free models)",
        "tier": "free_inference",
        "gpu": "نماذج مجانية عبر وسيط",
        "quota_ar": "نماذج بلاحقة :free عند التوفر",
        "session_limit_ar": "حسب المزوّد الفرعي",
        "api_key_env": ["OPENROUTER_API_KEY"],
        "signup_url": "https://openrouter.ai/",
        "api_docs": "https://openrouter.ai/docs",
        "best_for": "استدلال مجاني متعدد النماذج",
        "nsm_module": None,
        "supports_api_submit": True,
        "priority": 10,
    },
    {
        "id": "vast",
        "name": "Vast.ai",
        "tier": "marketplace",
        "gpu": "سوق أسعار منخفضة",
        "quota_ar": "مدفوع حسب العرض",
        "session_limit_ar": "حسب المثيل",
        "api_key_env": ["VAST_API_KEY"],
        "signup_url": "https://vast.ai/",
        "api_docs": "https://vast.ai/docs/",
        "best_for": "تدريب رخيص نسبياً",
        "nsm_module": "ai.remote_gpu_provider",
        "supports_api_submit": True,
        "priority": 7,
    },
]


def list_free_gpu_providers(
    include_paid: bool = True,
    training_only: bool = False,
) -> List[Dict[str, Any]]:
    rows = []
    for p in FREE_GPU_CATALOG:
        if not include_paid and p.get("tier") in ("paid_with_credits", "marketplace"):
            continue
        if training_only and p.get("tier") == "free_inference":
            continue
        rows.append(dict(p))
    rows.sort(key=lambda x: int(x.get("priority", 99)))
    return rows


def provider_env_status(provider_id: Optional[str] = None) -> Dict[str, Any]:
    """هل مفاتيح API مضبوطة في البيئة؟ (بدون إظهار القيم)."""
    out: Dict[str, Any] = {}
    items = FREE_GPU_CATALOG
    if provider_id:
        items = [p for p in FREE_GPU_CATALOG if p["id"] == provider_id]
    for p in items:
        envs = p.get("api_key_env") or []
        configured = {}
        for k in envs:
            configured[k] = bool(os.environ.get(k, "").strip())
        out[p["id"]] = {
            "name": p["name"],
            "keys_required": envs,
            "keys_present": configured,
            "ready": all(configured.values()) if envs else p.get("supports_api_submit") is False,
            "supports_api_submit": p.get("supports_api_submit"),
            "signup_url": p.get("signup_url"),
        }
    return out


def recommended_stack_ar() -> str:
    return (
        "## 🆓 مزوّدو GPU مجاني / شبه مجاني لـ NSM\n\n"
        "### للتدريب (أفضل استمرارية)\n"
        "1. **Kaggle** — 30 ساعة/أسبوع تقريباً + API (`KAGGLE_USERNAME` + `KAGGLE_KEY`)\n"
        "2. **Modal** — رصيد مجاني + API (`MODAL_TOKEN_ID` / `MODAL_TOKEN_SECRET`)\n"
        "3. **Lightning AI** — رصيد شهري (`LIGHTNING_API_KEY`)\n"
        "4. **Hugging Face** — `HF_TOKEN` لمهام مرتبطة بـ Hub\n\n"
        "### للتجربة السريعة\n"
        "- **Google Colab** — بدون API GPU رسمي؛ استخدم دفاتر المشروع (قد ينقطع)\n\n"
        "### استدلال مجاني (ليس تدريب أوزان كامل)\n"
        "- Groq · OpenRouter :free · NVIDIA NIM\n\n"
        "### مستقر وطويل (مدفوع غالباً)\n"
        "- Vast.ai\n\n"
        "**نصيحة NSM:** للتدريب دون انقطاع Colab → **Kaggle API** أولاً، ثم Modal/Lightning."
    )


def plan_for_provider(provider_id: str, notebook_id: str = "") -> Dict[str, Any]:
    meta = next((p for p in FREE_GPU_CATALOG if p["id"] == provider_id), None)
    if not meta:
        return {"ok": False, "error": f"مزوّد غير معروف: {provider_id}"}
    env = provider_env_status(provider_id).get(provider_id, {})
    steps = []
    if meta["id"] == "kaggle":
        steps = [
            "أنشئ حساباً وتحقق بالهاتف على kaggle.com",
            "أنشئ API token وضع KAGGLE_USERNAME و KAGGLE_KEY في Streamlit Secrets",
            "استخدم ai.kaggle_provider أو connectors.kaggle_training_connector",
            "ارفع الدفتر/الـ kernel وشغّل على GPU T4/P100",
        ]
    elif meta["id"] == "modal":
        steps = [
            "pip install modal && modal setup",
            "اضبط MODAL_TOKEN_ID و MODAL_TOKEN_SECRET",
            "لف سكربت التدريب في @app.function(gpu='T4')",
            "modal run / modal deploy",
        ]
    elif meta["id"] == "lightning":
        steps = [
            "سجّل في lightning.ai واحصل على API key",
            "LIGHTNING_API_KEY في Secrets",
            "lightning run model --accelerator gpu …",
        ]
    elif meta["id"] == "huggingface":
        steps = [
            "HF_TOKEN من huggingface.co/settings/tokens",
            "huggingface_hub أو Jobs حسب التوثيق الحالي",
        ]
    elif meta["id"] == "colab":
        steps = [
            "افتح notebooks/*Colab*.ipynb",
            "Runtime → GPU",
            "scripts/colab_bootstrap.py لربط المستودع",
        ]
    else:
        steps = [f"راجع {meta.get('api_docs')}", f"مفاتيح: {', '.join(meta.get('api_key_env') or [])}"]

    return {
        "ok": True,
        "provider": meta,
        "env": env,
        "notebook_id": notebook_id,
        "steps": steps,
        "ready_to_submit": bool(env.get("ready")) and bool(meta.get("supports_api_submit")),
    }
