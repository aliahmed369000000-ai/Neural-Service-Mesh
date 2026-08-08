"""
Production Activation — تفعيل الإنتاجية القصوى على الكود الحالي
===============================================================
يجمع حالة الربط للخطوات الأربع:
  1) DeepRouting داخل reasoning_pipeline
  2) تدريب ذاتي مستمر
  3) موصل Kaggle
  4) Multi-tenancy / AIaaS
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional


def activation_status() -> Dict[str, Any]:
    st: Dict[str, Any] = {"layers": {}}
    # 1 deep routing
    try:
        from ai.reasoning_pipeline import ReasoningPipeline
        import inspect
        sig = str(inspect.signature(ReasoningPipeline.__init__))
        st["layers"]["deep_routing_in_pipeline"] = "use_deep_routing" in sig
    except Exception as e:
        st["layers"]["deep_routing_in_pipeline"] = f"error:{e}"
    try:
        from ai.deep_routing_network import get_default_deep_network
        net = get_default_deep_network()
        st["layers"]["deep_routing_network"] = True
        st["layers"]["deep_routing_name"] = getattr(net, "name", "ok")
    except Exception as e:
        st["layers"]["deep_routing_network"] = str(e)
    # 2 continuous
    try:
        from ai.continuous_training_agent import assess_answer_quality
        st["layers"]["continuous_training"] = True
    except Exception as e:
        st["layers"]["continuous_training"] = str(e)
    # 3 kaggle connector
    try:
        from connectors.kaggle_training_connector import status
        st["layers"]["kaggle_connector"] = status()
    except Exception as e:
        st["layers"]["kaggle_connector"] = str(e)
    # 4 multi-tenant
    try:
        from ai.aiaas_platform import load_tenants_index, PLANS
        idx = load_tenants_index()
        st["layers"]["aiaas_tenants"] = len((idx or {}).get("tenants") or {})
        st["layers"]["aiaas_plans"] = list(PLANS.keys())
    except Exception as e:
        st["layers"]["aiaas_tenants"] = str(e)
    try:
        from ai.commercial_economy import dashboard
        st["layers"]["commercial"] = dashboard().get("ledger")
    except Exception as e:
        st["layers"]["commercial"] = str(e)
    return st


def roadmap_ar() -> str:
    st = activation_status()
    lines = [
        "## 🚀 ماذا بعد؟ — تفعيل الإنتاجية على الكود الحالي",
        "",
        "### 1) Orchestration Activation (Deep Routing)",
        f"- مدمج في `ReasoningPipeline`: **{st['layers'].get('deep_routing_in_pipeline')}**",
        f"- `DeepRoutingNetwork`: **{st['layers'].get('deep_routing_network')}**",
        "- الاستخدام: أنشئ `ReasoningPipeline(use_deep_routing=True)` — المزج الافتراضي 45%.",
        "",
        "### 2) تدريب ذاتي مستمر",
        f"- الوحدة: **{st['layers'].get('continuous_training')}**",
        "- أمر: `تدريب مستمر` أو `راقب جودة`",
        "- عند الضعف: خطة رفع epochs + تجهيز Kaggle (بدون دفع أعمى).",
        "",
        "### 3) أتمتة Kaggle/Colab",
        f"- الموصل: `{st['layers'].get('kaggle_connector')}`",
        "- أوامر: `حالة kaggle` · `جهّز kaggle` · `درّب بعيد kaggle وادفع` · `حمّل kaggle`",
        "- بعد التحقق: حدّث `models/` ثم commit مدروس.",
        "",
        "### 4) تجاري Multi-Tenancy",
        f"- مستأجرو AIaaS: **{st['layers'].get('aiaas_tenants')}** · خطط `{st['layers'].get('aiaas_plans')}`",
        "- الواجهة: تبويب **☁️ AIaaS والاقتصاد**",
        "- أوامر: `لوحة الاقتصاد` · إنشاء مستأجر من الواجهة",
        "",
        "### ترتيب تنفيذ مقترح هذا الأسبوع",
        "1. فعّل Deep Routing على مسار الأسئلة الحيّة (QA).",
        "2. شغّل `تدريب مستمر` على عيّنة إجابات ضعيفة.",
        "3. دورة Kaggle واحدة لأوزان ArabicTransformer مع اختبار قبل الدمج.",
        "4. أنشئ مستأجر Pro تجريبي وفاتورة تقديرية.",
        "",
        "```json",
        json.dumps(st, ensure_ascii=False, indent=2)[:2500],
        "```",
    ]
    return "\n".join(lines)


def handle_production_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(ماذا\s*بعد|خريط[ةه]\s*تطوير|production\s*activation|تفعيل\s*انتاج|انتاجيه\s*قصوى)", text, re.I):
        return roadmap_ar()
    if re.search(r"(حاله\s*التفعيل|activation\s*status)", text, re.I):
        return "## حالة التفعيل\n\n```json\n" + json.dumps(activation_status(), ensure_ascii=False, indent=2)[:3000] + "\n```"
    return None
