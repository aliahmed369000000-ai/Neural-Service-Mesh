#!/usr/bin/env python3
"""
Agent User Assist — طبقة ودّية فوق الجسر والوكلاء
================================================
تحوّل عبارات المستخدم الطبيعية إلى أفعال واضحة + ردود قصيرة مفهومة
مع اقتراحات «ماذا بعد» — بدون صلاحيات خطرة.
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple


def _next_steps(*items: str) -> str:
    lines = ["", "### 👉 ماذا بعد؟"]
    for i, it in enumerate(items, 1):
        lines.append(f"{i}. {it}")
    return "\n".join(lines)


def welcome_card() -> str:
    return (
        "## 👋 أهلاً — أنا مساعدك داخل NSM\n\n"
        "أفهم هدفك بالعربية وأختار الأداة المناسبة (تدريب، MoE، فحص، اختبارات…).\n\n"
        "| ماذا تريد؟ | اكتب مثلاً |\n"
        "|------------|------------|\n"
        "| فهم قدراتي | `مساعدة` أو `ماذا تستطيع؟` |\n"
        "| تصنيف سؤال | `صنّف: ما حكم الصيام؟` |\n"
        "| صحة الخبراء | `صحة moe` |\n"
        "| فحص المشروع | `افحص المشروع` |\n"
        "| تدريب تجريبي | `مهمة تدريب data/samples/classification_demo.csv الهدف=label` |\n"
        "| تنفيذ آمن | `نفّذ بأمان: افحص المشروع وشغّل اختبارات` |\n"
        "| حالة النمو | `حالة نمو الوكيل` |\n"
        "| نبض المشروع | `تقرير النظام` أو `كيف حال النظام` |\n"
        + _next_steps(
            "جرّب `تقرير النظام` لنظرة شاملة",
            "أو `صنّف: كود بايثون لفرز قائمة`",
            "أو افتح تبويب **🧩 MoE والوكيل**",
        )
    )


def handle_user_assist(user_input: str) -> Optional[str]:
    """
    يعالج عبارات المساعدة والنوايا العامة.
    يعيد نصاً أو None لتمرير الطلب لطبقات أخرى.
    """
    text = (user_input or "").strip()
    if not text:
        return welcome_card()

    low = text.lower().strip()

    # ترحيب / مساعدة
    if re.search(
        r"^(مساعدة|help|ماذا\s*تستطيع|ما\s*قدراتك|كيف\s*أستخدم|ابدأ|مرحبا|السلام|"
        r"hi|hello|start|what\s*can\s*you)\b",
        text,
        re.I,
    ) or low in {"?", "؟", "أوامر", "commands"}:
        return welcome_card()

    # حالة / تقرير النظام ككل
    if re.search(
        r"(كيف\s*حال|حالة\s*النظام|هل\s*كل\s*شيء\s*يعمل|status\s*check|"
        r"تقرير\s*النظام|نبض\s*المشروع)",
        text,
        re.I,
    ):
        try:
            from ai.system_hub import format_system_report
            return format_system_report()
        except Exception as e:
            return f"تعذّر تقرير النظام: {e}"

    # نية طبيعية: افحص المشروع (بدون بادئة خطة:)
    if re.search(r"^(افحص|فحص)\s*(المشروع|النظام)?$", text, re.I):
        try:
            from ai.agent_growth_loop import run_safe_mission
            return run_safe_mission("افحص المشروع", execute=True)
        except Exception as e:
            return f"تعذّر الفحص: {e}"

    # نية: شغّل اختبارات
    if re.search(r"^(شغ[ّ]?ل|نف[ّ]?ذ|run)\s*(ال)?اختبارات?$", text, re.I):
        try:
            from ai.agent_growth_loop import run_safe_mission
            return run_safe_mission("شغّل اختبارات آمنة", execute=True)
        except Exception as e:
            return f"تعذّر تشغيل الاختبارات: {e}"

    # شرح مبسّط لـ MoE
    if re.search(r"(ما\s*هو\s*moe|اشرح\s*moe|خليط\s*الخبراء|ما\s*فائدة\s*الخبراء)", text, re.I):
        return (
            "## 🧩 ما هو Hierarchical MoE هنا؟\n\n"
            "نظام يختار **فئة** ثم **خبراء** محدودين (Top-K) لكل سؤال بدل تفعيل كل الشبكة.\n"
            "- أسرع مع مئات الخبراء\n"
            "- يمكن إضافة خبير جديد دون إعادة بناء الكل\n"
            "- يظهر في الإجابة شريط: الفئة · الثقة · الخبراء\n"
            + _next_steps(
                "`صنّف: ما حكم الصلاة؟`",
                "`صحة moe`",
                "افتح تبويب **🧩 MoE والوكيل**",
            )
        )

    # توجيه ودي إذا بدا الطلب غامضاً جداً وقصيراً
    if len(text) <= 2 and not re.search(r"[\u0600-\u06FF]{3,}|[a-zA-Z]{3,}", text):
        return welcome_card()

    return None


def suggest_after_bridge(reply: str, user_input: str) -> str:
    """يلحق اقتراحات قصيرة بعد رد الجسر إن لم تكن موجودة."""
    if not reply:
        return reply
    if "ماذا بعد" in reply or "👉" in reply:
        return reply
    u = (user_input or "").lower()
    tips: List[str] = []
    if "moe" in u or "صن" in user_input or "خبراء" in user_input:
        tips = ["`إحصاء moe`", "`حالة نمو الوكيل`", "جرّب سؤالاً آخر للتصنيف"]
    elif "تدريب" in user_input or "train" in u or "csv" in u:
        tips = [
            "للتشغيل الفعلي أضف `نفّذ`",
            "`سجل مهام التدريب`",
            "`مساعدة`",
        ]
    elif "اختبار" in user_input or "test" in u:
        tips = ["`حالة نمو الوكيل`", "`افحص المشروع`"]
    else:
        tips = ["`مساعدة` إن احتجت قائمة الأوامر", "`كيف حال النظام`"]
    return reply.rstrip() + _next_steps(*tips)
