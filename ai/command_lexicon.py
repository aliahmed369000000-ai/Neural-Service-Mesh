"""
Command Lexicon — صياغة أوامر عربية طبيعية وموحّدة
=================================================
  • تطبيع النص (همزات، تشكيل، مسافات)
  • قاموس مرادفات → نية (intent)
  • مساعدة أوامر مصنّفة للطبقات: تدريب، منصات، معماري، عالِم

يُستدعى من model_training_agent قبل الموجّهات الفرعية.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple


def normalize_ar(text: str) -> str:
    """تطبيع خفيف للصياغة العربية العامية/الفصحى."""
    if not text:
        return ""
    t = unicodedata.normalize("NFKC", text).strip()
    # إزالة التشكيل
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    # توحيد الألف والياء والتاء
    repl = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ى": "ي",
        "ة": "ه",
        "ؤ": "و",
        "ئ": "ي",
        "ـ": "",
    }
    for a, b in repl.items():
        t = t.replace(a, b)
    # تنوين شائع
    t = t.replace("ً", "").replace("ٌ", "").replace("ٍ", "")
    # حروف عربية شائعة تُكتب بدل اللاتينية في أسماء المنصات
    arab_latin = {
        "كaggle": "kaggle",
        "كاجل": "kaggle",
        "كاغل": "kaggle",
        "كولاب": "colab",
        "كوالب": "colab",
    }
    for a, b in arab_latin.items():
        t = t.replace(a, b)
    # مسافات
    t = re.sub(r"\s+", " ", t)
    return t.lower()


# نية → عبارات طبيعية (بعد التطبيع)
# الأطول أولاً عند المطابقة
INTENT_PHRASES: Dict[str, List[str]] = {
    # مساعدة
    "help_all": [
        "اوامر",
        "الاوامر",
        "قائمه الاوامر",
        "مساعدة",
        "المساعده",
        "help",
        "ماذا تستطيع",
        "ايش تقدر",
        "وش تقدر تسوي",
        "كيف استخدمك",
    ],
    "help_train": ["اوامر التدريب", "مساعدة تدريب", "help training"],
    "help_platforms": ["اوامر المنصات", "مساعدة kaggle", "مساعدة colab"],
    "help_architect": ["اوامر المعماري", "مساعدة معماري"],
    "help_scientist": ["اوامر العالم", "مساعدة العالم", "اوامر الامن", "اوامر التكلفه"],
    # Meta-AI
    "meta_status": ["meta", "meta-ai", "حاله meta", "ذكاء خارق", "ميتا"],
    "meta_cycle": ["دوره meta", "شغل meta", "meta cycle"],
    "reason": ["فكر عميق", "سيناريو تدريب", "tree of thought", "reasoning"],
    "reflect": ["نقد ذاتي", "لماذا فشل", "مراجعه فشل"],
    "nas": ["تطور جيني", "neuroevolution", "nas", "جيل شبكات"],
    "hardware": ["تحسين عتاد", "hardware", "كرت الشاشه", "gpu profile"],
    "remember": ["تذكر", "احفظ خبره"],
    "recall": ["استرجع", "خبرات سابق", "ذاكر مشابه"],
    "mem_stats": ["احصاء الذاكره", "memory stats"],
    "super_status": ["super", "super-ai", "foundation", "منظومه فائق", "حاله super"],
    "super_cycle": ["دوره super", "شغل super", "super cycle"],
    "parallel3d": ["توازي ثلاثي", "3d parallel", "deepspeed", "megatron"],
    "synthetic": ["مصنع بيانات", "بيانات اصطناع", "synthetic data"],
    "agent_evo": ["تطور ذاتي", "سجل الوكيل", "agent v2"],
    "swarm": ["سرب", "swarm", "mesh", "وكلاء لامركز"],
    "economic_dash": ["لوحه الاقتصاد", "محرك اقتصاد", "ايرادات", "marketplace"],
    "roadmap": ["ماذا بعد", "خريطه تطوير", "تفعيل انتاج"],
    "continuous": ["تدريب مستمر", "راقب جوده", "صيان تدريب"],

    # جرد / حالة عامة
    "inventory": ["جرد", "مخزون", "ما المتاح", "بيئه التدريب", "inventory", "وش عندك"],
    "plan": ["خطه تدريب", "خطه", "دوره حياه", "lifecycle", "plan"],
    # معماراري
    "architect_status": ["حاله المعماري", "المهندس المعماري", "قدرات المعماري", "architect"],
    "architect_cycle": ["دوره معماريه", "شغل المعماري", "architect cycle"],
    "judge": [
        "حكم نموذج",
        "حكم النماذج",
        "تحكيم",
        "قيم نموذج",
        "تقرير تقييم",
        "قاضي النماذج",
        "judge",
    ],
    "tune": [
        "بحث فائق",
        "هندسه معلمات",
        "اضبط المعلمات",
        "ولّف معلمات",
        "ولف معلمات",
        "tuning",
        "hyperparam",
        "bayesian",
    ],
    "quantize": ["كمم نموذج", "تكميم", "quantiz", "int8"],
    "prune": ["قلم نموذج", "تقليم", "prun"],
    "compress": ["اضغط نموذج", "ضغط النموذج", "compression", "خفف نموذج"],
    "federated": [
        "تدريب اتحادي",
        "تعلم موحد",
        "خصوصيه البيانات",
        "fedavg",
        "federat",
        "بدون رفع بيانات",
    ],
    # عالم
    "scientist_status": ["حاله العالم", "العالم المبتكر", "scientist", "مدير مالي"],
    "scientist_cycle": ["دوره العالم", "شغل العالم", "scientist cycle"],
    "discover": [
        "اكتشف تنشيط",
        "ابحث تنشيط",
        "دوال تنشيط",
        "ابتكار خوارزم",
        "activation search",
    ],
    "merge": ["دمج نماذج", "ادمج الاوزان", "دمج الادمغه", "model merg"],
    "cost": ["تكلفه تدريب", "تكلفه", "ميزانيه", "عائد التكلفه", "roi", "cost"],
    "cheapest": ["ارخص مسار", "ارخص سيرفر", "spot", "سعر gpu"],
    "redteam": [
        "red team",
        "اختبار امني",
        "امن نموذج",
        "تحصين نموذج",
        "تسميم بيانات",
        "اختراق ذاتي",
    ],
    "promote": ["ترقيه نموذج", "promote model", "نشر نموذج"],
    "registry": ["سجل نماذج", "registry", "اعرض السجل"],
    # منصات
    "platforms": ["حاله المنصات", "platforms", "المنصات البعيده"],
    "kaggle_status": ["حاله kaggle", "kaggle status", "وضع كاجل", "وضع كaggle"],
    "kaggle_prepare": [
        "جهز kaggle",
        "جهز كاجل",
        "تدريب على kaggle",
        "حضر kaggle",
        "درب kaggle",
        "تدريب kaggle",
        "prepare kaggle",
    ],
    "kaggle_push": ["ادفع kaggle", "ارفع kaggle", "push kaggle", "شغل kaggle"],
    "kaggle_download": ["حمل kaggle", "نزل kaggle", "download kaggle"],
    "colab_mission": ["مهمه colab", "خلايا colab", "جهز colab", "colab mission"],
    "remote_train_kaggle": [
        "درب بعيد kaggle",
        "تدريب بعيد kaggle",
        "efficient kaggle",
        "درب كاجل فعال",
    ],
    "efficient_plan": ["خطه كفاءه", "تدريب فعال", "efficient plan"],
    # تدريب أساسي
    "ckg_status": ["حاله ckg", "ckg", "وضع ckg"],
    "train_clf": ["درب تصنيف تجريبي", "تصنيف تجريبي"],
    "train_reg": ["درب انحدار تجريبي", "انحدار تجريبي"],
    "train_torch": ["درب شبكه torch", "شبكه torch"],
    "gpu_status": ["حاله gpu", "وضع gpu", "gpu"],
}


# صياغة موصى بها للعرض في المساعدة (فصحى واضحة)
HELP_CATALOG: List[Tuple[str, List[Tuple[str, str]]]] = [
    (
        "أساسية",
        [
            ("جرد", "عرض المكتبات والبيانات والسكربتات"),
            ("خطة", "دورة حياة تدريب نموذج"),
            ("أوامر", "هذه القائمة"),
        ],
    ),
    (
        "تدريب محلي",
        [
            ("درّب تصنيف تجريبي", "نموذج تصنيف سريع"),
            ("درّب انحدار تجريبي", "نموذج انحدار"),
            ("درّب شبكة torch", "MLP بـ PyTorch"),
            ("حالة ckg", "تقدّم تدريب CKG"),
            ("حالة gpu", "جهاز CUDA الحالي"),
        ],
    ),
    (
        "منصات بعيدة",
        [
            ("حالة المنصات", "Kaggle + Colab + الجهاز"),
            ("جهّز kaggle", "مهمة Kernel جاهزة للرفع"),
            ("ادفع kaggle <id>", "تشغيل على GPU"),
            ("حمّل kaggle <id>", "تنزيل الأوزان"),
            ("مهمة colab", "خلايا جاهزة للصق في Colab"),
            ("درّب بعيد kaggle وادفع", "تدريب فعّال + دفع"),
            ("خطة كفاءة", "AMP / DataParallel / Early stop"),
        ],
    ),
    (
        "مهندس معماري",
        [
            ("حالة المعماري", "القدرات الهندسية"),
            ("دورة معمارية", "بحث فائق + تحكيم + ضغط + اتحاد"),
            ("حكّم نموذج", "تقرير تقييم + خطة إصلاح"),
            ("بحث فائق 12 تجربة", "أفضل معلمات"),
            ("كمّم نموذج", "ضغط int8"),
            ("قلّم نموذج 40%", "إزالة أوزان ضعيفة"),
            ("اضغط نموذج", "تقليم ثم تكميم"),
            ("تدريب اتحادي 5 عملاء", "FedAvg بخصوصية"),
        ],
    ),
    (
        "عالِم ومال وأمن ونشر",
        [
            ("حالة العالم", "طبقة البحث/المال/الأمن"),
            ("دورة العالم", "تشغيل القدرات الأربع"),
            ("اكتشف تنشيط", "مقارنة دوال التنشيط"),
            ("دمج نماذج", "دمج أوزان A+B"),
            ("تكلفة تدريب", "قرار عائد/تكلفة"),
            ("أرخص مسار", "ترتيب أسعار GPU"),
            ("اختبار أمني", "Red Team دفاعي"),
            ("ترقية نموذج score=0.91", "سجل + قرار نشر"),
            ("سجل نماذج", "البطل والمنافسون"),
        ],
    ),
    (
        "Meta-AI",
        [
            ("حالة meta", "تفكير / NAS / عتاد / ذاكرة"),
            ("دورة meta", "تشغيل قدرات Meta"),
            ("فكر عميق …", "Tree of Thoughts"),
            ("تطور جيني N أجيال", "Neuroevolution"),
            ("تحسين عتاد", "توصيات GPU"),
            ("تذكر / استرجع", "ذاكرة متجهة"),
        ],
    ),
    (
        "Super AI Orchestrator",
        [
            ("حالة super", "سقف المنظومة"),
            ("دورة super", "حوسبة+بيانات+تطور+سرب"),
            ("توازي ثلاثي 7B 8 gpu", "DP/TP/PP + DeepSpeed"),
            ("مصنع بيانات 100 عينة", "توليد وتصفية"),
            ("تطور ذاتي score=0.85", "إصدار وكيل جديد"),
            ("سجل الوكيل", "إصدارات Foundation Agent"),
            ("حالة السرب", "شبكة Mesh"),
            ("حاكِ مزامنة كوكبية", "تبادل خبرات"),
            ("لوحة الاقتصاد", "إيرادات القنوات الأربع"),
            ("انشر نموذج", "كتالوج السوق"),
            ("هامش spot", "Compute arbitrage"),
            ("بيع بيانات", "تسعير اصطناعي"),
        ],
    ),
]


def match_intent(text: str) -> Optional[str]:
    """يعيد أول نية تطابق عبارة كاملة أو ضمن النص بعد التطبيع."""
    n = normalize_ar(text)
    if not n:
        return None
    # مطابقة العبارات الأطول أولاً
    candidates: List[Tuple[int, str]] = []
    for intent, phrases in INTENT_PHRASES.items():
        for ph in phrases:
            pn = normalize_ar(ph)
            if not pn:
                continue
            if n == pn or pn in n:
                candidates.append((len(pn), intent))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]


def help_markdown(section: Optional[str] = None) -> str:
    lines = [
        "## أوامر وكيل NSM (صياغة موحّدة)",
        "",
        "اكتب الأمر بعبارة طبيعية؛ الهمزات والتشكيل لا تهم غالباً.",
        "أمثلة: `جهز كaggle` = `جهّز kaggle` · `حكم نموذج` = `حكّم نموذج`",
        "",
    ]
    for title, rows in HELP_CATALOG:
        if section and section not in title and normalize_ar(section) not in normalize_ar(title):
            continue
        lines.append(f"### {title}")
        for cmd, desc in rows:
            lines.append(f"- `{cmd}` — {desc}")
        lines.append("")
    lines.append("للتفصيل: `أوامر التدريب` · `أوامر المنصات` · `أوامر المعماري` · `أوامر العالم`")
    return "\n".join(lines)


def rewrite_to_canonical(text: str) -> str:
    """
    إن وُجدت نية واضحة، يُعاد نصاً قانونياً يفهمه الموجّهون الحاليون.
    وإلا يُعاد النص الأصلي.
    """
    intent = match_intent(text)
    if not intent:
        return text
    canonical = {
        "help_all": "أوامر",
        "help_train": "أوامر التدريب",
        "help_platforms": "أوامر المنصات",
        "help_architect": "أوامر المعماري",
        "help_scientist": "أوامر العالم",
        "inventory": "جرد",
        "plan": "خطة",
        "architect_status": "حالة المعماري",
        "architect_cycle": "دورة معمارية",
        "judge": "حكّم نموذج",
        "tune": text if re.search(r"\d+", text) else "بحث فائق 10 تجارب",
        "quantize": "كمّم نموذج",
        "prune": text if "%" in text or re.search(r"\d+", text) else "قلّم نموذج 50%",
        "compress": "اضغط نموذج",
        "federated": text if re.search(r"\d+", text) else "تدريب اتحادي 5 عملاء 6 جولات",
        "scientist_status": "حالة العالم",
        "scientist_cycle": "دورة العالم",
        "discover": "اكتشف تنشيط",
        "merge": text if "alpha" in text.lower() or "α" in text else "دمج نماذج",
        "cost": "تكلفة تدريب",
        "cheapest": "أرخص مسار",
        "redteam": "red team",
        "promote": text if "score" in text.lower() else "ترقية نموذج score=0.85",
        "registry": "سجل نماذج",
        "platforms": "حالة المنصات",
        "kaggle_status": "حالة kaggle",
        "kaggle_prepare": "جهّز kaggle",
        "kaggle_push": text,  # يحتاج id
        "kaggle_download": text,
        "colab_mission": "مهمة colab",
        "remote_train_kaggle": text if "ادفع" in text or "دفع" in text else "درّب بعيد kaggle",
        "efficient_plan": "خطة كفاءة",
        "ckg_status": "حالة ckg",
        "train_clf": "درّب تصنيف تجريبي",
        "train_reg": "درّب انحدار تجريبي",
        "train_torch": "درّب شبكة torch",
        "gpu_status": "حالة gpu",
        "meta_status": "حالة meta",
        "meta_cycle": "دورة meta",
        "reason": text,
        "reflect": text,
        "nas": text,  # يحافظ على عدد الأجيال/الشبكات
        "hardware": text,
        "remember": text,
        "recall": text,
        "mem_stats": "إحصاء الذاكرة",
        "super_status": "حالة super",
        "super_cycle": "دورة super",
        "parallel3d": text,
        "synthetic": text,
        "agent_evo": text,
        "swarm": text,
        "economic_dash": "لوحة الاقتصاد",
        "roadmap": "ماذا بعد",
        "continuous": "تدريب مستمر",
    }.get(intent, text)
    return canonical


def handle_help_command(user_input: str) -> Optional[str]:
    n = normalize_ar(user_input or "")
    intent = match_intent(user_input or "")
    if intent == "help_train":
        return help_markdown("تدريب")
    if intent == "help_platforms":
        return help_markdown("منصات")
    if intent == "help_architect":
        return help_markdown("مهندس")
    if intent == "help_scientist":
        return help_markdown("عالِم")
    if intent == "help_all" or n in ("اوامر", "مساعدة", "help"):
        return help_markdown()
    # عبارات مساعدة عامة
    if re.search(r"(اوامر|مساعده|help|ماذا تستطيع|ايش تقدر|وش تقدر)", n):
        return help_markdown()
    return None
