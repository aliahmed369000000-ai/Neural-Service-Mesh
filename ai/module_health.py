"""
ai/module_health.py
فحص تشخيصي لكل الوحدات الاختيارية التي يستوردها app_core.py ضمن كتل
try/except Exception. تلك الكتل تُسكِت سبب الفشل الفعلي (لا `as e`، لا
تسجيل) — وهذا بالضبط ما أخفى عطل RoutingEngine (ai/models/initial_weights.npy
المفقود) لمدة طويلة دون أي إشارة في الواجهة.

هذا الملف لا يُعدّل app_core.py؛ بدلاً من ذلك يُعيد محاولة نفس الاستيرادات
بشكل مستقل عبر importlib ويلتقط رسالة الاستثناء الحقيقية، ليعرضها تبويب
"صحة النظام" بدل الاختفاء الصامت.
"""
from __future__ import annotations

import importlib
import traceback
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModuleCheck:
    flag: str            # اسم متغير الحالة في app_core.py (مثال: _NSM_BRIDGE_OK)
    label: str            # اسم عربي مقروء
    module: str            # المسار المستورَد فعلياً (importlib.import_module)
    note: str = ""         # سياق إضافي مختصر
    ok: bool = False
    reason: str = ""       # نص الاستثناء الحقيقي عند الفشل
    traceback_tail: str = ""


# ── قائمة الوحدات الاختيارية — يجب أن تبقى متوافقة مع كتل try/except في
#    app_core.py (الأسطر ~256–529 وقت كتابة هذا الملف). أي وحدة جديدة
#    تُضاف هناك بنفس النمط يُستحسن إضافتها هنا أيضاً.
_OPTIONAL_MODULES: List[ModuleCheck] = [
    ModuleCheck("_ROUTE_LOG_DB_OK",     "سجل التوجيه (SQLite)",         "ai.route_log_store"),
    ModuleCheck("_QUALITY_SCORER_OK",   "مقيّم جودة الرد",              "ai.quality_scorer"),
    ModuleCheck("_STT_OK",              "تفريغ صوت→نص (STT)",           "ai.stt_engine"),
    ModuleCheck("_TTS_OK",              "قراءة نص→صوت (TTS)",           "ai.tts_engine"),
    ModuleCheck("_HARM_CLASSIFIER_OK",  "طبقة فحص الأمان الأولى",       "ai.harm_classifier"),
    ModuleCheck("_AGENTS_HUB_OK",       "مركز وكلاء AI المتخصصون",      "ai.agent_categories"),
    ModuleCheck("_FABLE_OK",            "محرك السرد الإبداعي 🎭",       "ai.fable_engine"),
    ModuleCheck("_PDF_EXPORT_OK",       "تصدير PDF",                    "ai.pdf_export"),
    ModuleCheck("_WEB_SEARCH_OK",       "أداة البحث في الويب",          "ai.web_search_tool"),
    ModuleCheck("_ARABIC_NLP_OK",       "محرك المعالجة اللغوية العربية", "ai.arabic_nlp"),
    ModuleCheck("_EPISODIC_OK",         "الذاكرة العرضية (Episodic)",   "ai.episodic_memory"),
    ModuleCheck("_CONSOLIDATOR_OK",     "مُوحِّد الذاكرة",               "ai.memory_consolidator"),
    ModuleCheck("_CHECKPOINT_OK",       "نقاط حفظ الدماغ (Checkpoint)", "ai.brain_checkpoint"),
    ModuleCheck("_GITHUB_SYNC_OK",      "مزامنة GitHub",                "ai.github_sync"),
    ModuleCheck("_AUTOTUNE_OK",         "الضبط التلقائي للتغذية الراجعة", "ai.autotune_feedback"),
    ModuleCheck("_SELF_AWARE_OK",       "محرك الوعي الذاتي",            "ai.self_awareness"),
    ModuleCheck("_NEURAL_CORE_OK",      "النواة العصبية (NeuralCore)",  "ai.neural_core"),
    ModuleCheck("_WORLD_FEED_OK",       "تغذية العالم / جودة / مناعة",  "ai.world_feed",
                note="يشمل أيضاً ai.quality_engine و ai.immune_system"),
    ModuleCheck("_SELF_NARRATIVE_OK",   "السرد الذاتي",                 "ai.self_narrative"),
    ModuleCheck("_GOAL_PLANNER_OK",     "مخطِّط الأهداف",               "ai.goal_planner"),
    ModuleCheck("_META_REASONER_OK",    "المُحلِّل الفوقي (Meta)",       "ai.meta_reasoner"),
    ModuleCheck("_ORCHESTRATOR_OK",     "منسّق الوكلاء (godmode)",      "ai.godmode"),
    ModuleCheck("_SWARM_OK",            "🐝 السرب الذكي",               "ai.agent_factory",
                note="يشمل أيضاً ai.swarm_coordinator"),
    ModuleCheck("_ULTRAPLINIAN_OK",     "Ultraplinian (سباق نماذج)",    "ai.ultraplinian"),
    ModuleCheck("_NSM_BRIDGE_OK",       "NSM Router Bridge",            "ai.nsm_router_bridge",
                note="RoutingEngine + ScoringEngine + MemoryEngine + LearningValidator"),
    ModuleCheck("_NSM_SEMANTIC_OK",     "الموجّه الدلالي (Semantic Router)", "ai.semantic_router"),
]


def run_module_health_checks(deep: bool = True) -> List[ModuleCheck]:
    """
    يعيد محاولة استيراد كل وحدة اختيارية بشكل مستقل ويلتقط سبب الفشل
    الحقيقي. لا يُعدّل حالة app_core.py الموجودة أصلاً — فحص تشخيصي فقط.

    deep=True: لبعض الوحدات (مثل nsm_router_bridge) يُستدعى أيضاً
    فحص جاهزية إضافي (is_ready()) إن وُجد، لأن نجاح import وحده لا يكفي
    لبعضها (استيراد قد ينجح لكن التهيئة الداخلية تفشل).
    """
    results: List[ModuleCheck] = []
    for spec in _OPTIONAL_MODULES:
        check = ModuleCheck(spec.flag, spec.label, spec.module, spec.note)
        try:
            mod = importlib.import_module(spec.module)
            check.ok = True
            if deep and spec.module == "ai.nsm_router_bridge" and hasattr(mod, "is_ready"):
                try:
                    ready = mod.is_ready()
                    if not ready:
                        check.ok = False
                        check.reason = "الاستيراد نجح لكن is_ready() أعادت False (فشل تهيئة داخلي)"
                except Exception as e_ready:
                    check.ok = False
                    check.reason = f"الاستيراد نجح لكن is_ready() فشلت: {e_ready}"
        except Exception as e:
            check.ok = False
            check.reason = str(e) or type(e).__name__
            check.traceback_tail = "".join(traceback.format_exception_only(type(e), e)).strip()
        results.append(check)
    return results


def module_health_summary(results: Optional[List[ModuleCheck]] = None) -> dict:
    """ملخص رقمي مختصر لعرضه في مقياس (st.metric) أو تقرير."""
    results = results if results is not None else run_module_health_checks()
    ok = sum(1 for r in results if r.ok)
    return {
        "total": len(results),
        "ok": ok,
        "failed": len(results) - ok,
        "failed_labels": [r.label for r in results if not r.ok],
    }
