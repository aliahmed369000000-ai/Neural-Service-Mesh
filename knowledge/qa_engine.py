"""
qa_engine.py
============
محرك الأسئلة والأجوبة القرآني — Quran Knowledge Q&A Engine
يستخدم فقط:
  - 6236 آية قرآنية
  - 173 مفهوم في الـ CKG
  - 2149 علاقة دلالية
  - 633 جذر عربي مفهرس

لا يضيف طبقات عصبية جديدة ولا مصادر خارجية — يعمل فوق البنية الحالية فقط.
"""

from __future__ import annotations

import re
import logging
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# طبقة تعزيز عصبي اختيارية (ArabicTransformer 40M) — إضافية فوق قواعد qa_engine
# ═══════════════════════════════════════════════════════════════════════════
# مبدأ التصميم: qa_engine.py يبقى يعمل بالكامل بمفرده حتى لو فشلت هذه الطبقة
# تماماً (استيراد، تحميل أوزان، أو أي استثناء أثناء التنفيذ) — أي خطأ هنا
# يُصمَّت ويُرجَع الترتيب الأصلي دون تعديل. لا يوجد أي مسار يعتمد على نجاح
# هذه الطبقة.

_NEURAL_BOOSTER = None       # ArabicTransformer المحمَّل، أو False لو فشل نهائياً
_NEURAL_BOOST_TRIED = False  # نحاول التحميل مرة واحدة فقط (لا نعيد المحاولة كل سؤال)


def _neural_booster_checkpoint_ready(weights_dir: str = "models/transformer_ckg_v3") -> bool:
    """
    فحص خفيف جداً (لا يبني الشبكة، لا يحمّل الأوزان كاملة في الذاكرة) قبل
    استنشاء ArabicTransformer — الذي يخصّص فعلياً ~4-5GB بالمعمارية
    الحالية (D_MODEL×N_LAYERS الكبيرين، ~1 مليار معامل) حتى بلا أي
    checkpoint فعلي، لأن __init__ يبني كل الطبقات بأوزان عشوائية أولاً
    ثم يحاول load() لاحقاً.

    ملاحظة (تحديث): weights_dir الافتراضي كان يشير سابقاً إلى
    models/transformer_ckg_v1 — مسار مؤرشَف من معمارية قديمة (40M) لا
    علاقة له بمسار التدريب الفعلي الحالي. train_batch_v3.py يحفظ فعلياً
    في models/transformer_ckg_v3 (WEIGHTS_DIR)، وهو نفس المسار الذي
    يستثنيه .gitignore عمداً لأنه ينتج محلياً/عبر Kaggle، لذا صُحِّح
    الافتراضي هنا ليطابق مخرجات التدريب الحقيقية بدل مسار لم يعد موجوداً.

    الفحص الفعلي (توافق الأبعاد عبر mmap) منقول الآن إلى دالة مشتركة
    ai.arabic_transformer.checkpoint_dims_ready() تُستخدم أيضاً من
    reasoning_pipeline.py لتفادي ازدواج نفس المنطق الحرِج في مكانين.

    بدون هذا الفحص: أي استدعاء answer_question() في بيئة محدودة الذاكرة
    يتعرّض لخطر OOM حقيقي لبناء نموذج لن يُحمَّل منه شيء أصلاً، وحتى في
    بيئة بذاكرة كافية كان سيُستخدَم بأوزان عشوائية غير مدرَّبة بصمت
    لإعادة ترتيب المفاهيم (_apply_neural_boost) — تشويش على دقة الإجابات
    لا مجرد مشكلة أداء.
    """
    try:
        from ai.arabic_transformer import checkpoint_dims_ready
        return checkpoint_dims_ready(weights_dir)
    except Exception:
        return False


def _get_neural_booster():
    """يُحمِّل ArabicTransformer مرة واحدة (lazy singleton). يُرجع None عند أي فشل."""
    global _NEURAL_BOOSTER, _NEURAL_BOOST_TRIED
    if _NEURAL_BOOST_TRIED:
        return _NEURAL_BOOSTER if _NEURAL_BOOSTER is not False else None
    _NEURAL_BOOST_TRIED = True
    if not _neural_booster_checkpoint_ready():
        _NEURAL_BOOSTER = False
        logger.info(
            "[qa_engine] لا يوجد checkpoint متوافق لـArabicTransformer "
            "(models/transformer_ckg_v3) — تخطّي التعزيز العصبي بأمان "
            "(الترتيب الأساسي يبقى كما هو، بلا استنشاء نموذج بلا فائدة)."
        )
        return None
    try:
        from ai.arabic_transformer import ArabicTransformer
        t = ArabicTransformer()
        t.load("models/transformer_ckg_v3")
        _NEURAL_BOOSTER = t
        logger.info("[qa_engine] طبقة التعزيز العصبي (ArabicTransformer) محمَّلة بنجاح")
    except Exception as e:
        _NEURAL_BOOSTER = False
        logger.warning(f"[qa_engine] تعذّر تحميل طبقة التعزيز العصبي — "
                        f"سيعمل qa_engine بقواعده الأصلية فقط: {e}")
    return _NEURAL_BOOSTER if _NEURAL_BOOSTER is not False else None


# ═══════════════════════════════════════════════════════════════════════════
# LoRA: تدريب خفيف إضافي من ملاحظات المستخدمين (لا يمسّ الأوزان الأساسية)
# ═══════════════════════════════════════════════════════════════════════════
_LORA_ADAPTER = None
_LORA_TRIED = False
_LORA_FEEDBACK_COUNT = 0
LORA_SAVE_DIR = "models/lora_feedback_v1"
LORA_SAVE_EVERY = 10  # حفظ الـadapter كل 10 ملاحظات إيجابية

_LORA_LOSS_WINDOW: List[float] = []  # آخر قيم loss لتقدير الجودة عند الحفظ

_ROLLBACK_GUARD = None
_ROLLBACK_GUARD_TRIED = False


def _get_lora_rollback_guard():
    """
    يُحمِّل CheckpointGuard (ai/rollback_guard.py) مرة واحدة. يُرجع None عند
    أي فشل — عندها يعود الحفظ لسلوكه القديم بلا حماية (fallback آمن، لا كسر).
    """
    global _ROLLBACK_GUARD, _ROLLBACK_GUARD_TRIED
    if _ROLLBACK_GUARD_TRIED:
        return _ROLLBACK_GUARD if _ROLLBACK_GUARD is not False else None
    _ROLLBACK_GUARD_TRIED = True
    try:
        from ai.rollback_guard import CheckpointGuard
        _ROLLBACK_GUARD = CheckpointGuard(asset="lora_feedback_v1")
        logger.info("[qa_engine] CheckpointGuard محمَّل — حفظ LoRA محميّ من التراجع الآن")
    except Exception as e:
        _ROLLBACK_GUARD = False
        logger.warning(f"[qa_engine] تعذّر تحميل rollback_guard — الحفظ سيتم بلا حماية: {e}")
    return _ROLLBACK_GUARD if _ROLLBACK_GUARD is not False else None


def _save_lora_with_guard(lora) -> None:
    """
    يحفظ LoRA adapter عبر CheckpointGuard إن تحمَّل بنجاح: لقطة احتياطية قبل
    الحفظ، وتقييم متوسط الخسارة الأخيرة بعده — لو تدهورت الجودة عن آخر نسخة
    مقبولة يُستعاد الملف القديم تلقائياً بدل الكتابة فوقه (هذا بالضبط ما كان
    مفقوداً سابقاً حسب توثيق rollback_guard.py الأصلي). عند فشل تحميل الحارس
    نفسه، يحفظ مباشرة كما كان يحدث دائماً — لا يتوقف شيء بسبب هذه الإضافة.
    """
    guard = _get_lora_rollback_guard()
    if guard is None:
        lora.save(LORA_SAVE_DIR)
        return

    import glob as _glob
    files = sorted(_glob.glob(f"{LORA_SAVE_DIR}_*"))

    def _eval_fn() -> float:
        if not _LORA_LOSS_WINDOW:
            return 0.0
        import statistics
        # الأعلى = أفضل حسب واجهة CheckpointGuard → نعكس إشارة الخسارة
        return -statistics.mean(_LORA_LOSS_WINDOW[-LORA_SAVE_EVERY:])

    decision = guard.guarded_update(
        files=files,
        update_fn=lambda: lora.save(LORA_SAVE_DIR),
        eval_fn=_eval_fn,
        tolerance=-0.1,  # يسمح بتذبذب طبيعي في الخسارة قبل اعتباره تراجعاً حقيقياً
        label=f"lora_feedback_step_{_LORA_FEEDBACK_COUNT}",
    )
    logger.info(f"[qa_engine] {decision.summary()}")


def _get_lora_adapter():
    """
    يُغلّف ArabicTransformer بـ LoRA adapters (rank=8) لتدريب خفيف جداً
    (~548K باراميتر بدل 27M) من ملاحظات المستخدمين، دون أي تعديل على
    الأوزان الأساسية المدرَّبة. يُرجع None لو الشبكة الأساسية غير متاحة.
    """
    global _LORA_ADAPTER, _LORA_TRIED
    if _LORA_TRIED:
        return _LORA_ADAPTER if _LORA_ADAPTER is not False else None
    _LORA_TRIED = True
    booster = _get_neural_booster()
    if booster is None:
        _LORA_ADAPTER = False
        return None
    try:
        from ai.lora_adapter import LoRATransformerAdapter
        import os
        lora = LoRATransformerAdapter(booster, rank=8, alpha=16.0)
        if os.path.exists(f"{LORA_SAVE_DIR}_meta.json"):
            lora.load(LORA_SAVE_DIR)
            logger.info(f"[qa_engine] LoRA adapter محمَّل من {LORA_SAVE_DIR}")
        _LORA_ADAPTER = lora
    except Exception as e:
        _LORA_ADAPTER = False
        logger.warning(f"[qa_engine] تعذّر تهيئة LoRA adapter: {e}")
    return _LORA_ADAPTER if _LORA_ADAPTER is not False else None


def record_positive_feedback(question: str, answer_summary: str = "", lr: float = 5e-3) -> bool:
    """
    يُسجِّل ملاحظة إيجابية من المستخدم على إجابة معيّنة: خطوة تدريب LoRA
    واحدة صغيرة تُعزِّز نمط السؤال (+الإجابة إن وُجدت) داخل الشبكة، دون
    أي تعديل على الأوزان الأساسية (~27M) — فقط adapters صغيرة (~548K).

    يُستدعى مستقبلاً من واجهة streamlit_app.py عند ضغط المستخدم على 👍.
    يُرجع True لو نجحت خطوة التدريب، False عند أي فشل (fallback آمن —
    لا يكسر أي شيء، فقط لا يحدث تحسين).
    """
    global _LORA_FEEDBACK_COUNT
    lora = _get_lora_adapter()
    if lora is None:
        return False
    try:
        text = (question + " " + answer_summary).strip() if answer_summary else question
        ids, targets = lora.prepare_lm_sample(text)
        if ids is None:
            return False
        loss = lora.train_step(ids, targets, lr=lr)
        _LORA_LOSS_WINDOW.append(float(loss))
        if len(_LORA_LOSS_WINDOW) > LORA_SAVE_EVERY * 3:
            del _LORA_LOSS_WINDOW[: -LORA_SAVE_EVERY * 3]

        _LORA_FEEDBACK_COUNT += 1
        if _LORA_FEEDBACK_COUNT % LORA_SAVE_EVERY == 0:
            _save_lora_with_guard(lora)
            logger.info(f"[qa_engine] LoRA adapter محفوظ (محمي) بعد {_LORA_FEEDBACK_COUNT} ملاحظة")
        return True
    except Exception as e:
        logger.warning(f"[qa_engine] فشل تسجيل ملاحظة LoRA: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# التوليد الحر الاختياري (Yemeni LLM) — طبقة إضافية فوق qa_engine الرمزي
# ═══════════════════════════════════════════════════════════════════════════
# نفس مبدأ _get_neural_booster: qa_engine يعمل بالكامل حتى لو فشلت هذه الطبقة
# تماماً (torch غير مثبّت، لا يوجد checkpoint، أي استثناء أثناء التوليد).
# لا يوجد أي مسار افتراضي يعتمد على نجاح هذه الطبقة — generation_mode=False
# (الافتراضي) لا يلمسها إطلاقاً.

_YEMENI_DECODER = None    # YemeniDecoder المحمَّل، أو False لو فشل نهائياً
_YEMENI_TOKENIZER = None  # YemeniTokenizer المحمَّل، أو False لو فشل نهائياً
_YEMENI_TRIED = False     # نحاول التحميل مرة واحدة فقط


def _get_yemeni_generator():
    """
    يُحمِّل YemeniDecoder + YemeniTokenizer مرة واحدة (lazy singleton).
    يُرجع (decoder, tokenizer) أو (None, None) عند أي فشل (torch غير متاح،
    لا يوجد checkpoint، إلخ). لا يرمي أي استثناء للخارج أبداً.
    """
    global _YEMENI_DECODER, _YEMENI_TOKENIZER, _YEMENI_TRIED
    if _YEMENI_TRIED:
        ok = _YEMENI_DECODER is not False and _YEMENI_TOKENIZER is not False
        return (_YEMENI_DECODER, _YEMENI_TOKENIZER) if ok else (None, None)
    _YEMENI_TRIED = True
    try:
        from ai.arabic_transformer import get_yemeni_decoder
        from ai.yemeni_tokenizer import get_yemeni_tokenizer
        _YEMENI_DECODER = get_yemeni_decoder()
        _YEMENI_TOKENIZER = get_yemeni_tokenizer()
        logger.info("[qa_engine] YemeniDecoder + YemeniTokenizer محمَّلان بنجاح "
                     "(التوليد الحر متاح — قد يكون بأوزان غير مدرَّبة إن لم "
                     "يوجد checkpoint في models/yemeni_decoder)")
    except Exception as e:
        _YEMENI_DECODER = False
        _YEMENI_TOKENIZER = False
        logger.warning(f"[qa_engine] تعذّر تحميل طبقة التوليد الحر — "
                        f"generation_mode سيُعامَل كأنه False: {e}")
    ok = _YEMENI_DECODER is not False and _YEMENI_TOKENIZER is not False
    return (_YEMENI_DECODER, _YEMENI_TOKENIZER) if ok else (None, None)


_LLM_FALLBACK = None        # LLMFallback المحمَّل، أو False لو فشل نهائياً
_LLM_FALLBACK_TRIED = False

_YEMENI_DIALECT_COMMON_RULES = (
    "قواعد صارمة يجب الالتزام بها دون استثناء، بغض النظر عن اللهجة:\n"
    "1. الحقائق المعطاة لك (الملخص، الآيات، المفاهيم، درجة الثقة) هي المصدر "
    "الوحيد المسموح به للإجابة. لا تُضِف معلومة دينية أو تاريخية واحدة غير "
    "موجودة فيها، ولا تحذف تحذيراً أو قيداً ورد فيها.\n"
    "2. أي آية قرآنية تُذكر يجب نقلها حرفياً بلا تغيير ولا لحن ولا حتى تليين "
    "لهجي — النص القرآني وحده يبقى بالفصحى دائماً مهما كانت لهجة بقية الرد.\n"
    "3. مهمتك محصورة في الصياغة الأسلوبية فقط: حوّل الجمل الرسمية لأسلوب "
    "المحادثة الشفهية بلا حشو أو مبالغة أو إطالة غير مبرَّرة.\n"
    "4. إن كانت الحقائق المعطاة غير كافية للإجابة على السؤال، قل ذلك صراحة "
    "بلهجتك بدلاً من الاختلاق أو التخمين.\n"
    "5. لا تذكر كلمة 'CKG' أو 'نظام' أو 'قاعدة بيانات' أو أي مصطلح تقني "
    "داخلي — تحدّث كأنك شخص يعرف الجواب، لا برنامج يسترجعه."
)

_YEMENI_DIALECT_STYLE_HINTS = {
    "صنعانية": (
        "اكتب باللهجة الصنعانية (وسط اليمن): استخدم 'ايش' للاستفهام، "
        "'كدا/كذا' للإشارة، صيغة المضارع بـ'ي' المفتوحة المعتادة في صنعاء "
        "(مثل 'يقول' لا 'يقوّل')، ونبرة هادئة رسمية-ودودة معتدلة، قريبة من "
        "الفصحى في اختيار الألفاظ لكنها محكية في تركيب الجملة."
    ),
    "عدنية": (
        "اكتب باللهجة العدنية (الساحل الجنوبي): جملة أخف وأسرع إيقاعاً، "
        "استخدم 'شنو' أو 'وش' للاستفهام، 'زين' للموافقة/الإقرار، أسلوب "
        "مباشر وعملي بلا إطالة، مع نبرة ودودة عفوية طابعها التبادل التجاري "
        "التاريخي لعدن (مفردات بسيطة وواضحة، لا تكلّف)."
    ),
    "حضرمية": (
        "اكتب باللهجة الحضرمية (حضرموت والمهرة): استخدم 'كيف حالك' الممدودة "
        "والنبرة الهادئة المتأنية المعروفة عن أهل حضرموت، مع ميل لتراكيب "
        "أقرب قليلاً للفصحى من عدن وصنعاء لكن بمفردات محلية مميزة (مثل "
        "'شوي' للقليل)، ونبرة مهذبة تحفظ الوقار في الحديث عن أمور دينية."
    ),
    "عام": (
        "اكتب بلهجة يمنية عامة مفهومة لكل المناطق (لا تحدد صنعانية أو عدنية "
        "أو حضرمية بعينها) — تجنّب المفردات الحادة الخاصة بمنطقة واحدة، "
        "واستخدم أسلوباً محكياً بسيطاً ومهذباً يفهمه أي يمني."
    ),
}


def _build_yemeni_system_prompt(dialect: str = "عام") -> str:
    """
    يبني system prompt مخصصاً للهجة المطلوبة (صنعانية/عدنية/حضرمية/عام)،
    مع الإبقاء على قواعد منع التحريف الدينية/التاريخية مشتركة وصارمة لكل
    الحالات. أي قيمة غير معروفة تُعامَل كـ 'عام' تلقائياً (fallback آمن).
    """
    style = _YEMENI_DIALECT_STYLE_HINTS.get(dialect, _YEMENI_DIALECT_STYLE_HINTS["عام"])
    return (
        "أنت مساعد يصوغ إجابات دينية/معرفية موثوقة بلهجة يمنية طبيعية "
        "لقارئ يمني عادي.\n\n"
        f"أسلوب اللهجة المطلوب:\n{style}\n\n"
        f"{_YEMENI_DIALECT_COMMON_RULES}"
    )


def _get_llm_fallback():
    """يُحمِّل LLMFallback مرة واحدة (lazy singleton). يُرجع None عند أي فشل."""
    global _LLM_FALLBACK, _LLM_FALLBACK_TRIED
    if _LLM_FALLBACK_TRIED:
        return _LLM_FALLBACK if _LLM_FALLBACK is not False else None
    _LLM_FALLBACK_TRIED = True
    try:
        from ai.llm_fallback import LLMFallback
        _LLM_FALLBACK = LLMFallback(max_tokens=500, temperature=0.6)
        logger.info("[qa_engine] LLMFallback محمَّل — صياغة يمنية عبر API متاحة")
    except Exception as e:
        _LLM_FALLBACK = False
        logger.warning(f"[qa_engine] تعذّر تحميل LLMFallback — "
                        f"سيُستخدم المسار الرمزي فقط: {e}")
    return _LLM_FALLBACK if _LLM_FALLBACK is not False else None


def _rephrase_in_yemeni_dialect(
    question: str,
    result: Dict[str, Any],
    dialect: str = "عام",
    arabic_roots: Optional[Dict[str, Any]] = None,
    surah_profiles: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Optional[str]:
    """
    يمرّر الإجابة الرمزية المؤسَّسة (result['summary'] + الآيات + المفاهيم)
    عبر LLMFallback لإعادة صياغتها بلهجة يمنية طبيعية، دون تحريف الحقائق.
    يُرجع None عند أي فشل (fallback آمن كامل — answer_question يتجاهل
    النتيجة ويُبقي result['summary'] الرمزي كما هو دون أي تغيير).

    arabic_roots/surah_profiles اختياريان: إثراء إضافي صادق (جذور لغوية
    لأسماء المفاهيم المطابقة + محاور السور الموضوعية) يُضاف لحقائق
    السياق المرسلة للـLLM — نفس مصدر الحقيقة (CKG)، فقط سياق أعمق.
    """
    llm = _get_llm_fallback()
    if llm is None:
        return None
    try:
        facts_lines = [f"الملخص الرمزي المسترجَع: {result.get('summary', '')}"]
        verses = result.get("verses") or []
        for v in verses[:3]:
            text = v.get("text", "")
            if text:
                facts_lines.append(
                    f"آية — سورة {v.get('surah', '')} آية {v.get('ayah', '')}: {text}"
                )
        related = result.get("related_concepts") or []
        related_names = [
            c.get("name", "") if isinstance(c, dict) else str(c)
            for c in related[:5]
        ]
        if related_names:
            facts_lines.append(f"مفاهيم مرتبطة: {'، '.join(related_names)}")
        facts_lines.append(f"درجة الثقة: {result.get('confidence', 0.0)}")

        primary_names = [
            c.get("name", "") if isinstance(c, dict) else str(c)
            for c in (result.get("primary_concepts") or [])
        ]
        facts_lines.extend(
            _enrich_with_arabic_roots(primary_names + related_names, arabic_roots or {})
        )
        facts_lines.extend(_enrich_with_surah_context(verses, surah_profiles or {}))

        if conversation_history:
            prev = conversation_history[-1]
            prev_q = prev.get("question", "")
            prev_s = prev.get("summary", "")
            if prev_q and prev_s:
                facts_lines.append(
                    f"سياق من سؤال سابق في نفس المحادثة — السؤال: {prev_q} "
                    f"| ملخص الإجابة عليه: {prev_s[:200]}"
                )

        # حقن أمثلة RAG لهجية إن وُجدت في النتيجة (تلقائي من answer_question)
        rag_ctx = (result.get("dialect_rag_context") or "").strip()
        if rag_ctx:
            facts_lines.append(
                "أمثلة لهجية مسترجَعة من مدونة يمنية (للاستئناس بالأسلوب فقط، "
                "لا تختلق حقائق دينية منها):\n" + rag_ctx[:800]
            )

        query = (
            f"السؤال الأصلي: {question}\n\n"
            f"الحقائق المسترجَعة (لا تُغيّرها، فقط أعد صياغتها بلهجة يمنية "
            f"{dialect}):\n" + "\n".join(facts_lines)
        )

        fb_result = llm.generate(query, system_prompt=_build_yemeni_system_prompt(dialect))
        text = (fb_result.text or "").strip()
        return text or None
    except Exception as e:
        logger.warning(f"[qa_engine] فشلت صياغة اللهجة اليمنية عبر LLMFallback — "
                        f"سيُستخدم الملخص الرمزي فقط: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════
# طبقة الأمان (ai/nova_system.py) — كانت غير متصلة إطلاقاً بالمسار الحي
# ═══════════════════════════════════════════════════════════════════════════
_SAFETY_CHECK_FN = None
_SAFETY_CHECK_TRIED = False


def _get_safety_checker():
    """يُحمِّل run_safety_checks مرة واحدة. يُرجع None عند أي فشل (لا يعطّل النظام)."""
    global _SAFETY_CHECK_FN, _SAFETY_CHECK_TRIED
    if _SAFETY_CHECK_TRIED:
        return _SAFETY_CHECK_FN
    _SAFETY_CHECK_TRIED = True
    try:
        from ai.nova_system import run_safety_checks
        _SAFETY_CHECK_FN = run_safety_checks
        logger.info("[qa_engine] طبقة أمان nova_system.py محمَّلة ومتصلة")
    except Exception as e:
        _SAFETY_CHECK_FN = None
        logger.warning(f"[qa_engine] تعذّر تحميل طبقة الأمان — سيستمر النظام بدونها: {e}")
    return _SAFETY_CHECK_FN


# ═══════════════════════════════════════════════════════════════════════════
# تخزين مؤقت (ai/agent_performance_cache.py) — لتسريع الأسئلة المتكررة
# ═══════════════════════════════════════════════════════════════════════════
# يُخزَّن الناتج فقط للمسار الرمزي القطعي (لا توليد حر عشوائي، لا سياق محادثة
# سابقة يغيّر المعنى، لا فحص تأسيس مكلف) — بقية الحالات تمرّ دون كاش دائماً.
_AGENT_CACHE = None
_AGENT_CACHE_TRIED = False


def _get_agent_cache():
    """يُحمِّل AgentCache مرة واحدة. يُرجع None عند أي فشل (لا يعطّل النظام)."""
    global _AGENT_CACHE, _AGENT_CACHE_TRIED
    if _AGENT_CACHE_TRIED:
        return _AGENT_CACHE
    _AGENT_CACHE_TRIED = True
    try:
        from ai.agent_performance_cache import AgentCache
        _AGENT_CACHE = AgentCache(max_size=2000, ttl_seconds=1800)
        logger.info("[qa_engine] طبقة تخزين مؤقت agent_performance_cache.py محمَّلة ومتصلة")
    except Exception as e:
        _AGENT_CACHE = None
        logger.warning(f"[qa_engine] تعذّر تحميل طبقة التخزين المؤقت — سيستمر النظام بدونها: {e}")
    return _AGENT_CACHE


def _cacheable(
    generation_mode: bool,
    include_faithfulness_check: bool,
    conversation_history: Optional[List[Dict[str, str]]],
) -> bool:
    """معايير الأهلية للتخزين المؤقت — أي حالة فيها عشوائية أو سياق أو تحقق
    مكلف تُستبعد كلياً كي لا يُرجَع ناتج قديم أو غير مناسب للسياق الحالي."""
    return (
        not generation_mode
        and not include_faithfulness_check
        and not conversation_history
    )


def _blocked_response(question: str, domain: str, hint: str) -> Dict[str, Any]:
    """رد موحَّد الشكل عند رفض فحص الأمان — بنفس بنية نتيجة answer_question العادية تماماً."""
    return {
        "question": question,
        "summary": hint or "عذراً، لا يمكنني المساعدة في هذا الطلب.",
        "primary_concepts": [],
        "related_concepts": [],
        "verses": [],
        "confidence": 0.0,
        "safety_blocked": True,
        "safety_domain": domain,
        "reasoning_trace": None,
        "images": [],
        "generation_used": False,
        "generated_text": None,
        "generation_backend": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# أثر التفكير (ai/chain_of_thought.py) — "لماذا هذه الإجابة؟"
# ═══════════════════════════════════════════════════════════════════════════
_COT_BUILDER = None
_COT_BUILDER_TRIED = False


def _get_cot_builder(ckg: Dict[str, Any]):
    global _COT_BUILDER, _COT_BUILDER_TRIED
    if _COT_BUILDER_TRIED:
        return _COT_BUILDER if _COT_BUILDER is not False else None
    _COT_BUILDER_TRIED = True
    try:
        from ai.chain_of_thought import ChainOfThoughtBuilder
        _COT_BUILDER = ChainOfThoughtBuilder(ckg=ckg)
        logger.info("[qa_engine] ChainOfThoughtBuilder محمَّل")
    except Exception as e:
        _COT_BUILDER = False
        logger.warning(f"[qa_engine] تعذّر تحميل chain_of_thought — سيُتجاهل reasoning_trace: {e}")
    return _COT_BUILDER if _COT_BUILDER is not False else None


# ═══════════════════════════════════════════════════════════════════════════
# صور توضيحية (ai/image_sources.py) — للواجهة الجديدة
# ═══════════════════════════════════════════════════════════════════════════
def _try_fetch_illustration(concept_name: str) -> list:
    if not concept_name:
        return []
    try:
        from ai.image_sources import search_stock_images
        results = search_stock_images(concept_name, per_page=2)
        return [
            {
                "url": r.url,
                "source": r.source.value if hasattr(r.source, "value") else str(r.source),
            }
            for r in results
        ]
    except Exception as e:
        logger.warning(f"[qa_engine] فشل جلب الصور التوضيحية لـ '{concept_name}': {e}")
        return []


# ═══════════════════════════════════════════════════════════════════════════
# درجة ثقة معزَّزة (ai/self_awareness_deep.py) — تكامل جزئي صريح
# ⚠️ DeepSelfAwareness مصمَّم أصلاً لمعايرة قرارات توجيه رقمية (feature_vec +
# predicted_target ضمن NeuralCore)، وليس لتقييم إجابات نصية. التكامل هنا
# "محوَّل" (adapter): نبني feature_vec تقريبياً من إشارات qa_engine المتاحة
# فعلاً، ونستخدم before_decision() فقط كرأي ثانٍ يُمزَج مع الثقة الرمزية
# الأصلية — لا يستبدلها. لا توجد حلقة تغذية راجعة (after_decision) لأن
# answer_question لا يملك "الإجابة الصحيحة الفعلية" ليقارن بها، لذلك وحدة
# WeaknessDetector/التعلّم بمرور الوقت في الملف الأصلي غير مُستخدَمة هنا —
# فقط طبقة ConfidenceEstimator الآنية.
# ═══════════════════════════════════════════════════════════════════════════
_DEEP_AWARENESS = None
_DEEP_AWARENESS_TRIED = False


def _get_deep_awareness():
    global _DEEP_AWARENESS, _DEEP_AWARENESS_TRIED
    if _DEEP_AWARENESS_TRIED:
        return _DEEP_AWARENESS if _DEEP_AWARENESS is not False else None
    _DEEP_AWARENESS_TRIED = True
    try:
        from ai.self_awareness_deep import DeepSelfAwareness
        _DEEP_AWARENESS = DeepSelfAwareness()
        logger.info("[qa_engine] DeepSelfAwareness محمَّل (adapter جزئي لدرجة الثقة)")
    except Exception as e:
        _DEEP_AWARENESS = False
        logger.warning(f"[qa_engine] تعذّر تحميل self_awareness_deep — الثقة تبقى كما هي: {e}")
    return _DEEP_AWARENESS if _DEEP_AWARENESS is not False else None


def _refine_confidence(result: Dict[str, Any]) -> float:
    """يمزج الثقة الرمزية الأصلية مع رأي ConfidenceEstimator (متوسط بسيط)."""
    base_confidence = float(result.get("confidence", 0.0))
    awareness = _get_deep_awareness()
    if awareness is None:
        return base_confidence
    try:
        n_concepts = len(result.get("primary_concepts", []))
        n_related  = len(result.get("related_concepts", []))
        n_verses   = len(result.get("verses", []))
        top_match  = (result.get("primary_concepts") or [{}])[0].get("match", 0.0)

        feature_vec = [
            min(n_concepts / 5.0, 1.0),
            min(n_related / 8.0, 1.0),
            min(n_verses / 5.0, 1.0),
            min(float(top_match), 1.0),
        ]
        conf, _details = awareness.confidence.estimate(feature_vec, base_confidence)
        return round((base_confidence + conf) / 2.0, 4)
    except Exception as e:
        logger.warning(f"[qa_engine] فشل تحسين درجة الثقة عبر self_awareness_deep: {e}")
        return base_confidence


# ═══════════════════════════════════════════════════════════════════════════
# فحص تأسيس الإجابة على المصدر (ai/nsm_answer_verifier.py) — DeepEval
# FaithfulnessMetric + حَكَم مجاني بالكامل عبر ai/free_router.py (Groq/
# Gemini/Cloudflare). استيراد كسول + تدهور آمن كامل، بنفس نمط
# _get_deep_awareness أعلاه: أي فشل (لا deepeval، لا مفتاح مجاني، مشكلة
# شبكة) يُرجع {"available": False, ...} بدل رمي استثناء يوقف
# answer_question() كاملة.
# ═══════════════════════════════════════════════════════════════════════════
def _check_answer_faithfulness(question: str, result: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from ai.nsm_answer_verifier import verify_answer_faithfulness
        return verify_answer_faithfulness(question, result)
    except Exception as e:
        logger.warning(f"[qa_engine] تعذّر تحميل ai.nsm_answer_verifier: {e}")
        return {
            "available": False, "faithful": None, "score": None,
            "reason": f"تعذّر تحميل وحدة التحقق: {e}",
        }


def _enrich_with_arabic_roots(
    names: List[str],
    arabic_roots: Dict[str, Any],
    max_roots: int = 3,
) -> List[str]:
    """
    يربط أسماء مفاهيم مطابقة بجذورها العربية المحفوظة في CKG
    (knowledge/cognitive_graph.json → arabic_roots: {root: {frequency,
    top_token}}). الربط يتم عبر top_token (الصيغة الأشهر لكل جذر) لأن
    أسماء المفاهيم غالباً صيغ مشتقة كاملة وليست جذوراً مجرَّدة — التقاطع
    المباشر بالجذر نفسه ضعيف (70 من 7,338)، بينما عبر top_token أغنى
    بكثير (335) وصادق (لا نخترع جذوراً، فقط نستخدم ما هو محفوظ فعلياً).
    """
    if not arabic_roots or not names:
        return []
    # فهرس عكسي: top_token → (root, frequency) — يُبنى مرة لكل استدعاء
    # (839 عنصر فقط، رخيص، لا يستحق تخزيناً مؤقتاً عبر الاستدعاءات)
    by_top_token: Dict[str, Any] = {}
    for root, info in arabic_roots.items():
        tt = info.get("top_token") if isinstance(info, dict) else None
        if tt:
            by_top_token[tt] = (root, info.get("frequency", 0))

    lines: List[str] = []
    for name in names:
        hit = by_top_token.get(name)
        if hit:
            root, freq = hit
            lines.append(f"الجذر اللغوي لـ«{name}»: «{root}» (يتكرر {freq} مرة في القرآن)")
        if len(lines) >= max_roots:
            break
    return lines


def _enrich_with_surah_context(
    verses: List[Dict[str, Any]],
    surah_profiles: Dict[str, Any],
    max_surahs: int = 2,
    max_concepts_per_surah: int = 3,
) -> List[str]:
    """
    يضيف السياق الموضوعي لكل سورة ظهرت في الآيات المسترجَعة، من
    surah_profiles المحفوظة في CKG (رقم السورة → أهم مفاهيمها بالوزن).
    يساعد الـLLM يفهم الإطار الموضوعي العام للسورة، لا فقط نص الآية
    المفردة.
    """
    if not surah_profiles or not verses:
        return []
    seen_surahs: List[str] = []
    for v in verses:
        s = str(v.get("surah", "")).strip()
        if s and s not in seen_surahs:
            seen_surahs.append(s)
        if len(seen_surahs) >= max_surahs:
            break

    lines: List[str] = []
    for s in seen_surahs:
        profile = surah_profiles.get(s)
        if not profile:
            continue
        top = sorted(profile, key=lambda p: p.get("weight", 0), reverse=True)[:max_concepts_per_surah]
        names = "، ".join(p.get("concept", "") for p in top if p.get("concept"))
        if names:
            lines.append(f"المحاور الموضوعية الرئيسية لسورة رقم {s}: {names}")
    return lines


def _build_grounding_text(
    concept_matches: List[Tuple[str, float]],
    verses: List[Dict[str, Any]],
    max_concepts: int = 5,
    max_verses: int = 3,
    arabic_roots: Optional[Dict[str, Any]] = None,
    surah_profiles: Optional[Dict[str, Any]] = None,
) -> str:
    """
    يبني نص سياق تأسيسي (grounding) من نفس المعطيات الرمزية التي يحسبها
    answer_question أصلاً (مفاهيم مطابقة + آيات داعمة) — لا يستدعي أي
    مصدر خارجي جديد، فقط يعيد صياغتها كنص متصل يُغذّى للـ decoder.

    arabic_roots/surah_profiles اختياريان: لو مُمرَّران (من ckg الخام)
    يُضاف إثراء إضافي صادق (جذور لغوية + محاور موضوعية للسور) — عند
    غيابهما يبقى السلوك مطابقاً تماماً للسابق.
    """
    parts: List[str] = []
    top_names = [c for c, _ in concept_matches[:max_concepts]]
    if top_names:
        parts.append(f"المفاهيم ذات الصلة: {'، '.join(top_names)}.")
    if verses:
        for v in verses[:max_verses]:
            text = v.get("text", "")
            if text:
                parts.append(f"سورة {v.get('surah', '')} آية {v.get('ayah', '')}: {text}")
    parts.extend(_enrich_with_arabic_roots(top_names, arabic_roots or {}))
    parts.extend(_enrich_with_surah_context(verses, surah_profiles or {}))
    return " ".join(parts)


def _try_generate_free_text(
    question: str,
    concept_matches: List[Tuple[str, float]],
    verses: List[Dict[str, Any]],
    temperature: float,
    top_p: float,
    top_k: int,
    arabic_roots: Optional[Dict[str, Any]] = None,
    surah_profiles: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """
    يحاول توليد نص حر عبر YemeniDecoder، مؤسَّس (grounded) على المفاهيم
    والآيات المسترجَعة رمزياً. يُرجع None عند أي فشل (fallback آمن كامل —
    answer_question يتجاهل النتيجة ويكمل بمساره الرمزي الافتراضي).
    """
    decoder, tokenizer = _get_yemeni_generator()
    if decoder is None or tokenizer is None:
        return None
    try:
        import torch

        grounding_text = _build_grounding_text(
            concept_matches, verses,
            arabic_roots=arabic_roots, surah_profiles=surah_profiles,
        )
        grounding_ids = tokenizer.encode(grounding_text, add_bos=False, add_eos=False) \
            if grounding_text else None
        prompt_ids = tokenizer.encode(question, add_bos=True, add_eos=False)

        grounding_tensor = (
            torch.tensor(grounding_ids, dtype=torch.int64).unsqueeze(0)
            if grounding_ids is not None and len(grounding_ids) > 0 else None
        )
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.int64).unsqueeze(0)

        with torch.no_grad():
            out_ids = decoder.generate(
                prompt_ids=prompt_tensor,
                grounding_ids=grounding_tensor,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )

        generated_ids = out_ids[0].tolist()
        return tokenizer.decode(generated_ids, skip_special=True, stop_at_eos=True)
    except Exception as e:
        logger.warning(f"[qa_engine] فشل التوليد الحر (Yemeni LLM) — "
                        f"سيُستخدم المسار الرمزي فقط: {e}")
        return None


def _apply_neural_boost(
    question: str, related_concepts: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    يعزّز ترتيب related_concepts دلالياً عبر مقارنة token IDs من WordTokenizer
    (أو HashTokenizer إن فُعّل يدوياً). عند أي فشل: يُرجع القائمة كما هي.
    """
    if not related_concepts:
        return related_concepts
    booster = _get_neural_booster()
    if booster is None:
        return related_concepts
    try:
        top = booster.predict_next(question, top_k=15)
        top_ids = [i for i, _ in top]
        top_id_set = set(top_ids)
        n = len(top_ids)

        def boost_of(name: str) -> float:
            tok = booster.tokenizer
            if hasattr(tok, "content_ids"):
                ids = tok.content_ids(name, 3)
            else:
                ids = tok.encode(name, 8)
                special = {getattr(tok, "PAD", 0), getattr(tok, "BOS", 2),
                           getattr(tok, "EOS", 3), getattr(tok, "SEP", 4),
                           getattr(tok, "MASK", 5)}
                ids = [int(x) for x in ids if int(x) not in special]
            if len(ids) == 0:
                return 0.0
            first_id = int(ids[0])
            if first_id not in top_id_set:
                return 0.0
            rank = top_ids.index(first_id)
            return 1.0 - (rank / n)

        for r in related_concepts:
            r["neural_boost"] = round(boost_of(r["concept"]), 4)

        # إعادة الترتيب: الوزن الأصلي يبقى العامل الأساسي، والتعزيز
        # العصبي يُستخدم فقط لكسر التعادل/تحسين طفيف (وزن صغير 0.1)
        related_concepts.sort(
            key=lambda r: (r.get("weight", 0.0) * (1.0 + 0.1 * r.get("neural_boost", 0.0))),
            reverse=True,
        )
        return related_concepts
    except Exception as e:
        logger.warning(f"[qa_engine] فشل التعزيز العصبي لهذا السؤال — "
                        f"استخدام الترتيب الأصلي: {e}")
        return related_concepts


# ═══════════════════════════════════════════════════════════════════════════
# تطبيع النص العربي (نفس منطق streamlit_app.py)
# ═══════════════════════════════════════════════════════════════════════════
_TASHKEEL  = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_ALEF      = re.compile(r'[أإآٱ]')
_BOM       = re.compile(r'\ufeff')
_SPACES    = re.compile(r'\s+')


def normalize_arabic(text: str) -> str:
    text = _TASHKEEL.sub('', text)
    text = _ALEF.sub('ا', text)
    text = _BOM.sub('', text)
    text = _SPACES.sub(' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# كلمات أداة / وقف عربية — تُستثنى من استخراج المفاهيم من السؤال
# ═══════════════════════════════════════════════════════════════════════════
ARABIC_STOPWORDS = {
    "ما", "ماذا", "من", "هل", "كيف", "لماذا", "متى", "اين", "أين",
    "في", "على", "عن", "الى", "إلى", "مع", "هو", "هي", "هم", "نحن",
    "انت", "أنت", "انتم", "أنتم", "كان", "يكون", "قال", "يقول",
    "الذي", "التي", "الذين", "هذا", "هذه", "ذلك", "تلك",
    "لا", "لم", "لن", "قد", "كل", "بل", "أم", "او", "أو", "ثم",
    "اذا", "إذا", "حتى", "كما", "لكن", "وإن", "وان", "بين",
    "علاقة", "علاقه", "يقول", "تقول", "نص", "آية", "ايه", "ايات", "آيات",
    "القران", "القرآن", "الكريم", "حول", "بخصوص", "بشأن", "بشان",
    "موضوع", "معنى", "تفسير", "شرح", "وضح", "اشرح", "بين", "وضّح",
}

# اختصار: كلمات سؤال شائعة + حروف عطف نزيلها من بدايات/نهايات الكلمات
PREFIX_STRIP = ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك"]

# مفاهيم "إطارية" تُستخدم غالباً في صياغة السؤال نفسه ولا تمثل موضوعه
# (مثل: "ماذا يقول القرآن عن X؟") — تُخفَّض أولويتها إذا وُجد مفهوم آخر معها
META_CONCEPTS = {"قرآن", "كتاب", "وحي", "أنبياء", "رسالة"}

# عبارات استفهامية تدل على أن السؤال عن "كيان" (شخص/أمة/شخصية)
# وليس عن مفهوم عام — تُستخدم لتفعيل طبقة الكيانات المعرفية
ENTITY_QUESTION_PATTERNS = [
    "من هو", "من هي", "من هم",
    "ما هو", "ما هي",
    "من ", "ما قصة", "حدثني عن", "تحدث عن", "اخبرني عن", "أخبرني عن",
    "عرفني ب", "عرّفني ب",
]


def _strip_prefixes(word: str) -> str:
    """إزالة أل التعريف وحروف الجر/العطف الشائعة من بداية الكلمة."""
    for p in PREFIX_STRIP:
        if word.startswith(p) and len(word) - len(p) >= 2:
            return word[len(p):]
    return word


# ═══════════════════════════════════════════════════════════════════════════
# 1) استخراج المفاهيم من السؤال
# ═══════════════════════════════════════════════════════════════════════════
def extract_concepts_from_question(question: str, concepts_db: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    يحلل سؤالاً بالعربية ويستخرج المفاهيم الموجودة في الـ CKG التي تطابقه.
    يعيد قائمة (اسم المفهوم، درجة التطابق) مرتبة تنازلياً.
    """
    q_norm = normalize_arabic(question)

    # تقسيم السؤال إلى كلمات وتنظيفها من علامات الترقيم
    raw_words = re.split(r'[\s\u060C\u061F\u061B,.!?؟،؛]+', q_norm)
    words = [w for w in raw_words if w and w not in ARABIC_STOPWORDS]

    matches: Dict[str, float] = {}

    # حد أدنى لطول المفهوم لاعتباره صالحاً لمطابقة "تطابق جزئي" (substring)
    # يمنع مفاهيم قصيرة جداً (مثل "بر") من المطابقة الخاطئة داخل كلمات أطول
    MIN_LEN_FOR_SUBSTRING = 3

    # تجهيز نسخ مطبّعة من كل اسم مفهوم (مع وبدون إزالة السوابق)
    for cname, cdata in concepts_db.items():
        c_norm = normalize_arabic(cname)
        c_len  = len(c_norm)

        score = 0.0

        # (أ) تطابق المفهوم كاملاً كسلسلة فرعية من السؤال (لمفاهيم >= 3 حروف فقط)
        if c_len >= MIN_LEN_FOR_SUBSTRING and c_norm in q_norm:
            score = max(score, 1.0)

        # (ب) تطابق على مستوى الكلمات المفردة
        for w in words:
            w_clean = _strip_prefixes(w)
            if not w_clean:
                continue

            if w == c_norm or w_clean == c_norm:
                score = max(score, 1.0)
            elif c_len >= MIN_LEN_FOR_SUBSTRING and (c_norm in w or c_norm in w_clean):
                score = max(score, 0.85)
            elif (
                c_len >= MIN_LEN_FOR_SUBSTRING
                and (w_clean in c_norm or w in c_norm)
                and len(w_clean) >= MIN_LEN_FOR_SUBSTRING
                and len(w_clean) / c_len >= 0.7  # الكلمة تغطي معظم اسم المفهوم (يمنع تطابق "ايمان" مع "ايمان زواج")
            ):
                score = max(score, 0.7)
            elif c_len >= MIN_LEN_FOR_SUBSTRING and " " in c_norm:
                # (ج) مفاهيم مركّبة (تحتوي مسافة، مثل "خمر ومسكرات"):
                # نطابق على مستوى كل كلمة من كلمات المفهوم على حدة
                concept_words = [cw for cw in c_norm.split(" ") if len(cw) >= 3]
                for cw in concept_words:
                    if w_clean == cw or w == cw:
                        score = max(score, 0.8)
                    elif len(w_clean) >= 3 and cw[:3] == w_clean[:3] and abs(len(cw) - len(w_clean)) <= 2:
                        score = max(score, 0.4)
            else:
                # تشابه جذري بسيط: أول 3 حروف متطابقة (لمفاهيم وكلمات >=3 حروف)
                # بشرط أن يكون طول الكلمة والمفهوم متقاربين (يمنع تطابق كلمة قصيرة
                # مع بداية مفهوم مركّب أطول بكثير، مثل "ايمان" مع "ايمان زواج")
                if (
                    len(w_clean) >= 3 and c_len >= 3
                    and w_clean[:3] == c_norm[:3]
                    and abs(len(w_clean) - c_len) <= 2
                ):
                    score = max(score, 0.4)

        if score > 0:
            matches[cname] = score

    # ── خفض أولوية "المفاهيم الإطارية" إن وُجد مفهوم آخر غير إطاري معها ──
    non_meta = [c for c in matches if c not in META_CONCEPTS]
    if non_meta:
        for c in list(matches.keys()):
            if c in META_CONCEPTS:
                matches[c] *= 0.3

    # ── ترتيب أفضل قبل إرسال المفاهيم للـLLM/العلاقات ───────────────────
    # تعزيز تماسك موضوعي (cluster coherence): عند تعادل درجات المطابقة
    # النصية (شائع هنا لأن score قيم منفصلة: 1.0/0.85/0.8/0.7/0.4)، نفضّل
    # مفهوماً يشارك cluster أعلى مطابقة — يقلّل ضجيج مطابقات حرفية عرضية
    # (مثل تشابه أول 3 حروف بشرط تقارب الطول) لا علاقة موضوعية حقيقية
    # لها بمحور السؤال.
    #
    # ملاحظة: جُرِّب أيضاً استبدال frequency الخام بـ strength (log-
    # normalized) كعامل ترجيح ثانٍ، لكن اختبار حقيقي على بيانات CKG كشف
    # أن strength تتشبّع قرب 1.0 لمعظم المفاهيم متوسطة-عالية التكرار (لأنها
    # مُطبَّعة نسبةً لأعلى تكرار في كامل الـCKG، وهو ضخم جداً) — فتفقد
    # الدقة بالضبط حيث نحتاجها، وأنتجت ترتيباً أسوأ فعلياً في حالة حقيقية
    # (سؤال عن "الإيمان": رفعت "بالله"/"إلآ" فوق "إيمان" نفسها). لذا أُبقي
    # frequency الخام كعامل الترجيح الثاني كما كان — مُثبَت الصحة.
    if matches:
        top_name = max(matches, key=lambda c: matches[c])
        top_cluster = concepts_db.get(top_name, {}).get("cluster")
    else:
        top_cluster = None

    def _rank_key(name: str):
        info = concepts_db.get(name, {})
        coherence_bonus = (
            0.05 if top_cluster and info.get("cluster") == top_cluster else 0.0
        )
        return (-(matches[name] + coherence_bonus), -info.get("frequency", 0))

    sorted_names = sorted(matches.keys(), key=_rank_key)
    sorted_matches = [(name, matches[name]) for name in sorted_names]
    return sorted_matches


# ═══════════════════════════════════════════════════════════════════════════
# 1.5) كشف أسئلة "الكيانات" (من هو/ما هي...) — لتفعيل طبقة الكيانات المعرفية
# ═══════════════════════════════════════════════════════════════════════════
def is_entity_question(question: str) -> bool:
    """
    يحدد إن كان السؤال يسأل عن "كيان" (شخص، نبي، أمة، شخصية)
    بدلاً من سؤال عام عن مفهوم أو علاقة.
    """
    q_norm = normalize_arabic(question)
    return any(q_norm.startswith(p) or f" {p}" in q_norm for p in ENTITY_QUESTION_PATTERNS)


def find_entity_match(
    concept_matches: List[Tuple[str, float]],
    entities_db: Dict[str, Any],
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    يبحث بين المفاهيم المستخرجة من السؤال عن أول مفهوم
    يملك إدخالاً في طبقة الكيانات المعرفية (entities.json).
    يعيد (اسم الكيان، بيانات الكيان) أو None.
    """
    for cname, _score in concept_matches:
        if cname in entities_db:
            return cname, entities_db[cname]
    return None


# ═══════════════════════════════════════════════════════════════════════════
# 2) إيجاد المفاهيم المرتبطة عبر العلاقات في CKG
# ═══════════════════════════════════════════════════════════════════════════
# الحد الأدنى لتكرار المفهوم المرتبط في القرآن ليُعتبر "ذا دلالة كافية"
# لعرضه في قائمة المفاهيم المرتبطة. مفاهيم نادرة جداً (مثل "سخرية" بـ4 تكرارات)
# قد تحصل على "weight" مرتفع كاذب رياضياً (count / min(freq)) لمجرد أن
# عدد ظهوراتها القليل تزامن مع مفهوم شائع — فنستثني هذه الحالات هنا.
MIN_RELATED_FREQUENCY = 8


def _relation_rank_score(weight: float, count: int, other_freq: int) -> float:
    """
    يحسب درجة ترتيب أكثر توازناً من "weight" الخام المخزّن في CKG.

    weight الخام = count / min(freq_a, freq_b) → ينحاز للمفاهيم النادرة
    (قاسم صغير يرفع النسبة حتى مع عدد تزامن قليل جداً).

    الدرجة الجديدة تأخذ في الحسبان أيضاً:
      - عدد مرات التزامن الفعلي (count) — أدلة أكثر = أوثق
      - تكرار المفهوم الآخر في القرآن (other_freq) — مفهوم له حضور
        حقيقي في النص، لا مجرد ذكر عابر
    """
    import math
    count_factor = math.log(count + 1) if count > 0 else 0.3  # علاقات بلا evidence (semantic/narrative) تحصل على عامل ثابت معتدل
    freq_factor  = math.log(other_freq + 1)
    return weight * count_factor * freq_factor


def find_related_concepts(
    primary_concepts: List[str],
    relations_db: Dict[str, Any],
    concepts_db: Optional[Dict[str, Any]] = None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    يبحث في جدول العلاقات عن المفاهيم المرتبطة بالمفاهيم الأساسية
    المستخرجة من السؤال، مع ترتيب متوازن يتجنب طغيان المفاهيم
    النادرة جداً ذات "weight" مرتفع كاذب رياضياً (انظر _relation_rank_score).
    """
    concepts_db = concepts_db or {}
    primary_norm = {normalize_arabic(c) for c in primary_concepts}
    related: Dict[str, Dict[str, Any]] = {}

    for rel_key, rel in relations_db.items():
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        src_n, tgt_n = normalize_arabic(src), normalize_arabic(tgt)

        other = None
        if src_n in primary_norm and tgt_n not in primary_norm:
            other = tgt
        elif tgt_n in primary_norm and src_n not in primary_norm:
            other = src

        if other is None:
            continue

        # استثناء أسماء الجذور (root:...) من قائمة "المفاهيم المرتبطة"
        # المعروضة للمستخدم — تبقى متاحة عبر علاقات root_link لأغراض أخرى
        if other.startswith("root:"):
            continue

        other_freq = concepts_db.get(other, {}).get("frequency", 0)
        # استثناء المفاهيم النادرة جداً (دلالة ضعيفة إحصائياً)
        if concepts_db and other_freq < MIN_RELATED_FREQUENCY:
            continue

        weight = rel.get("weight", 0.0)
        count  = rel.get("count", 0)
        rtype  = rel.get("relation_type", "")
        score  = _relation_rank_score(weight, count, other_freq)

        # إن وجد المفهوم بأكثر من علاقة، نحتفظ بالأعلى بحسب الدرجة المتوازنة
        existing = related.get(other)
        if existing is None or score > existing["_score"]:
            related[other] = {
                "concept":       other,
                "weight":        weight,
                "relation_type": rtype,
                "evidence":      rel.get("evidence", []),
                "_score":        score,
            }

    ranked = sorted(related.values(), key=lambda x: -x["_score"])

    # ── ضمان تنوّع أنواع العلاقات في النتائج النهائية ──────────────────
    # الترتيب الخام قد يُهيمن عليه نوع واحد (عادة co_occurrence ذو evidence
    # كثيرة)، فتغيب علاقات semantic/narrative_sequence القيّمة موضوعياً
    # حتى لو كانت أقل توثيقاً إحصائياً. نضمن ظهور أعلى نتيجة من كل نوع
    # متاح أولاً، ثم نكمل الباقي بالترتيب العام.
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    for r in ranked:
        by_type.setdefault(r["relation_type"], []).append(r)

    diversified: List[Dict[str, Any]] = []
    seen_concepts = set()

    # الجولة الأولى: أفضل نتيجة من كل نوع علاقة (بترتيب ظهور الأنواع
    # في القائمة الأصلية، أي الأنواع الأقوى عموماً أولاً)
    type_order = list(dict.fromkeys(r["relation_type"] for r in ranked))
    for rtype in type_order:
        candidates = by_type.get(rtype, [])
        if candidates and candidates[0]["concept"] not in seen_concepts:
            diversified.append(candidates[0])
            seen_concepts.add(candidates[0]["concept"])
        if len(diversified) >= top_k:
            break

    # الجولة الثانية: إكمال الباقي بالترتيب العام حتى الوصول لـ top_k
    if len(diversified) < top_k:
        for r in ranked:
            if r["concept"] in seen_concepts:
                continue
            diversified.append(r)
            seen_concepts.add(r["concept"])
            if len(diversified) >= top_k:
                break

    # إزالة الحقل الداخلي _score قبل الإعادة
    for r in diversified:
        r.pop("_score", None)
    return diversified[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# 3) استرجاع الآيات الداعمة
# ═══════════════════════════════════════════════════════════════════════════
def _ref_to_surah_ayah(ref: str) -> Tuple[int, int]:
    """يحوّل مرجعاً مثل 'quran:2:153' إلى (سورة، آية)."""
    try:
        parts = ref.split(":")
        return int(parts[-2]), int(parts[-1])
    except Exception:
        return (0, 0)


def retrieve_supporting_verses(
    concept_matches: List[Tuple[str, float]],
    concepts_db: Dict[str, Any],
    ayat_by_ref: Dict[Tuple[int, int], Dict[str, Any]],
    max_verses: int = 5,
) -> List[Dict[str, Any]]:
    """
    يجمع الآيات الداعمة من حقل sources لكل مفهوم مطابق،
    مع ترتيبها بحسب قوة تطابق المفهوم.
    """
    seen_refs = set()
    verses: List[Dict[str, Any]] = []

    for cname, score in concept_matches:
        cdata = concepts_db.get(cname, {})
        sources = cdata.get("sources", [])
        for ref in sources:
            sa = _ref_to_surah_ayah(ref)
            if sa in seen_refs or sa == (0, 0):
                continue
            ayah_data = ayat_by_ref.get(sa)
            if not ayah_data:
                continue
            seen_refs.add(sa)
            verses.append({
                "surah":   sa[0],
                "ayah":    sa[1],
                "text":    ayah_data.get("text", ""),
                "concept": cname,
                "score":   score,
            })
            if len(verses) >= max_verses:
                return verses
    return verses


# ═══════════════════════════════════════════════════════════════════════════
# 4) درجة الثقة
# ═══════════════════════════════════════════════════════════════════════════
def compute_confidence(
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
) -> float:
    """
    درجة ثقة مبنية على:
      - وجود مفاهيم مباشرة مطابقة (40%)
      - وجود علاقات دلالية مستنتجة (25%)
      - وجود آيات داعمة (35%)
    """
    confidence = 0.0

    if concept_matches:
        best_score = max(s for _, s in concept_matches)
        confidence += 0.40 * best_score

    if related_concepts:
        confidence += 0.25 * min(len(related_concepts) / 5, 1.0)

    if verses:
        confidence += 0.35 * min(len(verses) / 5, 1.0)

    return round(min(confidence, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════
# 5) توليد إجابة منظمة
# ═══════════════════════════════════════════════════════════════════════════
# ── عبارات افتتاحية طبيعية بحسب المجموعة المعرفية للمفهوم الأساسي ──
CLUSTER_OPENERS = {
    "توحيد":   "في باب التوحيد وأسماء الله وصفاته، يتحدث القرآن الكريم عن",
    "عبادة":   "في باب العبادات، يبيّن القرآن الكريم أحكام ومعاني",
    "أخلاق":   "من القيم الأخلاقية التي يدعو إليها القرآن الكريم",
    "إيمان":   "في باب العقيدة والإيمان، يتناول القرآن الكريم",
    "آخرة":    "في وصف الدار الآخرة، يذكر القرآن الكريم",
    "نبوة":    "في سياق قصص الأنبياء والرسالات، يذكر القرآن الكريم",
    "معرفة":   "في مجال العلم والمعرفة، يوجّه القرآن الكريم إلى",
    "مجتمع":   "في تنظيم شؤون المجتمع، يبيّن القرآن الكريم أحكام",
    "كون":     "من آيات الله في الكون، يذكر القرآن الكريم",
    "قصص":     "من القصص القرآني، يروي القرآن الكريم خبر",
    "روح":     "في تزكية النفس وأحوالها، يتحدث القرآن الكريم عن",
    "حكم":     "في باب الأحكام والقضاء، يبيّن القرآن الكريم",
    "اقتصاد":  "في باب المعاملات المالية، ينظّم القرآن الكريم أحكام",
    "سلوك":    "من السلوكيات التي يحذّر القرآن الكريم منها أو يدعو إليها",
    "فقه":     "في الأحكام الفقهية، يبيّن القرآن الكريم حكم",
    "باطن":    "في أعمال القلوب والإيمان الباطن، يتحدث القرآن الكريم عن",
}

# علاقات لها صياغة طبيعية خاصة عند ذكرها
RELATION_PHRASES = {
    "co_occurrence":     "وترد هذه الفكرة مرتبطة في آيات عديدة بمفهوم",
    "semantic":          "وهي مرتبطة من حيث المعنى بمفهوم",
    "thematic_cluster":  "وتتكرر هذه الفكرة جنباً إلى جنب مع مفهوم",
    "root_link":         "ويتصل لفظياً بجذر",
    "narrative_sequence": "وترتبط في السياق القصصي بـ",
    "episodic_rule":     "وقد لوحظ تكرار ربطها بمفهوم",
}


def generate_entity_answer(
    question: str,
    entity_name: str,
    entity_data: Dict[str, Any],
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    concepts_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    يبني إجابة مباشرة لسؤال عن "كيان" (من هو/ما هي...) باستخدام
    طبقة الكيانات المعرفية (entities.json):

      كيان → وصف وصفات → مفاهيم مرتبطة → آيات داعمة

    بدلاً من البدء بالعلاقات الإحصائية فقط.
    """
    confidence = compute_confidence(concept_matches, related_concepts, verses)
    # سؤال كيان واضح + وجود وصف جاهز → ثقة عالية بطبيعتها
    confidence = max(confidence, 0.9)

    summary_parts = [entity_data.get("summary", "").strip()]

    attributes = entity_data.get("attributes", [])
    if attributes:
        attrs_joined = "، ".join(attributes)
        possessive = "صفاتها" if entity_data.get("gender") == "f" else "صفاته"
        summary_parts.append(f"من {possessive}: {attrs_joined}.")

    # ── المفاهيم المرتبطة: نُفضّل المرتبطة المعرّفة يدوياً في entities.json ──
    entity_related_names = entity_data.get("related_concepts", [])
    extra_related = []
    seen_related_names = {r["concept"] for r in related_concepts}
    for rname in entity_related_names:
        # تطابق مرن مع أسماء CKG (قد تختلف في التطبيع، مثل "ابرهيم" vs "إبراهيم")
        match = None
        if rname in concepts_db:
            match = rname
        else:
            rn = normalize_arabic(rname)
            match = next((c for c in concepts_db if normalize_arabic(c) == rn), None)
        if match and match not in seen_related_names and match != entity_name:
            extra_related.append({
                "concept":       match,
                "weight":        0.9,
                "relation_type": "entity_attribute",
                "evidence":      [],
            })
            seen_related_names.add(match)

    combined_related = extra_related + related_concepts

    if combined_related:
        names = "، ".join(f"«{r['concept']}»" for r in combined_related[:5])
        summary_parts.append(f"وترتبط هذه الشخصية بمفاهيم: {names}.")

    # ── ذكر الآيات بصياغة استشهادية ──
    if verses:
        refs = "، ".join(f"({v['surah']}:{v['ayah']})" for v in verses[:3])
        extra = f"، وغيرها من {len(verses)} آية" if len(verses) > 3 else ""
        summary_parts.append(f"ومن الآيات الدالة على ذلك: {refs}{extra}.")

    summary = " ".join(p for p in summary_parts if p)

    # ── تفاصيل المفاهيم الأساسية (تبقى كما هي للعرض في الواجهة) ──
    primary_details = []
    for cname, score in concept_matches[:5]:
        cdata = concepts_db.get(cname, {})
        primary_details.append({
            "name":      cname,
            "cluster":   cdata.get("cluster", "غير مصنّف"),
            "frequency": cdata.get("frequency", 0),
            "match":     score,
        })

    return {
        "question":         question,
        "summary":          summary,
        "primary_concepts": primary_details,
        "related_concepts": combined_related,
        "verses":           verses,
        "confidence":       round(min(confidence, 1.0), 4),
        "entity": {
            "name": entity_name,
            "type": entity_data.get("type", ""),
        },
    }


def generate_answer(
    question: str,
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    concepts_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    يبني إجابة منظمة بصياغة طبيعية احترافية (ملخص + مفاهيم مرتبطة +
    آيات داعمة + درجة ثقة) اعتماداً فقط على بيانات الـ CKG والقرآن الموجودة.

    الصياغة تتجنب الإشارة إلى "النظام" أو "CKG" أو الأرقام الداخلية،
    وتقدّم الإجابة كشرح معرفي مباشر يستشهد بالآيات كدليل.
    """
    confidence = compute_confidence(concept_matches, related_concepts, verses)

    if not concept_matches:
        return {
            "question":         question,
            "summary":          "لم يتم العثور على مفهوم واضح يطابق هذا السؤال في قاعدة المعرفة الحالية. "
                                 "حاول إعادة صياغة السؤال باستخدام مصطلح قرآني أوضح (مثل: الصبر، العدل، التوحيد، الصلاة).",
            "primary_concepts": [],
            "related_concepts": [],
            "verses":           [],
            "confidence":       0.0,
        }

    # ── المفاهيم الأساسية المكتشفة ──
    primary_names = [c for c, _ in concept_matches[:3]]

    # استبعاد المفاهيم "الإطارية" (مثل: قرآن، كتاب) من صياغة الملخص
    # إن وُجد معها مفهوم آخر أكثر دلالة على موضوع السؤال
    non_meta_names = [c for c in primary_names if c not in META_CONCEPTS]
    topic_names = non_meta_names if non_meta_names else primary_names

    main_concept = topic_names[0]
    main_cdata   = concepts_db.get(main_concept, {})
    main_cluster = main_cdata.get("cluster", "")
    opener = CLUSTER_OPENERS.get(main_cluster, "يتحدث القرآن الكريم عن")

    # إزالة المفاهيم الثانوية التي تشترك بكلمتها الأولى مع المفهوم الأساسي
    # أو مع مفهوم ثانوي آخر سبقه (مثل "حكمة" و"حكمة عملية")
    # لتجنب التكرار في الصياغة
    seen_first_words = {main_concept.split(" ")[0]}
    secondary_names = []
    for c in topic_names[1:]:
        first_word = c.split(" ")[0]
        if first_word in seen_first_words:
            continue
        seen_first_words.add(first_word)
        secondary_names.append(c)

    # ── بناء ملخص الإجابة بصياغة طبيعية ──
    summary_parts = []

    if not secondary_names:
        summary_parts.append(f"{opener} «{main_concept}».")
    else:
        secondary = "، ".join(f"«{c}»" for c in secondary_names)
        summary_parts.append(f"{opener} «{main_concept}»، وتتصل هذه الفكرة أيضاً بـ {secondary}.")

    # ── ذكر العلاقات المستنتجة بصياغة طبيعية ──
    if related_concepts:
        # نختار أقوى علاقة من كل نوع متاح (حتى نتنوع في الصياغة) بحد أقصى 3
        seen_types = {}
        for r in related_concepts:
            rtype = r.get("relation_type", "")
            if rtype not in seen_types and r["concept"] != main_concept:
                seen_types[rtype] = r
            if len(seen_types) >= 3:
                break

        rel_sentences = []
        for rtype, r in seen_types.items():
            phrase = RELATION_PHRASES.get(rtype, "وترتبط بمفهوم")
            target = r["concept"]
            # إزالة بادئة "root:" إن وُجدت في أسماء الجذور
            target_display = target.replace("root:", "")
            rel_sentences.append(f"{phrase} «{target_display}»")

        if rel_sentences:
            summary_parts.append("، ".join(rel_sentences) + ".")

    # ── ذكر الآيات بصياغة استشهادية ──
    if verses:
        if len(verses) == 1:
            v = verses[0]
            summary_parts.append(f"ومن الآيات الدالة على ذلك قوله تعالى في سورة {v['surah']} الآية {v['ayah']}.")
        else:
            refs = "، ".join(f"({v['surah']}:{v['ayah']})" for v in verses[:3])
            extra = f"، وغيرها من {len(verses)} آية" if len(verses) > 3 else ""
            summary_parts.append(f"ومن الآيات الدالة على ذلك: {refs}{extra}.")
    else:
        summary_parts.append("ولم يُعثر على آيات مرتبطة مباشرة بهذا المفهوم في الفهرس الحالي.")

    summary = " ".join(summary_parts)

    # ── تفاصيل المفاهيم الأساسية ──
    primary_details = []
    for cname, score in concept_matches[:5]:
        cdata = concepts_db.get(cname, {})
        primary_details.append({
            "name":      cname,
            "cluster":   cdata.get("cluster", "غير مصنّف"),
            "frequency": cdata.get("frequency", 0),
            "match":     score,
        })

    return {
        "question":         question,
        "summary":          summary,
        "primary_concepts": primary_details,
        "related_concepts": related_concepts,
        "verses":           verses,
        "confidence":       confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6) الدالة الرئيسية — تجميع كل المراحل
# ═══════════════════════════════════════════════════════════════════════════

def _auto_yemeni_rag_boost(
    question: str,
    concept_matches: List[Tuple[str, float]],
    concepts_db: Dict[str, Any],
    dialect_score_threshold: float = 0.25,
) -> Tuple[List[Tuple[str, float]], Dict[str, Any]]:
    """
    حقن تلقائي لسياق RAG اللهجي عند ارتفاع درجة اللهجة اليمنية.

    - يوسّع استخراج المفاهيم عبر صيغ الاستعلام البديلة (فصيح↔لهجة)
    - يُرجع (concept_matches المحدّثة, معلومات dialect_boost)
    لا يفشل المسار أبداً: عند أي خطأ يُرجع المدخلات كما هي.
    """
    meta: Dict[str, Any] = {
        "dialect_score": 0.0,
        "is_yemeni": False,
        "rag_context": "",
        "rag_hits": [],
        "expanded_queries": [],
        "expansion_terms": [],
        "injected": False,
    }
    try:
        from ai.dialect_boost import analyze_and_boost
        info = analyze_and_boost(question, top_k_rag=3)
        meta.update({
            "dialect_score": float(info.get("dialect_score") or 0.0),
            "is_yemeni": bool(info.get("is_yemeni")),
            "rag_context": info.get("rag_context") or "",
            "rag_hits": info.get("rag_hits") or [],
            "expanded_queries": info.get("expanded_queries") or [],
            "expansion_terms": info.get("expansion_terms") or [],
        })
    except Exception as e:
        logger.debug(f"[qa_engine] dialect_boost غير متاح: {e}")
        return concept_matches, meta

    if meta["dialect_score"] < dialect_score_threshold:
        return concept_matches, meta

    # توسيع المفاهيم من الاستعلامات البديلة
    try:
        existing = {c for c, _ in concept_matches}
        for q2 in (meta.get("expanded_queries") or [])[1:4]:
            for name, score in extract_concepts_from_question(q2, concepts_db):
                if name not in existing:
                    concept_matches.append((name, round(float(score) * 0.85, 4)))
                    existing.add(name)
        for term in (meta.get("expansion_terms") or [])[:8]:
            for name, score in extract_concepts_from_question(term, concepts_db):
                if name not in existing:
                    concept_matches.append((name, round(float(score) * 0.75, 4)))
                    existing.add(name)
        concept_matches.sort(key=lambda x: -x[1])
        meta["injected"] = True
    except Exception as e:
        logger.warning(f"[qa_engine] فشل توسيع المفاهيم اللهجي: {e}")

    return concept_matches, meta


def answer_question(
    question: str,
    ckg: Dict[str, Any],
    ayat: List[Dict[str, Any]],
    max_verses: int = 5,
    max_related: int = 8,
    entities: Optional[Dict[str, Any]] = None,
    generation_mode: bool = False,
    generation_backend: str = "llm_fallback",  # "llm_fallback" أو "yemeni_decoder"
    dialect: str = "عام",
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    include_reasoning_trace: bool = False,
    include_images: bool = False,
    include_faithfulness_check: bool = False,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """غلاف رقيق حول _answer_question_impl() يضيف تخزيناً مؤقتاً اختيارياً.

    يُستخدَم الكاش فقط للحالات القطعية 100% (راجع _cacheable أعلاه) —
    أي فشل بالكاش نفسه (لا استيراد، لا مساحة قرص، إلخ) يُصمَّت تماماً
    وتُحسَب الإجابة من الصفر كالمعتاد؛ لا يوجد أي مسار يعتمد على الكاش.
    """
    cache = None
    cache_key = None
    if _cacheable(generation_mode, include_faithfulness_check, conversation_history):
        cache = _get_agent_cache()
        if cache is not None:
            try:
                cache_key = cache._make_key(
                    "qa_v1", question.strip(), max_verses, max_related,
                    include_reasoning_trace, include_images,
                )
                cached = cache.get(cache_key)
                if cached is not None:
                    import copy
                    result = copy.deepcopy(cached)
                    result["from_cache"] = True
                    return result
            except Exception as e:
                logger.warning(f"[qa_engine] فشل قراءة الكاش — سيُحسَب السؤال بشكل طبيعي: {e}")

    result = _answer_question_impl(
        question, ckg, ayat,
        max_verses=max_verses, max_related=max_related, entities=entities,
        generation_mode=generation_mode, generation_backend=generation_backend,
        dialect=dialect, temperature=temperature, top_p=top_p, top_k=top_k,
        include_reasoning_trace=include_reasoning_trace, include_images=include_images,
        include_faithfulness_check=include_faithfulness_check,
        conversation_history=conversation_history,
    )

    if cache is not None and cache_key is not None and not result.get("safety_blocked"):
        try:
            cache.set(cache_key, result)
        except Exception as e:
            logger.warning(f"[qa_engine] فشل كتابة الكاش — لا يؤثر على الإجابة الحالية: {e}")

    return result


def _answer_question_impl(
    question: str,
    ckg: Dict[str, Any],
    ayat: List[Dict[str, Any]],
    max_verses: int = 5,
    max_related: int = 8,
    entities: Optional[Dict[str, Any]] = None,
    generation_mode: bool = False,
    generation_backend: str = "llm_fallback",  # "llm_fallback" أو "yemeni_decoder"
    dialect: str = "عام",
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    include_reasoning_trace: bool = False,
    include_images: bool = False,
    include_faithfulness_check: bool = False,
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """
    نقطة الدخول الرئيسية لمحرك الأسئلة والأجوبة.

    المراحل:
      0. إن كان السؤال عن "كيان" (من هو/ما هي...) ووُجد له وصف
         في طبقة الكيانات المعرفية (entities)، تُبنى الإجابة من:
         كيان → وصف وصفات → مفاهيم مرتبطة → آيات داعمة
      1. استخراج المفاهيم من السؤال
      2. البحث في العلاقات عن مفاهيم مرتبطة
      3. استرجاع الآيات الداعمة من sources
      4. توليد إجابة منظمة مع درجة ثقة
      5. (اختياري) generation_mode=True: توليد نص حر إضافي عبر YemeniDecoder،
         مؤسَّس على نفس المفاهيم/الآيات المسترجَعة في الخطوات 1-3. لا يستبدل
         أبداً الإجابة الرمزية أعلاه — يُضاف كحقل إضافي فقط، وعند أي فشل
         (لا torch، لا checkpoint، إلخ) يُتجاهل بصمت والإجابة الرمزية تبقى
         كما هي دون أي تغيير في السلوك الافتراضي (generation_mode=False).
      6. (اختياري) include_faithfulness_check=True: يتحقق عبر
         ai/nsm_answer_verifier.py (DeepEval FaithfulnessMetric + حَكَم
         مجاني بالكامل عبر ai/free_router.py) أن result["summary"] مؤسَّس فعلاً على result["verses"]
         ولا يحتوي اختلاقاً. False افتراضياً (يستدعي LLM حَكَم بطيء
         ومكلِف، لا يجب تشغيله في كل استعلام مستخدم عادي)، ويتدهور بأمان
         كامل لو DeepEval غير مثبَّت أو لا مفتاح Anthropic متاح.
    """
    concepts_db  = ckg.get("concepts", {})
    relations_db = ckg.get("relations", {})
    entities_db  = entities or {}
    # إثراء إضافي محفوظ فعلياً في CKG (فُقد سابقاً بباغ save() الآن مُصلَح
    # — انظر knowledge/cognitive_graph.py). اختياري تماماً، يبقى {} بأمان
    # لو الملف لا يحتويهما (توافق خلفي كامل).
    arabic_roots   = ckg.get("arabic_roots", {})
    surah_profiles = ckg.get("surah_profiles", {})

    # 0. فحص الأمان أولاً — قبل أي معالجة أخرى. عند فشل تحميل الطبقة (لا
    # استيراد، إلخ) يستمر النظام بدون حظر (fallback آمن)، لكن عند نجاح
    # التحميل وثبوت أن السؤال غير آمن، يُرجَع رد محظور فوراً دون المرور
    # على CKG أو LLMFallback إطلاقاً.
    safety_checker = _get_safety_checker()
    if safety_checker is not None:
        try:
            safety_result = safety_checker(question)
            if not safety_result.is_safe:
                return _blocked_response(
                    question, safety_result.domain, safety_result.response_hint,
                )
        except Exception as e:
            logger.warning(f"[qa_engine] فشل فحص الأمان نفسه — سيستمر النظام دون حظر: {e}")

    # فهرسة الآيات بحسب (سورة، آية) لتسريع البحث
    ayat_by_ref = {(a.get("surah"), a.get("ayah")): a for a in ayat}

    # 1. استخراج المفاهيم
    concept_matches = extract_concepts_from_question(question, concepts_db)

    # 1.b حقن RAG لهجي تلقائي (توسيع مفاهيم + سياق أمثلة عند لهجة يمنية)
    concept_matches, _yemeni_rag_meta = _auto_yemeni_rag_boost(
        question, concept_matches, concepts_db,
    )

    # دعم أسئلة المتابعة: لو السؤال الحالي لم يستخرج مفاهيم كافية (أو حتى
    # لو استخرج)، نضيف مفاهيم السؤال السابق كمرشّحين إضافيين بوزن مخفَّض
    # (0.5×) — لا يستبدلون مفاهيم السؤال الحالي أبداً، فقط يسدّون الفجوة
    # لأسئلة قصيرة مثل "ليش؟"/"وماذا عن الزكاة؟" التي لا تحمل سياقاً كافياً
    # بذاتها. لا تأثير إطلاقاً لو conversation_history=None (الافتراضي).
    if conversation_history:
        prev_question = conversation_history[-1].get("question", "")
        if prev_question:
            prev_matches = extract_concepts_from_question(prev_question, concepts_db)
            existing_names = {c for c, _ in concept_matches}
            for name, score in prev_matches:
                if name not in existing_names:
                    concept_matches.append((name, round(score * 0.5, 4)))
                    existing_names.add(name)
            concept_matches.sort(key=lambda x: -x[1])

    # 2. المفاهيم المرتبطة — نستثني المفاهيم "الإطارية" (مثل: قرآن، كتاب) من
    # بذور البحث عن العلاقات، لأنها تملك علاقات قوية عامة (مثل قرآن↔كتاب)
    # قد تطغى على علاقات المفهوم الموضوعي الفعلي للسؤال
    all_primary = [c for c, _ in concept_matches[:3]]
    non_meta_primary = [c for c in all_primary if c not in META_CONCEPTS]
    primary_names = non_meta_primary if non_meta_primary else all_primary
    related_concepts = find_related_concepts(primary_names, relations_db, concepts_db, top_k=max_related) if primary_names else []
    related_concepts = _apply_neural_boost(question, related_concepts)

    # 3. الآيات الداعمة
    verses = retrieve_supporting_verses(concept_matches, concepts_db, ayat_by_ref, max_verses=max_verses)

    # 0/4. تفعيل طبقة الكيانات المعرفية لأسئلة "من هو/ما هي..."
    result: Optional[Dict[str, Any]] = None
    if entities_db and is_entity_question(question):
        entity_match = find_entity_match(concept_matches, entities_db)
        if entity_match:
            entity_name, entity_data = entity_match
            result = generate_entity_answer(
                question, entity_name, entity_data,
                concept_matches, related_concepts, verses, concepts_db,
            )

    # 4. الإجابة المنظمة (المسار العام: مفهوم → علاقات → آيات) — إن لم يُعثر
    # على كيان مطابق أعلاه
    if result is None:
        result = generate_answer(question, concept_matches, related_concepts, verses, concepts_db)

    result["safety_blocked"] = False
    result["safety_domain"] = "benign"

    # 4.5 أثر التفكير — اختياري (يبني "لماذا هذه الإجابة؟" لواجهة المستخدم)
    if include_reasoning_trace:
        cot = _get_cot_builder(ckg)
        if cot is not None:
            try:
                trace = cot.build_trace(question)
                result["reasoning_trace"] = trace.to_display()
            except Exception as e:
                logger.warning(f"[qa_engine] فشل بناء أثر التفكير: {e}")
                result["reasoning_trace"] = None
        else:
            result["reasoning_trace"] = None
    else:
        result["reasoning_trace"] = None

    # 4.6 صور توضيحية — اختياري
    if include_images:
        top_concept = (result.get("primary_concepts") or [{}])[0].get("name", "")
        result["images"] = _try_fetch_illustration(top_concept)
    else:
        result["images"] = []

    # 4.7 تحسين درجة الثقة عبر self_awareness_deep (adapter جزئي — انظر
    # التعليق التوثيقي أعلى _refine_confidence لحدود هذا التكامل)
    result["confidence"] = _refine_confidence(result)

    # 5. توليد حر اختياري (Yemeni LLM) — إضافي فقط، لا يمسّ أي من المسارين أعلاه
    result["generation_used"] = False
    result["generated_text"] = None
    result["generation_backend"] = None
    result["conversation_context_used"] = bool(
        conversation_history and conversation_history[-1].get("question")
    )
    if generation_mode:
        generated: Optional[str] = None
        if generation_backend == "llm_fallback":
            generated = _rephrase_in_yemeni_dialect(
                question, result, dialect,
                arabic_roots=arabic_roots, surah_profiles=surah_profiles,
                conversation_history=conversation_history,
            )
            backend_used = "llm_fallback"
            if generated is None:
                generated = _try_generate_free_text(
                    question, concept_matches, verses, temperature, top_p, top_k,
                    arabic_roots=arabic_roots, surah_profiles=surah_profiles,
                )
                backend_used = "yemeni_decoder" if generated else None
        else:
            generated = _try_generate_free_text(
                question, concept_matches, verses, temperature, top_p, top_k,
                arabic_roots=arabic_roots, surah_profiles=surah_profiles,
            )
            backend_used = "yemeni_decoder" if generated else None

        if generated:
            result["generated_text"] = generated
            result["generation_used"] = True
            result["generation_backend"] = backend_used

    # 6. فحص تأسيس الإجابة على المصدر — اختياري تماماً (انظر توثيق
    # include_faithfulness_check أعلاه). None افتراضياً لو لم يُطلَب.
    result["faithfulness_check"] = (
        _check_answer_faithfulness(question, result) if include_faithfulness_check else None
    )

    # 7. بيانات حقن RAG اللهجي (للواجهة والمسارات اللاحقة)
    try:
        result["dialect_score"] = float(_yemeni_rag_meta.get("dialect_score") or 0.0)
        result["dialect_is_yemeni"] = bool(_yemeni_rag_meta.get("is_yemeni"))
        result["dialect_rag_injected"] = bool(_yemeni_rag_meta.get("injected"))
        result["dialect_rag_context"] = _yemeni_rag_meta.get("rag_context") or ""
        result["dialect_rag_hits"] = _yemeni_rag_meta.get("rag_hits") or []
        result["dialect_expanded_queries"] = _yemeni_rag_meta.get("expanded_queries") or []
        # إلحاق مقتطف أمثلة لهجية بالملخص فقط عند حقن فعلي ووجود سياق
        if (
            result.get("dialect_rag_injected")
            and result.get("dialect_rag_context")
            and result.get("summary")
        ):
            # لا نطيل الملخص أكثر من اللازم — سطران كحد
            ctx = result["dialect_rag_context"].strip()
            if ctx and ctx not in result["summary"]:
                result["dialect_context_note"] = ctx
    except Exception:
        result.setdefault("dialect_score", 0.0)
        result.setdefault("dialect_rag_injected", False)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7) أداة مساعدة: تشابه نصي بين سؤالين (للذاكرة التجريبية لاحقاً)
# ═══════════════════════════════════════════════════════════════════════════
def question_similarity(q1: str, q2: str) -> float:
    """
    تشابه بسيط بين سؤالين بناءً على تقاطع الكلمات (Jaccard)
    بعد التطبيع وإزالة كلمات الوقف.
    """
    def tokenize(q: str) -> set:
        norm = normalize_arabic(q)
        raw = re.split(r'[\s\u060C\u061F\u061B,.!?؟،؛]+', norm)
        return {_strip_prefixes(w) for w in raw if w and w not in ARABIC_STOPWORDS}

    t1, t2 = tokenize(q1), tokenize(q2)
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return round(inter / union, 4) if union else 0.0
