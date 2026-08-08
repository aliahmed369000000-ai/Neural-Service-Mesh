"""
ai/nsm_answer_verifier.py
==========================
مُتحقِّق تأسيس إجابات knowledge/qa_engine.py على المصدر (Faithfulness
Verifier) — يتحقق أن ملخص الإجابة (result["summary"]) مبني فعلاً على
الآيات/المفاهيم المسترجَعة (result["verses"] + result["primary_concepts"])
ولا "يختلق" معلومة غير موجودة في السياق المسترجَع فعلياً.

⚠️ هذه وحدة مختلفة تماماً عن ai/nsm_verifier.py:
  - ai/nsm_verifier.py    → يحكم هل ناتج تنفيذ الوكيل البرمجي لمهمة
                            (كود/ملفات) يطابق وصف المهمة. حَكَم LLM
                            داخلي مبني خصيصاً لهذا النظام.
  - ai/nsm_answer_verifier.py (هذا الملف) → يحكم هل *محتوى معرفي* (إجابة
                            دينية/قرآنية) مؤسَّس فعلاً على المصدر
                            المسترجَع، أم اختلاق. يستخدم مقياساً جاهزاً
                            من مكتبة DeepEval مفتوحة المصدر
                            (https://github.com/confident-ai/deepeval)
                            — FaithfulnessMetric، مصمَّم خصيصاً لأنظمة
                            RAG: يستخرج "truths" من retrieval_context ثم
                            "claims" من actual_output، ويحكم حَكَم LLM هل
                            كل claim مدعوم بـtruths فعلاً.

لماذا DeepEval تحديداً (من بين RAGAS/DeepEval وغيرها):
  - يوفّر CLI فعلي (`deepeval test run`) وتكامل pytest أصيل — يطابق
    اصطلاح الاختبارات الحالي في المشروع (test_*.py بجانب الوحدات).
  - FaithfulnessMetric لا يتطلب OpenAI حصراً: واجهة DeepEvalBaseLLM
    تسمح بربط أي نموذج — هنا موجّه النماذج المجانية نفسه المستخدَم في
    بقية المشروع (ai/free_router.py: Groq → Gemini → Cloudflare)، فلا
    اعتماد على أي مزوّد مدفوع أو مفتاح Anthropic إضافي.
  - رخصة Apache 2.0 مفتوحة تماماً، وصيانة نشطة.

مجاني بالكامل — بلا أي اعتماد مدفوع:
  - الحَكَم (judge LLM) هو ai/free_router.py::chat_free نفسه المستخدَم
    في بقية المشروع (Groq → Gemini → Cloudflare، بالترتيب حتى ينجح
    أحدها) — لا مفتاح Anthropic ولا أي مزوّد مدفوع آخر مطلوب إطلاقاً.
  - DeepEval نفسها مكتبة Apache 2.0 مجانية بالكامل.

تدهور آمن كامل (نفس نمط بقية الوحدات الاختيارية في المشروع، مثل
_get_deep_awareness في qa_engine.py):
  - DeepEval غير مثبَّت (اعتماد اختياري — غير مُدرَج في requirements.txt
    الأساسي لتفادي إثقال بيئة Streamlit Cloud محدودة الذاكرة، بل في
    requirements-verifier.txt منفصل) → available=False، بلا استثناء.
  - لا يوجد أي مفتاح نموذج مجاني (GROQ_API_KEY / GOOGLE_API_KEY أو
    CF_API_TOKEN+CF_ACCOUNT_ID) → available=False، بلا استثناء.
  - أي فشل تقني أثناء القياس نفسه (شبكة، فشل كل المزوّدات المجانية،
    JSON غير صالح من الحَكَم، ...) → available=False + سبب واضح، بلا
    استثناء يوقف answer_question().

الاستخدام النموذجي:
    from ai.nsm_answer_verifier import verify_answer_faithfulness
    report = verify_answer_faithfulness(question, qa_result)
    if report["available"] and report["faithful"] is False:
        logger.warning(f"إجابة قد تحتوي اختلاقاً: {report['reason']}")

أو ضمن answer_question() نفسها عبر include_faithfulness_check=True
(اختياري تماماً، False افتراضياً — لأنه يستدعي LLM حَكَم إضافياً وقد
يكون بطيئاً نسبياً حتى لو مجانياً، ولا يجب أن يشغَّل في كل استعلام
مستخدم عادي).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NSMAnswerVerifier")

_DEEPEVAL_AVAILABLE: Optional[bool] = None


def _deepeval_importable() -> bool:
    """يفحص توفر مكتبة deepeval مرة واحدة فقط ويخزّن النتيجة (نفس نمط
    _get_llm_fallback/_get_deep_awareness في بقية المشروع)."""
    global _DEEPEVAL_AVAILABLE
    if _DEEPEVAL_AVAILABLE is None:
        try:
            import deepeval  # noqa: F401
            _DEEPEVAL_AVAILABLE = True
        except Exception:
            _DEEPEVAL_AVAILABLE = False
    return _DEEPEVAL_AVAILABLE


class _FreeRouterJudge:
    """يكسو ai.free_router.chat_free بواجهة deepeval.models.DeepEvalBaseLLM
    — الحَكَم (judge LLM) المستخدَم لقياس FaithfulnessMetric هو موجّه
    النماذج المجانية نفسه المستخدَم في بقية المشروع (Groq → Gemini →
    Cloudflare)، بدل اعتماد DeepEval الافتراضي على OpenAI أو أي مزوّد
    مدفوع. اسم النموذج الفعلي الذي نجح يُحدَّث بعد كل استدعاء (يختلف
    حسب أي مزوّد استجاب أولاً)."""

    def __init__(self, model_name: str = "free-router") -> None:
        self._model_name = model_name

    def load_model(self):
        return self

    def _ask(self, prompt: str) -> str:
        from ai.free_router import chat_free
        text, model_used = chat_free(
            [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=1024,
        )
        self._model_name = model_used
        return text

    def generate(self, prompt: str, schema: Optional[Any] = None) -> Any:
        """يطابق عقد DeepEvalBaseLLM.generate: بلا schema يُعاد نص خام،
        ومع schema (نموذج pydantic تستخدمه مقاييس DeepEval الداخلية مثل
        FaithfulnessMetric لاستخراج truths/claims بصيغة منظّمة) يُعاد
        كائن الـschema نفسه بعد تحليل رد الحَكَم كـJSON."""
        if schema is None:
            return self._ask(prompt)
        return self._generate_structured(prompt, schema)

    async def a_generate(self, prompt: str, schema: Optional[Any] = None) -> Any:
        # AnthropicAdvanced متزامن بالكامل (urllib.request) — لا نسخة
        # async حقيقية متاحة حالياً، فنعيد استخدام المسار المتزامن.
        return self.generate(prompt, schema)

    def _generate_structured(self, prompt: str, schema: Any) -> Any:
        format_hint = (
            "\n\nأجب بصيغة JSON صالحة فقط تطابق تماماً هذا المخطط، بلا أي "
            f"نص أو شرح خارج الـJSON نفسه:\n{schema.model_json_schema()}"
        )
        raw = self._ask(prompt + format_hint)
        for attempt_text in (raw, _extract_json_block(raw)):
            if not attempt_text:
                continue
            try:
                return schema.model_validate_json(attempt_text)
            except Exception:
                continue
        # محاولة أخيرة: إعادة سؤال الحَكَم صراحة بإعادة تهيئة نفس الرد فقط
        retry_raw = self._ask(
            "أعد صياغة هذا كـJSON صالح فقط يطابق نفس المخطط أعلاه، بلا أي "
            f"نص خارج الـJSON:\n{raw}"
        )
        return schema.model_validate_json(retry_raw)

    def get_model_name(self) -> str:
        return self._model_name


def _extract_json_block(text: str) -> str:
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return ""


def _build_retrieval_context(qa_result: Dict[str, Any]) -> List[str]:
    """يبني قائمة نصوص السياق المسترجَع (retrieval_context) من نفس
    البيانات التي بُنيت منها الإجابة أصلاً في answer_question() — الآيات
    الداعمة الفعلية (verses[].text) والمفاهيم الأساسية، بدل إعادة
    استرجاع منفصل قد يختلف عمّا استُخدم فعلياً في بناء الإجابة."""
    verses = qa_result.get("verses") or []
    context = [
        f"سورة {v.get('surah')} آية {v.get('ayah')}: {v.get('text', '')}".strip()
        for v in verses
        if v.get("text")
    ]
    primary = qa_result.get("primary_concepts") or []
    names = "، ".join(p.get("name", "") for p in primary if p.get("name"))
    if names:
        context.append(f"المفاهيم الأساسية المستخرَجة من السؤال: {names}")
    return context



def _normalize_ar_tokens(text: str) -> set:
    """تطبيع خفيف + تقسيم إلى كلمات عربية/لاتينية ذات معنى (≥2 أحرف)."""
    import re
    if not text:
        return set()
    # إزالة التشكيل والأرقام الزائدة الشائعة
    text = re.sub(r"[\u064B-\u065F\u0670\u0640]", "", text)
    text = re.sub(r"[^\u0600-\u06FFa-zA-Z0-9\s]", " ", text)
    tokens = {t.lower() for t in text.split() if len(t) >= 2}
    # كلمات وظيفية شائعة لا تفيد في قياس التأسيس
    stop = {
        "من", "في", "على", "إلى", "عن", "أن", "إن", "ما", "لا", "لم", "لن",
        "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "هو", "هي", "هم", "ثم",
        "أو", "و", "ف", "ب", "ك", "ل", "ال", "قد", "كان", "يكون", "بين",
        "the", "a", "an", "of", "to", "in", "on", "and", "or", "is", "are",
    }
    return {t for t in tokens if t not in stop}


def _lexical_faithfulness_score(summary: str, retrieval_context: list) -> dict:
    """مقياس تأسيس معجمي محلي بالكامل — يعمل بدون deepeval وبدون أي LLM.
    مفيد في الوضع المغلق (NSM_OFFLINE_MODE) أو عند غياب مفاتيح النماذج المجانية.

    score ≈ نسبة كلمات الإجابة التي تظهر أيضاً في السياق المسترجَع.
    ليس بديلاً كاملاً عن FaithfulnessMetric (لا يفهم المعنى)، لكنه إشارة
    عملية سريعة ومنعدمة التكلفة.
    """
    sum_toks = _normalize_ar_tokens(summary)
    if not sum_toks:
        return {"score": None, "overlap": 0, "summary_tokens": 0, "context_tokens": 0}
    ctx_toks: set = set()
    for c in retrieval_context:
        ctx_toks |= _normalize_ar_tokens(c)
    if not ctx_toks:
        return {"score": 0.0, "overlap": 0, "summary_tokens": len(sum_toks), "context_tokens": 0}
    overlap = sum_toks & ctx_toks
    score = len(overlap) / max(len(sum_toks), 1)
    return {
        "score": round(float(score), 4),
        "overlap": len(overlap),
        "summary_tokens": len(sum_toks),
        "context_tokens": len(ctx_toks),
    }


def verify_answer_faithfulness(
    question: str,
    qa_result: Dict[str, Any],
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    يتحقق أن qa_result["summary"] مؤسَّس فعلاً على qa_result["verses"] +
    qa_result["primary_concepts"] — أي لا يحتوي معلومة مُختلَقة غير
    موجودة في السياق المسترجَع.

    يُرجع دائماً dict بنفس الشكل التالي، بلا رمي استثناء أبداً:
        {
            "available": bool,        # هل تم القياس فعلياً؟
            "faithful":  bool | None, # None لو لا يمكن الحكم
            "score":     float | None,
            "method":    str | None,  # "deepeval" | "lexical" | None
            "reason":    str,
        }
    عند غياب deepeval أو مفاتيح الحَكَم أو في NSM_OFFLINE_MODE يُستخدم
    مقياس معجمي محلي (method="lexical") بدل الفشل الصامت.
    """
    retrieval_context = _build_retrieval_context(qa_result)
    if not retrieval_context:
        return {
            "available": True, "faithful": None, "score": None,
            "method": None,
            "reason": "لا يوجد سياق مسترجَع (لا آيات ولا مفاهيم أساسية) — لا معنى لفحص التأسيس على مصدر غير موجود أصلاً.",
        }

    summary = qa_result.get("summary", "")
    if not summary:
        return {
            "available": True, "faithful": None, "score": None,
            "method": None,
            "reason": "لا يوجد نص إجابة (summary) لفحصه.",
        }

    def _lexical_report(extra_reason: str = "") -> Dict[str, Any]:
        lex = _lexical_faithfulness_score(summary, retrieval_context)
        score = lex.get("score")
        faithful = bool(score is not None and score >= threshold) if score is not None else None
        reason = (
            f"مقياس معجمي محلي (بدون LLM): تداخل {lex.get('overlap', 0)}/"
            f"{lex.get('summary_tokens', 0)} كلمة من الإجابة مع السياق. "
            f"{extra_reason}"
        ).strip()
        return {
            "available": True,
            "faithful": faithful,
            "score": score,
            "method": "lexical",
            "reason": reason,
            "lexical_detail": lex,
        }

    # تفضيل DeepEval + حَكَم مجاني عند التوفّر؛ وإلا fallback معجمي
    # (يعمل في الوضع المغلق وبدون أي مفاتيح API).
    use_deepeval = _deepeval_importable()
    has_key = False
    if use_deepeval:
        try:
            from ai.free_router import has_any_free_key
            has_key = bool(has_any_free_key())
        except Exception:
            has_key = False

    # الوضع المغلق: لا نحاول استدعاء مزوّدين سحابيين أصلاً
    try:
        from ai.offline_mode import is_offline
        if is_offline():
            return _lexical_report("NSM_OFFLINE_MODE=1 — استُخدم المسار المعجمي.")
    except Exception:
        pass

    if not use_deepeval or not has_key:
        why = (
            "deepeval غير مثبَّت (انظر requirements-verifier.txt)."
            if not use_deepeval
            else "لا يوجد مفتاح نموذج مجاني للحَكَم (GROQ/Gemini/Cloudflare)."
        )
        return _lexical_report(why)

    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        judge = _FreeRouterJudge()
        metric = FaithfulnessMetric(
            threshold=threshold, model=judge, include_reason=True,
            async_mode=False,
        )
        test_case = LLMTestCase(
            input=question, actual_output=summary,
            retrieval_context=retrieval_context,
        )
        metric.measure(test_case)
        score = float(metric.score) if metric.score is not None else None
        return {
            "available": True,
            "faithful": bool(score is not None and score >= threshold),
            "score": score,
            "method": "deepeval",
            "reason": metric.reason or "",
        }
    except Exception as e:
        logger.warning(f"[nsm_answer_verifier] فشل قياس FaithfulnessMetric: {e} — انتقال للمعجمي")
        return _lexical_report(f"فشل DeepEval ({e})؛ استُخدم المسار المعجمي.")
