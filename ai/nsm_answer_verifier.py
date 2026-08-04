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
            "faithful":  bool | None, # None لو available=False
            "score":     float | None,
            "reason":    str,
        }
    """
    if not _deepeval_importable():
        return {
            "available": False, "faithful": None, "score": None,
            "reason": (
                "deepeval غير مثبَّت (اعتماد اختياري — انظر "
                "requirements-verifier.txt). هذا التحقق اختياري بالكامل "
                "ولا يؤثر على answer_question() نفسها."
            ),
        }
    from ai.free_router import has_any_free_key
    if not has_any_free_key():
        return {
            "available": False, "faithful": None, "score": None,
            "reason": (
                "لا يوجد أي مفتاح نموذج مجاني متاح (GROQ_API_KEY أو "
                "GOOGLE_API_KEY أو CF_API_TOKEN+CF_ACCOUNT_ID) — الحَكَم "
                "(judge LLM) يحتاج واحداً منها على الأقل، وكلها مجانية."
            ),
        }

    retrieval_context = _build_retrieval_context(qa_result)
    if not retrieval_context:
        return {
            "available": True, "faithful": None, "score": None,
            "reason": "لا يوجد سياق مسترجَع (لا آيات ولا مفاهيم أساسية) — لا معنى لفحص التأسيس على مصدر غير موجود أصلاً.",
        }

    summary = qa_result.get("summary", "")
    if not summary:
        return {
            "available": True, "faithful": None, "score": None,
            "reason": "لا يوجد نص إجابة (summary) لفحصه.",
        }

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
            "reason": metric.reason or "",
        }
    except Exception as e:
        logger.warning(f"[nsm_answer_verifier] فشل قياس FaithfulnessMetric: {e}")
        return {
            "available": False, "faithful": None, "score": None,
            "reason": f"فشل تقني أثناء القياس: {e}",
        }
