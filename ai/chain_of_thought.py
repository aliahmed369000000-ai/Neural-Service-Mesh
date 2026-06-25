"""
Chain-of-Thought Reasoning Builder — التفكير التسلسلي + Few-shot
====================================================================
الأولوية #2 من تقرير التحليل النهائي (الصورة المرفقة):
  "Few-shot Prompting + Chain-of-Thought — يجعل النظام 'يشرح تفكيره'
   ويتعلم من أمثلة قليلة."

المشكلة المكتشفة في التحليل:
  ai/prompt_engine.py (FewShotPromptEngine) موجود ويعمل، لكنه **غير مُستخدَم
  في أي مكان** (orphan module). النتيجة: النظام لا يبني أمثلة few-shot فعلية،
  ولا توجد أي خطوة "تفكير" مرئية أو ضمنية قبل الإجابة — فقط dictionary lookup
  مباشر (راجع نتيجة التحليل: "Few-shot learning from prompt: ❌").

الحل (هذا الملف):
  ChainOfThoughtBuilder يربط 3 مصادر معرفة موجودة فعلاً في المشروع:
    1. FewShotPromptEngine  (ai/prompt_engine.py)   → أمثلة مشابهة من الذاكرة
    2. CognitiveKnowledgeGraph (knowledge/cognitive_graph.py) → مفاهيم مرتبطة
    3. السؤال نفسه                                  → استخراج كلمات مفتاحية

  ويبني منها ReasoningTrace يحتوي:
    • خطوات تفكير صريحة ومقروءة (تعمل حتى بدون أي LLM خارجي — استرجاع حتمي)
    • Prompt مُعزَّز بالأمثلة + تعليمات CoT جاهز للإرسال لأي LLM حقيقي

  هذا يحقق فعلياً:
    ✅ Few-shot Prompting  — أمثلة حقيقية تُسترجع وتُدمج في الـ prompt
    ✅ Chain-of-Thought    — خطوات تفكير صريحة (شفّافة) قبل كل إجابة
    ✅ يعمل في كل الأحوال  — مع LLM حي، أو مع CKG synthesis فقط، أو حتى
                             بدون أي شيء (يرجع إلى أمثلة احتياطية مدمجة)

التكامل (بدون تعديل أي ملف موجود):
    from ai.chain_of_thought import ChainOfThoughtBuilder

    cot = ChainOfThoughtBuilder(ckg=my_ckg)          # ckg اختياري
    trace = cot.build_trace("ما حكم الزكاة على الراتب؟")

    print(trace.to_display())     # خطوات التفكير الكاملة (شفافة)
    llm_query = trace.llm_prompt  # تُمرَّر مباشرة كـ query لـ LLMFallback.generate()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from ai.prompt_engine import FewShotPromptEngine, RetrievedExample

logger = logging.getLogger(__name__)

_STOP_WORDS = {
    "هل", "ما", "من", "في", "عن", "على", "إلى", "هو", "هي", "كيف",
    "لماذا", "متى", "أين", "الذي", "التي", "كان", "كانت", "كل", "بعض",
    "the", "is", "are", "what", "how", "why", "when", "where", "a", "an",
}

_COT_INSTRUCTION = (
    "فكّر في السؤال خطوة بخطوة بالاستناد إلى الأمثلة والمفاهيم أعلاه، "
    "ثم اكتب فقط الإجابة النهائية المختصرة (3-5 جمل) بالعربية الفصحى — "
    "بدون إظهار خطوات تفكيرك الداخلية في الرد."
)


# ════════════════════════════════════════════════════════════════════════════
# ReasoningTrace — أثر التفكير الكامل لسؤال واحد
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class ReasoningTrace:
    """أثر تفكير شفّاف وقابل للعرض — كل خطوة موثّقة بشكل حتمي وقابل للتدقيق."""
    query:          str
    concepts:       List[str]                 = field(default_factory=list)
    examples:       List[RetrievedExample]     = field(default_factory=list)
    ckg_related:    List[Tuple[str, float]]    = field(default_factory=list)
    steps:          List[str]                  = field(default_factory=list)
    llm_prompt:     str                        = ""

    def to_display(self) -> str:
        """يبني نصاً عربياً يعرض خطوات التفكير كاملة — للشفافية / التصحيح."""
        lines = ["🧠 خطوات التفكير:"]
        for i, step in enumerate(self.steps, 1):
            lines.append(f"  {i}. {step}")
        return "\n".join(lines)

    def has_examples(self) -> bool:
        return len(self.examples) > 0

    def has_ckg_links(self) -> bool:
        return len(self.ckg_related) > 0


# ════════════════════════════════════════════════════════════════════════════
# ChainOfThoughtBuilder — المحرك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class ChainOfThoughtBuilder:
    """
    يبني ReasoningTrace لكل سؤال: استخراج مفاهيم → أمثلة مشابهة (few-shot)
    → مفاهيم مرتبطة من CKG (إن وُجد) → prompt مُعزَّز جاهز لـ LLM.

    لا يعتمد على أي LLM داخلياً — هو طبقة "تجهيز" فقط، يبقى استدعاء
    النموذج (أو CKG synthesis، أو القاموس الثابت) مسؤولية الطبقة الأعلى
    (مثل NSMChatPlus / LLMFallback) كما هي بدون أي تعديل.
    """

    def __init__(
        self,
        ckg: Any = None,
        few_shot_engine: Optional[FewShotPromptEngine] = None,
        k_examples: int = 3,
        k_concepts: int = 5,
        show_internal_steps: bool = False,
    ):
        """
        Args:
            ckg:                 كائن CognitiveKnowledgeGraph (اختياري)
            few_shot_engine:     FewShotPromptEngine جاهز (يُنشأ تلقائياً إن لم يُعطَ)
            k_examples:          عدد أمثلة few-shot المُسترجعة
            k_concepts:          عدد المفاهيم المرتبطة المُسترجعة من CKG
            show_internal_steps: إن True، يُطلب من LLM إظهار خطوات تفكيره
                                  في الرد (مفيد للتصحيح فقط — الإجابات الافتراضية
                                  تبقى مختصرة كما في النظام الأصلي)
        """
        self.ckg          = ckg
        self.engine       = few_shot_engine or FewShotPromptEngine(k=k_examples)
        self.k_examples   = k_examples
        self.k_concepts   = k_concepts
        self.show_steps   = show_internal_steps

    # ── واجهة عامة ──────────────────────────────────────────────────────────

    def build_trace(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> ReasoningTrace:
        """يبني أثر التفكير الكامل لسؤال واحد."""
        steps: List[str] = []

        # الخطوة 1 — استخراج المفاهيم/الكلمات المفتاحية من السؤال
        concepts = self._extract_concepts(query)
        if concepts:
            steps.append(f"استخراج المفاهيم الجوهرية من السؤال: {', '.join(concepts)}.")
        else:
            steps.append("لم تُستخرج مفاهيم جوهرية واضحة من نص السؤال.")

        # الخطوة 2 — استرجاع أمثلة مشابهة (Few-shot)
        examples = self.engine.retrieve(query, k=self.k_examples)
        if examples:
            ex_desc = "؛ ".join(
                f"«{ex.question.strip()[:40]}…» (تشابه {ex.similarity:.2f}, مصدر {ex.source})"
                for ex in examples
            )
            steps.append(f"استرجاع {len(examples)} مثال مشابه من الذاكرة: {ex_desc}.")
        else:
            steps.append("لم يُعثر على أمثلة مشابهة كافية في الذاكرة.")

        # الخطوة 3 — مفاهيم مرتبطة من الرسم المعرفي (إن وُجد CKG)
        ckg_related = self._retrieve_ckg_related(concepts)
        if ckg_related:
            rel_desc = "، ".join(f"{name} ({w:.2f})" for name, w in ckg_related)
            steps.append(f"مفاهيم مرتبطة عُثر عليها في الرسم المعرفي (CKG): {rel_desc}.")
        elif self.ckg is not None:
            steps.append("لا توجد مفاهيم مرتبطة في الرسم المعرفي لهذا السؤال.")

        # الخطوة 4 — التوليف النهائي
        steps.append(
            "تركيب الإجابة النهائية بالاستناد إلى الأمثلة والمفاهيم المسترجعة أعلاه."
        )

        llm_prompt = self._build_llm_prompt(query, examples, ckg_related, history)

        return ReasoningTrace(
            query=query,
            concepts=concepts,
            examples=examples,
            ckg_related=ckg_related,
            steps=steps,
            llm_prompt=llm_prompt,
        )

    def build_llm_query(
        self,
        query: str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> str:
        """اختصار: يُرجع فقط نص الـ prompt المُعزَّز (بدون بناء trace كامل)."""
        return self.build_trace(query, history=history).llm_prompt

    # ── الآليات الداخلية ──────────────────────────────────────────────────────

    def _extract_concepts(self, query: str) -> List[str]:
        words = [w.strip("؟!.,،") for w in query.split()]
        return [w for w in words if len(w) > 2 and w.lower() not in _STOP_WORDS][: self.k_concepts]

    def _retrieve_ckg_related(self, concepts: List[str]) -> List[Tuple[str, float]]:
        if self.ckg is None or not concepts:
            return []
        candidates = {}
        for word in concepts:
            try:
                related = self.ckg.query_related(word, top_k=self.k_concepts)
                for name, weight in related:
                    candidates[name] = max(candidates.get(name, 0.0), weight)
            except Exception as exc:
                logger.debug(f"[CoT] query_related فشل لـ '{word}': {exc}")
        ranked = sorted(candidates.items(), key=lambda x: -x[1])
        return ranked[: self.k_concepts]

    def _build_llm_prompt(
        self,
        query: str,
        examples: List[RetrievedExample],
        ckg_related: List[Tuple[str, float]],
        history: Optional[List[Tuple[str, str]]],
    ) -> str:
        parts: List[str] = []

        if examples:
            parts.append("أمثلة مرجعية مشابهة (Few-shot):")
            for i, ex in enumerate(examples, 1):
                q = ex.question.strip()[:120]
                a = ex.answer.strip()[:200]
                parts.append(f"  {i}) س: {q}\n     ج: {a}")

        if ckg_related:
            names = "، ".join(name for name, _ in ckg_related)
            parts.append(f"\nمفاهيم ذات صلة من الرسم المعرفي للنظام: {names}.")

        parts.append(f"\nالسؤال: {query.strip()}")

        instruction = _COT_INSTRUCTION
        if self.show_steps:
            instruction = (
                "اشرح خطوات تفكيرك بإيجاز أولاً (نقطة واحدة لكل خطوة)، ثم اكتب "
                "الإجابة النهائية بوضوح تحت عنوان 'الإجابة:'."
            )
        parts.append(instruction)

        return "\n".join(parts)

    # ── إحصاءات ──────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        return {
            "k_examples": self.k_examples,
            "k_concepts": self.k_concepts,
            "has_ckg":    self.ckg is not None,
            "show_internal_steps": self.show_steps,
            "engine_stats": self.engine.get_stats(),
        }
