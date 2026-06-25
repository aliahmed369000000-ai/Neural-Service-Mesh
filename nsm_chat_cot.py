"""
NSMChat-CoT — امتداد إضافي يُفعّل Few-shot + Chain-of-Thought فعلياً
======================================================================
يُطبّق الأولوية #2 من تقرير التحليل النهائي على مستوى المحادثة الفعلية.

البنية (كل طبقة تُبنى فوق التي قبلها، بدون تعديل أي ملف موجود):

    NSMChat        (nsm_chat.py)        — القاموس الثابت + الذاكرة
        ↓ يُمتد بدون تعديل
    NSMChatPlus    (nsm_chat_plus.py)   — + توليد LLM حقيقي عند الحاجة
        ↓ يُمتد بدون تعديل
    NSMChatCoT     (هذا الملف)          — + Few-shot حقيقي + CoT شفّاف

ماذا يُضيف هذا الملف فعلياً (الذي كان مفقوداً):
  ✅ عند اللجوء لطبقة التوليد (KB score منخفض)، يُبنى prompt يحتوي على
     أمثلة Few-shot حقيقية مُسترجعة من ذاكرة المحادثات + مفاهيم من CKG،
     مع تعليمة Chain-of-Thought صريحة — وليس فقط استدعاء النموذج بالسؤال
     الخام كما كان يحدث سابقاً.
  ✅ أثر تفكير شفّاف (ReasoningTrace) متاح عبر `last_reasoning` لكل سؤال —
     يعمل حتى بدون أي مفتاح LLM (يعتمد على استرجاع حتمي من الذاكرة + CKG).
  ✅ مسار القاموس الثابت (KB) يبقى تماماً كما هو — سريع وحرفي، بدون أي
     تغيير في سلوكه أو نتائجه. لا يُلمس إلا مسار "التوليد" فقط.

الاستخدام:
    from nsm_chat_cot import NSMChatCoT
    from knowledge.cognitive_graph import get_ckg

    bot = NSMChatCoT(ckg=get_ckg())     # ckg اختياري تماماً
    reply = bot.chat("ما حكم الزكاة على الراتب؟")
    print(reply)
    print(bot.last_reasoning.to_display())   # عرض خطوات التفكير (شفافية)

للتوافق الكامل مع الواجهات الموجودة (nsm_chat_ui.py وغيرها)، الكلاس يطابق
تماماً واجهة NSMChat / NSMChatPlus: chat(text)->str، history، clear_history، إلخ.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple

from nsm_chat_plus import NSMChatPlus
from ai.chain_of_thought import ChainOfThoughtBuilder, ReasoningTrace
from ai.llm_fallback import Provider

logger = logging.getLogger("NSMChatCoT")


class NSMChatCoT(NSMChatPlus):
    """
    NSMChatPlus مُعزَّز بطبقة Few-shot + Chain-of-Thought حقيقية.

    التسلسل عند كل سؤال (مطابق تماماً لـ NSMChatPlus إلا في خطوة واحدة):
      1. تغنية الاستعلام بالسياق (كما في NSMChatPlus / NSMChat الأصلي)
      2. بحث في القاموس الثابت (_KB) — بدون أي تغيير
      3. score >= threshold → إجابة القاموس مباشرة (لا تغيير في هذا المسار)
      4. score <  threshold → ★ الجديد هنا ★:
           a. يُبنى ReasoningTrace (مفاهيم + أمثلة few-shot + روابط CKG)
           b. إن وُجد LLM حي (Groq/OpenAI/HF) → يُرسَل trace.llm_prompt
              المُعزَّز بالأمثلة وتعليمة CoT، بدل السؤال الخام فقط
           c. إن لم يوجد LLM حي (CKG synthesis فقط) → يُستخدم السؤال
              الأصلي تماماً كما في NSMChatPlus (لا تغيير — نتجنّب إرباك
              منطق استخراج الكلمات المفتاحية في _ckg_synthesize)
    """

    def __init__(
        self,
        ckg=None,
        k_examples: int = 3,
        k_concepts: int = 5,
        show_internal_steps: bool = False,
    ):
        super().__init__(ckg=ckg)
        self.cot = ChainOfThoughtBuilder(
            ckg=ckg,
            k_examples=k_examples,
            k_concepts=k_concepts,
            show_internal_steps=show_internal_steps,
        )
        self.last_reasoning: Optional[ReasoningTrace] = None
        logger.info("[NSMChatCoT] جاهز | Few-shot + CoT مُفعَّلان لمسار التوليد")

    # ── override chat ────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        if not user_input.strip():
            return "الرجاء كتابة سؤالك."

        # ❶ تغنية الاستعلام بالسياق — مطابق تماماً للأصل
        query = user_input
        if self.memory and self.memory.needs_context(user_input):
            query = self.memory.enrich_query(user_input)

        # ❷ بحث في القاموس الثابت — مطابق تماماً للأصل، بدون أي تغيير
        answer, score = self._find(query)
        self._last_score = score

        if score >= self._kb_threshold():
            # ✅ إجابة القاموس — لا تغيير، نفس سلوك NSMChatPlus بالضبط
            self._last_source  = "kb"
            self.last_reasoning = None

        else:
            # ❸ ★ الجديد: نبني أثر التفكير (Few-shot + CKG + CoT) ★
            trace = self.cot.build_trace(query, history=self.history[-4:])
            self.last_reasoning = trace

            # نستخدم الـ prompt المُعزَّز فقط مع LLM حي — وإلا نترك السلوك
            # الأصلي (CKG synthesis على السؤال الخام) بدون أي تغيير
            effective_query = (
                trace.llm_prompt if self.fallback.has_live_llm() else query
            )

            result = self.fallback.generate(
                query=effective_query,
                history=self.history[-4:],
            )
            answer = result.text
            self._last_source = (
                "llm"
                if result.provider in (Provider.GROQ, Provider.OPENAI, Provider.HUGGINGFACE)
                else "ckg"
            )

        # ❹ حفظ في الذاكرة — مطابق تماماً للأصل
        if self.memory:
            self.memory.add(user_input, answer, self._last_topic)
        self.history.append((user_input, answer))
        return answer

    def chat_with_trace(self, user_input: str) -> Tuple[str, Optional[ReasoningTrace]]:
        """مثل chat() لكنه يُرجع أيضاً أثر التفكير الكامل — مفيد للواجهات/التصحيح."""
        reply = self.chat(user_input)
        return reply, self.last_reasoning

    # ── ملاحظة: إصلاح عدم تطابق الترميز (Encoding Mismatch) ──────────────────
    # كان هذا الإصلاح override محصوراً هنا فقط، وبطلب من المستخدم تم تعميمه
    # مباشرة في NSMChat._encode_raw (nsm_chat.py) — فأصبح NSMChat،
    # NSMChatPlus، و NSMChatCoT الثلاثة يستخدمون text_to_vec بدل ميزات
    # ArabicNLPEngine تلقائياً، دون حاجة لأي override هنا بعد الآن.

    # ── أدوات مساعدة ─────────────────────────────────────────────────────

    def _kb_threshold(self) -> float:
        """يقرأ نفس الحد المستخدم في NSMChatPlus دون تكراره يدوياً."""
        import nsm_chat_plus
        return nsm_chat_plus._KB_THRESHOLD

    def reasoning_display(self) -> str:
        """نص جاهز للعرض في الواجهة — خطوات التفكير لآخر سؤال (أو رسالة بديلة)."""
        if self.last_reasoning is None:
            return "📚 آخر إجابة جاءت من القاموس الثابت مباشرة (لا حاجة لتفكير إضافي)."
        return self.last_reasoning.to_display()


# ══════════════════════════════════════════════════════════════════
# CLI — مطابق لنفس أسلوب nsm_chat.py / nsm_chat_plus.py
# ══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  NSM Chat — CoT — Few-shot + Chain-of-Thought مُفعَّلان")
    print("=" * 60)

    ckg = None
    try:
        from knowledge.cognitive_graph import get_ckg
        ckg = get_ckg()
        print(f"✓ CKG محمّل: {len(ckg._concepts)} مفهوم")
    except Exception as exc:
        print(f"⚠ تعذّر تحميل CKG ({exc}) — سيعمل النظام بدونه")

    bot = NSMChatCoT(ckg=ckg)
    print("اكتب سؤالك. 'فكر' لعرض آخر خطوات تفكير. 'خروج' للإنهاء.\n")

    while True:
        try:
            u = input("أنت: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not u:
            continue
        if u.lower() in ("خروج", "exit", "quit"):
            print("NSM: مع السلامة!")
            break
        if u == "فكر":
            print(f"\n{bot.reasoning_display()}\n")
            continue

        reply = bot.chat(u)
        print(f"\nNSM [{bot.source_badge()}]: {reply}\n")


if __name__ == "__main__":
    main()
