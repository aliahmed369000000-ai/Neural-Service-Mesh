"""
NSMChat+ — امتداد NSMChat بطبقة التوليد الحقيقي
=================================================
Drop-in replacement لـ NSMChat يضيف:
  - LLM Generative Fallback عندما score < KB_THRESHOLD
  - شارة مصدر الإجابة: 📚 قاموس | 🤖 LLM | 🕸️ رسم معرفي
  - دعم Multi-turn context window كامل للـ LLM
  - احتفاظ كامل بتوافق الواجهة مع NSMChat الأصلي
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import nsm_chat as _nsm_chat_module
from nsm_chat import (
    NSMChat,
    _handle_code_command,
    _AGENT_TRIGGERS,
)
from ai.llm_fallback import LLMFallback, Provider

logger = logging.getLogger("NSMChatPlus")

# الحد الأدنى لقبول إجابة مباشرة من القاموس الثابت.
# الأصل في NSMChat هو 0.12 — نرفعه قليلاً لتوسيع نطاق تدخّل الـ LLM.
_KB_THRESHOLD = 0.18

_SOURCE_BADGES: Dict[str, str] = {
    "kb":  "📚 قاموس NSM",
    "llm": "🤖 LLM",
    "ckg": "🕸️ رسم معرفي",
}


class NSMChatPlus(NSMChat):
    """
    NSMChat مُعزَّز بطبقة توليد LLM حقيقي.

    التسلسل عند كل سؤال:
      1. يُثري الاستعلام بالسياق (مثل NSMChat الأصلي)
      2. يبحث في _KB (keyword + embedding cosine)
      3. إذا score >= KB_THRESHOLD → إجابة القاموس (سريع، دقيق)
      4. إذا score < KB_THRESHOLD  → LLMFallback:
           a. Groq / OpenAI / HuggingFace  إذا وُجد GROQ_API_KEY أو ما يعادله
           b. CKG Synthesis                 إذا لا مفتاح (يولّد من cognitive_graph.json)

    الاستخدام المباشر:
        bot = NSMChatPlus()
        reply = bot.chat("ما حكم الصلاة في الإسلام؟")
        print(reply)
        print(bot.source_badge())   # "📚 قاموس NSM" أو "🤖 llama-3.1-8b-instant" ...
    """

    def __init__(self, ckg=None):
        super().__init__()
        self._ckg         = ckg
        self.fallback     = LLMFallback(ckg=ckg)
        self._last_source = "kb"
        self._last_score  = 0.0
        logger.info(
            f"[NSMChatPlus] جاهز | fallback: {self.fallback.provider.value}"
            f" | نموذج: {self.fallback.model}"
        )

    # ── override chat ────────────────────────────────────────────────────

    def chat(self, user_input: str) -> str:
        if not user_input.strip():
            return "الرجاء كتابة سؤالك."

        # ❶ NSM Agent الذكي (Groq) — أولوية 1 للطلبات الطبيعية
        if _nsm_chat_module._HAS_NSM_AGENT and _nsm_chat_module._nsm_agent and any(
            user_input.strip().startswith(t) for t in _AGENT_TRIGGERS
        ):
            response = _nsm_chat_module._nsm_agent.run(user_input)
            self._last_source = "nsm_agent:groq"
            self.history.append((user_input, response))
            return response

        # ❷ Code Agent المباشر — أولوية 2 للأوامر الدقيقة (افحص/قائمة/ارفع)
        agent_response = _handle_code_command(user_input)
        if agent_response is not None:
            self._last_source = "code_agent"
            self.history.append((user_input, agent_response))
            return agent_response

        # ❸ تغنية الاستعلام بالسياق (pronoun resolution)
        query = user_input
        if self.memory and self.memory.needs_context(user_input):
            query = self.memory.enrich_query(user_input)

        # ❹ بحث في القاموس الثابت
        answer, score = self._find(query)
        self._last_score = score

        if score >= _KB_THRESHOLD:
            # ✅ إجابة جيدة من القاموس
            self._last_source = "kb"

        else:
            # ❺ اللجوء إلى طبقة التوليد
            result = self.fallback.generate(
                query=query,
                history=self.history[-4:],
            )
            answer = result.text
            self._last_source = (
                "llm"
                if result.provider in (
                    Provider.GROQ,
                    Provider.OPENAI,
                    Provider.TOGETHER,
                    Provider.GEMINI,
                )
                else "ckg"
            )
            # إلحاق بيان الكمون إذا كان LLM حقيقياً
            if result.latency_ms and self._last_source == "llm":
                logger.debug(
                    f"[NSMChatPlus] LLM latency: {result.latency_ms}ms"
                )

        # ❻ حفظ في الذاكرة
        if self.memory:
            self.memory.add(user_input, answer, self._last_topic)
        self.history.append((user_input, answer))
        return answer

    # ── معلومات الإجابة ──────────────────────────────────────────────────

    @property
    def last_source(self) -> str:
        """مصدر آخر إجابة: 'kb' | 'llm' | 'ckg'"""
        return self._last_source

    @property
    def last_score(self) -> float:
        """نتيجة cosine similarity لآخر بحث في القاموس"""
        return self._last_score

    def source_badge(self) -> str:
        """شارة نصية لعرض مصدر الإجابة في الواجهة"""
        if self._last_source == "llm":
            return f"🤖 {self.fallback.model}"
        return _SOURCE_BADGES.get(self._last_source, "❓")

    def fallback_info(self) -> Dict[str, str]:
        """معلومات المزوّد الحالي للعرض في لوحة الإعدادات"""
        return self.fallback.info()

    def is_generative(self) -> bool:
        """هل النظام في وضع التوليد الحقيقي (LLM حي)؟"""
        return self.fallback.has_live_llm()
