"""
NSMChat+ — امتداد NSMChat بطبقة التوليد الحقيقي
=================================================
Drop-in replacement لـ NSMChat يضيف:
  - كل الردود تمر عبر NSM Agent / Code Agent / LLM حصراً — لا قاموس ثابت
  - شارة مصدر الإجابة: 🧠 NSM Agent | 🛠️ Code Agent | 🤖 LLM | 🕸️ رسم معرفي
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
from ai.llm_fallback import LLMFallback, Provider, LIVE_LLM_PROVIDERS

logger = logging.getLogger("NSMChatPlus")

# ⚠️ غير مُستخدَم في هذا الملف بعد حذف القاموس — أُبقي عليه فقط لأن
# nsm_chat_cot.py القديم يستورده. لا تأثير له على مسار الردود هنا.
_KB_THRESHOLD = 0.18

_SOURCE_BADGES: Dict[str, str] = {
    "llm":       "🤖 LLM",
    "ckg":       "🕸️ رسم معرفي",
    "nsm_agent": "🧠 NSM Agent",
    "code_agent":"🛠️ Code Agent",
}


class NSMChatPlus(NSMChat):
    """
    NSMChat مُعزَّز بطبقة توليد LLM حقيقي.

    التسلسل عند كل سؤال (لا يوجد قاموس ثابت — أُزيل نهائياً):
      1. NSM Agent (أوامر بناء/تعديل عبر _AGENT_TRIGGERS)
      2. Code Agent (أوامر دقيقة: افحص/قائمة/ارفع/عدل)
      3. LLMFallback مباشرة لكل ما تبقى:
           a. Cloudflare / Groq / Gemini / OpenRouter / OpenAI حسب المتاح
           b. CKG Synthesis فقط إذا لا يوجد أي مفتاح API

    الاستخدام المباشر:
        bot = NSMChatPlus()
        reply = bot.chat("ما حكم الصلاة في الإسلام؟")
        print(reply)
        print(bot.source_badge())   # "🧠 NSM Agent" أو "🤖 llama-3.1-8b-instant" ...
    """

    def __init__(self, ckg=None, system_prompt: str = None):
        super().__init__()
        self._ckg           = ckg
        self.fallback       = LLMFallback(ckg=ckg)
        self._last_source   = "nsm_agent"
        self._last_score    = 0.0
        self._system_prompt = system_prompt  # NSM_SYSTEM_PROMPT يُمرَّر من streamlit_app.py
        logger.info(
            f"[NSMChatPlus] جاهز | fallback: {self.fallback.provider.value}"
            f" | نموذج: {self.fallback.model}"
            f" | system_prompt: {'مخصص' if system_prompt else 'افتراضي'}"
        )

    # ── override chat ────────────────────────────────────────────────────

    def chat(self, user_input: str, system_prompt: str = None) -> str:
        if not user_input.strip():
            return "الرجاء كتابة سؤالك."

        # ❶ NSM Agent الذكي — أولوية 1 للطلبات البرمجية
        if _nsm_chat_module._HAS_NSM_AGENT and _nsm_chat_module._nsm_agent and any(
            user_input.strip().startswith(t) for t in _AGENT_TRIGGERS
        ):
            response = _nsm_chat_module._nsm_agent.run(user_input)
            self._last_source = "nsm_agent"
            self.history.append((user_input, response))
            return response

        # ❷ Code Agent المباشر — أولوية 2 للأوامر الدقيقة (افحص/قائمة/ارفع)
        agent_response = _handle_code_command(user_input)
        if agent_response is not None:
            self._last_source = "code_agent"
            self.history.append((user_input, agent_response))
            return agent_response

        # ❸ تغنية الاستعلام بالسياق (pronoun resolution + ذاكرة الحقائق)
        query = user_input
        if self.memory and self.memory.needs_context(user_input):
            query = self.memory.enrich_query(user_input)

        # ❸.5 حقن كتلة الذاكرة القوية (حقائق + محادثات ذات صلة دلالياً)
        #     — best-effort تماماً: أي خطأ يُتجاهل ولا يوقف الرد أبداً
        if self.memory:
            try:
                mem_block = self.memory.build_memory_context(user_input)
                if mem_block:
                    query = f"[{mem_block}]\n{query}"
            except Exception as _mem_err:
                logger.debug(f"memory context injection skipped: {_mem_err}")

        # ❸.6 تثبيت قرآني/معرفي — نفس الآلية الموجودة أصلاً في nsm_chat.py
        # (الفئة الأساسية) لكنها لم تكن مطبَّقة هنا: NSMChatPlus.chat()
        # يُعيد كتابة chat() بالكامل (override) دون استدعاء super().chat()،
        # وهي الفئة الفعلية المستخدمة بالواجهة الحية (`from nsm_chat_plus
        # import NSMChatPlus as NSMChat`) — أما NSMChat الأصلية بتثبيتها
        # فتُستخدَم فقط كـ fallback نادر عند فشل استيراد nsm_chat_plus.
        # النتيجة العملية قبل هذا الإصلاح: التثبيت موجود بالكود لكن لا
        # يعمل أبداً على المسار الحقيقي الذي يمر به كل سؤال مستخدم.
        _context_blocks: List[str] = []
        if _nsm_chat_module._HAS_QURAN_GROUNDING and _nsm_chat_module._build_quran_context is not None:
            try:
                _qctx = _nsm_chat_module._build_quran_context(user_input)
            except Exception:
                _qctx = None
            if _qctx:
                _context_blocks.append(_qctx)
        if _nsm_chat_module._HAS_DOMAIN_GROUNDING and _nsm_chat_module._search_domain is not None:
            try:
                _dmatches = _nsm_chat_module._search_domain(user_input, limit=3)
            except Exception:
                _dmatches = []
            if _dmatches:
                _refs = "\n".join(
                    f"- [{m['domain_ar']}] {m['concept']}: {m['text']}"
                    for m in _dmatches
                )
                _context_blocks.append(
                    "[معلومات مرجعية دقيقة من قاعدة معرفة NSM التعليمية — "
                    "استخدمها إن كانت ذات صلة]\n" + _refs
                )
        if _context_blocks:
            query = "\n\n".join(_context_blocks) + f"\n\n[سؤال المستخدم]\n{query}"

        # ❹ LLM مباشرة — القاموس محذوف من مسار الردود
        # الأولوية: system_prompt الممرَّر في الاستدعاء → self._system_prompt → الافتراضي
        _sp = system_prompt or self._system_prompt
        result = self.fallback.generate(
            query=query,
            history=self.history[-4:],
            system_prompt=_sp,
        )
        answer = result.text
        self._last_source = (
            "llm" if result.provider in LIVE_LLM_PROVIDERS else "ckg"
        )

        # ❺ حفظ في الذاكرة
        if self.memory:
            self.memory.add(user_input, answer, self._last_topic)
            # استخلاص حقائق جديدة من هذا الحوار (Mem0-style) — لا يوقف الرد أبداً
            try:
                self.memory.extract_and_remember_facts(
                    user_input, answer, llm_fallback=self.fallback
                )
            except Exception as _fact_err:
                logger.debug(f"fact extraction skipped: {_fact_err}")
        self.history.append((user_input, answer))
        return answer

    # ── معلومات الإجابة ──────────────────────────────────────────────────

    @property
    def last_source(self) -> str:
        """مصدر آخر إجابة: 'nsm_agent' | 'code_agent' | 'llm' | 'ckg'"""
        return self._last_source

    @property
    def last_score(self) -> float:
        """محجوزة للتوافق الخلفي — غير مُستخدَمة بعد حذف القاموس"""
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
