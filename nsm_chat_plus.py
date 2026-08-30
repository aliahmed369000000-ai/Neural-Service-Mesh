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
import re
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("NSMChatPlus")

import nsm_chat as _nsm_chat_module
from nsm_chat import (
    NSMChat,
    _handle_code_command,
    _AGENT_TRIGGERS,
)
from ai.llm_fallback import LLMFallback, Provider, LIVE_LLM_PROVIDERS

try:
    from ai.nova_search_copyright import is_song_lyrics_request, SONG_LYRICS_REFUSAL
    _HAS_COPYRIGHT_CHECK = True
except Exception:
    _HAS_COPYRIGHT_CHECK = False
    is_song_lyrics_request = None
    SONG_LYRICS_REFUSAL = ""

# 🆕 الحزمة 4: الذاكرة الدلالية للمحادثات — Qdrant أولًا ثم SQLite محلي
try:
    from ai.qdrant_semantic_memory import QdrantSemanticMemory
    _QDRANT_MEM: Optional[QdrantSemanticMemory] = QdrantSemanticMemory()
except Exception as _qsm_init_err:
    logger.debug(f"[NSMChatPlus] الذاكرة الدلالية غير متاحة: {_qsm_init_err}")
    _QDRANT_MEM = None

# ⚠️ غير مُستخدَم في هذا الملف بعد حذف القاموس — أُبقي عليه فقط لأن
# nsm_chat_cot.py القديم يستورده. لا تأثير له على مسار الردود هنا.
_KB_THRESHOLD = 0.18

_SOURCE_BADGES: Dict[str, str] = {
    "llm":       "🤖 LLM",
    "ckg":       "🕸️ رسم معرفي",
    "nsm_agent": "🧠 NSM Agent",
    "code_agent":"🛠️ Code Agent",
    "copyright_guard": "©️ حقوق نشر",
}

# عبارات صريحة تسمح بتمرير الطلب إلى الوكيل حتى إن لم تبدأ الجملة بفعل
# موجود في _AGENT_TRIGGERS، مع إبقاء التوجيه محافظاً لتجنب تنفيذ عرضي.
_AGENT_INTENT_RE = re.compile(
    r"(?:نفّذ|نفذ|شغّل|شغل|طبّق|طبق|اختبر|حلّل|حلل|افحص|عدّل|عدل|"""
    r"أنشئ|انشئ|طوّر|طور|أضف|اضف|ارفع|صحّح|صحح)\b",
    re.IGNORECASE,
)

_MAX_PROMPT_HISTORY_TURNS = 8
_MAX_PROMPT_CHARS = 18000



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
        self._turn_count = 0
        self._last_latency_ms = 0.0
        self._last_metadata: Dict[str, Any] = {}
        logger.info(
            f"[NSMChatPlus] جاهز | fallback: {self.fallback.provider.value}"
            f" | نموذج: {self.fallback.model}"
            f" | system_prompt: {'مخصص' if system_prompt else 'افتراضي'}"
        )

    # ── أدوات سياق وتوجيه مشتركة ─────────────────────────────────────────

    @staticmethod
    def _is_agent_request(user_input: str) -> bool:
        """تحديد محافظ لطلبات التنفيذ البرمجية أو التحليلية العميقة."""
        text = user_input.strip()
        if any(text.startswith(trigger) for trigger in _AGENT_TRIGGERS):
            return True
        # لا نستخدم البحث الجزئي إلا مع أفعال تنفيذ صريحة في بداية الطلب؛
        # ذلك يمنع تحويل سؤال عادي يحتوي كلمة «حلل» عرضاً إلى تنفيذ.
        return bool(_AGENT_INTENT_RE.match(text))

    def _history_for_prompt(self) -> List[Tuple[str, str]]:
        """يبني سياقاً مضبوط الحجم مع الحفاظ على أحدث الأدوار كاملة."""
        turns = list(self.history[-_MAX_PROMPT_HISTORY_TURNS:])
        if not turns:
            return []
        selected: List[Tuple[str, str]] = []
        chars = 0
        for user_text, assistant_text in reversed(turns):
            pair_chars = len(user_text) + len(assistant_text)
            if selected and chars + pair_chars > _MAX_PROMPT_CHARS:
                break
            selected.append((user_text, assistant_text))
            chars += pair_chars
        return list(reversed(selected))

    def _set_metadata(self, started: float, *, source: str,
                      used_memory: bool = False, route: str = "chat") -> None:
        self._turn_count += 1
        self._last_latency_ms = round((time.perf_counter() - started) * 1000, 1)
        self._last_metadata = {
            "turn": self._turn_count,
            "source": source,
            "route": route,
            "used_memory": used_memory,
            "latency_ms": self._last_latency_ms,
            "history_turns": len(self.history),
        }

    @property
    def last_metadata(self) -> Dict[str, Any]:
        """بيانات قابلة للعرض في لوحة المراقبة دون كشف محتوى سري."""
        return dict(self._last_metadata)

    # ── override chat ────────────────────────────────────────────────────

    def chat(self, user_input: str, system_prompt: str = None) -> str:
        started = time.perf_counter()
        if not user_input.strip():
            self._set_metadata(started, source="validation", route="chat")
            return "الرجاء كتابة سؤالك."

        # ⓪ حقوق النشر — رفض استباقي لطلبات كلمات أغاني/قصائد كاملة (nova_search_copyright.py
        # كانت وحدة يتيمة غير مستوردة إطلاقاً؛ الوظيفة البرمجية الحقيقية الوحيدة منها
        # المُوصولة هنا: is_song_lyrics_request(). تُستبعد عمداً check_response_copyright()
        # لأنها تتعارض مع تثبيت القرآن أعلاه — آيات طويلة موثوقة ليست انتهاكاً.
        if _HAS_COPYRIGHT_CHECK and is_song_lyrics_request(user_input):
            self._last_source = "copyright_guard"
            self.history.append((user_input, SONG_LYRICS_REFUSAL))
            self._set_metadata(started, source="copyright_guard", route="guard")
            return SONG_LYRICS_REFUSAL

        # ❶ NSM Agent الذكي — أولوية 1 للطلبات البرمجية
        if _nsm_chat_module._HAS_NSM_AGENT and _nsm_chat_module._nsm_agent and self._is_agent_request(user_input):
            response = _nsm_chat_module._nsm_agent.run(user_input)
            self._last_source = "nsm_agent"
            self.history.append((user_input, response))
            self._set_metadata(started, source="nsm_agent", route="agent")
            return response

        # ❷ Code Agent المباشر — أولوية 2 للأوامر الدقيقة (افحص/قائمة/ارفع)
        agent_response = _handle_code_command(user_input)
        if agent_response is not None:
            self._last_source = "code_agent"
            self.history.append((user_input, agent_response))
            self._set_metadata(started, source="code_agent", route="code")
            return agent_response

        # 🆕 الحزمة 4: الذاكرة الدلالية — استحضار أقرب محادثات سابقة ذات صلة
        # بالسياق (Qdrant/bge-m3 عربي، وعند تعذّره ترتيب محلي TF عربي)
        # لتحسين دقة الرد في المواضيع المتكررة.
        _semantic_context = ""
        _used_semantic_memory = False
        if _QDRANT_MEM is not None:
            try:
                _convo_hits = _QDRANT_MEM.search_conversations(user_input)
                if _convo_hits:
                    _hits_text = []
                    for _sc, _sp in _convo_hits:
                        _hits_text.append(
                            f"سؤال سابق مشابه: {_sp.get('user_text','')[:300]}\n"
                            f"جوابه السابق: {_sp.get('assistant_text','')[:500]}"
                        )
                    _semantic_context = (
                        "محادثات سابقة ذات صلة بالسؤال الحالي:\n"
                        + "\n---\n".join(_hits_text)
                    )
                    _used_semantic_memory = True
            except Exception as _sc_err:
                logger.debug(f"[NSMChatPlus] استحضار دلالي فاشل (صامت): {_sc_err}")

        # ❸ تغنية الاستعلام بالسياق (pronoun resolution + ذاكرة الحقائق)
        query = user_input
        used_memory = _used_semantic_memory
        if self.memory and self.memory.needs_context(user_input):
            query = self.memory.enrich_query(user_input)
            used_memory = True

        # ❸.5 حقن كتلة الذاكرة القوية (حقائق + محادثات ذات صلة دلالياً)
        #     — best-effort تماماً: أي خطأ يُتجاهل ولا يوقف الرد أبداً
        if self.memory:
            try:
                mem_block = self.memory.build_memory_context(user_input)
                if mem_block:
                    query = f"[{mem_block}]\n{query}"
                    used_memory = True
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
        # 🆕 الحزمة 4: إلحاق المحادثات الدلالية المستحضرة كسياق إضافي
        if _semantic_context and _context_blocks:
            _context_blocks.append(_semantic_context)
        elif _semantic_context:
            query = _semantic_context + f"\n\n[سؤال المستخدم]\n{query}"
        if _context_blocks:
            query = "\n\n".join(_context_blocks) + f"\n\n[سؤال المستخدم]\n{query}"

        # ❹ LLM مباشرة — القاموس محذوف من مسار الردود
        # الأولوية: system_prompt الممرَّر في الاستدعاء → self._system_prompt → الافتراضي
        _sp = system_prompt or self._system_prompt
        result = self.fallback.generate(
            query=query,
            history=self._history_for_prompt(),
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
        self._set_metadata(
            started,
            source=self._last_source,
            used_memory=used_memory or _used_semantic_memory,
            route="llm" if self._last_source in {"llm", "ckg"} else "chat",
        )
        # 🆕 الحزمة 4: حفظ المحادثة الحالية في الذاكرة الدلالية (Qdrant أولًا
        # ثم SQLite محلي) — score هنا درجة جودة الرد المتاحة من _last_score
        if _QDRANT_MEM is not None and (user_input.strip() and answer.strip()):
            try:
                _QDRANT_MEM.add_conversation(
                    convo_id=f"conv_{int(time.time()*1000)}_{self._turn_count}",
                    user_text=user_input,
                    assistant_text=answer,
                    relevance=float(self._last_score or 0.0),
                )
            except Exception as _save_err:
                logger.debug(f"[NSMChatPlus] حفظ دلالي فاشل (صامت): {_save_err}")
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
