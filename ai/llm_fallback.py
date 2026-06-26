"""
LLM Generative Fallback Engine — NSM v18.2
============================================
يوفر طبقة توليد نصي حقيقي عندما لا يجد NSMChat إجابة كافية في قاموسه الثابت.

الأولوية في اختيار المزوّد (auto-detect من env vars):
  1. OpenAI API      (OPENAI_API_KEY)   — GPT-4o-mini  ✅ متاح من Replit
  2. Together.xyz    (TOGETHER_API_KEY) — Llama-3/Mixtral مجاني  ✅ متاح من Replit
  3. Google Gemini   (GOOGLE_API_KEY)   — Gemini 1.5 Flash  ✅ متاح من Replit
  4. CKG Synthesis   (بدون مفتاح)      — يولّد من الرسم المعرفي دائماً

ملاحظة: Groq (GROQ_API_KEY) محجوب من Replit IPs عبر Cloudflare (error 1010).
         للاستخدام على Streamlit Community Cloud أو server خاص — يعمل بلا مشكلة.

الاستخدام:
    from ai.llm_fallback import LLMFallback

    fb = LLMFallback(ckg=my_ckg_instance)
    result = fb.generate("ما هو مفهوم التوحيد في الإسلام؟", history=[...])
    print(result.text)
    print(result.provider.value)  # "openai" | "together" | "gemini" | "ckg_synthesis"
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("LLMFallback")


# ════════════════════════════════════════════════════════════════════════════
# Provider Enum
# ════════════════════════════════════════════════════════════════════════════

class Provider(Enum):
    OPENAI    = "openai"
    TOGETHER  = "together"
    GEMINI    = "gemini"
    GROQ      = "groq"        # متاح خارج Replit فقط
    CKG_SYNTH = "ckg_synthesis"


# ════════════════════════════════════════════════════════════════════════════
# System Prompt المتخصص في المعرفة العربية الإسلامية
# ════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "أنت NSM (Neural Service Mesh) — مساعد ذكاء اصطناعي عربي متخصص في:\n"
    "- المعرفة الإسلامية: القرآن الكريم، الحديث النبوي الشريف، العقيدة، الفقه، التاريخ الإسلامي\n"
    "- اللغة العربية: نحو، صرف، بلاغة، أدب\n"
    "- العلوم والتكنولوجيا والرياضيات باللغة العربية\n\n"
    "قواعد الإجابة:\n"
    "1. أجب دائماً بالعربية الفصحى الواضحة والمختصرة (3-5 جمل كحد أقصى)\n"
    "2. للمسائل الشرعية، استند للقرآن والسنة الصحيحة مع ذكر المصدر\n"
    "3. إذا لم تعرف الإجابة، قل ذلك بصراحة ولا تتخمّن\n"
    "4. لا تُشر إلى نفسك كـ GPT أو Claude أو أي نموذج آخر — أنت NSM"
)

_OPENAI_MODEL   = "gpt-4o-mini"
_TOGETHER_MODEL = "meta-llama/Llama-3-8b-chat-hf"
_GEMINI_MODEL   = "gemini-1.5-flash"
_GROQ_MODELS    = ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]


# ════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class FallbackResult:
    text:       str
    provider:   Provider
    model:      str   = ""
    latency_ms: float = 0.0
    error:      Optional[str] = None


# ════════════════════════════════════════════════════════════════════════════
# HTTP helper (بدون مكتبات خارجية)
# ════════════════════════════════════════════════════════════════════════════

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 15) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req  = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ════════════════════════════════════════════════════════════════════════════
# CKG Synthesis — توليد من الرسم المعرفي بدون LLM خارجي
# ════════════════════════════════════════════════════════════════════════════

def _ckg_synthesize(query: str, ckg) -> str:
    """
    يبني إجابة من cognitive_graph.json بدون LLM.
    الخوارزمية: استخراج كلمات مفتاحية → بحث في CKG → تركيب جملة عربية.
    """
    if ckg is None:
        return _generic_fallback()
    try:
        stop_words = {
            "هل", "ما", "من", "في", "عن", "على", "إلى", "هو", "هي",
            "كيف", "لماذا", "متى", "أين", "ماذا", "التي", "الذي",
        }
        words = [
            w.strip("؟.,!:;") for w in query.split()
            if len(w) > 2 and w not in stop_words
        ]

        candidates: Dict[str, float] = {}
        for word in words[:6]:
            try:
                for name, weight in ckg.query_related(word, top_k=5):
                    candidates[name] = max(candidates.get(name, 0.0), weight)
            except Exception:
                pass

        if not candidates:
            return _generic_fallback()

        ranked   = sorted(candidates.items(), key=lambda x: -x[1])[:8]
        top      = [n for n, _ in ranked]

        clusters: Dict[str, str] = {}
        for name, _ in ranked[:4]:
            c = ckg._concepts.get(name)
            if c and c.cluster:
                clusters[name] = c.cluster

        core = "، ".join(top[:3])
        ans  = f"يرتبط سؤالك بالمفاهيم المعرفية التالية: {core}."
        if len(top) > 3:
            ans += f" كما يتصل بـ: {' ، '.join(top[3:6])}."
        unique_cl = list(set(clusters.values()))
        if len(unique_cl) == 1:
            ans += f" هذه المفاهيم تنتمي إلى مجال: {unique_cl[0]}."
        elif unique_cl:
            ans += f" تغطي مجالات: {' | '.join(unique_cl[:3])}."
        ans += " (مُستخلَص من الرسم المعرفي — للحصول على إجابة أدق أضف OPENAI_API_KEY)"
        return ans
    except Exception as exc:
        logger.warning(f"[CKGSynth] {exc}")
        return _generic_fallback()


def _generic_fallback() -> str:
    return (
        "سؤالك خارج نطاق معرفتي المباشرة حالياً. "
        "يمكنني المساعدة في: الإسلام والقرآن الكريم، الذكاء الاصطناعي، "
        "الرياضيات، اللغة العربية، التاريخ الإسلامي، والبرمجة. "
        "لتفعيل التوليد الكامل، أضف OPENAI_API_KEY أو TOGETHER_API_KEY في الـ Secrets."
    )


# ════════════════════════════════════════════════════════════════════════════
# LLMFallback — المحرك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class LLMFallback:
    """
    طبقة التوليد الذكي. تُفعَّل عند score < threshold في NSMChat.

    أولوية المزوّدين (حسب الإمكانية من Replit):
      1. OpenAI   (OPENAI_API_KEY)   ← أفضل جودة
      2. Together (TOGETHER_API_KEY) ← مجاني، نماذج مفتوحة
      3. Gemini   (GOOGLE_API_KEY)   ← سريع ومجاني
      4. Groq     (GROQ_API_KEY)     ← يعمل خارج Replit فقط
      5. CKG Synthesis               ← دائماً متاح

    مثال:
        fb = LLMFallback(ckg=my_ckg)
        r  = fb.generate("ما حكم الزكاة في الإسلام؟", history=[...])
        print(r.text, r.provider.value, r.latency_ms)
    """

    def __init__(
        self,
        ckg=None,
        max_tokens:  int   = 350,
        temperature: float = 0.4,
        timeout:     int   = 14,
    ):
        self.ckg         = ckg
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout

        self._provider, self._api_key, self._model = self._detect_provider()
        logger.info(
            f"[LLMFallback] مزوّد: {self._provider.value} | نموذج: {self._model}"
        )

    # ── اكتشاف المزوّد تلقائياً ─────────────────────────────────────────

    def _detect_provider(self) -> Tuple[Provider, str, str]:
        # 1) Groq — الأسرع والمجاني، يعمل على Streamlit Cloud
        k = os.getenv("GROQ_API_KEY", "").strip()
        if k:
            return Provider.GROQ, k, _GROQ_MODELS[0]

        # 2) OpenAI
        k = os.getenv("OPENAI_API_KEY", "").strip()
        if k:
            return Provider.OPENAI, k, _OPENAI_MODEL

        # 3) Together.xyz — مجاني
        k = os.getenv("TOGETHER_API_KEY", "").strip()
        if k:
            return Provider.TOGETHER, k, _TOGETHER_MODEL

        # 4) Google Gemini — مجاني
        k = os.getenv("GOOGLE_API_KEY", "").strip()
        if k:
            return Provider.GEMINI, k, _GEMINI_MODEL

        # 5) CKG Synthesis — لا مفتاح مطلوب
        return Provider.CKG_SYNTH, "", "ckg-synthesis-v1"

    # ── الواجهة العامة ───────────────────────────────────────────────────

    def generate(
        self,
        query:   str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> FallbackResult:
        """
        يولّد إجابة للاستعلام مع السياق متعدد الأدوار.

        Args:
            query:   نص سؤال المستخدم
            history: [(user_msg, bot_msg), ...] آخر N رسائل
        Returns:
            FallbackResult
        """
        # إعادة فحص المفتاح (يدعم الحقن المتأخر من Streamlit Secrets)
        if not self._api_key or self._provider.value == "ckg_synthesis":
            self._provider, self._api_key, self._model = self._detect_provider()

        t0      = time.time()
        history = history or []

        try:
            if self._provider == Provider.OPENAI:
                result = self._call_openai(query, history)
            elif self._provider == Provider.TOGETHER:
                result = self._call_together(query, history)
            elif self._provider == Provider.GEMINI:
                result = self._call_gemini(query, history)
            elif self._provider == Provider.GROQ:
                result = self._call_groq(query, history)
            else:
                text   = _ckg_synthesize(query, self.ckg)
                result = FallbackResult(
                    text=text, provider=Provider.CKG_SYNTH, model=self._model
                )
        except Exception as exc:
            logger.error(f"[LLMFallback] {self._provider.value} فشل: {exc}")
            result = FallbackResult(
                text=_ckg_synthesize(query, self.ckg),
                provider=Provider.CKG_SYNTH,
                model="ckg-synthesis-v1",
                error=str(exc),
            )

        result.latency_ms = round((time.time() - t0) * 1000, 1)
        return result

    # ── خصائص ───────────────────────────────────────────────────────────

    @property
    def provider(self) -> Provider:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    @property
    def available(self) -> bool:
        """هل يوجد مزوّد LLM حقيقي متاح؟ (يُستخدم من nsm_chat.py)"""
        return self._provider != Provider.CKG_SYNTH and bool(self._api_key)

    def has_live_llm(self) -> bool:
        """هل يوجد LLM حقيقي يعمل (وليس CKG synthesis فقط)؟"""
        return self._provider != Provider.CKG_SYNTH

    def info(self) -> Dict[str, str]:
        return {
            "provider": self._provider.value,
            "model":    self._model,
            "live_llm": "✅" if self.has_live_llm() else "❌ (CKG synthesis)",
            "api_key":  "✅ موجود" if self._api_key else "❌ غير موجود",
        }

    # ── OpenAI ───────────────────────────────────────────────────────────

    def _call_openai(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            {
                "model":       self._model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
            },
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            self.timeout,
        )
        return FallbackResult(
            text=data["choices"][0]["message"]["content"].strip(),
            provider=Provider.OPENAI,
            model=self._model,
        )

    # ── Together.xyz ─────────────────────────────────────────────────────

    def _call_together(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        data = _post_json(
            "https://api.together.xyz/v1/chat/completions",
            {
                "model":       self._model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
            },
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            self.timeout,
        )
        return FallbackResult(
            text=data["choices"][0]["message"]["content"].strip(),
            provider=Provider.TOGETHER,
            model=self._model,
        )

    # ── Google Gemini ─────────────────────────────────────────────────────

    def _call_gemini(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        # بناء تاريخ المحادثة بصيغة Gemini
        contents = []
        for u, a in history[-4:]:
            contents += [
                {"role": "user",  "parts": [{"text": u}]},
                {"role": "model", "parts": [{"text": a}]},
            ]
        contents.append({"role": "user", "parts": [{"text": query}]})

        url  = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self._model}:generateContent?key={self._api_key}"
        )
        body = {
            "system_instruction": {"parts": [{"text": _SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": self.max_tokens,
                "temperature":     self.temperature,
            },
        }
        data = _post_json(url, body, {"Content-Type": "application/json"}, self.timeout)
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return FallbackResult(
            text=text, provider=Provider.GEMINI, model=self._model
        )

    # ── Groq ────────────────────────────────────────────────────────────

    def _call_groq(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        # نماذج بديلة عند 403
        groq_models = [
            self._model,
            "llama3-8b-8192",
            "gemma2-9b-it",
            "llama-3.3-70b-versatile",
        ]
        # إزالة المكررات مع الحفاظ على الترتيب
        seen = set()
        groq_models = [m for m in groq_models if not (m in seen or seen.add(m))]

        last_err = None
        for model in groq_models:
            try:
                data = _post_json(
                    "https://api.groq.com/openai/v1/chat/completions",
                    {
                        "model":       model,
                        "messages":    messages,
                        "max_tokens":  self.max_tokens,
                        "temperature": self.temperature,
                        "stream":      False,
                    },
                    {
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type":  "application/json",
                    },
                    self.timeout,
                )
                return FallbackResult(
                    text=data["choices"][0]["message"]["content"].strip(),
                    provider=Provider.GROQ,
                    model=model,
                )
            except urllib.error.HTTPError as e:
                if e.code == 403:
                    last_err = f"403 على {model}"
                    continue
                raise
            except Exception as e:
                last_err = str(e)
                continue

        raise Exception(f"فشلت كل نماذج Groq: {last_err}")
