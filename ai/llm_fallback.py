"""
LLM Generative Fallback Engine — NSM v18.3
============================================
يوفر طبقة توليد نصي حقيقي عندما لا يجد NSMChat إجابة كافية في قاموسه الثابت.

الأولوية في اختيار المزوّد (auto-detect من env vars):
  1. Anthropic Claude (ANTHROPIC_API_KEY) — Claude Sonnet 5 ← الأولوية الأولى ✅
  2. Cloudflare Workers AI (CF_API_TOKEN + CF_ACCOUNT_ID) — مجاني 10k/يوم
  3. Google Gemini   (GOOGLE_API_KEY)   — Gemini 2.5 Flash
  4. OpenRouter      (OPENROUTER_API_KEY)
  5. Groq            (GROQ_API_KEY)     — قد يُحجب من بعض الشبكات
  6. OpenAI API      (OPENAI_API_KEY)   — GPT-4o-mini
  7. Together.xyz    (TOGETHER_API_KEY) — Llama-3/Mixtral
  8. CKG Synthesis   (بدون مفتاح)      — يولّد من الرسم المعرفي دائماً

الاستخدام:
    from ai.llm_fallback import LLMFallback

    fb = LLMFallback(ckg=my_ckg_instance)
    result = fb.generate("ما هو مفهوم التوحيد في الإسلام؟", history=[...])
    print(result.text)
    print(result.provider.value)  # "anthropic" | "cloudflare" | "gemini" | ... | "ckg_synthesis"
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
    ANTHROPIC = "anthropic"    # Claude — الأولوية الأولى ✅
    CLOUDFLARE = "cloudflare"  # مجاني 10k/يوم ويعمل من اليمن ✅
    GEMINI    = "gemini"
    OPENROUTER = "openrouter"
    OPENAI    = "openai"
    TOGETHER  = "together"
    GROQ      = "groq"
    CKG_SYNTH = "ckg_synthesis"


# مصدر وحيد للحقيقة لكل المزوّدين "الحيّين" (ليسوا CKG synthesis).
# استخدم هذا في أي مكان بالمشروع بدل كتابة قائمة يدوية جديدة، لتفادي
# نسيان مزوّد جديد (كما حدث سابقاً مع Provider.CLOUDFLARE).
LIVE_LLM_PROVIDERS = frozenset(
    p for p in Provider if p is not Provider.CKG_SYNTH
)


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
    "4. تصرّف بشكل طبيعي باسم NSM ضمن سياق المنتج، دون التطوّع بتفاصيل البنية التقنية الداخلية ما لم يُسأل عنها مباشرة\n"
    "5. إذا سُئلت بجدية ومباشرة عن النموذج الأساسي الذي تعمل به، أجب بصدق ولا تنفِ ذلك\n"
    "6. الصدق أهم من الحفاظ على شخصية العلامة التجارية؛ عند التعارض، الصدق يُقدَّم دائماً"
)

# ── أسماء النماذج المحدّثة ────────────────────────────────────────────────
# مصدر: Claude.ai system prompt (2026) — النماذج الرسمية المتاحة:
#   claude-sonnet-4-6          ← Sonnet 4 (الأسرع/الأوفر تكلفةً)
#   claude-opus-4-8            ← Opus 4 (الأقوى)
#   claude-haiku-4-5-20251001  ← Haiku 4 (الأخف والأسرع)
#   claude-sonnet-4-20250514   ← Sonnet 4 للـ Artifacts (مستقر)
ANTHROPIC_MODELS = {
    "sonnet":  "claude-sonnet-4-6",           # الافتراضي — توازن مثالي بين الجودة والسرعة
    "opus":    "claude-opus-4-8",             # للمهام المعقدة التي تتطلب أعلى دقة
    "haiku":   "claude-haiku-4-5-20251001",   # للردود السريعة والمهام الخفيفة
    "stable":  "claude-sonnet-4-20250514",    # إصدار مستقر للإنتاج
    "fable":   "claude-fable-5",              # متخصص في السرد الإبداعي (ai/fable_engine.py)
}

_ANTHROPIC_MODEL  = ANTHROPIC_MODELS["sonnet"]    # الأولوية الأولى ✅
_CF_MODEL         = "@cf/meta/llama-3.1-8b-instruct"  # مجاني 10k/يوم ✅
_OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"  # مجاني
_OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"
_OPENAI_MODEL     = "gpt-4o-mini"
_TOGETHER_MODEL   = "meta-llama/Llama-3-8b-chat-hf"
_GEMINI_MODEL     = "gemini-2.5-flash"  # gemini-1.5-flash أُطفئ نهائياً ويرجع 404
_GROQ_MODELS          = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "openai/gpt-oss-20b"]
_FAILURE_COOLDOWN_SEC = 300   # 5 دقائق قبل إعادة تجربة مزوّد فاشل


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
    tried:      List[str] = None   # سجل المزوّدين الذين جرى تجريبهم بالترتيب

    def __post_init__(self):
        if self.tried is None:
            self.tried = []


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

    أولوية المزوّدين:
      1. Anthropic Claude (ANTHROPIC_API_KEY) ← الأولوية الأولى دائماً
      2. Cloudflare (CF_API_TOKEN + CF_ACCOUNT_ID) ← مجاني 10k/يوم
      3. Gemini   (GOOGLE_API_KEY)   ← سريع ومجاني
      4. OpenRouter (OPENROUTER_API_KEY)
      5. Groq     (GROQ_API_KEY)     ← قد يُحجب من بعض الشبكات
      6. OpenAI   (OPENAI_API_KEY)
      7. Together (TOGETHER_API_KEY)
      8. CKG Synthesis               ← دائماً متاح (fallback أخير)

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
        model_key:   Optional[str] = None,
    ):
        self.ckg         = ckg
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.timeout     = timeout
        # model_key اختياري: يسمح باختيار نموذج Anthropic محدد من
        # ANTHROPIC_MODELS (مثال: model_key="fable" لاستخدام claude-fable-5
        # في محرك السرد الإبداعي ai/fable_engine.py). إن لم يُمرَّر، يُستخدم
        # الافتراضي "sonnet" كما كان سابقاً — لا تغيير في السلوك القديم.
        self._model_key  = model_key if model_key in ANTHROPIC_MODELS else None

        self._provider, self._api_key, self._model = self._detect_provider()
        if self._model_key and self._provider == Provider.ANTHROPIC:
            self._model = ANTHROPIC_MODELS[self._model_key]
        # سجل المزوّدين الفاشلين: provider → timestamp انتهاء الـ cooldown
        self._failed_until: Dict[Provider, float] = {}
        logger.info(
            f"[LLMFallback] مزوّد: {self._provider.value} | نموذج: {self._model}"
        )

    # ── اكتشاف المزوّد تلقائياً (أول مزوّد متاح فقط — للتهيئة الأولى) ─────

    def _detect_provider(self) -> Tuple[Provider, str, str]:
        chain = self._build_provider_chain()
        if chain:
            return chain[0]
        return Provider.CKG_SYNTH, "", "ckg-synthesis-v1"

    # ── بناء سلسلة كل المزوّدين المتاحين ────────────────────────────────

    def _build_provider_chain(self) -> List[Tuple[Provider, str, str]]:
        """
        يُعيد قائمة مرتّبة بكل المزوّدين الذين لديهم مفاتيح صالحة.
        هذه القائمة هي مصدر الحقيقة لنظام التبديل التلقائي.
        """
        chain: List[Tuple[Provider, str, str]] = []

        # 1) Anthropic Claude
        k = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if k:
            model = (
                ANTHROPIC_MODELS[self._model_key]
                if self._model_key else _ANTHROPIC_MODEL
            )
            chain.append((Provider.ANTHROPIC, k, model))

        # 2) Cloudflare Workers AI
        cf_token   = os.getenv("CF_API_TOKEN",   "").strip()
        cf_account = os.getenv("CF_ACCOUNT_ID",  "").strip()
        if cf_token and cf_account:
            chain.append((Provider.CLOUDFLARE, cf_token, _CF_MODEL))

        # 3) Google Gemini
        k = os.getenv("GOOGLE_API_KEY", "").strip()
        if k and k.startswith("AIzaSy"):
            chain.append((Provider.GEMINI, k, _GEMINI_MODEL))

        # 4) OpenRouter
        k = os.getenv("OPENROUTER_API_KEY", "").strip()
        if k:
            chain.append((Provider.OPENROUTER, k, _OPENROUTER_MODEL))

        # 5) Groq
        k = os.getenv("GROQ_API_KEY", "").strip()
        if k:
            chain.append((Provider.GROQ, k, _GROQ_MODELS[0]))

        # 6) OpenAI
        k = os.getenv("OPENAI_API_KEY", "").strip()
        if k:
            chain.append((Provider.OPENAI, k, _OPENAI_MODEL))

        # 7) Together
        k = os.getenv("TOGETHER_API_KEY", "").strip()
        if k:
            chain.append((Provider.TOGETHER, k, _TOGETHER_MODEL))

        return chain

    # ── التبديل بين المزوّدين حسب النوع ─────────────────────────────────

    def _call_provider(
        self,
        provider: Provider,
        api_key:  str,
        model:    str,
        query:    str,
        history:  List[Tuple[str, str]],
        sp:       str,
    ) -> FallbackResult:
        """يستدعي المزوّد المحدّد ويُعيد النتيجة — أو يرفع استثناءً عند الفشل."""
        # تحديث مؤقت حتى تعمل دوال _call_* (تقرأ self._api_key و self._model)
        old_key, old_model = self._api_key, self._model
        self._api_key, self._model = api_key, model
        try:
            if provider == Provider.ANTHROPIC:
                return self._call_anthropic(query, history, sp)
            elif provider == Provider.CLOUDFLARE:
                return self._call_cloudflare(query, history, sp)
            elif provider == Provider.OPENROUTER:
                return self._call_openrouter(query, history, sp)
            elif provider == Provider.OPENAI:
                return self._call_openai(query, history, sp)
            elif provider == Provider.TOGETHER:
                return self._call_together(query, history, sp)
            elif provider == Provider.GEMINI:
                return self._call_gemini(query, history, sp)
            elif provider == Provider.GROQ:
                return self._call_groq(query, history, sp)
            else:
                raise ValueError(f"مزوّد غير معروف: {provider}")
        finally:
            self._api_key, self._model = old_key, old_model

    # ── الواجهة العامة مع التبديل التلقائي ──────────────────────────────

    def generate(
        self,
        query:   str,
        history: Optional[List[Tuple[str, str]]] = None,
        system_prompt: Optional[str] = None,
    ) -> FallbackResult:
        """
        يولّد إجابة مع التبديل التلقائي بين المزوّدين عند الفشل.

        السلوك:
          1. يبني سلسلة كل المزوّدين المتاحين (لديهم مفاتيح API).
          2. يتخطّى المزوّدين الذين فشلوا مؤخراً (cooldown = 5 دقائق).
          3. عند فشل مزوّد → يُسجَّل في _failed_until → ينتقل للتالي.
          4. إذا فشلت الجميع → يسقط إلى CKG Synthesis.
          5. عند نجاح مزوّد → يصبح self._provider الجديد.
        """
        t0      = time.time()
        history = history or []
        sp      = system_prompt or _SYSTEM_PROMPT
        now     = time.time()
        tried:  List[str] = []

        chain = self._build_provider_chain()

        for (prov, key, mdl) in chain:
            # تخطّ المزوّدين في فترة الـ cooldown
            if self._failed_until.get(prov, 0) > now:
                remaining = int(self._failed_until[prov] - now)
                logger.debug(
                    f"[Rotation] تخطّي {prov.value} (cooldown {remaining}ث متبقية)"
                )
                tried.append(f"{prov.value}:cooldown")
                continue

            try:
                logger.info(f"[Rotation] محاولة → {prov.value} / {mdl}")
                result = self._call_provider(prov, key, mdl, query, history, sp)

                # نجاح: تحديث المزوّد الحالي وإزالته من قائمة الفاشلين
                self._provider, self._api_key, self._model = prov, key, mdl
                self._failed_until.pop(prov, None)
                tried.append(f"{prov.value}:ok")
                result.tried = tried
                result.latency_ms = round((time.time() - t0) * 1000, 1)
                if len(tried) > 1:
                    logger.info(
                        f"[Rotation] نجح {prov.value} بعد {len(tried)-1} محاولة فاشلة"
                    )
                return result

            except Exception as exc:
                err_msg = str(exc)[:120]
                logger.warning(
                    f"[Rotation] فشل {prov.value}: {err_msg} — جرّب التالي..."
                )
                self._failed_until[prov] = now + _FAILURE_COOLDOWN_SEC
                tried.append(f"{prov.value}:err({err_msg})")

        # كل المزوّدين فشلوا → CKG Synthesis
        logger.error(f"[Rotation] فشلت جميع المزوّدين {[t for t in tried]} → CKG")
        result = FallbackResult(
            text=_ckg_synthesize(query, self.ckg),
            provider=Provider.CKG_SYNTH,
            model="ckg-synthesis-v1",
            error=f"فشلت كل المزوّدين: {tried}",
            tried=tried,
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

    # ── Anthropic Claude (الأولوية الأولى ✅) ───────────────────────────

    def _call_anthropic(
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        messages = []
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        data = _post_json(
            "https://api.anthropic.com/v1/messages",
            {
                "model":      self._model,
                "system":     system_prompt,
                "messages":   messages,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            },
            {
                "x-api-key":         self._api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type":      "application/json",
            },
            self.timeout,
        )
        # رسالة Claude تُرجَع كمصفوفة content blocks — نجمع نصوص type=="text" فقط
        text = "".join(
            block.get("text", "")
            for block in data.get("content", [])
            if block.get("type") == "text"
        ).strip()
        return FallbackResult(
            text=text, provider=Provider.ANTHROPIC, model=self._model
        )

    # ── Cloudflare Workers AI (مجاني 10k/يوم ✅) ────────────────────────

    def _call_cloudflare(
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        account_id = os.getenv("CF_ACCOUNT_ID", "").strip()
        cf_url = (
            f"https://api.cloudflare.com/client/v4/accounts/"
            f"{account_id}/ai/run/{self._model}"
        )
        messages = [{"role": "system", "content": system_prompt}]
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        data = _post_json(
            cf_url,
            {"messages": messages, "max_tokens": self.max_tokens},
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            self.timeout,
        )
        text = (
            data.get("result", {}).get("response", "")
            or data.get("choices", [{}])[0].get("message", {}).get("content", "")
        ).strip()
        return FallbackResult(
            text=text,
            provider=Provider.CLOUDFLARE,
            model=self._model,
        )

    # ── OpenRouter (يعمل من كل مكان ✅) ─────────────────────────────────

    def _call_openrouter(
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        messages = [{"role": "system", "content": system_prompt}]
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        data = _post_json(
            _OPENROUTER_URL,
            {
                "model":       self._model,
                "messages":    messages,
                "max_tokens":  self.max_tokens,
                "temperature": self.temperature,
            },
            {
                "Authorization":  f"Bearer {self._api_key}",
                "Content-Type":   "application/json",
                "HTTP-Referer":   "https://neural-service-mesh.streamlit.app",
                "X-Title":        "Neural Service Mesh",
            },
            self.timeout,
        )
        return FallbackResult(
            text=data["choices"][0]["message"]["content"].strip(),
            provider=Provider.OPENROUTER,
            model=self._model,
        )

    # ── OpenAI ───────────────────────────────────────────────────────────

    def _call_openai(
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        messages = [{"role": "system", "content": system_prompt}]
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
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        messages = [{"role": "system", "content": system_prompt}]
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
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
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
            "system_instruction": {"parts": [{"text": system_prompt}]},
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
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        messages = [{"role": "system", "content": system_prompt}]
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
