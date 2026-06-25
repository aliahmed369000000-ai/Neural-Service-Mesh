"""
LLM Generative Fallback Engine — NSM v18.1
============================================
يوفر طبقة توليد نصي حقيقي عندما لا يجد NSMChat إجابة كافية في قاموسه الثابت.

الأولوية في اختيار المزوّد (auto-detect من env vars):
  1. Groq API       (GROQ_API_KEY)   — مجاني، سريع جداً، Llama-3 / Mixtral
  2. OpenAI API     (OPENAI_API_KEY) — GPT-4o-mini
  3. HuggingFace    (HF_API_TOKEN)   — Mistral-7B-Instruct
  4. CKG Synthesis  (بدون مفتاح)    — يولّد من الرسم المعرفي (cognitive_graph.json)

الاستخدام:
    from ai.llm_fallback import LLMFallback

    fb = LLMFallback(ckg=my_ckg_instance)
    result = fb.generate("ما هو مفهوم التوحيد في الإسلام؟", history=[...])
    print(result.text)          # النص المولَّد
    print(result.provider.value) # "groq" | "openai" | "huggingface" | "ckg_synthesis"
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("LLMFallback")


# ════════════════════════════════════════════════════════════════════════════
# Provider Enum
# ════════════════════════════════════════════════════════════════════════════

class Provider(Enum):
    GROQ        = "groq"
    OPENAI      = "openai"
    HUGGINGFACE = "huggingface"
    CKG_SYNTH   = "ckg_synthesis"


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

_GROQ_MODELS  = ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama3-8b-8192"]
_OPENAI_MODEL = "gpt-4o-mini"
_HF_MODEL     = "mistralai/Mistral-7B-Instruct-v0.2"


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
    يبني إجابة من الرسم المعرفي المحلي (cognitive_graph.json).
    يُفعَّل تلقائياً عندما لا يوجد مفتاح API.

    الخوارزمية:
      1. استخراج الكلمات المفتاحية من الاستعلام
      2. البحث عن المفاهيم المرتبطة في CKG بالكلمات المفتاحية
      3. ترتيب المفاهيم بالوزن والقوة
      4. تركيب جملة معلوماتية عربية من المفاهيم المُسترجَعة
    """
    if ckg is None:
        return _generic_fallback()

    try:
        # استخراج الكلمات الجوهرية (أكثر من حرفين)
        stop_words = {"هل", "ما", "من", "في", "عن", "على", "إلى", "هو", "هي", "كيف", "لماذا", "متى"}
        words = [w.strip("؟.,!") for w in query.split() if len(w) > 2 and w not in stop_words]

        # جمع المفاهيم المرتبطة
        candidates: Dict[str, float] = {}
        for word in words[:6]:
            try:
                related = ckg.query_related(word, top_k=5)
                for name, weight in related:
                    candidates[name] = max(candidates.get(name, 0.0), weight)
            except Exception:
                pass

        if not candidates:
            return _generic_fallback()

        # ترتيب بالوزن
        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:8]
        top_names = [name for name, _ in ranked]

        # استخراج المجموعات (clusters) لأبرز المفاهيم
        clusters: Dict[str, str] = {}
        for name, _ in ranked[:4]:
            concept = ckg._concepts.get(name)
            if concept and concept.cluster:
                clusters[name] = concept.cluster

        # بناء جملة الإجابة
        if len(top_names) >= 3:
            core     = "، ".join(top_names[:3])
            extended = "، ".join(top_names[3:6]) if len(top_names) > 3 else ""
            answer   = f"يرتبط سؤالك بالمفاهيم المعرفية التالية في الرسم المعرفي للنظام: {core}."
            if extended:
                answer += f" كما يتصل بـ: {extended}."
            # إضافة المجموعة المعرفية إذا كانت موحّدة
            unique_clusters = list(set(clusters.values()))
            if len(unique_clusters) == 1:
                answer += f" هذه المفاهيم تنتمي إلى مجال: {unique_clusters[0]}."
            elif unique_clusters:
                answer += f" تغطي مجالات: {' | '.join(unique_clusters[:3])}."
        else:
            answer = f"المعرفة المتاحة عن سؤالك تتمحور حول: {' ، '.join(top_names)}."

        answer += " (مُستخلَص من الرسم المعرفي — لمزيد من الدقة أضف مفتاح API)"
        return answer

    except Exception as exc:
        logger.warning(f"[CKGSynth] فشل التوليد: {exc}")
        return _generic_fallback()


def _generic_fallback() -> str:
    return (
        "سؤالك خارج نطاق معرفتي المباشرة حالياً. "
        "يمكنني المساعدة في: الإسلام والقرآن الكريم، الذكاء الاصطناعي، "
        "الرياضيات، اللغة العربية، التاريخ الإسلامي، والبرمجة. "
        "لتفعيل التوليد الكامل، أضف مفتاح GROQ_API_KEY في إعدادات البيئة."
    )


# ════════════════════════════════════════════════════════════════════════════
# LLMFallback — المحرك الرئيسي
# ════════════════════════════════════════════════════════════════════════════

class LLMFallback:
    """
    طبقة التوليد الذكي تُفعَّل عند فشل NSMChat في إيجاد إجابة بنتيجة كافية.

    مثال:
        fb = LLMFallback(ckg=my_ckg)
        result = fb.generate("اشرح مفهوم التوحيد", history=[("كيف حالك؟","بخير!")])
        print(result.text)
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
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key:
            return Provider.GROQ, key, _GROQ_MODELS[0]

        key = os.getenv("OPENAI_API_KEY", "").strip()
        if key:
            return Provider.OPENAI, key, _OPENAI_MODEL

        key = os.getenv("HF_API_TOKEN", "").strip()
        if key:
            return Provider.HUGGINGFACE, key, _HF_MODEL

        return Provider.CKG_SYNTH, "", "ckg-synthesis-v1"

    # ── الواجهة العامة ───────────────────────────────────────────────────

    def generate(
        self,
        query:   str,
        history: Optional[List[Tuple[str, str]]] = None,
    ) -> FallbackResult:
        """
        يولّد إجابة للاستعلام.

        Args:
            query:   نص سؤال المستخدم
            history: قائمة (user_msg, bot_msg) لآخر N رسائل
        Returns:
            FallbackResult مع النص والمزوّد والكمون
        """
        t0      = time.time()
        history = history or []

        try:
            if self._provider == Provider.GROQ:
                result = self._call_groq(query, history)
            elif self._provider == Provider.OPENAI:
                result = self._call_openai(query, history)
            elif self._provider == Provider.HUGGINGFACE:
                result = self._call_hf(query, history)
            else:
                text   = _ckg_synthesize(query, self.ckg)
                result = FallbackResult(
                    text=text, provider=Provider.CKG_SYNTH, model=self._model
                )
        except (urllib.error.URLError, TimeoutError, KeyError, IndexError) as exc:
            logger.error(f"[LLMFallback] {self._provider.value} فشل: {exc}")
            text   = _ckg_synthesize(query, self.ckg)
            result = FallbackResult(
                text=text,
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

    def has_live_llm(self) -> bool:
        """هل يوجد نموذج LLM حقيقي (وليس CKG synthesis فقط)؟"""
        return self._provider != Provider.CKG_SYNTH

    def info(self) -> Dict[str, str]:
        return {
            "provider": self._provider.value,
            "model":    self._model,
            "live_llm": "✅" if self.has_live_llm() else "❌ (CKG synthesis)",
            "api_key":  "✅ موجود" if self._api_key else "❌ غير موجود",
        }

    # ── Groq ─────────────────────────────────────────────────────────────

    def _call_groq(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for u, a in history[-4:]:
            messages.append({"role": "user",      "content": u})
            messages.append({"role": "assistant",  "content": a})
        messages.append({"role": "user", "content": query})

        payload = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
            "stream":      False,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        data = _post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            payload, headers, self.timeout,
        )
        text = data["choices"][0]["message"]["content"].strip()
        return FallbackResult(text=text, provider=Provider.GROQ, model=self._model)

    # ── OpenAI ───────────────────────────────────────────────────────────

    def _call_openai(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for u, a in history[-4:]:
            messages.append({"role": "user",      "content": u})
            messages.append({"role": "assistant",  "content": a})
        messages.append({"role": "user", "content": query})

        payload = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  self.max_tokens,
            "temperature": self.temperature,
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        data = _post_json(
            "https://api.openai.com/v1/chat/completions",
            payload, headers, self.timeout,
        )
        text = data["choices"][0]["message"]["content"].strip()
        return FallbackResult(text=text, provider=Provider.OPENAI, model=self._model)

    # ── HuggingFace Inference API ─────────────────────────────────────────

    def _call_hf(
        self, query: str, history: List[Tuple[str, str]]
    ) -> FallbackResult:
        conv = ""
        for u, a in history[-3:]:
            conv += f"[INST] {u} [/INST] {a} </s>"
        conv += f"[INST] {query} [/INST]"
        full_prompt = f"<s>{_SYSTEM_PROMPT}\n\n{conv}"

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens":  self.max_tokens,
                "temperature":     self.temperature,
                "return_full_text": False,
            },
        }
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type":  "application/json",
        }
        url  = f"https://api-inference.huggingface.co/models/{self._model}"
        data = _post_json(url, payload, headers, self.timeout)
        text = (
            data[0]["generated_text"].strip()
            if isinstance(data, list)
            else str(data)
        )
        return FallbackResult(
            text=text, provider=Provider.HUGGINGFACE, model=self._model
        )
