"""
LLM Generative Fallback Engine — NSM v18.4
============================================
يوفر طبقة توليد نصي حقيقي عندما لا يجد NSMChat إجابة كافية في قاموسه الثابت.

الأولوية في اختيار المزوّد (auto-detect من env vars):
  1. Groq            (GROQ_API_KEY)     ← الأولوية الأولى ✅ gpt-oss-120b
                                           (مجاني: 14k طلب/دقيقة — ~1000 توكن/ث)
  1.5. Cerebras      (CEREBRAS_API_KEY) — احتياطي فوري لـGroq، نفس أوزان
                                           gpt-oss-120b على عتاد وحصة مستقلة
  2. Cloudflare Workers AI (CF_API_TOKEN + CF_ACCOUNT_ID) — مجاني 10k/يوم
  3. Google Gemini   (GOOGLE_API_KEY)   — Gemini 2.5 Flash
  4. OpenRouter      (OPENROUTER_API_KEY) — نماذج مجانية تلقائياً، أو Kimi K3
                                             (moonshotai/kimi-k3) عبر model_key="kimi"
  5. Anthropic Claude (ANTHROPIC_API_KEY) — Claude Sonnet 5 — يُضاف بعد النماذج
                                            المجانية لأنه مدفوع بالإنشاء
  6. OpenAI API      (OPENAI_API_KEY)   — GPT-4o-mini
  7. Together.xyz    (TOGETHER_API_KEY) — Llama-3/Mixtral
  8. Hugging Face    (HUGGINGFACE_API_KEY أو HF_TOKEN) — Falcon-Arabic-7B-Instruct (مجاني)
  9. نموذج محلي (Ollama) — فقط إذا حُدِّد NSM_LOCAL_LLM_URL صراحة
 10. CKG Synthesis   (بدون مفتاح)      — يولّد من الرسم المعرفي دائماً

🆕 NSM_LLM_PROVIDER_PREF (اختياري) — فرض مزوّد واحد دون غيره:
   NSM_LLM_PROVIDER_PREF=groq        → Groq فقط (ثم CKG إن فشل)
   NSM_LLM_PROVIDER_PREF=openrouter  → OpenRouter فقط
   NSM_LLM_PROVIDER_PREF=gemini      → Gemini فقط
   NSM_LLM_PROVIDER_PREF=cf          → Cloudflare فقط
   NSM_LLM_PROVIDER_PREF=anthropic   → Anthropic فقط
   NSM_LLM_PROVIDER_PREF=auto        → السلسلة الكاملة أعلاه (الافتراضي)

🔒 وضع النشر المغلق (بدون إنترنت خارجي — سيرفرات الجهة المشترية):
   NSM_OFFLINE_MODE=1 يجعل النموذج المحلي (Ollama) المزوّد الوحيد في
   السلسلة تماماً — لا تُجرَّب أي مزوّدات سحابية إطلاقاً مهما كانت مفاتيح
   الـAPI موجودة في env. متغيرات التهيئة:
     NSM_OFFLINE_MODE=1                       (تفعيل الوضع المغلق)
     NSM_LOCAL_LLM_URL=http://localhost:11434 (عنوان خادم Ollama، اختياري)
     NSM_LOCAL_MODEL=qwen2.5:7b-instruct-q4_K_M (اختياري)

🆕 RAG على الـ CKG: قبل استدعاء أي مزوّد حيّ، يبحث generate() في الرسم
المعرفي (self.ckg) عن مفاهيم ذات صلة بالسؤال ويُرفقها كسياق إضافي ضمن
الـ system prompt (انظر _build_ckg_context) — هذا يفيد كل المزوّدين
وليس فقط Falcon، بدون أي تدريب إضافي.

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
    ANTHROPIC = "anthropic"    # Claude — مدفوع بالإنشاء (بعد المجانيات) ✅
    CLOUDFLARE = "cloudflare"  # مجاني 10k/يوم ويعمل من اليمن ✅
    GEMINI    = "gemini"
    OPENROUTER = "openrouter"
    OPENAI    = "openai"
    TOGETHER  = "together"
    GROQ      = "groq"       # الأولوية الأولى ✅ — مجاني وسريع جداً
    CEREBRAS  = "cerebras"      # احتياطي فوري لـGroq — نفس أوزان gpt-oss-120b
                                 # على عتاد مختلف وحصة مجانية منفصلة (dual-homing)
    HUGGINGFACE = "huggingface"  # Falcon-Arabic-7B-Instruct — مجاني (HF Inference API)
    LOCAL     = "local"  # نموذج محلي عبر Ollama — للنشر المغلق بدون إنترنت (NSM_OFFLINE_MODE)
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
# Kimi K3 (Moonshot AI) — أحدث نموذج مفتوح المصدر (2.8T معامل، MoE، 896
# خبيراً/16 نشِط لكل توكن)، أُطلق 16 يوليو 2026 وصدرت أوزانه الكاملة في
# 27 يوليو 2026. أداء من فئة frontier (منافس لنماذج مغلقة) عبر OpenRouter.
# مدفوع ($3/$15 لكل مليون توكن دخل/خرج) → لا يُختار تلقائياً ضمن الاكتشاف
# المجاني، بل فقط عند تحديده صراحةً عبر model_key="kimi" (انظر OPENROUTER_MODELS).
_OPENROUTER_KIMI_MODEL = "moonshotai/kimi-k3"
_OPENROUTER_URL   = "https://openrouter.ai/api/v1/chat/completions"

# نماذج OpenRouter القابلة للاختيار صراحةً (مثل ANTHROPIC_MODELS أدناه) —
# استخدمها عبر LLMFallback(model_key="kimi") لتجاوز الاكتشاف التلقائي
# للنماذج المجانية واستخدام Kimi K3 تحديداً.
OPENROUTER_MODELS = {
    "free": _OPENROUTER_MODEL,       # الافتراضي — نموذج مجاني (Llama 3.1 8B)
    "kimi": _OPENROUTER_KIMI_MODEL,  # Kimi K3 — أقوى نموذج مفتوح المصدر متاح حالياً (مدفوع)
}
_OPENAI_MODEL     = "gpt-4o-mini"
_TOGETHER_MODEL   = "meta-llama/Llama-3-8b-chat-hf"
_GEMINI_MODEL     = "gemini-2.5-flash"  # gemini-1.5-flash أُطفئ نهائياً ويرجع 404
_GROQ_MODELS          = ["openai/gpt-oss-120b", "llama-3.3-70b-versatile", "llama-3.1-8b-instant", "openai/gpt-oss-20b"]
# gpt-oss-120b أولاً: أقوى نموذج مفتوح المصدر متاح مجاناً على Groq حالياً
# (116B معامل، Apache 2.0) — نفس الاختيار المطبَّق في ai/ultraplinian.py
# (FREE_DIRECT_MODELS و_GROQ_FALLBACK_MODELS)، تحقّق بتاريخ أغسطس 2026.
# ملاحظة: _GROQ_MODELS[0] فقط هو المستخدم فعلياً في سلسلة LLMFallback
# (انظر _build_provider_chain أدناه) — البقية للتوثيق فقط حالياً.
# Cerebras — نفس أوزان gpt-oss-120b بالضبط، لكن على عتاد Cerebras WSE
# وبحصة مجانية منفصلة تماماً عن حصة Groq (1M توكن/يوم، بدون بطاقة).
# يُستخدم كاحتياطي فوري إذا حُجب Groq أو استُنفدت حصته — بنفس النموذج
# تماماً فلا يتغيّر شكل الردود عند التبديل (dual-homing، تحقّق أغسطس 2026).
_CEREBRAS_MODEL       = "gpt-oss-120b"
_CEREBRAS_URL         = "https://api.cerebras.ai/v1/chat/completions"
# Falcon-Arabic-7B-Instruct — نموذج لغوي عربي عام (وليس متخصصاً دينياً فقط)
# مبني على Falcon3-7B من TII، مجاني بالكامل عبر HF Inference API.
_HF_MODEL             = "tiiuae/Falcon-Arabic-7B-Instruct"
_HF_INFERENCE_URL     = f"https://api-inference.huggingface.co/models/{_HF_MODEL}"
_FAILURE_COOLDOWN_SEC = 300   # 5 دقائق قبل إعادة تجربة مزوّد فاشل

# ── النموذج المحلي (Ollama) — للنشر داخل شبكة الجهة المغلقة ──────────────
# سيرفر عادي CPU فقط أو GPU صغير → نموذج مصغّر مكمَّم (quantized) بصيغة
# GGUF عبر Ollama. Qwen2.5-7B-Instruct له دعم عربي جيد ونسخة مكمَّمة q4_K_M
# تعمل على CPU بذاكرة معقولة (~5GB RAM). يمكن استبداله بأي tag آخر مثبَّت
# على خادم Ollama (مثال: نسخة Yemeni LoRA المدرَّبة مسبقاً إن حُوِّلت لـGGUF).
_LOCAL_MODEL          = "qwen2.5:7b-instruct-q4_K_M"
_LOCAL_BASE_URL       = "http://localhost:11434"   # عنوان خادم Ollama الافتراضي
_LOCAL_TIMEOUT_SEC    = 90   # الاستدلال على CPU أبطأ بكثير من الـAPI السحابية

# NSM_OFFLINE_MODE=1 → يجعل النموذج المحلي المزوّد الوحيد في السلسلة، ولا
# تُجرَّب أي مزوّدات سحابية إطلاقاً (لا توجد محاولات اتصال بالإنترنت الخارجي
# حتى لو كانت مفاتيح API موجودة بالخطأ في env — مهم لبيئة الجهة المغلقة).

# ── اكتشاف نماذج OpenRouter المجانية تلقائياً ────────────────────────────
_OPENROUTER_MODELS_URL   = "https://openrouter.ai/api/v1/models"
_OPENROUTER_CACHE_PATH   = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "openrouter_models_cache.json"
)
_OPENROUTER_CACHE_TTL    = 6 * 3600   # 6 ساعات قبل إعادة الاكتشاف
_OPENROUTER_MAX_MODELS   = 5          # أقصى عدد نماذج نجرّبها بالتتابع

# ترتيب تفضيلي (الأفضل أولاً) لعائلات النماذج المجانية المعروفة بجودة عالية
# على OpenRouter. قائمة ":free" على OpenRouter تتغيّر أسبوعياً (نماذج تُضاف
# وتُحذف باستمرار — مثلاً DeepSeek وGLM كانا مجانيين ثم أصبحا مدفوعَين في
# فترات مختلفة)، لذا تعمّدنا عدم تثبيت اسم نموذج واحد كما فعلنا سابقاً مع
# Kimi K3 (مدفوع دائماً)، بل نُرتّب أي نموذج مجاني *متاح فعلياً الآن* حسب
# عائلته، بحيث يُجرَّب الأقوى أولاً تلقائياً دون تدخّل يدوي مع كل دورة.
_FREE_MODEL_QUALITY_FAMILIES = [
    "glm-4.6", "glm-4.5", "glm-",                 # Zhipu GLM — قريب من Sonnet
    "deepseek-v3", "deepseek-r1", "deepseek",     # DeepSeek — استدلال قوي
    "qwen3-", "qwen-",                            # Qwen — دعم عربي/متعدد لغات جيد
    "llama-4", "llama-3.3", "llama-3.1",          # Meta Llama
    "gemini",                                     # Google Gemini (نسخ مجانية أحياناً)
    "gpt-oss", "hermes", "gemma", "mistral",
]


def _free_model_rank(model_id: str) -> int:
    """رتبة جودة تقديرية لنموذج مجاني (أصغر رقم = أفضل)؛ تُستخدم لترتيب
    النماذج المكتشفة تلقائياً بدل الاعتماد على ترتيب استجابة الـAPI العشوائي.
    نموذج غير معروف يحصل على أدنى أولوية (يُجرَّب أخيراً)."""
    mid = model_id.lower()
    for i, family in enumerate(_FREE_MODEL_QUALITY_FAMILIES):
        if family in mid:
            return i
    return len(_FREE_MODEL_QUALITY_FAMILIES)


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


def _get_json(url: str, headers: dict, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def discover_openrouter_models(
    api_key: str,
    cache_path: str = _OPENROUTER_CACHE_PATH,
    ttl_sec: int = _OPENROUTER_CACHE_TTL,
    max_models: int = _OPENROUTER_MAX_MODELS,
    fetcher=_get_json,
) -> List[str]:
    """
    يكتشف نماذج OpenRouter المجانية المتاحة تلقائياً (بدل الاعتماد على نموذج
    واحد ثابت). يُخزَّن الناتج في cache محلي (TTL) لتفادي طلب /models في كل
    استدعاء. عند فشل الاكتشاف أو انتهاء الشبكة، يعود لآخر cache صالح، ثم
    أخيراً للنموذج الثابت الافتراضي _OPENROUTER_MODEL.

    الترتيب: النماذج المجانية (":free" أو pricing.prompt == "0") تُرتَّب
    حسب جودة العائلة المعروفة (_free_model_rank) بحيث تُجرَّب أقواها أولاً
    (مثال: GLM/DeepSeek/Qwen إن كانت مجانية حالياً)، مع إبقاء
    _OPENROUTER_MODEL دائماً كخيار أخير مضمون.
    """
    now = time.time()

    # 1) جرّب الـ cache أولاً إن كان ما زال صالحاً
    cached = _load_openrouter_cache(cache_path)
    if cached and (now - cached.get("ts", 0)) < ttl_sec:
        models = cached.get("models", [])
        if models:
            return _ensure_default_last(models, max_models)

    # 2) اكتشاف حي عبر واجهة OpenRouter
    try:
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        data = fetcher(_OPENROUTER_MODELS_URL, headers, 10)
        free_models = []
        for m in data.get("data", []):
            mid = m.get("id", "")
            pricing = m.get("pricing", {}) or {}
            is_free = mid.endswith(":free") or str(pricing.get("prompt", "")) == "0"
            if mid and is_free:
                free_models.append(mid)
        if free_models:
            free_models.sort(key=_free_model_rank)
            _save_openrouter_cache(cache_path, free_models, now)
            return _ensure_default_last(free_models, max_models)
    except Exception as exc:
        logger.warning(f"[OpenRouterDiscover] فشل الاكتشاف الحي: {exc}")

    # 3) استخدم آخر cache معروف حتى لو منتهي الصلاحية (أفضل من لا شيء)
    if cached and cached.get("models"):
        return _ensure_default_last(cached["models"], max_models)

    # 4) fallback نهائي: النموذج الثابت فقط
    return [_OPENROUTER_MODEL]


def _ensure_default_last(models: List[str], max_models: int) -> List[str]:
    ordered = [m for m in models if m != _OPENROUTER_MODEL][:max_models - 1]
    ordered.append(_OPENROUTER_MODEL)
    seen = set()
    return [m for m in ordered if not (m in seen or seen.add(m))]


def _load_openrouter_cache(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _save_openrouter_cache(path: str, models: List[str], ts: float) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"ts": ts, "models": models}, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.warning(f"[OpenRouterDiscover] فشل حفظ الـ cache: {exc}")


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
        # 🆕 عند تمرير prompt مُعزَّز (مثلاً من ChainOfThoughtBuilder، يحتوي
        # أمثلة few-shot + مفاهيم مرتبطة قبل السؤال نفسه)، السؤال الفعلي
        # يكون بعد آخر سطر يبدأ بـ"السؤال:" — نستخرج الكلمات المفتاحية
        # منه حصراً بدل أول 6 كلمات من كامل الـ prompt (غالباً نص الأمثلة
        # لا السؤال الحقيقي).
        _marker = "السؤال:"
        effective = query
        if _marker in query:
            effective = query.rsplit(_marker, 1)[-1]

        stop_words = {
            "هل", "ما", "من", "في", "عن", "على", "إلى", "هو", "هي",
            "كيف", "لماذا", "متى", "أين", "ماذا", "التي", "الذي",
        }
        words = [
            w.strip("؟.,!:;") for w in effective.split()
            if len(w) > 2 and w not in stop_words
        ]

        candidates: Dict[str, float] = {}
        for word in words[:6]:
            # المطابقة في CKG تامة بالاسم المخزَّن؛ المفاهيم غالباً مجرَّدة
            # ("صبر") بينما كلمات السؤال مُعرَّفة بـ"ال" ("الصبر") فلا
            # تُطابِق إطلاقاً — نجرّب الكلمة كما هي ثم بعد إزالة "ال".
            variants = [word]
            if word.startswith("ال") and len(word) > 3:
                variants.append(word[2:])
            for variant in variants:
                try:
                    related = ckg.query_related(variant, top_k=5)
                except Exception:
                    related = []
                if related:
                    for name, weight in related:
                        candidates[name] = max(candidates.get(name, 0.0), weight)
                    break

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


def _build_ckg_context(query: str, ckg, max_concepts: int = 6) -> str:
    """
    🆕 RAG حقيقي: يبحث في الرسم المعرفي (CKG) عن مفاهيم ذات صلة بالسؤال،
    ويُعيدها كسطر سياق قصير يُرفَق مع الـ system prompt قبل استدعاء أي
    مزوّد LLM حيّ (Anthropic, Falcon-Arabic عبر HF, Gemini, ...).

    بخلاف _ckg_synthesize (اللي يبني إجابة نهائية جاهزة بدون LLM)، هذه
    الدالة تُعيد فقط "سياق مساعد" يُترك للنموذج نفسه صياغة الإجابة منه —
    يفيد أي سؤال (عام أو ديني) طالما له مفاهيم مطابقة في الـ CKG، ولا
    يفرض أي توجّه ديني إن لم يكن السؤال كذلك.

    تُعيد سلسلة فارغة "" إن لم يوجد ckg أو لم تُطابَق أي مفاهيم.
    """
    if ckg is None:
        return ""
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
            variants = [word]
            if word.startswith("ال") and len(word) > 3:
                variants.append(word[2:])
            for variant in variants:
                try:
                    related = ckg.query_related(variant, top_k=5)
                except Exception:
                    related = []
                if related:
                    for name, weight in related:
                        candidates[name] = max(candidates.get(name, 0.0), weight)
                    break

        if not candidates:
            return ""

        ranked = sorted(candidates.items(), key=lambda x: -x[1])[:max_concepts]
        top    = [n for n, _ in ranked]
        return (
            "سياق إضافي من قاعدة المعرفة الخاصة بالمشروع (استخدمها إن كانت "
            f"ذات صلة، وتجاهلها إن لم تكن): {'، '.join(top)}."
        )
    except Exception as exc:
        logger.warning(f"[CKGContext] {exc}")
        return ""


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

    أولوية المزوّدين (auto):
      1. Groq      (GROQ_API_KEY) ← الأولوية الأولى ✅ gpt-oss-120b
      1.5. Cerebras (CEREBRAS_API_KEY) ← احتياطي فوري لـGroq (نفس gpt-oss-120b)
      2. Cloudflare (CF_API_TOKEN + CF_ACCOUNT_ID) ← مجاني 10k/يوم
      3. Gemini   (GOOGLE_API_KEY)   ← سريع ومجاني
      4. OpenRouter (OPENROUTER_API_KEY)
      5. Anthropic (ANTHROPIC_API_KEY) ← مدفوع بالإنشاء — بعد المجانيات
      6. OpenAI   (OPENAI_API_KEY)
      7. Together (TOGETHER_API_KEY)
      8. CKG Synthesis               ← دائماً متاح (fallback أخير)

    قابل للتخصيص عبر NSM_LLM_PROVIDER_PREF (انظر توثيق أعلى الملف).

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
        # model_key اختياري: يسمح باختيار نموذج محدد من ANTHROPIC_MODELS
        # (مثال: model_key="fable" لاستخدام claude-fable-5 في محرك السرد
        # الإبداعي ai/fable_engine.py) أو من OPENROUTER_MODELS (مثال:
        # model_key="kimi" لاستخدام Kimi K3 عبر OpenRouter بدل الاكتشاف
        # التلقائي للنماذج المجانية). إن لم يُمرَّر أو لم يُطابق أياً منهما،
        # يُستخدم السلوك الافتراضي القديم — لا تغيير في السلوك القديم.
        self._model_key  = (
            model_key if model_key in ANTHROPIC_MODELS or model_key in OPENROUTER_MODELS
            else None
        )

        self._openrouter_models: List[str] = [_OPENROUTER_MODEL]
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
        Groq أولاً (مجاني وسريع جداً)، ثم المجانيات الأخرى، ثم المدفوعون.
        NSM_LLM_PROVIDER_PREF يفرض مزوّداً واحداً إن حُدِّد صراحة.
        """
        chain: List[Tuple[Provider, str, str]] = []
        pref = os.getenv("NSM_LLM_PROVIDER_PREF", "auto").strip().lower() or "auto"
        if pref not in (
            "auto", "groq", "cerebras", "cf", "cloudflare", "gemini",
            "openrouter", "anthropic", "openai", "together",
            "huggingface", "hf",
        ):
            logger.warning(
                "[LLMFallback] قيمة NSM_LLM_PROVIDER_PREF غير معروفة "
                "(%r) — تُستخدم auto", pref
            )
            pref = "auto"

        def _keep(prov: Provider) -> bool:
            """هل يُسمح بهذا المزوّد في الوضع الحالي؟"""
            if pref == "auto":
                return True
            if pref == "cf" and prov is Provider.CLOUDFLARE:
                return True
            if pref == "hf" and prov is Provider.HUGGINGFACE:
                return True
            return prov.value == pref

        # 0) وضع النشر المغلق (بدون إنترنت خارجي): النموذج المحلي فقط —
        #    نتوقف هنا فوراً ولا نضيف أي مزوّد سحابي للسلسلة إطلاقاً.
        offline_mode = os.getenv("NSM_OFFLINE_MODE", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        local_model = os.getenv("NSM_LOCAL_MODEL", _LOCAL_MODEL).strip() or _LOCAL_MODEL
        if offline_mode:
            chain.append((Provider.LOCAL, "", local_model))
            return chain

        # 0.5) في الوضع المتصل: النموذج المحلي يبقى خياراً اختيارياً إضافياً
        #      فقط إذا حُدِّد عنوان خادم صراحة (لا نفترض وجود Ollama افتراضياً
        #      على بيئات سحابية مثل Streamlit Cloud لتفادي تأخير/أخطاء غير
        #      ضرورية)، ويُضاف كخيار قبل CKG synthesis النهائي.
        local_url = os.getenv("NSM_LOCAL_LLM_URL", "").strip()

        # 1) Groq — gpt-oss-120b: مجاني (14k طلب/دقيقة) وسريع جداً (~1000
        #    توكن/ثانية). يُضاف قبل الجميع لأنه مجاني وسريع — إذا حُجب أو
        #    استُنفدت حصته ينتقل النظام تلقائياً للمجانيات الأخرى (cooldown)
        #    قبل أي مزوّد مدفوع.
        k = os.getenv("GROQ_API_KEY", "").strip()
        if k and _keep(Provider.GROQ):
            chain.append((Provider.GROQ, k, _GROQ_MODELS[0]))

        # 1.5) Cerebras — احتياطي فوري لنفس gpt-oss-120b (حصة مجانية مستقلة
        #      عن Groq تماماً). يُضاف مباشرة بعد Groq حتى لو فشل الأخير
        #      (حجب شبكي، انتهاء حصة)، يُجرَّب نفس النموذج على مزوّد آخر
        #      قبل النزول لمزوّدين أضعف.
        k = os.getenv("CEREBRAS_API_KEY", "").strip()
        if k and _keep(Provider.CEREBRAS):
            chain.append((Provider.CEREBRAS, k, _CEREBRAS_MODEL))

        # 2) Cloudflare Workers AI
        cf_token   = os.getenv("CF_API_TOKEN",   "").strip()
        cf_account = os.getenv("CF_ACCOUNT_ID",  "").strip()
        if cf_token and cf_account and _keep(Provider.CLOUDFLARE):
            chain.append((Provider.CLOUDFLARE, cf_token, _CF_MODEL))

        # 3) Google Gemini
        k = os.getenv("GOOGLE_API_KEY", "").strip()
        if k and k.startswith("AIzaSy") and _keep(Provider.GEMINI):
            chain.append((Provider.GEMINI, k, _GEMINI_MODEL))

        # 4) OpenRouter — نموذج مُختار صراحةً (مثال: "kimi" لـKimi K3) يتجاوز
        #    الاكتشاف التلقائي، وإلا يُكتشف أفضل نموذج مجاني متاح تلقائياً.
        k = os.getenv("OPENROUTER_API_KEY", "").strip()
        if k and _keep(Provider.OPENROUTER):
            if self._model_key in OPENROUTER_MODELS:
                chosen = OPENROUTER_MODELS[self._model_key]
                self._openrouter_models = [chosen]
                chain.append((Provider.OPENROUTER, k, chosen))
            else:
                models = discover_openrouter_models(k)
                self._openrouter_models = models
                chain.append((Provider.OPENROUTER, k, models[0]))

        # 5) Anthropic Claude — مدفوع بالإنشاء، يُضاف بعد المجانيات:
        #    المجانيات تكفي لمعظم الردود، والمدفوع يبقى خياراً احتياطياً
        #    دون استهلاك رصيد ما لم تُستنفد المجانيات.
        k = os.getenv("ANTHROPIC_API_KEY", "").strip()
        if k and _keep(Provider.ANTHROPIC):
            model = (
                ANTHROPIC_MODELS[self._model_key]
                if self._model_key else _ANTHROPIC_MODEL
            )
            chain.append((Provider.ANTHROPIC, k, model))

        # 6) OpenAI
        k = os.getenv("OPENAI_API_KEY", "").strip()
        if k and _keep(Provider.OPENAI):
            chain.append((Provider.OPENAI, k, _OPENAI_MODEL))

        # 7) Together
        k = os.getenv("TOGETHER_API_KEY", "").strip()
        if k and _keep(Provider.TOGETHER):
            chain.append((Provider.TOGETHER, k, _TOGETHER_MODEL))

        # 8) Hugging Face — Falcon-Arabic-7B-Instruct (مجاني بالكامل)
        k = os.getenv("HUGGINGFACE_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()
        if k and _keep(Provider.HUGGINGFACE):
            chain.append((Provider.HUGGINGFACE, k, _HF_MODEL))

        # 9) نموذج محلي (Ollama) — فقط إذا حُدِّد عنوان الخادم صراحة عبر
        #    NSM_LOCAL_LLM_URL. يُضاف كآخر خيار حي قبل CKG synthesis.
        if local_url:
            chain.append((Provider.LOCAL, "", local_model))

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
            elif provider == Provider.CEREBRAS:
                return self._call_cerebras(query, history, sp)
            elif provider == Provider.HUGGINGFACE:
                return self._call_huggingface(query, history, sp)
            elif provider == Provider.LOCAL:
                return self._call_local(query, history, sp)
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

        # 🆕 RAG: أرفق سياق الـ CKG (إن وُجد) قبل استدعاء أي مزوّد حيّ.
        # يفيد كل المزوّدين (Anthropic, Falcon-Arabic, Gemini, ...) بنفس
        # الطريقة، دون تدريب إضافي ودون التأثير على مسار CKG Synthesis
        # الحالي (الذي يبقى كما هو كـ fallback أخير بلا LLM).
        ckg_context = _build_ckg_context(query, self.ckg)
        if ckg_context:
            sp = f"{sp}\n\n{ckg_context}"

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
        if self._provider == Provider.CKG_SYNTH:
            return False
        # المزوّد المحلي (Ollama) لا يحتاج مفتاح API — لا نطلب api_key له
        return self._provider == Provider.LOCAL or bool(self._api_key)

    def has_live_llm(self) -> bool:
        """هل يوجد LLM حقيقي يعمل (وليس CKG synthesis فقط)؟"""
        return self._provider != Provider.CKG_SYNTH

    def info(self) -> Dict[str, str]:
        pref = os.getenv("NSM_LLM_PROVIDER_PREF", "auto").strip().lower() or "auto"
        return {
            "provider": self._provider.value,
            "model":    self._model,
            "live_llm": "✅" if self.has_live_llm() else "❌ (CKG synthesis)",
            "api_key":  "✅ موجود" if self._api_key else "❌ غير موجود",
            "pref_mode": f"مفروض ({pref})" if pref != "auto" else "تلقائي",
        }

    # ── Anthropic Claude (بعد المجانيات — مدفوع بالإنشاء) ────────────

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

        # جرّب النماذج المكتشَفة تلقائياً بالتتابع (نفس نمط Groq) —
        # كل نموذج مجاني مكتشَف يُعامَل كـ "عقدة" بديلة عند فشل الأول.
        candidates = list(dict.fromkeys(self._openrouter_models + [self._model]))

        last_err = None
        for model in candidates:
            try:
                data = _post_json(
                    _OPENROUTER_URL,
                    {
                        "model":       model,
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
                    model=model,
                )
            except Exception as e:
                last_err = str(e)
                continue

        raise Exception(f"فشلت كل نماذج OpenRouter المكتشَفة: {last_err}")

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

        # نماذج بديلة عند 403 — gemma2-9b-it وllama3-8b-8192 أُزيلا لأنهما
        # لم يعودا ضمن قائمة Groq الرسمية (يسببان فشلاً صامتاً بكل محاولة).
        groq_models = [
            self._model,
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-20b",
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

    # ── Cerebras — احتياطي فوري لـGroq بنفس أوزان gpt-oss-120b تماماً ────
    # عتاد مختلف (Cerebras WSE بدل Groq LPU) وحصة مجانية مستقلة (1M
    # توكن/يوم)، لكن نفس النموذج بالضبط — فلا يتغيّر شكل الردود عند
    # التبديل التلقائي. واجهة OpenAI-compatible مطابقة لـGroq تقريباً.

    def _call_cerebras(
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
            _CEREBRAS_URL,
            {
                "model":       self._model or _CEREBRAS_MODEL,
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
            provider=Provider.CEREBRAS,
            model=self._model or _CEREBRAS_MODEL,
        )

    # ── Hugging Face — Falcon-Arabic-7B-Instruct (مجاني) ─────────────────

    def _call_huggingface(
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        """
        يستدعي Falcon-Arabic-7B-Instruct عبر HF Inference API (المجانية).
        النموذج نفسه نموذج لغوي عربي عام (غير متخصص دينياً)، مبني على
        Falcon3-7B من TII. نبني الـ prompt يدوياً بصيغة نصية بسيطة لأن
        الـ Inference API القديمة (serverless) تستخدم "text-generation"
        وليس "chat/completions".
        """
        # بناء prompt نصي واحد يضمّ التعليمات + آخر جولات المحادثة + السؤال
        parts = [system_prompt.strip()]
        for u, a in history[-4:]:
            parts.append(f"المستخدم: {u}\nالمساعد: {a}")
        parts.append(f"المستخدم: {query}\nالمساعد:")
        prompt = "\n\n".join(parts)

        data = _post_json(
            _HF_INFERENCE_URL,
            {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": self.max_tokens,
                    "temperature":    max(self.temperature, 0.01),
                    "return_full_text": False,
                },
                "options": {"wait_for_model": True},
            },
            {
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type":  "application/json",
            },
            self.timeout,
        )

        # صيغة الاستجابة المعتادة: [{"generated_text": "..."}]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            text = data[0]["generated_text"].strip()
        elif isinstance(data, dict) and "generated_text" in data:
            text = data["generated_text"].strip()
        else:
            raise Exception(f"صيغة استجابة غير متوقعة من HF: {str(data)[:150]}")

        if not text:
            raise Exception("HF أعاد نصاً فارغاً")

        return FallbackResult(
            text=text, provider=Provider.HUGGINGFACE, model=self._model,
        )

    # ── نموذج محلي عبر Ollama (نشر مغلق بدون إنترنت) ────────────────────

    def _call_local(
        self, query: str, history: List[Tuple[str, str]],
        system_prompt: str = _SYSTEM_PROMPT,
    ) -> FallbackResult:
        """
        يستدعي نموذجاً محلياً عبر خادم Ollama (أو أي خادم متوافق مع
        /api/chat بنفس الصيغة) — لا يخرج أي اتصال خارج شبكة الجهة المضيفة.
        عنوان الخادم قابل للتهيئة عبر NSM_LOCAL_LLM_URL (افتراضي:
        http://localhost:11434)، والنموذج عبر NSM_LOCAL_MODEL.
        """
        base_url = os.getenv("NSM_LOCAL_LLM_URL", _LOCAL_BASE_URL).strip().rstrip("/") or _LOCAL_BASE_URL
        messages = [{"role": "system", "content": system_prompt}]
        for u, a in history[-4:]:
            messages += [
                {"role": "user",      "content": u},
                {"role": "assistant", "content": a},
            ]
        messages.append({"role": "user", "content": query})

        timeout = max(self.timeout, _LOCAL_TIMEOUT_SEC)
        data = _post_json(
            f"{base_url}/api/chat",
            {
                "model":    self._model,
                "messages": messages,
                "stream":   False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            },
            {"Content-Type": "application/json"},
            timeout,
        )

        text = (data.get("message") or {}).get("content", "").strip()
        if not text:
            raise Exception(f"النموذج المحلي أعاد استجابة فارغة/غير متوقعة: {str(data)[:150]}")

        return FallbackResult(
            text=text, provider=Provider.LOCAL, model=self._model,
        )
