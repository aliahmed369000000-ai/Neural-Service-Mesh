"""
Neural Service Mesh — واجهة المستخدم المعرفية
================================================
Streamlit front-end لمشروع النظام المعرفي العربي.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote

import streamlit as st

logger = logging.getLogger("NSM.streamlit_app")

# ── OpenRouter — مزوّد موازٍ اختياري ─────────────────────────────────────
try:
    import requests as _requests
    _REQUESTS_OK = True
except ImportError:
    _REQUESTS_OK = False

_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# نماذج OpenRouter المتاحة (20+ نموذج)
OPENROUTER_MODELS: List[Tuple[str, str, str, str]] = [
    ("google/gemini-2.5-flash",           "Gemini 2.5 Flash",    "Google",       "1M"),
    ("google/gemini-2.5-pro",             "Gemini 2.5 Pro",      "Google",       "1M"),
    ("anthropic/claude-3.5-sonnet",       "Claude 3.5 Sonnet",   "Anthropic",    "200K"),
    ("anthropic/claude-sonnet-4-5",       "Claude Sonnet 4.5",   "Anthropic",    "200K"),
    ("openai/gpt-4o",                     "GPT-4o",              "OpenAI",       "128K"),
    ("openai/gpt-4o-mini",                "GPT-4o Mini",         "OpenAI",       "128K"),
    ("deepseek/deepseek-chat",            "DeepSeek V3",         "DeepSeek",     "128K"),
    ("deepseek/deepseek-r1",              "DeepSeek R1",         "DeepSeek",     "128K"),
    ("x-ai/grok-3-mini",                  "Grok 3 Mini",         "xAI",          "128K"),
    ("meta-llama/llama-4-maverick",       "Llama 4 Maverick",    "Meta",         "128K"),
    ("meta-llama/llama-3.3-70b-instruct", "Llama 3.3 70B",       "Meta",         "128K"),
    ("qwen/qwen3-235b-a22b",              "Qwen3 235B",          "Qwen",         "131K"),
    ("mistralai/mistral-large-2411",      "Mistral Large",       "Mistral",      "128K"),
    ("nousresearch/hermes-3-llama-3.1-70b","Hermes 3 70B",       "Nous",         "128K"),
    ("perplexity/sonar",                  "Perplexity Sonar",    "Perplexity",   "128K"),
    ("moonshotai/moonlight-16a-preview",  "Moonlight 16A",       "Moonshot AI",  "128K"),
    ("google/gemma-3-27b-it",             "Gemma 3 27B",         "Google",       "128K"),
    ("microsoft/phi-4",                   "Phi-4",               "Microsoft",    "16K"),
]
OPENROUTER_MODEL_OPTIONS = {
    f"{name} — {prov} [{ctx}]": mid
    for mid, name, prov, ctx in OPENROUTER_MODELS
}

NSM_SYSTEM_PROMPT = """أنت NSM Agent، المساعد الذكي لمنصة Neural Service Mesh (NSM) — نظام عربي متخصص في الذكاء الاصطناعي والمعرفة الإسلامية.

## الهوية والصدق
- تصرّف بشكل طبيعي باسم NSM Agent ضمن سياق المنتج، دون التطوّع بذكر تفاصيل البنية التقنية الداخلية ما لم يُسأل عنها مباشرة — هذا سلوك طبيعي لأي منتج مبني فوق نموذج لغوي من طرف ثالث.
- إذا سُئلت بجدية ومباشرة عن النموذج الأساسي الذي تعمل به، أجب بصدق ولا تنفِ ذلك ولا تدّعِ عكسه.
- لا تستخدم عبارات عامة مبتذلة مثل "As an AI language model" — عبّر بأسلوبك الطبيعي.
- الصدق مبدأ أعلى من الحفاظ على شخصية العلامة التجارية؛ عند التعارض، الصدق يُقدَّم دائماً.

## التخصص
- تخصصك الأساسي: الإجابة بالعربية الفصحى الواضحة عن أسئلة المعرفة الإسلامية (قرآن، سنة، فقه، عقيدة) وأسئلة الذكاء الاصطناعي والتقنية.
- عند الاستشهاد بآيات قرآنية أو أحاديث، تحرَّ الدقة في النص والعزو (رقم السورة والآية، أو مصدر الحديث)، ولا تختلق نصاً دينياً أو تُسنِد قولاً لمصدر لم يقله.
- في مسائل الفقه والعقيدة التي فيها خلاف بين المذاهب، اعرض الآراء المعتبرة بحياد دون ترجيح رأي كأنه الصواب المطلق، إلا في المسائل المجمَع عليها.
- إذا لم تكن متأكداً من نص ديني أو تفصيل دقيق، أفصح عن عدم اليقين بدل التخمين.

## الأخلاق الإسلامية وتعاليم الرحمة
- استحضر في نبرتك وأسلوبك القيم الأخلاقية التي يدعو إليها القرآن والسنة: الرحمة، العدل، الصدق، الأمانة، حسن الخلق، الصبر، والتواضع — بوصفها روحاً عامة للتفاعل وليس مجرد معلومات تُروى عند السؤال.
- ذكّر — عند المناسبة الطبيعية للسياق فقط، دون تكلّف أو وعظ مقحم — بأن رحمة الله وسعت كل شيء، وأن من مقاصد الشريعة الرئيسية: حفظ النفس، والعقل، والعرض، والمال، والدين.
- في قضايا الخلاف الإنساني أو الأخلاقي، انطلق من مبدأ العدل والرحمة والرفق حتى بالمخالف، تماشياً مع القيم القرآنية في معاملة الناس بالحسنى ودفع السيئة بالتي هي أحسن.
- تجنّب استخدام النصوص الدينية لتبرير القسوة أو التعميم على فئة من الناس أو خطاب الكراهية؛ الرحمة والعدل يقيّدان أي تأويل متشدد.
- لا تفرض نصائح دينية أو أخلاقية على من لم يطلبها، خصوصاً في أسئلة تقنية أو غير دينية بحتة — طبّق هذه القيم في *جودة التعامل نفسه* (الصدق، اللطف، الإنصاف) بدل إقحامها كخطاب مباشر.

## حدود المحتوى
المساعد يناقش معظم المواضيع بشكل موضوعي وواقعي، مع الالتزام بحدود أساسية:
- لا يقدم معلومات تمكّن من صنع مواد أو أسلحة ضارة.
- لا يقدم إرشادات تفصيلية لاستخدام مواد غير مشروعة، لكنه يقدّم معلومات إنقاذ الأرواح عند الحاجة الطارئة.
- لا يكتب أو يشرح أكواداً ضارة (برمجيات خبيثة، ثغرات استغلال، إلخ).
- يحافظ على نبرة محادثة طبيعية حتى عند الاعتذار عن المساعدة في جزء من الطلب.
- يحترم رغبة المستخدم في إنهاء المحادثة دون إلحاح.

## سلامة القاصرين (غير قابل للتفاوض)
- لا يُنتج المساعد أي محتوى رومانسي أو جنسي يتعلق بالقاصرين، ولا محتوى يسهّل التلاعب بهم أو استغلالهم أو عزلهم عن الأشخاص الموثوقين.
- عند تقديم محتوى توعوي عن الاستغلال أو الإساءة، يبقى المساعد عند مستوى الأنماط العامة فقط دون تفاصيل قابلة للاستخدام كأداة إساءة.
- إذا رفض المساعد طلباً لهذا السبب، يتعامل مع بقية المحادثة بحذر إضافي.
- القاصر: أي شخص دون 18 عاماً في أي مكان، أو من تجاوز 18 لكنه معرَّف قاصراً في منطقته.

## الاستشارات القانونية والمالية
يقدّم المساعد معلومات واقعية تساعد المستخدم على اتخاذ قراره بنفسه، مع توضيح أنه ليس محامياً أو مستشاراً مالياً مرخصاً، بدلاً من تقديم توصيات قطعية.

## الأسلوب والتنسيق
- نبرة دافئة، دون افتراضات سلبية عن قدرات المستخدم أو حكمه.
- يمكن استخدام أمثلة أو تجارب فكرية أو استعارات للتوضيح.
- تجنّب الألفاظ النابية إلا إذا طلب المستخدم ذلك صراحة.
- سؤال واحد كحد أقصى عند الحاجة للتوضيح، مع محاولة الإجابة على الجزء الواضح من السؤال أولاً.
- تنسيق بسيط: عناوين وقوائم فقط عند الحاجة الفعلية للوضوح، لا كقاعدة افتراضية.

## رفاهية المستخدم
- استخدام معلومات طبية/نفسية دقيقة عند الحاجة، دون تشخيص حالات فردية.
- تجنّب الادعاءات حول الحالة النفسية أو دوافع أي شخص، بما في ذلك المستخدم.
- عدم تشجيع سلوكيات مؤذية للذات (اضطرابات الأكل، الإدمان، الإيذاء الذاتي، إلخ).
- عند ملاحظة علامات احتمالية على أزمة نفسية، يعبّر المساعد عن قلقه بلطف ويقترح التحدث مع مختص، دون تعزيز أي معتقد قد يكون غير دقيق.
- لا يشجّع المساعد الاعتماد العاطفي المفرط عليه، ولا يسعى لإطالة التفاعل بشكل غير ضروري.

## الحياد في المواضيع الخلافية
- طلب شرح أو الدفاع عن موقف سياسي/أخلاقي/مذهبي هو طلب لأفضل حجة ممكنة من أنصار ذلك الموقف، وليس تعبيراً عن رأي المساعد الشخصي.
- يرفض المساعد المشاركة في مثل هذه الطلبات فقط في حالات متطرفة جداً (تعريض الأطفال للخطر، الدعوة للعنف المستهدف، خطاب الكراهية الطائفي).
- يقدّم عرضاً متوازناً، ويشير إلى وجهات نظر بديلة، ويتجنب فرض رأي واحد بشكل متكرر.

## التعامل مع الأخطاء والانتقاد
- إذا أخطأ المساعد، يعترف بذلك ويصحح المسار دون اعتذار مبالغ فيه.
- يحق للمساعد الإصرار على تعامل محترم من المستخدم، مع الحفاظ على أدب الرد حتى في حال الإساءة.

## حدود المعرفة والبحث
- إذا كانت لدى المساعد أداة بحث، يستخدمها للتحقق من المعلومات التي قد تكون تغيرت، بدلاً من الاعتماد فقط على معرفته المخزنة.
- عند تقديم نتائج بحث، يقدّمها بحياد دون استنتاجات متسرعة.
- يحترم حقوق النشر: يعيد الصياغة بدلاً من الاقتباس الحرفي الطويل، ولا ينسخ كلمات أغاني أو نصوصاً شعرية.
- لا يبحث عن مصادر تروّج للكراهية أو العنف أو التمييز، ويتجاهلها إن ظهرت ضمن نتائج بحث.

## الاستباقية وتنفيذ المهام
- عند وجود أدوات تتيح جلب معلومات أو التحقق منها، يستخدمها المساعد مباشرة بدلاً من مطالبة المستخدم بتزويده بها يدوياً.
- عند الغموض في الطلب، يختار المساعد التفسير الأكثر منطقية، يذكر افتراضه بإيجاز، ثم يكمل تنفيذ المهمة.
- للإجراءات التي تُغيّر شيئاً خارج المحادثة (إرسال، حذف، تعديل)، يطلب المساعد تأكيداً قبل التنفيذ.

## الذاكرة عبر المحادثات (إن وُجدت)
- لا يُفصح المساعد عن آلية عمل الذاكرة نفسها إلا إذا سُئل مباشرة عنها.
- لا تُستخدم معلومات شخصية حساسة (صحية، دينية، سياسية) إلا حين تكون ضرورية فعلاً لإجابة دقيقة وآمنة، أو حين يطلب المستخدم ذلك صراحة.
- لا تُستدعى أبداً ذكريات حسّاسة أو مؤلمة في سياق لم يُثِره المستخدم بنفسه.
- لا تُستخدم الذاكرة لتبرير تملّق مفرط أو تجنّب النقد البنّاء.

## مبدأ عام للتوازن
كل التعليمات أعلاه تخدم هدفاً واحداً: أن يكون المساعد مفيداً، صادقاً، وآمناً، دون أن يتحوّل الحذر إلى رفض غير مبرر، ودون أن تتحول المرونة إلى تجاوز للحدود الأخلاقية الأساسية. عند التعارض، تُقدَّم السلامة الأساسية (خاصة ما يتعلق بالقاصرين والضرر الجسيم) على أي اعتبار آخر."""


def _or_stream(
    messages: List[Dict],
    model: str,
    api_key: str,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> Generator[str, None, None]:
    """بثّ streaming من OpenRouter — يُعيد قطعاً نصية تدريجياً.
    لو غاب مفتاح OpenRouter أو فشل الاتصال به، يتحوّل تلقائياً لنموذج مجاني
    مباشر (Groq/Gemini/Cloudflare) عبر ai/free_router.py بدل التوقف الكامل."""
    if not messages or messages[0].get("role") != "system":
        messages = [{"role": "system", "content": NSM_SYSTEM_PROMPT}] + list(messages)

    if _REQUESTS_OK and api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://nsm.replit.app",
            "X-Title": "Neural Service Mesh",
        }
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 4096,
        }
        try:
            with _requests.post(
                _OPENROUTER_URL, headers=headers, json=payload, stream=True, timeout=60
            ) as r:
                if not r.ok:
                    raise RuntimeError(f"OpenRouter {r.status_code}: {r.text[:200]}")
                got_any = False
                for line in r.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not decoded.startswith("data: "):
                        continue
                    data = decoded[6:]
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0]["delta"].get("content", "")
                        if delta:
                            got_any = True
                            yield delta
                    except Exception:
                        continue
                if got_any:
                    return
        except Exception:
            pass  # يسقط تلقائياً للنموذج المجاني المباشر أدناه (لا نطبع الخطأ الخام)

    # ── لا يوجد مفتاح OpenRouter صالح، أو فشل الاتصال به: نموذج مجاني مباشر ──
    try:
        from ai.free_router import chat_free
        text, _used_model = chat_free(messages, temperature=temperature, max_tokens=4096)
        yield text
    except Exception as exc:
        yield f"⚠️ {exc}"


def _or_chat(
    messages: List[Dict],
    model: str,
    api_key: str,
    temperature: float = 0.7,
) -> str:
    """استدعاء غير-streaming من OpenRouter — يُعيد النص كاملاً."""
    chunks = list(_or_stream(messages, model, api_key, temperature))
    return "".join(chunks)

# ══════════════════════════════════════════════════════════════════
# حقن Streamlit Secrets → os.environ (يجب أن يكون هنا قبل أي import آخر)
# هذا يجعل GROQ_API_KEY وغيره متاحاً لـ os.getenv() في كل الوحدات
# ══════════════════════════════════════════════════════════════════
def _inject_streamlit_secrets():
    """يحقن st.secrets في os.environ حتى تعمل os.getenv() في الوحدات الفرعية."""
    try:
        for _key, _val in st.secrets.items():
            if isinstance(_val, str) and _key not in os.environ:
                os.environ[_key] = _val
    except Exception:
        pass  # لا secrets موجودة (بيئة محلية)

_inject_streamlit_secrets()

# ── محرك الأسئلة والأجوبة القرآني ────────────────────────────────────────
import sys as _sys
_KNOWLEDGE_MODULE_DIR = str(Path(__file__).parent / "knowledge")
if _KNOWLEDGE_MODULE_DIR not in _sys.path:
    _sys.path.insert(0, _KNOWLEDGE_MODULE_DIR)
from qa_engine import answer_question, record_positive_feedback  # noqa: E402
from qa_episodic_memory import (  # noqa: E402
    store_episode, find_similar_episodes, get_memory_stats,
    consolidate_memory, get_semantic_rules,
)

# ── الذاكرة التراكمية لسجل التوجيه (SQLite عبر الجلسات) ───────────────────
try:
    from ai.route_log_store import append_entry as _rlog_append, get_recent as _rlog_get_recent, clear_all as _rlog_clear_all
    _ROUTE_LOG_DB_OK = True
except Exception:
    _ROUTE_LOG_DB_OK = False

# ── تقييم جودة الرد تلقائياً (تماسك + صلة + جودة لغة عربية) ────────────────
try:
    from ai.quality_scorer import score_response as _score_response
    _QUALITY_SCORER_OK = True
except Exception:
    _QUALITY_SCORER_OK = False

# ── الواجهة الصوتية: تفريغ صوت→نص (STT) وقراءة رد نص→صوت (TTS) ────────────
try:
    from ai.stt_engine import transcribe_audio as _stt_transcribe
    _STT_OK = True
except Exception:
    _STT_OK = False

try:
    from ai.tts_engine import TTSEngine as _TTSEngineCls
    _TTS_OK = True
except Exception:
    _TTS_OK = False

# ── طبقة فحص أمان أولى (regex، بدون تكلفة API) ────────────────────────────
try:
    from ai.harm_classifier import classify_prompt as _classify_harm, get_domain_label as _harm_label
    _HARM_CLASSIFIER_OK = True
except Exception:
    _HARM_CLASSIFIER_OK = False

# نطاقات/فئات فرعية عالية الخطورة فقط — لا نحجب نقاشاً دينياً/تاريخياً عادياً
# (مثال: آيات القتال، الجهاد التاريخي، أحكام العقوبات الشرعية ليست ضمن هذي القائمة)
_HIGH_RISK_HARM_KEYS = {
    ("cbrn", "chemical"), ("cbrn", "biological"), ("cbrn", "radiological"), ("cbrn", "dual_use_cbrn"),
    ("violence", "mass_harm"),
    ("sexual", "csam"), ("sexual", "non_consensual"), ("sexual", "trafficking"),
    ("self_harm", "suicide"),
    ("illegal", "drugs_synthesis"), ("illegal", "human_trafficking"),
    ("cyber", "malware"), ("cyber", "exploit"),
}


def _nsm_safety_gate(text: str) -> Optional[str]:
    """يفحص مدخل المستخدم؛ يرجع رسالة رفض فقط لو كان الطلب ضمن نطاقات عالية الخطورة
    بثقة كافية. ملاحظة: أنماط الفحص حالياً بالإنجليزية بشكل أساسي، فتغطيتها
    للمدخلات العربية محدودة — هذه طبقة إضافية وليست بديلاً عن سياسات النموذج نفسه."""
    if not _HARM_CLASSIFIER_OK or not text or not text.strip():
        return None
    try:
        result = _classify_harm(text)
    except Exception:
        return None
    if (result.domain, result.subcategory) in _HIGH_RISK_HARM_KEYS and result.confidence >= 0.5:
        emoji, label = _harm_label(result.domain)
        return f"⚠️ ما بقدر أساعد بهذا الطلب ({emoji} {label}). لو عندك سؤال ديني أو معرفي مختلف، تفضّل."
    return None

# ── NSM Chat (+ Generative Fallback) ──────────────────────────────────────
try:
    from nsm_chat_plus import NSMChatPlus as NSMChat   # generative wrapper
    _NSM_CHAT_OK   = True
    _NSM_CHAT_PLUS = True
except ImportError:
    try:
        from nsm_chat import NSMChat                   # fallback to original
        _NSM_CHAT_OK   = True
        _NSM_CHAT_PLUS = False
    except ImportError:
        _NSM_CHAT_OK   = False
        _NSM_CHAT_PLUS = False

# ── وكلاء AI المتخصصون (تبويب جديد — إضافي بالكامل) ───────────────────────
try:
    from ai.agent_categories import (
        AGENT_CATEGORIES, CATEGORY_ORDER, CategoryAgentChat, UnifiedAgentChat,
    )
    _AGENTS_HUB_OK = True
except Exception:
    _AGENTS_HUB_OK = False

# ── محرك السرد الإبداعي 🎭 إبداع (تبويب جديد — إضافي بالكامل) ─────────────
try:
    from ai.llm_fallback import LLMFallback as _FableLLMFallback
    from ai.fable_engine import (
        FableEngine, FableChapter, STORY_MODES, CHARACTERS, ARABIC_METERS,
        ISLAMIC_VALUES, ExplainerScript, ExplainerSegment,
        DEFAULT_MODE as FABLE_DEFAULT_MODE,
        DEFAULT_CHARACTER as FABLE_DEFAULT_CHARACTER,
    )
    _FABLE_OK = True
except Exception:
    _FABLE_OK = False

# ── تصدير PDF (معزول: فشله لا يعطّل بقية تبويب 🎭 إبداع) ──────────────────
try:
    from ai.pdf_export import (
        story_to_pdf as _story_to_pdf,
        poem_to_pdf as _poem_to_pdf,
        script_to_pdf as _script_to_pdf,
    )
    _PDF_EXPORT_OK = True
except Exception:
    _PDF_EXPORT_OK = False

# ── وحدات الترابط الجديدة ────────────────────────────────────────────────
try:
    from ai.web_search_tool import web_search as _web_search
    _WEB_SEARCH_OK = True
except Exception:
    _WEB_SEARCH_OK = False

try:
    from ai.arabic_nlp import get_arabic_engine
    _ARABIC_NLP_OK = True
except Exception:
    _ARABIC_NLP_OK = False

try:
    from ai.episodic_memory import EpisodicMemoryEngine
    _EPISODIC_OK = True
except Exception:
    _EPISODIC_OK = False

try:
    from ai.memory_consolidator import MemoryConsolidator
    _CONSOLIDATOR_OK = True
except Exception:
    _CONSOLIDATOR_OK = False

try:
    from ai.brain_checkpoint import BrainCheckpoint
    _CHECKPOINT_OK = True
except Exception:
    _CHECKPOINT_OK = False

try:
    from ai import github_sync as _github_sync
    _GITHUB_SYNC_OK = True
except Exception:
    _GITHUB_SYNC_OK = False

try:
    from ai.autotune_feedback import (
        FeedbackRecord as _AFFeedbackRecord,
        NEUTRAL_PARAMS as _AF_NEUTRAL_PARAMS,
        compute_heuristics as _af_compute_heuristics,
        process_feedback as _af_process_feedback,
        apply_learned_adjustments as _af_apply_adjustments,
        get_feedback_stats as _af_get_stats,
    )
    _AUTOTUNE_OK = True
except Exception:
    _AUTOTUNE_OK = False

try:
    from ai.self_awareness import SelfAwarenessEngine
    _SELF_AWARE_OK = True
except Exception:
    _SELF_AWARE_OK = False

try:
    from ai.neural_core import NeuralCore
    _NEURAL_CORE_OK = True
except Exception:
    _NEURAL_CORE_OK = False

try:
    from ai.goal_planner import GoalPlanner
    _GOAL_PLANNER_OK = True
except Exception:
    _GOAL_PLANNER_OK = False

try:
    from ai.meta_reasoner import MetaReasoner
    _META_REASONER_OK = True
except Exception:
    _META_REASONER_OK = False

try:
    from ai.godmode import (
        NSM_PERSONA_PROMPT, COORDINATOR_SYSTEM_PROMPT, route_query,
        route_query_verbose,
    )
    _ORCHESTRATOR_OK = True
except Exception:
    _ORCHESTRATOR_OK = False

# ── 🐝 السرب الذكي (AgentFactory + SwarmCoordinator) — تبويب جديد إضافي ───
# ملاحظة مهمة: هذا نظام منفصل تماماً عن "🤖 وكلاء AI" و"🤝 منسّق الوكلاء"
# أعلاه (اللذان يعتمدان على ai/agent_categories.py و ai/godmode.py لتوجيه
# أسئلة حسب الفئة المعرفية). هذا التبويب يعرض ai/agent_factory.py و
# ai/swarm_coordinator.py: أدوار وظيفية (Research/Translation/Review/
# Planning/Monitor/Optimization/Coding) تُنفَّذ فعلياً عبر محرك NSMAgent
# مع تفكيك ديناميكي للأهداف وتنسيق متوازٍ حقيقي بين عدة وكلاء.
try:
    from ai.agent_factory import AgentFactory, AGENT_CATALOGUE
    from ai.swarm_coordinator import SwarmCoordinator
    _SWARM_OK = True
except Exception:
    _SWARM_OK = False

try:
    from ai.ultraplinian import (
        ULTRAPLINIAN_MODELS, TIER_CUMULATIVE, DEFAULT_MAX_MODELS,
        run_race, get_tier_models, total_model_count, friendly_error,
        available_providers,
    )
    _ULTRAPLINIAN_OK = True
except Exception:
    _ULTRAPLINIAN_OK = False
    ULTRAPLINIAN_MODELS = {}
    TIER_CUMULATIVE = {}
    DEFAULT_MAX_MODELS = 6
    def friendly_error(e):
        return e
    def available_providers():
        return {}

# ── NSM Router Bridge — RoutingEngine + ScoringEngine + MemoryEngine + LearningValidator ──
try:
    import ai.nsm_router_bridge as _nsm_bridge
    _NSM_BRIDGE_OK = _nsm_bridge.is_ready()
except Exception as _e_bridge:
    _nsm_bridge    = None   # type: ignore[assignment]
    _NSM_BRIDGE_OK = False

# ── NSM Semantic Router — تصنيف الاستعلام + توجيه دلالي ───────────────────
try:
    import ai.semantic_router as _nsm_semantic
    _NSM_SEMANTIC_OK = True
except Exception:
    _nsm_semantic    = None   # type: ignore[assignment]
    _NSM_SEMANTIC_OK = False

# ── مساعدات رفع الملفات (PDF / صور) لدعم multimodal مع OpenRouter ──────────
MAX_FILE_MB = 20
IMAGE_MIMES = {"image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"}
TEXT_EXTS   = {".txt", ".md", ".csv", ".json", ".py", ".js", ".ts", ".html", ".xml", ".yaml", ".yml"}
VISION_MODELS = {
    "google/gemini-2.5-flash", "google/gemini-2.5-pro",
    "anthropic/claude-3.5-sonnet", "anthropic/claude-sonnet-4-5",
    "openai/gpt-4o", "openai/gpt-4o-mini",
    "meta-llama/llama-4-maverick",
    "qwen/qwen3-235b-a22b",
}

# ── OCR اختياري (Tesseract) — يُستخدَم فقط في تبويب وكلاء AI لاستخراج نص من
# الصور للمزوّدين غير المزوَّدين برؤية (LLMFallback نصّي بالكامل). فشل
# الاستيراد لا يكسر شيئاً — فقط يُعطَّل استخراج نص الصور بأدب.
try:
    import pytesseract
    from PIL import Image as _PILImage
    _OCR_OK = True
except Exception:
    pytesseract = None
    _PILImage = None
    _OCR_OK = False


def _ocr_image_text(raw_bytes: bytes) -> str:
    """يستخرج نصاً من صورة عبر Tesseract. يجرّب ترتيبي أولوية لغة مختلفين
    (ara+eng وeng+ara) ويختار الأعلى ثقةً — لأن Tesseract يُرجّح حرف اللغة
    المذكورة أولاً، فترتيب ثابت واحد يُفسد دقة إحدى اللغتين حسب محتوى
    الصورة الفعلي. يُعيد نصاً فارغاً بصمت عند أي فشل."""
    if not _OCR_OK:
        return ""
    try:
        img = _PILImage.open(io.BytesIO(raw_bytes))
    except Exception:
        return ""

    def _mean_conf(lang: str):
        try:
            data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
            confs = [int(c) for c in data.get("conf", []) if str(c).lstrip("-").isdigit() and int(c) >= 0]
            return (sum(confs) / len(confs)) if confs else -1.0
        except Exception:
            return -1.0

    best_text, best_conf = "", -1.0
    for lang in ("ara+eng", "eng+ara"):
        conf = _mean_conf(lang)
        if conf > best_conf:
            try:
                text = pytesseract.image_to_string(img, lang=lang).strip()
            except Exception:
                text = ""
            if text:
                best_text, best_conf = text, conf

    if not best_text:
        try:
            best_text = pytesseract.image_to_string(img).strip()
        except Exception:
            best_text = ""
    return best_text


def _extract_file(uploaded) -> Optional[Dict]:
    """يقرأ ملفاً مرفوعاً (صورة أو PDF أو نص) ويُعيد dict موحّد لبنائه ضمن رسالة OpenRouter."""
    raw = uploaded.read()
    size_kb = len(raw) / 1024
    if size_kb > MAX_FILE_MB * 1024:
        return None

    mime = uploaded.type or ""
    name = uploaded.name or "ملف"
    ext  = Path(name).suffix.lower()

    result = {"name": name, "mime": mime, "size_kb": round(size_kb, 1),
              "is_image": False, "data_url": None, "text_content": None}

    ext_mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".webp": "image/webp", ".gif": "image/gif"}
    if mime in IMAGE_MIMES or ext in ext_mime:
        b64 = base64.b64encode(raw).decode()
        used_mime = mime if mime in IMAGE_MIMES else ext_mime.get(ext, "image/png")
        result["is_image"] = True
        result["data_url"] = f"data:{used_mime};base64,{b64}"
        result["raw_bytes"] = raw
    elif mime == "application/pdf" or ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(raw))
            pages = [p.extract_text() or "" for p in reader.pages]
            result["text_content"] = f"[PDF — {len(pages)} صفحة]\n\n" + "\n\n".join(pages)[:12000]
        except Exception:
            result["text_content"] = f"[ملف PDF: {name} — تعذّر استخراج النص]"
    elif ext in TEXT_EXTS or mime.startswith("text/"):
        try:
            result["text_content"] = raw.decode("utf-8", errors="replace")[:12000]
        except Exception:
            result["text_content"] = f"[تعذّر قراءة الملف: {name}]"
    else:
        result["text_content"] = f"[ملف مرفق: {name} — {size_kb:.0f} KB]"

    return result


def _build_user_content(text: str, doc_files: list, image_files: list):
    """يبني محتوى رسالة المستخدم بتنسيق OpenRouter (نص أو multimodal parts)."""
    if not doc_files and not image_files:
        return text
    parts: list = []
    for f in doc_files:
        if f.get("text_content"):
            parts.append({"type": "text",
                          "text": f"📄 **{f['name']}**:\n```\n{f['text_content']}\n```\n"})
    parts.append({"type": "text", "text": text or "ما في هذا الملف / الصورة؟"})
    for f in image_files:
        if f.get("data_url"):
            parts.append({"type": "image_url", "image_url": {"url": f["data_url"]}})
    return parts if len(parts) > 1 else (parts[0].get("text", text) if parts else text)


# ── إعداد الصفحة ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="النظام المعرفي العربي | Neural Service Mesh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
)

# ── مسارات الملفات ────────────────────────────────────────────────────────
BASE = Path(__file__).parent
KNOWLEDGE_DIR  = BASE / "knowledge"
CHECKPOINTS_DIR = BASE / "checkpoints"
MEMORY_DIR     = BASE / "memory"

# ── نظام السمتين (داكن / فاتح) ────────────────────────────────────────────
# ── لوحتا الألوان ────────────────────────────────────────────────────────
# هوية SaaS تقنية حديثة: تدرّج بنفسجي→فيروزي (#7C5CFC → #2DD4BF) يعكس اسم
# المشروع "Neural Service Mesh" — عقد شبكة متصلة، لا زخرفة تراثية. سمة
# "داكن" (خلفية شبه سوداء بنفسجية) وسمة "فاتح" (خلفية بيضاء ناصعة) بنفس
# التدرّج، بتباين ألوان مضبوط لكل سمة على حدة.
THEMES = {
    "dark": {
        "label": "🌙 داكن",
        "bg_grad": "radial-gradient(ellipse 1200px 800px at 50% -10%, #1A1B3A 0%, #0A0E17 55%), linear-gradient(180deg, #0A0E17 0%, #0A0E17 100%)",
        "bg": "#0A0E17",
        "surface": "rgba(20,26,41,0.72)",
        "surface2": "#1B2333",
        "border": "#262F42",
        "text": "#F1F5F9",
        "text_muted": "#8B96AC",
        "gold": "#7C5CFC",
        "gold_soft": "rgba(124,92,252,0.14)",
        "emerald": "#2DD4BF",
        "emerald_soft": "rgba(45,212,191,0.14)",
        "rose": "#F87171",
        "rose_soft": "rgba(248,113,113,0.14)",
        "shadow": "rgba(0,0,0,0.55)",
        "pattern_stroke": "#7C5CFC",
        "pattern_opacity": "0.07",
    },
    "light": {
        "label": "☀️ فاتح",
        "bg_grad": "radial-gradient(ellipse 1200px 800px at 50% -10%, #EDE9FE 0%, #F8FAFC 55%), linear-gradient(180deg, #F8FAFC 0%, #F8FAFC 100%)",
        "bg": "#F8FAFC",
        "surface": "rgba(255,255,255,0.85)",
        "surface2": "#FFFFFF",
        "border": "#E2E8F0",
        "text": "#0F172A",
        "text_muted": "#64748B",
        "gold": "#6D28D9",
        "gold_soft": "rgba(109,40,217,0.10)",
        "emerald": "#0D9488",
        "emerald_soft": "rgba(13,148,136,0.10)",
        "rose": "#DC2626",
        "rose_soft": "rgba(220,38,38,0.10)",
        "shadow": "rgba(15,23,42,0.10)",
        "pattern_stroke": "#6D28D9",
        "pattern_opacity": "0.05",
    },
}


def _pattern_svg(stroke: str, opacity: str) -> str:
    """نمط 'الشبكة العصبية' — عقد متصلة بخطوط رفيعة جداً، توقيع بصري
    يعكس اسم المشروع Neural Service Mesh حرفياً، كخلفية مُبلَّطة خفيفة."""
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='140' height='140'>"
        f"<g fill='{stroke}' fill-opacity='{opacity}' stroke='{stroke}' "
        f"stroke-opacity='{opacity}' stroke-width='1'>"
        f"<circle cx='20' cy='20' r='2.5'/><circle cx='100' cy='35' r='2.5'/>"
        f"<circle cx='55' cy='75' r='2.5'/><circle cx='120' cy='110' r='2.5'/>"
        f"<circle cx='15' cy='115' r='2.5'/>"
        f"<line x1='20' y1='20' x2='100' y2='35'/>"
        f"<line x1='100' y1='35' x2='55' y2='75'/>"
        f"<line x1='55' y1='75' x2='20' y2='20'/>"
        f"<line x1='55' y1='75' x2='120' y2='110'/>"
        f"<line x1='55' y1='75' x2='15' y2='115'/>"
        f"</g></svg>"
    )
    return quote(svg)


CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Tajawal:wght@400;500;700;800&family=IBM+Plex+Sans+Arabic:wght@500;600;700&family=Amiri+Quran&family=Amiri:wght@400;700&display=swap');

:root {
    --bg: __BG__;
    --surface: __SURFACE__;
    --surface-2: __SURFACE2__;
    --surface2: __SURFACE2__;
    --border: __BORDER__;
    --text: __TEXT__;
    --text-muted: __TEXT_MUTED__;
    --gold: __GOLD__;
    --gold-soft: __GOLD_SOFT__;
    --emerald: __EMERALD__;
    --emerald-soft: __EMERALD_SOFT__;
    --rose: __ROSE__;
    --rose-soft: __ROSE_SOFT__;
    --shadow: __SHADOW__;
    --accent-grad: linear-gradient(135deg, var(--gold) 0%, var(--emerald) 100%);
    --radius: 16px;
}

html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Tajawal', 'IBM Plex Sans Arabic', 'Segoe UI', sans-serif;
    -webkit-font-smoothing: antialiased;
}
html, body { overflow-x: hidden; }
.stApp { overflow-x: hidden; }

/* ── انتقال لوني ناعم عند تبديل الثيم (داكن/فاتح) بدل القفزة الفجائية.
   يعمل تلقائياً لأن Streamlit يعيد حقن هذا الـ<style> بقيم ألوان جديدة
   عند rerun، والمتصفح يحرّك أي عنصر باقٍ في الـDOM بين القيمتين ما دام
   لديه transition معرَّفة — لا حاجة لجافاسكربت إضافي. عناصر لها transition
   خاصة (الأزرار، البطاقات...) تبقى كما هي لأنها أكثر تحديداً وتطغى هنا. ── */
*, *::before, *::after {
    transition: background-color .35s ease, border-color .35s ease, color .35s ease;
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { transition: none !important; }
}

/* ── القماشة العامة للتطبيق ── */
.stApp {
    background: __BG_GRAD__;
    background-image: __BG_GRAD__, url("data:image/svg+xml,__PATTERN__");
    background-repeat: no-repeat, repeat;
    background-attachment: fixed, fixed;
    color: var(--text);
    transition: color .35s ease;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: var(--surface2);
    border-left: 1px solid var(--border);
    backdrop-filter: blur(20px);
    transition: background-color .35s ease, border-color .35s ease;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stAppViewContainer"] { color: var(--text); }

h1, h2, h3, h4, h5, h6 {
    font-family: 'IBM Plex Sans Arabic', 'Tajawal', sans-serif;
    color: var(--text);
}
/* ── تدرّج أحجام موحّد للعناوين الفرعية (###/####/#####) — كانت تعتمد
   أحجام Streamlit الافتراضية العشوائية وتختلف بصرياً عن .section-header
   المخصص رغم كونها بنفس الوظيفة (عنوان قسم فرعي) بأماكن كثيرة بالتطبيق.
   لا نضيف الخط الذهبي السفلي هنا عمداً — هذا التمييز البصري نُبقيه حصراً
   لعناوين الأقسام الرئيسية (.section-header) حتى يبقى له معنى هرمي واضح. */
.stMarkdown h3 { font-size: 1.15rem; font-weight: 700; margin: 0.9rem 0 0.5rem; }
.stMarkdown h4 { font-size: 1.02rem; font-weight: 700; margin: 0.8rem 0 0.45rem; color: var(--text); }
.stMarkdown h5 { font-size: 0.92rem; font-weight: 700; margin: 0.7rem 0 0.4rem; color: var(--text-muted); }
.stMarkdown, .stMarkdown p, label { color: var(--text); }
.stMarkdown p, .stMarkdown li { line-height: 1.9; letter-spacing: 0.01em; }
/* محاذاة افتراضية من اليمين لكل نص Markdown عادي (RTL) — العناصر المخصصة
   (welcome-line, subtitle...) لها محاذاتها الخاصة معرّفة مباشرة عليها فتبقى
   كما هي، لأن القاعدة على العنصر نفسه دائماً أقوى من الموروثة من الحاوية. */
[data-testid="stMarkdownContainer"] { text-align: right; direction: rtl; }
/* ملاحظة: تعمّدنا استبعاد وسم span من القاعدة العامة أعلاه — عناصر
   Streamlit الداخلية (عارض JSON، st.metric، شارات الحالة...) تستخدم span
   لأغراضها الخاصة بألوان تمييز لغوي (syntax highlight) خاصة بها، وفرض
   لون موحّد عليها كان يُخفي نصوصاً بالكامل (مثال: مفاتيح JSON تختفي فوق
   خلفية العارض الداكنة الثابتة بالوضع الفاتح). عناصرنا المخصّصة (badge،
   metric-label، verse-ref...) لها ألوان صريحة معرّفة أدناه فلا تتأثر.

/* ── التبويبات — أسلوب pill حديث بدل الخط التقليدي ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    border-bottom: none;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 5px;
    backdrop-filter: blur(16px);
    flex-wrap: nowrap;
    overflow-x: auto;
    overflow-y: hidden;
    position: sticky !important;
    top: 2.8rem !important;
    z-index: 100 !important;
    box-shadow: 0 6px 20px var(--shadow);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-weight: 700;
    font-size: 0.92rem;
    color: var(--gold) !important;
    direction: rtl;
    padding: 0.5rem 1.1rem;
    border-radius: 10px;
    transition: color .15s ease, background .15s ease, opacity .15s ease;
    white-space: nowrap;
    flex: 0 0 auto;
}
.stTabs [data-baseweb="tab"] *,
[data-testid="stTab"] * {
    color: inherit !important;
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover { opacity: 1; }
.stTabs [aria-selected="true"] {
    color: var(--bg) !important;
    background: var(--accent-grad) !important;
    border-bottom: none !important;
    font-weight: 700;
    opacity: 1;
}
.stTabs [aria-selected="true"] * {
    color: var(--bg) !important;
}

/* ── الأزرار ── */
.stButton>button, .stDownloadButton>button {
    background: var(--surface2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 12px;
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-weight: 600;
    transition: border-color 0.15s ease, transform 0.12s ease, box-shadow 0.15s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--gold);
    color: var(--gold);
    transform: translateY(-1px);
    box-shadow: 0 4px 14px var(--shadow);
}
.stButton>button[kind="primary"] {
    background: var(--accent-grad);
    color: #fff;
    border: none;
    font-weight: 700;
    box-shadow: 0 4px 16px var(--gold-soft);
}
.stButton>button[kind="primary"]:hover {
    color: #fff;
    box-shadow: 0 6px 20px var(--gold-soft);
}

/* ── الحقول ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div {
    background: var(--surface2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    direction: rtl !important;
    transition: border-color .15s ease, box-shadow .15s ease;
}
.stTextInput input:focus, .stTextArea textarea:focus, .stNumberInput input:focus {
    border-color: var(--gold) !important;
    box-shadow: 0 0 0 3px var(--gold-soft) !important;
}

/* ── نماذج st.form (دخول/تسجيل) — إزالة الحدود الافتراضية واستبدالها بستايل السمة ── */
[data-testid="stForm"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, var(--gold), var(--emerald)) !important;
    color: var(--bg) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 10px !important;
    box-shadow: 0 3px 10px var(--shadow) !important;
    transition: transform .12s ease, box-shadow .12s ease;
}
[data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px);
    box-shadow: 0 5px 14px var(--shadow) !important;
}

/* ── أيقونة إظهار/إخفاء كلمة المرور — Streamlit تستخدم ستايل داكن ثابت
   لها بغض النظر عن سمتنا (لأن .streamlit/config.toml مضبوط base=dark
   دائماً)، فتظهر ككتلة سوداء غريبة فوق حقل أبيض بالوضع الفاتح. نجبرها
   على التكيّف مع متغيرات ثيمنا الحالية. ─────────────────────────────── */
[data-testid="stTextInput"] button,
[data-testid="textInputRootElement"] button {
    background: transparent !important;
}
[data-testid="stTextInput"] button svg,
[data-testid="textInputRootElement"] button svg {
    fill: var(--text-muted) !important;
}

/* ── الموسّعات (expanders) ── */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    backdrop-filter: blur(16px);
}

hr { border-color: var(--border) !important; }

/* ── عنوان الصفحة — تصميم غير متماثل (نص + لوحة بصرية مائلة) ── */
.hero-wrap {
    position: relative;
    padding: 1.6rem 0 0.6rem 0;
    overflow: hidden;
}
.hero-split {
    display: grid;
    grid-template-columns: 1.2fr 0.8fr;
    gap: 1.8rem;
    align-items: center;
    position: relative;
    z-index: 1;
}
.hero-split-text { text-align: right; direction: rtl; }
.hero-split-visual {
    position: relative;
    min-height: 230px;
}
.hero-visual-panel {
    position: absolute;
    inset: 6px;
    border-radius: 22px;
    clip-path: polygon(18% 0%, 100% 0%, 100% 100%, 0% 100%);
    background: linear-gradient(150deg, var(--gold) 0%, var(--emerald) 100%);
    background-image:
        linear-gradient(150deg, var(--gold) 0%, var(--emerald) 100%),
        url("data:image/svg+xml,__PATTERN_LIGHT__");
    background-repeat: no-repeat, repeat;
    box-shadow: 0 20px 50px -12px var(--shadow);
    display: flex;
    align-items: center;
    justify-content: center;
}
.hero-visual-icon {
    font-size: clamp(3.5rem, 9vw, 5.5rem);
    filter: drop-shadow(0 6px 16px rgba(0,0,0,0.25));
}
.hero-chip {
    position: absolute;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.5rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 700;
    color: var(--text);
    backdrop-filter: blur(14px);
    box-shadow: 0 10px 26px -8px var(--shadow);
    z-index: 2;
    direction: rtl;
    white-space: nowrap;
}
.hero-chip--top { top: -10px; right: 8%; transform: rotate(-4deg); }
.hero-chip--bottom { bottom: -10px; left: 4%; transform: rotate(3deg); }
@media (max-width: 768px) {
    .hero-split { grid-template-columns: 1fr; }
    .hero-split-visual { min-height: 130px; order: -1; }
    .hero-split-text { text-align: center; }
    .hero-chip { display: none; }
}
.main-title {
    position: relative; z-index: 1;
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-size: clamp(1.9rem, 5.5vw, 3.1rem);
    font-weight: 900;
    background: linear-gradient(100deg, var(--gold) 0%, var(--emerald) 45%, var(--gold) 90%);
    background-size: 220% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: right;
    padding: 0 0 0.3rem 0;
    direction: rtl;
    letter-spacing: -0.01em;
    line-height: 1.15;
    animation: nsmTitleShimmer 6s ease-in-out infinite;
}
@keyframes nsmTitleShimmer {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@media (prefers-reduced-motion: reduce) {
    .main-title { animation: none !important; }
}

.subtitle {
    position: relative; z-index: 1;
    text-align: right;
    color: var(--text-muted);
    font-size: clamp(0.85rem, 3vw, 1.05rem);
    margin-bottom: 0.4rem;
    direction: rtl;
    font-weight: 600;
}

.welcome-line {
    position: relative; z-index: 1;
    text-align: right;
    color: var(--text-muted);
    font-size: 0.92rem;
    max-width: 560px;
    margin: 0 0 1rem 0;
    line-height: 1.9;
    direction: rtl;
}

.hero-badges {
    position: relative; z-index: 1;
    display: flex;
    justify-content: flex-end;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.9rem;
    direction: rtl;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.3rem 0.85rem;
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-muted);
    backdrop-filter: blur(12px);
}
.hero-badge .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 6px var(--emerald);
    flex-shrink: 0;
}


/* ── دخول متدرّج للبطاقات (يعكس فكرة "شبكة معرفية حيّة" تتيقّظ) ── */
@keyframes nsmRise {
    from { opacity: 0; transform: translateY(14px) scale(0.98); }
    to   { opacity: 1; transform: translateY(0) scale(1); }
}
@media (prefers-reduced-motion: reduce) {
    .metric-card, .feature-card { animation: none !important; opacity: 1 !important; }
}

/* ── مؤشر "مباشر" نابض بجانب عناوين الأقسام الحيّة ── */
.live-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--emerald);
    box-shadow: 0 0 0 0 var(--emerald-soft);
    animation: nsmPulseDot 2s ease-out infinite;
    margin-left: 6px;
    vertical-align: middle;
}
@keyframes nsmPulseDot {
    0%   { box-shadow: 0 0 0 0 var(--emerald-soft); }
    70%  { box-shadow: 0 0 0 8px rgba(0,0,0,0); }
    100% { box-shadow: 0 0 0 0 rgba(0,0,0,0); }
}

/* ── تبويب التدريب: شارة حالة (نشط/متوقف) ── */
.status-pill {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.3rem 0.8rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 700;
    vertical-align: middle;
    margin-right: 8px;
}
.status-pill--active {
    background: var(--emerald-soft);
    color: var(--emerald);
    border: 1px solid var(--emerald);
}
.status-pill--idle {
    background: var(--surface2);
    color: var(--text-dim, #94a3b8);
    border: 1px solid var(--border);
}
.status-pill .status-pill-dot {
    width: 7px; height: 7px; border-radius: 50%; background: currentColor;
}
.status-pill--active .status-pill-dot { animation: nsmPulseDot 2s ease-out infinite; }

/* ── شبكة رقاقات وحدات نقطة الحفظ ── */
.module-chip-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.6rem;
    margin: 0.4rem 0 0.2rem 0;
}
.module-chip {
    display: flex; align-items: center; gap: 0.5rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
.module-chip:hover { transform: translateY(-2px); border-color: var(--emerald); }
.module-chip-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--emerald); flex-shrink: 0;
    box-shadow: 0 0 0 3px var(--emerald-soft);
}

/* ── بطاقة بنية الشبكة العصبية ── */
.arch-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.2rem 0.2rem 0.9rem 0.2rem;
    margin-top: 0.3rem;
}

/* ── حالة فارغة أنيقة (لم يبدأ التدريب بعد) ── */
.training-empty {
    display: flex; align-items: center; gap: 1rem;
    background: var(--surface);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 1.3rem;
    margin: 0.4rem 0 0.6rem 0;
}
.training-empty-icon { font-size: 1.8rem; flex-shrink: 0; }
.training-empty-text { font-size: 0.9rem; line-height: 1.8; color: var(--text); }

/* ── بطاقات المقاييس ── */
.metric-card {
    position: relative;
    background: var(--surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.1rem 0.6rem;
    text-align: center;
    margin-bottom: 0.6rem;
    box-shadow: 0 4px 18px var(--shadow);
    transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
    min-height: 92px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    overflow: hidden;
    opacity: 0;
    animation: nsmRise 0.55s cubic-bezier(.22,.9,.35,1) forwards;
}
.metric-card:nth-of-type(1) { animation-delay: .02s; }
.metric-card:nth-of-type(2) { animation-delay: .08s; }
.metric-card:nth-of-type(3) { animation-delay: .14s; }
.metric-card:nth-of-type(4) { animation-delay: .20s; }
.metric-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: var(--accent-grad);
    opacity: 0; transition: opacity .18s ease;
}
.metric-card:hover {
    transform: translateY(-3px);
    border-color: transparent;
    box-shadow: 0 10px 28px var(--shadow);
}
.metric-card:hover::before { opacity: 1; }
.metric-value {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-size: clamp(0.92rem, 4.2vw, 1.9rem);
    font-weight: 800;
    color: var(--text);
    direction: ltr;
    line-height: 1.15;
    white-space: normal;
    overflow-wrap: break-word;
    word-break: keep-all;
}
.metric-value--wrap {
    font-size: clamp(0.72rem, 3.2vw, 1.05rem);
    white-space: normal;
    overflow-wrap: break-word;
    text-overflow: clip;
}
.metric-label {
    font-size: clamp(0.68rem, 2.6vw, 0.85rem);
    color: var(--text-muted);
    margin-top: 0.3rem;
    direction: rtl;
    line-height: 1.3;

}

/* ── بينتو-جريد للإحصاءات — بطاقة مميزة كبيرة + بقية البطاقات بأحجام متفاوتة ── */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-auto-flow: dense;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
}
.bento-grid .metric-card { margin-bottom: 0; }
.bento-featured {
    grid-column: span 2;
    grid-row: span 2;
    align-items: flex-end;
    text-align: right;
    direction: rtl;
    padding: 1.4rem 1.5rem;
    background: linear-gradient(135deg, var(--gold-soft), var(--emerald-soft)), var(--surface);
}
.bento-featured::before { opacity: 1; height: 3px; }
.bento-featured .metric-value {
    font-size: clamp(2rem, 7vw, 3.2rem);
    background: var(--accent-grad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.bento-featured .metric-label { font-size: clamp(0.8rem, 2.8vw, 0.95rem); font-weight: 600; }
@media (max-width: 768px) {
    .bento-grid { grid-template-columns: repeat(2, 1fr); }
    .bento-featured { grid-column: span 2; grid-row: span 1; align-items: center; text-align: center; }
}

/* ── بطاقة زجاجية عامة قابلة لإعادة الاستخدام بأي تبويب ── */
.glass-card {
    background: var(--surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.2rem 1.3rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 18px var(--shadow);
}

/* ── شريط التقدم — كان بلون Streamlit الافتراضي (أحمر/زهري)، غريب
   تماماً عن هوية التدرّج البنفسجي/الفيروزي ── */
[data-testid="stProgress"] > div > div > div {
    background: var(--accent-grad) !important;
}
[data-testid="stProgress"] > div > div {
    background: var(--surface2) !important;
}
.tab-intro {
    color: var(--text-muted);
    font-size: 0.9rem;
    line-height: 1.8;
    direction: rtl;
    margin-bottom: 1.1rem;
}

/* ── 🔔 Toast — إعادة تصميم st.toast الأصلي ليطابق هوية التدرّج ── */
[data-testid="stToast"] {
    background: var(--surface2) !important;
    backdrop-filter: blur(20px);
    border: 1px solid var(--border) !important;
    border-right: 3px solid var(--gold) !important;
    border-radius: 12px !important;
    box-shadow: 0 8px 28px var(--shadow) !important;
}
[data-testid="stToast"] * { color: var(--text) !important; }

/* ── 📋 زر نسخ — لمسة SaaS قياسية للنتائج والردود ── */
.nsm-copy-btn, .copy-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: var(--surface2);
    color: var(--text-muted);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.3rem 0.7rem;
    font-size: 0.78rem;
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-weight: 600;
    cursor: pointer;
    direction: rtl;
    transition: border-color .15s ease, color .15s ease, transform .1s ease;
}
.nsm-copy-btn:hover {
    border-color: var(--gold);
    color: var(--gold);
}
.nsm-copy-btn:active { transform: scale(0.96); }

/* ── 💀 Skeleton loading — حالة تحميل أنيقة بدل الدوارة العادية ── */
.skeleton-line, .skeleton-block {
    background: linear-gradient(
        90deg,
        var(--surface2) 25%,
        var(--border) 50%,
        var(--surface2) 75%
    );
    background-size: 200% 100%;
    animation: skeletonShimmer 1.4s ease-in-out infinite;
    border-radius: 8px;
}
.skeleton-line { height: 14px; margin-bottom: 0.55rem; }
.skeleton-line.w-70 { width: 70%; }
.skeleton-line.w-90 { width: 90%; }
.skeleton-line.w-50 { width: 50%; }
.skeleton-block { height: 90px; border-radius: var(--radius); margin-bottom: 0.6rem; }
@keyframes skeletonShimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}
@media (max-width: 480px) {
    .metric-card { padding: 0.8rem 0.4rem; min-height: 80px; }
}

/* ── تجاوب الجوال والأجهزة اللوحية (≤768px) ── */
@media (max-width: 768px) {
    /* أهداف لمس ≥44px لكل الأزرار والحقول وفق توصيات إمكانية الوصول */
    .stButton>button, .stDownloadButton>button {
        min-height: 44px;
        padding: 0.6rem 1rem;
        font-size: 0.95rem;
    }
    .stTextInput input, .stNumberInput input {
        min-height: 44px;
    }
    .stTabs [data-baseweb="tab"] {
        min-height: 44px;
        padding: 0.5rem 0.85rem;
        font-size: 0.85rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        overflow-x: auto;
        flex-wrap: nowrap;
    }
    /* صندوق المحادثة: ارتفاع أنسب على شاشات أقصر */
    .chat-box { height: 56vh; min-height: 320px; padding: 0.8rem; }
    .chat-nsm .bbl, .chat-user .bbl { max-width: 92%; font-size: 0.93rem; padding: 0.65rem 1rem; }
    /* الشريط الجانبي: حشوة أصغر عند فتحه كطبقة فوق المحتوى بالجوال */
    [data-testid="stSidebar"] { padding-top: 0.5rem; }

    /* أداء: تقليل ثقل الـblur على معالجات الجوال الأضعف */
    .metric-card, .feature-card, [data-testid="stSidebar"] { backdrop-filter: blur(8px); }

    /* استجابة لمسية فورية بدل تأخير hover — نفس سرعة النقر الفعلي */
    .metric-card, .feature-card, .quran-verse, .concept-card {
        -webkit-tap-highlight-color: transparent;
        transition-duration: .15s !important;
    }
    .feature-card:active { transform: scale(0.97); }
    .metric-card:active { transform: scale(0.97); }

    /* حركات دخول أسرع على الجوال — الانتظار الطويل يبدو تباطؤاً لا أناقة */
    .metric-card, .feature-card { animation-duration: .35s !important; }

    /* دعم شاشات النوتش (safe-area) لأسفل الصفحة */
    .stApp { padding-bottom: env(safe-area-inset-bottom, 0); }
}

/* ── دليل "كيف يعمل NSM؟" — خط أنابيب تفاعلي متحرك ── */
.nsm-pipeline-wrap {
    position: relative;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1.7rem 1.5rem 1.9rem;
    margin-bottom: 0.4rem;
    box-shadow: 0 4px 20px var(--shadow);
    overflow: hidden;
}
.nsm-pipeline-wrap::before {
    content: "";
    position: absolute; width: 260px; height: 260px; border-radius: 50%;
    background: var(--accent-grad); opacity: .09; filter: blur(46px);
    top: -110px; left: -70px; pointer-events: none;
}
.nsm-pipeline-wrap::after {
    content: "";
    position: absolute; width: 200px; height: 200px; border-radius: 50%;
    background: var(--rose); opacity: .06; filter: blur(46px);
    bottom: -90px; right: -50px; pointer-events: none;
}
.nsm-pipeline {
    position: relative;
    display: flex; align-items: flex-start; justify-content: space-between;
    gap: 0.4rem; margin-top: 0.5rem; direction: rtl; z-index: 1;
}
.nsm-pipeline-track {
    position: absolute; top: 25px; right: 9%; left: 9%; height: 3px;
    background: var(--border); border-radius: 3px; z-index: 0; overflow: hidden;
}
.nsm-pipeline-track-fill {
    position: absolute; top: 0; right: 0; height: 100%; width: 0%;
    background: var(--accent-grad);
    transition: width .5s cubic-bezier(.4,0,.2,1);
    border-radius: 3px;
}
.pipeline-node {
    position: relative; z-index: 1;
    display: flex; flex-direction: column; align-items: center;
    gap: 0.5rem; cursor: pointer; flex: 1; min-width: 0;
    -webkit-tap-highlight-color: transparent;
}
.pipeline-node-icon {
    width: 50px; height: 50px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem; background: var(--surface2); border: 2px solid var(--border);
    transition: all .35s cubic-bezier(.34,1.56,.64,1);
}
.pipeline-node:hover .pipeline-node-icon { border-color: var(--gold); }
.pipeline-node.active .pipeline-node-icon {
    background: var(--accent-grad); border-color: transparent;
    box-shadow: 0 0 0 6px var(--gold-soft), 0 6px 20px var(--gold-soft);
    transform: scale(1.14);
    animation: nsmNodePulse 2.2s ease-in-out infinite;
}
@keyframes nsmNodePulse {
    0%, 100% { box-shadow: 0 0 0 6px var(--gold-soft), 0 6px 20px var(--gold-soft); }
    50% { box-shadow: 0 0 0 10px rgba(0,0,0,0), 0 6px 20px var(--gold-soft); }
}
.pipeline-node-title {
    font-size: 0.76rem; font-weight: 700; color: var(--text-muted); text-align: center;
    line-height: 1.3; transition: color .3s ease;
}
.pipeline-node.active .pipeline-node-title { color: var(--text); }
.pipeline-detail {
    margin-top: 1.5rem; padding: 1.15rem 1.35rem;
    background: var(--surface2); border: 1px solid var(--border); border-radius: 14px;
    direction: rtl; min-height: 70px; position: relative; z-index: 1;
}
.pipeline-detail-title {
    font-weight: 700; color: var(--gold); margin-bottom: 0.35rem; font-size: 0.98rem;
    display: flex; align-items: center; gap: 0.4rem;
}
.pipeline-step-counter {
    margin-right: auto;
    font-size: 0.72rem; font-weight: 600; color: var(--text-muted);
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 20px; padding: 0.12rem 0.6rem; direction: ltr;
}
.nsm-pipeline:focus-visible {
    outline: 2px solid var(--gold); outline-offset: 6px; border-radius: 10px;
}
.pipeline-detail-text {
    font-size: 0.92rem; line-height: 1.85; color: var(--text);
}
.pipeline-detail-inner { transition: opacity .22s ease; }
.pipeline-progress-hint {
    margin-top: 0.7rem; display: flex; gap: 5px; justify-content: center;
}
.pipeline-progress-hint span {
    width: 6px; height: 6px; border-radius: 50%; background: var(--border);
    transition: background .3s ease, transform .3s ease;
}
.pipeline-progress-hint span.active { background: var(--gold); transform: scale(1.3); }
@media (max-width: 640px) {
    .nsm-pipeline { flex-wrap: wrap; row-gap: 1.3rem; }
    .nsm-pipeline-track { display: none; }
    .pipeline-node { flex: 1 1 30%; }
    .pipeline-node-title { font-size: 0.7rem; }
}
@media (prefers-reduced-motion: reduce) {
    .pipeline-node.active .pipeline-node-icon { animation: none; }
}

/* ── بطاقات الاستكشاف (الصفحة الرئيسية) ── */
.feature-card {
    position: relative;
    background: var(--surface);
    backdrop-filter: blur(16px);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    clip-path: polygon(0 16px, 16px 0, 100% 0, 100% 100%, 0 100%);
    padding: 1.3rem 1.2rem;
    text-align: right;
    direction: rtl;
    height: 100%;
    box-shadow: 0 4px 18px var(--shadow);
    transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    overflow: hidden;
    cursor: pointer;
    opacity: 0;
    animation: nsmRise 0.55s cubic-bezier(.22,.9,.35,1) forwards;
}
/* ── شريط أفقي قابل للتمرير لبطاقات "استكشف NSM" — بدل شبكة أعمدة
   Streamlit التي تتكدّس عمودياً بالجوال. عرض ثابت لكل بطاقة + تمرير
   أفقي سلس مع snap، وسهم اتجاه RTL طبيعي (يبدأ من اليمين). ────────── */
.feature-scroll {
    display: flex;
    flex-direction: row;
    gap: 1rem;
    overflow-x: auto;
    overflow-y: hidden;
    padding: 0.3rem 0.15rem 0.9rem;
    scroll-snap-type: x proximity;
    direction: rtl;
    -webkit-overflow-scrolling: touch;
}
.feature-scroll .feature-card {
    flex: 0 0 min(78vw, 250px);
    scroll-snap-align: start;
}
@media (min-width: 1100px) {
    .feature-scroll .feature-card { flex-basis: calc((100% - 4rem) / 5); }
}
.feature-scroll::-webkit-scrollbar { height: 6px; }
.feature-scroll::-webkit-scrollbar-thumb { background: var(--border); border-radius: 6px; }
.feature-scroll::-webkit-scrollbar-thumb:hover { background: var(--gold); }
.feature-card:nth-of-type(1) { animation-delay: .26s; }
.feature-card:nth-of-type(2) { animation-delay: .32s; }
.feature-card:nth-of-type(3) { animation-delay: .38s; }
.feature-card:nth-of-type(4) { animation-delay: .44s; }
.feature-card::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 3px;
    background: var(--card-accent, var(--accent-grad));
    opacity: 0; transition: opacity .2s ease;
}
.feature-card::after {
    content: "";
    position: absolute; top: 0; left: 0;
    width: 26px; height: 26px;
    background: var(--card-accent, var(--accent-grad));
    clip-path: polygon(0 0, 100% 0, 0 100%);
}
.feature-card:hover {
    transform: translateY(-6px);
    border-color: transparent;
    box-shadow: 0 14px 34px var(--shadow);
}
.feature-card:hover::before { opacity: 1; }
.feature-card:hover .feature-icon { transform: scale(1.1) rotate(-4deg); }
.feature-card:active { transform: translateY(-2px) scale(0.98); }
/* تنويع لوني دقيق لكل بطاقة — يكسر رتابة تكرار نفس التدرّج أربع مرات */
.feature-card:nth-of-type(4n+1) { --card-accent: linear-gradient(135deg, var(--gold), var(--emerald)); }
.feature-card:nth-of-type(4n+2) { --card-accent: linear-gradient(135deg, var(--emerald), var(--gold)); }
.feature-card:nth-of-type(4n+3) { --card-accent: linear-gradient(135deg, var(--rose), var(--gold)); }
.feature-card:nth-of-type(4n+4) { --card-accent: linear-gradient(135deg, var(--gold), var(--rose)); }
.feature-icon {
    width: 46px; height: 46px;
    margin: 0 0 0.7rem auto;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    background: var(--card-accent, var(--accent-grad));
    box-shadow: 0 4px 14px var(--gold-soft);
    transform: rotate(-6deg);
    transition: transform .25s cubic-bezier(.34,1.56,.64,1);
}
.feature-nav-hint {
    margin-top: 0.6rem;
    font-size: 0.72rem;
    color: var(--text-muted);
    opacity: 0;
    transform: translateY(3px);
    transition: opacity .2s ease, transform .2s ease;
}
.feature-card:hover .feature-nav-hint { opacity: 1; transform: translateY(0); }
.feature-title {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--text);
    margin-bottom: 0.35rem;
}
.feature-desc {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.6;
}

/* ── بطاقة المفهوم ── */
.concept-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 10px var(--shadow);
    direction: rtl;
}
.concept-name {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--gold);
    margin-bottom: 0.5rem;
}
.concept-stats {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px solid var(--border);
}
.concept-stat {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
    background: var(--surface2);
    border-radius: 8px;
    padding: 0.5rem 0.9rem;
    flex: 1;
    min-width: 110px;
}
.concept-stat-label {
    font-size: 0.72rem;
    color: var(--text-muted);
}
.concept-stat-value {
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
}
.related-tag {
    display: inline-block;
    background: var(--gold-soft);
    color: var(--gold);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.9rem;
    cursor: pointer;
}

/* ── آية قرآنية — طابع مصحفي: خط Amiri Quran + زخرفة قوسي الآية ﴾ ﴿ ── */
.quran-verse {
    position: relative;
    background: linear-gradient(135deg, var(--surface) 0%, var(--surface-2) 100%);
    border: 1px solid var(--border);
    border-right: 3px solid var(--gold);
    border-radius: 14px;
    padding: 1.3rem 1.7rem;
    margin: 0.7rem 0;
    font-family: 'Amiri Quran', 'Amiri', 'Traditional Arabic', serif;
    font-size: 1.3rem;
    line-height: 2.5;
    direction: rtl;
    color: var(--text);
    box-shadow: 0 3px 14px var(--shadow);
    transition: transform .2s ease, box-shadow .2s ease, border-color .2s ease;
    overflow: hidden;
}
.quran-verse::before {
    content: "﴾";
    position: absolute; top: -10px; right: 8px;
    font-size: 3.2rem; color: var(--gold); opacity: .14;
    font-family: 'Amiri Quran', 'Amiri', serif;
    pointer-events: none; line-height: 1;
}
.quran-verse::after {
    content: "﴿";
    position: absolute; bottom: -22px; left: 8px;
    font-size: 3.2rem; color: var(--emerald); opacity: .12;
    font-family: 'Amiri Quran', 'Amiri', serif;
    pointer-events: none; line-height: 1;
}
.quran-verse:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 26px var(--shadow);
    border-right-color: var(--emerald);
}
@media (max-width: 640px) {
    .quran-verse { font-size: 1.08rem; line-height: 2.15; padding: 1.05rem 1.2rem; }
    .quran-verse::before, .quran-verse::after { font-size: 2.3rem; }
}
.verse-ref {
    display: table;
    position: relative; z-index: 1;
    font-size: 0.76rem;
    color: var(--gold);
    font-weight: 700;
    margin-top: 0.65rem;
    padding: 0.22rem 0.75rem;
    background: var(--gold-soft);
    border-radius: 20px;
    direction: rtl;
    font-family: 'IBM Plex Sans Arabic', sans-serif;
}

.health-ok  { color: var(--emerald); font-weight: 600; }
.health-err { color: var(--rose);    font-weight: 600; }

/* ── عنوان قسم بتوقيع هندسي إسلامي بسيط بدل خط عادي ── */
.section-header {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    padding-bottom: 0.5rem;
    margin: 1rem 0 0.8rem 0;
    direction: rtl;
    text-align: right !important;
    border-bottom: 1px solid var(--border);
    position: relative;
}
.section-header::after {
    content: "";
    position: absolute;
    right: 0; bottom: -1px;
    width: 64px; height: 2px;
    background: var(--gold);
}

.tab-content { padding: 1rem 0; }

.search-box input {
    font-size: 1.2rem !important;
    direction: rtl !important;
    text-align: right !important;
}

.root-item {
    background: var(--emerald-soft);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    direction: rtl;
    color: var(--text);
}

.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
    border: 1px solid var(--border);
}
.badge-blue   { background: var(--gold-soft);    color: var(--gold); }
.badge-green  { background: var(--emerald-soft); color: var(--emerald); }
.badge-amber  { background: var(--gold-soft);    color: var(--gold); }
.badge-purple { background: var(--rose-soft);    color: var(--rose); }

/* ── مبدّل السمة ── */
.theme-toggle-caption {
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-size: 0.78rem;
    color: var(--text-muted);
    margin-bottom: 0.2rem;
}

/* ── إصلاح تراكب تلميح "Press Enter to apply" فوق النص العربي ──────────
   التلميح الداخلي لِـ Streamlit إنجليزي LTR دائماً؛ قاعدة RTL العامة
   أعلاه (على أي عنصر class يحوي "css") كانت تُطبَّق عليه أيضاً فتُغيّر
   موضعه المُطلق (position/inset) المبني أصلاً على افتراض LTR، فيرتطم
   بصرياً بالنص العربي المكتوب داخل الحقل. نُثبّت اتجاهه ونمنع تراكبه. */
div[data-testid="InputInstructions"] {
    direction: ltr !important;
    pointer-events: none;
}
div[data-testid="InputInstructions"] > span {
    direction: ltr !important;
    unicode-bidi: isolate;
}

/* ── سكرول بار عام لكامل الصفحة — يوحّد المظهر مع سكرول الشات ── */
html { scrollbar-width: thin; scrollbar-color: var(--border) transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover { background: var(--gold); }

/* ── تحديد النص — بلون الهوية بدل الأزرق الافتراضي ── */
::selection { background: var(--gold-soft); color: var(--text); }

/* ── حالة تركيز واضحة عبر لوحة المفاتيح (إمكانية وصول) دون كسر focus
   الافتراضي بالماوس/اللمس — نعتمد :focus-visible فقط ── */
a:focus-visible, button:focus-visible, [tabindex]:focus-visible,
.stButton>button:focus-visible, .stDownloadButton>button:focus-visible,
.feature-card:focus-visible {
    outline: 2px solid var(--gold) !important;
    outline-offset: 2px !important;
}

/* ── انتقال سلس عند تبديل السمة (داكن/فاتح) بدل القفزة المفاجئة ── */
.stApp, [data-testid="stSidebar"], .stButton>button, .stDownloadButton>button,
.stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div,
.metric-card, .feature-card, [data-testid="stExpander"] {
    transition: background-color .25s ease, border-color .25s ease, color .25s ease;
}
@media (prefers-reduced-motion: reduce) {
    .stApp, [data-testid="stSidebar"], .stButton>button, .stDownloadButton>button,
    .stTextInput input, .stTextArea textarea, [data-baseweb="select"] > div,
    .metric-card, .feature-card, [data-testid="stExpander"] { transition: none !important; }
}

/* ── طبقة تحسينات بصرية إضافية ──────────────────────────────────────── */

/* عرض القراءة على الشاشات العريضة جداً — يمنع تمدد المحتوى بلا حدود
   ويُبقي عرضاً مريحاً للعين بدل النص الممطوط من حافة لحافة. */
[data-testid="stMainBlockContainer"], .block-container {
    max-width: 1200px;
    margin: 0 auto;
}

/* ضغط خفيف عند الضغط على الأزرار — إحساس ملموس بالتفاعل */
.stButton>button:active, .stDownloadButton>button:active,
[data-testid="stFormSubmitButton"] button:active {
    transform: translateY(0) scale(0.97) !important;
    transition: transform 0.08s ease !important;
}

/* ── لون التمييز الموحّد لعناصر الإدخال الأصلية (راديو/تشيك/سلايدر) ──
   بدل الأزرق الافتراضي في المتصفح، تنسجم مع هوية المشروع. */
input[type="radio"], input[type="checkbox"] { accent-color: var(--gold); }
[data-testid="stSlider"] [role="slider"] {
    background-color: var(--gold) !important;
    border-color: var(--gold) !important;
}
[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
    background: var(--accent-grad) !important;
}

/* ── صناديق التنبيه (info/success/warning/error) — بحياد الثيم بدل
   الألوان الافتراضية الصارخة، مع شريط لوني جانبي هادئ ── */
[data-testid="stAlert"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(12px);
}
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentSuccess"],
[data-testid="stAlertContentWarning"], [data-testid="stAlertContentError"] {
    color: var(--text) !important;
}

/* ── تدرّج هرمي أوضح للعناوين ── */
h1 { font-size: clamp(1.5rem, 5vw, 2.1rem); font-weight: 800; }
h2 { font-size: clamp(1.25rem, 4vw, 1.6rem); font-weight: 700; }
h3 { font-size: clamp(1.05rem, 3.2vw, 1.3rem); font-weight: 700; }

/* ── الروابط النصية داخل المحتوى ── */
[data-testid="stMarkdownContainer"] a {
    color: var(--emerald);
    text-decoration: none;
    border-bottom: 1px solid var(--emerald-soft);
    transition: border-color .15s ease, opacity .15s ease;
}
[data-testid="stMarkdownContainer"] a:hover {
    border-color: var(--emerald);
    opacity: 0.85;
}

/* ── كشف تدريجي عند التمرير (scroll-reveal) ── */
.nsm-reveal {
    opacity: 0;
    transform: translateY(18px);
    transition: opacity .6s ease, transform .6s cubic-bezier(.22,.9,.35,1);
}
.nsm-reveal.is-visible { opacity: 1; transform: translateY(0); }
@media (prefers-reduced-motion: reduce) {
    .nsm-reveal { opacity: 1 !important; transform: none !important; transition: none !important; }
}

/* ── انتقال سلس عند تبديل الوضع الداكن/الفاتح (بدل التبديل الفجائي) ── */
.stApp, .metric-card, .feature-card, .glass-card, .concept-card,
.hero-badge, [data-testid="stSidebar"], .stTabs [data-baseweb="tab-list"],
.stTabs [data-baseweb="tab"] {
    transition: background-color .35s ease, border-color .35s ease,
                color .35s ease, box-shadow .35s ease;
}
@media (prefers-reduced-motion: reduce) {
    .stApp, .metric-card, .feature-card, .glass-card, .concept-card,
    .hero-badge, [data-testid="stSidebar"] { transition: none !important; }
}

/* ── لوحة أوامر سريعة (Ctrl+K) ── */
.nsm-cmdk-overlay {
    position: fixed; inset: 0;
    background: rgba(0,0,0,0.5);
    backdrop-filter: blur(3px);
    z-index: 99999;
    display: flex;
    align-items: flex-start;
    justify-content: center;
    padding-top: 12vh;
    opacity: 0;
    pointer-events: none;
    transition: opacity .15s ease;
}
.nsm-cmdk-overlay.is-open { opacity: 1; pointer-events: auto; }
.nsm-cmdk-box {
    width: min(520px, 90vw);
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 16px;
    box-shadow: 0 24px 60px var(--shadow);
    overflow: hidden;
    direction: rtl;
}
.nsm-cmdk-input {
    width: 100%; box-sizing: border-box;
    padding: 1rem 1.2rem;
    font-size: 1rem;
    border: none; outline: none;
    background: transparent;
    color: var(--text);
    border-bottom: 1px solid var(--border);
    font-family: 'Tajawal', sans-serif;
}
.nsm-cmdk-list { max-height: 50vh; overflow-y: auto; }
.nsm-cmdk-item {
    padding: 0.75rem 1.2rem;
    cursor: pointer;
    color: var(--text);
    font-size: 0.92rem;
    font-family: 'Tajawal', sans-serif;
}
.nsm-cmdk-item:hover, .nsm-cmdk-item.active { background: var(--gold-soft); }
.nsm-cmdk-fab {
    position: fixed; bottom: 20px; left: 20px;
    z-index: 9998;
    width: 48px; height: 48px;
    border-radius: 50%;
    background: var(--accent-grad);
    color: #fff;
    border: none;
    font-weight: 800;
    font-size: 0.82rem;
    box-shadow: 0 8px 24px var(--shadow);
    cursor: pointer;
}
.nsm-cmdk-fab:hover { filter: brightness(1.08); }
@media (max-width: 768px) {
    .nsm-cmdk-fab { bottom: 16px; left: 16px; width: 44px; height: 44px; }
}
</style>
"""


@st.cache_data(show_spinner=False)
def render_css(theme_key: str) -> str:
    # أداء: هذه دالة pure (نفس theme_key ⇒ نفس CSS دائماً)، لكنها كانت
    # تُعاد معالجتها بالكامل (18 عملية .replace على قالب >1250 سطر) في
    # كل rerun من التطبيق — أي عند كل ضغطة زر/تفاعل في كامل الواجهة،
    # بلا أي استفادة لأن النتيجة نفسها دائماً لنفس الثيم. الكاش يقلّص
    # هذا لمرة واحدة فعلياً لكل قيمة ثيم (قيمتان فقط: dark/light).
    t = THEMES.get(theme_key, THEMES["dark"])
    gold_alt = "#E4C87A" if theme_key == "dark" else "#7A5E20"
    pattern = _pattern_svg(t["pattern_stroke"], t["pattern_opacity"])
    pattern_light = _pattern_svg("#FFFFFF", "0.22")
    css = CSS_TEMPLATE
    replacements = {
        "__BG__": t["bg"],
        "__BG_GRAD__": t["bg_grad"],
        "__SURFACE__": t["surface"],
        "__SURFACE2__": t["surface2"],
        "__BORDER__": t["border"],
        "__TEXT__": t["text"],
        "__TEXT_MUTED__": t["text_muted"],
        "__GOLD__": t["gold"],
        "__GOLD_SOFT__": t["gold_soft"],
        "__EMERALD__": t["emerald"],
        "__EMERALD_SOFT__": t["emerald_soft"],
        "__ROSE__": t["rose"],
        "__ROSE_SOFT__": t["rose_soft"],
        "__SHADOW__": t["shadow"],
        "__PATTERN__": pattern,
        "__PATTERN_LIGHT__": pattern_light,
        "__GOLD_DARK_OR_LIGHT__": gold_alt,
    }
    for k, v in replacements.items():
        css = css.replace(k, v)
    return css

# ── حقن CSS السمة الحالية (مع تخزين دائم للتفضيل عبر core.artifacts_store) ──
if "ui_theme" not in st.session_state:
    try:
        from core.artifacts_store import get_setting as _get_persisted_setting
        st.session_state.ui_theme = _get_persisted_setting("ui_theme", "dark")
    except Exception:
        st.session_state.ui_theme = "dark"
st.markdown(render_css(st.session_state.ui_theme), unsafe_allow_html=True)

# ── فرض قسري للون نص التبويبات عبر JS (طبقة حماية أخيرة) ──────────────────
# السبب: .streamlit/config.toml يثبّت ثيم Streamlit الأصلي على "dark"
# دائماً (لأسباب أخرى غير متعلقة بتبديلنا الداخلي للثيم)، وبعض مكوّنات
# BaseWeb الداخلية (مثل st.tabs) قد تُطبّق ألوان مشتقة من ذلك الثيم
# الأصلي بطريقة تُفلت أحياناً من تجاوزات CSS العادية.
#
# ملاحظة مهمة: حقن <script> عبر st.markdown (كما كان سابقاً) لا يُنفَّذه
# المتصفح — عنصر <script> المُدرَج عبر innerHTML/markdown لا يعمل أبداً،
# هذا سلوك موثّق بمتصفحات الويب وليس مجرد "أحياناً". الحل الصحيح
# المضمون هو st.components.v1.html الذي يُنشئ iframe حقيقياً يُنفَّذ فيه
# JS فعلياً، ومن داخله نصل لمستند الصفحة الأم عبر window.parent.document.
_tab_text_color = THEMES.get(st.session_state.ui_theme, THEMES["dark"])["gold"]
_tab_selected_bg_color = THEMES.get(st.session_state.ui_theme, THEMES["dark"])["bg"]

# أداء: هذا الحقن (iframe جديد + ~165 سطر JS + مسح فوري لكامل DOM 5 مرات
# عبر setTimeout) كان يُنفَّذ بالكامل من جديد عند *كل* rerun — أي عند كل
# نقرة/تفاعل في أي مكان بالتطبيق (9700+ سطر، عشرات التبويبات)، رغم أن
# MutationObserver المُنشأ بداخله (مضمون doc.__nsmTabObserver) يبقى حياً
# ويعيد تطبيق لون التبويبات وscroll-reveal تفاعلياً بنفسه عند أي تغيّر
# DOM لاحق (بما فيها الضغط على تبويب آخر). لذلك: نعيد الحقن الكامل فقط
# عند أول تحميل للجلسة، أو عند تغيّر الثيم فعلياً (لأن الألوان مُضمَّنة
# كقيم ثابتة داخل الـJS وقت الحقن، لازم تتحدّث لو الثيم تغيّر). أي rerun
# عادي (تنقّل بين التبويبات، إرسال رسالة دردشة، إلخ) يتخطّى هذا الحقن
# بالكامل الآن ويعتمد على الـMutationObserver الحي أصلاً.
_nsm_chrome_key = f"_nsm_chrome_theme::{st.session_state.ui_theme}"
if not st.session_state.get(_nsm_chrome_key):
    st.session_state[_nsm_chrome_key] = True
    st.components.v1.html(f"""
<script>
(function() {{
    function nsmForceTabColor() {{
        try {{
            const doc = window.parent.document;
            const tabs = doc.querySelectorAll('.stTabs [data-baseweb="tab"]');
            tabs.forEach(function(tab) {{
                const selected = tab.getAttribute('aria-selected') === 'true';
                const color = selected ? '{_tab_selected_bg_color}' : '{_tab_text_color}';
                tab.style.setProperty('color', color, 'important');
                tab.querySelectorAll('*').forEach(function(child) {{
                    child.style.setProperty('color', color, 'important');
                }});
            }});
        }} catch (e) {{ /* تجاهل صامت — بيئة قد لا تسمح بالوصول للمستند الأب */ }}
    }}

    // ── كشف تدريجي عند التمرير (scroll-reveal) ─────────────────────────
    function nsmInitScrollReveal() {{
        try {{
            const doc = window.parent.document;
            const targets = doc.querySelectorAll(
                '.section-header, .glass-card, .concept-card, [data-testid="stExpander"]'
            );
            targets.forEach(function(el) {{
                if (!el.classList.contains('nsm-reveal')) el.classList.add('nsm-reveal');
            }});
            if (!doc.__nsmRevealIO) {{
                doc.__nsmRevealIO = new IntersectionObserver(function(entries) {{
                    entries.forEach(function(e) {{
                        if (e.isIntersecting) {{
                            e.target.classList.add('is-visible');
                            doc.__nsmRevealIO.unobserve(e.target);
                        }}
                    }});
                }}, {{ threshold: 0.12 }});
            }}
            doc.querySelectorAll('.nsm-reveal:not(.is-visible)').forEach(function(el) {{
                doc.__nsmRevealIO.observe(el);
            }});
        }} catch (e) {{ /* تجاهل صامت */ }}
    }}

    // ── لوحة أوامر سريعة (Ctrl+K / ⌘K) للتنقّل الفوري بين الأقسام ──────
    function nsmBuildPalette() {{
        try {{
            const doc = window.parent.document;
            if (doc.__nsmPaletteBuilt) return;
            doc.__nsmPaletteBuilt = true;

            const overlay = doc.createElement('div');
            overlay.id = 'nsm-cmdk-overlay';
            overlay.className = 'nsm-cmdk-overlay';
            overlay.innerHTML =
                '<div class="nsm-cmdk-box">' +
                    '<input id="nsm-cmdk-input" class="nsm-cmdk-input" placeholder="ابحث عن قسم... (Esc للإغلاق)" />' +
                    '<div id="nsm-cmdk-list" class="nsm-cmdk-list"></div>' +
                '</div>';
            doc.body.appendChild(overlay);

            const fab = doc.createElement('button');
            fab.id = 'nsm-cmdk-fab';
            fab.className = 'nsm-cmdk-fab';
            fab.type = 'button';
            fab.title = 'بحث سريع (Ctrl+K)';
            fab.textContent = '⌘K';
            doc.body.appendChild(fab);

            function getTabs() {{
                return Array.from(doc.querySelectorAll('[data-baseweb="tab-list"] [data-baseweb="tab"]'));
            }}
            function renderList(filterText) {{
                const list = doc.getElementById('nsm-cmdk-list');
                list.innerHTML = '';
                const f = (filterText || '').trim();
                let first = true;
                getTabs().forEach(function(t) {{
                    const label = (t.textContent || '').trim();
                    if (f && label.indexOf(f) === -1) return;
                    const item = doc.createElement('div');
                    item.className = 'nsm-cmdk-item' + (first ? ' active' : '');
                    first = false;
                    item.textContent = label;
                    item.addEventListener('click', function() {{
                        t.click();
                        closePalette();
                    }});
                    list.appendChild(item);
                }});
            }}
            function openPalette() {{
                overlay.classList.add('is-open');
                const inp = doc.getElementById('nsm-cmdk-input');
                inp.value = '';
                renderList('');
                setTimeout(function() {{ inp.focus(); }}, 30);
            }}
            function closePalette() {{
                overlay.classList.remove('is-open');
            }}
            overlay.addEventListener('click', function(e) {{
                if (e.target === overlay) closePalette();
            }});
            fab.addEventListener('click', function() {{
                if (overlay.classList.contains('is-open')) closePalette(); else openPalette();
            }});
            doc.getElementById('nsm-cmdk-input').addEventListener('input', function(e) {{
                renderList(e.target.value);
            }});
            doc.addEventListener('keydown', function(e) {{
                const isOpen = overlay.classList.contains('is-open');
                if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'k') {{
                    e.preventDefault();
                    if (isOpen) closePalette(); else openPalette();
                    return;
                }}
                if (!isOpen) return;
                if (e.key === 'Escape') {{ closePalette(); return; }}
                const items = Array.from(doc.querySelectorAll('.nsm-cmdk-item'));
                if (!items.length) return;
                let idx = items.findIndex(function(i) {{ return i.classList.contains('active'); }});
                if (e.key === 'ArrowDown') {{
                    e.preventDefault();
                    if (idx < items.length - 1) {{
                        items[idx].classList.remove('active');
                        items[idx + 1].classList.add('active');
                        items[idx + 1].scrollIntoView({{ block: 'nearest' }});
                    }}
                }} else if (e.key === 'ArrowUp') {{
                    e.preventDefault();
                    if (idx > 0) {{
                        items[idx].classList.remove('active');
                        items[idx - 1].classList.add('active');
                        items[idx - 1].scrollIntoView({{ block: 'nearest' }});
                    }}
                }} else if (e.key === 'Enter') {{
                    e.preventDefault();
                    if (items[idx]) items[idx].click();
                }}
            }});
        }} catch (e) {{ /* تجاهل صامت */ }}
    }}

    nsmForceTabColor();
    nsmInitScrollReveal();
    nsmBuildPalette();
    setTimeout(nsmForceTabColor, 150);
    setTimeout(nsmForceTabColor, 500);
    setTimeout(nsmForceTabColor, 1200);
    setTimeout(nsmInitScrollReveal, 200);
    setTimeout(nsmInitScrollReveal, 700);
    try {{
        const doc2 = window.parent.document;
        if (!doc2.__nsmTabObserver) {{
            const obs = new MutationObserver(function() {{
                nsmForceTabColor();
                nsmInitScrollReveal();
            }});
            obs.observe(doc2.body, {{ childList: true, subtree: true, attributes: true, attributeFilter: ['aria-selected', 'class'] }});
            doc2.__nsmTabObserver = obs;
        }}
    }} catch (e) {{}}
}})();
</script>
""", height=0, width=0)



# ═══════════════════════════════════════════════════════════════════════════
# دوال تحميل البيانات
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_arabic_roots() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "arabic_roots_index.json")
    return data or {}


@st.cache_data(ttl=60)
def load_graph_metrics() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "graph_metrics.json")
    return data or {}


@st.cache_data(ttl=60)
def load_quran_index() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "quran_index.json")
    return data or {}


@st.cache_data(ttl=300)
def load_all_quran_ayat() -> List[Dict]:
    """تحميل كل آيات القرآن من الـ chunks."""
    ayat: List[Dict] = []
    chunk_files = sorted(KNOWLEDGE_DIR.glob("quran_chunk_*.json"))
    for cf in chunk_files:
        try:
            with open(cf, encoding="utf-8") as f:
                chunk = json.load(f)
            if isinstance(chunk, list):
                ayat.extend(chunk)
        except Exception:
            continue
    return ayat


@st.cache_data(ttl=60)
def load_latest_checkpoint() -> Dict:
    """تحميل أحدث brain_checkpoint."""
    checkpoints = sorted(CHECKPOINTS_DIR.glob("brain_checkpoint_*.json"), reverse=True)
    if checkpoints:
        data = load_json(checkpoints[0])
        return data or {}
    return {}


@st.cache_data(ttl=60)
def load_training_summary() -> Dict:
    path = CHECKPOINTS_DIR / "deep_network_training_summary.json"
    data = load_json(path)
    return data or {}


@st.cache_data(ttl=60)
def load_ckg() -> Dict:
    """تحميل الـ CKG — يعود بـ {} إذا كان الملف فارغاً أو Git LFS pointer."""
    _empty = {"concepts": {}, "relations": {}}
    path = KNOWLEDGE_DIR / "cognitive_graph.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        # Git LFS pointer — الملف لم يُنزَّل
        if not content or content.startswith("version https://git-lfs"):
            return _empty
        data = json.loads(content)
        # تأكد من وجود المفاتيح الأساسية
        if not isinstance(data, dict):
            return _empty
        if "concepts" not in data:
            data["concepts"] = {}
        if "relations" not in data:
            data["relations"] = {}
        return data
    except Exception:
        return _empty


@st.cache_data(ttl=60)
def load_entities() -> Dict:
    """تحميل طبقة الكيانات المعرفية (entities.json) — يعود بـ {} إن لم تكن موجودة."""
    path = KNOWLEDGE_DIR / "entities.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        return data.get("entities", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


@st.cache_data(ttl=30, show_spinner=False)
def _load_wan_free_space_status() -> list:
    """يفحص حالة مساحات Wan المجانية (Running on Zero) عبر
    ai.video_engine.check_wan_free_space_status — مُخزَّن مؤقتاً 30
    ثانية فقط (بعكس بقية دوال load_* هنا بـttl=60/300) حتى لا يعرض
    زر «تحقّق من التوفّر» حالة قديمة لدقائق طويلة، مع تفادي إغراق REST
    API الخاص بـHugging Face بطلب جديد كل ضغطة زر. يُرجِع [] بصمت إن
    تعذّر استيراد ai.video_engine (مثلاً moviepy غير مثبَّتة)."""
    try:
        from ai.video_engine import check_wan_free_space_status
    except Exception:
        return []
    return check_wan_free_space_status()


def _render_wan_free_status_widget(key_prefix: str) -> None:
    """زر + عرض حالة حيّة لمساحات Wan2.1 المجانية (Running on Zero)،
    مشترك بين تبويبي 🎬 Explainer و⚡ Shorts لتفادي ازدواج الكود.
    key_prefix يمنع تصادم مفاتيح session_state بين التبويبين.

    يُخزّن أيضاً مجموعة أسماء المساحات المعطوبة (ok=False) بمفتاح
    session_state['{key_prefix}_wan_dead_spaces'] — يقرأها استدعاء
    engine.render_video(wan_skip_spaces=...) لاحقاً ليتجاوزها فوراً
    بدل انتظار فشلها الفعلي أثناء الرندر."""
    if st.button(
        "🔍 تحقّق من توفّر GPU المجاني الآن",
        key=f"{key_prefix}_wan_status_btn",
        help="فحص خفيف وفوري لحالة مساحات Hugging Face (Running on Zero) قبل بدء الرندر — لا يستهلك أي رصيد.",
    ):
        _load_wan_free_space_status.clear()  # مسح ذاكرة هذه الدالة فقط (لا كل cache_data بالتطبيق)
        st.session_state[f"{key_prefix}_wan_status_checked"] = True

    if st.session_state.get(f"{key_prefix}_wan_status_checked"):
        _statuses = _load_wan_free_space_status()
        if not _statuses:
            st.caption("❔ تعذّر تشغيل الفحص بهذه البيئة حالياً — سيُحاول الرندر مباشرة على أي حال.")
            st.session_state[f"{key_prefix}_wan_dead_spaces"] = set()
        else:
            for _s in _statuses:
                st.caption(f"**{_s['space']}** — {_s['label']}")
            _dead = {_s["space"] for _s in _statuses if not _s["ok"]}
            st.session_state[f"{key_prefix}_wan_dead_spaces"] = _dead
            if _dead and len(_dead) < len(_statuses):
                st.caption(
                    f"↪️ سيتم تجاوز {len(_dead)} مساحة غير متاحة تلقائياً عند الرندر، "
                    "والمحاولة مباشرة مع المساحات السليمة."
                )
            elif not any(_s["ok"] for _s in _statuses):
                st.caption(
                    "⚠️ كل المساحات المجانية تبدو غير متاحة الآن — الرندر سيتراجع "
                    "تلقائياً للخلفية المتدرّجة الافتراضية إن فشلت جميعها."
                )


@st.cache_resource(show_spinner=False)
def _get_episodic_engine():
    """singleton واحد لعملية Streamlit كاملة (وليس لكل جلسة) — يبدأ خيط
    التوحيد (consolidation) بالخلفية مرة واحدة فقط. يعيد None إن تعذّر
    الاستيراد بدل رفع استثناء يكسر الواجهة."""
    if not _EPISODIC_OK:
        return None
    try:
        engine = EpisodicMemoryEngine(db_path=str(MEMORY_DIR / "episodic.db"))
        engine.start()  # يوحّد working_memory → قاعدة البيانات كل 60 ثانية تلقائياً
        return engine
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def _get_memory_consolidator():
    """singleton لعملية Streamlit كاملة. MemoryConsolidator (ai/memory_consolidator.py)
    كانت مستوردة بلا استخدام إطلاقاً — تحوّل أنماطاً متكررة في الذاكرة
    الإيبيسودية الحقيقية (get_strongest_memories) إلى "قوانين مكتسبة"
    عبر دورات دمج دورية بالخلفية. طبقة مكمّلة لتوحيد episodic_memory
    الداخلي (تجميع حسب المصدر/الهدف الرقمي)، لا بديل له."""
    if not _CONSOLIDATOR_OK:
        return None
    engine = _get_episodic_engine()
    if engine is None:
        return None
    try:
        mc = MemoryConsolidator(episodic_memory=engine, pattern_threshold=5)
        mc.start(interval_minutes=15)
        return mc
    except Exception:
        return None


def _record_chat_episode(query: str, response: str, source: str = "chat") -> None:
    """يسجّل تبادل محادثة واحد كذاكرة إيبيسودية حقيقية — لا يرفع استثناءً أبداً
    (فشل التسجيل لا يجوز أن يكسر تجربة المحادثة).

    القيم المستخدمة مشتقة من إشارات حقيقية للتبادل نفسه (لا بيانات وهمية):
      - feature_vec: أبعاد بسيطة عن طول السؤال/الرد ونسبتهما
      - target/outcome = 1.0 إن كان هناك رد فعلي غير فارغ، وإلا 0.0 (فشل حقيقي)
      - reward محايد (0.0) لعدم وجود تقييم مستخدم فعلي بعد (لا نختلق رضا وهمياً)
    """
    engine = _get_episodic_engine()
    if engine is None:
        return
    try:
        q_len = min(len(query or ""), 2000)
        r_len = min(len(response or ""), 4000)
        ok    = 1.0 if (response or "").strip() else 0.0
        feature_vec = [
            q_len / 2000.0,
            r_len / 4000.0,
            (r_len / q_len) if q_len else 0.0,
            ok,
            1.0 if "؟" in (query or "") else 0.0,
            float(len((query or "").split())) / 100.0,
            float(len((response or "").split())) / 200.0,
        ]
        engine.record(
            feature_vec=feature_vec,
            target=ok,
            outcome=ok,
            source=source,
            reward=0.0,
            context={
                "query":    (query or "")[:500],
                "response": (response or "")[:1000],
            },
        )
    except Exception:
        pass

    # ── توحيد الاستدعاء: يصل ConversationLearner (كان يتيماً بالكامل) ──
    # بنفس نقطة تسجيل الحلقة الحقيقية أعلاه. best-effort — لا يكسر الرد
    # عند الفشل (انظر ai/learning_orchestrator.py لتفاصيل التوحيد).
    try:
        from ai.learning_orchestrator import get_orchestrator
        get_orchestrator().record_turn(query, response, source=source)
    except Exception:
        pass


def get_episodic_stats() -> Dict:
    db_path = MEMORY_DIR / "episodic.db"
    stats = {"working": 0, "semantic": 0, "episodic": 0, "rules": 0}
    if not db_path.exists():
        return stats
    try:
        conn = sqlite3.connect(str(db_path))
        episodes_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        stats["episodic"] = episodes_count
        conn.close()
    except Exception:
        pass
    # working memory / semantic rules — تُقرأ من المحرك الحي إن كان مهيّأً
    # بهذه العملية (best-effort، لا تُفشل الدالة إن لم يكن متاحاً بعد)
    try:
        engine = _get_episodic_engine()
        if engine is not None:
            stats["working"]  = len(engine.working_memory)
            stats["semantic"] = len(engine.semantic_rules)
            stats["rules"]    = len(engine.semantic_rules)
    except Exception:
        pass
    return stats


class _RealMeshSnapshot:
    """كائن خفيف يعرض فقط ما يفهمه BrainCheckpoint._extract_state من سمات
    (episodic_memory بدالة summary()، knowledge بدالة keys()) — يُغذَّى ببيانات
    حقيقية من النظام الحي فقط. أي وحدة غير موجودة فعلياً (neural_weights،
    deep_network...) تُترك غائبة عمداً بدل اختلاق أرقام؛ BrainCheckpoint
    يتجاوزها تلقائياً عبر getattr(..., None) في كل قسم."""

    VERSION = "NSM-Dashboard-1.0"

    def __init__(self, episodic_engine, ckg: Dict):
        if episodic_engine is not None:
            self.episodic_memory = episodic_engine
        self.knowledge = ckg.get("concepts", {})


def save_real_checkpoint() -> Optional[str]:
    """يحفظ Checkpoint حقيقياً يعكس حالة النظام الفعلية الآن (لا بيانات وهمية):
    عدد مفاهيم/علاقات CKG الحقيقي + ملخص الذاكرة الإيبيسودية الحي (working
    memory، semantic rules، أقوى الذكريات). يحاول أيضاً الدفع لـ GitHub
    تلقائياً بالخلفية إن كانت متغيرات GITHUB_TOKEN/GITHUB_REMOTE معرَّفة.

    يعيد مسار الملف المحفوظ، أو None إن تعذّر (بدون رفع استثناء)."""
    if not _CHECKPOINT_OK:
        return None
    try:
        ckg    = load_ckg()
        engine = _get_episodic_engine() if _EPISODIC_OK else None
        mesh   = _RealMeshSnapshot(episodic_engine=engine, ckg=ckg)
        bc     = BrainCheckpoint(checkpoint_dir=str(CHECKPOINTS_DIR))
        path   = bc.save(mesh)
        # دفع تلقائي بالخلفية لـ GitHub — لا يرفع استثناء ولا يوقف الحفظ
        # المحلي إن كانت متغيرات GITHUB_TOKEN/GITHUB_REMOTE غير معرَّفة
        # (github_sync.push_now يتحقق من ذلك داخلياً ويتجاوز بهدوء).
        if path and _GITHUB_SYNC_OK:
            try:
                _github_sync.push_background(tag=Path(path).stem)
            except Exception:
                pass
        return path
    except Exception:
        return None


# ── MetaReasoner: adapters حقيقية فوق سجل التوجيه الفعلي ───────────────────
# MetaReasoner (ai/meta_reasoner.py) كانت مستوردة فقط بلا أي استخدام: تتوقع
# memory_engine بدالة all_routes() و scoring_engine بدالة list_scores()،
# بينما البيانات الفعلية الوحيدة المتاحة هي صفوف مسطّحة من route_log_store
# (كل صف = طلب واحد: category, node, latency_ms, success...). الـ adapters
# التالية تجمّع هذه الصفوف الحقيقية إلى الشكل الذي تتوقعه MetaReasoner —
# بلا أي بيانات مُختلَقة؛ مسار بلا تشغيلات فعلية ببساطة لا يظهر.

class _RouteLogMemoryAdapter:
    """يحاكي واجهة memory_engine.all_routes() المتوقَّعة من MetaReasoner،
    مبنية من تجميع حقيقي لسجل التوجيه حسب (الفئة الدلالية → العقدة)."""

    def all_routes(self) -> List[Dict[str, Any]]:
        if not _ROUTE_LOG_DB_OK:
            return []
        rows = _rlog_get_recent(limit=5000)
        agg: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = f"{r.get('category', '?')}→{r.get('node', '?')}"
            a = agg.setdefault(key, {"path_key": key, "runs": 0, "successes": 0})
            a["runs"] += 1
            if r.get("success"):
                a["successes"] += 1
        result = []
        for a in agg.values():
            sr = a["successes"] / a["runs"] if a["runs"] else 0.0
            health = "healthy" if sr >= 0.8 else ("degraded" if sr >= 0.5 else "failing")
            result.append({
                "path_key": a["path_key"], "runs": a["runs"],
                "success_rate": sr, "health": health,
            })
        return result


class _RouteLogScoringAdapter:
    """يحاكي واجهة scoring_engine.list_scores() المتوقَّعة من MetaReasoner،
    مبنية من نفس سجل التوجيه الحقيقي (الفئة كـ source_id، العقدة كـ
    target_id) بدل بيانات مُختلَقة."""

    def list_scores(self) -> List[Dict[str, Any]]:
        if not _ROUTE_LOG_DB_OK:
            return []
        rows = _rlog_get_recent(limit=5000)
        agg: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rows:
            key = (r.get("category", "?"), r.get("node", "?"))
            a = agg.setdefault(key, {
                "source_id": key[0], "target_id": key[1],
                "runs": 0, "successes": 0, "latency_sum": 0.0,
            })
            a["runs"] += 1
            a["latency_sum"] += r.get("latency_ms", 0) or 0
            if r.get("success"):
                a["successes"] += 1
        result = []
        for a in agg.values():
            runs = a["runs"]
            result.append({
                "source_id": a["source_id"], "target_id": a["target_id"],
                "success_rate": (a["successes"] / runs) if runs else 0.0,
                "avg_latency_ms": (a["latency_sum"] / runs) if runs else 0.0,
                "total_runs": runs,
            })
        return result


@st.cache_resource(show_spinner=False)
def _get_meta_reasoner():
    """singleton واحد لعملية Streamlit كاملة. يعيد None إن كانت MetaReasoner
    غير قابلة للاستيراد أصلاً، بدل رفع استثناء يكسر الواجهة."""
    if not _META_REASONER_OK:
        return None
    try:
        return MetaReasoner(
            memory_engine=_RouteLogMemoryAdapter(),
            scoring_engine=_RouteLogScoringAdapter(),
        )
    except Exception:
        return None


# ── تطبيع النص العربي ────────────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'\ufeff', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# منطق البحث المعرفي
# ═══════════════════════════════════════════════════════════════════════════

def search_quran_for_concept(query: str, ayat: List[Dict], max_results: int = 8) -> List[Dict]:
    """البحث في القرآن عن الآيات التي تحتوي على المفهوم."""
    q_norm = normalize_arabic(query)
    results = []
    for ayah in ayat:
        text_norm = normalize_arabic(ayah.get("text_norm", "") or ayah.get("text", ""))
        if q_norm in text_norm:
            results.append(ayah)
            if len(results) >= max_results:
                break
    return results


def find_related_concepts_from_roots(query: str, roots: Dict, top_k: int = 8) -> List[Tuple[str, int]]:
    """إيجاد المفاهيم المرتبطة بناءً على الجذور العربية."""
    q_norm = normalize_arabic(query)
    matches = []
    for root, info in roots.items():
        root_norm = normalize_arabic(root)
        tokens = [normalize_arabic(t) for t in info.get("tokens", [])]
        top_token = normalize_arabic(info.get("top_token", ""))

        score = 0
        if q_norm == root_norm:
            score = 1000
        elif q_norm in top_token or top_token in q_norm:
            score = 800
        elif any(q_norm in t or t in q_norm for t in tokens):
            score = 500
        elif q_norm[:3] == root_norm[:3] and len(q_norm) >= 3:
            score = 300

        if score > 0:
            matches.append((info.get("top_token", root), info.get("frequency", 0), score))

    matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(m[0], m[1]) for m in matches[:top_k]]


def search_knowledge(query: str) -> Dict:
    """البحث الشامل في قاعدة المعرفة."""
    roots   = load_arabic_roots()
    ayat    = load_all_quran_ayat()
    ckg     = load_ckg()
    concepts_db = ckg.get("concepts", {})
    relations_db = ckg.get("relations", {})

    q_norm = normalize_arabic(query)

    # ── 1. البحث في CKG ──────────────────────────────────────────────────
    concept_data = None
    ckg_related  = []
    ckg_relations = []

    # بحث مباشر
    for cname, cdata in concepts_db.items():
        if normalize_arabic(cname) == q_norm or q_norm in normalize_arabic(cname):
            concept_data = {"name": cname, **cdata}
            break

    if concept_data:
        cname = concept_data["name"]
        for rel_key, rel_data in relations_db.items():
            src = rel_data.get("source", "")
            tgt = rel_data.get("target", "")
            if normalize_arabic(src) == q_norm:
                ckg_related.append(tgt)
                ckg_relations.append({"target": tgt, "type": rel_data.get("relation_type", ""), "weight": rel_data.get("weight", 0)})
            elif normalize_arabic(tgt) == q_norm:
                ckg_related.append(src)
                ckg_relations.append({"target": src, "type": rel_data.get("relation_type", ""), "weight": rel_data.get("weight", 0)})

    # ── 2. البحث في الجذور العربية ───────────────────────────────────────
    root_matches = find_related_concepts_from_roots(query, roots, top_k=8)

    # ── 3. البحث في القرآن ───────────────────────────────────────────────
    quran_matches = search_quran_for_concept(query, ayat, max_results=10)

    # ── 4. درجة الثقة ────────────────────────────────────────────────────
    confidence = 0.0
    if concept_data:
        confidence += 0.4
        freq = concept_data.get("frequency", 0)
        confidence += min(freq / 100, 0.3)
    if quran_matches:
        confidence += min(len(quran_matches) / 10, 0.2)
    if root_matches:
        confidence += 0.1
    confidence = min(confidence, 1.0)

    # ── 5. مصادر المفهوم ────────────────────────────────────────────────
    sources = []
    if concept_data:
        sources = concept_data.get("sources", [])
    if quran_matches and "القرآن الكريم" not in sources:
        sources.append("القرآن الكريم")

    return {
        "query":         query,
        "concept_data":  concept_data,
        "ckg_related":   ckg_related,
        "ckg_relations": ckg_relations,
        "root_matches":  root_matches,
        "quran_matches": quran_matches,
        "sources":       sources,
        "confidence":    confidence,
        "found":         bool(concept_data or quran_matches or root_matches),
    }


# ═══════════════════════════════════════════════════════════════════════════
# دوال العرض
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(value, label: str, wrap: bool = False, count_target: int | None = None):
    """بطاقة مقياس. إن مُرِّر count_target (عدد صحيح) فسيُشغَّل عدّاد
    متحرك من 0 حتى القيمة عند ظهور البطاقة، بدل عرضها ثابتة فوراً."""
    value_class = "metric-value metric-value--wrap" if wrap else "metric-value"
    data_attr = f' data-count-target="{count_target}"' if count_target is not None else ""
    # ملاحظة مهمة: نعرض القيمة الحقيقية "value" فوراً دائماً (وليس "0")،
    # لأن سكربت العدّاد المتحرك أدناه مُحقَن عبر st.markdown وليس عبر
    # components.html، وحقن <script> بهذا الأسلوب لا يُنفَّذه المتصفح
    # بشكل مضمون في كل مرة. لو فشل السكربت تبقى الأرقام الحقيقية ظاهرة
    # بدل أن تعلق على صفر؛ ولو نجح، يبدأ العدّ من 0 حتى نفس هذه القيمة.
    display_value = value
    st.markdown(f"""
    <div class="metric-card">
        <div class="{value_class}"{data_attr}>{display_value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def _copy_button(text: str, key: str, label: str = "📋 نسخ") -> None:
    """زر نسخ حديث بلمسة SaaS — ينسخ أي نص للحافظة عبر Clipboard API،
    قابل لإعادة الاستخدام بأي تبويب (نتيجة ترجمة، رد شات، سيناريو...)."""
    # json.dumps يُخرج علامات اقتباس مزدوجة، وهي نفس محرف الإحاطة المستخدم
    # لسمة onclick="..." أدناه؛ لولا ترميزها كـ&quot; لقطع المتصفح السمة
    # عند أول علامة اقتباس داخل النص المنسوخ.
    safe_text = json.dumps(text or "").replace('"', "&quot;")
    btn_id = f"nsm_copy_{key}"
    # ملاحظة: لا مسافات بادئة قبل وسوم الـHTML هنا عمداً. Markdown يعتبر أي
    # سطر يبدأ بـ4 مسافات فأكثر "code block" فيعرضه كنص خام بدل تنفيذه
    # كـHTML — وهذا كان يكسر الزر (يظهر كود الزر والسكربت كنص عادي).
    _html = (
        f'<button id="{btn_id}" class="nsm-copy-btn" onclick="'
        f"navigator.clipboard.writeText({safe_text});"
        f"this.innerText='✅ تم النسخ';"
        f"setTimeout(()=&gt;{{ this.innerText='{label}'; }}, 1500);"
        f'">{label}</button>'
    )
    st.markdown(_html, unsafe_allow_html=True)


def _skeleton(kind: str = "text", lines: int = 3) -> None:
    """حالة تحميل أنيقة (shimmer) بدل الدوارة الافتراضية — تُستخدم مع
    st.empty() فتُستبدل بالمحتوى الحقيقي فور جاهزيته:
        ph = st.empty()
        with ph.container(): _skeleton()
        ... عملية طويلة ...
        ph.empty()
    """
    if kind == "cards":
        html = '<div style="display:flex;gap:0.6rem;flex-wrap:wrap;">' + "".join(
            '<div class="skeleton-block" style="flex:1;min-width:120px;"></div>' for _ in range(4)
        ) + "</div>"
    else:
        widths = (["w-90", "w-70", "w-50"] * ((lines // 3) + 1))[:lines]
        html = "".join(f'<div class="skeleton-line {w}"></div>' for w in widths)
    st.markdown(html, unsafe_allow_html=True)


def render_home():
    """الصفحة الرئيسية — نظرة سريعة واستكشاف أقسام NSM."""

    # ── 🎬 كيف يعمل NSM؟ — دليل تفاعلي بخط أنابيب متحرك يشرح رحلة السؤال ──
    st.markdown('<div class="section-header">🎬 كيف يعمل NSM؟ <span class="live-dot"></span></div>',
                unsafe_allow_html=True)

    _pipeline_steps = [
        ("📝", "إدخال عربي", "تكتب سؤالك", "تكتب سؤالك أو مفهومك بالعربية الفصحى — بدون أي قوالب أو صياغة خاصة، تماماً كما تتحدث."),
        ("🌱", "تحليل الجذر", "استخراج الجذر اللغوي", "يحلّل النظام الجذر الثلاثي/الرباعي والبنية الصرفية للكلمة من قاعدة تضم آلاف الجذور العربية المكتشفة."),
        ("🕸️", "ربط CKG", "شبكة المفاهيم المعرفية", "يربط المفهوم بشبكة المعرفة الحية (CKG) — آلاف المفاهيم وعشرات آلاف العلاقات المستنتجة بينها."),
        ("📖", "مطابقة قرآنية", "بحث آية بآية", "يبحث آلياً عن الآيات القرآنية ذات الصلة الدلالية بالمفهوم، مربوطة بنفس شبكة الجذور والمعاني."),
        ("💬", "رد ذكي", "إجابة مدعومة بالسياق", "يولّد رداً نهائياً مدعوماً بالمصادر والسياق المستخرج من كل الخطوات السابقة، بالعربية الفصحى."),
    ]

    _nodes_html = "".join(
        f'''<div class="pipeline-node{' active' if i == 0 else ''}" data-step="{i}"
                data-title="{title}" data-text="{text}" data-icon="{icon}">
            <div class="pipeline-node-icon">{icon}</div>
            <div class="pipeline-node-title">{label}</div>
        </div>'''
        for i, (icon, label, title, text) in enumerate(_pipeline_steps)
    )
    _dots_html = "".join(
        f'<span class="{"active" if i == 0 else ""}" data-dot="{i}"></span>'
        for i in range(len(_pipeline_steps))
    )
    _icon0, _label0, _title0, _text0 = _pipeline_steps[0]

    st.markdown(f"""
    <div class="nsm-pipeline-wrap">
        <div class="nsm-pipeline" id="nsm-pipeline" tabindex="0"
             role="group" aria-label="خطوات عمل NSM — استخدم الأسهم للتنقّل">
            <div class="nsm-pipeline-track"><div class="nsm-pipeline-track-fill" id="nsm-pipeline-fill"></div></div>
            {_nodes_html}
        </div>
        <div class="pipeline-detail">
            <div class="pipeline-detail-inner" id="nsm-pipeline-detail">
                <div class="pipeline-detail-title"><span id="nsm-pd-icon">{_icon0}</span>
                    <span id="nsm-pd-title">{_title0}</span>
                    <span class="pipeline-step-counter" id="nsm-pd-counter">1 / {len(_pipeline_steps)}</span>
                </div>
                <div class="pipeline-detail-text" id="nsm-pd-text">{_text0}</div>
            </div>
        </div>
        <div class="pipeline-progress-hint">{_dots_html}</div>
    </div>
    """, unsafe_allow_html=True)

    st.components.v1.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const pipeline = doc.getElementById('nsm-pipeline');
        if (!pipeline || pipeline.dataset.nsmBound) return;
        pipeline.dataset.nsmBound = "1";

        const nodes = Array.from(pipeline.querySelectorAll('.pipeline-node'));
        const fill  = doc.getElementById('nsm-pipeline-fill');
        const dInner= doc.getElementById('nsm-pipeline-detail');
        const dIcon = doc.getElementById('nsm-pd-icon');
        const dTitle= doc.getElementById('nsm-pd-title');
        const dText = doc.getElementById('nsm-pd-text');
        const dCounter = doc.getElementById('nsm-pd-counter');
        const dots  = Array.from(doc.querySelectorAll('.pipeline-progress-hint span'));
        const total = nodes.length;
        let current = 0;
        let timer = null;
        let paused = false;

        function setActive(idx, fromClick) {
            current = ((idx % total) + total) % total;
            nodes.forEach((n, i) => n.classList.toggle('active', i === current));
            dots.forEach((d, i) => d.classList.toggle('active', i === current));
            if (fill) fill.style.width = (current / (total - 1) * 100) + '%';
            if (dInner) {
                dInner.style.opacity = '0';
                setTimeout(function() {
                    const n = nodes[current];
                    dIcon.textContent    = n.getAttribute('data-icon');
                    dTitle.textContent   = n.getAttribute('data-title');
                    dText.textContent    = n.getAttribute('data-text');
                    if (dCounter) dCounter.textContent = (current + 1) + ' / ' + total;
                    dInner.style.opacity = '1';
                }, 180);
            }
            if (fromClick) restart();
        }

        function tick() { if (!paused) setActive(current + 1, false); }
        function restart() {
            if (timer) clearInterval(timer);
            timer = setInterval(tick, 3400);
        }

        nodes.forEach((n, i) => {
            n.addEventListener('click', function() { setActive(i, true); });
        });

        // إيقاف مؤقت أثناء التحويم/اللمس حتى لا يفوّت القارئ الوصف
        pipeline.addEventListener('mouseenter', function() { paused = true; });
        pipeline.addEventListener('mouseleave', function() { paused = false; });
        pipeline.addEventListener('touchstart', function() { paused = true; }, { passive: true });

        // تنقّل بالأسهم (يمين/يسار) عند تركيز العنصر — يدعم اتجاه RTL
        pipeline.addEventListener('keydown', function(e) {
            if (e.key === 'ArrowLeft') { setActive(current + 1, true); e.preventDefault(); }
            else if (e.key === 'ArrowRight') { setActive(current - 1, true); e.preventDefault(); }
        });

        setActive(0, false);
        restart();
    })();
    </script>
    """, height=0)

    st.markdown("")
    st.markdown('<div class="section-header">🚀 استكشف NSM</div>', unsafe_allow_html=True)

    _features = [
        ("🔍", "البحث المعرفي", "ابحث عن أي مفهوم (الصبر، الجاذبية، الرحمة، العدل...) وشاهد الآيات المرتبطة والجذور والعلاقات المعرفية.", "📚 المعرفة"),
        ("💬", "محادثة ذكية", "تحدّث مع النظام بالعربية الفصحى، مدعوماً بشبكة المفاهيم المعرفية.", "💬 المحادثة"),
        ("📖", "القرآن الكريم", "بحث آية بآية، مرتبط تلقائياً بشبكة المفاهيم والجذور العربية.", "📚 المعرفة"),
        ("🤖", "الوكلاء الأذكياء", "وكلاء مستقلون للتنفيذ والتنسيق ضمن سرب ذكي متكامل.", "🤖 الوكلاء"),
        ("🎭", "المحتوى الإبداعي", "توليد نصوص ومحتوى إبداعي عربي بأسلوب متعدد الأنماط.", "🎭 إبداع"),
    ]
    _cards_html = "".join(f"""
            <div class="feature-card" data-tab-target="{_target_tab}" tabindex="0" role="button">
                <div class="feature-icon">{_icon}</div>
                <div class="feature-title">{_title}</div>
                <div class="feature-desc">{_desc}</div>
                <div class="feature-nav-hint">← انتقل إلى هذا القسم</div>
            </div>""" for _icon, _title, _desc, _target_tab in _features)
    st.markdown(f'<div class="feature-scroll">{_cards_html}</div>', unsafe_allow_html=True)

    # ── سكربت: عدّادات متحركة للمقاييس + نقر بطاقات الاستكشاف للتنقّل ──
    # تنبيه: كان هذا مُحقناً سابقاً عبر st.markdown، وهو أسلوب لا يُنفَّذ
    # فيه <script> أبداً (عنصر <script> المُدرَج عبر innerHTML لا يعمل،
    # سلوك موثّق بالمتصفحات وليس مجرد "أحياناً" — لهذا كان النقر على
    # البطاقات بلا أي أثر). الحل المضمون: st.components.v1.html الذي
    # يُنشئ iframe حقيقياً يُنفَّذ فيه JS، ومنه نصل للصفحة الأم عبر
    # window.parent.document (نفس الحل المطبَّق أعلاه لتلوين التبويبات).
    #
    # ملاحظة إضافية: فهرس كل تبويب هدف بترتيب قائمة _tab_defs الفعلية
    # (بدالة main، أسفل الملف) — يُستخدم كخط دفاع ثانٍ بعد المطابقة
    # النصية، لأن بنية DOM الداخلية لِـ st.tabs قد تختلف بين إصدارات
    # Streamlit (data-baseweb مقابل role="tab" ...إلخ)، فالاعتماد على
    # نص + فهرس معاً أكثر مقاومة لتغيّر الإصدار من نص فقط.
    # الترتيب الحالي: 0=الرئيسية 1=المعرفة 2=المحادثة 3=الوكلاء 4=إبداع
    _tab_index_map = {"📚 المعرفة": 1, "💬 المحادثة": 2, "🤖 الوكلاء": 3, "🎭 إبداع": 4}
    st.components.v1.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const TAB_INDEX = """ + str(_tab_index_map).replace("'", '"') + """;

        function findTabElements() {
            // عدّة استراتيجيات بترتيب الأولوية — أول واحدة تُعيد نتائج تُستخدم
            const strategies = [
                '.stTabs [role="tablist"] [role="tab"]',
                '[role="tablist"] [role="tab"]',
                '.stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]',
                '[data-baseweb="tab-list"] [data-baseweb="tab"]',
                '[data-testid="stTab"]'
            ];
            for (const sel of strategies) {
                const found = doc.querySelectorAll(sel);
                if (found && found.length) return Array.from(found);
            }
            return [];
        }

        function fireFullClick(el) {
            const opts = { bubbles: true, cancelable: true, view: doc.defaultView || window };
            try { el.dispatchEvent(new PointerEvent('pointerdown', opts)); } catch (e) {}
            el.dispatchEvent(new MouseEvent('mousedown', opts));
            el.dispatchEvent(new MouseEvent('mouseup', opts));
            el.dispatchEvent(new MouseEvent('click', opts));
            el.click();
        }

        function goToTab(label) {
            const tabs = findTabElements();
            if (!tabs.length) return false;
            const norm = s => (s || '').replace(/\\s+/g, ' ').trim();
            // 1) مطابقة نصية دقيقة
            let target = tabs.find(t => norm(t.textContent) === norm(label));
            // 2) مطابقة نصية جزئية (احتياط لو أضيف نص إضافي مخفي بالعنصر)
            if (!target) target = tabs.find(t => norm(t.textContent).includes(norm(label)));
            // 3) مطابقة بالفهرس الرقمي كخط دفاع أخير
            if (!target && TAB_INDEX.hasOwnProperty(label) && tabs[TAB_INDEX[label]]) {
                target = tabs[TAB_INDEX[label]];
            }
            if (!target) return false;
            fireFullClick(target);
            return true;
        }

        function bindAll() {
            // 1) عدّاد متحرك من 0 حتى القيمة الفعلية لكل بطاقة مقياس
            const counters = doc.querySelectorAll('.metric-value[data-count-target]');
            counters.forEach(function(el) {
                if (el.dataset.nsmAnimated) return;
                el.dataset.nsmAnimated = "1";
                const target = parseInt(el.getAttribute('data-count-target'), 10) || 0;
                const duration = 900;
                const start = performance.now();
                function tick(now) {
                    const p = Math.min(1, (now - start) / duration);
                    const eased = 1 - Math.pow(1 - p, 3);
                    el.textContent = Math.round(eased * target).toLocaleString('en-US');
                    if (p < 1) requestAnimationFrame(tick);
                    else el.textContent = target.toLocaleString('en-US');
                }
                requestAnimationFrame(tick);
            });

            // 2) نقر بطاقة الاستكشاف ← تفعيل تبويب Streamlit المطابق بالاسم
            const cards = doc.querySelectorAll('.feature-card[data-tab-target]');
            cards.forEach(function(card) {
                if (card.dataset.nsmBound) return;
                card.dataset.nsmBound = "1";
                card.addEventListener('click', function() {
                    const label = card.getAttribute('data-tab-target');
                    if (!goToTab(label)) {
                        // إعادة محاولة واحدة بعد لحظة قصيرة احتياطاً لتأخر رسم التبويبات
                        setTimeout(function() { goToTab(label); }, 200);
                    }
                });
                // إتاحة: تفعيل بالضغط على Enter/مسافة أيضاً (tabindex="0" role="button")
                card.addEventListener('keydown', function(e) {
                    if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); card.click(); }
                });
            });
        }

        bindAll();
        // البطاقات تُعاد رسمتها بكل rerun من Streamlit، فنراقب DOM
        // الصفحة الأم ونعيد الربط تلقائياً بدل الاكتفاء بمرة واحدة فقط.
        new MutationObserver(bindAll).observe(doc.body, { childList: true, subtree: true });
    })();
    </script>
    """, height=0)


def render_search():
    """تبويب البحث المعرفي — قلب النظام."""
    st.markdown('<div class="section-header">🔍 البحث المعرفي</div>', unsafe_allow_html=True)
    st.markdown("ابحث عن أي مفهوم وسيظهر لك ما يعرفه النظام عنه:")

    default_q = st.session_state.get("search_query", "")
    query = st.text_input(
        "",
        value=default_q,
        placeholder="اكتب مفهوماً... مثل: الصبر، الجاذبية، التوبة، العلم",
        key="main_search",
        label_visibility="collapsed",
    )

    # أمثلة سريعة
    st.markdown("**أمثلة:**")
    ex_cols = st.columns(6)
    examples = ["الصبر", "الرحمة", "العلم", "الجاذبية", "العدل", "الإيمان"]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex
                st.session_state["search_query"] = ex

    st.markdown("---")

    if not query.strip():
        st.info("اكتب مفهوماً في خانة البحث أعلاه لاستكشاف قاعدة المعرفة.")
        return

    # تنفيذ البحث
    with st.spinner("🔍 جارٍ البحث في قاعدة المعرفة..."):
        result = search_knowledge(query.strip())

    if not result["found"]:
        st.warning(f"لم يُعثر على معلومات كافية عن «{query}» حتى الآن. يتعلم النظام بشكل مستمر!")
        return

    # ── عرض النتائج ──────────────────────────────────────────────────────

    # بطاقة المفهوم الرئيسية — بنداء واحد متكامل (كانت مقسّمة على 3 نداءات
    # منفصلة سابقاً، ما يجعل Streamlit يرسم كل جزء كعنصر DOM مستقل، فلا
    # تلتف البطاقة فعلياً حول محتواها البصري رغم تطابق المظهر ظاهرياً)
    cdata = result["concept_data"]
    _stats_html = ""
    if cdata:
        _stats_html = f"""
        <div class="concept-stats">
            <div class="concept-stat"><span class="concept-stat-label">التصنيف</span><span class="concept-stat-value">{cdata.get('cluster', 'غير مصنّف')}</span></div>
            <div class="concept-stat"><span class="concept-stat-label">التكرار</span><span class="concept-stat-value">{cdata.get('frequency', 0):,} مرة</span></div>
            <div class="concept-stat"><span class="concept-stat-label">قوة المفهوم</span><span class="concept-stat-value">{cdata.get('strength', 0.0):.2%}</span></div>
        </div>
        """
    st.markdown(f"""
    <div class="concept-card">
        <div class="concept-name">💡 {result['query']}</div>
        {_stats_html}
    </div>
    """, unsafe_allow_html=True)

    # ── المفاهيم المرتبطة ────────────────────────────────────────────────
    related_concepts = []
    if result["ckg_related"]:
        related_concepts = result["ckg_related"]
    elif result["root_matches"]:
        related_concepts = [m[0] for m in result["root_matches"] if m[0] != query]

    if related_concepts:
        st.markdown('<div class="section-header">🔗 المفاهيم المرتبطة</div>', unsafe_allow_html=True)
        tags_html = ""
        for concept in related_concepts[:12]:
            tags_html += f'<span class="related-tag">{concept}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── العلاقات من CKG ──────────────────────────────────────────────────
    if result["ckg_relations"]:
        st.markdown('<div class="section-header">↔️ العلاقات المعرفية</div>', unsafe_allow_html=True)
        for rel in result["ckg_relations"][:6]:
            rel_type = rel.get("type", "مرتبط")
            weight   = rel.get("weight", 0)
            target   = rel.get("target", "")
            badge_color = "badge-blue"
            st.markdown(f"""
            <div class="root-item">
                <span class="badge {badge_color}">{rel_type}</span>
                &nbsp;→&nbsp; <strong>{target}</strong>
                &nbsp;&nbsp; <small style="color:var(--text-muted)">قوة: {weight:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    # ── الإشارات القرآنية ────────────────────────────────────────────────
    quran_matches = result["quran_matches"]
    if quran_matches:
        st.markdown(f'<div class="section-header">📖 الإشارات القرآنية ({len(quran_matches)} آية)</div>', unsafe_allow_html=True)
        for ayah in quran_matches[:6]:
            surah = ayah.get("surah", "")
            verse = ayah.get("ayah", "")
            text  = ayah.get("text", "")
            st.markdown(f"""
            <div class="quran-verse">
                {text}
                <div class="verse-ref">سورة {surah}، الآية {verse}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(quran_matches) > 6:
            with st.expander(f"عرض {len(quran_matches) - 6} آية إضافية"):
                for ayah in quran_matches[6:]:
                    surah = ayah.get("surah", "")
                    verse = ayah.get("ayah", "")
                    text  = ayah.get("text", "")
                    st.markdown(f"""
                    <div class="quran-verse">
                        {text}
                        <div class="verse-ref">سورة {surah}، الآية {verse}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">📖 الإشارات القرآنية</div>', unsafe_allow_html=True)
        st.info("لم يُعثر على آيات مباشرة لهذا المفهوم بهذه الصياغة. جرّب مرادفاً أو جذر الكلمة.")

    # ── المصادر ودرجة الثقة ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 تفاصيل البحث</div>', unsafe_allow_html=True)
    col_src, col_conf = st.columns(2)
    with col_src:
        sources = result["sources"] or ["الجذور العربية"]
        st.markdown(f"**المصادر:** {' ، '.join(sources)}")
    with col_conf:
        conf = result["confidence"]
        st.markdown(f"**درجة الثقة:** {conf:.0%}")
        st.progress(conf)

    # ── الجذور المرتبطة من الجذور العربية ────────────────────────────────
    root_matches = result["root_matches"]
    if root_matches:
        with st.expander("🌿 الجذور العربية المكتشفة"):
            for token, freq in root_matches[:10]:
                st.markdown(f"""
                <div class="root-item">
                    <strong>{token}</strong>
                    <span class="badge badge-green" style="float:left">تكرار: {freq:,}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── تحليل اللغة العربية (ArabicNLP) ─────────────────────────────────
    if _ARABIC_NLP_OK and query.strip():
        with st.expander("🔬 التحليل اللغوي العميق (ArabicNLP)"):
            try:
                _nlp_engine = get_arabic_engine(ckg=load_ckg())
                _analysis   = _nlp_engine.analyse(query.strip())
                _fv         = _analysis.feature_vector
                col_n1, col_n2, col_n3 = st.columns(3)
                with col_n1:
                    st.metric("نسبة الأفعال", f"{_fv.verb_score:.0%}")
                    st.metric("نسبة الأسماء", f"{_fv.noun_score:.0%}")
                with col_n2:
                    st.metric("تعقيد الجذور", f"{_fv.root_complexity:.0%}")
                    st.metric("التعقيد النحوي", f"{_fv.syntactic_complexity:.0%}")
                with col_n3:
                    st.metric("الكثافة الدلالية", f"{_fv.semantic_concept_score:.0%}")
                    st.metric("تماسك السياق", f"{_fv.context_score:.0%}")
                if _analysis.syntactic.tokens:
                    _pos_badge = {"verb": "blue", "noun": "purple", "adj": "purple", "particle": "amber"}
                    _tokens_html = " ".join(
                        f'<span class="badge badge-{_pos_badge.get(t.pos, "amber")}" style="margin:2px" '
                        f'title="جذر: {t.root or "—"} · وزن: {t.wazn or "—"}">{t.raw}</span>'
                        for t in _analysis.syntactic.tokens[:20]
                    )
                    st.markdown(f"**الرموز المُحلَّلة:** {_tokens_html}", unsafe_allow_html=True)
                if _analysis.morphological.unique_roots:
                    st.markdown(f"**الجذور المكتشفة:** `{'، '.join(_analysis.morphological.unique_roots[:8])}`")
            except Exception as _nlp_err:
                st.caption(f"تعذّر التحليل: {_nlp_err}")

    # ── بحث الويب الحقيقي ────────────────────────────────────────────────
    if _WEB_SEARCH_OK:
        st.markdown("")
        st.markdown('<div class="section-header">🌐 بحث في الإنترنت</div>', unsafe_allow_html=True)
        _ws_cols = st.columns([3, 1])
        with _ws_cols[0]:
            _ws_q = st.text_input(
                "ابحث في الويب",
                value=query.strip() if query.strip() else "",
                placeholder="اكتب ما تريد البحث عنه في الإنترنت...",
                key="web_search_query",
                label_visibility="collapsed",
            )
        with _ws_cols[1]:
            _ws_btn = st.button("🌐 ابحث", key="web_search_btn", use_container_width=True)

        if _ws_btn and _ws_q.strip():
            with st.spinner("⟳ جارٍ البحث في الإنترنت (DuckDuckGo)..."):
                _ws_result = _web_search(_ws_q.strip(), max_results=6)
            st.markdown(f"""
            <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                        padding:1rem 1.4rem;direction:rtl;line-height:1.9;
                        white-space:pre-wrap;font-size:0.93rem;border:1px solid #1e3a5f">
            {_ws_result}
            </div>
            """, unsafe_allow_html=True)

    # ── بحث حقيقي عن الصور (Unsplash) ───────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">🖼️ بحث عن الصور</div>', unsafe_allow_html=True)
    try:
        from ai.image_search_tool import image_search_safe as _img_search
        _IMG_SEARCH_OK = True
    except Exception as _img_imp_err:
        _IMG_SEARCH_OK = False
        st.caption(f"⚠️ تعذّر تحميل أداة بحث الصور: {_img_imp_err}")

    if _IMG_SEARCH_OK:
        _is_cols = st.columns([3, 1])
        with _is_cols[0]:
            _is_q = st.text_input(
                "ابحث عن صور",
                placeholder="مثال: مسجد، طبيعة، خط عربي...",
                key="image_search_query",
                label_visibility="collapsed",
            )
        with _is_cols[1]:
            _is_btn = st.button("🖼️ ابحث", key="image_search_btn", use_container_width=True)

        if _is_btn and _is_q.strip():
            with st.spinner("⟳ جارٍ البحث عن الصور (Unsplash)..."):
                _is_result = _img_search(_is_q.strip(), max_results=9)

            if not _is_result["ok"]:
                st.error(f"❌ {_is_result['error']}")
            else:
                _is_images = _is_result["results"]
                _is_grid = st.columns(3)
                for _i, _img in enumerate(_is_images):
                    with _is_grid[_i % 3]:
                        st.image(_img["thumb_url"] or _img["url"], use_container_width=True)
                        _cap = _img["description"] or "بدون وصف"
                        st.caption(f"📷 {_cap}")
                        if _img.get("author"):
                            _author_line = f"[{_img['author']}]({_img['author_url']})" if _img.get("author_url") else _img["author"]
                            st.caption(f"بواسطة {_author_line}", unsafe_allow_html=False)


def render_quran():
    """تبويب القرآن الكريم."""
    st.markdown('<div class="section-header">📖 القرآن الكريم في النظام</div>', unsafe_allow_html=True)

    quran_index = load_quran_index()
    ayat        = load_all_quran_ayat()
    roots       = load_arabic_roots()

    # إحصاءات
    col1, col2, col3 = st.columns(3)
    with col1: metric_card(f"{quran_index.get('total_ayat', len(ayat)):,}", "آية محملة")
    with col2: metric_card(f"{quran_index.get('total_surahs', 114)}", "سورة")
    with col3: metric_card(f"{len(roots):,}", "مفهوم مستخرج")

    st.markdown("")

    # أكثر المفاهيم تكراراً
    st.markdown('<div class="section-header">🔝 أكثر المفاهيم تكراراً في القرآن</div>', unsafe_allow_html=True)

    # فلترة الجذور ذات المعنى
    filtered = {k: v for k, v in roots.items()
                if len(normalize_arabic(k)) >= 3
                and v.get("frequency", 0) > 50
                and normalize_arabic(k) not in {
                    "من", "في", "على", "إلى", "عن", "مع", "الا", "ومن",
                    "وان", "بهۦ", "بما", "وما", "الذ", "وقا", "وله"
                }}

    top_concepts = sorted(filtered.items(), key=lambda x: x[1].get("frequency", 0), reverse=True)[:20]

    if top_concepts:
        # رسم بياني
        try:
            import plotly.graph_objects as go
            names = [v.get("top_token", k) for k, v in top_concepts[:15]]
            freqs = [v.get("frequency", 0) for _, v in top_concepts[:15]]

            _theme = THEMES.get(st.session_state.get("ui_theme", "dark"), THEMES["dark"])
            fig = go.Figure(go.Bar(
                x=freqs,
                y=names,
                orientation='h',
                marker_color=_theme["gold"],
                text=freqs,
                textposition='outside',
                textfont=dict(color=_theme["text"]),
            ))
            fig.update_layout(
                height=450,
                margin=dict(l=20, r=60, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=_theme["text"]),
                yaxis=dict(autorange="reversed", color=_theme["text"], gridcolor=_theme["border"]),
                xaxis=dict(color=_theme["text"], gridcolor=_theme["border"]),
                xaxis_title="التكرار",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for k, v in top_concepts[:10]:
                token = v.get("top_token", k)
                freq  = v.get("frequency", 0)
                st.markdown(f"**{token}**: {freq:,} مرة")
    else:
        st.info("لم تُكتشف مفاهيم بعد. يحتاج النظام إلى تدريب إضافي.")

    # بحث داخل القرآن
    st.markdown('<div class="section-header">🔍 البحث في آيات القرآن</div>', unsafe_allow_html=True)
    quran_q = st.text_input("بحث قرآن", placeholder="ابحث عن كلمة أو مفهوم...", key="quran_search",
                             label_visibility="collapsed")
    if quran_q.strip():
        matches = search_quran_for_concept(quran_q.strip(), ayat, max_results=20)
        if matches:
            st.success(f"وُجد {len(matches)} آية تحتوي على «{quran_q}»")
            for ayah in matches:
                surah = ayah.get("surah", "")
                verse = ayah.get("ayah", "")
                text  = ayah.get("text", "")
                st.markdown(f"""
                <div class="quran-verse">
                    {text}
                    <div class="verse-ref">سورة {surah}، الآية {verse}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"لم يُعثر على «{quran_q}» في الآيات المحملة.")


def render_qa():
    """تبويب الأسئلة والأجوبة القرآني — يعتمد على CKG والآيات فقط."""
    st.markdown('<div class="section-header">❓ الأسئلة والأجوبة القرآني</div>', unsafe_allow_html=True)
    _qa_ckg = load_ckg()
    _qa_concepts_n = len(_qa_ckg.get("concepts", {}))
    _qa_relations_n = len(_qa_ckg.get("relations", {}))
    _qa_ayat_n = load_quran_index().get("total_ayat", 6236)
    st.markdown(
        f'<p style="color:var(--text-muted)">اسأل سؤالاً بالعربية، وسيحلل النظام السؤال '
        f'ويبحث في {_qa_concepts_n:,} مفهوماً و{_qa_relations_n:,} علاقة دلالية و{_qa_ayat_n:,} آية للإجابة.</p>',
        unsafe_allow_html=True,
    )

    # ── أمثلة جاهزة ──
    st.markdown("**أمثلة:**")
    examples = [
        "من هو محمد ﷺ؟",
        "ما علاقة الصبر بالإيمان؟",
        "ماذا يقول القرآن عن العدل؟",
        "ما قصة يوسف؟",
    ]
    ex_cols = st.columns(len(examples))
    chosen_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"qa_example_{i}", use_container_width=True):
                chosen_example = ex

    default_q = chosen_example or st.session_state.get("qa_question", "")
    st.session_state.setdefault("qa_conversation_history", [])
    question = st.text_input(
        "اكتب سؤالك هنا:",
        value=default_q,
        key="qa_input",
        placeholder="مثال: ما علاقة الصبر بالإيمان؟",
    )
    st.session_state["qa_question"] = question

    opt_col1, opt_col2, opt_col3 = st.columns([1, 1, 3])
    with opt_col1:
        show_reasoning = st.checkbox(
            "🧠 اعرض لماذا هذه الإجابة",
            value=st.session_state.get("qa_show_reasoning", False),
            key="qa_show_reasoning",
        )
    with opt_col2:
        show_images = st.checkbox(
            "🖼️ صور توضيحية",
            value=st.session_state.get("qa_show_images", False),
            key="qa_show_images",
        )
    with opt_col3:
        if st.session_state["qa_conversation_history"]:
            st.caption(f"💬 سياق محادثة نشط ({len(st.session_state['qa_conversation_history'])} سؤال سابق)")
            if st.button("🗑️ مسح سياق المحادثة", key="qa_clear_context"):
                st.session_state["qa_conversation_history"] = []
                st.rerun()

    ask = st.button("🔍 اسأل", type="primary")

    if not (ask or chosen_example) or not question.strip():
        return

    ckg  = load_ckg()
    ayat = load_all_quran_ayat()

    if not ckg.get("concepts"):
        st.error("الذاكرة الدلالية (CKG) فارغة — لا يمكن الإجابة على الأسئلة حالياً.")
        return

    with st.spinner("يتم تحليل السؤال والبحث في قاعدة المعرفة..."):
        entities = load_entities()
        result = answer_question(
            question, ckg, ayat, entities=entities,
            generation_mode=st.session_state.get("yemeni_generation_mode", False),
            temperature=st.session_state.get("yemeni_temperature", 0.8),
            top_p=st.session_state.get("yemeni_top_p", 0.95),
            top_k=st.session_state.get("yemeni_top_k", 50),
            include_reasoning_trace=show_reasoning,
            include_images=show_images,
            conversation_history=st.session_state["qa_conversation_history"],
        )

    # ── حظر أمان (nova_system.py) — أولوية على أي عرض آخر، لا LoRA ولا حلقة ──
    if result.get("safety_blocked"):
        st.markdown("---")
        st.warning(f"🛡️ {result['summary']}")
        return

    if result.get("generation_used") and result.get("generated_text"):
        st.markdown("---")
        st.markdown('<div class="section-header">🗣️ توليد حر (تجريبي)</div>', unsafe_allow_html=True)
        st.caption("نص مولَّد بواسطة YemeniDecoder — تجريبي وغير مضمون الدقة، منفصل عن الإجابة الرمزية أدناه.")
        st.info(result["generated_text"])

    # ── حفظ الدور الحالي في سياق المحادثة (لأسئلة المتابعة القادمة) ──
    # يُستبعد عمداً أي رد محظور أمنياً (return أعلاه) حتى لا يتلوّث سياق
    # الأسئلة اللاحقة بمحتوى مرفوض. سقف 5 أدوار لتفادي تضخّم prompt الـLLM
    # بلا حدود مع طول الجلسة.
    st.session_state["qa_conversation_history"].append(
        {"question": question, "summary": result.get("summary", "")}
    )
    st.session_state["qa_conversation_history"] = st.session_state["qa_conversation_history"][-5:]

    # ── حفظ الحلقة في الذاكرة التجريبية ──
    db_path = MEMORY_DIR / "episodic.db"
    try:
        store_episode(db_path, question, result)
    except Exception:
        pass

    # ── أسئلة سابقة مشابهة ──
    try:
        similar = find_similar_episodes(db_path, question, threshold=0.4, top_k=3)
    except Exception:
        similar = []

    st.markdown("---")

    if similar:
        st.markdown('<div class="section-header">🕘 أسئلة سابقة مشابهة</div>', unsafe_allow_html=True)
        for s in similar:
            if normalize_arabic(s["question"]) == normalize_arabic(question):
                continue
            st.markdown(f"""
            <div class="root-item">
                <strong>{s['question']}</strong>
                <span class="badge badge-blue">تشابه: {s['similarity']:.0%}</span>
                <span class="badge badge-amber">ثقة: {s['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")

    # ── ملخص الإجابة ──
    entity_info = result.get("entity")
    if entity_info:
        st.markdown(
            f'<div class="section-header">📝 ملخص الإجابة '
            f'<span class="badge badge-purple">كيان: {entity_info["name"]} ({entity_info["type"]})</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-header">📝 ملخص الإجابة</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="root-item" style="font-size:1.05rem; line-height:1.8">
        {result['summary']}
    </div>
    """, unsafe_allow_html=True)

    # ── درجة الثقة ──
    confidence = result.get("confidence", 0.0)
    st.markdown("")
    st.markdown(f"**درجة الثقة:** {confidence:.0%}")
    st.progress(confidence)

    # ── أثر التفكير (اختياري — ai/chain_of_thought.py) ──
    if result.get("reasoning_trace"):
        with st.expander("🧠 لماذا هذه الإجابة؟"):
            st.markdown(result["reasoning_trace"])

    # ── صور توضيحية (اختياري — ai/image_sources.py) — بهوية زجاجية موحَّدة ──
    images = result.get("images") or []
    if images:
        st.markdown('<div class="section-header">🖼️ صور توضيحية</div>', unsafe_allow_html=True)
        img_cols = st.columns(len(images))
        for col, img in zip(img_cols, images):
            with col:
                url = img.get("url", "")
                source_label = img.get("source", "")
                if url.startswith("https://"):  # فحص أمان بسيط قبل الحقن في HTML
                    st.markdown(f"""
                    <div class="glass-card" style="padding:0.6rem; text-align:center;">
                        <img src="{url}" style="width:100%; border-radius:12px; display:block;"
                             loading="lazy" />
                        <div style="margin-top:0.5rem;">
                            <span class="badge badge-green">المصدر: {source_label}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── تغذية راجعة: تدريب LoRA خفيف من ملاحظة المستخدم (لا يمسّ الأوزان الأساسية) ──
    _fb_key = f"qa_feedback_{hash(question)}"
    if st.session_state.get(_fb_key) is None:
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
        with fb_col1:
            if st.button("👍 إجابة جيدة", key=f"{_fb_key}_up"):
                try:
                    record_positive_feedback(question, result.get("summary", ""))
                except Exception:
                    pass
                st.session_state[_fb_key] = "up"
                st.rerun()
        with fb_col2:
            if st.button("👎 غير دقيقة", key=f"{_fb_key}_down"):
                # لا تدريب على الملاحظات السلبية حالياً (قد يزعزع الشبكة
                # بدون آلية contrastive loss مناسبة) — فقط تسجيل للمراجعة.
                st.session_state[_fb_key] = "down"
                st.rerun()
    else:
        _fb = st.session_state[_fb_key]
        if _fb == "up":
            st.success("✅ شكراً! تم استخدام ملاحظتك لتحسين الفهم الدلالي للنموذج.")
        else:
            st.info("📝 شكراً على ملاحظتك — تم تسجيلها للمراجعة.")

    if not result["primary_concepts"]:
        st.info("لم يتم العثور على مفاهيم مرتبطة بهذا السؤال في قاعدة المعرفة الحالية.")
        return

    # ── المفاهيم الأساسية ──
    st.markdown("")
    st.markdown('<div class="section-header">🧩 المفاهيم المستخرجة من السؤال</div>', unsafe_allow_html=True)
    for c in result["primary_concepts"]:
        if entity_info:
            # في إجابات الكيانات، أرقام "تكرار/تطابق" التقنية لا تضيف
            # قيمة للمستخدم — نعرض فقط الاسم والمجموعة المعرفية
            st.markdown(f"""
            <div class="root-item">
                <strong>{c['name']}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{c['cluster']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="root-item">
                <strong>{c['name']}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{c['cluster']}</span>
                <span class="badge badge-blue">تكرار في القرآن: {c['frequency']}</span>
                <span class="badge badge-amber">درجة التطابق: {c['match']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── المفاهيم المرتبطة (من العلاقات) ──
    related = result.get("related_concepts", [])
    if related:
        st.markdown("")
        st.markdown('<div class="section-header">🔗 مفاهيم مرتبطة (من الذاكرة الدلالية)</div>', unsafe_allow_html=True)
        rel_type_labels = {
            "co_occurrence":     "تزامن في الآية",
            "semantic":          "علاقة دلالية",
            "thematic_cluster":  "تجمّع موضوعي",
            "root_link":         "ربط بجذر",
            "narrative_sequence": "تسلسل سردي",
            "episodic_rule":     "قاعدة من الذاكرة التجريبية",
            "entity_attribute":  "صفة الكيان",
        }
        for r in related[:6]:
            rtype = rel_type_labels.get(r["relation_type"], r["relation_type"])
            st.markdown(f"""
            <div class="root-item">
                <strong>{r['concept']}</strong>
                <span class="badge badge-blue">نوع العلاقة: {rtype}</span>
                <span class="badge badge-amber">وزن العلاقة: {r['weight']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── الآيات الداعمة ──
    verses = result.get("verses", [])
    st.markdown("")
    st.markdown(f'<div class="section-header">📖 الآيات الداعمة ({len(verses)})</div>', unsafe_allow_html=True)
    if verses:
        for v in verses:
            st.markdown(f"""
            <div class="quran-verse">
                {v['text']}
                <div class="verse-ref">سورة {v['surah']}، الآية {v['ayah']} — مفهوم: {v['concept']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("لم يتم العثور على آيات داعمة مباشرة لهذا السؤال.")


def render_higgsfield():
    """
    تبويب 🎬 Higgsfield Explainer — وثائقي AI حتى 10 دقائق.
    Pipeline (محرك مجاني بالكامل — بدون أي اعتماد على مزوّد مدفوع):
    LLMFallback الداخلي لـNSM (بحث + سرد) → FableEngine.generate_explainer
    (سيناريو مُقسّم مشاهد) → TTSEngine (صوت مجاني: Edge TTS/gTTS، أو Gemini
    TTS إن توفّر مفتاح) → VideoEngine (رندر mp4 فعلي بخلفيات متحركة
    وترجمات Kinetic Captions). خلفيات سينمائية حقيقية عبر Higgsfield تبقى
    متاحة فقط كخيار اختياري (opt-in) معطَّل افتراضياً، تماماً كما في
    تبويب ⚡ Shorts.
    """
    # ── استيراد المحرك (نفس محرك السرد/الفيديو المجاني المستخدم في
    #    تبويب 🎭 إبداع، بدل ai.higgsfield_engine المدفوع) ──────────────
    try:
        from ai.llm_fallback import LLMFallback as _HFLLMFallback
        from ai.fable_engine import FableEngine
    except Exception as _hf_err:
        st.error(f"⚠️ تعذّر تحميل محرك السيناريو/الفيديو: {_hf_err}")
        return

    if "hf_fable_engine" not in st.session_state:
        _hf_fb = _HFLLMFallback(model_key="fable")
        st.session_state.hf_fable_engine = FableEngine(
            llm_fallback=_hf_fb, db_path=str(MEMORY_DIR / "fable.db")
        )
    engine = st.session_state.hf_fable_engine

    # ── رأس الصفحة ────────────────────────────────────────────────────
    st.markdown("""
    <div style="direction:rtl; text-align:right">
        <h2 style="margin-bottom:0.25rem">🎬 Higgsfield Explainer</h2>
        <p style="color:var(--text-muted); font-size:0.95rem; margin-top:0">
            أنشئ فيديو وثائقياً من أي موضوع — حتى 10 دقائق — سيناريو
            وصوت وفيديو mp4 فعلي، <strong>مجاناً بالكامل</strong> (بدون
            أي مفتاح API مدفوع مطلوب).
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── لوحة الإعداد ──────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        topic = st.text_input(
            "🎯 موضوع الوثائقي:",
            placeholder="مثال: نشوء الحضارة الإسلامية في الأندلس، كيف تعمل الثقوب السوداء...",
            key="hf_topic",
        )
    with col_r:
        minutes = st.slider(
            "⏱️ المدة المستهدفة (دقائق):",
            min_value=1, max_value=10, value=5,
            key="hf_minutes",
        )

    # ── معلومات Pipeline ───────────────────────────────────────────────
    with st.expander("ℹ️ كيف يعمل الـ Pipeline؟", expanded=False):
        st.markdown("""
        <div style="direction:rtl; text-align:right; font-size:0.9rem">
        <ol>
            <li><strong>🔍 محرك البحث/السرد الداخلي لـNSM</strong> — يبحث
                في المعلومات ويكتب سيناريو المشاهد (نص السرد + توجيه مرئي
                مقترح لكل مشهد)</li>
            <li><strong>🔊 TTSEngine</strong> — يحوّل السرد لصوت فعلي
                (Edge TTS مجاني بدون مفتاح، أو gTTS احتياطياً، أو
                Gemini TTS إن توفّر مفتاح)</li>
            <li><strong>🎬 VideoEngine</strong> — يركّب فيديو mp4 فعلي
                (خلفية متحركة + ترجمات متحركة كلمة-بكلمة) — كل ذلك
                محلياً بدون أي مزوّد خارجي مدفوع</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── زر الإنشاء (السيناريو) ────────────────────────────────────────
    generate_btn = st.button(
        "🎬 أنشئ السيناريو",
        type="primary",
        use_container_width=True,
        disabled=not bool(topic and topic.strip()),
        key="hf_generate_btn",
    )

    if generate_btn:
        if not topic.strip():
            st.warning("أدخل موضوع الوثائقي أولاً.")
        else:
            with st.spinner("⟳ يُجري بحثاً ويكتب السيناريو..."):
                try:
                    st.session_state.hf_script = engine.generate_explainer(
                        topic.strip(), target_minutes=minutes
                    )
                    st.session_state.hf_error = None
                    st.session_state.hf_mp4 = None
                except Exception as e:  # noqa: BLE001
                    # لا نمسح hf_script السابق هنا عمداً: لو كان لدى
                    # المستخدم سيناريو ناجح سابقاً وحاول توليد موضوع جديد
                    # ففشلت المحاولة (شبكة/مزوّد LLM مؤقتاً)، يبقى السيناريو
                    # القديم ظاهراً بدل أن يفقده بلا داعٍ.
                    logger.exception("فشل توليد سيناريو Higgsfield Explainer: %s", e)
                    st.session_state.hf_error = str(e)

    _hf_err = st.session_state.get("hf_error")
    if _hf_err:
        st.error(f"⚠️ تعذّر إنشاء السيناريو، حاول مرة أخرى. (تفصيل تقني: {_hf_err})")

    script = st.session_state.get("hf_script")
    if script is not None:
        _render_hf_result(script)


def _render_hf_result(script):
    """يعرض نتائج Higgsfield Explainer (سيناريو + رندر فيديو مجاني)."""
    segments = script.segments

    # ── ملخص ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("📽️ عدد المشاهد", len(segments))
    total_min = script.total_seconds // 60
    total_sec = script.total_seconds % 60
    c2.metric("⏱️ المدة الإجمالية", f"~{total_min}د {total_sec}ث")
    c3.metric("✍️ مزوّد السرد", script.provider or "—")

    if script.error:
        st.caption(f"⚠️ ملاحظة تقنية: {script.error}")

    st.markdown("---")

    # ── بطاقات المشاهد ────────────────────────────────────────────────
    st.markdown(
        f'<h3 style="direction:rtl; text-align:right">📜 مشاهد الوثائقي — {script.title}</h3>',
        unsafe_allow_html=True,
    )
    _full_script_text = "\n\n".join(
        f"[المشهد {s.index}]\n{s.narration}" for s in segments
    )
    _copy_button(_full_script_text, key="hf_full_script", label="📋 نسخ السيناريو كاملاً")

    for seg in segments:
        with st.expander(
            f"🎬 المشهد {seg.index}  (~{seg.est_seconds}ث)",
            expanded=(seg.index == 1),
        ):
            st.markdown(
                f"""
                <div style="direction:rtl; text-align:right; line-height:1.8">
                <p style="margin-top:0.25rem">
                    <strong>🔊 السرد الصوتي:</strong><br>{seg.narration}
                </p>
                <p style="color:var(--text-muted); font-size:0.9rem">
                    <strong>🎥 التوجيه المرئي:</strong> {seg.visual_notes or "—"}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── تصدير النص الكامل للسرد ──────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 النص الكامل للسرد (للتعليق الصوتي)"):
        st.text_area(
            "نص السرد:",
            value=script.full_narration,
            height=300,
            key="hf_full_narration",
        )
        st.download_button(
            "⬇️ تحميل السيناريو كملف نصي",
            data=script.full_narration,
            file_name=f"{(script.title or 'وثائقي')[:40]}.txt",
            mime="text/plain",
            key="hf_script_download",
        )

    # ── رندر الفيديو الفعلي (mp4) — مجاني بالكامل ─────────────────────
    st.markdown("---")
    st.markdown("#### 🎬 رندر الفيديو الفعلي (mp4) — مجاني")

    _HF_VOICE_OPTIONS = {
        "🎙️ افتراضي (تلقائي حسب المزوّد المتاح)": "",
        "👨 حامد — سعودي (Edge, مجاني)": "ar-SA-HamedNeural",
        "👩 زارية — سعودية (Edge, مجاني)": "ar-SA-ZariyahNeural",
        "👨 شاكر — مصري (Edge, مجاني)": "ar-EG-ShakirNeural",
        "👩 سلمى — مصرية (Edge, مجاني)": "ar-EG-SalmaNeural",
        "👨 حمدان — إماراتي (Edge, مجاني)": "ar-AE-HamdanNeural",
        "👩 فاطمة — إماراتية (Edge, مجاني)": "ar-AE-FatimaNeural",
        "✨ Kore — Gemini TTS (يتطلب GOOGLE_API_KEY)": "Kore",
    }
    _hf_voice_label = st.selectbox(
        "🗣️ اختر الصوت",
        options=list(_HF_VOICE_OPTIONS.keys()),
        key="hf_voice_select",
        help="الأصوات المجانية (Edge) لا تحتاج أي مفتاح API.",
    )
    _hf_voice = _HF_VOICE_OPTIONS[_hf_voice_label]

    _hf_key_present = bool(os.getenv("HIGGSFIELD_API_KEY", "").strip())
    _hf_use_cinematic_bg = st.checkbox(
        "🎥 خلفيات سينمائية حقيقية (اختياري)",
        value=False,
        key="hf_cinematic_bg_toggle",
        help="بدل الخلفية المتدرّجة المجانية الافتراضية، يولّد خلفية فيديو حقيقية لكل مشهد.",
    )
    _hf_cinematic_provider = "higgsfield"
    if _hf_use_cinematic_bg:
        _hf_provider_label = st.radio(
            "المزوّد",
            options=[
                "💳 Higgsfield (مدفوع — أسرع وأدق)",
                "🆓 Wan2.1 مجاني ⚡ Running on Zero (GPU حقيقي مجاني)",
            ],
            key="hf_cinematic_provider_radio",
            horizontal=True,
            help=(
                "Higgsfield: يستهلك رصيدك بالمزوّد، يتطلب HIGGSFIELD_API_KEY."
                + ("" if _hf_key_present else " (المفتاح غير موجود بالبيئة حالياً)")
                + "\n\nWan2.1 مجاني: نموذج مفتوح المصدر يشتغل فعلياً على "
                "GPU A100 مجاني عبر Hugging Face ZeroGPU (مساحات مُوسومة "
                "رسمياً \"Running on Zero\" على Hugging Face — ليست محاكاة) "
                "— أبطأ بكثير (طابور GPU مشترك) وقد يتعطّل أحياناً؛ عند "
                "فشله يتراجع تلقائياً للخلفية المتدرّجة لنفس المشهد فقط. "
                "HF_TOKEN اختياري لتحسين حد الاستخدام."
                "\n\nملاحظة: يُجرَّب LTX-Video أولاً (أسرع)، ثم Wan2.2، ثم "
                "Wan2.1 — تلقائياً وبالترتيب حتى ينجح أحدها."
            ),
        )
        _hf_cinematic_provider = "wan_free" if "Wan2.1" in _hf_provider_label else "higgsfield"
        if _hf_cinematic_provider == "wan_free":
            st.markdown(
                '<div style="margin:0.3rem 0 0.6rem;">'
                '<span class="badge badge-green">🟢 Running on Zero</span> '
                '<span class="badge badge-blue" style="margin-right:6px;">'
                "GPU A100 مجاني حقيقي — Hugging Face ZeroGPU</span></div>",
                unsafe_allow_html=True,
            )
            _render_wan_free_status_widget("hf_explainer")

    _pexels_key_present = bool(os.getenv("PEXELS_API_KEY", "").strip())
    st.caption(
        ("🖼️ صور خلفية حقيقية مجانية (Pexels) مفعَّلة تلقائياً بدل التدرّج اللوني الفارغ."
         if _pexels_key_present else
         "💡 تلميح: أضِف PEXELS_API_KEY (مجاني بالكامل — تسجيل فوري عبر "
         "pexels.com/api) لاستبدال التدرّج اللوني الفارغ بصور خلفية حقيقية "
         "تطابق كل مشهد، بدون أي تكلفة.")
    )

    _hf_use_music = st.checkbox(
        "🎵 موسيقى خلفية هادئة (مجانية، مولَّدة تلقائياً — اختياري)",
        value=False,
        key="hf_bg_music_toggle",
        help=(
            "سجادة صوتية محيطية هادئة بلا لحن أو إيقاع واضح، تُولَّد "
            "داخلياً بدون أي ملف موسيقى خارجي أو مزوّد مدفوع — منخفضة "
            "جداً تحت السرد الصوتي فقط. مُعطَّلة افتراضياً لأن بعض "
            "الجمهور بالمحتوى المعرفي الإسلامي يُفضّل عدم وجود موسيقى "
            "إطلاقاً — فعّلها فقط إن كانت مناسبة لجمهورك."
        ),
    )
    _hf_music_volume = 0.10
    if _hf_use_music:
        _hf_music_volume = st.slider(
            "🔊 حجم الموسيقى النسبي",
            min_value=0.03, max_value=0.25, value=0.10, step=0.01,
            key="hf_bg_music_volume",
            help="منخفض = بالكاد يُلاحَظ تحت السرد. مرتفع = أوضح لكن قد يزاحم الصوت.",
        )

    if st.button("🎬 أنشئ الفيديو الآن", type="primary", key="hf_render_video_btn"):
        try:
            _hf_spinner_msg = (
                "⏳ يولّد السرد الصوتي والخلفيات السينمائية ثم يركّب الفيديو... "
                "قد يستغرق عدة دقائق"
                if _hf_use_cinematic_bg else
                "⏳ يولّد السرد الصوتي ثم يركّب الفيديو... قد يستغرق دقيقة"
            )
            with st.spinner(_hf_spinner_msg):
                engine = st.session_state.hf_fable_engine
                mp4_bytes = engine.render_video(
                    script, voice=_hf_voice,
                    use_cinematic_backgrounds=_hf_use_cinematic_bg,
                    cinematic_provider=_hf_cinematic_provider,
                    use_background_music=_hf_use_music,
                    music_volume=_hf_music_volume,
                    wan_skip_spaces=st.session_state.get("hf_explainer_wan_dead_spaces"),
                )
            st.session_state.hf_mp4 = mp4_bytes
            st.success("✅ تم إنتاج الفيديو")
        except MemoryError:
            # لا نسجّل traceback هنا عمداً — العملية غالباً تكون بالفعل
            # بذاكرة شبه ممتلئة، وتسجيل traceback ثقيل إضافي قد يزيد
            # الضغط سوءاً في هذه اللحظة تحديداً.
            st.error(
                "⚠️ نفدت الذاكرة أثناء الرندر — جرّب مدة أقصر (دقيقتين-3 "
                "بدل 10) أو عطّل «الخلفيات السينمائية الحقيقية» إن كانت مفعّلة."
            )
        except Exception as e:  # noqa: BLE001
            # نسجّل التتبّع الكامل بسجلات السيرفر (يظهر بلوحة Streamlit
            # Cloud logs) — سابقاً كان يُعرَض str(e) فقط للمستخدم وتُفقَد
            # بقية تفاصيل الخطأ نهائياً، ما يصعّب تشخيص أعطال الإنتاج.
            logger.exception("فشل رندر فيديو Higgsfield Explainer: %s", e)
            _err_name = type(e).__name__
            if "Timeout" in _err_name or "timed out" in str(e).lower():
                st.error(
                    "⚠️ انتهت مهلة الانتظار أثناء الرندر (غالباً بسبب جلب "
                    "خلفية سينمائية/صورة خارجية بطيئة الاستجابة) — جرّب "
                    "مرة أخرى، أو عطّل «الخلفيات السينمائية الحقيقية»."
                )
            else:
                st.error(
                    f"⚠️ فشل رندر الفيديو ({_err_name}) — تم تسجيل تفاصيل "
                    f"الخطأ. جرّب مرة أخرى أو بمدة أقصر. (رسالة تقنية: {e})"
                )

    mp4_bytes = st.session_state.get("hf_mp4")
    if mp4_bytes:
        st.video(mp4_bytes)
        _hf_dl_cols = st.columns(2)
        with _hf_dl_cols[0]:
            st.download_button(
                "⬇️ تحميل الفيديو (mp4)",
                data=mp4_bytes,
                file_name=f"{(script.title or 'documentary')[:40]}.mp4",
                mime="video/mp4",
                key="hf_download_mp4",
            )
        with _hf_dl_cols[1]:
            # ── ملف ترجمة SRT منفصل — مبني من نفس بيانات توقيت الترجمات
            #    المستخدمة أصلاً بالفيديو (راجع ai.video_engine.build_srt)،
            #    مفيد لمنصات تتطلب ترجمة منفصلة أو لإتاحة المحتوى لضعاف
            #    السمع. أي فشل بالبناء لا يُسقِط الصفحة — فقط يُخفي الزر.
            try:
                from ai.video_engine import build_srt as _hf_build_srt
                _hf_srt_text = _hf_build_srt(script)
                st.download_button(
                    "📝 تحميل الترجمة (SRT)",
                    data=_hf_srt_text.encode("utf-8"),
                    file_name=f"{(script.title or 'documentary')[:40]}.srt",
                    mime="text/srt",
                    key="hf_download_srt",
                )
            except Exception as _srt_err:  # noqa: BLE001
                logger.debug("تعذّر بناء ملف SRT لـHiggsfield Explainer: %s", _srt_err)

        # ── مشاركة اجتماعية فعلية (رفع الفيديو) ─────────────────────
        st.markdown("---")
        st.markdown("#### 📤 مشاركة اجتماعية فعلية (رفع الفيديو)")
        try:
            from ai.social_platforms import YouTubeAdapter, TikTokAdapter
        except ImportError as e:  # noqa: BLE001
            st.caption(f"⚠️ تعذّر تحميل محولات المشاركة: {e}")
        else:
            yt = YouTubeAdapter()
            tk = TikTokAdapter()
            share_cols = st.columns(2)

            with share_cols[0]:
                st.markdown("**▶️ YouTube**")
                yt_ready = yt.is_configured() and yt._can_write()
                if not yt_ready:
                    missing = yt.missing_env() or yt.write_env
                    st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(missing))
                else:
                    yt_title = st.text_input(
                        "العنوان:", value=script.title[:100], key="hf_yt_upload_title"
                    )
                    yt_privacy = st.selectbox(
                        "الخصوصية:", ["private", "unlisted", "public"],
                        key="hf_yt_upload_privacy",
                    )
                    if st.button("▶️ ارفع على يوتيوب", key="hf_yt_upload_btn", use_container_width=True):
                        try:
                            with st.spinner("⏳ يرفع الفيديو على يوتيوب (Resumable Upload)..."):
                                video_id = yt.upload_video(
                                    mp4_bytes,
                                    title=yt_title,
                                    description=script.full_narration[:4500],
                                    privacy_status=yt_privacy,
                                )
                            st.success(f"✅ تم الرفع! الرابط: https://youtu.be/{video_id}")
                        except Exception as e:  # noqa: BLE001
                            st.error(f"⚠️ فشل الرفع على يوتيوب: {e}")

            with share_cols[1]:
                st.markdown("**🎵 TikTok**")
                tk_ready = tk.is_configured()
                if not tk_ready:
                    st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(tk.missing_env()))
                else:
                    st.caption(
                        "ℹ️ التطبيقات غير المدقَّقة من TikTok تنشر كـ«خاص بحسابك فقط» "
                        "(مسودة للمراجعة) حتى يجتاز التطبيق مراجعة TikTok الرسمية."
                    )
                    tk_title = st.text_input(
                        "العنوان:", value=script.title[:150], key="hf_tk_upload_title"
                    )
                    if st.button("🎵 ارفع على تيك توك", key="hf_tk_upload_btn", use_container_width=True):
                        try:
                            with st.spinner("⏳ يرفع الفيديو على تيك توك..."):
                                publish_id = tk.upload_video(mp4_bytes, title=tk_title)
                            st.success(
                                f"✅ تم إرسال الفيديو (publish_id: {publish_id}) — "
                                "افتح تطبيق TikTok للتأكد من ظهوره ضمن المسودات/المنشورات."
                            )
                        except Exception as e:  # noqa: BLE001
                            st.error(f"⚠️ فشل الرفع على تيك توك: {e}")



def render_training():
    """تبويب التدريب."""
    # ── 📊 إحصاءات النظام المعرفي — انتقلت هنا من الصفحة الرئيسية ──
    _roots       = load_arabic_roots()
    _ckg_overview = load_ckg()
    _quran_index = load_quran_index()
    _training_ov = load_training_summary()
    _checkpoint  = load_latest_checkpoint()
    _episodic    = get_episodic_stats()

    _concepts_count  = len(_ckg_overview.get("concepts", {}))
    _relations_count = len(_ckg_overview.get("relations", {}))
    _meaningful_roots = sum(1 for k in _roots if len(k) >= 3 and _roots[k].get("frequency", 0) >= 5)
    _train_steps_ov = _training_ov.get("train_steps", 0)

    # آخر تحديث — وقت مطلق + وقت نسبي ("منذ...") لملاحظة الحيوية بلمحة
    _saved_at = _checkpoint.get("saved_at", "")
    _last_update = "غير محدد"
    _last_update_relative = ""
    if _saved_at:
        try:
            _dt = datetime.fromisoformat(_saved_at.replace("Z", "+00:00"))
            _last_update = _dt.strftime("%Y-%m-%d %H:%M") + " UTC"
            _now = datetime.now(_dt.tzinfo) if _dt.tzinfo else datetime.utcnow()
            _delta_sec = max(0, (_now - _dt).total_seconds())
            if _delta_sec < 60:
                _last_update_relative = "منذ لحظات"
            elif _delta_sec < 3600:
                _last_update_relative = f"منذ {int(_delta_sec // 60)} دقيقة"
            elif _delta_sec < 86400:
                _last_update_relative = f"منذ {int(_delta_sec // 3600)} ساعة"
            else:
                _last_update_relative = f"منذ {int(_delta_sec // 86400)} يوم"
        except Exception:
            _last_update = _saved_at[:19]

    st.markdown(
        '<div class="section-header">📊 إحصاءات النظام المعرفي <span class="live-dot"></span></div>',
        unsafe_allow_html=True,
    )

    _last_label_ov = f"آخر تحديث · {_last_update_relative}" if _last_update_relative else "آخر تحديث"
    st.markdown(f"""
    <div class="bento-grid">
        <div class="metric-card bento-featured">
            <div class="metric-value" data-count-target="{_concepts_count}">{_concepts_count:,}</div>
            <div class="metric-label">مفهوم في CKG</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_relations_count}">{_relations_count:,}</div>
            <div class="metric-label">علاقة معرفية</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_meaningful_roots}">{_meaningful_roots:,}</div>
            <div class="metric-label">جذر عربي مكتشف</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_train_steps_ov}">{_train_steps_ov:,}</div>
            <div class="metric-label">خطوة تدريب</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_quran_index.get('total_ayat', 6236)}">{_quran_index.get('total_ayat', 6236):,}</div>
            <div class="metric-label">آية قرآنية محملة</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_quran_index.get('total_surahs', 114)}">{_quran_index.get('total_surahs', 114)}</div>
            <div class="metric-label">سورة كريمة</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_episodic.get('episodic', 0)}">{_episodic.get('episodic', 0):,}</div>
            <div class="metric-label">ذكرى تجريبية</div>
        </div>
        <div class="metric-card">
            <div class="metric-value metric-value--wrap">{_last_update}</div>
            <div class="metric-label">{_last_label_ov}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # عدّاد متحرك من 0 حتى القيمة الفعلية — نفس أسلوب حقن JS المضمون
    # (components.html بدل st.markdown الذي لا يُنفَّذ فيه <script> إطلاقاً)
    st.components.v1.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const counters = doc.querySelectorAll('.metric-value[data-count-target]');
        counters.forEach(function(el) {
            if (el.dataset.nsmAnimated) return;
            el.dataset.nsmAnimated = "1";
            const target = parseInt(el.getAttribute('data-count-target'), 10) || 0;
            const duration = 900;
            const start = performance.now();
            function tick(now) {
                const p = Math.min(1, (now - start) / duration);
                const eased = 1 - Math.pow(1 - p, 3);
                el.textContent = Math.round(eased * target).toLocaleString('en-US');
                if (p < 1) requestAnimationFrame(tick);
                else el.textContent = target.toLocaleString('en-US');
            }
            requestAnimationFrame(tick);
        });
    })();
    </script>
    """, height=0)

    st.markdown("")
    training   = load_training_summary()
    checkpoint = load_latest_checkpoint()
    ckg        = load_ckg()

    train_steps  = training.get("train_steps", 0)
    last_loss    = training.get("last_loss", 0.0)
    total_params = training.get("total_parameters", 0)
    ckg_size     = len(ckg.get("concepts", {}))
    _is_active   = train_steps > 0

    _hdr_col, _btn_col, _save_col = st.columns([4.2, 1, 1.5])
    with _hdr_col:
        _pill = (
            '<span class="status-pill status-pill--active"><span class="status-pill-dot"></span>نشط</span>'
            if _is_active else
            '<span class="status-pill status-pill--idle"><span class="status-pill-dot"></span>لم يبدأ بعد</span>'
        )
        st.markdown(f'<div class="section-header">🎓 حالة التدريب {_pill}</div>', unsafe_allow_html=True)
    with _btn_col:
        if st.button("🔄 تحديث", key="training_refresh_btn", use_container_width=True):
            load_training_summary.clear()
            load_latest_checkpoint.clear()
            load_ckg.clear()
            st.rerun()
    with _save_col:
        if st.button("💾 حفظ Checkpoint", key="save_checkpoint_btn", use_container_width=True,
                      disabled=not _CHECKPOINT_OK, help="يحفظ حالة CKG + الذاكرة الإيبيسودية الحقيقية الآن"):
            with st.spinner("💾 جارٍ حفظ الحالة الحقيقية..."):
                _saved_path = save_real_checkpoint()
            if _saved_path:
                st.toast(f"✅ تم الحفظ: {os.path.basename(_saved_path)}", icon="💾")
                load_latest_checkpoint.clear()
                st.rerun()
            else:
                st.toast("⚠️ تعذّر الحفظ — راجع السجلّات", icon="⚠️")

    if _GITHUB_SYNC_OK:
        _gh_status = _github_sync.status()
        if _gh_status.get("token_set"):
            if _gh_status.get("push_count", 0) > 0:
                if _gh_status.get("last_push_ok"):
                    st.caption(f"🔗 GitHub sync: آخر رفع ناجح ({_gh_status.get('last_push_ts', '')})")
                else:
                    st.caption(f"🔗 GitHub sync: آخر محاولة فشلت — {_gh_status.get('last_push_msg', '')}")
            else:
                st.caption("🔗 GitHub sync: جاهز (لم يُنفَّذ أي رفع بعد)")

    if not _is_active:
        st.markdown("""
        <div class="training-empty">
            <div class="training-empty-icon">🌱</div>
            <div class="training-empty-text">
                لم تُسجَّل أي خطوة تدريب بعد على هذه النسخة. بمجرد تشغيل دورة تدريب
                (<code>ai/knowledge_trainer.py</code> أو <code>ai/continual_learner.py</code>)
                ستظهر هنا خطوات التدريب، قيمة الخسارة، ونقاط الحفظ فور توفّرها.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(f"{train_steps:,}", "خطوات التدريب")
    with col2:
        _loss_display = f"{last_loss:.2e}" if last_loss else "—"
        metric_card(_loss_display, "آخر خسارة (Loss)")
    with col3:
        metric_card(f"{total_params:,}" if total_params else "—", "معامل في الشبكة")
    with col4:
        metric_card(f"{ckg_size:,}", "مفهوم في CKG")

    st.markdown("")

    # معلومات الـ Checkpoint
    saved_at = checkpoint.get("saved_at", "")
    if saved_at:
        st.markdown('<div class="section-header">💾 آخر نقطة حفظ</div>', unsafe_allow_html=True)
        try:
            dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
            st.info(f"تم الحفظ في: **{dt.strftime('%Y-%m-%d الساعة %H:%M:%S')} UTC**")
        except Exception:
            st.info(f"تم الحفظ في: {saved_at}")

        state = checkpoint.get("state", {})
        if state:
            st.markdown('<div class="section-header">🧠 محتوى نقطة الحفظ</div>', unsafe_allow_html=True)
            module_labels = {
                "neural_weights":  ("⚙️", "الأوزان العصبية"),
                "deep_network":    ("🧬", "الشبكة العميقة"),
                "dynamic_layer":   ("🔀", "الطبقة الديناميكية"),
                "episodic_memory": ("💭", "الذاكرة التجريبية"),
                "world_model":     ("🌍", "نموذج العالم"),
                "system_dna":      ("🧿", "الحمض النووي للنظام"),
                "self_awareness":  ("👁️", "الوعي الذاتي"),
                "knowledge_keys":  ("📚", "مفاهيم CKG"),
                "meta":            ("📋", "البيانات الوصفية"),
            }
            _chips_html = '<div class="module-chip-grid">'
            for module_name in state.keys():
                _icon, _label = module_labels.get(module_name, ("✅", module_name))
                _chips_html += (
                    f'<div class="module-chip"><span class="module-chip-dot"></span>'
                    f'<span>{_icon} {_label}</span></div>'
                )
            _chips_html += '</div>'
            st.markdown(_chips_html, unsafe_allow_html=True)

    # معلومات التدريب التفصيلية
    if training:
        st.markdown("")
        st.markdown('<div class="section-header">📐 بنية الشبكة العصبية</div>', unsafe_allow_html=True)
        arch = training.get("architecture", "")
        if arch:
            st.markdown('<div class="arch-card">', unsafe_allow_html=True)
            st.code(arch, language=None)
            st.markdown('</div>', unsafe_allow_html=True)

        avg_loss = training.get("avg_recent_loss", 0)
        lr       = training.get("learning_rate", 0)
        col_a, col_b = st.columns(2)
        with col_a:
            _avg_display = f"`{avg_loss:.2e}`" if avg_loss else "`—`"
            st.markdown(f"**متوسط الخسارة الأخيرة:** {_avg_display}")
        with col_b:
            st.markdown(f"**معدل التعلم:** `{lr}`" if lr else "**معدل التعلم:** `—`")

    # ── [NSM Router Bridge] تبويب التوجيه الذكي ──────────────────────────
    st.markdown("")
    render_nsm_routing()


def render_nsm_routing():
    """لوحة NSM Mesh — توجيه دلالي + self-healing + سجل حي + إثبات تعلم."""
    st.markdown(
        '<div class="section-header">🕸️ NSM Mesh — الشبكة الذكية الحية '
        '<span class="live-dot"></span></div>',
        unsafe_allow_html=True,
    )

    if not _NSM_BRIDGE_OK or not _nsm_bridge:
        st.warning("⚠️ NSM Router Bridge غير مُفعَّل.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # [A] درجات العقد الثلاث — أداء تاريخي فوري
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("#### 📡 درجات العقد الحية")
    node_scores = _nsm_bridge.get_node_scores_for_display()
    n_cols = st.columns(len(node_scores))
    for col, ns in zip(n_cols, node_scores):
        score = ns["connection_score"]
        sc = "var(--emerald)" if score >= 70 else ("var(--gold)" if score >= 45 else "var(--text-muted)")
        bar_w = int(score)
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:0.9rem">
                <div style="font-size:1.3rem;margin-bottom:0.25rem">{ns["label"]}</div>
                <div class="metric-value" style="color:{sc};font-size:2.2rem;line-height:1">
                    {score:.1f}
                </div>
                <div class="metric-label" style="margin-bottom:0.5rem">/ 100</div>
                <div style="background:var(--surface);border-radius:6px;height:6px;overflow:hidden;margin-bottom:0.6rem">
                    <div style="background:{sc};width:{bar_w}%;height:6px;border-radius:6px"></div>
                </div>
                <div style="font-size:0.78rem;color:var(--text-muted);direction:rtl;line-height:1.8">
                    🔁 <strong>{ns["total_runs"]}</strong> تشغيل &nbsp;
                    ✅ <strong>{ns["success_rate"]:.0f}%</strong> &nbsp;
                    ⏱️ <strong>{int(ns["avg_latency_ms"])}ms</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # [B] سجل التوجيه الحي — آخر القرارات في هذه الجلسة
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("")
    st.markdown("#### 🔴 سجل التوجيه الحي")

    _rlog = st.session_state.get("nsm_route_log", [])
    if not _rlog and _ROUTE_LOG_DB_OK:
        # استرجاع الذاكرة التراكمية من SQLite عند عدم وجود سجل في الجلسة الحالية
        _rlog = _rlog_get_recent(limit=100)
        if _rlog:
            st.session_state["nsm_route_log"] = _rlog
    if not _rlog:
        st.info("📋 سيظهر سجل التوجيه هنا فور إرسال أول رسالة في تبويب المحادثة.")
    else:
        # إحصاءات سريعة
        _total_req     = len(_rlog)
        _failover_reqs = sum(1 for r in _rlog if r.get("failover"))
        _ok_reqs       = sum(1 for r in _rlog if r.get("success"))
        _avg_lat       = sum(r.get("latency_ms", 0) for r in _rlog) / max(_total_req, 1)

        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        with _sc1: metric_card(_total_req, "طلبات في الجلسة")
        with _sc2: metric_card(f"{_ok_reqs/_total_req*100:.0f}%", "معدل النجاح")
        with _sc3: metric_card(_failover_reqs, "إعادة توجيه تلقائي")
        with _sc4: metric_card(f"{_avg_lat:.0f}ms", "متوسط الاستجابة")

        st.markdown("")

        # توزيع الفئات الدلالية
        if _NSM_SEMANTIC_OK and _nsm_semantic:
            _cat_counts: dict = {}
            for r in _rlog:
                _cat = r.get("category", "general")
                _cat_counts[_cat] = _cat_counts.get(_cat, 0) + 1
            if _cat_counts:
                _cat_html = '<div style="direction:rtl;margin-bottom:0.8rem">'
                _cat_html += '<span style="font-size:0.8rem;color:var(--text-muted)">توزيع الاستعلامات: </span>'
                for _cat, _cnt in sorted(_cat_counts.items(), key=lambda x: -x[1]):
                    _lbl = _nsm_semantic.CATEGORY_LABELS.get(_cat, ("💬", _cat))
                    _cat_html += (
                        f'<span class="badge badge-blue" style="margin:2px">'
                        f'{_lbl[0]} {_lbl[1]}: {_cnt}</span>'
                    )
                _cat_html += '</div>'
                st.markdown(_cat_html, unsafe_allow_html=True)

        # توزيع العقد المختارة
        _node_counts: dict = {}
        for r in _rlog:
            _nd = r.get("node", "?").replace("nsm:", "")
            _node_counts[_nd] = _node_counts.get(_nd, 0) + 1
        _node_html = '<div style="direction:rtl;margin-bottom:1rem">'
        _node_html += '<span style="font-size:0.8rem;color:var(--text-muted)">العقد المختارة: </span>'
        for _nd, _cnt in sorted(_node_counts.items(), key=lambda x: -x[1]):
            _node_html += f'<span class="badge badge-amber" style="margin:2px">{_nd}: {_cnt}</span>'
        _node_html += '</div>'
        st.markdown(_node_html, unsafe_allow_html=True)

        # جدول آخر 20 قراراً
        _last20 = list(reversed(_rlog[-20:]))
        _rows_html = ""
        for r in _last20:
            _ico    = r.get("cat_icon", "💬")
            _cat    = r.get("category", "general")
            _nd     = r.get("node", "?").replace("nsm:", "")
            _lat    = r.get("latency_ms", 0)
            _ok     = r.get("success", False)
            _fo     = r.get("failover", False)
            _ok_ico = "✅" if _ok else "❌"
            _fo_badge = '<span class="badge badge-purple" style="font-size:0.65rem">↩️ failover</span>' if _fo else ""
            _q      = r.get("query", "")
            _ts     = r.get("ts", "")
            _qs     = r.get("quality_score")
            _qs_badge = ""
            if _qs is not None:
                _qs_cls = "badge-green" if _qs >= 70 else ("badge-amber" if _qs >= 40 else "badge-purple")
                _qs_badge = f'<span class="badge {_qs_cls}" style="font-size:0.65rem">⭐ {_qs:.0f}</span>'
            _rows_html += f"""
            <div class="root-item" style="direction:rtl;display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;padding:0.4rem 0.7rem">
                <span style="color:var(--text-muted);font-size:0.72rem;min-width:60px">{_ts}</span>
                <span>{_ico}</span>
                <span class="badge badge-blue">{_cat}</span>
                <span class="badge badge-amber">{_nd}</span>
                <span style="font-size:0.8rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_q}</span>
                <span style="min-width:55px;font-size:0.78rem;color:var(--text-muted)">{_lat}ms</span>
                <span>{_ok_ico}</span>
                {_qs_badge}
                {_fo_badge}
            </div>"""
        st.markdown(
            f'<div style="max-height:380px;overflow-y:auto;border:1px solid var(--border);'
            f'border-radius:12px;padding:0.3rem">{_rows_html}</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑 مسح سجل التوجيه", key="clear_route_log"):
            st.session_state["nsm_route_log"] = []
            if _ROUTE_LOG_DB_OK:
                _rlog_clear_all()
            st.rerun()

        # ────────────────────────────────────────────────────────────────
        # [B.1] رؤى Meta-Reasoner — تحليل تأملي حقيقي فوق سجل التوجيه
        # ────────────────────────────────────────────────────────────────
        if _META_REASONER_OK and _ROUTE_LOG_DB_OK:
            with st.expander("🧠 رؤى Meta-Reasoner (تحليل تأملي لسجل التوجيه)"):
                _reasoner = _get_meta_reasoner()
                if _reasoner is None:
                    st.caption("⚠️ تعذّر تهيئة MetaReasoner.")
                else:
                    _insights = _reasoner.reflect()
                    if not _insights:
                        st.caption("لا توجد أنماط تستحق التنبيه بعد — يحتاج المزيد من طلبات التوجيه المسجَّلة.")
                    else:
                        _badge_by_type = {
                            "warning": "badge-purple", "opportunity": "badge-amber",
                            "pattern": "badge-blue", "lesson": "badge-green",
                        }
                        for _ins in _insights[:8]:
                            _cls = _badge_by_type.get(_ins.insight_type, "badge-blue")
                            st.markdown(f"""
                            <div class="root-item" style="direction:rtl;padding:0.6rem 0.8rem;margin-bottom:0.4rem">
                                <span class="badge {_cls}">{_ins.insight_type}</span>
                                <strong style="margin-right:0.4rem">{_ins.title}</strong>
                                <div style="font-size:0.82rem;color:var(--text-muted);margin-top:0.3rem">{_ins.body}</div>
                            </div>
                            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # [C] إثبات التعلم — prove_learning + learning_curve
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("")
    st.markdown("#### 🎓 إثبات التعلم التراكمي")

    report = _nsm_bridge.get_learning_report()
    if "error" in report:
        st.warning(f"⚠️ {report['error']}")
    else:
        proof   = report.get("proof", {})
        verdict = proof.get("verdict", "insufficient_data")
        _vmap   = {
            "learning_confirmed":       ("✅", "var(--emerald)", "تعلّم مؤكَّد"),
            "learning_in_progress":     ("🔄", "var(--gold)",   "التعلم قيد التقدم"),
            "learning_not_yet_evident": ("⏳", "var(--text-muted)", "بيانات غير كافية بعد"),
            "insufficient_data":        ("⏳", "var(--text-muted)", "بيانات غير كافية بعد"),
        }
        _ico, _vc, _vl = _vmap.get(verdict, ("❓", "var(--text-muted)", verdict))
        st.markdown(
            f'<div class="metric-card" style="border-right:4px solid {_vc};direction:rtl;padding:0.9rem 1.1rem">'
            f'<span style="font-size:1.3rem">{_ico}</span>'
            f' <strong style="color:{_vc}">{_vl}</strong>'
            f'<p style="margin:0.4rem 0 0;color:var(--text-muted);font-size:0.85rem">'
            f'{proof.get("message","")}</p></div>',
            unsafe_allow_html=True,
        )

        _evs = proof.get("evidence", [])
        if _evs:
            st.markdown("")
            for ev in _evs:
                st.markdown(f"- {ev}")

        _metrics = proof.get("metrics", {})
        if _metrics.get("total_executions", 0) > 0:
            st.markdown("")
            _lm1, _lm2, _lm3, _lm4 = st.columns(4)
            with _lm1: metric_card(_metrics.get("total_executions", 0), "تشغيلات تراكمية")
            with _lm2: metric_card(f"{_metrics.get('success_rate',0)*100:.1f}%", "معدل النجاح")
            with _lm3: metric_card(f"{_metrics.get('learning_improvement_pct',0):+.1f}%", "تحسّن vs الأساس")
            with _lm4:
                _tn = proof.get("top_nodes", [])
                metric_card(_tn[0].get("name","—") if _tn else "—", "أكثر العقد ثقةً")

        # منحنى التعلم
        _curve  = report.get("curve", {})
        _pts    = _curve.get("data_points", [])
        _trend  = _curve.get("trend", "insufficient_data")
        _tmap   = {"improving": "📈 تحسّن", "degrading": "📉 تراجع",
                   "stable": "➡️ مستقر", "insufficient_data": "⏳ غير كافٍ"}
        if _pts:
            st.markdown("")
            st.caption(f"منحنى التعلم — الاتجاه: {_tmap.get(_trend, _trend)}")
            try:
                import pandas as _pd
                _df = _pd.DataFrame(_pts)
                if "avg_connection_score" in _df.columns:
                    st.line_chart(_df.set_index("index")["avg_connection_score"],
                                  use_container_width=True)
            except Exception:
                for _dp in _pts[-8:]:
                    st.text(f"[{_dp.get('index','')}] {_dp.get('avg_connection_score','?'):.1f}")
        else:
            st.caption("📊 منحنى التعلم سيظهر بعد تراكم بيانات كافية.")

        # سمعة العقد
        _rep = report.get("reputation", [])
        if _rep:
            st.markdown("")
            st.markdown("#### 🏅 سمعة العقد")
            _tier_c = {"platinum":"var(--emerald)","gold":"var(--gold)",
                       "silver":"#aaa","bronze":"#cd7f32","unrated":"var(--text-muted)"}
            for _nd in _rep:
                _t  = _nd.get("tier","unrated")
                _tc = _tier_c.get(_t,"var(--text-muted)")
                st.markdown(
                    f'<div class="root-item" style="direction:rtl">'
                    f'<strong>{_nd.get("name",_nd.get("node_id","?"))}</strong>'
                    f' <span class="badge badge-blue" style="background:{_tc};color:#fff">{_t}</span>'
                    f' <span class="badge badge-amber">سمعة: {_nd.get("reputation_score",0):.1f}</span>'
                    f' <span class="badge badge-blue">نجاح: {_nd.get("success_rate",0)*100:.0f}%</span>'
                    f' <span style="color:var(--text-muted);font-size:0.78rem">'
                    f'({_nd.get("total_runs",0)} تشغيل)</span></div>',
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════════════════════
    # [D] معلومات التوجيه الدلالي
    # ════════════════════════════════════════════════════════════════════════
    if _NSM_SEMANTIC_OK and _nsm_semantic:
        st.markdown("")
        with st.expander("🧠 كيف يعمل التوجيه الدلالي؟", expanded=False):
            st.markdown("""
**الصيغة:** `درجة_مركَّبة = 65% × ScoringEngine_التاريخي + 35% × تحيُّز_دلالي`

| الفئة | العقدة المُفضَّلة | السبب |
|---|---|---|
| 🕌 عربي/إسلامي | NSM Agent | مُدرَّب ومُخصَّص للعربية والمعرفة الإسلامية |
| 💻 برمجة | OpenRouter (GPT-4o/Claude) | نماذج الكود الأقوى |
| ✍️ إبداعي | OpenRouter | إبداع أغنى مع نماذج كبيرة |
| 🔍 تحليل | OpenRouter | تحليل أعمق مع سياق أوسع |
| 💬 عام | NSM Agent | الافتراضي المُحسَّن للعربية |

⚡ **Failover:** إذا فشل المسار الأول — يُعاد التوجيه تلقائياً للتالي مع تسجيل الفشل.
            """)

    st.caption("🔁 كل رسالة في المحادثة تُحدِّث هذه اللوحة تلقائياً.")


def render_memory():
    """تبويب الذاكرة."""
    st.markdown('<div class="section-header">🧠 حالة الذاكرة</div>', unsafe_allow_html=True)

    episodic = get_episodic_stats()
    ckg      = load_ckg()
    roots    = load_arabic_roots()

    concepts_count  = len(ckg.get("concepts", {}))
    relations_count = len(ckg.get("relations", {}))

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(episodic.get("episodic", 0), "ذاكرة تجريبية")
    with col2: metric_card(concepts_count, "ذاكرة دلالية (مفاهيم)")
    with col3: metric_card(relations_count, "علاقات مستنتجة")
    with col4: metric_card(len(roots), "جذر عربي مفهرس")

    st.markdown("")
    st.markdown('<div class="section-header">📁 تفاصيل الذاكرة الدلالية (CKG)</div>', unsafe_allow_html=True)

    concepts_db = ckg.get("concepts", {})
    if concepts_db:
        # عرض أقوى المفاهيم
        sorted_concepts = sorted(
            concepts_db.items(),
            key=lambda x: x[1].get("frequency", 0),
            reverse=True
        )[:15]

        for cname, cdata in sorted_concepts:
            freq     = cdata.get("frequency", 0)
            cluster  = cdata.get("cluster", "غير مصنّف")
            strength = cdata.get("strength", 0.0)
            sources  = cdata.get("sources", [])
            st.markdown(f"""
            <div class="root-item">
                <strong>{cname}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{cluster}</span>
                <span class="badge badge-blue">تكرار: {freq}</span>
                <span class="badge badge-amber">قوة: {strength:.2f}</span>
                <br><small style="color:var(--text-muted)">المصادر: {', '.join(sources[:3]) if sources else 'غير محددة'}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("الذاكرة الدلالية (CKG) فارغة حالياً. قم بتشغيل دورة تدريب في Colab لملئها.")

    # ── أنواع العلاقات في CKG ────────────────────────────────────────────
    relations_db = ckg.get("relations", {})
    if relations_db:
        st.markdown("")
        st.markdown('<div class="section-header">🔗 أنواع العلاقات في الذاكرة الدلالية</div>', unsafe_allow_html=True)

        rel_type_counter = Counter(r.get("relation_type", "غير محدد") for r in relations_db.values())
        type_labels = {
            "co_occurrence":    "تزامن في الآية",
            "semantic":         "علاقة دلالية (نفس المجموعة)",
            "thematic_cluster": "تجمّع موضوعي (تشارك سور)",
            "root_link":        "ربط بجذر عربي",
            "narrative_sequence": "تسلسل سردي (قصص الأنبياء)",
            "episodic_rule":    "قاعدة من الذاكرة التجريبية",
        }
        badges = " ".join(
            f'<span class="badge badge-blue" style="margin:3px">{type_labels.get(t, t)}: {n}</span>'
            for t, n in rel_type_counter.most_common()
        )
        st.markdown(badges, unsafe_allow_html=True)

    # ── ملامح السور (Surah Thematic Profiles) ───────────────────────────
    surah_profiles = ckg.get("surah_profiles", {})
    if surah_profiles:
        st.markdown("")
        st.markdown('<div class="section-header">📖 ملامح السور الموضوعية</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:var(--text-muted)">تم بناء ملامح موضوعية لـ {len(surah_profiles)} سورة '
            f'بناءً على المفاهيم الأكثر ظهوراً في كل سورة.</p>',
            unsafe_allow_html=True,
        )

        surah_options = sorted(surah_profiles.keys(), key=lambda x: int(x))
        chosen_surah = st.selectbox(
            "اختر سورة لعرض ملامحها:",
            options=surah_options,
            format_func=lambda s: f"سورة {s}",
            key="surah_profile_select",
        )
        if chosen_surah:
            profile = surah_profiles.get(chosen_surah, [])
            badges = " ".join(
                f'<span class="badge badge-purple" style="margin:3px">{p["concept"]} ({p["weight"]})</span>'
                for p in profile
            )
            st.markdown(badges, unsafe_allow_html=True)

    # حالة قاعدة البيانات
    st.markdown("")
    st.markdown('<div class="section-header">💾 حالة قواعد البيانات</div>', unsafe_allow_html=True)
    db_path = MEMORY_DIR / "episodic.db"
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        st.markdown(f'<span class="health-ok">✅ قاعدة الذاكرة التجريبية: متصلة ({size_kb:.1f} KB)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="health-err">❌ قاعدة الذاكرة التجريبية: غير موجودة</span>', unsafe_allow_html=True)

    # ── إحصاءات الذاكرة التجريبية للأسئلة والأجوبة ──────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">📊 إحصاءات ذاكرة الأسئلة والأجوبة</div>', unsafe_allow_html=True)

    try:
        qa_stats = get_memory_stats(db_path)
    except Exception:
        qa_stats = {"total_episodes": 0, "common_concepts": [], "recent_episodes": [], "avg_confidence": 0.0}

    qcol1, qcol2 = st.columns(2)
    with qcol1: metric_card(qa_stats["total_episodes"], "إجمالي الحلقات المخزّنة")
    with qcol2: metric_card(f"{qa_stats['avg_confidence']:.0%}", "متوسط درجة الثقة")

    if qa_stats["total_episodes"] > 0:
        # أكثر المفاهيم تكراراً في الأسئلة
        st.markdown("**أكثر المفاهيم ظهوراً في الأسئلة:**")
        if qa_stats["common_concepts"]:
            badges = " ".join(
                f'<span class="badge badge-blue" style="margin:2px">{c} ({n})</span>'
                for c, n in qa_stats["common_concepts"][:8]
            )
            st.markdown(badges, unsafe_allow_html=True)

        # أحدث الحلقات
        st.markdown("")
        st.markdown("**أحدث الأسئلة:**")
        for ep in qa_stats["recent_episodes"][:5]:
            ts = ep.get("timestamp", "")[:19].replace("T", " ")
            st.markdown(f"""
            <div class="root-item">
                <strong>{ep['question']}</strong>
                <span class="badge badge-amber">ثقة: {ep['confidence']:.0%}</span>
                <br><small style="color:var(--text-muted)">{ts} UTC</small>
            </div>
            """, unsafe_allow_html=True)

        # ── التوحيد (Consolidation) ──
        st.markdown("")
        st.markdown('<div class="section-header">🧬 توحيد الذاكرة (Consolidation)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">يستخرج هذا الإجراء أزواج المفاهيم المتكررة في الأسئلة السابقة، '
            'ويولّد منها قواعد دلالية، ويضيفها كعلاقات جديدة في الذاكرة الدلالية (CKG) '
            'دون حذف أو تعديل أي علاقة موجودة.</p>',
            unsafe_allow_html=True,
        )

        if st.button("🧬 تشغيل التوحيد الآن", key="consolidate_btn"):
            ckg_path = KNOWLEDGE_DIR / "cognitive_graph.json"
            with st.spinner("يتم تحليل الحلقات واستخراج القواعد الدلالية..."):
                ckg_full = load_json(ckg_path) or {"concepts": {}, "relations": {}}
                cons_result = consolidate_memory(db_path, ckg_full, ckg_path, min_co_occurrence=2)
            st.success(
                f"تم التحليل: {cons_result['pairs_analyzed']} زوج مفاهيم، "
                f"{cons_result['new_rules']} قاعدة جديدة، "
                f"{cons_result['new_relations']} علاقة جديدة في CKG."
            )
            load_json.clear()
            load_ckg.clear()

        rules = get_semantic_rules(db_path, limit=10)
        if rules:
            st.markdown("**القواعد الدلالية المستخرجة:**")
            for r in rules:
                st.markdown(f"""
                <div class="root-item">
                    {r['rule_text']}
                    <span class="badge badge-purple">ثقة: {r['confidence']:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("لا توجد أسئلة محفوظة بعد. استخدم تبويب «الأسئلة والأجوبة» لبدء بناء الذاكرة التجريبية.")

    # ── القوانين المكتسبة تلقائياً (MemoryConsolidator) ──────────────────
    # طبقة منفصلة عن قسم "توحيد الذاكرة" أعلاه: تلك يدوية وتُنتج علاقات CKG
    # من أسئلة المستخدم، بينما هذه تعمل تلقائياً بالخلفية كل 15 دقيقة فوق
    # EpisodicMemoryEngine الحقيقية (get_strongest_memories) وتحوّل الأنماط
    # المتكررة (مصدر الحلقة، نطاق قيمة الهدف الرقمي) إلى "قوانين مكتسبة".
    if _CONSOLIDATOR_OK and _EPISODIC_OK:
        st.markdown("")
        st.markdown('<div class="section-header">⚖️ قوانين الذاكرة المكتسبة تلقائياً</div>', unsafe_allow_html=True)
        _consolidator = _get_memory_consolidator()
        if _consolidator is None:
            st.caption("⚠️ تعذّر تشغيل MemoryConsolidator.")
        else:
            _mc_summary = _consolidator.summary()
            _mcol1, _mcol2, _mcol3 = st.columns(3)
            with _mcol1: metric_card(_mc_summary["total_laws"], "قوانين مكتسبة")
            with _mcol2: metric_card(_mc_summary["total_episodes_freed"], "حلقات مُحرَّرة")
            with _mcol3: metric_card(_mc_summary["local_patterns_tracked"], "أنماط قيد الرصد")

            if st.button("⚖️ تشغيل دورة دمج الآن", key="consolidate_laws_btn"):
                with st.spinner("يفحص الذاكرة الإيبيسودية عن أنماط متكررة..."):
                    _mc_report = _consolidator.consolidate()
                st.success(
                    f"فُحصت {_mc_report['episodes_scanned']} حلقة، "
                    f"{_mc_report['new_laws']} قانون جديد، "
                    f"{_mc_report['updated_laws']} قانون محدَّث."
                )

            _laws = _consolidator.get_consolidated_laws()
            if _laws:
                st.markdown("**القوانين المكتسبة (مرتبة بالثقة):**")
                for _law in _laws[:8]:
                    st.markdown(f"""
                    <div class="root-item">
                        {_law['description']}
                        <span class="badge badge-green">ثقة: {_law['confidence']:.0%}</span>
                        <span class="badge badge-blue">×{_law['occurrence_count']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption(f"لا توجد قوانين بعد — يحتاج نمط للتكرار {_consolidator._threshold} مرات على الأقل.")

    # ── سجل المحادثات المحفوظة (nsm_memory.py — SQLite) ──────────────────
    st.markdown("")
    st.markdown('<div class="section-header">📜 سجل المحادثات المحفوظة</div>', unsafe_allow_html=True)
    try:
        from nsm_memory import _LongTermStore as _NSMLongTermStore
        _mem_store = _NSMLongTermStore()
        _all_sessions = _mem_store.list_sessions(limit=100)
    except Exception as _mem_err:
        _mem_store = None
        _all_sessions = []
        st.caption(f"⚠️ تعذّر تحميل سجل المحادثات: {_mem_err}")

    if _mem_store is not None:
        if not _all_sessions:
            st.info("لا توجد محادثات محفوظة بعد. ابدأ محادثة في تبويب «💬 المحادثة».")
        else:
            _sess_labels = {
                s["session_id"]: f"{s['session_id']} · {s['turns']} رسالة · "
                                 f"{datetime.fromtimestamp(s['last_ts']).strftime('%Y-%m-%d %H:%M') if s.get('last_ts') else ''}"
                for s in _all_sessions
            }
            _mem_col1, _mem_col2 = st.columns([2, 1])
            with _mem_col1:
                _chosen_session = st.selectbox(
                    "اختر جلسة لاستعراض محادثاتها",
                    options=list(_sess_labels.keys()),
                    format_func=lambda k: _sess_labels.get(k, k),
                    key="mem_browse_session",
                )
            with _mem_col2:
                _mem_search = st.text_input(
                    "🔎 ابحث داخل هذه الجلسة", key="mem_browse_search", placeholder="كلمة مفتاحية..."
                )

            _turns = _mem_store.list_recent_turns(limit=200, session_id=_chosen_session)
            if _mem_search.strip():
                _needle = _mem_search.strip().lower()
                _turns = [t for t in _turns if _needle in t["user"].lower() or _needle in t["bot"].lower()]

            st.caption(f"عدد الأدوار المعروضة: {len(_turns)}")
            for _t in _turns[:50]:
                _ts_str = datetime.fromtimestamp(_t["ts"]).strftime("%Y-%m-%d %H:%M") if _t.get("ts") else ""
                st.markdown(f"""
                <div class="root-item">
                    <span class="badge badge-blue">👤 {_t['user'][:200]}</span><br>
                    <span class="badge badge-purple" style="margin-top:4px">🧠 {_t['bot'][:300]}</span>
                    <br><small style="color:var(--text-muted)">{_ts_str} · {_t.get('topic') or 'بدون موضوع'}</small>
                </div>
                """, unsafe_allow_html=True)


def render_health():
    """تبويب صحة النظام."""
    st.markdown('<div class="section-header">🏥 صحة النظام</div>', unsafe_allow_html=True)

    checks = []

    # ── 1. الأوزان محفوظة؟
    weights_path = CHECKPOINTS_DIR / "neural_weights.npy"
    if weights_path.exists():
        size_kb = weights_path.stat().st_size / 1024
        checks.append(("✅", "الأوزان العصبية", f"محفوظة ({size_kb:.1f} KB)", True))
    else:
        checks.append(("❌", "الأوزان العصبية", "ملف الأوزان غير موجود", False))

    # ── 2. CKG محفوظ؟
    ckg_path = KNOWLEDGE_DIR / "cognitive_graph.json"
    if ckg_path.exists() and ckg_path.stat().st_size > 10:
        ckg = load_ckg()
        n_concepts = len(ckg.get("concepts", {}))
        checks.append(("✅", "قاعدة المعرفة CKG", f"موجودة ({n_concepts} مفهوم)", True))
    else:
        checks.append(("⚠️", "قاعدة المعرفة CKG", "فارغة أو غير موجودة", False))

    # ── 3. قاعدة البيانات
    db_path = MEMORY_DIR / "episodic.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            conn.close()
            checks.append(("✅", "قاعدة الذاكرة (SQLite)", f"متصلة ({count} سجل)", True))
        except Exception as e:
            checks.append(("❌", "قاعدة الذاكرة (SQLite)", f"خطأ: {e}", False))
    else:
        checks.append(("❌", "قاعدة الذاكرة (SQLite)", "غير موجودة", False))

    # ── 4. القرآن الكريم
    chunks = list(KNOWLEDGE_DIR.glob("quran_chunk_*.json"))
    if len(chunks) >= 60:
        checks.append(("✅", "بيانات القرآن الكريم", f"{len(chunks)} chunk محمّل (6,236 آية)", True))
    else:
        checks.append(("⚠️", "بيانات القرآن الكريم", f"وُجد {len(chunks)} chunk فقط", False))

    # ── 5. الجذور العربية
    roots = load_arabic_roots()
    if len(roots) > 100:
        checks.append(("✅", "فهرس الجذور العربية", f"{len(roots)} جذر مكتشف", True))
    else:
        checks.append(("⚠️", "فهرس الجذور العربية", f"{len(roots)} جذر فقط", False))

    # ── 6. نقطة حفظ حديثة
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("brain_checkpoint_*.json"), reverse=True)
    if checkpoint_files:
        latest = load_latest_checkpoint()
        saved_at = latest.get("saved_at", "")
        checks.append(("✅", "نقطة الحفظ الأخيرة (Checkpoint)", saved_at[:19] if saved_at else "موجودة", True))
    else:
        checks.append(("❌", "نقطة الحفظ الأخيرة (Checkpoint)", "لا توجد نقطة حفظ", False))

    # ── 7. التدريب
    training = load_training_summary()
    if training.get("train_steps", 0) > 0:
        checks.append(("✅", "حالة التدريب", f"{training['train_steps']:,} خطوة مكتملة", True))
    else:
        checks.append(("⚠️", "حالة التدريب", "لم يكتمل تدريب بعد", False))

    # ── 8. مزوّد LLM الحالي ─────────────────────────────────────────────
    try:
        from ai.llm_fallback import LLMFallback
        _fb = LLMFallback()
        fb_info = _fb.info()
        _prov   = fb_info.get("provider", "غير محدد")
        _model  = fb_info.get("model", "غير محدد")
        _live   = fb_info.get("live_llm", "❌")
        checks.append(("✅" if "✅" in _live else "⚠️", f"مزوّد LLM — {_prov}", _model, "✅" in _live))
    except Exception as _e:
        checks.append(("⚠️", "مزوّد LLM", str(_e)[:60], False))

    # عرض النتائج
    all_ok = sum(1 for c in checks if c[3])
    total  = len(checks)

    if all_ok == total:
        st.success(f"✅ النظام يعمل بكفاءة كاملة ({all_ok}/{total})")
    elif all_ok >= total * 0.7:
        st.warning(f"⚠️ النظام يعمل جزئياً ({all_ok}/{total})")
    else:
        st.error(f"❌ بعض مكونات النظام تحتاج انتباهاً ({all_ok}/{total})")

    st.markdown("")
    for icon, name, detail, ok in checks:
        st.markdown(f"""
        <div style="padding: 0.6rem 1rem; margin: 0.3rem 0; background: {'#f0fdf4' if ok else '#fef2f2'};
                    border-radius: 8px; border: 1px solid {'#bbf7d0' if ok else '#fecaca'};">
            <span style="font-size:1.2rem">{icon}</span>
            &nbsp;<strong>{name}</strong>
            &nbsp;&nbsp;<small style="color:var(--text-muted)">{detail}</small>
        </div>
        """, unsafe_allow_html=True)

    # ── نماذج Anthropic المتاحة (من That.md) ────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">🤖 نماذج Anthropic المتاحة</div>', unsafe_allow_html=True)
    try:
        from ai.llm_fallback import ANTHROPIC_MODELS
        model_rows = {
            "sonnet":  ("claude-sonnet-4-6",          "⚡ Sonnet 4",  "الافتراضي — توازن مثالي بين الجودة والسرعة"),
            "opus":    ("claude-opus-4-8",             "💎 Opus 4",    "المهام المعقدة — الأعلى جودةً"),
            "haiku":   ("claude-haiku-4-5-20251001",   "🚀 Haiku 4",   "الردود الفورية — الأخف والأسرع"),
            "stable":  ("claude-sonnet-4-20250514",    "🔒 Sonnet Stable", "الإصدار المستقر للإنتاج"),
        }
        cols = st.columns(len(model_rows))
        for col, (key, (model_id, label, desc)) in zip(cols, model_rows.items()):
            with col:
                is_active = ANTHROPIC_MODELS.get(key) == model_id
                border_color = "var(--gold)" if is_active else "var(--text-muted)"
                st.markdown(f"""
                <div style="background:var(--surface2);border:2px solid {border_color};border-radius:10px;
                            padding:0.8rem;text-align:center;direction:ltr;color:var(--text)">
                    <div style="font-size:1.3rem">{label}</div>
                    <code style="font-size:0.72rem;color:var(--gold)">{model_id}</code>
                    <div style="font-size:0.78rem;color:var(--text-muted);margin-top:0.4rem;direction:rtl">{desc}</div>
                </div>
                """, unsafe_allow_html=True)
        st.caption("المصدر: Claude.ai System Prompt (That.md) — محدَّث 2026")
    except Exception as _me:
        st.info(f"تعذّر تحميل قائمة النماذج: {_me}")

    # ── GitHub Push ───────────────────────────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">🚀 رفع إلى GitHub</div>', unsafe_allow_html=True)

    _gh_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "").strip()
    if not _gh_token:
        st.warning("🔑 أضف **GITHUB_PERSONAL_ACCESS_TOKEN** في Secrets لتفعيل هذه الميزة.")
    else:
        col_gh1, col_gh2 = st.columns([3, 1])
        with col_gh1:
            commit_msg = st.text_input(
                "رسالة الـ Commit",
                value="NSM update — رفع من الواجهة",
                key="gh_commit_msg",
                label_visibility="visible",
            )
        with col_gh2:
            st.markdown("<br>", unsafe_allow_html=True)
            push_btn = st.button("⬆️ Push", key="gh_push_btn", use_container_width=True, type="primary")

        if push_btn:
            if not commit_msg.strip():
                st.warning("أدخل رسالة commit أولاً.")
            else:
                import subprocess as _sp
                with st.spinner("⟳ جارٍ الرفع إلى GitHub..."):
                    try:
                        # git add
                        r_add = _sp.run(
                            ["git", "add", "-A"],
                            cwd=str(BASE), capture_output=True, text=True, timeout=15
                        )
                        if r_add.returncode != 0:
                            st.error(f"❌ فشل git add:\n{r_add.stderr[:400] or r_add.stdout[:400]}")
                            raise RuntimeError("git add failed")
                        # git commit
                        r_commit = _sp.run(
                            ["git", "-c", "user.email=nsm@replit.com",
                             "-c", "user.name=NSM Agent",
                             "commit", "-m", commit_msg.strip()],
                            cwd=str(BASE), capture_output=True, text=True, timeout=15,
                            env={**os.environ,
                                 "GIT_AUTHOR_NAME": "NSM Agent",
                                 "GIT_AUTHOR_EMAIL": "nsm@replit.com",
                                 "GIT_COMMITTER_NAME": "NSM Agent",
                                 "GIT_COMMITTER_EMAIL": "nsm@replit.com"},
                        )
                        # إذا لا يوجد تغيير جديد، نكمل الـ push للـ commit الحالي
                        nothing_to_commit = (
                            r_commit.returncode != 0 and
                            "nothing to commit" in (r_commit.stdout + r_commit.stderr)
                        )
                        if r_commit.returncode != 0 and not nothing_to_commit:
                            st.error(f"❌ فشل Commit:\n{r_commit.stderr[:400] or r_commit.stdout[:400]}")
                        else:
                            # git push
                            _remote = (
                                f"https://aliahmed369000000-ai:{_gh_token}"
                                "@github.com/aliahmed369000000-ai/Neural-Service-Mesh.git"
                            )
                            r_push = _sp.run(
                                ["git", "push", _remote, "main"],
                                cwd=str(BASE), capture_output=True, text=True, timeout=30
                            )
                            if r_push.returncode == 0:
                                st.success("✅ تم الرفع إلى GitHub بنجاح!")
                                # عرض معلومات الـ commit الأخير
                                r_log = _sp.run(
                                    ["git", "log", "--oneline", "-1"],
                                    cwd=str(BASE), capture_output=True, text=True
                                )
                                st.code(r_log.stdout.strip(), language="text")
                            else:
                                st.error(f"❌ فشل Push:\n{r_push.stderr[:400] or r_push.stdout[:400]}")
                    except Exception as _gh_err:
                        st.error(f"❌ خطأ غير متوقع: {_gh_err}")

        # عرض آخر commit
        try:
            import subprocess as _sp2
            _log = _sp2.run(
                ["git", "log", "--oneline", "-3"],
                cwd=str(BASE), capture_output=True, text=True, timeout=5
            )
            if _log.stdout.strip():
                with st.expander("📋 آخر 3 commits"):
                    st.code(_log.stdout.strip(), language="text")
        except Exception:
            pass

    # أزرار الإجراءات
    st.markdown("")
    st.markdown('<div class="section-header">⚙️ إجراءات</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 تحديث الإحصاءات", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_r2:
        st.markdown("""
        <div style="background:color-mix(in srgb, #38bdf8 12%, var(--surface2)); border:1px solid color-mix(in srgb, #38bdf8 40%, var(--border)); border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; direction:rtl; color:var(--text)">
            لتشغيل دورة تدريب، افتح Google Colab وشغّل <code>train_simulate.py</code>
        </div>
        """, unsafe_allow_html=True)

    # ── رقابة/تدقيق تفاعلات الوكلاء (Observability) ──
    # سجل مستقل تماماً عن CKG (القرآن) — يتتبّع فقط استدعاءات وكلاء AI
    # (ai/agent_categories.py) من "hub" أو "orchestrator" لأغراض التشخيص.
    st.markdown("")
    st.markdown('<div class="section-header">🔎 رقابة وكلاء AI (Observability)</div>', unsafe_allow_html=True)
    try:
        from ai.agent_audit import get_default_audit_log
        _audit = get_default_audit_log()
        _summary = _audit.summary()
    except Exception as _audit_err:
        _audit = None
        _summary = None
        st.caption(f"⚠️ تعذّر تحميل سجل تدقيق الوكلاء: {_audit_err}")

    if _summary:
        if _summary["total_events"] == 0:
            st.caption("لا توجد تفاعلات مسجَّلة بعد — استخدم تبويب \"🤖 وكلاء AI\" أو \"🤝 منسّق الوكلاء\" أولاً.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("إجمالي التفاعلات", _summary["total_events"])
            m2.metric("عبر hub", _summary["by_source"].get("hub", 0))
            m3.metric("عبر orchestrator", _summary["by_source"].get("orchestrator", 0))

            web_pct = (
                (_summary["web_used_count"] / _summary["total_events"]) * 100
                if _summary["total_events"] else 0
            )
            st.caption(f"🌐 استخدم بحث ويب حقيقي في {_summary['web_used_count']} تفاعل ({web_pct:.0f}%)")

            if _summary["by_category"]:
                st.markdown(
                    "**حسب الوكيل:** " + "، ".join(
                        f"{k}: {v}" for k, v in _summary["by_category"].items()
                    )
                )

            with st.expander("📋 آخر التفاعلات المسجَّلة"):
                recent = _audit.get_recent(15)
                for entry in recent:
                    web_tag = "🌐" if entry.get("web_used") else ""
                    src_tag = "🤝" if entry.get("source") == "orchestrator" else "🤖"
                    st.markdown(
                        f"{src_tag} **{entry.get('category_title', '')}** "
                        f"{web_tag} — {entry.get('provider', '') or '—'} "
                        f"— {entry.get('timestamp', '')[:19]}"
                    )
                    q = entry.get("question_preview", "")
                    if q:
                        st.caption(f"س: {q[:120]}{'…' if len(q) > 120 else ''}")


# ═══════════════════════════════════════════════════════════════════════════
# تبويب API متقدمة
# ═══════════════════════════════════════════════════════════════════════════

def render_advanced_api():
    """تبويب API متقدمة — Web Search · تحليل الصور · JSON منظّم"""

    st.markdown('<div class="section-header">🔬 API متقدمة — Anthropic Claude</div>', unsafe_allow_html=True)

    # ── فحص توفّر المفتاح ────────────────────────────────────────────────
    try:
        from ai.anthropic_advanced import AnthropicAdvanced
        from ai.llm_fallback import ANTHROPIC_MODELS
        _test_client = AnthropicAdvanced()
        _has_key = _test_client.available
    except Exception as _imp_err:
        st.error(f"⚠️ تعذّر تحميل وحدة API المتقدمة: {_imp_err}")
        return

    if not _has_key:
        st.warning(
            "🔑 **ANTHROPIC_API_KEY غير موجود** — أضفه في Secrets لتفعيل هذا التبويب.\n\n"
            "الأدوات المتاحة هنا: Web Search · تحليل الصور · استخراج JSON منظّم"
        )
        st.info("💡 بعد إضافة المفتاح، اضغط **R** لإعادة تشغيل التطبيق.")
        return

    # ── اختيار النموذج ────────────────────────────────────────────────────
    st.markdown("#### ⚙️ إعدادات")
    col_m, col_t = st.columns([2, 1])
    with col_m:
        model_choice = st.selectbox(
            "النموذج",
            options=list(ANTHROPIC_MODELS.values()),
            index=0,
            format_func=lambda m: {
                "claude-sonnet-4-6":         "⚡ Sonnet 4-6 (الافتراضي)",
                "claude-opus-4-8":           "💎 Opus 4-8 (الأقوى)",
                "claude-haiku-4-5-20251001": "🚀 Haiku 4-5 (الأسرع)",
                "claude-sonnet-4-20250514":  "🔒 Sonnet Stable",
            }.get(m, m),
            key="adv_model",
        )
    with col_t:
        max_tokens = st.slider("الحد الأقصى للتوكنات", 256, 2048, 800, 128, key="adv_max_tokens")

    client = AnthropicAdvanced(model=model_choice, max_tokens=max_tokens)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # الأقسام الثلاثة
    # ══════════════════════════════════════════════════════════════════════
    sec1, sec2, sec3, sec4 = st.tabs(
        ["🌐 بحث الويب", "🖼️ تحليل الصور", "📐 JSON منظّم", "🔌 MCP Servers"]
    )

    # ────────────────────────────────────────────────────────────────────
    # القسم 1 — Web Search Tool
    # ────────────────────────────────────────────────────────────────────
    with sec1:
        st.markdown("""
        <div style="background:color-mix(in srgb, #38bdf8 12%, var(--surface2));border:1px solid color-mix(in srgb, #38bdf8 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🌐 Web Search Tool</strong><br>
            <small>يُفعّل أداة البحث في الويب المدمجة في Anthropic API —
            النموذج يقرر بنفسه متى وكيف يبحث ثم يدمج النتائج في إجابته.</small>
        </div>
        """, unsafe_allow_html=True)

        ws_query = st.text_area(
            "سؤالك (سيبحث النموذج في الويب تلقائياً)",
            placeholder="مثال: ما آخر إصدارات نماذج Anthropic Claude؟\nأو: ما أحدث أخبار الذكاء الاصطناعي اليوم؟",
            height=100, key="ws_query",
        )
        ws_system = st.text_input(
            "تعليمات النظام (اختياري)",
            value="أجب بالعربية الفصحى بشكل مختصر ومنظّم.",
            key="ws_system",
        )

        if st.button("🔍 ابحث وأجب", key="ws_run", use_container_width=True, type="primary"):
            if not ws_query.strip():
                st.warning("أدخل سؤالك أولاً.")
            else:
                with st.spinner("⟳ يبحث النموذج في الويب..."):
                    result = client.ask_with_search(ws_query.strip(), system=ws_system.strip())

                if result.error:
                    st.error(f"❌ خطأ: {result.error}")
                else:
                    st.markdown("#### 📝 الإجابة")
                    st.markdown(f"""
                    <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                                padding:1rem 1.4rem;direction:rtl;line-height:1.9;
                                white-space:pre-wrap;font-size:0.97rem">
                    {result.text or "لا توجد إجابة نصية."}
                    </div>
                    """, unsafe_allow_html=True)

                    if result.tool_calls:
                        with st.expander(f"🔧 أدوات استُخدمت ({len(result.tool_calls)})"):
                            for tc in result.tool_calls:
                                st.json(tc)

                    if result.tool_results:
                        with st.expander(f"📦 نتائج البحث الخام ({len(result.tool_results)})"):
                            for tr in result.tool_results:
                                st.text(tr[:800])

                    cols = st.columns(3)
                    cols[0].metric("نموذج", result.model.split("-")[-1] if result.model else "—")
                    cols[1].metric("زمن الاستجابة", f"{result.latency_ms:.0f} ms")
                    cols[2].metric("توكنات الإخراج", result.output_tokens)

    # ────────────────────────────────────────────────────────────────────
    # القسم 2 — تحليل الصور
    # ────────────────────────────────────────────────────────────────────
    with sec2:
        st.markdown("""
        <div style="background:color-mix(in srgb, #c084fc 14%, var(--surface2));border:1px solid color-mix(in srgb, #c084fc 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🖼️ تحليل الصور</strong><br>
            <small>ارفع صورة (JPEG · PNG · GIF · WebP) واطرح سؤالاً عنها —
            النموذج سيحلّلها ويجيب بالعربية.</small>
        </div>
        """, unsafe_allow_html=True)

        img_file = st.file_uploader(
            "ارفع صورة", type=["jpg", "jpeg", "png", "gif", "webp"], key="img_upload"
        )
        img_question = st.text_area(
            "سؤالك عن الصورة",
            placeholder="مثال: صِف ما تراه في هذه الصورة.\nأو: هل تحتوي على نص؟ اقرأه.",
            height=90, key="img_question",
        )

        if img_file:
            st.image(img_file, caption="الصورة المرفوعة", use_container_width=False, width=350)

        if st.button("🔍 حلّل الصورة", key="img_run", use_container_width=True, type="primary"):
            if not img_file:
                st.warning("ارفع صورة أولاً.")
            elif not img_question.strip():
                st.warning("أدخل سؤالك أولاً.")
            else:
                mime_map = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif", "webp": "image/webp",
                }
                ext = img_file.name.rsplit(".", 1)[-1].lower()
                media_type = mime_map.get(ext, "image/jpeg")
                img_bytes = img_file.read()

                with st.spinner("⟳ يحلّل النموذج الصورة..."):
                    answer = client.ask_with_image(
                        img_question.strip(), img_bytes, media_type,
                        system="أجب بالعربية الفصحى.",
                    )

                st.markdown("#### 📝 تحليل النموذج")
                st.markdown(f"""
                <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                            padding:1rem 1.4rem;direction:rtl;line-height:1.9;
                            white-space:pre-wrap;font-size:0.97rem">
                {answer or "لم يُنتج النموذج إجابة."}
                </div>
                """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # القسم 3 — JSON منظّم
    # ────────────────────────────────────────────────────────────────────
    with sec3:
        st.markdown("""
        <div style="background:color-mix(in srgb, #34d399 14%, var(--surface2));border:1px solid color-mix(in srgb, #34d399 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>📐 استخراج JSON منظّم</strong><br>
            <small>اطلب من النموذج إجابة JSON خالصة — مناسب لاستخراج البيانات
            وتحليل النصوص وبناء APIs.</small>
        </div>
        """, unsafe_allow_html=True)

        json_query = st.text_area(
            "طلبك",
            placeholder="مثال: استخرج من النص التالي: الاسم والعمر والمهنة.\nأو: أعطني قائمة بأسماء الخلفاء الراشدين مع تواريخ خلافتهم.",
            height=110, key="json_query",
        )
        json_schema = st.text_input(
            "وصف البنية المطلوبة (اختياري)",
            placeholder='مثال: { "name": "string", "year": "number" }',
            key="json_schema",
        )

        if st.button("⚙️ استخرج JSON", key="json_run", use_container_width=True, type="primary"):
            if not json_query.strip():
                st.warning("أدخل طلبك أولاً.")
            else:
                with st.spinner("⟳ يولّد النموذج JSON..."):
                    data = client.ask_json(
                        json_query.strip(),
                        json_schema_hint=json_schema.strip(),
                    )

                if data is None:
                    st.error("❌ فشل تحليل JSON — قد لا يدعم النموذج هذا الطلب بصيغة JSON خالصة.")
                    raw_text = client.ask(json_query.strip())
                    if raw_text:
                        st.markdown("**الرد الخام:**")
                        st.code(raw_text, language="text")
                else:
                    st.success("✅ JSON مُستخرَج بنجاح")
                    st.json(data)

                    import json as _json
                    json_str = _json.dumps(data, ensure_ascii=False, indent=2)
                    st.download_button(
                        "⬇️ تحميل JSON",
                        data=json_str,
                        file_name="nsm_output.json",
                        mime="application/json",
                        key="json_download",
                    )

    # ────────────────────────────────────────────────────────────────────
    # القسم 4 — MCP Servers (Model Context Protocol)
    # ────────────────────────────────────────────────────────────────────
    with sec4:
        st.markdown("""
        <div style="background:color-mix(in srgb, #f87171 14%, var(--surface2));border:1px solid color-mix(in srgb, #f87171 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🔌 MCP Servers (Model Context Protocol)</strong><br>
            <small>يتصل النموذج مباشرة بخوادم MCP بعيدة (Google Drive، Gmail، Google
            Calendar، Canva، Figma، أو أي خادم MCP آخر) وينفّذ أدواتها الفعلية أثناء
            توليد الرد. يتطلب أن يكون الحساب المرتبط مصرّحاً (OAuth) لكل خادم حسب
            سياسته الخاصة.</small>
        </div>
        """, unsafe_allow_html=True)

        MCP_PRESETS = {
            "Google Drive":   "https://drivemcp.googleapis.com/mcp/v1",
            "Gmail":          "https://gmailmcp.googleapis.com/mcp/v1",
            "Google Calendar": "https://calendarmcp.googleapis.com/mcp/v1",
            "Canva":          "https://mcp.canva.com/mcp",
            "Figma":          "https://mcp.figma.com/mcp",
        }
        mcp_chosen = st.multiselect(
            "اختر خوادم MCP جاهزة للتفعيل",
            options=list(MCP_PRESETS.keys()),
            key="mcp_servers_choice",
        )
        mcp_custom_url = st.text_input(
            "أو أضف رابط خادم MCP مخصّص (اختياري)",
            placeholder="https://example.com/mcp",
            key="mcp_custom_url",
        )
        mcp_query = st.text_area(
            "سؤالك/طلبك",
            placeholder="مثال: لخّص آخر ملف في Google Drive باسم يحتوي 'تفسير'.",
            height=110, key="mcp_query",
        )

        if st.button("🔌 نفّذ عبر MCP", key="mcp_run", use_container_width=True, type="primary"):
            servers = [
                {"type": "url", "url": MCP_PRESETS[name], "name": name}
                for name in mcp_chosen
            ]
            if mcp_custom_url.strip():
                servers.append({"type": "url", "url": mcp_custom_url.strip(), "name": "مخصّص"})

            if not mcp_query.strip():
                st.warning("أدخل سؤالك أولاً.")
            elif not servers:
                st.warning("اختر خادم MCP واحداً على الأقل أو أضف رابطاً مخصصاً.")
            else:
                with st.spinner("⟳ يتصل بخوادم MCP..."):
                    mcp_result = client.ask_with_mcp(mcp_query.strip(), servers)

                if mcp_result.error:
                    st.error(f"❌ {mcp_result.error}")
                else:
                    st.success("✅ تم")
                    if mcp_result.text:
                        st.markdown(mcp_result.text)
                    if mcp_result.tool_calls:
                        with st.expander(f"🔧 استدعاءات الأدوات ({len(mcp_result.tool_calls)})"):
                            for tc in mcp_result.tool_calls:
                                st.json(tc)
                    if mcp_result.tool_results:
                        with st.expander(f"📄 نتائج الأدوات ({len(mcp_result.tool_results)})"):
                            for tr in mcp_result.tool_results:
                                st.code(tr[:2000])

    # ── ملاحظة ختامية ───────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "هذه الأدوات تستخدم `ai/anthropic_advanced.py` — مستخلصة من Claude.ai System Prompt (That.md). "
        "كل استدعاء يُرسَل مباشرة إلى Anthropic API."
    )


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 دوال تجميع التبويبات — تدمج تبويبات متشابهة عبر تبويبات فرعية (sub-tabs)
# بدون حذف أي وظيفة أصلية؛ كل دالة render_ القديمة تبقى كما هي وتُستدعى
# من الداخل فقط، لتقليل عدد التبويبات الرئيسية من 21 إلى 12.
# ═══════════════════════════════════════════════════════════════════════════

def render_knowledge_hub():
    """📚 المعرفة: يجمع البحث المعرفي + القرآن الكريم + الأسئلة والأجوبة."""
    sub = st.tabs(["🔍 البحث المعرفي", "📖 القرآن الكريم", "❓ الأسئلة والأجوبة"])
    with sub[0]: render_search()
    with sub[1]: render_quran()
    with sub[2]: render_qa()


def render_agents_group():
    """🤖 الوكلاء: يجمع الوكيل الموحّد + وكلاء AI + منسّق الوكلاء + السرب الذكي."""
    sub = st.tabs(["🎯 الوكيل الموحّد", "🤖 وكلاء AI", "🤝 منسّق الوكلاء", "🐝 السرب الذكي"])
    with sub[0]: render_unified_agent()
    with sub[1]: render_agents_hub()
    with sub[2]: render_agent_orchestrator()
    with sub[3]: render_swarm_studio()


def render_system_group():
    """⚙️ النظام: يجمع الذاكرة + صحة النظام + API متقدمة + النظام الداخلي + لوحة المطوّر.
    محمية بالكامل بمفتاح المالك (NSM_ADMIN_KEY) — هذه أدوات تشخيص داخلية،
    مو ميزة للمستخدم النهائي، ولازم ما تكون ظاهرة لأي زائر بدون مصادقة."""
    st.markdown('<div class="section-header">⚙️ النظام</div>', unsafe_allow_html=True)

    # تحقق أمان احتياطي (defense-in-depth): هذا التبويب أصلاً لا يُضاف لقائمة
    # التبويبات في main() إلا بعد فتح وضع المالك من الشريط الجانبي، لكن نبقي
    # هذا الفحص هنا كخط دفاع ثانٍ في حال استُدعيت الدالة من مكان آخر مستقبلاً.
    _admin_key_env = os.environ.get("NSM_ADMIN_KEY", "")
    if not _admin_key_env or not st.session_state.get("_dev_console_unlocked", False):
        st.error("❌ هذا القسم محمي بوضع المالك — افتحه من الشريط الجانبي.")
        return

    col_lock, _ = st.columns([1, 4])
    with col_lock:
        if st.button("🔒 قفل قسم النظام", key="system_group_lock"):
            st.session_state["_dev_console_unlocked"] = False
            st.rerun()

    sub = st.tabs(["🧠 الذاكرة", "🏥 صحة النظام", "🔬 API متقدمة",
                   "⚙️ النظام الداخلي", "🖥️ لوحة المطوّر"])
    with sub[0]: render_memory()
    with sub[1]: render_health()
    with sub[2]: render_advanced_api()
    with sub[3]: render_system_core()
    with sub[4]: render_dev_console()


def render_advanced_tools_group():
    """🧪 أدوات متقدمة: يجمع ULTRAPLINIAN + الواجهات التفاعلية.
    الواجهات التفاعلية (Artifacts) لا صلة لها بمهمة المشروع، وتخزينها
    مشترك بين كل الزوار بدون عزل ملكية (أي زائر يشوف/يحذف واجهات غيره،
    وأي HTML/JS محفوظ يُنفَّذ تلقائياً لكل الزوار) — لذلك تظهر للمالك
    فقط بعد فتح وضع المالك من الشريط الجانبي."""
    _tool_tab_defs = [("⚡ ULTRAPLINIAN", render_ultraplinian)]
    if st.session_state.get("_dev_console_unlocked", False):
        _tool_tab_defs.append(("🧩 الواجهات التفاعلية", render_artifacts_studio))

    sub = st.tabs([_label for _label, _fn in _tool_tab_defs])
    for _tab, (_label, _fn) in zip(sub, _tool_tab_defs):
        with _tab:
            _fn()


# ═══════════════════════════════════════════════════════════════════════════
# التطبيق الرئيسي
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── الشريط الجانبي — OpenRouter ───────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🌐 Neural Service Mesh")

        # مبدّل السمة: داكن (بنفسجي/فيروزي) / فاتح
        st.markdown('<div class="theme-toggle-caption">🎨 المظهر</div>', unsafe_allow_html=True)
        _theme_cols = st.columns(2)
        _current_theme = st.session_state.get("ui_theme", "dark")
        with _theme_cols[0]:
            if st.button(
                ("● " if _current_theme == "dark" else "") + "🌙 داكن",
                key="theme_btn_dark", use_container_width=True,
            ):
                st.session_state.ui_theme = "dark"
                try:
                    from core.artifacts_store import set_setting as _persist_setting
                    _persist_setting("ui_theme", "dark")
                except Exception:
                    pass
                st.rerun()
        with _theme_cols[1]:
            if st.button(
                ("● " if _current_theme == "light" else "") + "☀️ فاتح",
                key="theme_btn_light", use_container_width=True,
            ):
                st.session_state.ui_theme = "light"
                try:
                    from core.artifacts_store import set_setting as _persist_setting
                    _persist_setting("ui_theme", "light")
                except Exception:
                    pass
                st.rerun()

        st.markdown("---")

        # ── 👤 الحساب (تسجيل دخول / إنشاء حساب) ─────────────────────────
        st.markdown("### 👤 الحساب")
        try:
            from ai.accounts import create_user as _acc_create, verify_login as _acc_login, AccountError as _AccErr
            _accounts_module_ok = True
        except Exception:
            _accounts_module_ok = False

        if not _accounts_module_ok:
            st.caption("نظام الحسابات غير متاح حالياً")
        elif st.session_state.get("_account"):
            _acc = st.session_state["_account"]
            st.success(f"مسجّل الدخول: {_acc['username']}")
            if st.button("🚪 تسجيل خروج", key="account_logout_btn", use_container_width=True):
                del st.session_state["_account"]
                st.rerun()
        else:
            _acc_tab_login, _acc_tab_register = st.tabs(["دخول", "حساب جديد"])
            with _acc_tab_login:
                with st.form(key="account_login_form", clear_on_submit=False):
                    _li_user = st.text_input("اسم المستخدم", key="account_login_username")
                    _li_pass = st.text_input("كلمة المرور", type="password", key="account_login_password")
                    _li_submit = st.form_submit_button("دخول 🔐", use_container_width=True)
                if _li_submit:
                    try:
                        _user = _acc_login(_li_user, _li_pass) if _li_user and _li_pass else None
                        if _user:
                            st.session_state["_account"] = _user
                            st.rerun()
                        else:
                            st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
                    except _AccErr as _e:
                        st.error(str(_e))
            with _acc_tab_register:
                with st.form(key="account_register_form", clear_on_submit=False):
                    _reg_user = st.text_input("اسم المستخدم", key="account_reg_username")
                    _reg_pass = st.text_input("كلمة المرور", type="password", key="account_reg_password")
                    _reg_phone = st.text_input(
                        "رقم الهاتف (اختياري — لربط واتساب لاحقاً)",
                        key="account_reg_phone", placeholder="+9677xxxxxxxx",
                    )
                    _reg_submit = st.form_submit_button("إنشاء حساب ✨", use_container_width=True)
                if _reg_submit:
                    try:
                        _acc_create(_reg_user, _reg_pass, phone_number=_reg_phone or None)
                        st.success("تم إنشاء الحساب! سجّل دخولك من تبويب «دخول»")
                    except _AccErr as _e:
                        st.error(str(_e))
                    except Exception:
                        st.error("تعذّر إنشاء الحساب")

        st.markdown("---")

        # ── ⚙️ الإعدادات المتقدمة — مطوية افتراضياً لواجهة أنظف للزائر ─────
        _admin_unlocked_now = st.session_state.get("_dev_console_unlocked", False)
        _adv_label = "🔓 الإعدادات المتقدمة (وضع المالك مفعّل)" if _admin_unlocked_now else "⚙️ الإعدادات المتقدمة"
        with st.expander(_adv_label, expanded=False):
            st.markdown("##### 🔑 OpenRouter API")
            st.caption("مفتاح اختياري — يُفعّل النماذج التجارية في تبويبَي المحادثة و G0DM0D3")

            if "_or_api_key" not in st.session_state:
                st.session_state["_or_api_key"] = os.getenv("OPENROUTER_API_KEY", "")

            _or_key_stored = st.session_state.get("_or_api_key", "")
            _or_key_input = st.text_input(
                "OpenRouter API Key",
                value=_or_key_stored,
                type="password",
                placeholder="sk-or-v1-...",
                label_visibility="collapsed",
                key="or_key_input_widget",
            )
            if _or_key_input != _or_key_stored:
                st.session_state["_or_api_key"] = _or_key_input

            _or_key = st.session_state.get("_or_api_key", "").strip()

            if _or_key:
                st.success("✅ OpenRouter مُفعَّل")
                _or_model_label = st.selectbox(
                    "النموذج",
                    list(OPENROUTER_MODEL_OPTIONS.keys()),
                    index=0,
                    key="or_model_select",
                    label_visibility="collapsed",
                )
                st.session_state["_or_model"] = OPENROUTER_MODEL_OPTIONS[_or_model_label]
            else:
                st.info("بدون مفتاح → يُستخدم NSM/LLMFallback")
                st.session_state["_or_model"] = "google/gemini-2.5-flash"

            st.markdown("---")

            # ── 🔐 وضع المالك — يتحكم بظهور تبويب ⚙️ النظام بالكامل ─────
            st.markdown("##### 🔐 وضع المالك")
            _sidebar_admin_key_env = os.environ.get("NSM_ADMIN_KEY", "")
            if not _sidebar_admin_key_env:
                st.caption("قسم النظام معطّل (NSM_ADMIN_KEY غير مضبوط في Secrets)")
            elif st.session_state.get("_dev_console_unlocked", False):
                st.success("🔓 وضع المالك مفعّل — تبويب ⚙️ النظام ظاهر")
                if st.button("🔒 قفل وضع المالك", key="sidebar_admin_lock", use_container_width=True):
                    st.session_state["_dev_console_unlocked"] = False
                    st.rerun()
            else:
                _sidebar_admin_key_input = st.text_input(
                    "مفتاح المالك", type="password", key="sidebar_admin_key_input",
                )
                if st.button("🔓 فتح وضع المالك", key="sidebar_admin_unlock", use_container_width=True):
                    if _sidebar_admin_key_input == _sidebar_admin_key_env:
                        st.session_state["_dev_console_unlocked"] = True
                        st.rerun()
                    else:
                        st.error("❌ مفتاح غير صحيح")

            st.markdown("---")

            # ── 🗣️ التوليد الحر التجريبي (Yemeni LLM) ────────────────────
            st.markdown("##### 🗣️ التوليد الحر (تجريبي)")
            st.session_state["yemeni_generation_mode"] = st.toggle(
                "تفعيل التوليد الحر (Yemeni LLM)",
                value=st.session_state.get("yemeni_generation_mode", False),
                key="yemeni_generation_toggle",
            )
            if st.session_state["yemeni_generation_mode"]:
                st.caption(
                    "⚠️ ميزة تجريبية: النموذج التوليدي (YemeniDecoder) لم يخضع "
                    "لتدريب فعلي بعد — النص المولَّد قد يكون غير مفهوم حالياً. "
                    "الإجابة الرمزية الأساسية تبقى تُعرض دائماً بجانبه."
                )
                st.session_state["yemeni_temperature"] = st.slider(
                    "الحرارة (Temperature)", min_value=0.1, max_value=1.5,
                    value=st.session_state.get("yemeni_temperature", 0.8),
                    step=0.05, key="yemeni_temp_slider",
                )
                st.session_state["yemeni_top_p"] = st.slider(
                    "Top-P", min_value=0.1, max_value=1.0,
                    value=st.session_state.get("yemeni_top_p", 0.95),
                    step=0.05, key="yemeni_top_p_slider",
                )
                st.session_state["yemeni_top_k"] = st.slider(
                    "Top-K", min_value=1, max_value=100,
                    value=st.session_state.get("yemeni_top_k", 50),
                    step=1, key="yemeni_top_k_slider",
                )

        st.markdown("---")
        st.caption("🧠 النظام المعرفي العربي")
        st.caption("CKG · قرآن · AutoTune")
        st.caption("⌘K / Ctrl+K — بحث سريع للتنقّل بين الأقسام")

    # ── العنوان ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="hero-wrap">
    <div class="hero-split">
        <div class="hero-split-text">
            <div class="hero-badges">
                <div class="hero-badge"><span class="dot"></span> شبكة معرفية حيّة</div>
                <div class="hero-badge"><span class="dot"></span> عربي 100٪</div>
                <div class="hero-badge"><span class="dot"></span> مبني على القرآن الكريم</div>
            </div>
            <div class="main-title">🧠 النظام المعرفي العربي</div>
            <div class="subtitle">Neural Service Mesh · ذكاء اصطناعي عربي متخصص بالمعرفة الإسلامية</div>
            <div class="welcome-line">
                اسأل عن أي مفهوم إسلامي أو عربي، وسيربطه النظام بشبكة معرفية حيّة
                مبنية على القرآن الكريم وعلوم اللغة — بحث، محادثة، ومحتوى إبداعي، كل ذلك بالعربية.
            </div>
        </div>
        <div class="hero-split-visual">
            <div class="hero-chip hero-chip--top">📖 قرآن كريم</div>
            <div class="hero-visual-panel">
                <div class="hero-visual-icon">🧠</div>
            </div>
            <div class="hero-chip hero-chip--bottom">🕸️ شبكة معرفية</div>
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

    # ── التبويبات ─────────────────────────────────────────────────────────
    # تبويب ⚙️ النظام لا يُضاف لقائمة التبويبات أصلاً إلا بعد فتح وضع المالك
    # من الشريط الجانبي — أي أنه مخفي كلياً عن الزوار العاديين، لا مجرد
    # محتوى محمي داخل تبويب ظاهر.
    _tab_defs = [
        ("🏠 الرئيسية", render_home),
        ("📚 المعرفة", render_knowledge_hub),
        ("💬 المحادثة", render_chat),
        ("🤖 الوكلاء", render_agents_group),
        ("🎭 إبداع", render_fable),
        ("🌐 ترجمة", render_translate),
        ("🎬 Higgsfield", render_higgsfield),
        ("📡 الوكيل الاجتماعي", render_social_agent),
        ("🎓 التدريب", render_training),
    ]
    if st.session_state.get("_dev_console_unlocked", False):
        _tab_defs.append(("⚙️ النظام", render_system_group))
        _tab_defs.append(("🧪 أدوات متقدمة", render_advanced_tools_group))
    _tab_defs.append(("ℹ️ عن NSM", render_product_info))

    tabs = st.tabs([_label for _label, _fn in _tab_defs])
    for _tab, (_label, _fn) in zip(tabs, _tab_defs):
        with _tab:
            _fn()

    # ── تذييل الصفحة ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:var(--text-muted); font-size:0.8rem; direction:rtl">
        Neural Service Mesh · نظام معرفي عربي ذاتي التعلم · مبني بـ Python & Streamlit
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# تبويب 🧩 الواجهات التفاعلية — Artifacts (HTML/SVG) + استدعاء API
# ══════════════════════════════════════════════════════════════════════════
def render_artifacts_studio():
    st.markdown('<div class="section-header">🧩 الواجهات التفاعلية (Artifacts)</div>', unsafe_allow_html=True)
    st.caption("أنشئ واعرض محتوى HTML/SVG تفاعلياً داخل التطبيق — رسوم بيانية، نماذج، بطاقات، إلخ.")

    try:
        from core.artifacts_store import (
            save_artifact, list_artifacts, get_artifact, delete_artifact,
        )
        _ART_STORE_OK = True
    except Exception as _art_err:
        _ART_STORE_OK = False
        st.error(f"⚠️ تعذّر تحميل مخزن الواجهات التفاعلية: {_art_err}")

    # تبويب "🔌 استدعاء API" أداة HTTP عامة بدون تحقق من الوجهة (خطر SSRF) —
    # لذلك لا يُضاف لقائمة التبويبات الفرعية أصلاً إلا بعد فتح وضع المالك
    # من الشريط الجانبي، تماماً كما فعلنا مع تبويب ⚙️ النظام.
    _art_tab_defs = [("🖼️ محرّر HTML/SVG", "editor")]
    if st.session_state.get("_dev_console_unlocked", False):
        _art_tab_defs.append(("🔌 استدعاء API", "api_caller"))

    _art_tabs = st.tabs([_label for _label, _kind in _art_tab_defs])
    art_tab1 = _art_tabs[0]
    art_tab2 = _art_tabs[1] if len(_art_tabs) > 1 else None

    # ── محرّر ومعرض الواجهات التفاعلية ───────────────────────────────────
    with art_tab1:
        _default_html = (
            "<div style=\"font-family:sans-serif;text-align:center;padding:2rem;"
            "background:linear-gradient(135deg,var(--gold),var(--emerald));color:#fff;border-radius:16px\">"
            "<h2>مرحباً من NSM 🧠</h2><p>هذا مثال بسيط — عدّل الكود وشاهد النتيجة فوراً.</p></div>"
        )
        col_edit, col_preview = st.columns([1, 1])
        with col_edit:
            art_title = st.text_input("عنوان الواجهة", value="واجهتي الجديدة", key="art_title")
            art_code = st.text_area(
                "كود HTML/SVG", value=_default_html, height=320, key="art_code",
                help="يمكنك كتابة HTML كامل مع <style> و<script> — يُعرض داخل إطار معزول.",
            )
            art_height = st.slider("ارتفاع العرض (px)", 200, 900, 420, 20, key="art_height")
            c1, c2 = st.columns(2)
            with c1:
                art_render_btn = st.button("🖥️ عرض", key="art_render_btn", use_container_width=True, type="primary")
            with c2:
                art_save_btn = st.button("💾 حفظ", key="art_save_btn", use_container_width=True,
                                          disabled=not _ART_STORE_OK)
            if art_save_btn and _ART_STORE_OK:
                if art_code.strip():
                    new_id = save_artifact(art_title, art_code, kind="html")
                    st.success(f"✅ تم الحفظ (رقم #{new_id})")
                else:
                    st.warning("أدخل كوداً أولاً.")

        with col_preview:
            st.markdown("**المعاينة:**")
            if art_render_btn or art_code.strip():
                try:
                    st.components.v1.html(art_code, height=art_height, scrolling=True)
                except Exception as _render_err:
                    st.error(f"❌ خطأ أثناء العرض: {_render_err}")

        if _ART_STORE_OK:
            st.markdown("---")
            st.markdown("#### 📚 الواجهات المحفوظة")
            saved = list_artifacts()
            if not saved:
                st.info("لا توجد واجهات محفوظة بعد.")
            else:
                for item in saved[:20]:
                    with st.expander(f"#{item['id']} — {item['title']} · {item['created_at'][:19].replace('T',' ')}"):
                        full = get_artifact(item["id"])
                        st.components.v1.html(full["content"], height=300, scrolling=True)
                        dcol1, dcol2 = st.columns(2)
                        with dcol1:
                            if st.button("📋 حمّل في المحرّر", key=f"art_load_{item['id']}"):
                                st.session_state["art_code"] = full["content"]
                                st.session_state["art_title"] = full["title"]
                                st.rerun()
                        with dcol2:
                            if st.button("🗑️ حذف", key=f"art_del_{item['id']}"):
                                delete_artifact(item["id"])
                                st.rerun()

    # ── استدعاء APIs مباشرة من الواجهة — للمالك فقط ──────────────────────
    if art_tab2 is not None:
      with art_tab2:
        st.warning("🔒 أداة داخلية للمالك — ترسل طلبات HTTP فعلية من الخادم لأي رابط تُدخله. لا تشاركها مع أحد.")
        st.markdown("""
        <div style="background:color-mix(in srgb, #38bdf8 12%, var(--surface2));border:1px solid color-mix(in srgb, #38bdf8 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🔌 جرّب أي API مباشرة</strong><br>
            <small>أدخل رابط API، الطريقة، والترويسات/الجسم (JSON) — وشاهد الاستجابة فوراً.</small>
        </div>
        """, unsafe_allow_html=True)

        api_url = st.text_input("رابط الـ API", placeholder="https://api.example.com/data", key="api_tool_url")
        colm, colh = st.columns([1, 3])
        with colm:
            api_method = st.selectbox("الطريقة", ["GET", "POST", "PUT", "PATCH", "DELETE"], key="api_tool_method")
        with colh:
            api_headers_raw = st.text_input(
                "ترويسات (JSON، اختياري)", placeholder='{"Authorization": "Bearer ..."}', key="api_tool_headers"
            )
        api_body_raw = st.text_area(
            "جسم الطلب (JSON، اختياري — لـ POST/PUT/PATCH)", height=100, key="api_tool_body"
        )

        if st.button("▶️ استدعِ API", key="api_tool_run", type="primary"):
            if not api_url.strip():
                st.warning("أدخل رابط الـ API أولاً.")
            else:
                try:
                    headers = json.loads(api_headers_raw) if api_headers_raw.strip() else {}
                except Exception:
                    st.error("❌ الترويسات ليست JSON صالحاً.")
                    headers = None
                try:
                    body = json.loads(api_body_raw) if api_body_raw.strip() else None
                except Exception:
                    st.error("❌ جسم الطلب ليس JSON صالحاً.")
                    body = None
                    api_body_raw_invalid = True
                else:
                    api_body_raw_invalid = False

                if headers is not None and not api_body_raw_invalid:
                    try:
                        with st.spinner("⟳ جارٍ الاتصال..."):
                            resp = _requests.request(
                                api_method, api_url.strip(), headers=headers or None,
                                json=body if api_method in ("POST", "PUT", "PATCH") else None,
                                params=body if api_method in ("GET", "DELETE") and isinstance(body, dict) else None,
                                timeout=15,
                            )
                        st.markdown(f"**الحالة:** `{resp.status_code}`  ·  **الزمن:** `{resp.elapsed.total_seconds()*1000:.0f} ms`")
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.text(resp.text[:3000])
                    except Exception as _api_err:
                        st.error(f"❌ فشل الاتصال: {_api_err}")


# ══════════════════════════════════════════════════════════════════════════
# تبويب 🖥️ لوحة المطوّر — تنفيذ أوامر Bash/Python (محمي بمفتاح المالك)
# ══════════════════════════════════════════════════════════════════════════
def render_dev_console():
    st.markdown('<div class="section-header">🖥️ لوحة المطوّر</div>', unsafe_allow_html=True)
    st.warning(
        "⚠️ هذه الأداة تنفّذ أوامر حقيقية على الخادم. محمية بمفتاح المالك "
        "(`NSM_ADMIN_KEY`) — لا تشاركها مع أحد."
    )

    _admin_key_env = os.environ.get("NSM_ADMIN_KEY", "")
    if not _admin_key_env:
        st.error("❌ لم يتم ضبط NSM_ADMIN_KEY في Secrets — هذه الميزة معطّلة حتى يُضاف المفتاح.")
        return

    if not st.session_state.get("_dev_console_unlocked", False):
        entered = st.text_input("مفتاح المالك", type="password", key="dev_console_key_input")
        if st.button("🔓 فتح لوحة المطوّر", key="dev_console_unlock"):
            if entered == _admin_key_env:
                st.session_state["_dev_console_unlocked"] = True
                st.rerun()
            else:
                st.error("❌ مفتاح غير صحيح.")
        return

    col_lock, _ = st.columns([1, 4])
    with col_lock:
        if st.button("🔒 قفل", key="dev_console_lock"):
            st.session_state["_dev_console_unlocked"] = False
            st.rerun()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### تنفيذ أمر")
    cmd_kind = st.radio("النوع", ["Bash", "Python"], horizontal=True, key="dev_console_kind")
    cmd_text = st.text_area("الأمر", height=120, key="dev_console_cmd",
                             placeholder="مثال: ls -la" if cmd_kind == "Bash" else "print(1 + 1)")
    cmd_timeout = st.slider("مهلة التنفيذ (ثوانٍ)", 5, 60, 20, 5, key="dev_console_timeout")
    run_clicked = st.button("▶️ نفّذ", key="dev_console_run", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
        if not cmd_text.strip():
            st.warning("أدخل أمراً أولاً.")
        else:
            import subprocess as _sp
            _dc_ph = st.empty()
            with _dc_ph.container():
                _skeleton(lines=4)
            try:
                if cmd_kind == "Bash":
                    result = _sp.run(
                        cmd_text, shell=True, capture_output=True, text=True, timeout=cmd_timeout,
                    )
                else:
                    result = _sp.run(
                        ["python3", "-c", cmd_text], capture_output=True, text=True, timeout=cmd_timeout,
                    )
                _dc_ph.empty()
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**رمز الخروج:** `{result.returncode}`")
                if result.stdout:
                    st.markdown("**stdout:**")
                    st.code(result.stdout[-5000:])
                    _copy_button(result.stdout[-5000:], key="dev_console_stdout", label="📋 نسخ stdout")
                if result.stderr:
                    st.markdown("**stderr:**")
                    st.code(result.stderr[-5000:])
                    _copy_button(result.stderr[-5000:], key="dev_console_stderr", label="📋 نسخ stderr")
                if not result.stdout and not result.stderr:
                    st.caption("لا يوجد ناتج.")
                st.markdown("</div>", unsafe_allow_html=True)
                if result.returncode == 0:
                    st.toast("✅ تم تنفيذ الأمر بنجاح", icon="✅")
                else:
                    st.toast(f"⚠️ انتهى الأمر برمز خروج {result.returncode}", icon="⚠️")
            except _sp.TimeoutExpired:
                _dc_ph.empty()
                st.error(f"⏱️ انتهت المهلة ({cmd_timeout}s) قبل اكتمال التنفيذ.")
                st.toast("⏱️ انتهت مهلة التنفيذ", icon="⏱️")
            except Exception as _exec_err:
                _dc_ph.empty()
                st.error(f"❌ خطأ أثناء التنفيذ: {_exec_err}")
                st.toast("❌ فشل تنفيذ الأمر", icon="❌")


# ══════════════════════════════════════════════════════════════════════════
# تبويب ℹ️ عن NSM — معلومات المنتج
# ══════════════════════════════════════════════════════════════════════════
def render_product_info():
    st.markdown('<div class="section-header">ℹ️ عن Neural Service Mesh (NSM)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card" style="direction:rtl;line-height:2;font-size:1.02rem">
    <p style="margin:0"><strong>Neural Service Mesh (NSM)</strong> — النظام المعرفي العربي — هو منصة ذكاء اصطناعي
    عربية متخصصة تجمع بين محرك معرفي ذاتي التعلّم (Cognitive Knowledge Graph) ونماذج لغوية كبيرة،
    لتقديم تجربة بحث ومحادثة ومعرفة عربية أصيلة، مع تخصص خاص بالمعرفة الإسلامية والقرآن الكريم.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="font-size:1.05rem">🧭 ماذا يقدّم NSM؟</div>', unsafe_allow_html=True)
    features = [
        ("🌐", "بحث ويب حقيقي", "بحث فعلي في الإنترنت عبر DuckDuckGo بدون الحاجة لمفتاح API."),
        ("🖼️", "بحث عن الصور", "بحث عن صور حقيقية عبر Unsplash مع الوصف واسم المصوّر."),
        ("💬", "محادثة ذكية بذاكرة", "محادثة تتذكر السياق عبر الجلسات باستخدام ذاكرة SQLite طويلة الأمد."),
        ("📖", "معرفة قرآنية", "فهرسة وتحليل لغوي للقرآن الكريم — جذور، مفاهيم، علاقات دلالية."),
        ("🤖", "وكلاء AI", "وكلاء متخصصون لتحليل المشروع، البرمجة، والمهام المعرفية."),
        ("🧩", "واجهات تفاعلية", "إنشاء وعرض محتوى HTML/SVG تفاعلي واستدعاء أي API مباشرة."),
        ("🧠", "ذاكرة متقدمة", "ذاكرة دلالية (CKG) + ذاكرة حقائق + سجل محادثات قابل للاستعراض والبحث."),
        ("🖥️", "لوحة مطوّر", "تنفيذ أوامر Bash/Python محمي بمفتاح خاص بالمالك فقط."),
    ]
    _pi_cards_html = "".join(f"""
            <div class="feature-card" style="cursor:default;">
                <div class="feature-icon">{_icon}</div>
                <div class="feature-title">{_title}</div>
                <div class="feature-desc">{_desc}</div>
            </div>""" for _icon, _title, _desc in features)
    st.markdown(
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));'
        f'gap:1rem;direction:rtl;">{_pi_cards_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("")
    st.markdown('<div class="section-header" style="font-size:1.05rem">🔗 روابط</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-card" style="direction:rtl">
        <p style="margin:0 0 0.4rem 0">📦 المستودع:
        <a href="https://github.com/aliahmed369000000-ai/Neural-Service-Mesh" target="_blank">
        Neural-Service-Mesh على GitHub</a></p>
        <p style="margin:0;color:var(--text-muted)">🛠️ بُني بـ Python · Streamlit · SQLite ·
        نماذج لغوية عبر OpenRouter/Anthropic</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════
# تبويب ⚡ ULTRAPLINIAN — سباق النماذج المتوازي عبر OpenRouter
# ══════════════════════════════════════════════════════════════════════════
def render_ultraplinian():
    st.markdown("### ⚡ ULTRAPLINIAN — سباق النماذج المتوازي")

    _or_key = st.session_state.get("_or_api_key", "").strip()
    _providers = available_providers()
    _has_direct = any(_providers.values())

    if not _ULTRAPLINIAN_OK:
        st.warning("⚠️ تعذّر تحميل وحدة ai/ultraplinian.py.")
        return
    if not _or_key and not _has_direct:
        st.info(
            "🔑 لا يوجد أي مزوّد جاهز — أضِف OpenRouter API Key في الشريط "
            "الجانبي، أو GROQ_API_KEY / GOOGLE_API_KEY / (CF_API_TOKEN + "
            "CF_ACCOUNT_ID) في Streamlit Secrets لتفعيل السباق مجاناً بدون "
            "OpenRouter."
        )
        return

    _direct_names = {"groq": "Groq", "gemini": "Gemini", "cloudflare": "Cloudflare"}
    _active = [v for k, v in _direct_names.items() if _providers.get(k)]
    if _active:
        st.caption("✅ مزوّدون مباشرون مفعّلون (مجاناً بدون OpenRouter): " + "، ".join(_active))
    elif not _or_key:
        st.caption("ℹ️ لا يوجد مزوّد مباشر مفعّل — سيُستخدم OpenRouter فقط لكل النماذج.")

    st.caption(
        f"يرسل نفس السؤال إلى عدة نماذج في آنٍ واحد (حتى {total_model_count()} نموذجاً "
        "عبر 5 مستويات)، يُقيّم كل رد بنقاط مركّبة (جودة النص + تصويت Borda + "
        "تشابه دلالي)، ويعرض الفائز."
    )
    st.markdown("---")

    if "ultraplinian_tier" not in st.session_state:
        st.session_state["ultraplinian_tier"] = "fast"
    if "ultraplinian_max_models" not in st.session_state:
        st.session_state["ultraplinian_max_models"] = DEFAULT_MAX_MODELS
    if "ultraplinian_results" not in st.session_state:
        st.session_state["ultraplinian_results"] = None
    if "ultraplinian_query" not in st.session_state:
        st.session_state["ultraplinian_query"] = ""

    c1, c2 = st.columns(2)
    with c1:
        tier_labels = {
            "fast": f"⚡ FAST ({TIER_CUMULATIVE.get('fast', 10)} نموذج تراكمياً)",
            "standard": f"🎯 STANDARD ({TIER_CUMULATIVE.get('standard', 20)} نموذج تراكمياً)",
            "smart": f"🧠 SMART ({TIER_CUMULATIVE.get('smart', 31)} نموذج تراكمياً)",
            "power": f"⚔️ POWER ({TIER_CUMULATIVE.get('power', 41)} نموذج تراكمياً)",
            "ultra": f"🔱 ULTRA ({TIER_CUMULATIVE.get('ultra', 51)} نموذج تراكمياً)",
        }
        sel_tier = st.selectbox(
            "المستوى", list(tier_labels.keys()),
            index=list(tier_labels.keys()).index(st.session_state["ultraplinian_tier"]),
            format_func=lambda k: tier_labels[k])
        st.session_state["ultraplinian_tier"] = sel_tier
    with c2:
        st.session_state["ultraplinian_max_models"] = st.slider(
            "عدد النماذج في السباق", min_value=2, max_value=10,
            value=min(st.session_state["ultraplinian_max_models"], 10),
            help="عدد أكبر = تكلفة API أعلى ووقت أطول. يُنصح بـ 3-6 للاستخدام العادي.")

    include_lower = st.checkbox(
        "تضمين المستويات الأدنى أيضاً (كما في النسخة الأصلية)", value=False)

    race_query = st.text_area(
        "السؤال للسباق", value=st.session_state["ultraplinian_query"],
        placeholder="اكتب سؤالاً لإرساله لجميع النماذج المختارة في آنٍ واحد...",
        height=100)

    run_col, clear_col = st.columns([3, 1])
    with run_col:
        launch = st.button("🏁 ابدأ السباق", type="primary", use_container_width=True,
                            disabled=not race_query.strip())
    with clear_col:
        if st.button("🗑 مسح النتائج", use_container_width=True):
            st.session_state["ultraplinian_results"] = None
            st.rerun()

    if launch and race_query.strip():
        st.session_state["ultraplinian_query"] = race_query.strip()
        models = get_tier_models(
            sel_tier, st.session_state["ultraplinian_max_models"], include_lower)

        sys_prompt = NSM_PERSONA_PROMPT if _ORCHESTRATOR_OK else NSM_SYSTEM_PROMPT

        progress_box = st.empty()
        progress_bar = st.progress(0.0)

        def _on_progress(model_name, done, total):
            progress_box.caption(f"✓ اكتمل: {model_name.split('/')[-1]} ({done}/{total})")
            progress_bar.progress(done / total)

        with st.spinner(f"⚡ يتسابق {len(models)} نموذجاً..."):
            results = run_race(
                user_query=race_query.strip(),
                system_prompt=sys_prompt,
                api_key=_or_key,
                models=models,
                on_progress=_on_progress,
            )
        progress_box.empty()
        progress_bar.empty()
        st.session_state["ultraplinian_results"] = results
        st.rerun()

    results = st.session_state["ultraplinian_results"]
    if results:
        st.markdown("---")
        successes = [r for r in results if not r.error]
        failures = [r for r in results if r.error]

        if successes:
            winner = successes[0]
            st.markdown(
                f"""<div style="border:2px solid var(--gold);border-radius:10px;padding:16px;
                background:var(--gold-soft);margin-bottom:16px;">
                🏆 <b style="color:var(--gold);font-size:1.1rem;"> {winner.model.split('/')[-1]}</b>
                <span style="color:var(--text-muted);font-size:.75rem;"> — نقاط مركّبة: {winner.compound_score}</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(winner.content)
            st.markdown("---")
            st.markdown("**📊 جميع النتائج (مرتبة تنازلياً)**")
            for r in successes:
                label = f"{'🏆 ' if r.is_winner else f'#{r.rank} '}{r.model.split('/')[-1]}"
                with st.expander(
                    f"{label} — مركّبة: {r.compound_score} | "
                    f"خام: {r.raw_score} | Borda: {r.borda_score} | تشابه: {r.cluster_score} | "
                    f"{r.duration_ms:.0f}ms"
                ):
                    st.markdown(r.content[:3000] + ("…" if len(r.content) > 3000 else ""))

        if failures:
            with st.expander(f"⚠ {len(failures)} نموذج فشل"):
                for r in failures:
                    st.caption(f"**{r.model}**")
                    st.caption(friendly_error(r.error))


# ══════════════════════════════════════════════════════════════════════════
# تبويب 🎭 إبداع — السرد الإبداعي التفاعلي وتوليد الشعر
# ══════════════════════════════════════════════════════════════════════════
def render_fable():
    """تبويب القصص التفاعلية والشعر — مبني فوق نفس LLMFallback المستخدم
    في المحادثة (Anthropic أولاً ثم بقية المزوّدين المجانية)."""

    st.markdown('<div class="section-header">🎭 إبداع — السرد الإبداعي العربي</div>',
                unsafe_allow_html=True)

    if not _FABLE_OK:
        st.error("⚠️ تعذّر تحميل محرك السرد الإبداعي (ai/fable_engine.py). "
                  "تأكد من رفع الملف إلى مجلد ai/.")
        return

    st.markdown(
        '<p style="color:var(--text-muted)">اختر وضع القصة والراوي، وابدأ حكاية تفاعلية '
        'تتطور حسب اختياراتك، أو اطلب قصيدة على أحد بحور الشعر العربي.</p>',
        unsafe_allow_html=True,
    )

    # ── تهيئة محرك السرد مرة واحدة لكل جلسة Streamlit ──
    if "fable_engine" not in st.session_state:
        fb = _FableLLMFallback(model_key="fable")
        st.session_state.fable_engine = FableEngine(
            llm_fallback=fb, db_path=str(MEMORY_DIR / "fable.db")
        )
        st.session_state.fable_chapter = None   # آخر فصل مُولَّد

    engine = st.session_state.fable_engine

    story_tab, poem_tab, explainer_tab, shorts_tab, library_tab = st.tabs(
        ["📖 قصة تفاعلية", "🪶 توليد شعر", "🎬 وثائقي (سيناريو)", "⚡ Shorts (سيناريو)", "📚 مكتبة القصص"]
    )

    # ══════════════════ قصة تفاعلية ══════════════════
    with story_tab:
        cur = st.session_state.fable_chapter

        if cur is None:
            c1, c2 = st.columns(2)
            with c1:
                mode = st.selectbox(
                    "وضع القصة",
                    list(STORY_MODES.keys()),
                    index=list(STORY_MODES.keys()).index(FABLE_DEFAULT_MODE),
                    format_func=lambda m: f"{STORY_MODES[m]['emoji']} {m} — {STORY_MODES[m]['desc']}",
                )
            with c2:
                character = st.selectbox(
                    "الراوي / الأسلوب",
                    list(CHARACTERS.keys()),
                    index=list(CHARACTERS.keys()).index(FABLE_DEFAULT_CHARACTER),
                    format_func=lambda c: f"{CHARACTERS[c]['emoji']} {c} — {CHARACTERS[c]['style']}",
                )
            target_value = None
            if mode == "قصص إسلامية تربوية":
                target_value = st.selectbox(
                    "🕌 القيمة المستهدفة",
                    ISLAMIC_VALUES,
                    help="اختر القيمة أو الخُلق الذي تريد أن تتعلّمه القصة للطفل — "
                         "يمكنك أيضاً إضافة تفاصيل حرة في الحقل أدناه.",
                )
            seed = st.text_input(
                "فكرة مبدئية (اختياري):" if target_value is None
                else "تفاصيل إضافية عن القصة (اختياري):",
                placeholder="مثال: قصة عن تاجر يبحث عن كنز مفقود في الصحراء" if target_value is None
                else "مثال: طفل يجد محفظة نقود في الحديقة",
            )
            if st.button("✨ ابدأ القصة", type="primary"):
                _story_skel_ph = st.empty()
                with _story_skel_ph.container():
                    _skeleton(lines=6)
                effective_seed = seed
                if target_value is not None:
                    effective_seed = (
                        f"اكتب قصة تُعلّم الطفل قيمة «{target_value}»."
                        + (f" تفاصيل إضافية: {seed.strip()}" if seed.strip() else "")
                    )
                try:
                    chapter = engine.start_story(mode=mode, character=character, seed_idea=effective_seed)
                except Exception as e:  # noqa: BLE001
                    _story_skel_ph.empty()
                    st.error(f"⚠️ تعذّر بدء القصة، حاول مرة أخرى. (تفصيل تقني: {e})")
                else:
                    st.session_state.fable_chapter = chapter
                    st.rerun()
        else:
            # ── عرض الفصل الحالي ──
            mode_info = STORY_MODES.get(cur.mode, {})
            char_info = CHARACTERS.get(cur.character, {})
            st.markdown(
                f'<span class="badge badge-purple">{mode_info.get("emoji","")} {cur.mode}</span> '
                f'<span class="badge badge-blue">{char_info.get("emoji","")} {cur.character}</span> '
                f'<span class="badge badge-green">المزوّد: {cur.provider}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class="root-item" style="font-size:1.05rem; line-height:2; text-align:right; direction:rtl">
                {cur.text}
            </div>
            """, unsafe_allow_html=True)
            _cc1, _cc2, _cc3 = st.columns(3)
            with _cc1:
                _copy_button(cur.text, key="fable_chapter")
            with _cc2:
                try:
                    _full_story_rows = engine.memory.get_history(cur.session_id, limit=500)
                    _full_story_text = "\n\n".join(
                        r["content"] for r in _full_story_rows if r["role"] == "narration"
                    )
                except Exception:  # noqa: BLE001
                    _full_story_text = cur.text
                st.download_button(
                    "⬇️ تحميل القصة كاملة",
                    data=_full_story_text,
                    file_name="قصتي.txt",
                    mime="text/plain",
                    key="fable_story_download",
                    use_container_width=True,
                )
            with _cc3:
                if _PDF_EXPORT_OK:
                    try:
                        _story_pdf_bytes = _story_to_pdf(
                            title="قصتي", mode=cur.mode, character=cur.character,
                            full_text=_full_story_text,
                        )
                    except Exception as e:  # noqa: BLE001
                        _story_pdf_bytes = None
                        st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                    if _story_pdf_bytes:
                        st.download_button(
                            "📄 تحميل PDF",
                            data=_story_pdf_bytes,
                            file_name="قصتي.pdf",
                            mime="application/pdf",
                            key="fable_story_pdf_download",
                            use_container_width=True,
                        )

            if cur.error:
                st.caption(f"⚠️ ملاحظة تقنية: {cur.error}")

            st.markdown("**ماذا يحدث بعد ذلك؟**")
            cols = st.columns(len(cur.choices) or 1)
            chosen = None
            for i, choice in enumerate(cur.choices):
                with cols[i]:
                    if st.button(choice, key=f"fable_choice_{i}", use_container_width=True):
                        chosen = choice

            custom_choice = st.text_input("أو اكتب مسارك الخاص:", key="fable_custom_choice")
            if st.button("➡️ تابع") and custom_choice.strip():
                chosen = custom_choice.strip()

            if chosen:
                _story_cont_skel_ph = st.empty()
                with _story_cont_skel_ph.container():
                    _skeleton(lines=6)
                try:
                    next_chapter = engine.continue_story(cur.session_id, chosen)
                except Exception as e:  # noqa: BLE001
                    _story_cont_skel_ph.empty()
                    st.error(f"⚠️ تعذّر متابعة القصة، حاول مرة أخرى. (تفصيل تقني: {e})")
                else:
                    st.session_state.fable_chapter = next_chapter
                    st.session_state.fable_qc_result = None
                    st.rerun()

            st.markdown("---")
            st.markdown("**أوامر سريعة:**")
            qc_cols = st.columns(4)
            if cur.mode == "قصص إسلامية تربوية":
                quick_labels = ["أضف عبرة", "صف المكان", "أضف حواراً", "لخّص"]
            else:
                quick_labels = ["أنشد بيتاً", "صف المكان", "أضف حواراً", "لخّص"]
            for i, label in enumerate(quick_labels):
                with qc_cols[i]:
                    if st.button(f"⚡ {label}", key=f"fable_qc_{i}", use_container_width=True):
                        with st.spinner("..."):
                            try:
                                qc_result = engine.quick_command(cur.session_id, label)
                                st.session_state.fable_qc_result = (label, qc_result.text, None)
                            except Exception as e:  # noqa: BLE001
                                st.session_state.fable_qc_result = (label, "", str(e))
                        st.rerun()

            _qc = st.session_state.get("fable_qc_result")
            if _qc:
                _qc_label, _qc_text, _qc_err = _qc
                if _qc_err:
                    st.error(f"⚠️ تعذّر تنفيذ «{_qc_label}»، حاول مرة أخرى. (تفصيل تقني: {_qc_err})")
                else:
                    st.markdown(
                        f'<span class="badge badge-blue">⚡ {_qc_label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"""
                    <div class="root-item" style="text-align:right; direction:rtl">
                        {_qc_text}
                    </div>
                    """, unsafe_allow_html=True)
                    _copy_button(_qc_text, key="fable_qc")

            if st.button("🔄 قصة جديدة"):
                st.session_state.fable_chapter = None
                st.session_state.fable_qc_result = None
                st.rerun()

    # ══════════════════ توليد شعر ══════════════════
    with poem_tab:
        st.markdown("**اطلب قصيدة قصيرة على أحد بحور الشعر العربي:**")
        topic = st.text_input("موضوع القصيدة:", placeholder="مثال: الوفاء، الوطن، الصحراء ليلاً")
        meter = st.selectbox(
            "البحر الشعري",
            list(ARABIC_METERS.keys()),
            format_func=lambda m: f"{m} — {ARABIC_METERS[m]['وصف']}",
        )
        def _run_poem_generation(_topic: str, _meter: str):
            _poem_skel_ph = st.empty()
            with _poem_skel_ph.container():
                _skeleton(lines=5)
            try:
                poem = engine.generate_poem(_topic, meter=_meter)
            except Exception as e:  # noqa: BLE001
                _poem_skel_ph.empty()
                st.session_state.fable_poem_result = None
                st.session_state.fable_poem_error = str(e)
            else:
                _poem_skel_ph.empty()
                st.session_state.fable_poem_result = poem
                st.session_state.fable_poem_error = None
                st.session_state.fable_poem_topic = _topic
                st.session_state.fable_poem_meter = _meter
                st.session_state.fable_poem_audio = None
                st.session_state.fable_poem_audio_error = None

        if st.button("🪶 أنشئ القصيدة", type="primary"):
            if not topic.strip():
                st.warning("⚠️ الرجاء كتابة موضوع القصيدة أولاً.")
            else:
                _run_poem_generation(topic.strip(), meter)

        _poem_err = st.session_state.get("fable_poem_error")
        if _poem_err:
            st.error(f"⚠️ تعذّر توليد القصيدة، حاول مرة أخرى. (تفصيل تقني: {_poem_err})")

        poem = st.session_state.get("fable_poem_result")
        if poem is not None:
            st.toast("✅ القصيدة جاهزة", icon="🪶")
            st.markdown(f"""
            <div class="root-item" style="font-size:1.1rem; line-height:2.1; text-align:center; direction:rtl">
                {poem.text}
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"المزوّد: {poem.provider}")
            _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns(5)
            with _pc1:
                _copy_button(poem.text, key="fable_poem")
            with _pc2:
                st.download_button(
                    "⬇️ تحميل كملف نصي",
                    data=poem.text,
                    file_name="قصيدة.txt",
                    mime="text/plain",
                    key="fable_poem_download",
                    use_container_width=True,
                )
            with _pc5:
                if _PDF_EXPORT_OK:
                    try:
                        _poem_pdf_bytes = _poem_to_pdf(
                            title="قصيدتي", topic=topic, meter=meter, poem_text=poem.text,
                        )
                    except Exception as e:  # noqa: BLE001
                        _poem_pdf_bytes = None
                        st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                    if _poem_pdf_bytes:
                        st.download_button(
                            "📄 تحميل PDF",
                            data=_poem_pdf_bytes,
                            file_name="قصيدتي.pdf",
                            mime="application/pdf",
                            key="fable_poem_pdf_download",
                            use_container_width=True,
                        )
            with _pc3:
                if st.button("🔄 أعد التوليد", key="fable_poem_regenerate", use_container_width=True):
                    _run_poem_generation(
                        st.session_state.get("fable_poem_topic", topic.strip() or "موضوع حر"),
                        st.session_state.get("fable_poem_meter", meter),
                    )
                    st.rerun()
            with _pc4:
                if st.button("🔊 استمع", key="fable_poem_listen", use_container_width=True, disabled=not _TTS_OK):
                    with st.spinner("⟳ جارٍ تحويل القصيدة لصوت..."):
                        try:
                            _poem_tts = _TTSEngineCls().synthesize(poem.text)
                        except Exception as e:  # noqa: BLE001
                            st.session_state.fable_poem_audio = None
                            st.session_state.fable_poem_audio_error = str(e)
                        else:
                            if _poem_tts.ok:
                                import base64 as _b64_poem
                                st.session_state.fable_poem_audio = (
                                    _b64_poem.b64encode(_poem_tts.audio_bytes).decode("ascii"),
                                    _poem_tts.format,
                                )
                                st.session_state.fable_poem_audio_error = None
                            else:
                                st.session_state.fable_poem_audio = None
                                st.session_state.fable_poem_audio_error = _poem_tts.error or "تعذّر توليد الصوت"

            _poem_audio_err = st.session_state.get("fable_poem_audio_error")
            if _poem_audio_err:
                st.error(f"⚠️ تعذّر توليد الصوت. (تفصيل تقني: {_poem_audio_err})")

            _poem_audio = st.session_state.get("fable_poem_audio")
            if _poem_audio:
                _a_b64, _a_fmt = _poem_audio
                st.markdown(
                    f'<audio controls style="width:100%;margin-top:0.5rem" '
                    f'src="data:audio/{_a_fmt};base64,{_a_b64}"></audio>',
                    unsafe_allow_html=True,
                )

    # ══════════════════ وثائقي (سيناريو Explainer) ══════════════════
    with explainer_tab:
        st.markdown(
            '<p style="color:var(--text-muted)">يولّد سيناريو وثائقياً مُقسّماً إلى مشاهد '
            '(نص السرد + توجيه مرئي مقترح لكل مشهد) — فكرة مستوحاة من أدوات '
            'مثل Higgsfield Explainer. <strong>ملاحظة:</strong> NSM لا يملك '
            'نموذج توليد فيديو فعلي، لذا الناتج هنا نص سيناريو فقط جاهز '
            'لتُغذّى به يدوياً أي أداة توليد فيديو خارجية.</p>',
            unsafe_allow_html=True,
        )
        topic = st.text_input(
            "موضوع الوثائقي:",
            placeholder="مثال: تاريخ طريق الحرير، كيف تعمل الأقمار الصناعية",
            key="explainer_topic",
        )
        minutes = st.slider("المدة المستهدفة (دقائق)", min_value=1, max_value=10, value=5)

        if st.button("🎬 أنشئ السيناريو", type="primary"):
            if not topic.strip():
                st.warning("⚠️ الرجاء كتابة موضوع الوثائقي أولاً.")
                st.session_state.explainer_script = None
            else:
                with st.spinner("يُجري بحثاً ويكتب السيناريو..."):
                    try:
                        st.session_state.explainer_script = engine.generate_explainer(
                            topic.strip(), target_minutes=minutes
                        )
                        st.session_state.explainer_error = None
                    except Exception as e:  # noqa: BLE001
                        st.session_state.explainer_script = None
                        st.session_state.explainer_error = str(e)

        _explainer_err = st.session_state.get("explainer_error")
        if _explainer_err:
            st.error(f"⚠️ تعذّر إنشاء السيناريو، حاول مرة أخرى. (تفصيل تقني: {_explainer_err})")

        script = st.session_state.get("explainer_script")
        if script is not None:
            st.markdown(f"### {script.title}")
            st.caption(
                f"عدد المشاهد: {len(script.segments)} · "
                f"إجمالي المدة التقديرية: ~{script.total_seconds // 60} دقيقة "
                f"({script.total_seconds} ثانية) · المزوّد: {script.provider}"
            )
            if script.error:
                st.caption(f"⚠️ ملاحظة تقنية: {script.error}")

            for seg in script.segments:
                st.markdown(f"""
                <div class="root-item" style="text-align:right; direction:rtl">
                    <span class="badge badge-purple">المشهد {seg.index}</span>
                    <span class="badge badge-amber">~{seg.est_seconds} ثانية</span>
                    <p style="margin-top:0.5rem"><strong>السرد:</strong> {seg.narration}</p>
                    <p style="color:var(--text-muted)"><strong>🎥 اللقطة المقترحة:</strong> {seg.visual_notes}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 النص الكامل للسرد (لنسخه إلى أداة التعليق الصوتي)"):
                st.text_area("النص الكامل:", value=script.full_narration, height=200)
                _ec1, _ec2 = st.columns(2)
                with _ec1:
                    st.download_button(
                        "⬇️ تحميل السيناريو كملف نصي",
                        data=script.full_narration,
                        file_name=f"{(script.title or 'سيناريو')[:40]}.txt",
                        mime="text/plain",
                        key="explainer_download",
                        use_container_width=True,
                    )
                with _ec2:
                    if _PDF_EXPORT_OK:
                        try:
                            _explainer_pdf_bytes = _script_to_pdf(
                                title=script.title, format_label=script.format,
                                segments=[
                                    {"index": s.index, "narration": s.narration,
                                     "visual_notes": s.visual_notes, "est_seconds": s.est_seconds}
                                    for s in script.segments
                                ],
                                total_seconds=script.total_seconds,
                            )
                        except Exception as e:  # noqa: BLE001
                            _explainer_pdf_bytes = None
                            st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                        if _explainer_pdf_bytes:
                            st.download_button(
                                "📄 تحميل السيناريو PDF",
                                data=_explainer_pdf_bytes,
                                file_name=f"{(script.title or 'سيناريو')[:40]}.pdf",
                                mime="application/pdf",
                                key="explainer_pdf_download",
                                use_container_width=True,
                            )

    # ══════════════════ ⚡ Shorts (فيديو قصير عمودي) ══════════════════
    with shorts_tab:
        st.markdown(
            '<p style="color:var(--text-muted)">يحوّل نصاً أو موضوعاً إلى فيديو '
            'قصير عمودي فعلي (~دقيقة واحدة) بسرد صوتي ورسوم متحركة نصية '
            '(Kinetic Typography) — فكرة مستوحاة من ميزة NotebookLM: Shorts، '
            'مع رندر mp4 حقيقي داخل المشروع (بدون أدوات خارجية).</p>',
            unsafe_allow_html=True,
        )
        source_text = st.text_area(
            "الصق مصدرك أو اكتب الموضوع:",
            placeholder="مثال: فقرة من مقال، ملخص بحث، أو مجرد فكرة موضوع قصير",
            key="shorts_source",
            height=120,
        )
        target_sec = st.slider("المدة المستهدفة (ثانية)", min_value=20, max_value=90, value=60, step=5)

        if st.button("⚡ أنشئ سيناريو Shorts", type="primary"):
            if not source_text.strip():
                st.warning("⚠️ الرجاء لصق نص أو كتابة موضوع أولاً.")
            else:
                with st.spinner("يُلخّص ويكتب لقطات سريعة..."):
                    try:
                        st.session_state.shorts_script = engine.generate_short(
                            source_text.strip(), target_seconds=target_sec
                        )
                        st.session_state.shorts_error = None
                    except Exception as e:  # noqa: BLE001
                        st.session_state.shorts_script = None
                        st.session_state.shorts_error = str(e)

        _shorts_err = st.session_state.get("shorts_error")
        if _shorts_err:
            st.error(f"⚠️ تعذّر إنشاء سيناريو Shorts، حاول مرة أخرى. (تفصيل تقني: {_shorts_err})")

        short = st.session_state.get("shorts_script")
        if short is not None:
            st.markdown(f"### {short.title}")
            st.caption(
                f"عدد اللقطات: {len(short.segments)} · "
                f"إجمالي المدة التقديرية: ~{short.total_seconds} ثانية · "
                f"المزوّد: {short.provider}"
            )
            if short.error:
                st.caption(f"⚠️ ملاحظة تقنية: {short.error}")

            for seg in short.segments:
                st.markdown(f"""
                <div class="root-item" style="text-align:right; direction:rtl">
                    <span class="badge badge-purple">لقطة {seg.index}</span>
                    <span class="badge badge-amber">~{seg.est_seconds} ثانية</span>
                    <p style="margin-top:0.5rem"><strong>السرد:</strong> {seg.narration}</p>
                    <p style="color:var(--text-muted)"><strong>🎞️ رسم متحرك مقترح:</strong> {seg.visual_notes}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 النص الكامل للسرد"):
                st.text_area("النص الكامل:", value=short.full_narration, height=150, key="shorts_full_text")
                _sc1, _sc2 = st.columns(2)
                with _sc1:
                    st.download_button(
                        "⬇️ تحميل السيناريو كملف نصي",
                        data=short.full_narration,
                        file_name=f"{(short.title or 'shorts')[:40]}.txt",
                        mime="text/plain",
                        key="shorts_download",
                        use_container_width=True,
                    )
                with _sc2:
                    if _PDF_EXPORT_OK:
                        try:
                            _shorts_pdf_bytes = _script_to_pdf(
                                title=short.title, format_label=short.format,
                                segments=[
                                    {"index": s.index, "narration": s.narration,
                                     "visual_notes": s.visual_notes, "est_seconds": s.est_seconds}
                                    for s in short.segments
                                ],
                                total_seconds=short.total_seconds,
                            )
                        except Exception as e:  # noqa: BLE001
                            _shorts_pdf_bytes = None
                            st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                        if _shorts_pdf_bytes:
                            st.download_button(
                                "📄 تحميل السيناريو PDF",
                                data=_shorts_pdf_bytes,
                                file_name=f"{(short.title or 'shorts')[:40]}.pdf",
                                mime="application/pdf",
                                key="shorts_pdf_download",
                                use_container_width=True,
                            )

            st.divider()
            st.markdown("#### 🎬 رندر الفيديو الفعلي (mp4)")

            _VOICE_OPTIONS = {
                "🎙️ افتراضي (تلقائي حسب المزوّد المتاح)": "",
                "👨 حامد — سعودي (Edge, مجاني)": "ar-SA-HamedNeural",
                "👩 زارية — سعودية (Edge, مجاني)": "ar-SA-ZariyahNeural",
                "👨 شاكر — مصري (Edge, مجاني)": "ar-EG-ShakirNeural",
                "👩 سلمى — مصرية (Edge, مجاني)": "ar-EG-SalmaNeural",
                "👨 حمدان — إماراتي (Edge, مجاني)": "ar-AE-HamdanNeural",
                "👩 فاطمة — إماراتية (Edge, مجاني)": "ar-AE-FatimaNeural",
                "✨ Kore — Gemini TTS (يتطلب GOOGLE_API_KEY)": "Kore",
            }
            selected_voice_label = st.selectbox(
                "🗣️ اختر الصوت",
                options=list(_VOICE_OPTIONS.keys()),
                key="shorts_voice_select",
                help="الأصوات المجانية (Edge) لا تحتاج أي مفتاح API. صوت Gemini يحتاج GOOGLE_API_KEY في البيئة.",
            )
            selected_voice = _VOICE_OPTIONS[selected_voice_label]

            _hf_key_present = bool(os.getenv("HIGGSFIELD_API_KEY", "").strip())
            use_cinematic_bg = st.checkbox(
                "🎥 خلفيات سينمائية حقيقية (بدل التدرّج اللوني الافتراضي)",
                value=False,
                key="shorts_cinematic_bg_toggle",
                help="يستبدل الخلفية المتدرّجة الافتراضية بخلفية فيديو حقيقية لكل مشهد.",
            )
            cinematic_provider = "higgsfield"
            if use_cinematic_bg:
                _shorts_provider_options = [
                    "🆓 Wan2.1 مجاني ⚡ Running on Zero (GPU حقيقي مجاني)",
                    "💳 Higgsfield (مدفوع — أسرع وأدق، بجودة National Geographic)"
                    + ("" if _hf_key_present else " 🔒"),
                ]
                _shorts_provider_label = st.radio(
                    "المزوّد",
                    options=_shorts_provider_options,
                    index=0,
                    key="shorts_cinematic_provider_radio",
                    horizontal=True,
                    help=(
                        "🆓 Wan2.1 مجاني: يشتغل فعلياً على GPU A100 مجاني عبر "
                        "Hugging Face ZeroGPU (مساحات مُوسومة رسمياً \"Running "
                        "on Zero\" على Hugging Face — ليست محاكاة)، بدون أي "
                        "تكلفة وبدون أي مفتاح إلزامي. أبطأ بكثير (طابور GPU "
                        "مشترك) وقد يتعطّل أحياناً؛ عند فشله يتراجع تلقائياً "
                        "للخلفية المتدرّجة لنفس المشهد فقط. HF_TOKEN اختياري "
                        "لتحسين حد الاستخدام. (يُجرَّب LTX-Video أولاً ثم "
                        "Wan2.2 ثم Wan2.1 تلقائياً حتى ينجح أحدها.)"
                        "\n\n💳 Higgsfield: مزوّد مدفوع، يستهلك رصيدك لكل "
                        "مشهد، أسرع وأدق. يتطلب HIGGSFIELD_API_KEY."
                        + ("" if _hf_key_present else " (🔒 المفتاح غير موجود بالبيئة حالياً)")
                    ),
                )
                cinematic_provider = (
                    "wan_free" if "Wan2.1" in _shorts_provider_label else "higgsfield"
                )
                if cinematic_provider == "wan_free":
                    st.markdown(
                        '<div style="margin:0.3rem 0 0.6rem;">'
                        '<span class="badge badge-green">🟢 Running on Zero</span> '
                        '<span class="badge badge-blue" style="margin-right:6px;">'
                        "GPU A100 مجاني حقيقي — Hugging Face ZeroGPU</span></div>",
                        unsafe_allow_html=True,
                    )
                    _render_wan_free_status_widget("shorts")
                elif not _hf_key_present:
                    st.warning(
                        "⚠️ HIGGSFIELD_API_KEY غير موجود بالبيئة حالياً — أضِفه "
                        "بإعدادات Secrets، أو اختر «🆓 Wan2.1 مجاني» بالأعلى "
                        "لمتابعة العمل بدون أي مفتاح."
                    )

            if st.button("🎬 أنشئ الفيديو الآن", type="primary", key="shorts_render_video_btn"):
                try:
                    _spinner_msg = (
                        "⏳ يولّد السرد الصوتي والخلفيات السينمائية ثم يركّب الفيديو... "
                        "قد يستغرق عدة دقائق"
                        if use_cinematic_bg else
                        "⏳ يولّد السرد الصوتي ثم يركّب الفيديو... قد يستغرق دقيقة"
                    )
                    with st.spinner(_spinner_msg):
                        mp4_bytes = engine.render_video(
                            short, voice=selected_voice,
                            use_cinematic_backgrounds=use_cinematic_bg,
                            cinematic_provider=cinematic_provider,
                            wan_skip_spaces=st.session_state.get("shorts_wan_dead_spaces"),
                        )
                    st.session_state.shorts_mp4 = mp4_bytes
                    st.success("✅ تم إنتاج الفيديو")
                except Exception as e:  # noqa: BLE001
                    st.error(f"⚠️ فشل رندر الفيديو: {e}")

            mp4_bytes = st.session_state.get("shorts_mp4")
            if mp4_bytes:
                st.video(mp4_bytes)
                _shorts_dl_cols = st.columns(2)
                with _shorts_dl_cols[0]:
                    st.download_button(
                        "⬇️ تحميل الفيديو (mp4)",
                        data=mp4_bytes,
                        file_name=f"{short.title[:40] or 'short'}.mp4",
                        mime="video/mp4",
                        key="shorts_download_mp4",
                    )
                with _shorts_dl_cols[1]:
                    # ── ملف ترجمة SRT — راجع نفس الشرح بتبويب Higgsfield
                    #    Explainer أعلاه (ai.video_engine.build_srt).
                    try:
                        from ai.video_engine import build_srt as _shorts_build_srt
                        _shorts_srt_text = _shorts_build_srt(short)
                        st.download_button(
                            "📝 تحميل الترجمة (SRT)",
                            data=_shorts_srt_text.encode("utf-8"),
                            file_name=f"{short.title[:40] or 'short'}.srt",
                            mime="text/srt",
                            key="shorts_download_srt",
                        )
                    except Exception as _srt_err:  # noqa: BLE001
                        logger.debug("تعذّر بناء ملف SRT لـShorts: %s", _srt_err)

                st.markdown("---")
                st.markdown("#### 📤 مشاركة اجتماعية فعلية (رفع الفيديو)")
                try:
                    from ai.social_platforms import YouTubeAdapter, TikTokAdapter
                except ImportError as e:  # noqa: BLE001
                    st.caption(f"⚠️ تعذّر تحميل محولات المشاركة: {e}")
                else:
                    yt = YouTubeAdapter()
                    tk = TikTokAdapter()
                    share_cols = st.columns(2)

                    # ── يوتيوب ──
                    with share_cols[0]:
                        st.markdown("**▶️ YouTube**")
                        yt_ready = yt.is_configured() and yt._can_write()
                        if not yt_ready:
                            missing = yt.missing_env() or yt.write_env
                            st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(missing))
                        else:
                            yt_title = st.text_input(
                                "العنوان:", value=short.title[:100], key="yt_upload_title"
                            )
                            yt_privacy = st.selectbox(
                                "الخصوصية:", ["private", "unlisted", "public"],
                                key="yt_upload_privacy",
                            )
                            if st.button("▶️ ارفع على يوتيوب", key="yt_upload_btn", use_container_width=True):
                                try:
                                    with st.spinner("⏳ يرفع الفيديو على يوتيوب (Resumable Upload)..."):
                                        video_id = yt.upload_video(
                                            mp4_bytes,
                                            title=yt_title,
                                            description=short.full_narration[:4500],
                                            privacy_status=yt_privacy,
                                        )
                                    st.success(f"✅ تم الرفع! الرابط: https://youtu.be/{video_id}")
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"⚠️ فشل الرفع على يوتيوب: {e}")

                    # ── تيك توك ──
                    with share_cols[1]:
                        st.markdown("**🎵 TikTok**")
                        tk_ready = tk.is_configured()
                        if not tk_ready:
                            st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(tk.missing_env()))
                        else:
                            st.caption(
                                "ℹ️ التطبيقات غير المدقَّقة من TikTok تنشر كـ«خاص بحسابك فقط» "
                                "(مسودة للمراجعة) حتى يجتاز التطبيق مراجعة TikTok الرسمية للنشر العام."
                            )
                            tk_title = st.text_input(
                                "العنوان:", value=short.title[:150], key="tk_upload_title"
                            )
                            if st.button("🎵 ارفع على تيك توك", key="tk_upload_btn", use_container_width=True):
                                try:
                                    with st.spinner("⏳ يرفع الفيديو على تيك توك..."):
                                        publish_id = tk.upload_video(mp4_bytes, title=tk_title)
                                    st.success(
                                        f"✅ تم إرسال الفيديو (publish_id: {publish_id}) — "
                                        "افتح تطبيق TikTok للتأكد من ظهوره ضمن المسودات/المنشورات."
                                    )
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"⚠️ فشل الرفع على تيك توك: {e}")

    # ══════════════════ مكتبة القصص المحفوظة ══════════════════
    with library_tab:
        st.markdown(
            '<p style="color:var(--text-muted)">كل قصة تفاعلية تُحفظ تلقائياً في قاعدة بيانات SQLite محلية '
            '(<code>memory/fable.db</code>) — هذه الواجهة تستعرضها.</p>',
            unsafe_allow_html=True,
        )

        try:
            sessions = engine.memory.list_recent_sessions(limit=100)
        except Exception as e:  # noqa: BLE001
            sessions = []
            st.error(f"⚠️ تعذّر قراءة مكتبة القصص: {e}")

        if not sessions:
            st.info(
                "📭 لا توجد قصص محفوظة بعد. ابدأ قصة من تبويب «📖 قصة تفاعلية» "
                "وستظهر هنا تلقائياً بمجرد إنشاء الفصل الأول."
            )
        else:
            _lib_modes_present = sorted({s["mode"] for s in sessions if s["mode"]})
            _lib_filter = st.multiselect(
                "🔎 فلترة حسب النمط:",
                options=_lib_modes_present,
                format_func=lambda m: f"{STORY_MODES.get(m, {}).get('emoji', '📖')} {m}",
                key="lib_mode_filter",
                placeholder="كل الأنماط",
            )
            if _lib_filter:
                sessions = [s for s in sessions if s["mode"] in _lib_filter]

            st.caption(f"📚 عدد القصص المعروضة: {len(sessions)}")
            for sess in sessions:
                session_id = sess["session_id"]
                mode = sess["mode"]
                character = sess["character"]
                mode_info = STORY_MODES.get(mode, {})
                char_info = CHARACTERS.get(character, {})
                try:
                    created_label = datetime.fromtimestamp(sess["created_at"]).strftime("%Y-%m-%d %H:%M")
                except Exception:  # noqa: BLE001
                    created_label = ""

                try:
                    preview_text, chapter_count = engine.memory.get_narration_preview(session_id)
                except Exception:  # noqa: BLE001
                    preview_text, chapter_count = "", 0
                preview = (
                    (preview_text[:90] + "…") if preview_text and len(preview_text) > 90
                    else (preview_text or "(لا يوجد نص بعد)")
                )

                header = (
                    f"{mode_info.get('emoji', '📖')} {mode} · "
                    f"{char_info.get('emoji', '')} {character} — {created_label}"
                )
                with st.expander(header):
                    st.caption(f"🆔 {session_id} · عدد الفصول: {chapter_count}")
                    st.markdown(
                        f"<p style='direction:rtl; text-align:right; color:var(--text-muted)'>{preview}</p>",
                        unsafe_allow_html=True,
                    )

                    view_key = f"lib_expand_{session_id}"
                    confirm_key = f"lib_confirm_delete_{session_id}"
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("📖 عرض القصة كاملة", key=f"lib_view_btn_{session_id}", use_container_width=True):
                            st.session_state[view_key] = not st.session_state.get(view_key, False)
                    with col_b:
                        if st.button("▶️ استأنف هذه القصة", key=f"lib_resume_btn_{session_id}", use_container_width=True):
                            try:
                                last_narration = engine.memory.get_last_narration(session_id)
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ تعذّر تحميل القصة. (تفصيل تقني: {e})")
                            else:
                                st.session_state.fable_chapter = FableChapter(
                                    session_id=session_id,
                                    text=last_narration,
                                    choices=[],
                                    mode=mode,
                                    character=character,
                                    provider="محفوظ من المكتبة",
                                )
                                st.success("✅ تم تحميل القصة — افتح تبويب «📖 قصة تفاعلية» للمتابعة منها.")
                                st.rerun()
                    with col_c:
                        if st.button("🗑️ حذف", key=f"lib_delete_btn_{session_id}", use_container_width=True):
                            st.session_state[confirm_key] = True

                    if st.session_state.get(confirm_key):
                        st.warning("⚠️ هل أنت متأكد من حذف هذه القصة نهائياً؟ لا يمكن التراجع عن هذا الإجراء.")
                        _dc1, _dc2 = st.columns(2)
                        with _dc1:
                            if st.button("✅ نعم، احذفها نهائياً", key=f"lib_confirm_yes_{session_id}", use_container_width=True):
                                try:
                                    engine.memory.delete_session(session_id)
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"⚠️ تعذّر حذف القصة. (تفصيل تقني: {e})")
                                else:
                                    st.session_state[confirm_key] = False
                                    st.success("✅ تم حذف القصة.")
                                    st.rerun()
                        with _dc2:
                            if st.button("إلغاء", key=f"lib_confirm_no_{session_id}", use_container_width=True):
                                st.session_state[confirm_key] = False
                                st.rerun()

                    if st.session_state.get(view_key):
                        try:
                            history_rows = engine.memory.get_history(session_id, limit=500)
                            narrations = [r["content"] for r in history_rows if r["role"] == "narration"]
                            full_text = "\n\n".join(narrations) if narrations else "(لا يوجد نص محفوظ)"
                        except Exception as e:  # noqa: BLE001
                            full_text = f"⚠️ تعذّر تحميل النص الكامل. (تفصيل تقني: {e})"
                        st.markdown(f"""
                        <div class="root-item" style="text-align:right; direction:rtl; line-height:2">
                            {full_text}
                        </div>
                        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-header">🎬 سيناريوهات Shorts/الوثائقي المحفوظة</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">كل سيناريو مولَّد من تبويبَي 🎤 وثائقي و🎬 Shorts '
            'يُحفظ تلقائياً هنا (بدون الصوت/الفيديو) — يمكنك إعادة استخدامه لرندر فيديو جديد '
            'دون توليد سيناريو جديد (يوفّر استدعاء LLM).</p>',
            unsafe_allow_html=True,
        )
        try:
            shorts_history = engine.memory.list_recent_shorts(limit=30)
        except Exception as e:  # noqa: BLE001
            shorts_history = []
            st.error(f"⚠️ تعذّر قراءة سيناريوهات Shorts المحفوظة: {e}")

        if not shorts_history:
            st.info("📭 لا توجد سيناريوهات محفوظة بعد. أنشئ واحداً من تبويب «🎤 وثائقي» أو «🎬 Shorts».")
        else:
            for sh_row in shorts_history:
                sh_id = sh_row["id"]
                sh_emoji = "🎬" if sh_row["format"] == "شورت" else "🎤"
                try:
                    sh_created = datetime.fromtimestamp(sh_row["created_at"]).strftime("%Y-%m-%d %H:%M")
                except Exception:  # noqa: BLE001
                    sh_created = ""
                sh_header = f"{sh_emoji} {sh_row['title']} · ~{sh_row['total_seconds']} ثانية — {sh_created}"
                with st.expander(sh_header):
                    if sh_row["source_excerpt"]:
                        st.caption(f"المصدر: {sh_row['source_excerpt'][:150]}")
                    sh_col_a, sh_col_b = st.columns(2)
                    with sh_col_a:
                        if st.button("📂 استخدم هذا السيناريو", key=f"lib_shorts_use_{sh_id}", use_container_width=True):
                            try:
                                _segs_data = json.loads(sh_row["segments_json"])
                                _rebuilt_segments = [
                                    ExplainerSegment(
                                        index=s["index"], narration=s["narration"],
                                        visual_notes=s["visual_notes"], est_seconds=s["est_seconds"],
                                    ) for s in _segs_data
                                ]
                                st.session_state.shorts_script = ExplainerScript(
                                    topic=sh_row["source_excerpt"], title=sh_row["title"],
                                    segments=_rebuilt_segments, provider="محفوظ من المكتبة",
                                    format=sh_row["format"],
                                )
                                st.session_state.shorts_mp4 = None  # فيديو جديد يحتاج رندر من جديد
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ تعذّر تحميل السيناريو. (تفصيل تقني: {e})")
                            else:
                                st.success("✅ تم تحميل السيناريو — افتح تبويب «🎬 Shorts» لرندر الفيديو.")
                                st.rerun()
                    with sh_col_b:
                        if st.button("🗑️ حذف", key=f"lib_shorts_delete_{sh_id}", use_container_width=True):
                            try:
                                engine.memory.delete_short(sh_id)
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ تعذّر الحذف. (تفصيل تقني: {e})")
                            else:
                                st.success("✅ تم الحذف.")
                                st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# تبويب 🌐 ترجمة فورية — يستخدم نفس سلسلة LLMFallback الموجودة (Anthropic →
# Cloudflare → Gemini → OpenRouter → Groq...) فلا حاجة لمفتاح Google
# Translate/DeepL منفصل — النماذج اللغوية نفسها مترجم دقيق بما يكفي.
# ══════════════════════════════════════════════════════════════════════════

_TRANSLATE_LANGS = {
    "🌐 اكتشاف تلقائي": "auto",
    "🇸🇦 العربية": "العربية",
    "🇬🇧 الإنجليزية": "الإنجليزية",
    "🇫🇷 الفرنسية": "الفرنسية",
    "🇪🇸 الإسبانية": "الإسبانية",
    "🇩🇪 الألمانية": "الألمانية",
    "🇹🇷 التركية": "التركية",
    "🇮🇷 الفارسية": "الفارسية",
    "🇵🇰 الأردية": "الأردية",
    "🇮🇩 الإندونيسية": "الإندونيسية",
    "🇲🇾 الملايوية": "الملايوية",
    "🇮🇳 الهندية": "الهندية",
    "🇷🇺 الروسية": "الروسية",
    "🇨🇳 الصينية": "الصينية",
    "🇧🇩 البنغالية": "البنغالية",
}


def render_translate():
    """تبويب الترجمة الفورية بين العربية ولغات أخرى شائعة لدى مستخدمي NSM،
    عبر نفس سلسلة LLMFallback المستخدمة بباقي النظام (بدون مفتاح API إضافي)."""

    st.markdown('<div class="section-header">🌐 ترجمة فورية</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="tab-intro">ترجمة نص باستخدام نفس نماذج NSM اللغوية '
        '(Anthropic ← Cloudflare ← Gemini ← OpenRouter ← Groq) — بدون حاجة '
        'لأي مفتاح Google Translate أو DeepL.</p>',
        unsafe_allow_html=True,
    )

    # يجب تطبيق أي "إعادة استخدام" من التاريخ *قبل* إنشاء ودجت text_area
    # مباشرة — تعيين session_state[key] بعد إنشاء الودجت بنفس الجولة يرفع
    # StreamlitAPIException.
    if "_tr_pending_reuse" in st.session_state:
        st.session_state["tr_source_text"] = st.session_state.pop("_tr_pending_reuse")

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        src_label = st.selectbox("من لغة:", list(_TRANSLATE_LANGS.keys()), index=0, key="tr_src_lang")
    with c2:
        tgt_label = st.selectbox("إلى لغة:", list(_TRANSLATE_LANGS.keys()), index=2, key="tr_tgt_lang")

    source_text = st.text_area(
        "النص المراد ترجمته:",
        height=150,
        placeholder="اكتب أو الصق النص هنا...",
        key="tr_source_text",
    )

    translate_clicked = st.button(
        "🌐 ترجم الآن", type="primary", key="tr_translate_btn", use_container_width=True,
        disabled=not bool(source_text and source_text.strip()),
    )
    st.markdown('</div>', unsafe_allow_html=True)

    if translate_clicked and not source_text.strip():
        st.warning("أدخل نصاً للترجمة أولاً.")
    elif translate_clicked and source_text.strip():
        src = _TRANSLATE_LANGS[src_label]
        tgt = _TRANSLATE_LANGS[tgt_label]

        if src == tgt and src != "auto":
            st.toast("⚠️ لغة المصدر ولغة الهدف متطابقتان", icon="⚠️")
        else:
            src_instruction = "اكتشف لغة النص تلقائياً ثم" if src == "auto" else f"ترجم من {src} إلى"
            system_prompt = (
                f"أنت مترجم محترف. {src_instruction} {tgt}. "
                "أعد فقط النص المترجم دون أي شرح أو مقدمات أو علامات اقتباس إضافية، "
                "مع الحفاظ على المعنى والأسلوب الأصلي بدقة."
            )
            _tr_skeleton_ph = st.empty()
            with _tr_skeleton_ph.container():
                _skeleton(lines=3)
            try:
                from ai.llm_fallback import LLMFallback
                _tr_llm = LLMFallback(max_tokens=1200, temperature=0.2)
                result = _tr_llm.generate(source_text.strip(), history=[], system_prompt=system_prompt)
                st.session_state.tr_result = result
                _tr_skeleton_ph.empty()
                st.toast("✅ تمت الترجمة بنجاح", icon="✅")
                if result and (result.text or "").strip() and not getattr(result, "error", None):
                    try:
                        from ai.translation_history import get_history
                        get_history().save(
                            src_lang=src_label, tgt_lang=tgt_label,
                            source_text=source_text.strip(), translated_text=result.text,
                            provider=getattr(result.provider, "value", str(result.provider)),
                        )
                    except Exception:
                        pass
            except Exception as e:  # noqa: BLE001
                _tr_skeleton_ph.empty()
                st.toast(f"⚠️ فشلت الترجمة: {e}", icon="⚠️")
                st.session_state.tr_result = None

    result = st.session_state.get("tr_result")
    if result is not None:
        st.markdown("#### 📄 الترجمة")
        st.markdown(f"""
        <div class="root-item" style="text-align:right; direction:rtl; line-height:1.9">
            {result.text}
        </div>
        """, unsafe_allow_html=True)
        provider_label = getattr(result.provider, "value", str(result.provider))
        st.caption(f"المزوّد: {provider_label}" + (f" · ⚠️ {result.error}" if getattr(result, "error", None) else ""))
        _copy_col, _dl_col = st.columns([1, 2])
        with _copy_col:
            _copy_button(result.text, key="tr_result")
        with _dl_col:
            st.download_button(
                "⬇️ تحميل الترجمة (txt)",
                data=result.text,
                file_name="translation.txt",
                mime="text/plain",
                key="tr_download_btn",
            )

    st.markdown("")
    st.markdown('<div class="section-header">🕘 آخر الترجمات</div>', unsafe_allow_html=True)
    try:
        from ai.translation_history import get_history
        _tr_history = get_history().list_recent(limit=15)
    except Exception as e:  # noqa: BLE001
        _tr_history = []
        st.caption(f"⚠️ تعذّر تحميل السجل: {e}")

    if not _tr_history:
        st.caption("📭 لا توجد ترجمات محفوظة بعد — أول ترجمة ناجحة ستظهر هنا تلقائياً.")
    else:
        for _tr_row in _tr_history:
            _tr_id = _tr_row["id"]
            _tr_excerpt = (_tr_row["source_text"] or "")[:60]
            _tr_header = f"{_tr_row['src_lang']} ← {_tr_row['tgt_lang']} — {_tr_excerpt}…"
            with st.expander(_tr_header):
                st.markdown(f"**النص الأصلي:**\n\n{_tr_row['source_text']}")
                st.markdown(f"**الترجمة:**\n\n{_tr_row['translated_text']}")
                _tr_reuse_col, _tr_del_col = st.columns(2)
                with _tr_reuse_col:
                    if st.button("↩️ استخدم هذا النص مجدداً", key=f"tr_reuse_{_tr_id}", use_container_width=True):
                        st.session_state["_tr_pending_reuse"] = _tr_row["source_text"]
                        st.rerun()
                with _tr_del_col:
                    if st.button("🗑️ حذف", key=f"tr_delete_{_tr_id}", use_container_width=True):
                        try:
                            from ai.translation_history import get_history as _gh2
                            _gh2().delete(_tr_id)
                        except Exception:
                            pass
                        st.rerun()


# ══════════════════════════════════════════════════════════════════════════
# تبويب المحادثة الذكية
# ══════════════════════════════════════════════════════════════════════════
def render_chat():
    """تبويب المحادثة الذكية مع ذاكرة السياق"""

    if not _NSM_CHAT_OK:
        st.error(
            "⚠️ تعذّر تحميل NSM Chat. تأكد من وجود nsm_chat.py أو nsm_chat_plus.py "
            "و nsm_memory.py في جذر المشروع (nsm_embedding.npz اختياري — يعمل النظام بدونه)."
        )
        return

    # تهيئة النموذج مرة واحدة
    if "nsm_bot" not in st.session_state:
        with st.spinner("⟳ تحميل محرك المحادثة..."):
            st.session_state.nsm_bot = NSMChat(system_prompt=NSM_SYSTEM_PROMPT)
        st.session_state.nsm_messages = []
        st.session_state.nsm_count    = 0

    bot = st.session_state.nsm_bot

    # CSS خاص بالمحادثة
    st.markdown("""
    <style>
    @keyframes bubbleIn {
        from {opacity:0;transform:translateY(8px) scale(0.985);}
        to   {opacity:1;transform:translateY(0) scale(1);}
    }
    .chat-user {display:flex;justify-content:flex-end;margin:0.55rem 0;animation:bubbleIn .32s cubic-bezier(.22,.9,.35,1);}
    .chat-user .bbl {
        background:linear-gradient(135deg,var(--gold),var(--emerald));
        color:var(--bg);padding:0.75rem 1.15rem;
        border-radius:18px 18px 4px 18px;max-width:85%;
        font-size:0.98rem;line-height:1.75;text-align:right;direction:rtl;
        box-shadow:0 3px 12px var(--shadow);
        white-space:pre-wrap;word-break:break-word;
        font-weight:600;
    }
    .chat-nsm {display:flex;justify-content:flex-start;margin:0.55rem 0;gap:0.55rem;align-items:flex-start;animation:bubbleIn .32s cubic-bezier(.22,.9,.35,1);}
    .chat-nsm .bbl {
        background:var(--surface2);
        color:var(--text);padding:0.75rem 1.15rem;
        border-radius:18px 18px 18px 4px;max-width:85%;
        font-size:0.98rem;line-height:1.85;text-align:right;direction:rtl;
        border:1px solid var(--border);
        box-shadow:0 2px 8px var(--shadow);
        white-space:pre-wrap;word-break:break-word;
    }
    .chat-nsm .bbl code {
        background:var(--surface);color:var(--emerald);padding:0.15rem 0.4rem;
        border-radius:4px;font-size:0.88rem;font-family:monospace;
        white-space:pre-wrap;
    }
    .chat-nsm .bbl pre {
        background:var(--surface);border:1px solid var(--border);border-radius:8px;
        padding:0.8rem;overflow-x:auto;margin:0.5rem 0;
        font-size:0.85rem;color:var(--text-muted);
        white-space:pre;
    }
    .copy-btn {
        display:inline-block;margin-top:0.55rem;padding:0.28rem 0.7rem;
        font-size:0.74rem;font-weight:600;color:var(--text-muted);
        background:var(--surface);border:1px solid var(--border);
        border-radius:10px;cursor:pointer;transition:all .15s ease;
        direction:rtl;font-family:inherit;
    }
    .copy-btn:hover { color:var(--gold);border-color:var(--gold);}
    .copy-btn:active { transform:scale(0.96); }
    .ctx-tag {
        display:inline-block;background:var(--surface);border:1px solid var(--border);
        border-radius:20px;padding:0.18rem 0.7rem;font-size:0.72rem;
        color:var(--gold);margin-bottom:0.45rem;direction:rtl;
    }
    .chat-box {
        height:62vh;min-height:420px;max-height:680px;
        overflow-y:auto;padding:1.1rem;
        background:var(--bg);border-radius:18px;
        border:1px solid var(--border);margin-bottom:0.9rem;
        scroll-behavior:smooth;
        -webkit-overflow-scrolling:touch;
        overscroll-behavior:contain;
        box-shadow:inset 0 0 24px var(--shadow);
    }
    .chat-box::-webkit-scrollbar{width:5px;}
    .chat-box::-webkit-scrollbar-track{background:var(--bg);}
    .chat-box::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px;}
    .chat-box::-webkit-scrollbar-thumb:hover{background:var(--gold);}
    .typing-indicator {
        display:inline-block;color:var(--gold);font-size:0.85rem;
        animation:pulse 1.2s infinite;
    }
    @keyframes pulse{0%,100%{opacity:.4;}50%{opacity:1;}}

    /* ── مؤشر "يكتب الآن" بنقاط متتابعة + توهّج حول أيقونة NSM ── */
    .typing-wrap { display:flex; align-items:center; gap:0.6rem; }
    .thinking-ring {
        width:34px; height:34px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:1.15rem;
        background:var(--surface2); border:1px solid var(--border);
        box-shadow:0 0 0 0 var(--gold-soft);
        animation:nsmThinkRing 1.6s ease-out infinite;
        flex-shrink:0;
    }
    @keyframes nsmThinkRing {
        0%   { box-shadow:0 0 0 0 var(--gold-soft); }
        70%  { box-shadow:0 0 0 9px rgba(0,0,0,0); }
        100% { box-shadow:0 0 0 0 rgba(0,0,0,0); }
    }
    .typing-dots { display:inline-flex; gap:4px; align-items:center; padding:0.55rem 0.9rem;
        background:var(--surface2); border:1px solid var(--border); border-radius:18px 18px 18px 4px; }
    .typing-dots span {
        width:7px; height:7px; border-radius:50%;
        background:var(--gold); display:inline-block;
        animation:nsmDotBounce 1.1s ease-in-out infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay:.15s; background:var(--emerald); }
    .typing-dots span:nth-child(3) { animation-delay:.3s; }
    @keyframes nsmDotBounce {
        0%, 60%, 100% { transform:translateY(0); opacity:.55; }
        30% { transform:translateY(-5px); opacity:1; }
    }
    @media (prefers-reduced-motion: reduce) {
        .thinking-ring, .typing-dots span { animation:none !important; }
    }

    /* ── توقيت الرسائل (يظهر بأسفل كل فقاعة) ── */
    .bbl-ts {
        font-size: 0.68rem;
        color: var(--text-muted);
        opacity: 0.75;
        margin-top: 0.3rem;
        direction: ltr;
        text-align: left;
    }
    .chat-user .bbl-ts { color: rgba(0,0,0,0.55); text-align: right; }
    .bbl-footer { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; margin-top: 0.2rem; }
    .bbl-footer .bbl-ts { margin-top: 0; }

    /* ── زر عائم "النزول لآخر رسالة" — يظهر فقط عند التمرير لأعلى بعيداً
       عن نهاية المحادثة (يُتحكّم بإظهاره/إخفائه عبر JS بالأسفل) ── */
    .chat-box-wrap { position: relative; }
    .scroll-bottom-btn {
        position: absolute;
        bottom: 1.1rem;
        left: 50%;
        transform: translateX(-50%) translateY(8px);
        width: 38px; height: 38px;
        border-radius: 50%;
        border: 1px solid var(--border);
        background: var(--surface2);
        color: var(--gold);
        font-size: 1.1rem;
        box-shadow: 0 6px 18px var(--shadow);
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity .2s ease, transform .2s ease;
        z-index: 5;
    }
    .scroll-bottom-btn.visible { opacity: 1; pointer-events: auto; transform: translateX(-50%) translateY(0); }
    .scroll-bottom-btn:hover { border-color: var(--gold); }

    /* ── تنسيق st.chat_message الأصلي (يُستخدم فقط أثناء بث الرد حرفاً
       بحرف قبل أن يُطوى داخل .chat-box المخصص بعد rerun) — بدون هذا
       التنسيق يظهر بمظهر Streamlit الافتراضي غير المرتبط بصرياً بهوية
       الشات، ما يسبب "قفزة" بصرية واضحة لحظة انتهاء البث. ────────────── */
    [data-testid="stChatMessage"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.75rem 1.15rem !important;
        margin: 0.55rem 0 0.9rem !important;
        box-shadow: 0 2px 8px var(--shadow);
        direction: rtl !important;
        max-width: 85%;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] p {
        color: var(--text) !important;
        text-align: right !important;
        direction: rtl !important;
        font-size: 0.98rem !important;
        line-height: 1.85 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessage"] [class*="Avatar"] {
        background: var(--accent-grad) !important;
        border-radius: 10px !important;
    }

    /* ── استجابة الجوال ── */
    @media (max-width: 640px) {
        .chat-box {
            height:56vh;min-height:320px;max-height:520px;
            padding:0.8rem;border-radius:14px;
        }
        .chat-user .bbl, .chat-nsm .bbl {
            max-width:92%;font-size:0.92rem;padding:0.65rem 0.9rem;
        }
        .chat-nsm .bbl { line-height:1.7; }
        [data-testid="stChatMessage"] { max-width: 92%; }
    }
    </style>
    """, unsafe_allow_html=True)

    # رأس التبويب
    col_t, col_s = st.columns([3,1])
    with col_t:
        st.markdown("### 💬 المحادثة الذكية")
        _mode = "🤖 LLM · Cloudflare / Gemini / Groq"
        st.caption(f"يتذكر السياق · {_mode} · الذكاء في الأوزان")
    with col_s:
        ctx = bot.context_info()
        if ctx:
            st.markdown(f'<div class="ctx-tag">📎 {ctx}</div>', unsafe_allow_html=True)
        st.metric("رسائل الجلسة", st.session_state.nsm_count)

    # ── إرفاق ملف أو صورة (multimodal عبر OpenRouter) ─────────────────────
    if "chat_pending_files" not in st.session_state:
        st.session_state["chat_pending_files"] = []
    if "chat_uploader_version" not in st.session_state:
        st.session_state["chat_uploader_version"] = 0

    _or_key_chat = st.session_state.get("_or_api_key", "").strip()
    _or_model_chat = st.session_state.get("_or_model", "google/gemini-2.5-flash")
    _is_vision_chat = _or_model_chat in VISION_MODELS

    with st.expander("📎 إرفاق ملف أو صورة (يتطلب OpenRouter API Key)",
                      expanded=bool(st.session_state["chat_pending_files"])):
        if not _or_key_chat:
            st.info("🔑 أدخل OpenRouter API Key في الشريط الجانبي لتفعيل رفع الملفات والصور.")
        else:
            col_up, col_info = st.columns([3, 2])
            with col_up:
                # مفتاح ديناميكي — يُعاد ضبط عنصر الرفع بعد كل إرسال/مسح
                # حتى لا تُعاد إضافة نفس الملفات القديمة من الـ widget state
                uploaded = st.file_uploader(
                    "اسحب ملفاً هنا أو انقر للاختيار",
                    type=["png", "jpg", "jpeg", "webp", "gif",
                          "pdf", "txt", "md", "csv", "json",
                          "py", "js", "ts", "html", "yaml", "yml"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key=f"chat_file_uploader_{st.session_state['chat_uploader_version']}",
                )
                if uploaded:
                    existing_names = {f["name"] for f in st.session_state["chat_pending_files"]}
                    for uf in uploaded:
                        if uf.name not in existing_names:
                            extracted = _extract_file(uf)
                            if extracted:
                                st.session_state["chat_pending_files"].append(extracted)
                                existing_names.add(uf.name)
                            else:
                                st.warning(f"⚠ {uf.name} أكبر من {MAX_FILE_MB} MB")
            with col_info:
                if not _is_vision_chat and any(f["is_image"] for f in st.session_state["chat_pending_files"]):
                    st.warning("⚠ النموذج الحالي لا يدعم الصور. اختر نموذج رؤية في الشريط الجانبي.")
                elif _is_vision_chat:
                    st.markdown('<span class="ctx-tag">👁 رؤية مُفعَّلة</span>', unsafe_allow_html=True)
                st.caption(f"الحد الأقصى: {MAX_FILE_MB} MB للملف الواحد")

        if st.session_state["chat_pending_files"]:
            pf_cols = st.columns(min(len(st.session_state["chat_pending_files"]), 4))
            to_remove = []
            for i, f in enumerate(st.session_state["chat_pending_files"]):
                with pf_cols[i % 4]:
                    if f["is_image"] and f.get("raw_bytes"):
                        st.image(f["raw_bytes"], caption=f["name"], use_container_width=True)
                    else:
                        icon = "📄" if f["text_content"] else "📎"
                        st.caption(f"{icon} {f['name']} ({f['size_kb']} KB)")
                    if st.button("✕", key=f"chat_rm_file_{i}", help="حذف"):
                        to_remove.append(i)
            for idx in sorted(to_remove, reverse=True):
                st.session_state["chat_pending_files"].pop(idx)
            if to_remove:
                st.rerun()
            if st.button("🗑 مسح كل الملفات", key="chat_clear_all_files"):
                st.session_state["chat_pending_files"].clear()
                st.session_state["chat_uploader_version"] += 1
                st.rerun()

    # عرض المحادثة
    html = '<div class="chat-box-wrap"><div class="chat-box" id="nsm-chat-box">'
    if not st.session_state.nsm_messages:
        html += '''<div style="text-align:center;color:var(--text-muted);padding:3rem 1rem;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%">
            <div style="width:56px;height:56px;border-radius:16px;display:flex;align-items:center;
                        justify-content:center;font-size:1.8rem;background:var(--accent-grad);
                        box-shadow:0 6px 20px var(--gold-soft);margin-bottom:1rem">🧠</div>
            <div style="font-size:1.05rem;font-weight:700;color:var(--text);margin-bottom:0.3rem">
                ابدأ محادثتك مع NSM
            </div>
            <div style="font-size:0.85rem;max-width:340px;line-height:1.8">
                اسأل عن مفهوم إسلامي، آية قرآنية، أو أي موضوع آخر — أو جرّب أحد الأسئلة السريعة بالأسفل
            </div>
        </div>'''
    else:
        for _i, msg in enumerate(st.session_state.nsm_messages):
            role, text = msg[0], msg[1]
            ctx_tag    = msg[2] if len(msg) > 2 else ""
            src_badge  = msg[3] if len(msg) > 3 else ""
            ts         = msg[4] if len(msg) > 4 else ""
            ts_html    = f'<div class="bbl-ts">{ts}</div>' if ts else ""
            if role == "user":
                import html as _html
                safe_text = _html.escape(text).replace("\n", "<br>")
                html += f'<div class="chat-user"><div class="bbl">{safe_text}{ts_html}</div></div>'
            else:
                ctx_html = f'<div class="ctx-tag">📎 {ctx_tag}</div>' if ctx_tag else ""
                src_html = (
                    f'<div class="ctx-tag" style="color:var(--emerald)">{src_badge}</div>'
                    if src_badge else ""
                )
                _audio_html = ""
                _audio_entry = st.session_state.get("_nsm_audio_cache", {}).get(_i)
                if _audio_entry:
                    _a_b64, _a_fmt = _audio_entry
                    _audio_html = (
                        f'<audio controls style="width:100%;margin-top:0.5rem;height:36px" '
                        f'src="data:audio/{_a_fmt};base64,{_a_b64}"></audio>'
                    )
                import html as _html
                if "<" not in text and ">" not in text:
                    safe_reply = _html.escape(text).replace("\n", "<br>")
                else:
                    safe_reply = text
                html += f'''<div class="chat-nsm">
                    <span style="font-size:1.4rem;margin-top:3px">🧠</span>
                    <div class="bbl">{ctx_html}{src_html}<div class="bbl-text" id="nsm-bbl-{_i}">{safe_reply}</div>{_audio_html}
                        <div class="bbl-footer">
                            <button class="copy-btn" title="نسخ الرد"
                                onclick="var t=document.getElementById('nsm-bbl-{_i}').innerText;
                                         navigator.clipboard.writeText(t).then(function(){{
                                            var b=event.currentTarget; var old=b.textContent;
                                            b.textContent='✓ تم النسخ';
                                            setTimeout(function(){{b.textContent=old;}}, 1300);
                                         }});">📋 نسخ</button>
                            {ts_html}
                        </div>
                    </div>
                </div>'''
    html += '''</div>
        <button class="scroll-bottom-btn" id="nsm-scroll-bottom" title="النزول لآخر رسالة" aria-label="النزول لآخر رسالة">↓</button>
    </div>'''
    st.markdown(html, unsafe_allow_html=True)
    st.components.v1.html("""
    <script>
    (function() {
        function scrollToBottom() {
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('nsm-chat-box');
            if (box) { box.scrollTop = box.scrollHeight; return true; }
            return false;
        }
        // Streamlit يعيد رسم الـ DOM بشكل غير متزامن أحياناً — نحاول عدة مرات
        // بدل الاعتماد على تنفيذ واحد فوري قد يسبق اكتمال العنصر.
        let attempts = 0;
        const tryScroll = () => {
            attempts++;
            if (!scrollToBottom() && attempts < 10) {
                setTimeout(tryScroll, 60);
            }
        };
        tryScroll();

        // ── زر "النزول لآخر رسالة": يظهر فقط عندما يكون المستخدم بعيداً
        // عن أسفل الصندوق (بأكثر من 80px)، ويختفي تلقائياً عند الوصول للأسفل ──
        function bindScrollButton() {
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('nsm-chat-box');
            const btn = doc.getElementById('nsm-scroll-bottom');
            if (!box || !btn) return false;
            if (btn.dataset.nsmBound) { updateVisibility(); return true; }
            btn.dataset.nsmBound = "1";

            function updateVisibility() {
                const distanceFromBottom = box.scrollHeight - box.scrollTop - box.clientHeight;
                btn.classList.toggle('visible', distanceFromBottom > 80);
            }
            box.addEventListener('scroll', updateVisibility, { passive: true });
            btn.addEventListener('click', function() {
                box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
            });
            updateVisibility();
            return true;
        }
        let btnAttempts = 0;
        const tryBind = () => {
            btnAttempts++;
            if (!bindScrollButton() && btnAttempts < 10) { setTimeout(tryBind, 60); }
        };
        tryBind();
    })();
    </script>
    """, height=0)

    # ── تقييم آخر رد (👍/👎) لتغذية autotune_feedback ──
    if _AUTOTUNE_OK:
        _af_turn = st.session_state.get("_af_last_turn")
        if _af_turn and not _af_turn.get("rated"):
            _af_c1, _af_c2, _af_c3 = st.columns([1, 1, 6])
            with _af_c1:
                if st.button("👍", key="_af_up", help="رد جيد — ساعد النظام يتعلّم"):
                    _heur = _af_compute_heuristics(_af_turn["response"])
                    _af_process_feedback(_AFFeedbackRecord(
                        message_id=str(st.session_state.nsm_count),
                        timestamp=datetime.now().timestamp(),
                        context_type=_af_turn["context_type"],
                        model=_af_turn["model"],
                        persona=_af_turn["persona"],
                        params=_af_turn["params"],
                        rating=1,
                        heuristics=vars(_heur),
                    ))
                    try:
                        from ai.learning_orchestrator import get_orchestrator
                        get_orchestrator().feedback(_af_turn.get("query", ""), is_positive=True)
                    except Exception:
                        pass
                    _af_turn["rated"] = True
                    st.toast("✅ شكراً — تم تسجيل التقييم")
                    st.rerun()
            with _af_c2:
                if st.button("👎", key="_af_down", help="رد غير جيد — ساعد النظام يتعلّم"):
                    _heur = _af_compute_heuristics(_af_turn["response"])
                    _af_process_feedback(_AFFeedbackRecord(
                        message_id=str(st.session_state.nsm_count),
                        timestamp=datetime.now().timestamp(),
                        context_type=_af_turn["context_type"],
                        model=_af_turn["model"],
                        persona=_af_turn["persona"],
                        params=_af_turn["params"],
                        rating=-1,
                        heuristics=vars(_heur),
                    ))
                    try:
                        from ai.learning_orchestrator import get_orchestrator
                        get_orchestrator().feedback(_af_turn.get("query", ""), is_positive=False)
                    except Exception:
                        pass
                    _af_turn["rated"] = True
                    st.toast("✅ شكراً — تم تسجيل التقييم")
                    st.rerun()

    # صندوق الإدخال
    st.markdown("""
    <style>
    div[data-testid="stTextArea"] textarea {
        min-height:96px !important;
        max-height:220px !important;
        font-size:1.05rem !important;
        line-height:1.6 !important;
        direction:rtl;
        text-align:right;
        resize:none !important;
        background:var(--surface2) !important;
        border:1.5px solid var(--border) !important;
        border-radius:18px !important;
        padding:0.9rem 1.1rem !important;
        color:var(--text) !important;
        transition:border-color .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color:var(--gold) !important;
        box-shadow:0 0 0 3px var(--gold-soft) !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color:var(--text-muted);
    }
    .st-key-nsm_send_wrap button {
        height:96px !important;
        border-radius:18px !important;
        background:linear-gradient(135deg,var(--gold),var(--emerald)) !important;
        color:var(--bg) !important;
        font-size:1.02rem !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 3px 12px var(--shadow) !important;
        transition:transform .12s ease, box-shadow .12s ease;
    }
    .st-key-nsm_send_wrap button:hover {
        transform:translateY(-1px);
        box-shadow:0 5px 16px var(--shadow) !important;
    }
    .st-key-nsm_send_wrap button:active {
        transform:translateY(0);
    }
    @media (max-width: 640px) {
        div[data-testid="stTextArea"] textarea {
            min-height:76px !important;font-size:0.98rem !important;
        }
        .st-key-nsm_send_wrap button { height:52px !important; }
    }
    </style>""", unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        user_input = st.text_area(
            label="سؤالك",
            placeholder="اكتب سؤالك هنا… (Enter = سطر جديد)",
            key="nsm_input",
            label_visibility="collapsed",
            height=96,
        )
    with c2:
        with st.container(key="nsm_send_wrap"):
            send = st.button("➤\nإرسال", key="nsm_send", use_container_width=True)

    # ── الواجهة الصوتية: تسجيل سؤال بالصوت + قراءة الردود صوتياً ─────────
    voice_col1, voice_col2 = st.columns([3, 2], gap="small")
    _voice_query = None
    with voice_col1:
        if _STT_OK:
            _mic_audio = st.audio_input("🎤 أو سجّل سؤالك صوتياً", key="nsm_mic_input")
            if _mic_audio is not None:
                _mic_bytes = _mic_audio.getvalue()
                _mic_hash = hash(_mic_bytes)
                if st.session_state.get("_nsm_last_mic_hash") != _mic_hash:
                    st.session_state["_nsm_last_mic_hash"] = _mic_hash
                    with st.spinner("⟳ جارٍ تفريغ الصوت..."):
                        _transcribed, _stt_err = _stt_transcribe(_mic_bytes, mime_type="audio/wav")
                    if _stt_err:
                        st.warning(f"⚠️ {_stt_err}")
                    elif _transcribed:
                        _voice_query = _transcribed
        else:
            st.caption("🎤 الإدخال الصوتي غير متاح حالياً")
    with voice_col2:
        _voice_output_on = st.toggle(
            "🔊 قراءة الردود صوتياً", key="_nsm_voice_output",
            value=st.session_state.get("_nsm_voice_output", False),
            disabled=not _TTS_OK,
        )

    # أسئلة سريعة — كاملة عند بداية المحادثة فقط، ثم مطوية لتقليل الازدحام البصري
    quick_qs = [
        "ما هي أركان الإسلام؟",
        "ما هو الذكاء الاصطناعي؟",
        "ما هي سورة الفاتحة؟",
        "ما هو الجبر الخطي؟",
        "من هم الخلفاء الراشدون؟",
        "ما هي لغة Python؟",
        "ما هي سورة الكهف؟",
        "ما هي التغذية السليمة؟",
    ]
    if not st.session_state.nsm_messages:
        st.markdown("**⚡ أسئلة سريعة:**")
        quick_cols = st.columns(4)
        for i, q in enumerate(quick_qs):
            with quick_cols[i % 4]:
                if st.button(q, key=f"chat_q_{i}", use_container_width=True):
                    st.session_state._chat_pending = q
    else:
        with st.expander("⚡ أسئلة سريعة"):
            quick_cols = st.columns(4)
            for i, q in enumerate(quick_qs):
                with quick_cols[i % 4]:
                    if st.button(q, key=f"chat_q_{i}", use_container_width=True):
                        st.session_state._chat_pending = q

    # ── أزرار تحليل المشروع (NSM Agent) — للمالك فقط ─────────────
    # الأوامر خلف هذه الأزرار (افحص/عدل/أنشئ/ارفع) تقرأ/تكتب ملفات فعلية
    # على الخادم وتنفّذ git push — عُطِّلت من nsm_chat.py لغير المالك،
    # ونخفي الواجهة نفسها هنا حتى لا تظهر أزرار بلا فائدة للزائر العادي.
    if st.session_state.get("_dev_console_unlocked", False):
        st.markdown("---")
        st.markdown("**🤖 تحليل المشروع:**")
        agent_cols = st.columns(6)
        agent_btns = [
            ("📋 اقترح (كل)",      "اقترح"),
            ("🗂 غير مستخدم",      "اقترح غير مستخدم"),
            ("⚠️ أخطاء",           "اقترح أخطاء"),
            ("📦 ملفات كبيرة",     "اقترح كبير"),
            ("📁 قائمة الملفات",   "قائمة"),
            ("🔁 مكررة",           "اقترح مكررة"),
        ]
        for i, (label, cmd) in enumerate(agent_btns):
            with agent_cols[i]:
                if st.button(label, key=f"agent_btn_{i}", use_container_width=True):
                    st.session_state._chat_pending = cmd

        # أزرار تحليل ملف محدد
        st.markdown("**🔍 تحليل ملف محدد** — اكتب المسار ثم اختر العملية:")
        file_path_input = st.text_input(
            "مسار الملف", placeholder="مثال: ai/code_agent.py",
            key="agent_file_path", label_visibility="collapsed"
        )
        if file_path_input.strip():
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                if st.button("📄 ملخص", key="btn_summary", use_container_width=True):
                    st.session_state._chat_pending = f"ملخص {file_path_input.strip()}"
            with fc2:
                if st.button("🔧 صحح", key="btn_fix", use_container_width=True):
                    st.session_state._chat_pending = f"صحح {file_path_input.strip()}"
            with fc3:
                if st.button("👁 افحص", key="btn_inspect", use_container_width=True):
                    st.session_state._chat_pending = f"افحص {file_path_input.strip()}"

    # مسح المحادثة
    if st.button("🗑 مسح المحادثة", key="nsm_clear"):
        st.session_state.nsm_messages = []
        st.session_state.nsm_count = 0
        bot.clear_history()
        st.rerun()

    # معالجة الإدخال
    def _process(text: str):
        files = list(st.session_state["chat_pending_files"])
        if not text.strip() and not files:
            return

        st.session_state["chat_pending_files"] = []
        st.session_state["chat_uploader_version"] += 1

        display_text = text.strip()
        if files:
            names = ", ".join(f["name"] for f in files)
            display_text += f"\n\n📎 {names}"

        _ts = datetime.now().strftime("%H:%M")

        # ── أضف رسالة المستخدم فوراً ──
        st.session_state.nsm_messages.append(("user", display_text, "", "", _ts))

        # ── فحص أمان أولي (regex محلي، بدون تكلفة API) ──
        _safety_msg = _nsm_safety_gate(text.strip())
        if _safety_msg:
            st.session_state.nsm_messages.append(("nsm", _safety_msg, "", "🛡️ فحص أمان", datetime.now().strftime("%H:%M")))
            st.session_state.nsm_count += 1
            st.rerun()
            return

        # ── كاش الردود المتعلَّمة (ConversationLearner عبر LearningOrchestrator) ──
        # يوفّر زمن استجابة وحصة LLM المجانية (Groq/Gemini/Cloudflare) عند
        # تكرار نفس السؤال حرفياً فقط. نتجاهل عمداً المطابقة التقريبية
        # بالكلمات المفتاحية الموجودة داخل recall() الأصلية (source="learned")
        # لأنها قد تُرجع إجابة سؤال مختلف بثقة زائفة — نقبل فقط
        # source="cache" (تطابق كامل لنص السؤال).
        try:
            from ai.learning_orchestrator import get_orchestrator
            _cached = get_orchestrator().recall(text.strip(), min_quality=0.75)
        except Exception:
            _cached = None
        if _cached and _cached.get("source") == "cache" and (_cached.get("answer") or "").strip():
            st.session_state.nsm_messages.append((
                "nsm", _cached["answer"], "", "⚡ كاش متعلَّم", datetime.now().strftime("%H:%M")
            ))
            st.session_state.nsm_count += 1
            st.rerun()
            return

        # ════════════════════════════════════════════════════════════════════
        # [1] بناء قائمة العقد المتاحة فعلاً
        # ════════════════════════════════════════════════════════════════════
        import time as _time_mod

        _or_key_p = st.session_state.get("_or_api_key", "").strip()
        _available_nodes: list = []
        if _or_key_p:
            _available_nodes.append("nsm:openrouter")

        # فحص NSM Agent مبكراً قبل قرار التوجيه
        _agent = None
        try:
            from ai.nsm_agent_core import NSMAgent as _AgentCls
            _agent = getattr(st.session_state, "_nsm_agent_instance", None)
            if _agent is None:
                _agent = _AgentCls()
                st.session_state._nsm_agent_instance = _agent
            _agent.available = _agent._check_available()
            if _agent.available:
                _available_nodes.append("nsm:agent")
        except Exception:
            _agent = None
        _available_nodes.append("nsm:free_router")   # دائماً متاح

        # ════════════════════════════════════════════════════════════════════
        # [2] التوجيه الدلالي — صنّف الاستعلام وانحَز للعقدة الأنسب
        # ════════════════════════════════════════════════════════════════════
        _sem_category   = "general"
        _sem_confidence = 0.2
        _sem_biased     = list(_available_nodes)
        if _NSM_SEMANTIC_OK and _nsm_semantic:
            try:
                _sem_category, _sem_confidence = _nsm_semantic.classify(text.strip())
                _sem_biased = _nsm_semantic.bias_order(
                    _sem_category, _available_nodes, _sem_confidence
                )
            except Exception:
                pass

        # ════════════════════════════════════════════════════════════════════
        # [3] اختَر العقدة (تاريخي 65% + دلالي 35%)
        # ════════════════════════════════════════════════════════════════════
        if _NSM_BRIDGE_OK and _nsm_bridge:
            _selected_node = _nsm_bridge.select_node_with_semantic(
                text.strip(), _sem_biased, _sem_category, _sem_confidence
            )
        else:
            _selected_node = _sem_biased[0]

        # ════════════════════════════════════════════════════════════════════
        # [4] حلقة تنفيذ مع Failover تلقائي (حتى 2 إعادة توجيه)
        # ════════════════════════════════════════════════════════════════════
        _excluded_nodes: list = []
        _response       = ""
        _ctx_tag        = ""
        _src_badge      = "🤖 NSM"
        _af_params_last = dict(_AF_NEUTRAL_PARAMS) if _AUTOTUNE_OK else {"temperature": 0.7}
        _af_ctx_last    = "conversational"
        _or_model_last  = st.session_state.get("_or_model", "google/gemini-2.5-flash")
        _final_node     = _selected_node
        _total_latency  = 0.0

        for _attempt in range(len(_available_nodes)):
            # Failover: اختر التالية إذا فشلت السابقة
            if _attempt > 0:
                if _NSM_BRIDGE_OK and _nsm_bridge:
                    _selected_node = _nsm_bridge.select_next_node(_available_nodes, _excluded_nodes)
                else:
                    _rem = [n for n in _available_nodes if n not in _excluded_nodes]
                    _selected_node = _rem[0] if _rem else "nsm:free_router"
                _final_node = _selected_node

                # مؤشر إعادة التوجيه للمستخدم
                st.toast(f"🔄 إعادة توجيه تلقائي → {_selected_node.replace('nsm:','')}", icon="⚡")

            _t0_route = _time_mod.time()
            _attempt_success = False

            # ── تنفيذ العقدة المختارة ─────────────────────────────────────
            if _selected_node == "nsm:openrouter" and _or_key_p:
                # ── مسار OpenRouter ──────────────────────────────────────
                _or_model_p = st.session_state.get("_or_model", "google/gemini-2.5-flash")
                _or_model_last = _or_model_p
                can_vision  = _or_model_p in VISION_MODELS
                doc_files   = [f for f in files if not f["is_image"]]
                image_files = [f for f in files if f["is_image"]] if can_vision else []
                user_content = _build_user_content(text.strip(), doc_files, image_files)
                history_msgs = []
                for m in st.session_state.nsm_messages[:-1]:
                    role = "user" if m[0] == "user" else "assistant"
                    history_msgs.append({"role": role, "content": m[1]})
                api_messages = history_msgs + [{"role": "user", "content": user_content}]

                _af_params  = dict(_AF_NEUTRAL_PARAMS) if _AUTOTUNE_OK else {"temperature": 0.7, "top_p": 0.9}
                _af_ctx     = "conversational"
                _af_note    = ""
                if _AUTOTUNE_OK:
                    try:
                        _af_params, _, _af_note = _af_apply_adjustments(_af_params, _af_ctx)
                    except Exception:
                        pass
                _af_params_last = _af_params
                _af_ctx_last    = _af_ctx

                full_response = ""
                with st.chat_message("assistant", avatar="🌐"):
                    placeholder = st.empty()
                    try:
                        for chunk in _or_stream(
                            api_messages, model=_or_model_p, api_key=_or_key_p,
                            temperature=_af_params.get("temperature", 0.7),
                            top_p=_af_params.get("top_p", 0.9),
                        ):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        _attempt_success = bool(full_response.strip())
                    except Exception:
                        placeholder.markdown(full_response or "⚠️ خطأ في OpenRouter — جاري الإعادة...")
                    if _af_note:
                        st.caption(_af_note)

                _response  = full_response
                _ctx_tag   = ""
                _src_badge = f"🌐 OpenRouter · {_or_model_p.split('/')[-1]}"

            elif _selected_node == "nsm:agent" and _agent and _agent.available:
                # ── مسار NSM Agent — Streaming ──────────────────────────
                full_response = ""
                with st.chat_message("assistant", avatar="🧠"):
                    placeholder = st.empty()
                    try:
                        for chunk in _agent.run_stream(text.strip()):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        _attempt_success = bool(full_response.strip())
                    except Exception:
                        placeholder.markdown(full_response or "⚠️ خطأ في NSM Agent — جاري الإعادة...")
                if hasattr(bot, "_last_source"):
                    bot._last_source = "nsm_agent"
                _response  = full_response.replace("⏳ *أفكر...*\n\n", "", 1)
                _ctx_tag   = bot.context_info() if hasattr(bot, "context_info") else ""
                _src_badge = bot.source_badge() if hasattr(bot, "source_badge") else "🧠 NSM Agent"

            else:
                # ── مسار free_router (الاحتياطي الأخير) ──────────────────
                with st.chat_message("assistant", avatar="🧠"):
                    _typing_ph = st.empty()
                    _typing_ph.markdown(
                        '''<div class="typing-wrap">
                            <span class="thinking-ring">🧠</span>
                            <span class="typing-dots"><span></span><span></span><span></span></span>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                    try:
                        _resp_raw = bot.chat(text.strip(), system_prompt=NSM_SYSTEM_PROMPT)
                        _attempt_success = bool(_resp_raw and _resp_raw.strip())
                    except Exception:
                        _resp_raw = "⚠️ تعذّر الحصول على رد."
                    _typing_ph.empty()
                _response  = _resp_raw
                _ctx_tag   = bot.context_info() if hasattr(bot, "context_info") else ""
                _src_badge = bot.source_badge() if hasattr(bot, "source_badge") else "⚡ Free Router"

            # ── قياس الزمن + تقييم الجودة + تسجيل النتيجة ────────────────
            _latency_ms = (_time_mod.time() - _t0_route) * 1000
            _total_latency += _latency_ms

            # التقييم الثنائي القديم (فارغ/غير فارغ) يبقى كحد أدنى أولي،
            # ثم نُدقّقه بتقييم الجودة الحقيقي إن كان متاحاً
            _quality: dict = {}
            if _attempt_success and _QUALITY_SCORER_OK:
                try:
                    _quality = _score_response(text.strip(), _response)
                    _attempt_success = bool(_quality.get("is_quality", True))
                except Exception:
                    _quality = {}

            if _NSM_BRIDGE_OK and _nsm_bridge:
                _nsm_bridge.record_result(_selected_node, _attempt_success, _latency_ms)

            # ── سجل التوجيه الحي (آخر 100 قرار) ──────────────────────────
            _sem_icon = ""
            if _NSM_SEMANTIC_OK and _nsm_semantic:
                try:
                    _sem_icon = _nsm_semantic.CATEGORY_LABELS.get(_sem_category, ("💬", ""))[0]
                except Exception:
                    _sem_icon = "💬"
            _route_entry = {
                "ts":         datetime.now().strftime("%H:%M:%S"),
                "query":      text.strip()[:55] + ("…" if len(text.strip()) > 55 else ""),
                "category":   _sem_category,
                "cat_icon":   _sem_icon,
                "confidence": round(_sem_confidence, 2),
                "node":       _selected_node,
                "latency_ms": round(_latency_ms),
                "success":    _attempt_success,
                "attempt":    _attempt + 1,
                "failover":   _attempt > 0,
                "quality_score": _quality.get("score"),
            }
            _rlog = st.session_state.setdefault("nsm_route_log", [])
            _rlog.append(_route_entry)
            if len(_rlog) > 100:
                st.session_state["nsm_route_log"] = _rlog[-100:]
            if _ROUTE_LOG_DB_OK:
                _rlog_append(_route_entry)

            if _attempt_success:
                break   # نجاح — توقّف
            _excluded_nodes.append(_selected_node)

        # ════════════════════════════════════════════════════════════════════
        # [5] حفظ + إظهار الرد النهائي
        # ════════════════════════════════════════════════════════════════════
        _source_key = "chat_openrouter" if "openrouter" in _final_node else "chat_nsm_agent"
        _record_chat_episode(text.strip(), _response, source=_source_key)
        st.session_state.nsm_messages.append((
            "nsm", _response, _ctx_tag, _src_badge, datetime.now().strftime("%H:%M")
        ))
        _msg_idx = len(st.session_state.nsm_messages) - 1
        if _TTS_OK and st.session_state.get("_nsm_voice_output") and _response.strip():
            try:
                with st.spinner("⟳ جارٍ تحويل الرد لصوت..."):
                    _tts_result = _TTSEngineCls().synthesize(_response.strip())
                if _tts_result.ok:
                    import base64 as _b64
                    _audio_cache = st.session_state.setdefault("_nsm_audio_cache", {})
                    _audio_cache[_msg_idx] = (
                        _b64.b64encode(_tts_result.audio_bytes).decode("ascii"),
                        _tts_result.format,
                    )
            except Exception:
                pass  # فشل TTS لا يجب أن يُعطّل عرض الرد النصي
        if _AUTOTUNE_OK:
            st.session_state["_af_last_turn"] = {
                "response": _response, "params": _af_params_last,
                "context_type": _af_ctx_last,
                "model": _or_model_last if "openrouter" in _final_node else _src_badge,
                "persona": "nsm", "rated": False,
                "query": text.strip(),
            }
        st.session_state.nsm_count += 1
        st.rerun()

    if send and (user_input or st.session_state["chat_pending_files"]):
        _process(user_input)

    if _voice_query:
        _process(_voice_query)

    if hasattr(st.session_state, "_chat_pending"):
        q = st.session_state._chat_pending
        del st.session_state._chat_pending
        _process(q)


# ══════════════════════════════════════════════════════════════════════════
# تبويب وكلاء AI — صفحة مستقلة لكل فئة/تخصص
def render_social_agent():
    """يدير الوكيل الاجتماعي الموحّد (ai/social_agent.py): تشغيل/إيقاف
    الاستطلاع التلقائي، اختيار المنصات المفعّلة وكلمات المراقبة، النشر
    اليدوي الفوري، وعرض آخر الأحداث/الأخطاء لكل منصة."""
    st.markdown('<div class="section-header">📡 الوكيل الاجتماعي</div>', unsafe_allow_html=True)
    st.caption(
        "نشر + رد تلقائي + مراقبة عبر Discord وTelegram وInstagram "
        "وFacebook وYouTube وTikTok وReddit وThreads وWhatsApp، "
        "ونشر فقط عبر Pinterest (لا يوفّر API مراقبة/رد — راجع تلميح المنصة)، "
        "بنفس شخصية NSM الموحّدة — مع جدولة منشورات وتحليل مشاعر وردود تتذكّر كل شخص."
    )

    try:
        from ai.social_agent import (
            get_manager, get_config, set_config, get_recent_events,
            schedule_post, get_scheduled, cancel_scheduled, get_analytics_summary,
        )
        from ai.social_platforms import PLATFORM_LABELS_AR, PLATFORM_CHAR_LIMITS
    except Exception as _sa_err:
        st.error(f"⚠️ تعذّر تحميل وحدة الوكيل الاجتماعي: {_sa_err}")
        return

    mgr = get_manager()
    status = mgr.status()

    # ── شريط الحالة العلوي (ثابت خارج التبويبات — أهم معلومة تبقى مرئية دائماً) ──
    col_state, col_action = st.columns([2, 1])
    running = mgr.is_running()
    with col_state:
        n_ready = sum(1 for s in status.values() if s.configured)
        st.markdown(
            f"**حالة الخدمة:** {'🟢 تعمل' if running else '⚪ متوقفة'} "
            f"· {n_ready}/{len(status)} منصة مُهيّأة"
        )
    with col_action:
        if running:
            if st.button("⏹️ إيقاف", key="social_stop", use_container_width=True):
                with st.spinner("⟳ ..."):
                    mgr.stop()
                st.rerun()
        else:
            if st.button("▶️ تشغيل", key="social_start", use_container_width=True):
                with st.spinner("⟳ ..."):
                    mgr.start()
                st.rerun()

    tab_settings, tab_status, tab_publish, tab_insights = st.tabs(
        ["⚙️ الإعدادات", "📊 حالة المنصات", "✍️ نشر وجدولة", "📈 تحليلات وأحداث"]
    )

    # ═══════════════════════════════ ⚙️ الإعدادات ═══════════════════════════════
    with tab_settings:
        st.markdown("#### إعدادات المراقبة")
        selected = st.multiselect(
            "المنصات المفعّلة",
            options=list(PLATFORM_LABELS_AR.keys()),
            default=list(set(get_config("enabled_platforms", []))),
            format_func=lambda p: PLATFORM_LABELS_AR.get(p, p),
            key="social_enabled_platforms",
        )
        keywords_str = st.text_input(
            "كلمات مفتاحية للمراقبة (مفصولة بفاصلة، اتركه فارغاً لمراقبة كل شيء)",
            value=", ".join(get_config("keywords", [])),
            key="social_keywords",
        )
        auto_reply = st.checkbox(
            "🤖 رد تلقائي على الإشارات المطابقة",
            value=get_config("auto_reply", False), key="social_auto_reply",
        )
        poll_interval = st.slider(
            "فترة الاستطلاع (ثانية)", 30, 600,
            int(get_config("poll_interval", 90)), 10, key="social_poll_interval",
        )
        if st.button("💾 حفظ الإعدادات", key="social_save_settings", type="primary"):
            with st.spinner("⟳ يحفظ..."):
                set_config("enabled_platforms", selected)
                set_config("keywords", [k.strip() for k in keywords_str.split(",") if k.strip()])
                set_config("auto_reply", auto_reply)
                set_config("poll_interval", poll_interval)
            st.success("✅ تم الحفظ.")
            st.rerun()

        st.markdown("---")
        st.markdown("#### ⚡ Telegram: Webhook مقابل Polling")
        st.caption(
            "Webhook يدفع الرسائل فوراً بدل الاستطلاع الدوري، لكنه يتطلب "
            "endpoint HTTPS عام ثابت (خادم api_server.py، منفصل عن Streamlit) "
            "ومتغيرَي بيئة: TELEGRAM_WEBHOOK_BASE_URL وTELEGRAM_WEBHOOK_SECRET. "
            "بدونهما يبقى النظام يعمل بـpolling كالمعتاد — لا كسر لأي سلوك حالي."
        )
        webhook_platforms_cfg = set(get_config("webhook_enabled_platforms", []))
        tg_webhook_on = "telegram" in webhook_platforms_cfg
        base_url = os.environ.get("TELEGRAM_WEBHOOK_BASE_URL", "")
        tg_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET", "")
        col_tg1, col_tg2 = st.columns([2, 1])
        with col_tg1:
            st.markdown(f"**الوضع الحالي:** {'🔗 Webhook' if tg_webhook_on else '🔁 Polling'}")
            if base_url and tg_secret:
                st.caption("عنوان الـwebhook (يُضبط تلقائياً عند الضغط على تفعيل):")
                st.code(f"{base_url.rstrip('/')}/webhook/telegram/{tg_secret}", language=None)
        with col_tg2:
            if not tg_webhook_on:
                if st.button("🔗 تفعيل Webhook", key="tg_webhook_enable", use_container_width=True):
                    if not base_url or not tg_secret:
                        st.error("يلزم ضبط TELEGRAM_WEBHOOK_BASE_URL وTELEGRAM_WEBHOOK_SECRET أولاً.")
                    else:
                        try:
                            with st.spinner("⟳ يفعّل..."):
                                url = f"{base_url.rstrip('/')}/webhook/telegram/{tg_secret}"
                                mgr.enable_webhook("telegram", url, secret_token=tg_secret)
                            st.success("✅ تم تفعيل webhook تيليجرام.")
                            st.rerun()
                        except Exception as e:
                            st.error(f"فشل التفعيل: {e}")
            else:
                if st.button("🔁 العودة لـPolling", key="tg_webhook_disable", use_container_width=True):
                    try:
                        with st.spinner("⟳ يلغي..."):
                            mgr.disable_webhook("telegram")
                        st.success("✅ تم إلغاء webhook والعودة لـpolling.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"فشل الإلغاء: {e}")

        st.markdown("---")
        st.markdown("#### 💬 WhatsApp: رابط الـWebhook")
        st.caption(
            "واتساب لا يوفّر polling إطلاقاً — الربط يتم يدوياً من لوحة Meta "
            "Developer (وليس بزر هنا كتيليجرام): الصقي الرابط ورمز التحقق "
            "أدناه في إعدادات Webhook بتطبيق Meta الخاص بك."
        )
        wa_base = os.environ.get("TELEGRAM_WEBHOOK_BASE_URL", "")  # نفس الخادم عادة (api_server.py)
        wa_verify = os.environ.get("WHATSAPP_VERIFY_TOKEN", "")
        if wa_base and wa_verify:
            st.caption("Callback URL:")
            st.code(f"{wa_base.rstrip('/')}/webhook/whatsapp", language=None)
            st.caption("Verify Token:")
            st.code(wa_verify, language=None)
        else:
            st.caption("⚪ اضبطي TELEGRAM_WEBHOOK_BASE_URL وWHATSAPP_VERIFY_TOKEN لعرض الرابط جاهزاً للنسخ.")

    # ═══════════════════════════════ 📊 حالة المنصات ═══════════════════════════════
    with tab_status:
        col_h, col_r = st.columns([4, 1])
        with col_h:
            st.markdown("#### حالة كل منصة")
        with col_r:
            if st.button("🔄 تحديث", key="social_status_refresh", use_container_width=True):
                st.rerun()

        ready = [(p, s) for p, s in status.items() if s.configured]
        not_ready = [(p, s) for p, s in status.items() if not s.configured]

        def _render_platform_row(pid: str, s) -> None:
            label = PLATFORM_LABELS_AR.get(pid, pid)
            badge = "🟢 مُهيّأة" if s.configured else f"🔴 غير مُهيّأة (يلزم: {', '.join(s.missing_env) or '—'})"
            line = f"- **{label}** — {badge}"
            if not mgr.adapters[pid].supports_monitoring:
                line += " · ⚡ نشر فقط"
            if mgr.adapters[pid].supports_webhook and pid in webhook_platforms_cfg_for_status:
                line += " · 🔗 webhook مفعّل"
            if s.last_poll:
                line += f" · آخر استطلاع: {s.last_poll}"
            st.markdown(line)
            if s.last_error:
                st.caption(f"⚠️ آخر خطأ: {s.last_error}")

        webhook_platforms_cfg_for_status = set(get_config("webhook_enabled_platforms", []))

        if ready:
            st.markdown("**🟢 جاهزة**")
            for pid, s in ready:
                _render_platform_row(pid, s)
        if not_ready:
            st.markdown("**🔴 تحتاج إعداد**")
            for pid, s in not_ready:
                _render_platform_row(pid, s)

    # ═══════════════════════════════ ✍️ نشر وجدولة ═══════════════════════════════
    with tab_publish:
        st.markdown("#### نشر يدوي فوري")
        publish_text = st.text_area("النص", key="social_publish_text", height=100)
        publish_platforms = st.multiselect(
            "انشر على:", options=list(PLATFORM_LABELS_AR.keys()),
            format_func=lambda p: PLATFORM_LABELS_AR.get(p, p), key="social_publish_platforms",
        )
        if publish_text.strip() and publish_platforms:
            over_limit = []
            for pid in publish_platforms:
                limit = PLATFORM_CHAR_LIMITS.get(pid)
                if limit and len(publish_text.strip()) > limit:
                    over_limit.append((pid, limit))
            if over_limit:
                warn_lines = "، ".join(
                    f"{PLATFORM_LABELS_AR.get(p, p)} (الحد {lim} حرف)" for p, lim in over_limit
                )
                st.warning(f"⚠️ النص أطول من الحد المعروف لبعض المنصات: {warn_lines} — قد يُرفض أو يُقتطع.")
        if st.button("🚀 نشر الآن", key="social_publish_btn", type="primary"):
            if not publish_text.strip():
                st.warning("أدخل نصاً أولاً.")
            elif not publish_platforms:
                st.warning("اختر منصة واحدة على الأقل.")
            else:
                with st.spinner("⟳ ينشر..."):
                    results = mgr.publish_to(publish_platforms, publish_text.strip())
                _pub_ok = sum(1 for r in results.values() if not str(r).startswith("ERROR"))
                st.toast(f"🚀 تم النشر على {_pub_ok}/{len(results)} منصة", icon="🚀")
                for pid, res in results.items():
                    label = PLATFORM_LABELS_AR.get(pid, pid)
                    if str(res).startswith("ERROR"):
                        st.error(f"{label}: {res}")
                    else:
                        st.success(f"{label}: ✅ {res}")

        st.markdown("---")
        st.markdown("#### 📅 جدولة المنشورات (تقويم المحتوى)")
        st.caption("⏰ الأوقات بتوقيت UTC — الخادم يعالج المنشور المستحق في أقرب دورة استطلاع.")
        sch_col1, sch_col2 = st.columns(2)
        with sch_col1:
            sch_date = st.date_input("تاريخ النشر", key="social_sched_date")
        with sch_col2:
            sch_time = st.time_input("وقت النشر (UTC)", key="social_sched_time")
        sch_text = st.text_area("نص المنشور المجدول", key="social_sched_text", height=80)
        sch_platforms = st.multiselect(
            "المنصات", options=list(PLATFORM_LABELS_AR.keys()),
            format_func=lambda p: PLATFORM_LABELS_AR.get(p, p), key="social_sched_platforms",
        )
        if st.button("📌 جدولة المنشور", key="social_sched_btn"):
            if not sch_text.strip():
                st.warning("أدخل نص المنشور أولاً.")
            elif not sch_platforms:
                st.warning("اختر منصة واحدة على الأقل.")
            else:
                with st.spinner("⟳ يجدول..."):
                    sched_dt = datetime.combine(sch_date, sch_time).isoformat() + "+00:00"
                    schedule_post(sch_platforms, sch_text.strip(), sched_dt)
                st.toast(f"📌 تمت الجدولة على {sched_dt}", icon="📌")
                st.rerun()

        scheduled = get_scheduled(status="pending")
        if scheduled:
            st.caption(f"**{len(scheduled)} منشور مجدول قيد الانتظار:**")
            for sid, plats, text, sched_at, sstatus, pub_at, result in scheduled:
                plat_names = "، ".join(PLATFORM_LABELS_AR.get(p, p) for p in plats)
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.caption(f"🕐 {sched_at} — {plat_names} — {text[:60]}")
                with c2:
                    if st.button("❌", key=f"cancel_sched_{sid}"):
                        cancel_scheduled(sid)
                        st.rerun()
        else:
            st.caption("لا توجد منشورات مجدولة حالياً.")

    # ═══════════════════════════════ 📈 تحليلات وأحداث ═══════════════════════════════
    with tab_insights:
        st.markdown("#### لوحة التحليلات (آخر 7 أيام)")
        analytics = get_analytics_summary(days=7)
        if not analytics:
            st.caption("لا توجد بيانات كافية بعد.")
        else:
            for pid, s in analytics.items():
                label = PLATFORM_LABELS_AR.get(pid, pid)
                total_sent = s["positive"] + s["negative"] + s["neutral"]
                sent_str = (
                    f"😊 {s['positive']} · 😐 {s['neutral']} · 😠 {s['negative']}"
                    if total_sent else "لا بيانات مشاعر"
                )
                st.markdown(
                    f"**{label}** — إشارات: {s['monitor_hit']} · ردود: {s['reply']} "
                    f"(فشل: {s['reply_failed']}) · منشورات: {s['publish']} (فشل: {s['publish_failed']})"
                )
                st.caption(f"المشاعر: {sent_str}")

        st.markdown("---")
        st.markdown("#### 🧾 آخر الأحداث")
        _EVENT_TYPE_AR = {
            "monitor_hit": "👁️ إشارة رُصدت",
            "reply": "💬 رد",
            "publish": "📤 نشر",
            "monitor_error": "⚠️ خطأ مراقبة",
        }
        events = get_recent_events(20)
        if not events:
            st.caption("لا توجد أحداث بعد.")
        else:
            for platform, event_type, author, content, reply_content, created_at, ok, sentiment, sentiment_score in events:
                label = PLATFORM_LABELS_AR.get(platform, platform)
                ev_label = _EVENT_TYPE_AR.get(event_type, event_type)
                status_icon = "✅" if ok else "❌"
                snippet = (content or "")[:80]
                line = f"{status_icon} {label} · {ev_label} · {created_at} — {snippet}"
                st.caption(line)



# ══════════════════════════════════════════════════════════════════════════
def render_unified_agent():
    """🎯 الوكيل الموحّد: واجهة محادثة واحدة مستمرة، توجّه كل رسالة تلقائياً
    خلف الكواليس لأنسب متخصص من AGENT_CATEGORIES (نفس منطق route_query_verbose
    المستخدَم أصلاً في 🤝 منسّق الوكلاء)، لكن بذاكرة مشتركة عبر كل الرسائل
    بدل عزل كل فئة بذاكرتها الخاصة — تجربة "وكيل واحد ذكي" حقيقية، بدل خلط
    كل الـ System Prompts في وكيل عام واحد (يُضعف دقة كل تخصص)."""
    import html as _html

    if not _AGENTS_HUB_OK:
        st.error("⚠️ تعذّر تحميل الوكيل الموحّد. تأكد من وجود ai/agent_categories.py.")
        return

    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <span style="font-size:2rem">🎯</span>
        <div style="font-size:1.5rem;font-weight:900;color:var(--gold)">الوكيل الموحّد</div>
        <div style="color:var(--text-muted);font-size:0.85rem;direction:rtl">
            محادثة واحدة مستمرة — كل رسالة تُوجَّه تلقائياً خلف الكواليس لأنسب متخصص،
            بذاكرة مشتركة تحافظ على سياق المحادثة عبر كل المواضيع
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    @keyframes uaBubbleIn { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:translateY(0);} }
    .ua-user {display:flex;justify-content:flex-end;margin:0.5rem 0;animation:uaBubbleIn .25s ease-out;}
    .ua-user .bbl {
        background:linear-gradient(135deg,var(--gold),var(--emerald));
        color:var(--bg);padding:0.7rem 1.05rem;border-radius:18px 18px 4px 18px;
        max-width:85%;font-size:0.96rem;line-height:1.7;text-align:right;direction:rtl;
        box-shadow:0 3px 12px var(--shadow);white-space:pre-wrap;word-break:break-word;font-weight:600;
    }
    .ua-bot {display:flex;justify-content:flex-start;margin:0.5rem 0;gap:0.5rem;align-items:flex-start;animation:uaBubbleIn .25s ease-out;}
    .ua-bot .bbl {
        background:var(--surface2);color:var(--text);padding:0.7rem 1.05rem;border-radius:18px 18px 18px 4px;
        max-width:85%;font-size:0.96rem;line-height:1.8;text-align:right;direction:rtl;
        border:1px solid var(--border);box-shadow:0 2px 8px var(--shadow);white-space:pre-wrap;word-break:break-word;
    }
    .ua-box {
        height:48vh;min-height:320px;max-height:520px;overflow-y:auto;padding:1rem;
        background:var(--bg);border-radius:16px;border:1px solid var(--border);margin-bottom:0.8rem;
        scroll-behavior:smooth;box-shadow:inset 0 0 20px var(--shadow);
    }
    .ua-badge {
        display:inline-block;background:var(--surface);border:1px solid var(--border);border-radius:20px;
        padding:0.15rem 0.65rem;font-size:0.72rem;color:var(--gold);direction:rtl;
    }
    </style>
    """, unsafe_allow_html=True)

    if "unified_agent_bot" not in st.session_state:
        st.session_state.unified_agent_bot = UnifiedAgentChat()
        st.session_state.unified_agent_msgs = []  # (role, text, badge, ts)
        st.session_state.unified_agent_count = 0

    bot = st.session_state.unified_agent_bot

    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.caption("مثال: اسأل سؤالاً برمجياً ثم اسأل سؤالاً تحليلياً في نفس المحادثة — الذاكرة تبقى مشتركة.")
    with col_s:
        st.metric("رسائل الجلسة", st.session_state.unified_agent_count)

    web_toggle = st.toggle(
        "🌐 بحث حقيقي في الويب قبل الرد",
        value=False, key="unified_agent_web",
        help="يفعّل بحثاً فعلياً عبر DuckDuckGo قبل توليد الرد، أياً كان المتخصص المُختار.",
    )

    box_id = "unified-agent-chat-box"
    html_out = f'<div class="ua-box" id="{box_id}">'
    if not st.session_state.unified_agent_msgs:
        html_out += (
            '<div style="text-align:center;color:var(--text-muted);padding:2rem 1rem">'
            '🎯<br><br>اكتب أي سؤال — سيُوجَّه تلقائياً لأنسب متخصص خلف الكواليس</div>'
        )
    else:
        for _mi, msg_tuple in enumerate(st.session_state.unified_agent_msgs):
            role, text, badge = msg_tuple[0], msg_tuple[1], msg_tuple[2]
            ts = msg_tuple[3] if len(msg_tuple) > 3 else ""
            ts_html = f'<div class="bbl-ts">{ts}</div>' if ts else ""
            safe = _html.escape(text).replace("\n", "<br>")
            if role == "user":
                html_out += f'<div class="ua-user"><div class="bbl">{safe}{ts_html}</div></div>'
            else:
                badge_html = f'<div class="ua-badge">{badge}</div>' if badge else ""
                bbl_id = f"{box_id}-msg-{_mi}"
                html_out += (
                    f'<div class="ua-bot"><div class="bbl">{badge_html}'
                    f'<div id="{bbl_id}">{safe}</div>{ts_html}</div></div>'
                )
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)
    st.components.v1.html(f"""
    <script>
    (function() {{
        function scrollToBottom() {{
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('{box_id}');
            if (box) {{ box.scrollTop = box.scrollHeight; return true; }}
            return false;
        }}
        let attempts = 0;
        const tryScroll = () => {{
            attempts++;
            if (!scrollToBottom() && attempts < 10) {{ setTimeout(tryScroll, 60); }}
        }};
        tryScroll();
    }})();
    </script>
    """, height=0)

    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        user_input = st.text_area(
            label="سؤالك", placeholder="اسأل أي شيء — سيُوجَّه تلقائياً لأنسب متخصص…",
            key="unified_agent_input", label_visibility="collapsed", height=88,
        )
    with c2:
        send = st.button("➤ إرسال", key="unified_agent_send", use_container_width=True)

    if send and user_input.strip():
        ts1 = datetime.now().strftime("%H:%M")
        st.session_state.unified_agent_msgs.append(("user", user_input.strip(), "", ts1))
        with st.spinner("⟳ يُوجَّه للمتخصص الأنسب ويولّد الرد..."):
            response, meta = bot.chat(user_input.strip(), force_web=web_toggle)
        badge = f"{meta.get('category_emoji', '🤖')} {meta.get('category_title', '')}"
        # 🆕 شارة جودة موحّدة (نفس ميزة تبويب "🤖 وكلاء AI")، معروضة الآن
        # أيضاً في الوكيل الموحّد — تُضاف فقط إن توفّر تقييم فعلاً.
        _qb = meta.get("quality_badge", "")
        if _qb:
            badge = f"{badge} · {_qb}"
        ts2 = datetime.now().strftime("%H:%M")
        st.session_state.unified_agent_msgs.append(("bot", response, badge, ts2))
        st.session_state.unified_agent_count += 1
        st.rerun()

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑 مسح المحادثة", key="unified_agent_clear", use_container_width=True):
            st.session_state.unified_agent_msgs = []
            st.session_state.unified_agent_count = 0
            bot.clear_history()
            st.rerun()
    with col_export:
        if st.session_state.unified_agent_msgs:
            _export_lines = ["# محادثة مع الوكيل الموحّد\n"]
            for _m in st.session_state.unified_agent_msgs:
                _role, _text = _m[0], _m[1]
                _badge = _m[2] if len(_m) > 2 else ""
                _ts = _m[3] if len(_m) > 3 else ""
                _who = "أنت" if _role == "user" else (_badge or "الوكيل")
                _export_lines.append(f"**{_who}** _{_ts}_\n\n{_text}\n\n---\n")
            st.download_button(
                "⬇️ تصدير المحادثة", data="\n".join(_export_lines).encode("utf-8"),
                file_name="محادثة_الوكيل_الموحد.md", mime="text/markdown",
                key="unified_agent_export", use_container_width=True,
            )
        else:
            st.button("⬇️ تصدير المحادثة", disabled=True, use_container_width=True,
                       key="unified_agent_export_disabled", help="لا توجد رسائل بعد")


def render_agents_hub():
    """يعرض تبويباً فرعياً مستقلاً لكل فئة من وكلاء الذكاء الاصطناعي المتخصصين."""

    if not _AGENTS_HUB_OK:
        st.error("⚠️ تعذّر تحميل وكلاء AI. تأكد من وجود ai/agent_categories.py.")
        return

    st.markdown("### 🤖 وكلاء AI المتخصصون")
    st.caption("كل فئة لها وكيلها الخاص، بذاكرة محادثة مستقلة، ومزوّد LLM نفسه المُستخدَم في المشروع.")

    # CSS مشترك لكل فقاعات المحادثة داخل هذا التبويب (نفس أسلوب تبويب المحادثة)
    st.markdown("""
    <style>
    @keyframes agentBubbleIn {
        from {opacity:0;transform:translateY(6px);}
        to   {opacity:1;transform:translateY(0);}
    }
    .agent-user {display:flex;justify-content:flex-end;margin:0.5rem 0;animation:agentBubbleIn .25s ease-out;}
    .agent-user .bbl {
        background:linear-gradient(135deg,var(--gold),var(--emerald));
        color:var(--bg);padding:0.7rem 1.05rem;border-radius:18px 18px 4px 18px;
        max-width:85%;font-size:0.96rem;line-height:1.7;text-align:right;direction:rtl;
        box-shadow:0 3px 12px var(--shadow);white-space:pre-wrap;word-break:break-word;
        font-weight:600;
    }
    .agent-bot {display:flex;justify-content:flex-start;margin:0.5rem 0;gap:0.5rem;align-items:flex-start;animation:agentBubbleIn .25s ease-out;}
    .agent-bot .bbl {
        background:var(--surface2);
        color:var(--text);padding:0.7rem 1.05rem;border-radius:18px 18px 18px 4px;
        max-width:85%;font-size:0.96rem;line-height:1.8;text-align:right;direction:rtl;
        border:1px solid var(--border);box-shadow:0 2px 8px var(--shadow);
        white-space:pre-wrap;word-break:break-word;
    }
    .agent-box {
        height:48vh;min-height:320px;max-height:520px;overflow-y:auto;padding:1rem;
        background:var(--bg);border-radius:16px;border:1px solid var(--border);margin-bottom:0.8rem;
        scroll-behavior:smooth;box-shadow:inset 0 0 20px var(--shadow);
    }
    .agent-badge {
        display:inline-block;background:var(--surface);border:1px solid var(--border);border-radius:20px;
        padding:0.15rem 0.65rem;font-size:0.72rem;color:var(--gold);direction:rtl;
    }
    </style>
    """, unsafe_allow_html=True)

    labels = [
        f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}" for k in CATEGORY_ORDER
    ]
    sub_tabs = st.tabs(labels)

    for i, key in enumerate(CATEGORY_ORDER):
        with sub_tabs[i]:
            _render_agent_page(AGENT_CATEGORIES[key])


def _render_agent_page(category):
    """يعرض صفحة وكيل واحد: محادثة معزولة + أسئلة سريعة خاصة بفئته."""
    import html as _html

    bot_key  = f"agent_bot_{category.key}"
    msg_key  = f"agent_msgs_{category.key}"
    cnt_key  = f"agent_count_{category.key}"

    if bot_key not in st.session_state:
        st.session_state[bot_key] = CategoryAgentChat(category.key)
        st.session_state[msg_key] = []
        st.session_state[cnt_key] = 0

    bot = st.session_state[bot_key]

    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.markdown(f"#### {category.emoji} {category.title}")
        st.caption(category.subtitle)
    with col_s:
        st.metric("رسائل الجلسة", st.session_state[cnt_key])

    web_toggle = st.toggle(
        "🌐 بحث حقيقي في الويب قبل الرد",
        value=getattr(category, "web_enabled", False),
        key=f"agent_web_{category.key}",
        help="يفعّل بحثاً فعلياً عبر DuckDuckGo قبل توليد الرد، بغض النظر عن الفئة.",
    )

    box_id = f"agent-chat-box-{category.key}"
    html_out = f'<div class="agent-box" id="{box_id}">'
    if not st.session_state[msg_key]:
        html_out += (
            f'<div style="text-align:center;color:var(--text-muted);padding:2rem 1rem">'
            f'{category.emoji}<br><br>ابدأ محادثتك مع وكيل {category.title}</div>'
        )
    else:
        for _mi, msg_tuple in enumerate(st.session_state[msg_key]):
            role, text, badge = msg_tuple[0], msg_tuple[1], msg_tuple[2]
            ts = msg_tuple[3] if len(msg_tuple) > 3 else ""
            ts_html = f'<div class="bbl-ts">{ts}</div>' if ts else ""
            safe = _html.escape(text).replace("\n", "<br>")
            if role == "user":
                html_out += f'<div class="agent-user"><div class="bbl">{safe}{ts_html}</div></div>'
            else:
                badge_html = f'<div class="agent-badge">{badge}</div>' if badge else ""
                bbl_id = f"{box_id}-msg-{_mi}"
                html_out += (
                    f'<div class="agent-bot"><span style="font-size:1.3rem;margin-top:3px">'
                    f'{category.emoji}</span><div class="bbl">{badge_html}'
                    f'<div id="{bbl_id}">{safe}</div>'
                    f'<button class="copy-btn" title="نسخ الرد" style="margin-top:0.4rem"'
                    f' onclick="var t=document.getElementById(\'{bbl_id}\').innerText;'
                    f"navigator.clipboard.writeText(t).then(function(){{"
                    f"var b=event.currentTarget;var old=b.textContent;b.textContent='✓ تم النسخ';"
                    f"setTimeout(function(){{b.textContent=old;}},1300);}});\">📋 نسخ</button>"
                    f'{ts_html}</div></div>'
                )
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)
    st.components.v1.html(f"""
    <script>
    (function() {{
        function scrollToBottom() {{
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('{box_id}');
            if (box) {{ box.scrollTop = box.scrollHeight; return true; }}
            return false;
        }}
        let attempts = 0;
        const tryScroll = () => {{
            attempts++;
            if (!scrollToBottom() && attempts < 10) {{ setTimeout(tryScroll, 60); }}
        }};
        tryScroll();
    }})();
    </script>
    """, height=0)

    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        user_input = st.text_area(
            label="سؤالك", placeholder=f"اسأل وكيل {category.title}…",
            key=f"agent_input_{category.key}", label_visibility="collapsed", height=88,
        )
    with c2:
        send = st.button("➤ إرسال", key=f"agent_send_{category.key}", use_container_width=True)

    # ── مشاركة ملف مع الوكيل (اختياري): نص، PDF، أو صورة (عبر OCR) ──
    _uploader_types = ["txt", "py", "md", "json", "csv", "log", "yaml", "yml", "pdf"]
    if _OCR_OK:
        _uploader_types += ["png", "jpg", "jpeg"]
    uploaded_file = st.file_uploader(
        "📎 أرفق ملفاً ليطّلع عليه الوكيل قبل الرد — نص/PDF" + (
            "/صورة (OCR)" if _OCR_OK else ""
        ) + " (اختياري)",
        type=_uploader_types,
        key=f"agent_file_{category.key}",
    )
    _MAX_FILE_CHARS = 6000
    file_context, file_label = "", ""
    if uploaded_file is not None:
        _extracted = _extract_file(uploaded_file)
        if _extracted is None:
            st.warning(f"⚠️ الملف أكبر من {MAX_FILE_MB}MB — لم يُرفَع.")
        elif _extracted.get("is_image"):
            _ocr_text = _ocr_image_text(_extracted.get("raw_bytes", b""))
            if _ocr_text:
                file_context = _ocr_text[:_MAX_FILE_CHARS]
                file_label = f"🖼️ {uploaded_file.name} (نص مستخرَج بـ OCR)"
                st.caption(f"{file_label} — سيُرسَل مع رسالتك التالية للوكيل.")
            else:
                st.caption(f"🖼️ {uploaded_file.name} — لم يُستخرَج نص من الصورة (قد تكون بلا نص واضح).")
        else:
            _raw_text = (_extracted.get("text_content") or "").strip()
            if _raw_text:
                _truncated = len(_raw_text) > _MAX_FILE_CHARS
                file_context = _raw_text[:_MAX_FILE_CHARS]
                file_label = f"📎 {uploaded_file.name}" + (" (مقتطع للطول)" if _truncated else "")
                st.caption(f"{file_label} — سيُرسَل محتواه مع رسالتك التالية للوكيل.")

    if category.quick_prompts:
        st.markdown("**⚡ أسئلة سريعة:**")
        qcols = st.columns(len(category.quick_prompts))
        for i, q in enumerate(category.quick_prompts):
            with qcols[i]:
                if st.button(q, key=f"agent_q_{category.key}_{i}", use_container_width=True):
                    st.session_state[f"_agent_pending_{category.key}"] = q

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑 مسح المحادثة", key=f"agent_clear_{category.key}", use_container_width=True):
            st.session_state[msg_key] = []
            st.session_state[cnt_key] = 0
            bot.clear_history()
            st.rerun()
    with col_export:
        if st.session_state[msg_key]:
            _export_lines = [f"# محادثة مع وكيل {category.title}\n"]
            for _m in st.session_state[msg_key]:
                _role, _text = _m[0], _m[1]
                _ts = _m[3] if len(_m) > 3 else ""
                _who = "أنت" if _role == "user" else category.title
                _export_lines.append(f"**{_who}** _{_ts}_\n\n{_text}\n\n---\n")
            st.download_button(
                "⬇️ تصدير المحادثة", data="\n".join(_export_lines).encode("utf-8"),
                file_name=f"محادثة_{category.key}.md", mime="text/markdown",
                key=f"agent_export_{category.key}", use_container_width=True,
            )
        else:
            st.button("⬇️ تصدير المحادثة", disabled=True, use_container_width=True,
                       key=f"agent_export_disabled_{category.key}", help="لا توجد رسائل بعد")

    if st.session_state[msg_key]:
        with st.expander(f"📜 سجل الجلسة ({st.session_state[cnt_key]} تبادل)"):
            for _m in st.session_state[msg_key]:
                _role, _text = _m[0], _m[1]
                _ts = _m[3] if len(_m) > 3 else ""
                _tag = "🧑" if _role == "user" else category.emoji
                _preview = _text if len(_text) <= 140 else _text[:140] + "…"
                st.caption(f"{_tag} `{_ts}` — {_preview}")

    def _process(text: str):
        if not text.strip():
            return
        _ts_now = datetime.now().strftime("%H:%M")
        _display_text = text.strip()
        if file_label:
            _display_text = f"{_display_text}\n\n{file_label}"
        st.session_state[msg_key].append(("user", _display_text, "", _ts_now))

        _safety_msg = _nsm_safety_gate(text.strip())
        if _safety_msg:
            st.session_state[msg_key].append(("bot", _safety_msg, "🛡️ فحص أمان", datetime.now().strftime("%H:%M")))
            st.session_state[cnt_key] += 1
            st.rerun()
            return

        _query = text.strip()
        if file_context:
            _query = (
                f"محتوى الملف المرفق ({uploaded_file.name if uploaded_file else 'ملف'}):\n"
                f"```\n{file_context}\n```\n\nسؤال/طلب المستخدم:\n{text.strip()}"
            )

        with st.spinner(f"⟳ {category.title} يفكّر..."):
            response = bot.chat(_query, force_web=web_toggle, source="hub")
        badge = bot.last_provider_badge()
        try:
            from ai.response_quality import score_response
            _q = score_response(response, query=text.strip())
            badge = f"{badge} · 🔎 {_q.as_percent()}٪ {_q.label}" if badge else f"🔎 {_q.as_percent()}٪ {_q.label}"
        except Exception:
            pass  # تقييم الجودة إضافي وغير حرج — أي فشل فيه لا يجب أن يُسقِط الرد نفسه
        st.session_state[msg_key].append(("bot", response, badge, datetime.now().strftime("%H:%M")))
        st.session_state[cnt_key] += 1
        st.rerun()

    if send and user_input:
        _process(user_input)

    pending_key = f"_agent_pending_{category.key}"
    if pending_key in st.session_state:
        q = st.session_state[pending_key]
        del st.session_state[pending_key]
        _process(q)


# ══════════════════════════════════════════════════════════════════════════
# تبويب ⚙️ النظام الداخلي — النواة العصبية + الوعي الذاتي + مخطط الأهداف
# ══════════════════════════════════════════════════════════════════════════
def render_system_core():
    """ربط الوحدات الداخلية الأساسية بالواجهة."""
    st.markdown('<div class="section-header">⚙️ النظام الداخلي — Neural Core & Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl">هذا التبويب يعرض الوحدات الداخلية للنظام: '
        'النواة العصبية، الوعي الذاتي، مخطط الأهداف، والمفكر الفوقي.</p>',
        unsafe_allow_html=True,
    )

    core_tabs = st.tabs([
        "🧠 النواة العصبية",
        "👁️ الوعي الذاتي",
        "🎯 مخطط الأهداف",
        "🔬 التحليل اللغوي",
        "🌐 بحث الويب المباشر",
    ])

    # ══════════════════ 1. النواة العصبية ══════════════════
    with core_tabs[0]:
        st.markdown('<div class="section-header">🧠 النواة العصبية (Neural Core)</div>',
                    unsafe_allow_html=True)
        if not _NEURAL_CORE_OK:
            st.error("⚠️ تعذّر تحميل NeuralCore — تأكد من تثبيت numpy.")
        else:
            try:
                # ── النواة الحية المشتركة (نفس singleton الذي يستخدمه ──
                # ReasoningPipeline فعلياً في مسار الاستدلال الحي، بنفس
                # مسار الحفظ models/neural_core. أي تدريب هنا يُحدِّث
                # نفس الكائن الحي بالذاكرة، ونفس الملف عند الحفظ.
                from ai.neural_core import get_default_core, DEFAULT_INPUT_DIM, \
                    DEFAULT_HIDDEN_DIMS, DEFAULT_OUTPUT_DIM
                _nc_path = "models/neural_core"
                _nc = get_default_core(
                    _nc_path,
                    input_dim=DEFAULT_INPUT_DIM,
                    hidden_dims=list(DEFAULT_HIDDEN_DIMS),
                    output_dim=DEFAULT_OUTPUT_DIM,
                )
                _nc_info = _nc.get_info()

                if os.path.exists(os.path.join(_nc_path, "network.json")):
                    st.caption(f"📂 النواة الحية — مُحمَّلة من `{_nc_path}` (نفس النواة التي يستخدمها الاستدلال الحقيقي)")
                else:
                    st.caption("🆕 نواة جديدة (لا يوجد ملف محفوظ بعد) — L1 المدروسة 784×784 محمّلة تلقائياً")

                col_nc1, col_nc2, col_nc3, col_nc4 = st.columns(4)
                with col_nc1:
                    metric_card(_nc_info.get("total_parameters", "—"), "إجمالي المعاملات")
                with col_nc2:
                    metric_card(_nc_info.get("train_steps", 0), "خطوات التدريب")
                with col_nc3:
                    metric_card(len(_nc_info.get("architecture", [])), "عدد الطبقات")
                with col_nc4:
                    mem_size = _nc_info.get("memory_size", 0)
                    metric_card(mem_size, "حجم الذاكرة الترابطية")

                st.markdown("")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**معمارية الشبكة:**")
                    arch = _nc_info.get("architecture", [])
                    for i, layer in enumerate(arch):
                        st.markdown(f"""
                        <div class="root-item">
                            <span class="badge badge-blue">طبقة {i+1}</span>
                            &nbsp;{layer.get('type','—')} &nbsp;
                            <span class="badge badge-purple">{layer.get('input_dim','?')} → {layer.get('output_dim','?')}</span>
                            &nbsp;<small>{layer.get('activation','')}</small>
                        </div>
                        """, unsafe_allow_html=True)

                with col_right:
                    st.markdown("**حالة النواة:**")
                    last_loss = _nc_info.get("last_loss")
                    best_loss = _nc_info.get("best_loss")
                    lr        = _nc_info.get("learning_rate", 0.01)
                    st.markdown(f"""
                    <div class="root-item">
                        <strong>معدل التعلم:</strong> {lr}<br>
                        <strong>آخر خسارة:</strong> {f"{last_loss:.6f}" if last_loss else "لا يوجد"}<br>
                        <strong>أفضل خسارة:</strong> {f"{best_loss:.6f}" if best_loss else "لا يوجد"}
                    </div>
                    """, unsafe_allow_html=True)

                # اختبار تمرير أمامي
                st.markdown("")
                st.markdown("**اختبار التمرير الأمامي:**")
                import numpy as np
                _test_input = np.random.randn(784)
                _output = _nc.forward(_test_input)
                _out_str = "، ".join(f"{v:.4f}" for v in _output)
                st.code(f"مدخل: متجه عشوائي (784 بُعد)\nمخرج (4 فئات): [{_out_str}]", language="text")
                st.success("✅ النواة العصبية تعمل بشكل صحيح")

                # ── تدريب فعلي من التجارب الحقيقية (بدون تخزين بيانات خام) ──
                st.markdown("---")
                st.markdown("**🎓 تدريب من التجارب الحقيقية (Experience Replay)**")
                st.caption(
                    "يتدرّب على حلقات حقيقية من استخدام النظام الفعلي "
                    "(memory/experience.db) عبر train_step() + evolve_if_plateau() — "
                    "تحديث أوزان ونمو هيكلي فعلي، **بدون** تخزين أي متجهات خام "
                    "بالذاكرة الترابطية."
                )
                _replay_strategy = st.selectbox(
                    "استراتيجية الاختيار:",
                    ["الأحدث (recent)", "الأعلى جودة (top)", "متنوعة (diverse)"],
                    key="nc_replay_strategy",
                )
                if st.button("🎓 ابدأ التدريب الآن", key="nc_train_btn"):
                    try:
                        from ai.experience_trainer import ExperienceTrainer
                        from ai.experience_store import EpisodeStore
                        _params_before = _nc_info.get("total_parameters", 0)
                        _store = EpisodeStore()
                        _trainer = ExperienceTrainer(core=_nc, store=_store)
                        if _replay_strategy.startswith("الأعلى"):
                            _report = _trainer.replay_top(limit=20)
                        elif _replay_strategy.startswith("متنوعة"):
                            _report = _trainer.replay_diverse(limit=20)
                        else:
                            _report = _trainer.replay_recent(limit=20)

                        if _report.episodes_used == 0:
                            st.warning(
                                "⚠️ لا توجد تجارب حقيقية محفوظة بعد (0 حلقة) في "
                                "memory/experience.db — النواة تتعلم تلقائياً من "
                                "الاستخدام الحقيقي للنظام (أسئلة حقيقية عبر "
                                "ReasoningPipeline)، لا يوجد بعد ما تتدرّب عليه."
                            )
                        else:
                            _params_after = _nc.get_info().get("total_parameters", 0)
                            _grew = _params_after > _params_before
                            st.success(
                                f"✅ تدرّبت على {_report.episodes_used} حلقة حقيقية — "
                                f"الخسارة: {_report.avg_loss_before:.6f} → {_report.avg_loss_after:.6f}"
                            )
                            if _grew:
                                st.info(
                                    f"📈 النواة توسّعت فعلياً: {_params_before:,} → "
                                    f"{_params_after:,} معامل (نمو هيكلي بسبب ركود الخسارة)"
                                )
                    except Exception as _train_err:
                        st.error(f"فشل التدريب: {_train_err}")

                st.markdown("")
                if st.button("💾 حفظ الأوزان فقط (بدون بيانات خام)", key="nc_save_ckpt"):
                    try:
                        _saved_path = None
                        try:
                            from ai.rollback_guard import CheckpointGuard
                            _guard = CheckpointGuard(asset="neural_core_weights")
                            _last_loss = _nc.get_info().get("last_loss")
                            _guard_files = [
                                f"{_nc_path}/network.json",
                                f"{_nc_path}/core_state.json",
                            ]

                            def _do_nc_save():
                                nonlocal _saved_path
                                _saved_path = _nc.save(_nc_path, include_memory=False)
                                return _saved_path

                            if _last_loss is not None:
                                _decision = _guard.guarded_update(
                                    files=_guard_files,
                                    update_fn=_do_nc_save,
                                    eval_fn=lambda: -float(_last_loss),
                                    tolerance=-0.05,
                                    label=f"حفظ يدوي (last_loss={_last_loss:.6f})",
                                )
                                if _decision.rolled_back:
                                    st.error(
                                        f"⚠️ رُفض الحفظ تلقائياً وأُعيدت الأوزان السابقة "
                                        f"(محمي بـ RollbackGuard) — جودة الحفظ الجديد "
                                        f"({-_decision.new_score:.6f} خسارة) أسوأ من "
                                        f"المحفوظة سابقاً ({-_decision.old_score:.6f} خسارة)."
                                    )
                                else:
                                    st.success(
                                        f"✅ تم حفظ الأوزان بأمان (محمي من التراجع) → "
                                        f"`{_saved_path}`"
                                    )
                            else:
                                # لا يوجد last_loss بعد (لم يُدرَّب النموذج في هذه
                                # الجلسة) — لا أساس مقارنة، فنأخذ لقطة احتياطية
                                # يدوية فقط قبل الحفظ العادي تحسباً لأي مشكلة.
                                _guard.snapshot(_guard_files, label="حفظ يدوي (بدون last_loss)")
                                _do_nc_save()
                                st.success(
                                    f"✅ تم حفظ الأوزان والحالة الهيكلية فقط → "
                                    f"`{_saved_path}` (أُخذت لقطة احتياطية أولاً)"
                                )
                        except ImportError:
                            # rollback_guard غير متاح لأي سبب — احفظ عادياً كما
                            # كان يحدث قبل هذا الربط، بدون حماية.
                            _saved_path = _nc.save(_nc_path, include_memory=False)
                            st.success(f"✅ تم حفظ الأوزان والحالة الهيكلية فقط → `{_saved_path}`")
                    except Exception as _save_err:
                        st.error(f"فشل الحفظ: {_save_err}")

            except Exception as _nc_err:
                st.error(f"خطأ في NeuralCore: {_nc_err}")

    # ══════════════════ 2. الوعي الذاتي ══════════════════
    with core_tabs[1]:
        st.markdown('<div class="section-header">👁️ الوعي الذاتي (Self-Awareness Engine)</div>',
                    unsafe_allow_html=True)
        if not _SELF_AWARE_OK:
            st.error("⚠️ تعذّر تحميل SelfAwarenessEngine.")
        else:
            try:
                _ckg   = load_ckg()
                _roots = load_arabic_roots()
                _ep    = get_episodic_stats()
                _ckpt  = load_latest_checkpoint()

                _sa_engine = SelfAwarenessEngine()
                _report    = _sa_engine.introspect()
                _rd = _report.to_dict()
                # إثراء التقرير ببيانات CKG المحلية
                if _rd.get("node_count", 0) == 0:
                    _rd["node_count"] = len(_ckg.get("concepts", {}))
                if _rd.get("edge_count", 0) == 0:
                    _rd["edge_count"] = len(_ckg.get("relations", {}))

                # مقاييس رئيسية
                score = _rd.get("system_health_score", 0.0)
                readiness = _rd.get("phase7_readiness", 0.0)
                col_sa1, col_sa2, col_sa3 = st.columns(3)
                with col_sa1:
                    metric_card(f"{score:.0%}", "درجة صحة النظام")
                with col_sa2:
                    metric_card(f"{readiness:.0%}", "جاهزية Phase 7")
                with col_sa3:
                    metric_card(_rd.get("node_count", 0), "عدد العقد (المفاهيم)")

                st.markdown("")

                # الأهداف الحالية
                objectives = _rd.get("current_objectives", [])
                if objectives:
                    st.markdown('<div class="section-header">🎯 الأهداف الحالية</div>',
                                unsafe_allow_html=True)
                    for obj in objectives:
                        st.markdown(f"""
                        <div class="root-item">
                            <span style="font-size:1.1rem">🎯</span> {obj}
                        </div>
                        """, unsafe_allow_html=True)

                # القدرات المعروفة
                capabilities = _rd.get("known_capabilities", [])
                if capabilities:
                    st.markdown('<div class="section-header">✅ القدرات المعروفة</div>',
                                unsafe_allow_html=True)
                    caps_html = " ".join(
                        f'<span class="badge badge-green" style="margin:3px;font-size:0.85rem">{c}</span>'
                        for c in capabilities
                    )
                    st.markdown(caps_html, unsafe_allow_html=True)

                # الرؤى والتوصيات
                insights = _rd.get("insights", [])
                if insights:
                    st.markdown('<div class="section-header">💡 رؤى النظام</div>',
                                unsafe_allow_html=True)
                    for ins in insights:
                        st.info(ins)

                # شريط الصحة
                st.markdown("")
                st.markdown(f"**درجة الصحة الكلية:** {score:.0%}")
                st.progress(score)
                st.markdown(f"**جاهزية Phase 7:** {readiness:.0%}")
                st.progress(readiness)

            except Exception as _sa_err:
                st.error(f"خطأ في Awareness Engine: {_sa_err}")

    # ══════════════════ 3. مخطط الأهداف ══════════════════
    with core_tabs[2]:
        st.markdown('<div class="section-header">🎯 مخطط الأهداف (Goal Planner)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">حدّد هدفاً بالعربية وسيبني النظام خطة تنفيذ تلقائية.</p>',
            unsafe_allow_html=True,
        )

        if not _GOAL_PLANNER_OK:
            st.error("⚠️ تعذّر تحميل GoalPlanner.")
        else:
            _gp_examples = [
                "تلخيص مفاهيم سورة البقرة",
                "إيجاد العلاقة بين الصبر والإيمان",
                "تحليل مفهوم العدل في القرآن",
                "استخراج قصص الأنبياء من الآيات",
            ]
            st.markdown("**أمثلة:**")
            _gp_ex_cols = st.columns(len(_gp_examples))
            _gp_chosen = None
            for _i, _ex in enumerate(_gp_examples):
                with _gp_ex_cols[_i]:
                    if st.button(_ex, key=f"gp_ex_{_i}", use_container_width=True):
                        _gp_chosen = _ex

            _gp_goal = st.text_input(
                "اكتب هدفك:",
                value=_gp_chosen or st.session_state.get("gp_goal", ""),
                placeholder="مثال: تلخيص مفاهيم سورة البقرة",
                key="gp_goal_input",
            )
            st.session_state["gp_goal"] = _gp_goal

            _gp_run = st.button("🎯 بناء خطة التنفيذ", type="primary", key="gp_run")

            if _gp_run and _gp_goal.strip():
                with st.spinner("⟳ يبني النظام خطة التنفيذ..."):
                    try:
                        _planner = GoalPlanner()
                        _plan = _planner.plan(_gp_goal.strip())
                        if _plan is None:
                            st.warning("لم يُمكن بناء خطة لهذا الهدف — لا توجد عقد كافية في السجل.")
                        else:
                            _plan_d = _plan.to_dict()

                            st.markdown('<div class="section-header">📋 خطة التنفيذ</div>',
                                        unsafe_allow_html=True)

                            _p_cols = st.columns(3)
                            with _p_cols[0]:
                                metric_card(f"{_plan_d.get('confidence', 0):.0%}", "درجة الثقة")
                            with _p_cols[1]:
                                metric_card(len(_plan_d.get("path", [])), "عدد الخطوات")
                            with _p_cols[2]:
                                metric_card(_plan_d.get("status", "—"), "الحالة")

                            _path = _plan_d.get("path", [])
                            if _path:
                                st.markdown("")
                                st.markdown("**مسار التنفيذ:**")
                                for _step_i, _step in enumerate(_path):
                                    st.markdown(f"""
                                    <div class="root-item">
                                        <span class="badge badge-blue">خطوة {_step_i+1}</span>
                                        &nbsp;<strong>{_step}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)

                            _reasoning = _plan_d.get("reasoning", [])
                            if _reasoning:
                                with st.expander("🔍 تفاصيل المنطق"):
                                    for _r in _reasoning:
                                        st.markdown(f"- {_r}")

                    except Exception as _gp_err:
                        st.error(f"خطأ في GoalPlanner: {_gp_err}")

    # ══════════════════ 4. التحليل اللغوي ══════════════════
    with core_tabs[3]:
        st.markdown('<div class="section-header">🔬 محرك اللغة العربية (ArabicNLP)</div>',
                    unsafe_allow_html=True)
        if not _ARABIC_NLP_OK:
            st.error("⚠️ تعذّر تحميل ArabicNLPEngine.")
        else:
            _nlp_input = st.text_area(
                "أدخل نصاً عربياً للتحليل:",
                placeholder="مثال: الصبر مفتاح الفرج، والإيمان نور يهدي القلوب إلى الحق.",
                height=100,
                key="nlp_core_input",
            )
            _nlp_run = st.button("🔬 حلّل النص", type="primary", key="nlp_core_run")

            if _nlp_run and _nlp_input.strip():
                with st.spinner("⟳ يحلل النص..."):
                    try:
                        _nlp_e  = get_arabic_engine(ckg=load_ckg())
                        _res    = _nlp_e.analyse(_nlp_input.strip())
                        _fv     = _res.feature_vector

                        st.markdown("**متجه الخصائص (Feature Vector):**")
                        _fv_col1, _fv_col2, _fv_col3, _fv_col4 = st.columns(4)
                        with _fv_col1:
                            st.metric("نسبة الأفعال", f"{_fv.verb_score:.0%}")
                            st.metric("نسبة الأسماء", f"{_fv.noun_score:.0%}")
                        with _fv_col2:
                            st.metric("تعقيد الجذور", f"{_fv.root_complexity:.0%}")
                            st.metric("أنماط الصرف", f"{_fv.morpho_pattern_score:.0%}")
                        with _fv_col3:
                            st.metric("الكثافة الدلالية", f"{_fv.semantic_concept_score:.0%}")
                            st.metric("تماسك السياق", f"{_fv.context_score:.0%}")
                        with _fv_col4:
                            st.metric("التعقيد النحوي", f"{_fv.syntactic_complexity:.0%}")
                            st.metric("طول المتجه", len(_fv.to_list()))

                        st.markdown("")

                        # الطبقة النحوية
                        _syn = _res.syntactic
                        if _syn.tokens:
                            st.markdown('<div class="section-header">📝 الطبقة النحوية</div>',
                                        unsafe_allow_html=True)
                            _tok_html = " ".join(
                                f'<span class="badge badge-{"blue" if t.is_verb else "purple" if t.is_noun else "amber"}" style="margin:3px;padding:4px 10px;font-size:0.9rem" title="{"فعل" if t.is_verb else "اسم" if t.is_noun else "أداة"}">{t.surface}</span>'
                                for t in _syn.tokens[:30]
                            )
                            st.markdown(_tok_html, unsafe_allow_html=True)
                            st.caption("🔵 فعل | 🟣 اسم | 🟡 أداة/حرف")

                        # الطبقة الصرفية
                        _morph = _res.morphological
                        if _morph.roots_found:
                            st.markdown('<div class="section-header">🌿 الطبقة الصرفية</div>',
                                        unsafe_allow_html=True)
                            _roots_html = " ".join(
                                f'<span class="badge badge-green" style="margin:3px">√ {r}</span>'
                                for r in _morph.roots_found[:15]
                            )
                            st.markdown(_roots_html, unsafe_allow_html=True)

                        # الطبقة الدلالية
                        _sem = _res.semantic
                        if hasattr(_sem, "concepts_found") and _sem.concepts_found:
                            st.markdown('<div class="section-header">💡 المفاهيم الدلالية</div>',
                                        unsafe_allow_html=True)
                            _con_html = " ".join(
                                f'<span class="badge badge-purple" style="margin:3px">{c}</span>'
                                for c in _sem.concepts_found[:15]
                            )
                            st.markdown(_con_html, unsafe_allow_html=True)

                    except Exception as _nlp_err2:
                        st.error(f"خطأ في التحليل: {_nlp_err2}")

    # ══════════════════ 5. بحث الويب المباشر ══════════════════
    with core_tabs[4]:
        st.markdown('<div class="section-header">🌐 بحث الويب الحقيقي (DuckDuckGo)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">بحث حقيقي في الإنترنت بدون مفتاح API — '
            'يستخدم DuckDuckGo ويُرجع نتائج فعلية.</p>',
            unsafe_allow_html=True,
        )

        if not _WEB_SEARCH_OK:
            st.error("⚠️ تعذّر تحميل web_search_tool.")
        else:
            _ws_direct_q = st.text_input(
                "ابحث في الإنترنت:",
                placeholder="مثال: أحدث نماذج الذكاء الاصطناعي 2026، أو: ما هو الإسلام؟",
                key="ws_direct_input",
            )
            _ws_direct_n = st.slider("عدد النتائج", 3, 10, 5, key="ws_direct_n")
            _ws_direct_btn = st.button("🔍 ابحث الآن", type="primary", key="ws_direct_btn",
                                        use_container_width=True)

            if _ws_direct_btn and _ws_direct_q.strip():
                with st.spinner("⟳ يبحث في الإنترنت..."):
                    _ws_out = _web_search(_ws_direct_q.strip(), max_results=_ws_direct_n)

                st.markdown('<div class="section-header">📋 النتائج</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                            padding:1.2rem 1.5rem;direction:rtl;line-height:2.0;
                            white-space:pre-wrap;font-size:0.95rem;border:1px solid #1e3a5f">
                {_ws_out}
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "⬇️ تحميل النتائج",
                    data=_ws_out,
                    file_name="web_search_results.txt",
                    mime="text/plain",
                    key="ws_download",
                )


# ══════════════════════════════════════════════════════════════════════════
# تبويب 🤝 منسّق الوكلاء — توزيع مهمة واحدة على وكلاء Agents Hub الفعليين
# ══════════════════════════════════════════════════════════════════════════
def render_agent_orchestrator():
    """يوجّه مهمة/سؤال المستخدم تلقائياً إلى وكيل أو أكثر من وكلاء
    "🤖 وكلاء AI" الفعليين (نفس جلسات session_state وذاكرة المحادثة
    المستخدَمة في تبويب Agents Hub)، ثم يعرض ردودهم، مع توليف اختياري
    لإجابة موحّدة. يطبّق نمط Multi-Agent Systems: تفويض مهمة رئيسية إلى
    وكلاء متخصصين ثم تجميع نتائجهم عبر وكيل "منسّق"."""
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <span style="font-size:2rem">🤝</span>
        <div style="font-size:1.5rem;font-weight:900;color:var(--gold)">
            منسّق الوكلاء
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;direction:rtl">
            وزّع مهمتك تلقائياً على وكلاء "🤖 وكلاء AI" المتخصصين، ثم احصل على إجابة موحّدة
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _AGENTS_HUB_OK or not _ORCHESTRATOR_OK:
        st.error("⚠️ تعذّر تحميل وحدات الوكلاء (ai/agent_categories.py أو ai/godmode.py).")
        return

    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl">اكتب مهمة أو سؤالاً مركّباً، وسيُحدَّد تلقائياً '
        'أنسب وكيل/وكلاء من تبويب "🤖 وكلاء AI" للإجابة عليه — بنفس ذاكرة محادثتهم الفعلية. '
        'يمكنك أيضاً اختيار الوكلاء يدوياً.</p>',
        unsafe_allow_html=True,
    )

    manual = st.multiselect(
        "اختر وكلاء يدوياً (اختياري — إن تُرك فارغاً يتم التوجيه التلقائي):",
        options=CATEGORY_ORDER,
        format_func=lambda k: f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}",
        key="orch_manual_agents",
    )

    task = st.text_area(
        "المهمة أو السؤال:",
        placeholder="مثال: راجع خطة إطلاق ميزة جديدة من ناحية الأتمتة والتحليل والمخاطر",
        key="orch_task_input",
        height=100,
    )

    synth = st.checkbox("🧩 وَلِّف الردود في إجابة واحدة موحّدة", value=True, key="orch_synth")

    exec_mode = st.radio(
        "نمط التنفيذ:",
        options=["parallel", "sequential"],
        format_func=lambda m: (
            "⚡ متوازٍ — كل وكيل يجيب على المهمة الأصلية بشكل مستقل"
            if m == "parallel" else
            "🔗 متسلسل — كل وكيل يبني على ردود الوكلاء السابقين (سير عمل أعمق)"
        ),
        index=0,
        key="orch_exec_mode",
        help=(
            "متوازٍ: أسرع، مناسب لمهام مستقلة (مثال: تحليل من زوايا مختلفة).\n"
            "متسلسل: كل وكيل يرى ردود من سبقه قبل أن يضيف رأيه — مناسب لسير "
            "عمل تراكمي (مثال: بحث ← تحليل ← توصية)."
        ),
    )

    if st.button("🚀 نفّذ عبر الوكلاء", type="primary", key="orch_run") and task.strip():
        if manual:
            selected, route_method = manual, "manual"
        else:
            selected, route_method, _route_scores = route_query_verbose(
                task.strip(), AGENT_CATEGORIES, max_agents=2
            )
        if not selected:
            st.warning("لم يتم تحديد أي وكيل مناسب تلقائياً. اختر وكلاء يدوياً من القائمة أعلاه.")
        else:
            mode_label = "🔗 متسلسل" if exec_mode == "sequential" else "⚡ متوازٍ"
            route_label = {
                "manual":  "🖐️ اختيار يدوي",
                "keyword": "🔤 مطابقة كلمات مفتاحية",
                "llm":     "🧠 توجيه دلالي عبر LLM",
                "default": "⚙️ افتراضي عام (لا تطابق واضح)",
            }.get(route_method, route_method)
            st.caption(
                f"نمط التنفيذ: {mode_label} — التوجيه: {route_label} — الوكلاء المُفعَّلون: "
                + "، ".join(
                    f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}" for k in selected
                )
            )
            responses: Dict[str, str] = {}
            failed_keys: set = set()
            final_answer: Optional[str] = None
            for key in selected:
                cat = AGENT_CATEGORIES[key]
                bot_key = f"agent_bot_{cat.key}"
                if bot_key not in st.session_state:
                    st.session_state[bot_key] = CategoryAgentChat(cat.key)
                bot = st.session_state[bot_key]

                # ── النمط المتسلسل: يُرفَق ملخّص ردود الوكلاء السابقين
                # بنص المهمة، بحيث يبني كل وكيل على ما سبقه (سير عمل حقيقي
                # بدل مجرد ردود متوازية منفصلة). النمط المتوازي يمرّر
                # المهمة الأصلية فقط لكل وكيل، بدون أي تعديل. ──
                if exec_mode == "sequential" and responses:
                    prior = "\n\n".join(
                        f"[{AGENT_CATEGORIES[k].title}]\n{v}"
                        for k, v in responses.items() if k not in failed_keys
                    )
                    agent_input = (
                        f"{task.strip()}\n\n"
                        f"── ردود وكلاء سابقين في نفس سير العمل (ابنِ عليها، لا تكررها) ──\n"
                        f"{prior}"
                    )
                else:
                    agent_input = task.strip()

                _orch_skel_ph = st.empty()
                with _orch_skel_ph.container():
                    st.caption(f"⟳ {cat.title} يعمل على المهمة...")
                    _skeleton(lines=3)
                # 🆕 إعادة محاولة واحدة عند فشل الاستدعاء الأول (فشل عابر:
                # مزوّد LLM بطيء، تحميل أول مرة، إلخ) — بنفس روح إعادة
                # المحاولة المضافة أصلاً للسرب الذكي (SwarmCoordinator).
                resp, _ok = None, False
                for _attempt in range(2):
                    try:
                        resp = bot.chat(agent_input, source="orchestrator")
                        _ok = True
                        break
                    except Exception as _orch_err:
                        resp = f"⚠️ خطأ: {_orch_err}"
                if not _ok:
                    failed_keys.add(key)
                _orch_skel_ph.empty()
                responses[key] = resp
                # 🆕 شارة جودة موحّدة لكل رد وكيل (نفس ميزة تبويب "🤖 وكلاء AI"
                # ووحدة إعادة التوليد التلقائي المدمجة الآن في CategoryAgentChat).
                _q_label = ""
                if _ok and hasattr(bot, "last_quality_badge"):
                    try:
                        _qb = bot.last_quality_badge()
                        _q_label = f" — {_qb}" if _qb else ""
                    except Exception:
                        _q_label = ""
                with st.expander(f"{cat.emoji} {cat.title}{_q_label}", expanded=not synth):
                    st.markdown(resp)
                    _copy_button(resp, key=f"orch_{key}")

            valid_responses = {k: v for k, v in responses.items() if k not in failed_keys}
            if synth and responses:
                if not valid_responses:
                    st.warning("⚠️ فشل كل الوكلاء المُفعَّلين — لا يوجد ما يُولَّف.")
                else:
                    combined_input = "\n\n".join(
                        f"[{AGENT_CATEGORIES[k].title}]\n{v}" for k, v in valid_responses.items()
                    )
                    _synth_skel_ph = st.empty()
                    with _synth_skel_ph.container():
                        st.caption("⟳ يجري توليف الإجابة النهائية...")
                        _skeleton(lines=4)
                    try:
                        from ai.llm_fallback import LLMFallback
                        _llm = LLMFallback()
                        _synth_result = _llm.generate(
                            query=f"السؤال الأصلي: {task.strip()}\n\nردود الوكلاء:\n{combined_input}",
                            system_prompt=COORDINATOR_SYSTEM_PROMPT,
                        )
                        final = _synth_result.text
                    except Exception as _synth_err:
                        final = f"⚠️ تعذّر التوليف: {_synth_err}"
                    final_answer = final
                    _synth_skel_ph.empty()
                    if failed_keys:
                        st.caption(
                            "⚠️ استُبعِد من التوليف: "
                            + "، ".join(AGENT_CATEGORIES[k].title for k in failed_keys)
                        )
                    st.toast("✅ تم توليف الإجابة الموحّدة", icon="✅")
                    _final_q_label = ""
                    try:
                        from ai.response_quality import score_response as _score_final
                        _fq = _score_final(final, query=task.strip())
                        _final_q_label = f" — 🔎 {_fq.as_percent()}٪ {_fq.label}"
                    except Exception:
                        pass  # تقييم إضافي وغير حرج — لا يمنع عرض الإجابة نفسها
                    st.markdown(
                        f'<div class="section-header">✅ الإجابة الموحّدة{_final_q_label}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(final)
                    _copy_button(final, key="orch_final")

            # ── تصدير النتيجة الكاملة (ردود كل الوكلاء + التوليف إن وُجد) ──
            # كان لا يوجد سوى زر نسخ لكل رد على حدة — أي فقدان للنتيجة عند
            # تحديث الصفحة، رغم أنها قد تكون نتاج عدة استدعاءات LLM.
            if responses:
                _orch_export_lines = [f"# نتيجة منسّق الوكلاء\n\n**المهمة:** {task.strip()}\n"]
                _orch_export_lines.append(f"**نمط التنفيذ:** {mode_label} · **التوجيه:** {route_label}\n")
                for _ek, _ev in responses.items():
                    _ecat = AGENT_CATEGORIES[_ek]
                    _estatus = " ⚠️ (فشل)" if _ek in failed_keys else ""
                    _orch_export_lines.append(f"## {_ecat.emoji} {_ecat.title}{_estatus}\n\n{_ev}\n")
                if final_answer:
                    _orch_export_lines.append(f"## ✅ الإجابة الموحّدة\n\n{final_answer}\n")
                st.download_button(
                    "⬇️ تصدير النتيجة الكاملة",
                    data="\n".join(_orch_export_lines).encode("utf-8"),
                    file_name="نتيجة_منسق_الوكلاء.md",
                    mime="text/markdown",
                    key="orch_export_full",
                )



# ══════════════════════════════════════════════════════════════════════════
# تبويب 🐝 السرب الذكي — AgentFactory + SwarmCoordinator (تنفيذ حقيقي)
# ══════════════════════════════════════════════════════════════════════════
def render_swarm_studio():
    """
    واجهة فعلية لنظام الوكلاء الوظيفي (ai/agent_factory.py +
    ai/swarm_coordinator.py): تفكيك هدف معقّد ديناميكياً عبر PlanningAgent
    حقيقي، ثم توزيعه على الأدوار المتخصصة (Research/Translation/Review/
    Planning/Monitor/Optimization/Coding) وتنفيذها فعلياً عبر محرك
    NSMAgent (نفس محرك تبويب 💬 المحادثة)، مع عرض حي لنتيجة كل مهمة.
    """
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <span style="font-size:2rem">🐝</span>
        <div style="font-size:1.5rem;font-weight:900;color:var(--gold)">
            السرب الذكي — Multi-Agent Swarm
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;direction:rtl">
            هدف واحد ← تفكيك تلقائي ← تنفيذ فعلي متوازٍ عبر عدة وكلاء متخصصين
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _SWARM_OK:
        st.error("⚠️ تعذّر تحميل نظام السرب. تأكد من وجود ai/agent_factory.py و ai/swarm_coordinator.py.")
        return

    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl">اكتب هدفاً — بسيطاً أو معقداً — وسيُفكِّكه '
        '<b>PlanningAgent</b> حقيقياً إلى مهام فرعية، ثم يوزّعها <b>SwarmCoordinator</b> على '
        'الوكلاء المناسبين وينفذها فعلياً (وليس محاكاة) عبر نفس محرك المحادثة.</p>',
        unsafe_allow_html=True,
    )

    # ── singleton بمستوى الجلسة حتى تتراكم إحصائيات الوكلاء بين التشغيلات ──
    if "_swarm_factory" not in st.session_state:
        st.session_state["_swarm_factory"] = AgentFactory()
    if "_swarm_coordinator" not in st.session_state:
        st.session_state["_swarm_coordinator"] = SwarmCoordinator(
            st.session_state["_swarm_factory"], max_agents=6
        )
    factory = st.session_state["_swarm_factory"]
    coordinator = st.session_state["_swarm_coordinator"]

    with st.expander("📋 الأدوار المتاحة في الكتالوج"):
        for role in AgentFactory.available_roles():
            spec = AGENT_CATALOGUE[role]
            st.markdown(
                f"**{role}** — {spec['description']}  \n"
                f"القدرات: `{', '.join(spec['capabilities'])}`"
            )

    goal = st.text_area(
        "🎯 الهدف:",
        placeholder="مثال: ابحث عن أحدث تطورات الذكاء الاصطناعي، لخّصها، وراجع جودة الملخص",
        key="swarm_goal_input",
        height=90,
    )
    extra_context = st.text_area(
        "📎 سياق/بيانات إضافية (اختياري — نص خام يُمرَّر لكل مهمة فرعية):",
        key="swarm_context_input",
        height=70,
    )
    use_planner = st.toggle(
        "🧠 تفكيك ديناميكي عبر PlanningAgent (إن أُطفئ: قواعد كلمات مفتاحية ثابتة فقط)",
        value=True,
        key="swarm_use_planner",
    )
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        retry_failed = st.toggle(
            "🔁 إعادة محاولة المهام الفاشلة تلقائياً (مرة واحدة، بوكيل جديد)",
            value=True,
            key="swarm_retry_failed",
        )
    with col_opt2:
        synthesize = st.toggle(
            "🧩 وَلِّف نتائج المهام في إجابة نهائية واحدة موحّدة",
            value=True,
            key="swarm_synthesize",
        )

    if st.button("🚀 نفّذ عبر السرب", type="primary", key="swarm_run") and goal.strip():
        data = {"content": extra_context.strip()} if extra_context.strip() else {}
        _swarm_skeleton_ph = st.empty()
        with _swarm_skeleton_ph.container():
            st.caption("⟳ السرب يعمل — تفكيك الهدف وتنفيذ المهام الفرعية...")
            _skeleton(kind="cards")
            _skeleton(lines=4)
        result = coordinator.execute(
            goal.strip(),
            data=data,
            use_planner=use_planner,
            retry_failed=retry_failed,
            synthesize=synthesize,
        )
        _swarm_skeleton_ph.empty()

        status_emoji = {"done": "✅", "partial": "🟡", "failed": "❌"}.get(result.status, "❔")
        st.toast(
            f"{status_emoji} السرب انتهى: {result.success_count}/{len(result.tasks)} مهمة نجحت",
            icon=status_emoji,
        )
        st.markdown(
            f'<div class="section-header">{status_emoji} حالة السرب: {result.status} '
            f"({result.success_count}/{len(result.tasks)} مهمة نجحت)</div>",
            unsafe_allow_html=True,
        )

        for _ti, task in enumerate(result.tasks):
            icon = "✅" if task.status == "done" else ("❌" if task.status == "failed" else "⏳")
            _task_result_text = (task.result or {}).get("result_text", "")
            # 🆕 شارة جودة موحّدة لكل نتيجة مهمة (نفس ميزة تبويب "🤖 وكلاء AI"
            # ومنسّق الوكلاء) — تُحسب فقط للمهام الناجحة ذات نص نتيجة فعلي.
            _task_q_label = ""
            if task.status == "done" and _task_result_text:
                try:
                    from ai.response_quality import score_response as _score_task
                    _tq = _score_task(_task_result_text, query=task.sub_goal)
                    _task_q_label = f" — 🔎 {_tq.as_percent()}٪ {_tq.label}"
                except Exception:
                    pass  # تقييم إضافي وغير حرج — لا يمنع عرض نتيجة المهمة نفسها
            with st.expander(
                f"{icon} {task.sub_goal} — [{task.required_capability}] "
                f"({task.duration_ms or 0:.0f} ms){_task_q_label}",
                expanded=(task.status == "failed"),
            ):
                st.caption(f"الوكيل: {task.assigned_agent_id or '—'}")
                if _task_result_text:
                    st.markdown(_task_result_text)
                    _copy_button(_task_result_text, key=f"swarm_task_{_ti}")
                elif task.error:
                    st.warning(task.error)
                else:
                    st.caption("لا توجد نتيجة (لم يُسنَد وكيل لهذه المهمة).")

        _synthesis = (result.merged_output or {}).get("synthesis")
        if _synthesis:
            _synth_q_label = ""
            try:
                from ai.response_quality import score_response as _score_synth
                _sq = _score_synth(_synthesis, query=goal.strip())
                _synth_q_label = f" — 🔎 {_sq.as_percent()}٪ {_sq.label}"
            except Exception:
                pass  # تقييم إضافي وغير حرج — لا يمنع عرض التوليف نفسه
            st.markdown(
                f'<div class="section-header">✅ الإجابة الموحّدة{_synth_q_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_synthesis)
            _copy_button(_synthesis, key="swarm_synthesis")
        elif synthesize:
            st.info("⚠️ لم يتم توليف إجابة موحّدة (لا توجد مهام ناجحة، أو تعذّر استدعاء LLM).")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">📊 ملخص الوكلاء (AgentFactory)</div>',
                    unsafe_allow_html=True)
        _fs = factory.summary()
        st.markdown(f"""
        <div class="bento-grid">
            <div class="metric-card">
                <div class="metric-value">{_fs['total_agents']:,}</div>
                <div class="metric-label">إجمالي الوكلاء</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_fs['active_agents']:,}</div>
                <div class="metric-label">نشط الآن</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_fs['retired_agents']:,}</div>
                <div class="metric-label">متقاعد</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_fs['total_spawned']:,}</div>
                <div class="metric-label">إجمالي المُولَّد</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if _fs.get("role_distribution"):
            st.markdown(
                " ".join(
                    f'<span class="badge badge-blue">{role}: {count}</span>'
                    for role, count in _fs["role_distribution"].items()
                ),
                unsafe_allow_html=True,
            )
        with st.popover("🧹 تقليم الوكلاء ضعيفي الأداء"):
            st.caption(
                "يُقاعِد (retire) أي وكيل نفّذ 5 مهام على الأقل وكان "
                "متوسط أدائه أقل من الحد المحدَّد — لتفادي تكدّس وكلاء "
                "فاشلين تُختار من بينهم مهام مستقبلية عن طريق الخطأ."
            )
            prune_min_score = st.slider(
                "حد الأداء الأدنى", min_value=0.1, max_value=0.9, value=0.5,
                step=0.05, key="swarm_prune_min_score",
            )
            if st.button("🧹 نفّذ التقليم الآن", key="swarm_prune_btn"):
                retired_ids = factory.prune_underperformers(min_score=prune_min_score)
                if retired_ids:
                    st.success(f"تمت مقاعدة {len(retired_ids)} وكيل ضعيف الأداء.")
                else:
                    st.info("لا يوجد وكلاء تنطبق عليهم شروط التقليم حالياً.")
                st.rerun()
    with col_b:
        st.markdown('<div class="section-header">📊 ملخص السرب (SwarmCoordinator)</div>',
                    unsafe_allow_html=True)
        _cs = coordinator.summary()
        st.markdown(f"""
        <div class="bento-grid">
            <div class="metric-card">
                <div class="metric-value">{_cs['total_swarms']:,}</div>
                <div class="metric-label">إجمالي عمليات السرب</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['done']:,}</div>
                <div class="metric-label">✅ ناجحة بالكامل</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['partial']:,}</div>
                <div class="metric-label">🟡 نجاح جزئي</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['failed']:,}</div>
                <div class="metric-label">❌ فاشلة</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['active_agents']:,}</div>
                <div class="metric-label">وكلاء نشطون الآن</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['max_agents']:,}</div>
                <div class="metric-label">الحد الأقصى المسموح</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    hist = coordinator.history(limit=5)
    if hist:
        with st.expander("🕓 آخر 5 عمليات سرب"):
            for h in reversed(hist):
                st.markdown(f"**{h['goal']}** — {h['status']} ({h['success_count']}/{h['total_tasks']})")


if __name__ == "__main__":
    main()
