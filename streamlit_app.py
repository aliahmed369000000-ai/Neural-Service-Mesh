"""
Neural Service Mesh — واجهة المستخدم المعرفية
================================================
Streamlit front-end لمشروع النظام المعرفي العربي.
"""

from __future__ import annotations

import base64
import io
import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote

import streamlit as st

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
        AGENT_CATEGORIES, CATEGORY_ORDER, CategoryAgentChat,
    )
    _AGENTS_HUB_OK = True
except Exception:
    _AGENTS_HUB_OK = False

# ── محرك السرد الإبداعي 🎭 إبداع (تبويب جديد — إضافي بالكامل) ─────────────
try:
    from ai.llm_fallback import LLMFallback as _FableLLMFallback
    from ai.fable_engine import (
        FableEngine, FableChapter, STORY_MODES, CHARACTERS, ARABIC_METERS,
        DEFAULT_MODE as FABLE_DEFAULT_MODE,
        DEFAULT_CHARACTER as FABLE_DEFAULT_CHARACTER,
    )
    _FABLE_OK = True
except Exception:
    _FABLE_OK = False

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
    initial_sidebar_state="expanded",
)

# ── مسارات الملفات ────────────────────────────────────────────────────────
BASE = Path(__file__).parent
KNOWLEDGE_DIR  = BASE / "knowledge"
CHECKPOINTS_DIR = BASE / "checkpoints"
MEMORY_DIR     = BASE / "memory"

# ── نظام السمتين (الليل / المخطوطة) ─────────────────────────────────────
# ── لوحتا الألوان ────────────────────────────────────────────────────────
# مستوحاتان من عالم المخطوطات القرآنية: "الليل" (مخطوطة تحت ضوء قنديل مسجد
# ليلاً — نيلي عميق وذهب التذهيب)، و"المخطوطة" (ورق رَق/parchment نهاري
# بحبر سيبيا وتذهيب أفتح). كلا اللونين الذهبيين مختلفان فعلياً عن بعضهما
# (وليس نفس hex مع تغيير الخلفية فقط) لضمان تباين كافٍ بكل سمة.
THEMES = {
    "dark": {
        "label": "🌙 الليل",
        "bg_grad": "linear-gradient(180deg, #0B1220 0%, #121A2E 100%)",
        "bg": "#0B1220",
        "surface": "#141B2E",
        "surface2": "#1B2438",
        "border": "#2A3654",
        "text": "#EDE6D6",
        "text_muted": "#9AA5C0",
        "gold": "#C9A24B",
        "gold_soft": "rgba(201,162,75,0.15)",
        "emerald": "#2E9C77",
        "emerald_soft": "rgba(46,156,119,0.16)",
        "rose": "#C2686B",
        "rose_soft": "rgba(194,104,107,0.16)",
        "shadow": "rgba(0,0,0,0.45)",
        "pattern_stroke": "#C9A24B",
        "pattern_opacity": "0.05",
    },
    "light": {
        "label": "📜 المخطوطة",
        "bg_grad": "linear-gradient(180deg, #F6F0E1 0%, #EFE6CE 100%)",
        "bg": "#F3ECDA",
        "surface": "#FFFBF2",
        "surface2": "#F8F1DE",
        "border": "#D8C9A3",
        "text": "#241F16",
        "text_muted": "#6B5F47",
        "gold": "#9C7A2E",
        "gold_soft": "rgba(156,122,46,0.12)",
        "emerald": "#0F6B52",
        "emerald_soft": "rgba(15,107,82,0.10)",
        "rose": "#9C4A4D",
        "rose_soft": "rgba(156,74,77,0.10)",
        "shadow": "rgba(90,70,30,0.16)",
        "pattern_stroke": "#9C7A2E",
        "pattern_opacity": "0.06",
    },
}


def _pattern_svg(stroke: str, opacity: str) -> str:
    """نمط هندسي إسلامي بسيط (نجمة ثمانية من تقاطع مربعين) كخلفية مُبلَّطة
    خفيفة جداً — التوقيع البصري المميّز لهذا التصميم."""
    svg = (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='120' height='120'>"
        f"<g fill='none' stroke='{stroke}' stroke-opacity='{opacity}' stroke-width='1'>"
        f"<rect x='24' y='24' width='72' height='72'/>"
        f"<rect x='24' y='24' width='72' height='72' transform='rotate(45 60 60)'/>"
        f"</g></svg>"
    )
    return quote(svg)


CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&family=Noto+Kufi+Arabic:wght@500;700;800&display=swap');

:root {
    --bg: __BG__;
    --surface: __SURFACE__;
    --surface-2: __SURFACE2__;
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
}

html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Noto Naskh Arabic', 'Segoe UI', sans-serif;
}

/* ── القماشة العامة للتطبيق (تتجاوز سمة Streamlit المبنية مسبقاً) ── */
.stApp {
    background: __BG_GRAD__;
    background-image: __BG_GRAD__, url("data:image/svg+xml,__PATTERN__");
    background-repeat: no-repeat, repeat;
    background-attachment: fixed, fixed;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stSidebar"] {
    background: var(--surface);
    border-left: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stAppViewContainer"] { color: var(--text); }

h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown { color: var(--text); }

/* ── التبويبات بأسلوب "فصول مخطوطة" ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-muted);
    direction: rtl;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    color: var(--gold) !important;
    border-bottom: 2px solid var(--gold) !important;
}

/* ── الأزرار ── */
.stButton>button, .stDownloadButton>button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 10px;
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-weight: 600;
    transition: border-color 0.15s ease, transform 0.1s ease;
}
.stButton>button:hover, .stDownloadButton>button:hover {
    border-color: var(--gold);
    color: var(--gold);
}
.stButton>button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold) 0%, __GOLD_DARK_OR_LIGHT__ 100%);
    color: __BG__;
    border: none;
}

/* ── الحقول ── */
.stTextInput input, .stTextArea textarea, .stNumberInput input,
[data-baseweb="select"] > div {
    background: var(--surface-2) !important;
    color: var(--text) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    direction: rtl !important;
}

/* ── الموسّعات (expanders) ── */
[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
}

hr { border-color: var(--border) !important; }

/* ── عنوان الصفحة ── */
.main-title {
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--gold);
    text-align: center;
    padding: 1rem 0 0.3rem 0;
    direction: rtl;
}

.subtitle {
    text-align: center;
    color: var(--text-muted);
    font-size: 1rem;
    margin-bottom: 1.2rem;
    direction: rtl;
}

/* ── بطاقات المقاييس ── */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
    box-shadow: 0 2px 10px var(--shadow);
}
.metric-value {
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: var(--gold);
    direction: ltr;
}
.metric-label {
    font-size: 0.85rem;
    color: var(--text-muted);
    margin-top: 0.2rem;
    direction: rtl;
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
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: var(--gold);
    margin-bottom: 0.5rem;
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

/* ── آية قرآنية ── */
.quran-verse {
    background: var(--surface-2);
    border-right: 4px solid var(--gold);
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 1.15rem;
    line-height: 2.3;
    direction: rtl;
    color: var(--text);
}
.verse-ref {
    font-size: 0.8rem;
    color: var(--gold);
    font-weight: 600;
    margin-top: 0.3rem;
    direction: rtl;
}

.health-ok  { color: var(--emerald); font-weight: 600; }
.health-err { color: var(--rose);    font-weight: 600; }

/* ── عنوان قسم بتوقيع هندسي إسلامي بسيط بدل خط عادي ── */
.section-header {
    font-family: 'Noto Kufi Arabic', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    padding-bottom: 0.5rem;
    margin: 1rem 0 0.8rem 0;
    direction: rtl;
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
    font-family: 'Noto Kufi Arabic', sans-serif;
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
</style>
"""


def render_css(theme_key: str) -> str:
    t = THEMES.get(theme_key, THEMES["dark"])
    gold_alt = "#E4C87A" if theme_key == "dark" else "#7A5E20"
    pattern = _pattern_svg(t["pattern_stroke"], t["pattern_opacity"])
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
    return stats


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

def metric_card(value, label: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_home():
    """الصفحة الرئيسية — إحصاءات النظام."""
    roots         = load_arabic_roots()
    ckg           = load_ckg()
    quran_index   = load_quran_index()
    graph_metrics = load_graph_metrics()
    training      = load_training_summary()
    checkpoint    = load_latest_checkpoint()
    episodic      = get_episodic_stats()

    concepts_count  = len(ckg.get("concepts", {}))
    relations_count = len(ckg.get("relations", {}))

    # عدد الجذور ذات المعنى (أكثر من 3 أحرف)
    meaningful_roots = sum(1 for k in roots if len(k) >= 3 and roots[k].get("frequency", 0) > 10)

    train_steps = training.get("train_steps", 0)

    # آخر تحديث
    saved_at = checkpoint.get("saved_at", "")
    if saved_at:
        try:
            dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
            last_update = dt.strftime("%Y-%m-%d %H:%M") + " UTC"
        except Exception:
            last_update = saved_at[:19]
    else:
        last_update = "غير محدد"

    st.markdown('<div class="section-header">📊 إحصاءات النظام المعرفي</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(f"{concepts_count:,}", "مفهوم في CKG")
    with col2: metric_card(f"{relations_count:,}", "علاقة معرفية")
    with col3: metric_card(f"{meaningful_roots:,}", "جذر عربي مكتشف")
    with col4: metric_card(f"{train_steps:,}", "خطوة تدريب")

    st.markdown("")
    col5, col6, col7, col8 = st.columns(4)
    with col5: metric_card(f"{quran_index.get('total_ayat', 6236):,}", "آية قرآنية محملة")
    with col6: metric_card(f"{quran_index.get('total_surahs', 114)}", "سورة كريمة")
    with col7: metric_card(f"{episodic.get('episodic', 0):,}", "ذكرى تجريبية")
    with col8: metric_card(last_update, "آخر تحديث")

    st.markdown("")
    st.markdown('<div class="section-header">🔍 ابحث في المعرفة</div>', unsafe_allow_html=True)
    st.markdown("أدخل مفهوماً للبحث عنه مباشرةً في قلب النظام:")

    col_s, col_b = st.columns([4, 1])
    with col_s:
        quick_q = st.text_input("بحث", placeholder="مثال: الصبر، الجاذبية، الرحمة، العدل...",
                                key="home_search", label_visibility="collapsed")
    with col_b:
        if st.button("🔍 بحث", use_container_width=True, key="home_btn"):
            if quick_q.strip():
                st.session_state["search_query"] = quick_q.strip()
                st.session_state["active_tab"] = 1
                st.rerun()

    if quick_q.strip() and st.session_state.get("home_auto"):
        st.session_state["search_query"] = quick_q.strip()
        st.session_state["active_tab"] = 1
        st.rerun()


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

    # بطاقة المفهوم الرئيسية
    cdata = result["concept_data"]
    st.markdown(f"""
    <div class="concept-card">
        <div class="concept-name">💡 {result['query']}</div>
    """, unsafe_allow_html=True)

    if cdata:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**التصنيف:** {cdata.get('cluster', 'غير مصنّف')}")
        with col_b:
            freq = cdata.get("frequency", 0)
            st.markdown(f"**التكرار:** {freq:,} مرة")
        with col_c:
            strength = cdata.get("strength", 0.0)
            st.markdown(f"**قوة المفهوم:** {strength:.2%}")

    st.markdown("</div>", unsafe_allow_html=True)

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
                &nbsp;&nbsp; <small style="color:#999">قوة: {weight:.2f}</small>
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
                    _tokens_html = " ".join(
                        f'<span class="badge badge-{"blue" if t.is_verb else "purple" if t.is_noun else "amber"}" style="margin:2px">{t.surface}</span>'
                        for t in _analysis.syntactic.tokens[:20]
                    )
                    st.markdown(f"**الرموز المُحلَّلة:** {_tokens_html}", unsafe_allow_html=True)
                if _analysis.morphological.roots_found:
                    st.markdown(f"**الجذور المكتشفة:** `{'، '.join(_analysis.morphological.roots_found[:8])}`")
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
            <div style="background:#0f172a;color:#e2e8f0;border-radius:10px;
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

            fig = go.Figure(go.Bar(
                x=freqs,
                y=names,
                orientation='h',
                marker_color='#3b82f6',
                text=freqs,
                textposition='outside',
            ))
            fig.update_layout(
                height=450,
                margin=dict(l=20, r=60, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(autorange="reversed"),
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
    st.markdown(
        '<p style="color:#999">اسأل سؤالاً بالعربية، وسيحلل النظام السؤال '
        'ويبحث في 173 مفهوماً و2149 علاقة دلالية و6236 آية للإجابة.</p>',
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
    question = st.text_input(
        "اكتب سؤالك هنا:",
        value=default_q,
        key="qa_input",
        placeholder="مثال: ما علاقة الصبر بالإيمان؟",
    )
    st.session_state["qa_question"] = question

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
        result = answer_question(question, ckg, ayat, entities=entities)

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
    Pipeline: Gemini Omni Flash (بحث) → NSM Agent Fable 5 (سرد) → Higgsfield API (فيديو).
    """
    # ── استيراد المحرك ────────────────────────────────────────────────
    try:
        from ai.higgsfield_engine import (
            HiggsfieldEngine, build_gemini_llm, build_fable_llm
        )
    except Exception as _hf_err:
        st.error(f"⚠️ تعذّر تحميل محرك Higgsfield: {_hf_err}")
        return

    # ── رأس الصفحة ────────────────────────────────────────────────────
    st.markdown("""
    <div style="direction:rtl; text-align:right">
        <h2 style="margin-bottom:0.25rem">🎬 Higgsfield Explainer</h2>
        <p style="color:#aaa; font-size:0.95rem; margin-top:0">
            أنشئ فيديو وثائقياً من أي موضوع — حتى 10 دقائق —
            بالاستعانة بـ <strong>Gemini Omni Flash</strong> للبحث
            و<strong>NSM Agent Fable 5</strong> للسرد
            و<strong>Higgsfield API</strong> لتوليد الفيديو.
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
        style = st.selectbox(
            "🎨 نوع الوثائقي:",
            ["وثائقي عام", "تاريخي", "علمي", "ثقافي", "طبيعي", "تقني"],
            key="hf_style",
        )

    col_dur, col_vid = st.columns(2)
    with col_dur:
        minutes = st.slider(
            "⏱️ المدة المستهدفة (دقائق):",
            min_value=1, max_value=10, value=5,
            key="hf_minutes",
        )
    with col_vid:
        hf_key_input = st.text_input(
            "🔑 Higgsfield API Key (اختياري):",
            type="password",
            placeholder="اتركه فارغاً لتوليد السيناريو فقط",
            key="hf_api_key_input",
            help=(
                "⚠️ يجب أن يكون بصيغة KEY_ID:KEY_SECRET (المفتاح والسر معاً "
                "مفصولين بنقطتين رأسيتين ':') — كما بلوحة تحكم Higgsfield. "
                "مفتاح واحد بدون السر لن يعمل ويُرجع خطأ مصادقة 403."
            ),
        )
        hf_key = hf_key_input.strip() or os.getenv("HIGGSFIELD_API_KEY", "").strip()

    # ── معلومات Pipeline ───────────────────────────────────────────────
    with st.expander("ℹ️ كيف يعمل الـ Pipeline؟", expanded=False):
        st.markdown("""
        <div style="direction:rtl; text-align:right; font-size:0.9rem">
        <ol>
            <li><strong>🔍 Gemini Omni Flash</strong> — يبحث في المعلومات
                ويبني هيكل مشاهد الوثائقي (outline + حقائق موثّقة)</li>
            <li><strong>✍️ NSM Agent Fable 5</strong> — يصيغ نص السرد الصوتي
                بالعربية الفصحى + video prompt سينمائي بالإنجليزية لكل مشهد</li>
            <li><strong>🎬 Higgsfield API</strong> — يُولّد مقطع فيديو قصير
                (3-8 ثوانٍ) لكل مشهد. <em>يتطلب HIGGSFIELD_API_KEY</em></li>
        </ol>
        <p style="color:#888">بدون مفتاح Higgsfield تحصل على السيناريو الكامل
        جاهزاً للنسخ إلى أي أداة توليد فيديو خارجية.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── زر الإنشاء ────────────────────────────────────────────────────
    generate_btn = st.button(
        "🎬 أنشئ الوثائقي",
        type="primary",
        use_container_width=True,
        disabled=not bool(topic and topic.strip()),
        key="hf_generate_btn",
    )

    if not generate_btn:
        # عرض نتيجة سابقة إن وُجدت
        if "hf_result" in st.session_state:
            _render_hf_result(st.session_state["hf_result"])
        return

    if not topic.strip():
        st.warning("أدخل موضوع الوثائقي أولاً.")
        return

    # ── تنفيذ Pipeline ────────────────────────────────────────────────
    progress_bar  = st.progress(0, text="⟳ يبدأ الـ Pipeline...")
    status_text   = st.empty()

    def _prog(msg: str, pct: float):
        progress_bar.progress(int(min(pct, 100)), text=msg)
        status_text.markdown(
            f'<p style="color:#aaa; direction:rtl">{msg}</p>',
            unsafe_allow_html=True,
        )

    try:
        engine = HiggsfieldEngine(
            gemini_llm      = build_gemini_llm(),
            fable_llm       = build_fable_llm(),
            higgsfield_key  = hf_key,
        )
        result = engine.create_documentary(
            topic           = topic.strip(),
            target_minutes  = minutes,
            style           = style,
            generate_video  = bool(hf_key),
            progress_cb     = _prog,
        )
        st.session_state["hf_result"] = result
        progress_bar.progress(100, text="✅ اكتمل الوثائقي!")
        status_text.empty()

    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"❌ فشل إنشاء الوثائقي: {exc}")
        return

    _render_hf_result(result)


def _render_hf_result(result):
    """يعرض نتائج Higgsfield Explainer."""
    script  = result.script
    scenes  = script.scenes
    has_vid = result.api_used

    # ── ملخص ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📽️ عدد المشاهد", len(scenes))
    total_min = script.total_seconds // 60
    total_sec = script.total_seconds % 60
    c2.metric("⏱️ المدة الإجمالية", f"~{total_min}د {total_sec}ث")
    c3.metric("🔍 مزوّد البحث",
              script.research_provider or "—",
              delta=None)
    c4.metric("✍️ مزوّد السرد",
              script.narrative_provider or "—",
              delta=None)

    if has_vid:
        done  = sum(1 for s in scenes if s.video_status == "completed")
        fails = sum(1 for s in scenes if s.video_status in ("failed", "timeout"))
        st.caption(
            f"🎬 مقاطع الفيديو: {done} مكتملة · {fails} فاشلة "
            f"· {len(scenes)-done-fails} معلّقة"
        )
    else:
        st.info(
            "💡 لتوليد الفيديو الفعلي أضف **HIGGSFIELD_API_KEY** "
            "في الأسرار أو أدخله في الحقل أعلاه. "
            "السيناريو أدناه جاهز للنسخ إلى Higgsfield.ai يدوياً.",
            icon="ℹ️",
        )

    st.markdown("---")

    # ── بطاقات المشاهد ────────────────────────────────────────────────
    st.markdown(
        f'<h3 style="direction:rtl; text-align:right">📜 مشاهد الوثائقي — {script.title}</h3>',
        unsafe_allow_html=True,
    )

    for scene in scenes:
        # لون البادج بحسب حالة الفيديو
        vid_badge = {
            "completed":  '<span style="background:#22c55e;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">✅ فيديو جاهز</span>',
            "processing": '<span style="background:#f59e0b;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">⏳ يُعالَج</span>',
            "failed":     '<span style="background:#ef4444;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">❌ فشل</span>',
            "timeout":    '<span style="background:#ef4444;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">⏰ انتهت المهلة</span>',
            "no_api":     '<span style="background:#6b7280;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">🔑 بدون API</span>',
            "skipped":    '<span style="background:#6b7280;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">⏭️ متخطّى</span>',
            "pending":    '<span style="background:#6b7280;color:#fff;border-radius:4px;padding:2px 8px;font-size:0.75rem">⏳ معلّق</span>',
        }.get(scene.video_status, "")

        with st.expander(
            f"🎬 المشهد {scene.index} — {scene.title}  (~{scene.est_seconds}ث)",
            expanded=(scene.index == 1),
        ):
            # عرض الفيديو إن كان متاحاً
            if scene.video_url:
                st.video(scene.video_url)
            elif scene.video_error:
                st.caption(f"⚠️ {scene.video_error}")

            st.markdown(
                f"""
                <div style="direction:rtl; text-align:right; line-height:1.8">
                {vid_badge}
                <p style="margin-top:0.75rem">
                    <strong>🔊 السرد الصوتي:</strong><br>{scene.narration}
                </p>
                <p style="color:#aaa; font-size:0.9rem">
                    <strong>🎥 التوجيه المرئي:</strong> {scene.visual_notes or "—"}
                </p>
                <details>
                    <summary style="color:#888; cursor:pointer; font-size:0.85rem">
                        🎬 Higgsfield Video Prompt (إنجليزي)
                    </summary>
                    <pre style="background:#1e1e1e; padding:0.5rem; border-radius:6px;
                                font-size:0.8rem; color:#d4d4d4; white-space:pre-wrap">{scene.video_prompt}</pre>
                </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── تصدير السيناريو الكامل ────────────────────────────────────────
    st.markdown("---")
    col_exp, col_dl = st.columns(2)

    with col_exp:
        with st.expander("📋 النص الكامل للسرد (للتعليق الصوتي)"):
            st.text_area(
                "نص السرد:",
                value=script.full_narration,
                height=300,
                key="hf_full_narration",
            )

    with col_dl:
        with st.expander("🎬 Prompts لـ Higgsfield (للنسخ اليدوي)"):
            prompts_text = "\n\n".join(
                f"=== المشهد {s.index}: {s.title} ===\n{s.video_prompt}"
                for s in scenes
            )
            st.text_area(
                "Video Prompts:",
                value=prompts_text,
                height=300,
                key="hf_video_prompts",
            )

    # ── تجميع ومشاركة الوثائقي على مواقع التواصل ────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="direction:rtl; text-align:right">'
        '<h4 style="margin-bottom:0.3rem">📤 تصدير ومشاركة الوثائقي</h4>'
        '<p style="color:#aaa; font-size:0.85rem; margin-top:0">'
        'يجمّع مقاطع كل المشاهد المكتملة (من Higgsfield API) في فيديو واحد متسلسل، '
        'ثم يتيح رفعه مباشرة على يوتيوب أو تيك توك.'
        '</p></div>',
        unsafe_allow_html=True,
    )

    _completed_scenes = [s for s in scenes if s.video_status == "completed" and s.video_url]
    if not _completed_scenes:
        st.info(
            "ℹ️ لا توجد مشاهد مكتملة التوليد بعد. فعّل **Higgsfield API Key** أعلاه "
            "وانتظر اكتمال توليد المشاهد (🎬) حتى يظهر خيار التجميع والمشاركة هنا."
        )
    else:
        st.caption(f"🎬 عدد المشاهد الجاهزة للتجميع: {len(_completed_scenes)} / {len(scenes)}")

        if st.button("🎬 جمّع الفيديو الوثائقي الكامل", key="hf_assemble_btn", type="primary"):
            try:
                from ai.higgsfield_engine import assemble_documentary, DocumentaryAssemblyError
                with st.spinner("⏳ يُنزّل مقاطع المشاهد ويجمّعها بفيديو واحد... قد يستغرق دقائق"):
                    st.session_state.hf_assembled_mp4 = assemble_documentary(scenes)
                st.success("✅ تم تجميع الفيديو الوثائقي الكامل")
            except DocumentaryAssemblyError as e:
                st.error(f"⚠️ {e}")
            except Exception as e:  # noqa: BLE001
                st.error(f"⚠️ فشل التجميع: {e}")

        _assembled = st.session_state.get("hf_assembled_mp4")
        if _assembled:
            st.video(_assembled)
            st.download_button(
                "⬇️ تحميل الوثائقي الكامل (mp4)",
                data=_assembled,
                file_name=f"{script.title[:40] or 'documentary'}.mp4",
                mime="video/mp4",
                key="hf_download_assembled",
            )

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
                                        _assembled,
                                        title=yt_title,
                                        description=script.synopsis or script.full_narration[:4500],
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
                                    publish_id = tk.upload_video(_assembled, title=tk_title)
                                st.success(
                                    f"✅ تم إرسال الفيديو (publish_id: {publish_id}) — "
                                    "افتح تطبيق TikTok للتأكد من ظهوره ضمن المسودات/المنشورات."
                                )
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ فشل الرفع على تيك توك: {e}")


def render_training():
    """تبويب التدريب."""
    st.markdown('<div class="section-header">🎓 حالة التدريب</div>', unsafe_allow_html=True)

    training   = load_training_summary()
    checkpoint = load_latest_checkpoint()
    ckg        = load_ckg()

    train_steps = training.get("train_steps", 0)
    last_loss   = training.get("last_loss", 0.0)
    total_params= training.get("total_parameters", 0)
    ckg_size    = len(ckg.get("concepts", {}))

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(f"{train_steps:,}", "خطوات التدريب")
    with col2: metric_card(f"{last_loss:.2e}", "آخر خسارة (Loss)")
    with col3: metric_card(f"{total_params:,}", "معامل في الشبكة")
    with col4: metric_card(f"{ckg_size:,}", "مفهوم في CKG")

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
            for module_name in state.keys():
                module_labels = {
                    "neural_weights":  "الأوزان العصبية ✅",
                    "deep_network":    "الشبكة العميقة ✅",
                    "dynamic_layer":   "الطبقة الديناميكية ✅",
                    "episodic_memory": "الذاكرة التجريبية ✅",
                    "world_model":     "نموذج العالم ✅",
                    "system_dna":      "الحمض النووي للنظام ✅",
                    "self_awareness":  "الوعي الذاتي ✅",
                    "meta":            "البيانات الوصفية ✅",
                }
                label = module_labels.get(module_name, f"{module_name} ✅")
                st.markdown(f'<span class="badge badge-green">{label}</span>&nbsp;', unsafe_allow_html=True)

    # معلومات التدريب التفصيلية
    if training:
        st.markdown("")
        st.markdown('<div class="section-header">📐 بنية الشبكة العصبية</div>', unsafe_allow_html=True)
        arch = training.get("architecture", "")
        if arch:
            st.code(arch, language=None)

        avg_loss = training.get("avg_recent_loss", 0)
        lr       = training.get("learning_rate", 0)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**متوسط الخسارة الأخيرة:** `{avg_loss:.2e}`")
        with col_b:
            st.markdown(f"**معدل التعلم:** `{lr}`")


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
                <br><small style="color:#888">المصادر: {', '.join(sources[:3]) if sources else 'غير محددة'}</small>
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
            f'<p style="color:#999">تم بناء ملامح موضوعية لـ {len(surah_profiles)} سورة '
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
                <br><small style="color:#888">{ts} UTC</small>
            </div>
            """, unsafe_allow_html=True)

        # ── التوحيد (Consolidation) ──
        st.markdown("")
        st.markdown('<div class="section-header">🧬 توحيد الذاكرة (Consolidation)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#999">يستخرج هذا الإجراء أزواج المفاهيم المتكررة في الأسئلة السابقة، '
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
                    <br><small style="color:#888">{_ts_str} · {_t.get('topic') or 'بدون موضوع'}</small>
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
            &nbsp;&nbsp;<small style="color:#666">{detail}</small>
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
                border_color = "#1a73e8" if is_active else "#e2e8f0"
                st.markdown(f"""
                <div style="background:#f8faff;border:2px solid {border_color};border-radius:10px;
                            padding:0.8rem;text-align:center;direction:ltr">
                    <div style="font-size:1.3rem">{label}</div>
                    <code style="font-size:0.72rem;color:#1a73e8">{model_id}</code>
                    <div style="font-size:0.78rem;color:#555;margin-top:0.4rem;direction:rtl">{desc}</div>
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
        <div style="background:#f8faff; border:1px solid #c7d2fe; border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; direction:rtl">
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
        <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem">
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
                    <div style="background:#1e2a3a;color:#e2e8f0;border-radius:10px;
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
        <div style="background:#fdf4ff;border:1px solid #e9d5ff;border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem">
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
                <div style="background:#1e2a3a;color:#e2e8f0;border-radius:10px;
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
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem">
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
        <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem">
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
    """🤖 الوكلاء: يجمع وكلاء AI + منسّق الوكلاء + السرب الذكي."""
    sub = st.tabs(["🤖 وكلاء AI", "🤝 منسّق الوكلاء", "🐝 السرب الذكي"])
    with sub[0]: render_agents_hub()
    with sub[1]: render_agent_orchestrator()
    with sub[2]: render_swarm_studio()


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

        # مبدّل السمة: الليل (نيلي + تذهيب) / المخطوطة (ورق رَق + سيبيا)
        st.markdown('<div class="theme-toggle-caption">🎨 المظهر</div>', unsafe_allow_html=True)
        _theme_cols = st.columns(2)
        _current_theme = st.session_state.get("ui_theme", "dark")
        with _theme_cols[0]:
            if st.button(
                ("● " if _current_theme == "dark" else "") + "🌙 الليل",
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
                ("● " if _current_theme == "light" else "") + "📜 المخطوطة",
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
                _li_user = st.text_input("اسم المستخدم", key="account_login_username")
                _li_pass = st.text_input("كلمة المرور", type="password", key="account_login_password")
                if st.button("دخول", key="account_login_btn", use_container_width=True):
                    _user = _acc_login(_li_user, _li_pass) if _li_user and _li_pass else None
                    if _user:
                        st.session_state["_account"] = _user
                        st.rerun()
                    else:
                        st.error("اسم المستخدم أو كلمة المرور غير صحيحة")
            with _acc_tab_register:
                _reg_user = st.text_input("اسم المستخدم", key="account_reg_username")
                _reg_pass = st.text_input("كلمة المرور", type="password", key="account_reg_password")
                _reg_phone = st.text_input(
                    "رقم الهاتف (اختياري — لربط واتساب لاحقاً)",
                    key="account_reg_phone", placeholder="+9677xxxxxxxx",
                )
                if st.button("إنشاء حساب", key="account_reg_btn", use_container_width=True):
                    try:
                        _acc_create(_reg_user, _reg_pass, phone_number=_reg_phone or None)
                        st.success("تم إنشاء الحساب! سجّل دخولك من تبويب «دخول»")
                    except _AccErr as _e:
                        st.error(str(_e))
                    except Exception:
                        st.error("تعذّر إنشاء الحساب")

        st.markdown("---")

        st.markdown("### 🔑 OpenRouter API")
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

        # ── 🔐 وضع المالك — يتحكم بظهور تبويب ⚙️ النظام بالكامل ─────────────
        st.markdown("### 🔐 وضع المالك")
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
        st.caption("🧠 النظام المعرفي العربي")
        st.caption("CKG · قرآن · AutoTune")

    # ── العنوان ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-title">🧠 النظام المعرفي العربي</div>
    <div class="subtitle">Neural Service Mesh · ذكاء اصطناعي عربي متخصص بالمعرفة الإسلامية</div>
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
    <div style="text-align:center; color:#999; font-size:0.8rem; direction:rtl">
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
            "background:linear-gradient(135deg,#1a73e8,#0d47a1);color:#fff;border-radius:16px\">"
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
        <div style="background:#f0f9ff;border:1px solid #bae6fd;border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem">
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

    st.markdown("#### تنفيذ أمر")
    cmd_kind = st.radio("النوع", ["Bash", "Python"], horizontal=True, key="dev_console_kind")
    cmd_text = st.text_area("الأمر", height=120, key="dev_console_cmd",
                             placeholder="مثال: ls -la" if cmd_kind == "Bash" else "print(1 + 1)")
    cmd_timeout = st.slider("مهلة التنفيذ (ثوانٍ)", 5, 60, 20, 5, key="dev_console_timeout")

    if st.button("▶️ نفّذ", key="dev_console_run", type="primary"):
        if not cmd_text.strip():
            st.warning("أدخل أمراً أولاً.")
        else:
            import subprocess as _sp
            try:
                if cmd_kind == "Bash":
                    result = _sp.run(
                        cmd_text, shell=True, capture_output=True, text=True, timeout=cmd_timeout,
                    )
                else:
                    result = _sp.run(
                        ["python3", "-c", cmd_text], capture_output=True, text=True, timeout=cmd_timeout,
                    )
                st.markdown(f"**رمز الخروج:** `{result.returncode}`")
                if result.stdout:
                    st.markdown("**stdout:**")
                    st.code(result.stdout[-5000:])
                if result.stderr:
                    st.markdown("**stderr:**")
                    st.code(result.stderr[-5000:])
                if not result.stdout and not result.stderr:
                    st.caption("لا يوجد ناتج.")
            except _sp.TimeoutExpired:
                st.error(f"⏱️ انتهت المهلة ({cmd_timeout}s) قبل اكتمال التنفيذ.")
            except Exception as _exec_err:
                st.error(f"❌ خطأ أثناء التنفيذ: {_exec_err}")


# ══════════════════════════════════════════════════════════════════════════
# تبويب ℹ️ عن NSM — معلومات المنتج
# ══════════════════════════════════════════════════════════════════════════
def render_product_info():
    st.markdown('<div class="section-header">ℹ️ عن Neural Service Mesh (NSM)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="direction:rtl;line-height:2;font-size:1.02rem">
    <p><strong>Neural Service Mesh (NSM)</strong> — النظام المعرفي العربي — هو منصة ذكاء اصطناعي
    عربية متخصصة تجمع بين محرك معرفي ذاتي التعلّم (Cognitive Knowledge Graph) ونماذج لغوية كبيرة،
    لتقديم تجربة بحث ومحادثة ومعرفة عربية أصيلة، مع تخصص خاص بالمعرفة الإسلامية والقرآن الكريم.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### 🧭 ماذا يقدّم NSM؟")
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
    fcols = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        with fcols[i % 2]:
            st.markdown(f"""
            <div class="root-item">
                <strong>{icon} {title}</strong>
                <br><small style="color:#aaa">{desc}</small>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### 🔗 روابط")
    st.markdown(
        "- المستودع: [Neural-Service-Mesh على GitHub]"
        "(https://github.com/aliahmed369000000-ai/Neural-Service-Mesh)\n"
        "- بُني بـ Python · Streamlit · SQLite · نماذج لغوية عبر OpenRouter/Anthropic"
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
                f"""<div style="border:2px solid #a855f7;border-radius:10px;padding:16px;
                background:#a855f710;margin-bottom:16px;">
                🏆 <b style="color:#a855f7;font-size:1.1rem;"> {winner.model.split('/')[-1]}</b>
                <span style="color:#999;font-size:.75rem;"> — نقاط مركّبة: {winner.compound_score}</span>
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
        '<p style="color:#999">اختر وضع القصة والراوي، وابدأ حكاية تفاعلية '
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
            seed = st.text_input(
                "فكرة مبدئية (اختياري):",
                placeholder="مثال: قصة عن تاجر يبحث عن كنز مفقود في الصحراء",
            )
            if st.button("✨ ابدأ القصة", type="primary"):
                with st.spinner("يُنسج الفصل الأول..."):
                    chapter = engine.start_story(mode=mode, character=character, seed_idea=seed)
                st.session_state.fable_chapter = chapter
                st.rerun()
        else:
            # ── عرض الفصل الحالي ──
            mode_info = STORY_MODES.get(cur.mode, {})
            char_info = CHARACTERS.get(cur.character, {})
            st.markdown(
                f'<span class="badge badge-purple">{mode_info.get("emoji","")} {cur.mode}</span> '
                f'<span class="badge badge-blue">{char_info.get("emoji","")} {cur.character}</span> '
                f'<span class="badge badge-amber">المزوّد: {cur.provider}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class="root-item" style="font-size:1.05rem; line-height:2; text-align:right; direction:rtl">
                {cur.text}
            </div>
            """, unsafe_allow_html=True)

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
                with st.spinner("يُتابع نسج الأحداث..."):
                    st.session_state.fable_chapter = engine.continue_story(cur.session_id, chosen)
                st.rerun()

            st.markdown("---")
            st.markdown("**أوامر سريعة:**")
            qc_cols = st.columns(4)
            quick_labels = ["أنشد بيتاً", "صف المكان", "أضف حواراً", "لخّص"]
            for i, label in enumerate(quick_labels):
                with qc_cols[i]:
                    if st.button(f"⚡ {label}", key=f"fable_qc_{i}", use_container_width=True):
                        with st.spinner("..."):
                            result = engine.quick_command(cur.session_id, label)
                        st.markdown(f"""
                        <div class="root-item" style="text-align:right; direction:rtl">
                            {result.text}
                        </div>
                        """, unsafe_allow_html=True)

            if st.button("🔄 قصة جديدة"):
                st.session_state.fable_chapter = None
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
        if st.button("🪶 أنشئ القصيدة", type="primary") and topic.strip():
            with st.spinner("تُنظَم الأبيات..."):
                poem = engine.generate_poem(topic.strip(), meter=meter)
            st.markdown(f"""
            <div class="root-item" style="font-size:1.1rem; line-height:2.1; text-align:center; direction:rtl">
                {poem.text}
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"المزوّد: {poem.provider}")

    # ══════════════════ وثائقي (سيناريو Explainer) ══════════════════
    with explainer_tab:
        st.markdown(
            '<p style="color:#999">يولّد سيناريو وثائقياً مُقسّماً إلى مشاهد '
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

        if st.button("🎬 أنشئ السيناريو", type="primary") and topic.strip():
            with st.spinner("يُجري بحثاً ويكتب السيناريو..."):
                script = engine.generate_explainer(topic.strip(), target_minutes=minutes)

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
                    <p style="color:#999"><strong>🎥 اللقطة المقترحة:</strong> {seg.visual_notes}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 النص الكامل للسرد (لنسخه إلى أداة التعليق الصوتي)"):
                st.text_area("النص الكامل:", value=script.full_narration, height=200)

    # ══════════════════ ⚡ Shorts (فيديو قصير عمودي) ══════════════════
    with shorts_tab:
        st.markdown(
            '<p style="color:#999">يحوّل نصاً أو موضوعاً إلى فيديو '
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

        if st.button("⚡ أنشئ سيناريو Shorts", type="primary") and source_text.strip():
            with st.spinner("يُلخّص ويكتب لقطات سريعة..."):
                short = engine.generate_short(source_text.strip(), target_seconds=target_sec)
            st.session_state.shorts_script = short  # نحفظه بالجلسة لاستخدامه بزر الفيديو تحت

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
                    <p style="color:#999"><strong>🎞️ رسم متحرك مقترح:</strong> {seg.visual_notes}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 النص الكامل للسرد"):
                st.text_area("النص الكامل:", value=short.full_narration, height=150, key="shorts_full_text")

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
                "🎥 خلفيات سينمائية حقيقية (Higgsfield — بجودة National Geographic)",
                value=False,
                key="shorts_cinematic_bg_toggle",
                help=(
                    "بدل الخلفية المتدرّجة الافتراضية، يولّد خلفية فيديو حقيقية "
                    "لكل مشهد عبر Higgsfield. ⚠️ مزوّد مدفوع (بعكس بقية NSM "
                    "المجاني) — يستهلك رصيدك في Higgsfield لكل مشهد. "
                    "يتطلب HIGGSFIELD_API_KEY."
                    + ("" if _hf_key_present else " — غير مُفعَّل حالياً: المفتاح غير موجود بالبيئة.")
                ),
                disabled=not _hf_key_present,
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
                        )
                    st.session_state.shorts_mp4 = mp4_bytes
                    st.success("✅ تم إنتاج الفيديو")
                except Exception as e:  # noqa: BLE001
                    st.error(f"⚠️ فشل رندر الفيديو: {e}")

            mp4_bytes = st.session_state.get("shorts_mp4")
            if mp4_bytes:
                st.video(mp4_bytes)
                st.download_button(
                    "⬇️ تحميل الفيديو (mp4)",
                    data=mp4_bytes,
                    file_name=f"{short.title[:40] or 'short'}.mp4",
                    mime="video/mp4",
                )

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
            '<p style="color:#999">كل قصة تفاعلية تُحفظ تلقائياً في قاعدة بيانات SQLite محلية '
            '(<code>memory/fable.db</code>) — هذه الواجهة تستعرضها.</p>',
            unsafe_allow_html=True,
        )

        try:
            sessions = engine.memory.list_recent_sessions(limit=30)
        except Exception as e:  # noqa: BLE001
            sessions = []
            st.error(f"⚠️ تعذّر قراءة مكتبة القصص: {e}")

        if not sessions:
            st.info(
                "📭 لا توجد قصص محفوظة بعد. ابدأ قصة من تبويب «📖 قصة تفاعلية» "
                "وستظهر هنا تلقائياً بمجرد إنشاء الفصل الأول."
            )
        else:
            st.caption(f"📚 عدد القصص المحفوظة: {len(sessions)}")
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

                history_rows = engine.memory.get_history(session_id, limit=200)
                narrations = [r["content"] for r in history_rows if r["role"] == "narration"]
                preview = (narrations[0][:90] + "…") if narrations and len(narrations[0]) > 90 else (narrations[0] if narrations else "(لا يوجد نص بعد)")

                header = (
                    f"{mode_info.get('emoji', '📖')} {mode} · "
                    f"{char_info.get('emoji', '')} {character} — {created_label}"
                )
                with st.expander(header):
                    st.caption(f"🆔 {session_id} · عدد الفصول: {len(narrations)}")
                    st.markdown(
                        f"<p style='direction:rtl; text-align:right; color:#bbb'>{preview}</p>",
                        unsafe_allow_html=True,
                    )

                    view_key = f"lib_expand_{session_id}"
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("📖 عرض القصة كاملة", key=f"lib_view_btn_{session_id}", use_container_width=True):
                            st.session_state[view_key] = not st.session_state.get(view_key, False)
                    with col_b:
                        if st.button("▶️ استأنف هذه القصة", key=f"lib_resume_btn_{session_id}", use_container_width=True):
                            last_narration = narrations[-1] if narrations else ""
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

                    if st.session_state.get(view_key):
                        full_text = "\n\n".join(narrations) if narrations else "(لا يوجد نص محفوظ)"
                        st.markdown(f"""
                        <div class="root-item" style="text-align:right; direction:rtl; line-height:2">
                            {full_text}
                        </div>
                        """, unsafe_allow_html=True)


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
        '<p style="color:#999">ترجمة نص باستخدام نفس نماذج NSM اللغوية '
        '(Anthropic ← Cloudflare ← Gemini ← OpenRouter ← Groq) — بدون حاجة '
        'لأي مفتاح Google Translate أو DeepL.</p>',
        unsafe_allow_html=True,
    )

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

    if st.button("🌐 ترجم الآن", type="primary", key="tr_translate_btn") and source_text.strip():
        src = _TRANSLATE_LANGS[src_label]
        tgt = _TRANSLATE_LANGS[tgt_label]

        if src == tgt and src != "auto":
            st.warning("⚠️ لغة المصدر ولغة الهدف متطابقتان.")
        else:
            src_instruction = "اكتشف لغة النص تلقائياً ثم" if src == "auto" else f"ترجم من {src} إلى"
            system_prompt = (
                f"أنت مترجم محترف. {src_instruction} {tgt}. "
                "أعد فقط النص المترجم دون أي شرح أو مقدمات أو علامات اقتباس إضافية، "
                "مع الحفاظ على المعنى والأسلوب الأصلي بدقة."
            )
            with st.spinner("⏳ يترجم..."):
                try:
                    from ai.llm_fallback import LLMFallback
                    _tr_llm = LLMFallback(max_tokens=1200, temperature=0.2)
                    result = _tr_llm.generate(source_text.strip(), history=[], system_prompt=system_prompt)
                    st.session_state.tr_result = result
                except Exception as e:  # noqa: BLE001
                    st.error(f"⚠️ فشلت الترجمة: {e}")
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
        st.download_button(
            "⬇️ تحميل الترجمة (txt)",
            data=result.text,
            file_name="translation.txt",
            mime="text/plain",
            key="tr_download_btn",
        )


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
        from {opacity:0;transform:translateY(6px);}
        to   {opacity:1;transform:translateY(0);}
    }
    .chat-user {display:flex;justify-content:flex-end;margin:0.55rem 0;animation:bubbleIn .25s ease-out;}
    .chat-user .bbl {
        background:linear-gradient(135deg,#1a73e8,#0d47a1);
        color:#fff;padding:0.75rem 1.15rem;
        border-radius:18px 18px 4px 18px;max-width:85%;
        font-size:0.98rem;line-height:1.75;text-align:right;direction:rtl;
        box-shadow:0 3px 12px rgba(26,115,232,.3);
        white-space:pre-wrap;word-break:break-word;
    }
    .chat-nsm {display:flex;justify-content:flex-start;margin:0.55rem 0;gap:0.55rem;align-items:flex-start;animation:bubbleIn .25s ease-out;}
    .chat-nsm .bbl {
        background:linear-gradient(135deg,#1e2a3a,#162032);
        color:#e2e8f0;padding:0.75rem 1.15rem;
        border-radius:18px 18px 18px 4px;max-width:85%;
        font-size:0.98rem;line-height:1.85;text-align:right;direction:rtl;
        border:1px solid #2d4a6e;
        box-shadow:0 2px 8px rgba(0,0,0,.25);
        white-space:pre-wrap;word-break:break-word;
    }
    .chat-nsm .bbl code {
        background:#0d1b2a;color:#81e6d9;padding:0.15rem 0.4rem;
        border-radius:4px;font-size:0.88rem;font-family:monospace;
        white-space:pre-wrap;
    }
    .chat-nsm .bbl pre {
        background:#0d1b2a;border:1px solid #2d4a6e;border-radius:8px;
        padding:0.8rem;overflow-x:auto;margin:0.5rem 0;
        font-size:0.85rem;color:#a8d8ea;
        white-space:pre;
    }
    .ctx-tag {
        display:inline-block;background:#0f1923;border:1px solid #2d4a6e;
        border-radius:20px;padding:0.18rem 0.7rem;font-size:0.72rem;
        color:#90cdf4;margin-bottom:0.45rem;direction:rtl;
    }
    .chat-box {
        height:62vh;min-height:420px;max-height:680px;
        overflow-y:auto;padding:1.1rem;
        background:#0a0f1a;border-radius:18px;
        border:1px solid #1e2a3a;margin-bottom:0.9rem;
        scroll-behavior:smooth;
        box-shadow:inset 0 0 24px rgba(0,0,0,.25);
    }
    .chat-box::-webkit-scrollbar{width:5px;}
    .chat-box::-webkit-scrollbar-track{background:#0a0f1a;}
    .chat-box::-webkit-scrollbar-thumb{background:#2d4a6e;border-radius:6px;}
    .chat-box::-webkit-scrollbar-thumb:hover{background:#3d6a9e;}
    .typing-indicator {
        display:inline-block;color:#90cdf4;font-size:0.85rem;
        animation:pulse 1.2s infinite;
    }
    @keyframes pulse{0%,100%{opacity:.4;}50%{opacity:1;}}
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
    html = '<div class="chat-box" id="nsm-chat-box">'
    if not st.session_state.nsm_messages:
        html += '<div style="text-align:center;color:#2d4a6e;padding:2.5rem 1rem">🧠<br><br>ابدأ محادثتك — أسألني أي شيء</div>'
    else:
        for msg in st.session_state.nsm_messages:
            role, text = msg[0], msg[1]
            ctx_tag    = msg[2] if len(msg) > 2 else ""
            src_badge  = msg[3] if len(msg) > 3 else ""
            if role == "user":
                import html as _html
                safe_text = _html.escape(text).replace("\n", "<br>")
                html += f'<div class="chat-user"><div class="bbl">{safe_text}</div></div>'
            else:
                ctx_html = f'<div class="ctx-tag">📎 {ctx_tag}</div>' if ctx_tag else ""
                src_html = (
                    f'<div class="ctx-tag" style="color:#81e6d9">{src_badge}</div>'
                    if src_badge else ""
                )
                import html as _html
                if "<" not in text and ">" not in text:
                    safe_reply = _html.escape(text).replace("\n", "<br>")
                else:
                    safe_reply = text
                html += f'''<div class="chat-nsm">
                    <span style="font-size:1.4rem;margin-top:3px">🧠</span>
                    <div class="bbl">{ctx_html}{src_html}{safe_reply}</div>
                </div>'''
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    st.markdown("""
    <script>
    (function() {
        const box = window.parent.document.getElementById('nsm-chat-box');
        if (box) { box.scrollTop = box.scrollHeight; }
    })();
    </script>
    """, unsafe_allow_html=True)

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
        background:#0f1923 !important;
        border:1.5px solid #2d4a6e !important;
        border-radius:18px !important;
        padding:0.9rem 1.1rem !important;
        color:#e2e8f0 !important;
        transition:border-color .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color:#1a73e8 !important;
        box-shadow:0 0 0 3px rgba(26,115,232,.25) !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color:#5a7a9e;
    }
    .st-key-nsm_send_wrap button {
        height:96px !important;
        border-radius:18px !important;
        background:linear-gradient(135deg,#1a73e8,#0d47a1) !important;
        color:#fff !important;
        font-size:1.02rem !important;
        font-weight:600 !important;
        border:none !important;
        box-shadow:0 3px 12px rgba(26,115,232,.35) !important;
        transition:transform .12s ease, box-shadow .12s ease;
    }
    .st-key-nsm_send_wrap button:hover {
        transform:translateY(-1px);
        box-shadow:0 5px 16px rgba(26,115,232,.45) !important;
    }
    .st-key-nsm_send_wrap button:active {
        transform:translateY(0);
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

    # أسئلة سريعة
    st.markdown("**⚡ أسئلة سريعة:**")
    quick_cols = st.columns(4)
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

        # ── أضف رسالة المستخدم فوراً ──
        st.session_state.nsm_messages.append(("user", display_text, "", ""))

        # ── فحص أمان أولي (regex محلي، بدون تكلفة API) ──
        _safety_msg = _nsm_safety_gate(text.strip())
        if _safety_msg:
            st.session_state.nsm_messages.append(("nsm", _safety_msg, "", "🛡️ فحص أمان"))
            st.session_state.nsm_count += 1
            st.rerun()
            return

        # ── مسار OpenRouter مباشرة إذا تم إدخال مفتاح (يدعم الملفات/الصور) ──
        _or_key_p = st.session_state.get("_or_api_key", "").strip()
        if _or_key_p:
            _or_model_p = st.session_state.get("_or_model", "google/gemini-2.5-flash")
            can_vision = _or_model_p in VISION_MODELS
            doc_files   = [f for f in files if not f["is_image"]]
            image_files = [f for f in files if f["is_image"]] if can_vision else []
            user_content = _build_user_content(text.strip(), doc_files, image_files)

            history_msgs = []
            for m in st.session_state.nsm_messages[:-1]:
                role = "user" if m[0] == "user" else "assistant"
                history_msgs.append({"role": role, "content": m[1]})

            api_messages = history_msgs + [{"role": "user", "content": user_content}]

            with st.chat_message("assistant", avatar="🌐"):
                placeholder = st.empty()
                full_response = ""
                for chunk in _or_stream(api_messages, model=_or_model_p, api_key=_or_key_p):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            response = full_response
            ctx_tag = ""
            src_badge = f"🌐 OpenRouter · {_or_model_p.split('/')[-1]}"
            st.session_state.nsm_messages.append(("nsm", response, ctx_tag, src_badge))
            st.session_state.nsm_count += 1
            st.rerun()
            return

        # ── Streaming عبر NSM Agent مباشرة إذا كان متاحاً ──
        try:
            from ai.nsm_agent_core import NSMAgent as _AgentCls
            _agent = getattr(st.session_state, "_nsm_agent_instance", None)
            if _agent is None:
                _agent = _AgentCls()
                st.session_state._nsm_agent_instance = _agent
            _agent.available = _agent._check_available()
        except Exception:
            _agent = None

        if _agent and _agent.available:
            # ── Streaming: يظهر الرد حرفاً بحرف ──
            with st.chat_message("assistant", avatar="🧠"):
                placeholder = st.empty()
                full_response = ""
                for chunk in _agent.run_stream(text.strip()):
                    full_response += chunk
                    placeholder.markdown(full_response + "▌")
                placeholder.markdown(full_response)
            response = full_response.replace("⏳ *أفكر...*\n\n", "", 1)
            # ── مزامنة الشارة: bot.chat() لم يُستدعَ هنا، فنحدّث المصدر يدوياً ──
            if hasattr(bot, "_last_source"):
                bot._last_source = "nsm_agent"
        else:
            # ── fallback: bot.chat العادي ──
            response = bot.chat(text.strip(), system_prompt=NSM_SYSTEM_PROMPT)

        ctx_tag   = bot.context_info()
        src_badge = (
            bot.source_badge()
            if hasattr(bot, "source_badge") else "🤖 NSM Agent v3"
        )
        st.session_state.nsm_messages.append(("nsm",  response, ctx_tag, src_badge))
        st.session_state.nsm_count += 1
        st.rerun()

    if send and (user_input or st.session_state["chat_pending_files"]):
        _process(user_input)

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
                st.success(f"✅ تمت الجدولة على {sched_dt}")
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
        background:linear-gradient(135deg,#1a73e8,#0d47a1);
        color:#fff;padding:0.7rem 1.05rem;border-radius:18px 18px 4px 18px;
        max-width:85%;font-size:0.96rem;line-height:1.7;text-align:right;direction:rtl;
        box-shadow:0 3px 12px rgba(26,115,232,.3);white-space:pre-wrap;word-break:break-word;
    }
    .agent-bot {display:flex;justify-content:flex-start;margin:0.5rem 0;gap:0.5rem;align-items:flex-start;animation:agentBubbleIn .25s ease-out;}
    .agent-bot .bbl {
        background:linear-gradient(135deg,#1e2a3a,#162032);
        color:#e2e8f0;padding:0.7rem 1.05rem;border-radius:18px 18px 18px 4px;
        max-width:85%;font-size:0.96rem;line-height:1.8;text-align:right;direction:rtl;
        border:1px solid #2d4a6e;box-shadow:0 2px 8px rgba(0,0,0,.25);
        white-space:pre-wrap;word-break:break-word;
    }
    .agent-box {
        height:48vh;min-height:320px;max-height:520px;overflow-y:auto;padding:1rem;
        background:#0a0f1a;border-radius:16px;border:1px solid #1e2a3a;margin-bottom:0.8rem;
        scroll-behavior:smooth;box-shadow:inset 0 0 20px rgba(0,0,0,.25);
    }
    .agent-badge {
        display:inline-block;background:#0f1923;border:1px solid #2d4a6e;border-radius:20px;
        padding:0.15rem 0.65rem;font-size:0.72rem;color:#90cdf4;direction:rtl;
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
            f'<div style="text-align:center;color:#2d4a6e;padding:2rem 1rem">'
            f'{category.emoji}<br><br>ابدأ محادثتك مع وكيل {category.title}</div>'
        )
    else:
        for role, text, badge in st.session_state[msg_key]:
            safe = _html.escape(text).replace("\n", "<br>")
            if role == "user":
                html_out += f'<div class="agent-user"><div class="bbl">{safe}</div></div>'
            else:
                badge_html = f'<div class="agent-badge">{badge}</div>' if badge else ""
                html_out += (
                    f'<div class="agent-bot"><span style="font-size:1.3rem;margin-top:3px">'
                    f'{category.emoji}</span><div class="bbl">{badge_html}{safe}</div></div>'
                )
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)
    st.markdown(f"""
    <script>
    (function() {{
        const box = window.parent.document.getElementById('{box_id}');
        if (box) {{ box.scrollTop = box.scrollHeight; }}
    }})();
    </script>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        user_input = st.text_area(
            label="سؤالك", placeholder=f"اسأل وكيل {category.title}…",
            key=f"agent_input_{category.key}", label_visibility="collapsed", height=88,
        )
    with c2:
        send = st.button("➤ إرسال", key=f"agent_send_{category.key}", use_container_width=True)

    if category.quick_prompts:
        st.markdown("**⚡ أسئلة سريعة:**")
        qcols = st.columns(len(category.quick_prompts))
        for i, q in enumerate(category.quick_prompts):
            with qcols[i]:
                if st.button(q, key=f"agent_q_{category.key}_{i}", use_container_width=True):
                    st.session_state[f"_agent_pending_{category.key}"] = q

    if st.button("🗑 مسح المحادثة", key=f"agent_clear_{category.key}"):
        st.session_state[msg_key] = []
        st.session_state[cnt_key] = 0
        bot.clear_history()
        st.rerun()

    def _process(text: str):
        if not text.strip():
            return
        st.session_state[msg_key].append(("user", text.strip(), ""))

        _safety_msg = _nsm_safety_gate(text.strip())
        if _safety_msg:
            st.session_state[msg_key].append(("bot", _safety_msg, "🛡️ فحص أمان"))
            st.session_state[cnt_key] += 1
            st.rerun()
            return

        response = bot.chat(text.strip(), force_web=web_toggle, source="hub")
        st.session_state[msg_key].append(("bot", response, bot.last_provider_badge()))
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
        '<p style="color:#999;direction:rtl">هذا التبويب يعرض الوحدات الداخلية للنظام: '
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
            '<p style="color:#999">حدّد هدفاً بالعربية وسيبني النظام خطة تنفيذ تلقائية.</p>',
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
            '<p style="color:#999">بحث حقيقي في الإنترنت بدون مفتاح API — '
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
                <div style="background:#0f172a;color:#e2e8f0;border-radius:10px;
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
        <div style="font-size:1.5rem;font-weight:900;color:#38bdf8">
            منسّق الوكلاء
        </div>
        <div style="color:#999;font-size:0.85rem;direction:rtl">
            وزّع مهمتك تلقائياً على وكلاء "🤖 وكلاء AI" المتخصصين، ثم احصل على إجابة موحّدة
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _AGENTS_HUB_OK or not _ORCHESTRATOR_OK:
        st.error("⚠️ تعذّر تحميل وحدات الوكلاء (ai/agent_categories.py أو ai/godmode.py).")
        return

    st.markdown(
        '<p style="color:#999;direction:rtl">اكتب مهمة أو سؤالاً مركّباً، وسيُحدَّد تلقائياً '
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
        selected = manual if manual else route_query(task.strip(), AGENT_CATEGORIES, max_agents=2)
        if not selected:
            st.warning("لم يتم تحديد أي وكيل مناسب تلقائياً. اختر وكلاء يدوياً من القائمة أعلاه.")
        else:
            mode_label = "🔗 متسلسل" if exec_mode == "sequential" else "⚡ متوازٍ"
            st.caption(
                f"نمط التنفيذ: {mode_label} — الوكلاء المُفعَّلون: " + "، ".join(
                    f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}" for k in selected
                )
            )
            responses: Dict[str, str] = {}
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
                        f"[{AGENT_CATEGORIES[k].title}]\n{v}" for k, v in responses.items()
                    )
                    agent_input = (
                        f"{task.strip()}\n\n"
                        f"── ردود وكلاء سابقين في نفس سير العمل (ابنِ عليها، لا تكررها) ──\n"
                        f"{prior}"
                    )
                else:
                    agent_input = task.strip()

                with st.spinner(f"⟳ {cat.title} يعمل على المهمة..."):
                    try:
                        resp = bot.chat(agent_input, source="orchestrator")
                    except Exception as _orch_err:
                        resp = f"⚠️ خطأ: {_orch_err}"
                responses[key] = resp
                with st.expander(f"{cat.emoji} {cat.title}", expanded=not synth):
                    st.markdown(resp)

            if synth and responses:
                combined_input = "\n\n".join(
                    f"[{AGENT_CATEGORIES[k].title}]\n{v}" for k, v in responses.items()
                )
                with st.spinner("⟳ يجري توليف الإجابة النهائية..."):
                    try:
                        from ai.llm_fallback import LLMFallback
                        _llm = LLMFallback()
                        final = _llm.chat(messages=[
                            {"role": "system", "content": COORDINATOR_SYSTEM_PROMPT},
                            {"role": "user", "content":
                                f"السؤال الأصلي: {task.strip()}\n\nردود الوكلاء:\n{combined_input}"},
                        ])
                    except Exception as _synth_err:
                        final = f"⚠️ تعذّر التوليف: {_synth_err}"
                st.markdown('<div class="section-header">✅ الإجابة الموحّدة</div>', unsafe_allow_html=True)
                st.markdown(final)



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
        <div style="font-size:1.5rem;font-weight:900;color:#38bdf8">
            السرب الذكي — Multi-Agent Swarm
        </div>
        <div style="color:#999;font-size:0.85rem;direction:rtl">
            هدف واحد ← تفكيك تلقائي ← تنفيذ فعلي متوازٍ عبر عدة وكلاء متخصصين
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _SWARM_OK:
        st.error("⚠️ تعذّر تحميل نظام السرب. تأكد من وجود ai/agent_factory.py و ai/swarm_coordinator.py.")
        return

    st.markdown(
        '<p style="color:#999;direction:rtl">اكتب هدفاً — بسيطاً أو معقداً — وسيُفكِّكه '
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

    if st.button("🚀 نفّذ عبر السرب", type="primary", key="swarm_run") and goal.strip():
        data = {"content": extra_context.strip()} if extra_context.strip() else {}
        with st.spinner("⟳ السرب يعمل — تفكيك الهدف وتنفيذ المهام الفرعية..."):
            result = coordinator.execute(goal.strip(), data=data, use_planner=use_planner)

        status_emoji = {"done": "✅", "partial": "🟡", "failed": "❌"}.get(result.status, "❔")
        st.markdown(
            f'<div class="section-header">{status_emoji} حالة السرب: {result.status} '
            f"({result.success_count}/{len(result.tasks)} مهمة نجحت)</div>",
            unsafe_allow_html=True,
        )

        for task in result.tasks:
            icon = "✅" if task.status == "done" else ("❌" if task.status == "failed" else "⏳")
            with st.expander(
                f"{icon} {task.sub_goal} — [{task.required_capability}] "
                f"({task.duration_ms or 0:.0f} ms)",
                expanded=(task.status == "failed"),
            ):
                st.caption(f"الوكيل: {task.assigned_agent_id or '—'}")
                if task.result and task.result.get("result_text"):
                    st.markdown(task.result["result_text"])
                elif task.error:
                    st.warning(task.error)
                else:
                    st.caption("لا توجد نتيجة (لم يُسنَد وكيل لهذه المهمة).")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**📊 ملخص الوكلاء (AgentFactory)**")
        st.json(factory.summary())
    with col_b:
        st.markdown("**📊 ملخص السرب (SwarmCoordinator)**")
        st.json(coordinator.summary())

    hist = coordinator.history(limit=5)
    if hist:
        with st.expander("🕓 آخر 5 عمليات سرب"):
            for h in reversed(hist):
                st.markdown(f"**{h['goal']}** — {h['status']} ({h['success_count']}/{h['total_tasks']})")


if __name__ == "__main__":
    main()
