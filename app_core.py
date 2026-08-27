"""
Neural Service Mesh — واجهة المستخدم المعرفية
================================================
Streamlit front-end لمشروع النظام المعرفي العربي.
"""

from __future__ import annotations

import base64
import hmac
import io
import json
import logging
import os
import re
import sqlite3
import time
import uuid as _uuid
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple
from urllib.parse import quote
import sys

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

# أداء/سلامة: سقف طول محادثة الدردشة الرئيسية (تبويب 💬 المحادثة).
# يُستخدم في مكانين: (1) عدد الرسائل المعروضة في الواجهة (يمنع تضخّم
# HTML المُعاد بناؤه كل rerun مع طول المحادثة)، (2) عدد الرسائل المُرسَلة
# كسياق لمزوّد OpenRouter (يمنع نمو تكلفة/حجم التوكنز بلا حدود، ويتجنّب
# تجاوز حد نافذة السياق للنموذج في المحادثات الطويلة جداً). السجل الكامل
# يبقى محفوظاً في nsm_messages بلا حذف — هذا سقف عرض/سياق فقط، وليس حذفاً.
NSM_CHAT_DISPLAY_LIMIT = 40

# ═══════════════════════════════════════════════════════════════════════════
# ذاكرة المحادثة الطويلة — حد عرض متدرج + تلخيص سياقي تلقائي
# ═══════════════════════════════════════════════════════════════════════════
# 🆕 حد عرض متدرج (virtual scroll): بدلًا من ملاحظة سلبية «رسالة مخفية»,
# يظهر زر «⬆️ تحميل المزيد» يرفع سقف العرض تدريجيًا (40 → 80 → 120 ...).
# السقف محفوظ في session_state (`_nsm_chat_display_ceil`) ويُعاد إلى
# NSM_CHAT_DISPLAY_LIMIT تلقائيًا عند وصول رسالة جديدة (scroll للأسفل).
NSM_CHAT_DISPLAY_INCREMENT = 40

# 🆕 تلخيص سياقي تلقائي: عند تجاوز هذا العدد من الرسائل يُولَّد ملخص مضغوط
# للجزء الأقدم (دون حذف أي شيء — السجل الكامل يبقى في nsm_messages
# وchat_history_store). الملخص يعرض أعلى المحادثة ويقدم السياق للنموذج
# بدل السجل القديم الغائب، فيحافظ على خيوط الموضوع بلا نمو توكنات بلا حدود.
NSM_CHAT_MEMORY_SUMMARY_AT = 200
NSM_CHAT_SUMMARY_CHARS_PER_MSG = 60
NSM_CHAT_SUMMARY_MAX_CHARS = 3000


def summarize_chat_segment(segments: list) -> str:
    """يحوّل قطعًا من المحادثة إلى ملخص نصي مضغوط (بدون LLM).

    لكل رسالة: [المستخدم/NSM] + أول NSM_CHAT_SUMMARY_CHARS_PER_MSG حرفًا + «...».
    لا يحذف أي بيانات — ينتج فقط تمثيلًا مقروءًا لأقدم جزء.
    """
    if not segments:
        return ""
    lines = []
    for _seg in segments:
        role = _seg[0] if len(_seg) > 0 else "nsm"
        text = _seg[1] if len(_seg) > 1 else ""
        head = (text or "").strip().replace("\n", " ")[:NSM_CHAT_SUMMARY_CHARS_PER_MSG]
        label = "المستخدم" if str(role) == "user" else "NSM"
        tail = "..." if len((text or "").strip()) > NSM_CHAT_SUMMARY_CHARS_PER_MSG else ""
        lines.append(f"[{label}]: {head}{tail}")
    joined = " \u2022 ".join(lines)
    if len(joined) > NSM_CHAT_SUMMARY_MAX_CHARS:
        joined = joined[:NSM_CHAT_SUMMARY_MAX_CHARS] + "..."
    return joined


def build_chat_memory_summary(messages: list) -> str:
    """ملخص تراكمي للأقدم من المحادثة عند تجاوز NSM_CHAT_MEMORY_SUMMARY_AT.

    الجزء الأقدم (من 0 حتى total - SUMMARY_AT) يلخص؛ السجل كامل يبقى محفوظًا.
    """
    total = len(messages)
    if total <= NSM_CHAT_MEMORY_SUMMARY_AT:
        return ""
    cutoff = total - NSM_CHAT_MEMORY_SUMMARY_AT
    return summarize_chat_segment(messages[:cutoff])

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
# ── 🆕 الذاكرة الذاتية المستمرة (Long-Term Memory) ───────────────────────
# طبقة تعلّم تلقائي من المحادثات: تسجّل الأسئلة والأجوبة كذكريات طويلة
# المدى، وتستحضر الذكريات ذات الصلة عند الإجابة لاحقًا. كل فشل يُبتلَع
# بتحذير مسجّل فقط — لا يؤثر على تدفق المحادثة إطلاقًا.
try:
    from ai.long_term_memory import get_ltm as _get_ltm, reset_ltm_cache as _reset_ltm_cache
    _LTM_OK = True
except Exception:
    _LTM_OK = False
# ── 🆕 التفكير متعدد الخطوات (Multi-Step Reasoning) ─────────────────────
# يصنّف الأسئلة المعقدة حتميًا (بلا API)، يفكّكها إلى خطة خطوات مرتبة
# (مقارنة / سببية / عملية / تعداد / تحليل)، ويجمّعها في رد مخطط يُلحق
# كرسالة نظام قبل النافذة الأخيرة لمسار OpenRouter. كل فشل يُبتلَع
# بصمت — الأسئلة البسيطة وبقية المسارات لا تتأثر إطلاقًا.
try:
    from ai.multi_step_reasoner import (
        plan_system_prompt as _plan_system_prompt,
        is_complex_question as _is_complex_question,
    )
    _MSR_OK = True
except Exception:
    _MSR_OK = False
    _plan_system_prompt = None
    def _is_complex_question(t: str) -> bool:  # احتياطي آمن
        return False
# ── 🆕 المهام طويلة الأمد (Long-Horizon Tasks) ──────────────────────
# منظّم مهام بحثية/تقريرية متعددة الخطوات يعمل في خيوط خلفية (daemon)
# مع أدوات إنترنت آمنة (بحث متعدد المصادر / جلب صفحات / كتابة وقراءة
# ملفات في مساحة معزولة / بايثون محمي)، وخطة حتمية بلا API وحوكمة
# صارمة (سقف خطوات/طلبات/مدة/ملفات). كل فشل يُبتلَع بصمت — المحادثة
# وبقية المسارات لا تتأثر إطلاقًا.
try:
    from ai.long_horizon_tasks import (
        get_long_horizon_manager as _get_lht_manager,
    )
    _LHT_OK = True
except Exception:
    _LHT_OK = False
    def _get_lht_manager():  # احتياطي آمن
        raise RuntimeError("وحدة المهام طويلة الأمد غير متاحة")
# ── 🆕 التعاون في المهام الطويلة (Collaborative Tasks) ─────────────────
# فريق أدوار وكلاء متوازية (باحثون + مدقق نتائج) ينفّذ مهمة مركّبة ثم يُولّف
# المنسّق تقريرًا موحدًا من مخرجات الجميع عبر ناقل الأحداث المشترك. حتمي
# ومقصور (سقف أدوار/خطوات/تزامن/مدة) وكل فشل يُبتلَع بصمت.
try:
    from ai.collaborative_tasks import (
        get_collaborative_manager as _get_collab_manager,
    )
    _COOP_OK = True
except Exception:
    _COOP_OK = False
    def _get_collab_manager():  # احتياطي آمن
        raise RuntimeError("وحدة التعاون غير متاحة")
# ── 🆕 ناقل المعرفة المشترك (Shared Knowledge Base — Qdrant) ──────
# يتقاسم فيه أدوار فريق المهمة التعاونية نتائجهم لحظيًا: كل بحث/جلب
# ناجح يُشارك في الناقل، وكل دور يستحضر ما وجده الزملاء قبل بحثه
# (بحث دلالي عربي bge-m3 عبر Qdrant + fallback محلي SQLite صامت).
# أي فشل في Qdrant أو المكتبة يتحول للطبقة المحلية دون أي انقطاع.
try:
    from ai.shared_knowledge import (
        get_skb as _get_skb,
    )
    _SKB_OK = True
except Exception:
    _SKB_OK = False
    def _get_skb():  # احتياطي آمن
        raise RuntimeError("ناقل المعرفة المشترك غير متاح")
# ── 🆕 سجل الخبرات والقرارات الجماعية المتراكم (Team Experience Memory) ──
# ذاكرة ذاتية جماعية مستمرة: كل مهمة تعاونية/طويلة الأمد تُسجَّل فيها
# خبرات (قرار + نتيجته الفعلية success/partial/failure) تُستحضر قبل
# التخطيط للمهام المماثلة فتوجه الوكلاء نحو الأنجح وتتجنب الفاشل.
# استيراد اختياري (_TEM_OK) — أي فشل يعيد السلوك الأصلي بلا انقطاع.
try:
    from ai.team_experience import (
        get_experience_log as _get_experience_log,
    )
    _TEM_OK = True
except Exception:
    _TEM_OK = False
    def _get_experience_log():  # احتياطي آمن
        raise RuntimeError("سجل الخبرات الجماعية غير متاح")
# ── 🆕 نظام المكافآت الذاتية للأدوار (Role Rewards / XP) ─────────────────
# نقاط خبرة ومهارات متراكمة لكل دور عبر المهام: نجاح يرفع نقاطَه ويرقّي
# مهاراتَه، والفشل يخفض. الوحدات الأخرى تستحضر «أفضل دور لمهارة» فيوجَّه
# اختيار الأدوار تلقائيًا نحو الأنسب — سجل دائم عبر الجلسات.
try:
    from ai.role_rewards import (
        get_role_rewards as _get_role_rewards,
    )
    _RR_OK = True
except Exception:
    _RR_OK = False
    def _get_role_rewards():  # احتياطي آمن
        raise RuntimeError("نظام المكافآت غير متاح")
# ── 🆕 التخطيط الجماعي الاستباقي (Proactive Planning) ────────────────────
# قبل أي مهمة جديدة: يستحضر ما يشبهها من سجل الخبرات ويجمع أفضل الأدوار
# للمهارات المطلوبة، فيخرج خطة استباقية (توصيات + محاذير) تُحقن في
# المهمة التعاونية والمهام الطويلة قبل تخصيص الأدوار أو بناء الخطة.
try:
    import ai.proactive_planning as _pp_mod  # noqa: F401
    from ai.proactive_planning import (
        build_pre_task_plan as _build_pre_task_plan,
        plan_summary_text as _plan_summary_text,
    )
    _pp_mod._set_app_core(sys.modules[__name__])
    _PP_OK = True
except Exception:
    _PP_OK = False
    def _build_pre_task_plan(*a, **kw):  # احتياطي آمن
        raise RuntimeError("التخطيط الاستباقي غير متاح")
    def _plan_summary_text(*a, **kw):  # احتياطي آمن
        return ""
# ── 🆕 الأهداف المؤسسية طويلة الأمد (Long-Term Goals) ────────────────────
# سجل أهداف استراتيجية يتراكم عبر كل الجلسات: التقدم يتجدد عند كل إنجاز
# يُسجَّل ضدها، وخيط خلفية يقيّم الأهداف النشطة دوريًا (كل 24 ساعة)
# فيرفع تقدمها تلقائيًا بوتيرة بطيئة — مراقبة مسار المشروع نحو الذكاء العام.
try:
    from ai.long_term_goals import (
        get_long_term_goals as _get_long_term_goals,
        ltg_stats as _ltg_stats,
        ltg_list as _ltg_list,
        ltg_evaluate as _ltg_evaluate,
    )
    _LTG_OK = True
    # المقيّم الدوري لا يبدأ تلقائيًا عند التحميل (قد يتعارض مع ضبط فترة
    # الاختبار) — يمكن تشغيله صراحة عبر _start_ltg_evaluator من الواجهة.
    def _start_ltg_evaluator(period_hours=None):
        try:
            _ltg = _get_long_term_goals()
            if period_hours:
                _ltg.set_eval_period(period_hours * 3600)
            _ltg.start_evaluator()
        except Exception:
            pass
except Exception:
    _LTG_OK = False
    def _get_long_term_goals():  # احتياطي آمن
        raise RuntimeError("الأهداف طويلة الأمد غير متاحة")
    def _ltg_stats():
        return {"active": 0, "achieved": 0, "avg_progress": 0.0}
    def _ltg_list(*a, **kw):
        return []
    def _ltg_evaluate():
        return {"evaluated_at": 0.0, "goals_updated": 0}
# ── 🆕 التفكير ما قبل الفعل (Pre-Action Reasoning) ──────────────────────
# قبل كل "فعل" (خطوة مهمة طويلة الأمد أو دور في مهمة تعاونية) يفكر الوكيل
# أولًا: خطوات متوقعة + مخاطر متوقعة لكل خطوة + بدائل أمان + ثقة + حكم
# (proceed / revise) — تحليل نمطي محلي 100% بلا أي API خارجي، ويُسجَّل
# كل تفكير في قاعدة SQLite محلية (data/pre_action_reasoning.db) تتراكم.
try:
    from ai.pre_action_reasoning import (
        get_pre_action_reasoner as _get_par_reasoner,
        reason_task as _reason_task,
        par_stats as _par_stats,
        par_latest as _par_latest,
        par_recall as _par_recall,
        par_learn as _par_learn,
        par_learned_stats as _par_learned_stats,
        par_calibration as _par_calibration,
        par_role_accuracy as _par_role_accuracy,
        reason_multi_task as _reason_multi_task,
        reason_multi_role_task as _reason_multi_role_task,
        resolve_collective_task as _resolve_collective_task,
        record_conflict_task as _record_conflict_task,
        learn_conflict_task as _learn_conflict_task,
        recall_conflicts_task as _recall_conflicts_task,
        conflict_stats_task as _conflict_stats_task,
    )
    _PAR_OK = True
except Exception:
    _PAR_OK = False
    def _get_par_reasoner():  # احتياطي آمن
        raise RuntimeError("التفكير ما قبل الفعل غير متاح")
    def _reason_task(*a, **kw):
        return None
    def _par_stats():
        return {"reasoned": 0, "proceeded": 0, "revised": 0,
                "avg_confidence": 0.0}
    def _par_latest(*a, **kw):
        return None
    def _par_recall(*a, **kw):
        return []
    def _par_learn(*a, **kw):
        return None
    def _par_learned_stats():
        return {"learned": 0, "correct": 0, "accuracy": 0.0}
    def _par_calibration(*a, **kw):
        return {"role": "", "n": 0, "correct": 0, "accuracy": 0.0,
                "calibration_effect": 0.0, "reason": ""}
    def _par_role_accuracy():
        return {}
    def _reason_multi_task(*a, **kw):
        return None
    def _reason_multi_role_task(*a, **kw):
        return None
    def _resolve_collective_task(*a, **kw):
        return None
    def _record_conflict_task(*a, **kw):
        return None
    def _learn_conflict_task(*a, **kw):
        return None
    def _recall_conflicts_task(*a, **kw):
        return []
    def _conflict_stats_task():
        return {"resolutions": 0, "measured": 0,
                "correct": 0, "accuracy": 0.0}
# ── 🆕 منهجية NSM (نقل سرّ عمل Manus إلى الوكلاء) ───────────────────────
# محرك منهجي حتمي يوثّق دورة (خطة → فحص فعلي → تنفيذ منضبط → تحقق →
# تعلم من الأخطاء) لكل مهمة، ويُسجّلها في data/methodology.db، ويغذّي
# الـagent بمبادئ المنهجية السبعة عبر system prompt.
try:
    from ai.methodology_engine import (
        method_task_started as _method_task_started,
        method_step as _method_step,
        method_task_finished as _method_task_finished,
        method_stats as _method_stats,
        method_latest_task as _method_latest_task,
        method_task_steps as _method_task_steps,
        method_record_lesson as _method_record_lesson,
        method_recall_lessons as _method_recall_lessons,
        method_principles_prompt as _method_principles_prompt,
    )
    _METH_OK = True
except Exception:
    _METH_OK = False
    def _method_task_started(*a, **kw):  # احتياطي آمن
        return None
    def _method_step(*a, **kw):
        return None
    def _method_task_finished(*a, **kw):
        return None
    def _method_stats():
        return {"tasks": 0, "tasks_ok": 0, "accuracy": 0.0,
                "total_steps": 0, "step_types": {},
                "inspect_verify_ratio": 0.0, "lessons": 0,
                "lessons_applied": 0}
    def _method_latest_task():
        return None
    def _method_task_steps(*a, **kw):
        return []
    def _method_record_lesson(*a, **kw):
        return None
    def _method_recall_lessons(*a, **kw):
        return []
    def _method_principles_prompt():
        return ""
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
    from ai.perf_profiler import measure_latency
except Exception:
    # التزيين بدون وحدة القياس: دالة محايدة تعيد الوظيفة الأصلية بلا قياس.
    def measure_latency(label):
        def decorator(fn):
            return fn
        return decorator

# ── وحدة التحميل الكسول والفهارس المحوسبة لـ CKG ─────────────────────────
try:
    from ai.ckg_loader import get_indices, search_ckg_query as _ckg_search_via_indices
    _CKG_LOADER_OK = True
except Exception:
    _CKG_LOADER_OK = False
    def _ckg_search_via_indices(q_norm):
        return {"concept_data": None, "ckg_related": [], "ckg_relations": [], "found": False}

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
    from ai.world_feed import WorldFeed
    from ai.quality_engine import QualityEngine
    from ai.immune_system import ImmuneSystem
    _WORLD_FEED_OK = True
except Exception:
    _WORLD_FEED_OK = False

try:
    from ai.self_narrative import SelfNarrative
    _SELF_NARRATIVE_OK = True
except Exception:
    _SELF_NARRATIVE_OK = False

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

# ── تفعيل MeshLogger (Phase 2 structured logging) ─────────────────────────
# Streamlit يعيد تشغيل السكربت كاملاً عند كل تفاعل، لذا نحمي من إعادة
# التهيئة المتكررة (وبالتالي تكرار الـ handlers) عبر session_state.
if not st.session_state.get("_mesh_logger_ready"):
    try:
        from logs.mesh_logger import MeshLogger
        MeshLogger(log_dir=str(BASE / "logs"))
        st.session_state["_mesh_logger_ready"] = True
    except Exception as _mesh_logger_exc:  # لا نفشل تحميل الواجهة بسبب اللوغر
        logger.warning(f"تعذّر تفعيل MeshLogger: {_mesh_logger_exc}")

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

/* ── نقل الشريط الجانبي (تسجيل الدخول والإعدادات) إلى يمين الشاشة ──
   Streamlit يضع الشريط الجانبي دائماً في بداية حاوية flex الرئيسية
   بصرياً على اليسار، بصرف النظر عن اتجاه RTL للنصوص. نعكس ترتيبهما
   البصري فقط بخاصية order دون المساس بترتيبهما الفعلي في DOM.

   🛠️ إصلاح: خاصية order في CSS تُفسَّر حسب اتجاه (direction) الحاوية
   نفسها لا حسب اتجاه الصفحة العامة فقط. الحاوية هنا كانت ترث
   direction:rtl من القاعدة العامة (html, body, [class*="css"])، فيصبح
   "البداية" البصرية لمحور flex هو اليمين، فينعكس معنى order بالكامل:
   العنصر order:1 (المحتوى الرئيسي) ينتهي به المطاف على اليمين،
   وorder:2 (الشريط الجانبي) على اليسار — أي عكس ما تشرحه التعليقات
   وعكس ما يُفترض أن يحدث فعلياً (وهذا ما كان يظهر في الواجهة المنشورة:
   الشريط الجانبي على اليسار). الحل: نفرض direction:ltr صراحة على حاوية
   الـflex نفسها فقط (حتى تُفسَّر قيم order فيزيائياً من اليسار لليمين
   كما كانت مصمَّمة)، ثم نعيد direction:rtl صراحة على كل من الشريط
   الجانبي والمحتوى الرئيسي حتى يبقى نص كل منهما من اليمين لليسار كالمعتاد. */
[data-testid="stAppViewContainer"] { display: flex; flex-direction: row; direction: ltr; }
[data-testid="stSidebar"] { order: 2; direction: rtl; }
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] [data-testid="stMain"] { order: 1; flex: 1 1 auto; min-width: 0; direction: rtl; }
/* 🛠️ إصلاح نهائي (كان محاولة سابقة تُلغي هذا الترتيب على الجوال ظنّاً بأن
   Streamlit يعرض الشريط الجانبي كطبقة منزلقة (overlay) في هذا العرض، وهو
   افتراض غير صحيح فعلياً في هذا التطبيق — لقطات شاشة المستخدم على الجوال
   أثبتت أن الشريط يظهر كعمود flex ثابت بجانب المحتوى تماماً كسطح المكتب،
   لا كطبقة منزلقة. إلغاء order هناك (order:initial) كان يُعيد الشريط إلى
   ترتيبه الطبيعي في DOM (أولاً ⇐ يسار الشاشة فعلياً بما أن الحاوية
   direction:ltr)، وهذا بالضبط ما ظهر في الصورة: القائمة على اليسار، ويتكرر
   عند أي rerun (كل ضغطة زر) لأن Streamlit يعيد رسم كامل الصفحة بنفس الحكم
   المرتبط بعرض الشاشة في كل مرة. الحل الدائم: نفرض نفس ترتيب سطح المكتب
   على كل الأحجام دون استثناء عبر !important حتى لا يطغى عليها أي قالب
   داخلي لاحق خاص بالجوال. */
[data-testid="stSidebar"] { order: 2 !important; }
[data-testid="stAppViewContainer"] > .main,
[data-testid="stAppViewContainer"] [data-testid="stMain"] { order: 1 !important; }
@media (max-width: 768px) {
    /* ── محاولة جعل انزلاق الشريط على الجوال من اليمين بدل اليسار ──
       تجريبي: Streamlit لا يوثّق آلية الانزلاق الداخلية (transform أو
       position) بدقة لكل نسخة، لذا نغطّي الاحتمالين معاً بـ!important
       (يتجاوز حتى inline style غير !important حسب مواصفات CSS). لو
       الآلية الفعلية مختلفة، هذا الجزء ببساطة لا يُطابق شيئاً ولا يُفسد
       شيئاً — ولازم تأكيد بصري فعلي على جهاز حقيقي بعد النشر. */
    [data-testid="stSidebar"] {
        left: auto !important;
        right: 0 !important;
    }
    [data-testid="stSidebar"][aria-expanded="false"] {
        transform: translateX(100%) !important;
    }
    [data-testid="stSidebar"][aria-expanded="true"] {
        transform: translateX(0) !important;
    }
}

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
    background-position: 30% 50%;
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


/* ملاحظة: أُلغي سابقاً تأثير دخول متدرّج (fade/rise) للبطاقات عند كل
   تحميل صفحة — كان يُنتج تأخيراً بصرياً متكرراً غير مرغوب. المحتوى
   الآن يظهر فوراً (opacity: 1) بدون حركة، انظر .metric-card أدناه. */

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
}
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

/* ── بينتو-جريد للإحصاءات — بطاقة مميزة كبيرة + بقية البطاقات بأحجام متفاوتة ──
   🛠️ إصلاح RTL: نفس سبب مشكلة الشريط الجانبي (راجع [data-testid="stAppViewContainer"]
   أعلاه) — الحاوية هنا لم تكن تفرض direction:rtl على نفسها صراحةً، فكانت
   تعتمد فقط على التوارث من القاعدة العامة. أي عنصر أب لاحق (مثل حاوية
   Streamlit الداخلية لعمود التبويب) قد يكسر هذا التوارث فيصبح ترتيب
   شبكة CSS Grid (grid-auto-flow) من اليسار لليمين بدل اليمين لليسار،
   وتفقد البطاقات محاذاتها المقصودة — وهو ما ظهر في لقطة شاشة المستخدم
   (قسم "تفاصيل الشبكة المعرفية" يظهر كنص مكدَّس بلا شبكة منظمة). */
.bento-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    grid-auto-flow: dense;
    gap: 0.8rem;
    margin-bottom: 0.6rem;
    direction: rtl;
    text-align: right;
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

/* ── ⬆️ زر «تحميل المزيد» — حد العرض المتدرج للذاكرة الطويلة ── */
.nsm-load-more-btn,
button[data-testid="stBaseButton-secondary"][aria-label="⬆️ تحميل المزيد"] {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    background: linear-gradient(135deg, var(--gold-tint), var(--surface2));
    color: var(--text);
    border: 1px solid var(--gold-soft);
    border-radius: 10px;
    padding: 0.45rem 1.1rem;
    font-size: 0.82rem;
    font-family: 'IBM Plex Sans Arabic', sans-serif;
    font-weight: 600;
    cursor: pointer;
    direction: rtl;
    transition: border-color .15s ease, transform .1s ease, box-shadow .15s ease;
}
.nsm-load-more-btn:hover {
    border-color: var(--gold);
    box-shadow: 0 4px 16px var(--shadow);
}
.nsm-load-more-btn:active { transform: scale(0.97); }

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
    display: flex;
    align-items: center;
    gap: 0.4rem;
}
.nsm-cmdk-item:hover, .nsm-cmdk-item.active { background: var(--gold-soft); }
.nsm-cmdk-item-parent { color: var(--text-muted); font-size: 0.85rem; }
.nsm-cmdk-item-sep { color: var(--text-muted); opacity: 0.6; }
.nsm-cmdk-empty {
    padding: 1.4rem 1.2rem;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.88rem;
    font-family: 'Tajawal', sans-serif;
}
.nsm-cmdk-hint {
    padding: 0.5rem 1.2rem 0.85rem;
    color: var(--text-muted);
    font-size: 0.72rem;
    font-family: 'Tajawal', sans-serif;
    border-top: 1px solid var(--border);
}
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

/* ═══════════════════════════════════════════════════════════════════════════
   NSM Aesthetic Layer v2 — صقل بصري موحّد
   ═══════════════════════════════════════════════════════════════════════════ */

/* خلفية أنعم + عمق */
.stApp {
    background: var(--bg) !important;
}
[data-testid="stAppViewContainer"] > .main {
    background: transparent !important;
}
.main .block-container {
    padding-top: 1.1rem !important;
    padding-bottom: 2.5rem !important;
    max-width: 1200px !important;
}

/* شريط علوي زجاجي */
header[data-testid="stHeader"] {
    background: color-mix(in srgb, var(--bg) 78%, transparent) !important;
    backdrop-filter: blur(14px) saturate(1.2);
    border-bottom: 1px solid var(--border);
}

/* الشريط الجانبي */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, color-mix(in srgb, var(--surface2) 92%, var(--gold) 8%) 0%, var(--bg) 100%) !important;
    border-left: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--text-muted);
}

/* التبويبات — أوضح وأكثر أناقة */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.35rem !important;
    background: var(--surface) !important;
    padding: 0.4rem !important;
    border-radius: 14px !important;
    border: 1px solid var(--border) !important;
    backdrop-filter: blur(10px);
    flex-wrap: wrap !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px !important;
    padding: 0.45rem 0.9rem !important;
    font-weight: 700 !important;
    color: var(--text-muted) !important;
    border: 1px solid transparent !important;
    transition: all .18s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text) !important;
    background: var(--gold-soft) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-grad) !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 4px 14px var(--gold-soft) !important;
}
.stTabs [data-baseweb="tab-highlight"] {
    display: none !important;
}
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* البطاقات والمقاييس */
[data-testid="stMetric"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 0.85rem 1rem !important;
    box-shadow: 0 4px 18px var(--shadow);
}
[data-testid="stMetric"] label {
    color: var(--text-muted) !important;
    font-weight: 600 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-weight: 800 !important;
}

/* Expander أنظف */
[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    overflow: hidden;
}
[data-testid="stExpander"] details summary {
    font-weight: 700 !important;
}

/* رسائل النجاح/التحذير */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
}

/* شريط تمرير أنيق */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: color-mix(in srgb, var(--gold) 45%, var(--border));
    border-radius: 8px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--gold);
}

/* 🆕 حُذفت .nsm-section-title من هنا: كانت نسخة ثانية من "عنوان قسم" بحجم
   خط (1.15rem) وأسلوب حدّ سفلي (تدرّج شفاف) مختلفَين عن .section-header
   المُعرَّف أعلى بهذا الملف (1.3rem، حدّ سفلي صلب + خط ذهبي 64px) رغم أنها
   تُستخدم لنفس الغرض بالضبط (عنوان مجموعة/قسم). كل استخداماتها الأربعة
   (عناوين المجموعات بـstreamlit_app.py وagents_hub.py) حُوّلت الآن إلى
   .section-header — نفس الكلاس المستخدم فعلياً في 94+ مكاناً آخر بالتطبيق
   — بدل الإبقاء على نمطين متنافسين لنفس الوظيفة. */
.nsm-hero-panel {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg, var(--gold-soft), var(--emerald-soft)), var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1rem;
    box-shadow: 0 8px 28px var(--shadow);
    direction: rtl;
}
.nsm-hero-panel::before {
    content: "";
    position: absolute;
    inset: -40% auto auto -10%;
    width: 220px; height: 220px;
    background: radial-gradient(circle, var(--gold-soft), transparent 70%);
    pointer-events: none;
}
.nsm-hero-title {
    position: relative;
    font-size: 1.55rem;
    font-weight: 900;
    background: var(--accent-grad);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    margin: 0 0 0.35rem;
}
.nsm-hero-sub {
    position: relative;
    color: var(--text-muted);
    font-size: 0.92rem;
    line-height: 1.75;
    margin: 0;
}
.nsm-chip-row {
    display: flex; flex-wrap: wrap; gap: 0.4rem;
    margin-top: 0.85rem; direction: rtl;
}
.nsm-chip {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem; font-weight: 700;
    background: var(--surface2);
    border: 1px solid var(--border);
    color: var(--text-muted);
}
.nsm-chip--accent {
    background: var(--gold-soft);
    border-color: color-mix(in srgb, var(--gold) 35%, var(--border));
    color: var(--gold);
}

/* محادثة موحّدة — فقاعات أوضح */
.ua-box, .agent-box {
    background:
        radial-gradient(ellipse 80% 50% at 100% 0%, var(--gold-soft), transparent 55%),
        var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 18px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 8px 24px var(--shadow) !important;
}
.ua-user .bbl, .agent-user .bbl {
    box-shadow: 0 6px 18px var(--gold-soft) !important;
}
.ua-bot .bbl, .agent-bot .bbl {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
}
.ua-badge, .agent-badge {
    background: var(--gold-soft) !important;
    border-color: color-mix(in srgb, var(--gold) 30%, var(--border)) !important;
    color: var(--gold) !important;
    font-weight: 700 !important;
}

/* أزرار ثانوية متناسقة */
.stButton>button {
    border-radius: 12px !important;
}
.stButton>button[kind="secondary"] {
    background: var(--surface2) !important;
}

/* تحسين toggle */
[data-testid="stWidgetLabel"] p {
    font-weight: 600 !important;
}

/* فاصل بصري خفيف */
.nsm-divider {
    height: 1px;
    margin: 1rem 0;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
}

@media (max-width: 768px) {
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }
    .nsm-hero-title { font-size: 1.25rem; }
    .stTabs [data-baseweb="tab"] {
        padding: 0.35rem 0.65rem !important;
        font-size: 0.85rem !important;
    }
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
# 🛠️ إصلاح جوهري: هذا الحقن (CSS + JS اللوحة/التبويبات) كان قبل هذا الإصلاح
# كوداً على مستوى الوحدة (module-level) داخل app_core.py. بما أن بايثون
# يستورد كل وحدة مرة واحدة فقط لكل عملية خادم (sys.modules cache)، فإن
# `from app_core import *` في streamlit_app.py لا يُعيد تنفيذ جسد هذه
# الوحدة في أي rerun لاحق — أي أن هذا الكود كان يُنفَّذ فعلياً مرة واحدة
# فقط طوال عمر عملية الخادم (أول استيراد بعد إعادة التشغيل)، وليس عند كل
# rerun كما توحي التعليقات الأصلية بداخله. النتيجة المُلاحَظة: بعد إعادة
# التشغيل يظهر التصميم صحيحاً (الشريط الجانبي يميناً) لأن هذا هو التشغيل
# الوحيد الذي نفّذ الحقن فعلياً؛ ثم أي تفاعل لاحق (زر، تبديل، بحث...) يسبب
# rerun عادياً لا يُعيد حقن وسم <style> هذا إطلاقاً، فتختفي قواعد RTL/order
# الخاصة بالشريط الجانبي من الـDOM ويظهر التطبيق بترتيبه الافتراضي (يساراً)
# — وهذا نفس السبب الذي يُفسّر عودة التبويب النشط إلى «الرئيسية»: اختفاء
# هذا العنصر من تيار العناصر (delta stream) يُغيّر مواضع العناصر التالية
# له في الشجرة (بما فيها st.tabs الرئيسية)، فيفقد Streamlit تتبّع التبويب
# المختار سابقاً ويعود للتبويب الافتراضي (الأول).
#
# الحل: تحويل هذا الحقن إلى دالة حقيقية تُستدعى صراحة من main() في
# streamlit_app.py — أي دالة تُستدعى فعلياً في كل rerun (السكربت الرئيسي
# يُعاد تنفيذه بالكامل من Streamlit في كل مرة، بخلاف الوحدات المستورَدة)،
# فيضمن إعادة حقن CSS/JS في كل مرة كما كان يُفترض دائماً.
def apply_runtime_css_and_chrome() -> None:
    """يحقن CSS الثيم الحالي + JS اللوحة/تلوين التبويبات.
    يجب استدعاؤها من main() في كل تشغيل للسكربت (وليس تركها على مستوى
    الوحدة) حتى تُطبَّق فعلياً بعد أي rerun، لا فقط أول استيراد للوحدة.
    """
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
                        '<input id="nsm-cmdk-input" class="nsm-cmdk-input" placeholder="ابحث عن قسم أو قسم فرعي... (Esc للإغلاق)" />' +
                        '<div id="nsm-cmdk-list" class="nsm-cmdk-list"></div>' +
                        '<div class="nsm-cmdk-hint">↑↓ للتنقّل · Enter للفتح · Esc للإغلاق</div>' +
                    '</div>';
                doc.body.appendChild(overlay);

                const fab = doc.createElement('button');
                fab.id = 'nsm-cmdk-fab';
                fab.className = 'nsm-cmdk-fab';
                fab.type = 'button';
                fab.title = 'بحث سريع (Ctrl+K)';
                fab.textContent = '⌘K';
                doc.body.appendChild(fab);

                // ── فهرسة كل الأقسام: التبويبات الرئيسية + التبويبات الفرعية
                // المتداخلة داخلها (مثل «المعرفة ‹ القرآن الكريم»)، حتى يقدر
                // المستخدم يقفز مباشرة لأي قسم فرعي بدل الاقتصار على الرئيسية.
                // الربط بين كل تبويب فرعي وأصله الرئيسي يتم عبر aria-labelledby
                // (نمط ARIA القياسي لِـ tabpanel ← id تبويبه الأصل)، وهو أثبت
                // من محاولة حساب الفهرس يدوياً لأن كل التبويبات (حتى غير
                // النشطة) موجودة بالـDOM دوماً وقد تتشابه تسمياتها.
                function buildIndex() {{
                    const groups = Array.from(doc.querySelectorAll('.stTabs'));
                    if (!groups.length) return [];
                    const rootGroup = groups[0];
                    const items = [];

                    function ownTabs(group) {{
                        return Array.from(group.querySelectorAll('[data-baseweb="tab-list"] [role="tab"]'))
                            .filter(function(t) {{ return t.closest('.stTabs') === group; }});
                    }}

                    ownTabs(rootGroup).forEach(function(t) {{
                        items.push({{
                            label: (t.textContent || '').trim(),
                            parent: null,
                            run: function() {{ t.click(); }},
                        }});
                    }});

                    for (let i = 1; i < groups.length; i++) {{
                        const g = groups[i];
                        // نبحث عن حاوية اللوحة الأصل بأكثر من محدد احتياطاً لاختلاف
                        // إصدار BaseWeb — data-baseweb هو الأساسي، role=tabpanel احتياط ARIA قياسي.
                        const parentPanel = g.closest('[data-baseweb="tab-panel"], [role="tabpanel"]');
                        if (!parentPanel) continue;
                        const parentId = parentPanel.getAttribute('aria-labelledby');
                        const parentTab = parentId ? doc.getElementById(parentId) : null;
                        if (!parentTab) continue;
                        const parentLabel = (parentTab.textContent || '').trim();
                        ownTabs(g).forEach(function(st) {{
                            items.push({{
                                label: (st.textContent || '').trim(),
                                parent: parentLabel,
                                run: function() {{
                                    parentTab.click();
                                    setTimeout(function() {{ st.click(); }}, 60);
                                }},
                            }});
                        }});
                    }}
                    return items;
                }}

                function renderList(filterText) {{
                    const list = doc.getElementById('nsm-cmdk-list');
                    list.innerHTML = '';
                    const f = (filterText || '').trim();
                    const all = buildIndex();
                    const matches = all.filter(function(it) {{
                        if (!f) return true;
                        const hay = (it.parent ? it.parent + ' ' : '') + it.label;
                        return hay.indexOf(f) !== -1;
                    }});
                    if (!matches.length) {{
                        const empty = doc.createElement('div');
                        empty.className = 'nsm-cmdk-empty';
                        empty.textContent = 'لا توجد أقسام مطابقة';
                        list.appendChild(empty);
                        return;
                    }}
                    matches.forEach(function(it, i) {{
                        const item = doc.createElement('div');
                        item.className = 'nsm-cmdk-item' + (i === 0 ? ' active' : '');
                        if (it.parent) {{
                            const parentSpan = doc.createElement('span');
                            parentSpan.className = 'nsm-cmdk-item-parent';
                            parentSpan.textContent = it.parent;
                            const sepSpan = doc.createElement('span');
                            sepSpan.className = 'nsm-cmdk-item-sep';
                            sepSpan.textContent = '‹';
                            item.appendChild(parentSpan);
                            item.appendChild(sepSpan);
                        }}
                        const labelSpan = doc.createElement('span');
                        labelSpan.textContent = it.label;
                        item.appendChild(labelSpan);
                        item.addEventListener('click', function() {{
                            it.run();
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
@measure_latency("load_arabic_roots")
def load_arabic_roots() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "arabic_roots_index.json")
    return data or {}


@st.cache_data(ttl=60)
def load_graph_metrics() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "graph_metrics.json")
    return data or {}


@st.cache_data(ttl=60)
@measure_latency("load_quran_index")
def load_quran_index() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "quran_index.json")
    return data or {}


@st.cache_data(ttl=300)
@measure_latency("load_all_quran_ayat")
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
def load_ckg_stats() -> Dict[str, int]:
    """أعداد سريعة عن الشبكة المعرفية (مفاهيم/علاقات/جذور/عناقيد) لعرضها
    كبطاقات إحصاء بالصفحة الرئيسية. تُشتق من load_ckg() المخزَّن مسبقاً
    بالكاش — لا نحمّل أو نحتفظ بنسخة إضافية من الملف (~40MB) بالذاكرة."""
    # عبر الفهارس المحوسبة (أسرع: لا مرّة على 7,300+ مفهوم).
    if _CKG_LOADER_OK:
        try:
            _s = get_indices().summary()
            if _s:
                return _s
        except Exception:
            pass
    data = load_ckg()
    concepts_map = (data or {}).get("concepts") or {}
    if not concepts_map:
        return {}
    relations_map = (data or {}).get("relations") or {}
    meta  = (data or {}).get("meta") or {}
    _meta = (data or {}).get("_meta") or {}
    clusters = meta.get("clusters")
    if not clusters:
        clusters = {
            c.get("cluster") for c in concepts_map.values()
            if isinstance(c, dict) and c.get("cluster")
        }
    try:
        return {
            "concepts":  int(_meta.get("total_concepts") or meta.get("total_concepts") or len(concepts_map)),
            "relations": int(_meta.get("total_relations") or meta.get("total_relations") or len(relations_map)),
            "roots":     int(meta.get("arabic_roots") or len((data or {}).get("arabic_roots") or {})),
            "clusters":  int(len(clusters)),
        }
    except Exception:
        return {}


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


@st.cache_resource(show_spinner=False)
def _get_self_narrative():
    """singleton لعملية Streamlit كاملة. ai/self_narrative.py كان مكتوباً
    ومختبراً (يمنح الجهاز 'صوتاً ذاتياً': يومية، جملة هوية متطورة) لكن غير
    مستورد من أي مكان إطلاقاً. مربوط هنا بالذاكرة الإيبيسودية الحقيقية بعد
    إصلاح _link_to_episodic() (كانت تبني Episode(content=...) بمعامل غير
    موجود في التوقيع الحقيقي فتسقط دوماً صامتاً)."""
    if not _SELF_NARRATIVE_OK:
        return None
    try:
        episodic = _get_episodic_engine()
        return SelfNarrative(episodic_memory=episodic)
    except Exception:
        return None




@st.cache_resource(show_spinner=False)
def _get_autonomous_will():
    """إرادة + تشغيل تلقائي: بحث وتعلّم وفحص بدون أمر."""
    try:
        from ai.auto_runtime import get_auto_runtime
        get_auto_runtime(start=True)
        from ai.autonomous_will import get_autonomous_will
        will = get_autonomous_will(start=True)
        # ربط DriveEngine إن وُجد لاحقاً
        try:
            from ai.drive_engine import DriveEngine
            # لا يُنشأ هنا إلزاماً؛ يُربط عند توفر mesh
        except Exception:
            pass
        return will
    except Exception:
        return None

@st.cache_resource(show_spinner=False)
def _get_world_feed():
    """singleton لعملية Streamlit كاملة. ai/world_feed.py + ai/quality_engine.py +
    ai/immune_system.py كانت الثلاثة مكتوبة ومختبرة لكن غير مستوردة من أي مكان
    إطلاقاً في المشروع (Phase 15). هنا يُربط WorldFeed فعلياً بـImmuneSystem
    الحقيقي وQualityEngine الحقيقي (بعد إصلاح توقيعي inspect()/evaluate() في
    world_feed.py التي كانت لا تطابق الواجهة الفعلية لهذين الملفين) وبالذاكرة
    الإيبيسودية الحقيقية عبر memory_callback — كل عنصر مقبول (quality_score >= 60
    وسمح به الجهاز المناعي) يُسجَّل كـEpisode حقيقي وليس بيانات وهمية.
    لا يبدأ الاستطلاع التلقائي بالخلفية هنا؛ ذلك قرار صريح من المستخدم بزر."""
    if not _WORLD_FEED_OK:
        return None
    try:
        immune  = ImmuneSystem()
        quality = QualityEngine()
        # المصادر الافتراضية المدمجة في WorldFeed (arXiv/HackerNews/TechCrunch)
        # منافذ معروفة وموثوقة لكنها غير موجودة في قائمة الثقة الافتراضية لـ
        # QualityEngine (التي تحوي فقط system/core/verified_feed/...) — بدون
        # هذا السطر تسقط كل عناصرها الحقيقية دوماً تقريباً تحت عتبة الجودة 60.
        for _src in WorldFeed.DEFAULT_SOURCES:
            quality.trust(_src.name)
        episodic = _get_episodic_engine()

        def _on_accept(item: dict) -> None:
            if episodic is None:
                return
            try:
                content = item.get("content", "")
                title   = item.get("title", "")
                score   = float(item.get("quality_score", 0.0))
                feature_vec = [
                    min(len(content), 2000) / 2000.0,
                    min(len(title), 200) / 200.0,
                    score / 100.0,
                ]
                episodic.record(
                    feature_vec=feature_vec,
                    target=score / 100.0,
                    outcome=score / 100.0,
                    source=f"world_feed:{item.get('source', 'unknown')}",
                    reward=0.0,
                    context={
                        "title": title,
                        "content": content[:500],
                        "url": item.get("url", ""),
                    },
                )
            except Exception:
                pass  # فشل التسجيل لا يجوز أن يكسر دورة الاستطلاع

            _sn = _get_self_narrative()
            if _sn is not None:
                try:
                    _sn.record_event(
                        "world_feed",
                        {"source": item.get("source", ""), "message": title or content[:60]},
                        surprise_score=0.0,
                        importance=min(1.0, score / 100.0),
                    )
                except Exception:
                    pass

        wf = WorldFeed(immune_system=immune, quality_engine=quality, min_quality=60.0)
        wf.set_memory_callback(_on_accept)
        return wf
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

    # ── SelfNarrative (كان يتيماً بالكامل) — تسجيل الحدث بإشارة حقيقية
    # محسوبة محلياً وبأمان (لا اعتماد على متغيرات نطاق try سابق).
    try:
        _sn = _get_self_narrative()
        if _sn is not None:
            _ok_signal = 1.0 if (response or "").strip() else 0.0
            _sn.record_event(
                "decision",
                {"message": (query or "")[:80]},
                surprise_score=0.0,
                importance=0.5 if _ok_signal else 0.3,
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


def _persist_chat_message(session_id: str, role: str, content: str, source_badge: str = "") -> None:
    """يخزّن رسالة واحدة من تبويب المحادثة بشكل دائم عبر
    ai/chat_history_store.py (memory/chat_history.db) — يحل مشكلة فقدان
    st.session_state.nsm_messages بالكامل بانتهاء الجلسة. استيراد كسول +
    تدهور آمن كامل (نفس نمط _record_chat_episode أعلاه): أي فشل
    (وحدة غير موجودة، قرص ممتلئ، ...) يُبتلَع صامتاً ولا يكسر تجربة
    المحادثة الحيّة إطلاقاً."""
    try:
        from ai.chat_history_store import save_message
        save_message(session_id, role, content, source_badge)
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


@measure_latency("search_knowledge")
def search_knowledge(query: str, force_legacy_ckg: bool = False) -> Dict:
    """البحث الشامل في قاعدة المعرفة.

    `force_legacy_ckg` (اختباري) يُعطّل الفهارس المحوسبة ويرجع للسلوك القديم
    — يُستخدم في اختبارات التوافق للتأكد من أن الفهارس تعطي نفس النتائج.
    """
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

    # بحث مباشر — عبر الفهارس المحوسبة (ai.ckg_loader) إن كانت متاحة،
    # وإلا فبالسلوك القديم (حلقات normalize على كل المفاهيم والعلاقات).
    if _CKG_LOADER_OK and not force_legacy_ckg:
        try:
            _idx_result = _ckg_search_via_indices(q_norm)
            concept_data = _idx_result.get("concept_data")
            ckg_related = list(_idx_result.get("ckg_related") or [])
            ckg_relations = list(_idx_result.get("ckg_relations") or [])
        except Exception:
            concept_data, ckg_related, ckg_relations = None, [], []
    if not _CKG_LOADER_OK or force_legacy_ckg:
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


def _redact_secrets(text: str) -> str:
    """يخفي قيم أي متغير بيئة اسمه يحتوي KEY/TOKEN/SECRET/PASSWORD من نص
    مُعطى، قبل عرضه بالواجهة — يحمي من تسرّب أسرار عرضي عند تنفيذ أوامر
    تكشفها بالخطأ (مثلاً env أو printenv بلوحة المطوّر), دون منع تنفيذ
    الأمر نفسه أو حجب مخرجات غير حسّاسة."""
    if not text:
        return text
    for _name, _val in os.environ.items():
        if not _val or len(_val) < 6:
            continue
        if re.search(r"KEY|TOKEN|SECRET|PASSWORD", _name, re.IGNORECASE) and _val in text:
            text = text.replace(_val, f"***[{_name}]***")
    return text


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



def render_skip_link(target_id: str = "nsm-main-content", label: str = "تخطّى إلى المحتوى الرئيسي") -> None:
    """رابط تخطٍّ للوصول بلوحة المفاتيح/قارئ الشاشة (WCAG 2.4.1) — يظهر
    أولاً ضمن ترتيب الـtab، يأخذ المستخدم إلى بداية المحتوى الرئيسي بدل
    الاضطرار للمرور على كل عناصر التنقّل. خفيف الوزن لأنه مخفي بصرياً
    حتى يكتسب التركيز (:focus-visible)."""
    st.markdown(
        f'<a class="nsm-skip-link" href="#{target_id}" data-testid="nsm-skip-link">{label}</a>',
        unsafe_allow_html=True,
    )


def render_focus_styles() -> None:
    """حقن CSS لإطار تركيز مرئي وواضح على العناصر التفاعلية.
    إطار Streamlit الافتراضي باهت يصعب رؤيته على خلفية داكنة، لذا نُضيف
    حلقة (ring) عالية التباين. لا تطال الألوان الأساسية — tokens القائمة
    تبقى كما هي تماماً."""
    st.markdown(
        """
        <style>
        :focus-visible {
            outline: 2px solid var(--primary, #7C5CFC);
            outline-offset: 2px;
            border-radius: 6px;
        }
        button:focus-visible,
        [role="tab"]:focus-visible,
        [data-baseweb="tab"]:focus-visible,
        a:focus-visible {
            outline: 2px solid var(--gold, #D4AF37);
            outline-offset: 3px;
        }
        .nsm-skip-link {
            position: absolute;
            top: -40px;
            inset-inline-start: 0;
            z-index: 10000;
            background: var(--primary, #7C5CFC);
            color: #fff;
            padding: 0.5rem 1rem;
            border-radius: 0 0 6px 0;
            font-family: 'Tajawal', sans-serif;
            text-decoration: none;
            transition: top .2s ease;
        }
        .nsm-skip-link:focus-visible {
            top: 0;
            outline: 2px solid var(--gold, #D4AF37);
            outline-offset: 2px;
        }
        .nsm-copy-btn { margin-inline-start: 0.5rem; }

        /* ── حالة فارغة موحّدة (render_empty_state) ── */
        .empty-state {
            text-align: center;
            padding: 2.2rem 1rem;
            border: 1px dashed var(--border, rgba(255,255,255,0.18));
            border-radius: 12px;
            color: var(--text-muted, #94a3b8);
            font-family: 'Tajawal', sans-serif;
        }
        .empty-state-icon { font-size: 2rem; margin-bottom: 0.4rem; }
        .empty-state-title { font-weight: 700; font-size: 1.05rem; }
        .empty-state-hint { font-size: 0.85rem; margin-top: 0.3rem; }

        /* ── شريط KPI (render_kpi_strip) ── */
        .kpi-strip {
            display: flex;
            gap: 0.6rem;
            flex-wrap: wrap;
            margin: 0.6rem 0;
        }
        .kpi-strip-card {
            flex: 1;
            min-width: 110px;
            padding: 0.7rem 0.9rem;
            border-radius: 10px;
            background: var(--secondaryBackgroundColor, #1B2333);
            border: 1px solid var(--border, rgba(255,255,255,0.08));
            border-inline-start: 3px solid var(--primary, #7C5CFC);
            font-family: 'Tajawal', sans-serif;
        }
        .kpi-strip-card.accent-gold   { border-inline-start-color: var(--gold, #D4AF37); }
        .kpi-strip-card.accent-blue   { border-inline-start-color: #3b82f6; }
        .kpi-strip-card.accent-green  { border-inline-start-color: #22c55e; }
        .kpi-strip-card.accent-red    { border-inline-start-color: #ef4444; }
        .kpi-strip-value { font-size: 1.3rem; font-weight: 800; }
        .kpi-strip-label { font-size: 0.78rem; color: var(--text-muted, #94a3b8); }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, hint: str = "", icon: str = "🫥") -> None:
    """رسالة \"لا توجد بيانات\" موحَّدة بصرياً بدل الرسائل النصية المتباينة
    بين التبويبات. مقروءة لقارئ الشاشة عبر aria-live=\"polite\"، ومتّسقة
    الهوامش والأيقونة في كل موضع يستدعيها."""
    hint_html = f'<div class="empty-state-hint">{hint}</div>' if hint else ""
    st.markdown(
        f"""
        <div class="empty-state" role="status" aria-live="polite">
            <div class="empty-state-icon" aria-hidden="true">{icon}</div>
            <div class="empty-state-title">{title}</div>
            {hint_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kpi_strip(items: list) -> None:
    """شريط KPIs نحيف متجاوب — يستقبل قائمة [(value, label)] أو
    [(value, label, accent)] حيث accent أحد: gold | blue | green | red
    | primary (افتراضي). القيم تُعرَض فورياً كما هي. منفصل عن metric_card
    لأنها موجّهة لعرض أفقي مضغوط يلائم الشاشات الصغيرة."""
    if not items:
        return
    cards = []
    for entry in items:
        if len(entry) == 2:
            value, label = entry
            accent = "primary"
        else:
            value, label, accent = entry
        # تهريب كامل بالترتيب الصحيح: & أولاً (حتى لا نُعيد تهريب ما هُرّب)،
        # ثم < و> — يمنع حقن HTML من أي قيمة قادمة من بيانات المستخدم.
        safe_value = str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        safe_label = str(label).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        cards.append(
            f'<div class="kpi-strip-card accent-{accent}">'
            f'<div class="kpi-strip-value">{safe_value}</div>'
            f'<div class="kpi-strip-label">{safe_label}</div>'
            f"</div>"
        )
    st.markdown(
        f'<div class="kpi-strip" role="list">{"".join(cards)}</div>',
        unsafe_allow_html=True,
    )


__all__ = [
    _name for _name in dir()
    if not _name.startswith("__") and _name not in ("annotations", "_name")
]
# ── ملاحظة صيانة مهمة ────────────────────────────────────────────────────
# __all__ هنا مُحسَبة تلقائياً (كل اسم مُعرَّف بهذا الملف عدا dunder methods)
# بدل قائمة ثابتة يدوية. السبب: النسخة اليدوية السابقة فاتها دالة أُضيفت
# لاحقاً من جلسة أخرى موازية (load_ckg_stats عبر commit 2172695) لأن تلك
# الجلسة لم تكن تعرف بوجود قائمة __all__ يدوية يجب تحديثها — سبّب هذا
# NameError كامل عطّل الصفحة الرئيسية بالإنتاج. الحساب التلقائي هنا يجعل
# أي دالة/ثابت جديد يُضاف لهذا الملف (من أي جلسة/أداة) يُصدَّر تلقائياً
# لكل صفحات ui_pages/ عبر `from app_core import *` دون أي خطوة يدوية إضافية.
