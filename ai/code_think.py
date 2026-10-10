# -*- coding: utf-8 -*-
"""ai/code_think.py — نموذج برمجة مفتوح الأوزان للعقد عبر توكن Hugging Face نفسه.

يستدعي موجّه Inference Providers (https://router.huggingface.co/v1، متوافق مع
OpenAI) بتوكن HUGGINGFACE_API_KEY/HF_TOKEN الموجود أصلاً — لا مفتاح إضافي.

القائمة مرتّبة من الأقوى إلى الأرخص (قابلة للتغيير: NSM_CODE_MODELS مفصولة
بفواصل، لأن الترتيب يتغيّر كل شهر). تحتاج «تعمل بلا مشاكل» فكل ما يلي لا يرفع
استثناءً أبداً:
  • نموذج غير مدعوم/محذوف (400/404/422) → يُتخطّى 6 ساعات وينتقل للتالي.
  • نفاد رصيد HF الشهري المجاني (402) → إيقاف كل الاستدعاءات 6 ساعات (لا طلبات
    مهدورة) ثم إعادة المحاولة تلقائياً.
  • 429 → تبريد 5 دقائق لذلك النموذج. 401/403 → إيقاف ساعة (توكن خاطئ).
  • أي فشل آخر → تبريد دقيقتين.
  • مهلة إجمالية صارمة (≤13ث) أقل من مهلة اليد (15ث) مهما تعدّدت المحاولات.

تنبيه صريح: الطبقة المجانية من HF رصيدها شهري صغير (≈$0.10 وقت الكتابة)،
فالنماذج الضخمة تستهلكه بسرعة؛ لذلك القائمة تنتهي بنموذج برمجة أرخص.
التعطيل التلقائي فقط: NSM_NODE_NEURAL_AUTO=0 (الأداة اليدوية تبقى).
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import urllib.error
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ROUTER_URL = "https://router.huggingface.co/v1/chat/completions"

# أقوى → أرخص. أول نموذج موثَّق كمتاح عبر موجّه HF؛ الباقي يُتخطّى تلقائياً
# إن لم يكن مخدوماً (400/404) فلا ضرر من خطأ في المعرّف.
DEFAULT_CODE_MODELS: List[str] = [
    "Qwen/Qwen3-Coder-480B-A35B-Instruct",
    "deepseek-ai/DeepSeek-V3.2",
    "zai-org/GLM-4.6",
    "Qwen/Qwen3-Coder-30B-A3B-Instruct",
]

MAX_PROMPT_CHARS = 6000
MAX_TOKENS_CAP = 800
TOTAL_DEADLINE = 13.0
PER_CALL_TIMEOUT = 11.0

_BLOCK_NO_CREDITS = 6 * 3600.0
_BLOCK_BAD_MODEL = 6 * 3600.0
_BLOCK_BAD_TOKEN = 3600.0
_COOL_RATE_LIMIT = 300.0
_COOL_OTHER = 120.0

_lock = threading.Lock()
_model_blocked_until: Dict[str, float] = {}
_global_blocked_until: float = 0.0

_SYSTEM = (
    "You are an expert software engineer. Answer concisely and correctly. "
    "Give working code in fenced blocks and a brief explanation. "
    "Reply in the same language the user wrote in."
)

# كلمات شائعة (let/class/import/api/git...) أُسقطت عمداً: كل سؤال عام يُصنَّف
# برمجياً خطأً يستهلك الرصيد المجاني الشهري الصغير. نطلب أشكالاً برمجية صريحة.
_CODE_HINTS = re.compile(
    r"```|\bdef \w+\(|\bclass \w+\s*[:({]|\bimport \w+|\bfrom \w+ import\b|"
    r"\bfunction\s*\w*\s*\(|=>|\bconsole\.log|\bSELECT\b.+\bFROM\b|"
    r"traceback|stack ?trace|syntaxerror|nameerror|typeerror|segfault|"
    r"\b(?:regex|refactor|pytest|unit ?test|docker ?file|bash script|"
    r"python|javascript|typescript|golang|rustlang|c\+\+|streamlit|"
    r"debug|bug fix|compile error)\b|"
    r"\.(?:py|js|ts|rs|go|cpp|sql|sh)\b|"
    r"كود|برمج|دالة برمجية|خطأ برمجي|اكتب (?:لي )?(?:سكربت|برنامج|دالة|كلاس)|"
    r"باغ|تصحيح الخطأ|استعلام sql|ريجكس|سكربت",
    re.IGNORECASE | re.DOTALL,
)


def code_models() -> List[str]:
    env = os.getenv("NSM_CODE_MODELS", "").strip()
    if env:
        models = [m.strip() for m in env.split(",") if m.strip()]
        if models:
            return models
    return list(DEFAULT_CODE_MODELS)


def _hf_key() -> str:
    return os.getenv("HUGGINGFACE_API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()


def _auto_enabled() -> bool:
    return os.getenv("NSM_NODE_NEURAL_AUTO", "1").strip().lower() not in ("0", "false", "no", "off")


def code_available() -> bool:
    """مفتاح HF موجود ولا حظر عام نشط (نفاد رصيد/توكن خاطئ)."""
    return bool(_hf_key()) and time.time() >= _global_blocked_until


def is_code_prompt(text: str) -> bool:
    return bool(_CODE_HINTS.search(text or ""))


def _reset_state() -> None:  # للاختبارات
    global _global_blocked_until
    with _lock:
        _model_blocked_until.clear()
        _global_blocked_until = 0.0


def _classify(exc: Exception):
    """(حظر عام؟، مدة الحظر بالثواني)"""
    if isinstance(exc, urllib.error.HTTPError):
        c = exc.code
        if c == 402:
            return True, _BLOCK_NO_CREDITS
        if c in (401, 403):
            return True, _BLOCK_BAD_TOKEN
        if c in (400, 404, 422):
            return False, _BLOCK_BAD_MODEL
        if c == 429:
            return False, _COOL_RATE_LIMIT
    return False, _COOL_OTHER


def code_think(prompt: str, max_tokens: int = 600) -> Dict[str, Any]:
    """يرجع دائماً dict: {ok, text, model, provider, used_open_source_model, reason?}.
    لا يرفع استثناءً أبداً (التحقق من المدخلات فقط يرفع ValueError ليرفضه المستدعي)."""
    global _global_blocked_until
    prompt = (prompt or "").strip()
    if not prompt:
        raise ValueError("prompt فارغ")
    if len(prompt) > MAX_PROMPT_CHARS:
        raise ValueError(f"prompt أطول من {MAX_PROMPT_CHARS} حرف — اختصره قبل الاستدعاء")
    max_tokens = min(max(int(max_tokens), 1), MAX_TOKENS_CAP)

    def _fail(reason: str) -> Dict[str, Any]:
        return {"ok": False, "text": "", "model": None, "provider": "huggingface-router",
                "used_open_source_model": False, "reason": reason}

    key = _hf_key()
    if not key:
        return _fail("no_hf_key")
    now = time.time()
    if now < _global_blocked_until:
        return _fail("blocked_until_%d" % int(_global_blocked_until - now))

    from ai import llm_fallback as _lf  # _post_json قابلة للاستبدال في الاختبارات

    t0 = time.time()
    last_reason = "no_model_available"
    for model in code_models():
        remaining = TOTAL_DEADLINE - (time.time() - t0)
        if remaining < 2.0:
            last_reason = "deadline"
            break
        with _lock:
            if _model_blocked_until.get(model, 0.0) > time.time():
                continue
        try:
            data = _lf._post_json(
                ROUTER_URL,
                {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": _SYSTEM},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": 0.2,
                },
                {"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                int(min(PER_CALL_TIMEOUT, remaining)),
            )
            text = ""
            if isinstance(data, dict) and data.get("choices"):
                text = ((data["choices"][0] or {}).get("message") or {}).get("content") or ""
            text = text.strip()
            if not text:
                raise ValueError("رد فارغ")
            return {"ok": True, "text": text, "model": model, "provider": "huggingface-router",
                    "used_open_source_model": True,
                    "latency_ms": round((time.time() - t0) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            is_global, secs = _classify(exc)
            with _lock:
                if is_global:
                    _global_blocked_until = time.time() + secs
                else:
                    _model_blocked_until[model] = time.time() + secs
            last_reason = f"{type(exc).__name__}:{getattr(exc, 'code', '')}"
            logger.info("code_think: %s فشل (%s) — حظر %.0fث%s", model, last_reason, secs,
                        " (عام)" if is_global else "")
            if is_global:
                break
    return _fail(last_reason)


def code_first_generate(prompt: str, max_tokens: int = 400) -> Optional[Dict[str, Any]]:
    """للاستخدام التلقائي: None بلا أثر إن: معطَّل/لا مفتاح/ليس سؤال برمجة/فشل."""
    try:
        if not (_auto_enabled() and code_available() and is_code_prompt(prompt)):
            return None
        res = code_think(prompt[:MAX_PROMPT_CHARS], max_tokens=max_tokens)
        return res if res.get("ok") else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("code_first_generate failed: %s", exc)
        return None
