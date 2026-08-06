"""
ai/stt_engine.py
==================
تحويل صوت → نص عربي (Speech-to-Text) لدعم الإدخال الصوتي في المحادثة.

يستخدم Gemini 2.5 Flash متعدد الوسائط (inline_data audio) بنفس نمط
استدعاء REST الخام المستخدم في ai/nsm_agent_core.py (بدون SDK إضافي)،
لأن هذا هو المزوّد المتاح دائماً في هذا المشروع (GOOGLE_API_KEY).

الاستخدام:
    from ai.stt_engine import transcribe_audio
    text, error = transcribe_audio(audio_bytes, mime_type="audio/wav")
    # للهجة اليمنية: dialect_mode="yemeni" (يحافظ على ألفاظ اللهجة)
    text, error = transcribe_audio(audio_bytes, dialect_mode="yemeni")
"""
from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Optional, Tuple

from ai.offline_mode import is_offline, offline_message

_STT_MODEL = "gemini-2.5-flash"
_TRANSCRIBE_PROMPT_MSA = (
    "انسخ (فرّغ) هذا المقطع الصوتي حرفياً إلى نص عربي فصيح دون أي إضافة "
    "أو تعليق أو ترجمة — أعد النص المنطوق فقط كما هو. إن كان الصوت غير "
    "مفهوم أو صامتاً، أعد نصاً فارغاً."
)

_TRANSCRIBE_PROMPT_YEMENI = (
    "فرّغ هذا المقطع الصوتي إلى نص عربي كما نُطق حرفياً. "
    "إذا كانت اللهجة يمنية أو خليجية فحافظ على ألفاظ اللهجة "
    "(مثل: ايش، ليش، وين، ياخوي، ابشر، سدا) ولا تحوّلها إلى فصحى. "
    "لا تضف تعليقاً أو ترجمة — أعد النص المنطوق فقط. "
    "إن كان الصوت غير مفهوم أو صامتاً، أعد نصاً فارغاً."
)

# توافق خلفي
_TRANSCRIBE_PROMPT = _TRANSCRIBE_PROMPT_MSA

MAX_AUDIO_BYTES = 15 * 1024 * 1024  # 15MB سقف أمان قبل الإرسال للـAPI


def transcribe_audio(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    dialect_mode: str = "msa",
) -> Tuple[str, Optional[str]]:
    """
    يحوّل صوتاً إلى نص عربي. يعيد (النص, رسالة_خطأ).
    dialect_mode: "msa" | "yemeni" | "auto"
      - yemeni: يحافظ على ألفاظ اللهجة في التفريغ
      - msa: يفضّل صياغة فصيحة
    لا يرفع استثناءً أبداً.
    """
    if not audio_bytes:
        return "", "لم يصل أي صوت للتفريغ."

    if len(audio_bytes) > MAX_AUDIO_BYTES:
        return "", "المقطع الصوتي أطول من الحد المسموح (15MB)."

    if is_offline():
        return "", offline_message("تفريغ الصوت لنص (Gemini STT)")

    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        return "", "⚠️ لا يوجد مفتاح API — أضف GOOGLE_API_KEY في Streamlit Secrets"

    url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
           f"{_STT_MODEL}:generateContent?key={api_key}")

    mode = (dialect_mode or "msa").strip().lower()
    if mode in ("yemeni", "ye", "dialect"):
        prompt = _TRANSCRIBE_PROMPT_YEMENI
    else:
        prompt = _TRANSCRIBE_PROMPT_MSA

    body = {
        "contents": [{
            "role": "user",
            "parts": [
                {"text": prompt},
                {"inline_data": {"mime_type": mime_type, "data": base64.b64encode(audio_bytes).decode("ascii")}},
            ],
        }],
        "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.0},
    }

    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        return text, None
    except urllib.error.HTTPError as e:
        return "", f"تعذّر تفريغ الصوت (خطأ خادم {e.code})."
    except (urllib.error.URLError, TimeoutError):
        return "", "تعذّر الاتصال بخدمة تفريغ الصوت — تحقّق من الشبكة."
    except (KeyError, IndexError, json.JSONDecodeError):
        return "", "لم يتم التعرّف على أي كلام في المقطع الصوتي."
    except Exception:
        return "", "حدث خطأ غير متوقع أثناء تفريغ الصوت."
