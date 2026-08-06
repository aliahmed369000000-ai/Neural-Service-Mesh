"""
دمج المسار الصوتي مع اللهجة اليمنية — NSM

لا يوجد حالياً نموذج TTS/ASR يمني مفتوح جاهز للإنتاج بنفس سهولة Edge/Gemini،
لذا هذه الطبقة تُحسّن المسار العملي:

  STT  : تفريغ يحافظ على اللهجة (بدل إجبار الفصحى) + كشف لهجي بعد التفريغ
  TTS  : اختيار أصوات خليجية/عربية أقرب صوتياً + تطبيع نص قبل السرد
  خط أنابيب: صوت → نص يمني → (اختياري) تحليل dialect_boost

الاستخدام:
    from ai.yemeni_voice import transcribe_yemeni, speak_yemeni, voice_turn

    text, err, meta = transcribe_yemeni(audio_bytes)
    result = speak_yemeni("كيفك ياخوي")
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("yemeni_voice")

# أصوات Edge الأقرب للهجة اليمنية (خليج/جزيرة عربية) — مرتبة بالتفضيل
YEMENI_PREFERRED_EDGE_VOICES = [
    "ar-SA-HamedNeural",      # سعودي رجالي
    "ar-SA-ZariyahNeural",    # سعودي نسائي
    "ar-AE-HamdanNeural",     # إماراتي
    "ar-AE-FatimaNeural",
    "ar-KW-FahedNeural",      # كويتي
    "ar-QA-MoazNeural",       # قطري
    "ar-BH-AliNeural",
    "ar-OM-AbdullahNeural",
]

_YEMENI_STT_PROMPT = (
    "فرّغ هذا المقطع الصوتي إلى نص عربي كما نُطق حرفياً. "
    "إذا كانت اللهجة يمنية أو خليجية فحافظ على ألفاظ اللهجة "
    "(مثل: ايش، ليش، وين، ياخوي، ابشر، سدا، قات) ولا تحوّلها إلى فصحى. "
    "لا تضف تعليقاً أو ترجمة — أعد النص المنطوق فقط. "
    "إن كان الصوت غير مفهوم أو صامتاً، أعد نصاً فارغاً."
)

_MSA_STT_PROMPT = (
    "انسخ (فرّغ) هذا المقطع الصوتي حرفياً إلى نص عربي فصيح دون أي إضافة "
    "أو تعليق أو ترجمة — أعد النص المنطوق فقط كما هو. إن كان الصوت غير "
    "مفهوم أو صامتاً، أعد نصاً فارغاً."
)


@dataclass
class VoiceTurnResult:
    transcript: str = ""
    stt_error: Optional[str] = None
    dialect_score: float = 0.0
    is_yemeni: bool = False
    rag_context: str = ""
    audio_bytes: bytes = b""
    tts_error: Optional[str] = None
    tts_provider: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def ok_stt(self) -> bool:
        return bool(self.transcript) and not self.stt_error

    @property
    def ok_tts(self) -> bool:
        return bool(self.audio_bytes) and not self.tts_error


def _pick_edge_voice(preferred: str = "") -> str:
    if preferred:
        return preferred
    env = os.getenv("NSM_YEMENI_VOICE", "").strip()
    if env:
        return env
    return YEMENI_PREFERRED_EDGE_VOICES[0]


def transcribe_yemeni(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    preserve_dialect: bool = True,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """
    تفريغ صوتي موجّه للهجة.
    يُرجع (نص, خطأ, meta) حيث meta قد تتضمن dialect_score بعد التفريغ.
    """
    meta: Dict[str, Any] = {"preserve_dialect": preserve_dialect, "dialect_score": 0.0}
    try:
        from ai.stt_engine import transcribe_audio
    except Exception as e:
        return "", f"تعذّر تحميل stt_engine: {e}", meta

    # استدعاء مع prompt مخصّص إن دعم المحرك ذلك، وإلا fallback
    try:
        text, err = transcribe_audio(
            audio_bytes,
            mime_type=mime_type,
            dialect_mode="yemeni" if preserve_dialect else "msa",
        )
    except TypeError:
        # نسخة قديمة بدون dialect_mode
        text, err = transcribe_audio(audio_bytes, mime_type=mime_type)

    if err or not text:
        return text or "", err, meta

    try:
        from ai.yemeni_dialect import detect_yemeni_score, normalize_yemeni
        score = float(detect_yemeni_score(text))
        meta["dialect_score"] = score
        meta["is_yemeni"] = score >= 0.25
        meta["normalized"] = normalize_yemeni(text)
    except Exception:
        pass

    try:
        from ai.dialect_boost import analyze_and_boost
        info = analyze_and_boost(text, top_k_rag=2)
        meta["rag_context"] = info.get("rag_context") or ""
        meta["dialect_score"] = float(info.get("dialect_score") or meta.get("dialect_score") or 0.0)
        meta["is_yemeni"] = bool(info.get("is_yemeni") or meta.get("is_yemeni"))
    except Exception:
        pass

    return text, None, meta


def speak_yemeni(
    text: str,
    voice: str = "",
    use_dialect_voice: bool = True,
) -> Any:
    """
    سرد نص بصوت عربي مفضّل للهجة (خليجي/جزيرة).
    يُرجع TTSResult من tts_engine.
    """
    from ai.tts_engine import TTSEngine

    text = (text or "").strip()
    if not text:
        from ai.tts_engine import TTSResult, TTSProvider
        return TTSResult(audio_bytes=b"", provider=TTSProvider.EDGE, error="نص فارغ")

    try:
        from ai.yemeni_dialect import normalize_yemeni
        # تطبيع خفيف دون مسح طابع اللهجة بالكامل
        text_for_tts = normalize_yemeni(text) if use_dialect_voice else text
        # إن أفرغ التطبيع النص، ارجع للأصل
        if not text_for_tts.strip():
            text_for_tts = text
    except Exception:
        text_for_tts = text

    v = _pick_edge_voice(voice) if use_dialect_voice else (voice or "")
    engine = TTSEngine()
    # مرّر تلميح يمني للمحرك إن دعمه
    try:
        return engine.synthesize(text_for_tts, voice=v, dialect_hint="yemeni")
    except TypeError:
        return engine.synthesize(text_for_tts, voice=v)


def voice_turn(
    audio_bytes: bytes,
    mime_type: str = "audio/wav",
    also_speak_reply: Optional[str] = None,
) -> VoiceTurnResult:
    """
    دورة صوتية كاملة: تفريغ يمني → تحليل لهجة → (اختياري) سرد رد.
    """
    out = VoiceTurnResult()
    text, err, meta = transcribe_yemeni(audio_bytes, mime_type=mime_type)
    out.transcript = text
    out.stt_error = err
    out.dialect_score = float(meta.get("dialect_score") or 0.0)
    out.is_yemeni = bool(meta.get("is_yemeni"))
    out.rag_context = meta.get("rag_context") or ""
    out.meta = meta

    if also_speak_reply and also_speak_reply.strip():
        tts = speak_yemeni(also_speak_reply)
        out.audio_bytes = getattr(tts, "audio_bytes", b"") or b""
        out.tts_error = getattr(tts, "error", None)
        out.tts_provider = getattr(getattr(tts, "provider", None), "value", "") or str(
            getattr(tts, "provider", "")
        )
    return out


def list_yemeni_voices() -> List[str]:
    return list(YEMENI_PREFERRED_EDGE_VOICES)
