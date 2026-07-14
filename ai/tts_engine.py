"""
TTS Engine — محرك تحويل النص لصوت — NSM
=========================================
يوفّر طبقة سرد صوتي حقيقية لأي نص عربي/إنجليزي، بنفس فلسفة
LLMFallback (سلسلة مزوّدين مع تراجع تلقائي عند الفشل).

الأولوية في اختيار المزوّد (auto-detect):
  1. Google Gemini TTS  (GOOGLE_API_KEY)  — جودة صوت عصبية عالية، متعدد اللغات ✅
  2. ElevenLabs         (ELEVENLABS_API_KEY) — جودة استوديو ممتازة (مدفوع)
  3. Microsoft Edge TTS (بدون مفتاح)      — أصوات عصبية عربية مجانية ✅
  4. Google Translate TTS / gTTS (بدون مفتاح) — احتياطي أخير، جودة أبسط

الاستخدام:
    from ai.tts_engine import TTSEngine

    tts = TTSEngine()
    result = tts.synthesize("السلام عليكم ورحمة الله", voice="ar-SA-HamedNeural")
    with open("out.mp3", "wb") as f:
        f.write(result.audio_bytes)
    print(result.provider.value, result.duration_est_sec)

التثبيت (requirements.txt):
    edge-tts>=6.1.0
    gTTS>=2.5.0
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger("TTSEngine")


# ════════════════════════════════════════════════════════════════════════════
# Provider Enum
# ════════════════════════════════════════════════════════════════════════════

class TTSProvider(Enum):
    GEMINI = "gemini"
    ELEVENLABS = "elevenlabs"
    EDGE = "edge_tts"
    GTTS = "gtts"


LIVE_TTS_PROVIDERS = frozenset(TTSProvider)

_FAILURE_COOLDOWN_SEC = 300  # 5 دقائق قبل إعادة تجربة مزوّد فاشل

# أصوات عربية افتراضية جيدة لكل مزوّد (fallback إن لم يُحدَّد صوت)
_DEFAULT_VOICE = {
    TTSProvider.GEMINI: "Kore",              # صوت Gemini TTS متعدد اللغات
    TTSProvider.ELEVENLABS: "Rachid",        # اسم صوت عربي شائع في ElevenLabs (قد يختلف حسب الحساب)
    TTSProvider.EDGE: "ar-SA-HamedNeural",   # صوت رجالي سعودي عصبي مجاني
    TTSProvider.GTTS: "ar",                  # gTTS يستخدم رمز لغة لا اسم صوت
}

_GEMINI_TTS_MODEL = "gemini-2.5-flash-preview-tts"


# ════════════════════════════════════════════════════════════════════════════
# Result Dataclass
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class TTSResult:
    audio_bytes: bytes
    provider: TTSProvider
    format: str = "mp3"          # mp3 | wav
    voice: str = ""
    latency_ms: float = 0.0
    duration_est_sec: float = 0.0
    error: Optional[str] = None
    tried: List[str] = field(default_factory=list)
    # توقيت كل كلمة فعلياً بالصوت المولَّد — (النص, البداية بالثانية,
    # المدة بالثانية). يُملأ فقط عبر Edge TTS (الوحيد الذي يُصدر
    # WordBoundary events حقيقية)؛ يبقى [] لبقية المزوّدين، وVideoEngine
    # يتراجع تلقائياً لتقدير تناسبي عند غيابه (راجع ai/video_engine.py).
    word_timings: List[tuple] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return bool(self.audio_bytes) and self.error is None


# ════════════════════════════════════════════════════════════════════════════
# تقدير مدة الصوت من طول النص (عند عدم توفر المدة الحقيقية من المزوّد)
# ════════════════════════════════════════════════════════════════════════════

def _estimate_duration_sec(text: str, words_per_min: int = 150) -> float:
    words = max(1, len(text.split()))
    return round((words / words_per_min) * 60, 1)


# ════════════════════════════════════════════════════════════════════════════
# مزوّد 1: Google Gemini TTS
# ════════════════════════════════════════════════════════════════════════════

def _synthesize_gemini(text: str, voice: str, api_key: str) -> TTSResult:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_TTS_MODEL}:generateContent?key={api_key}"
    )
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice or _DEFAULT_VOICE[TTSProvider.GEMINI]}
                }
            },
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    b64_audio = (
        data["candidates"][0]["content"]["parts"][0]["inlineData"]["data"]
    )
    audio_bytes = base64.b64decode(b64_audio)
    # Gemini TTS يرجّع PCM خام (24kHz, 16-bit, mono) داخل WAV غير مُغلّف أحياناً؛
    # هنا نضيف رأس WAV بسيط إن كانت البيانات PCM خام.
    audio_bytes = _pcm_to_wav(audio_bytes) if not audio_bytes.startswith(b"RIFF") else audio_bytes
    return TTSResult(
        audio_bytes=audio_bytes,
        provider=TTSProvider.GEMINI,
        format="wav",
        voice=voice or _DEFAULT_VOICE[TTSProvider.GEMINI],
    )


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, bits: int = 16) -> bytes:
    """يغلّف بيانات PCM خام برأس WAV صالح (بدون أي مكتبات خارجية)."""
    import struct

    byte_rate = sample_rate * channels * bits // 8
    block_align = channels * bits // 8
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + len(pcm_bytes), b"WAVE",
        b"fmt ", 16, 1, channels, sample_rate, byte_rate, block_align, bits,
        b"data", len(pcm_bytes),
    )
    return header + pcm_bytes


# ════════════════════════════════════════════════════════════════════════════
# مزوّد 2: ElevenLabs
# ════════════════════════════════════════════════════════════════════════════

def _synthesize_elevenlabs(text: str, voice_id: str, api_key: str) -> TTSResult:
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        audio_bytes = resp.read()
    return TTSResult(audio_bytes=audio_bytes, provider=TTSProvider.ELEVENLABS, format="mp3", voice=voice_id)


# ════════════════════════════════════════════════════════════════════════════
# مزوّد 3: Microsoft Edge TTS (مجاني، بدون مفتاح) — يتطلب حزمة edge-tts
# ════════════════════════════════════════════════════════════════════════════

def _synthesize_edge(text: str, voice: str) -> TTSResult:
    try:
        import edge_tts  # type: ignore
    except ImportError as exc:
        raise RuntimeError("حزمة edge-tts غير مثبّتة. أضِف 'edge-tts>=6.1.0' لـ requirements.txt") from exc

    voice = voice or _DEFAULT_VOICE[TTSProvider.EDGE]

    async def _run():
        # boundary="WordBoundary" ضروري صريحاً — الافتراضي بمكتبة edge-tts
        # هو "SentenceBoundary" (لا يُصدر توقيت كل كلمة على حدة، فتبقى
        # word_timings فارغة دائماً بصمت دون هذا التصريح).
        communicate = edge_tts.Communicate(text, voice, boundary="WordBoundary")
        chunks = bytearray()
        # edge-tts يُصدر أحداث "WordBoundary" فعلية (توقيت حقيقي من محرك
        # النطق نفسه، لا تقدير) بجانب أحداث الصوت — نجمعها هنا لاستخدامها
        # لاحقاً في مزامنة الترجمات المتحركة بدقة (كاريوكي كلمة-بكلمة)
        # بمحرك الفيديو، بدل التقدير التناسبي القديم القائم على طول النص فقط.
        timings: list = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.extend(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # offset/duration بوحدة 100-نانوثانية (100ns ticks) حسب SSML/Edge
                start_sec = chunk["offset"] / 1e7
                dur_sec = chunk["duration"] / 1e7
                timings.append((chunk["text"], start_sec, dur_sec))
        return bytes(chunks), timings

    audio_bytes, word_timings = asyncio.run(_run())
    return TTSResult(
        audio_bytes=audio_bytes, provider=TTSProvider.EDGE, format="mp3", voice=voice,
        word_timings=word_timings,
    )


# ════════════════════════════════════════════════════════════════════════════
# مزوّد 4: gTTS (احتياطي أخير، بدون مفتاح) — يتطلب حزمة gTTS
# ════════════════════════════════════════════════════════════════════════════

def _synthesize_gtts(text: str, lang: str) -> TTSResult:
    try:
        from gtts import gTTS  # type: ignore
        import io
    except ImportError as exc:
        raise RuntimeError("حزمة gTTS غير مثبّتة. أضِف 'gTTS>=2.5.0' لـ requirements.txt") from exc

    buf = io.BytesIO()
    gTTS(text=text, lang=lang or _DEFAULT_VOICE[TTSProvider.GTTS]).write_to_fp(buf)
    return TTSResult(audio_bytes=buf.getvalue(), provider=TTSProvider.GTTS, format="mp3", voice=lang)


# ════════════════════════════════════════════════════════════════════════════
# TTSEngine — الواجهة الرئيسية (سلسلة تراجع تلقائي)
# ════════════════════════════════════════════════════════════════════════════

class TTSEngine:
    """محرك تحويل نص-لصوت بسلسلة مزوّدين وتراجع تلقائي عند الفشل.

    ترتيب المحاولة: Gemini (إن وُجد مفتاح) → ElevenLabs (إن وُجد مفتاح)
    → Edge TTS (مجاني) → gTTS (مجاني، احتياطي أخير).
    """

    def __init__(self) -> None:
        self._failed_until: dict[TTSProvider, float] = {}

    # ── بناء سلسلة المزوّدين المتاحين حسب مفاتيح البيئة ──────────────────
    def _build_chain(self) -> List[TTSProvider]:
        chain: List[TTSProvider] = []
        if os.getenv("GOOGLE_API_KEY", "").strip():
            chain.append(TTSProvider.GEMINI)
        if os.getenv("ELEVENLABS_API_KEY", "").strip():
            chain.append(TTSProvider.ELEVENLABS)
        chain.append(TTSProvider.EDGE)   # دائماً متاح (بدون مفتاح)
        chain.append(TTSProvider.GTTS)   # احتياطي أخير
        return chain

    def _is_cooling_down(self, provider: TTSProvider) -> bool:
        until = self._failed_until.get(provider)
        return bool(until and time.time() < until)

    def _mark_failed(self, provider: TTSProvider) -> None:
        self._failed_until[provider] = time.time() + _FAILURE_COOLDOWN_SEC

    # ── الواجهة العامة ────────────────────────────────────────────────
    def synthesize(self, text: str, voice: str = "") -> TTSResult:
        """يحوّل النص لصوت، مجرّباً كل مزوّد بالترتيب حتى ينجح أحدها."""
        text = (text or "").strip()
        if not text:
            return TTSResult(audio_bytes=b"", provider=TTSProvider.EDGE, error="نص فارغ")

        tried: List[str] = []
        for provider in self._build_chain():
            if self._is_cooling_down(provider):
                tried.append(f"{provider.value}(cooldown)")
                continue
            start = time.time()
            try:
                if provider == TTSProvider.GEMINI:
                    result = _synthesize_gemini(text, voice, os.getenv("GOOGLE_API_KEY", ""))
                elif provider == TTSProvider.ELEVENLABS:
                    result = _synthesize_elevenlabs(
                        text, voice or _DEFAULT_VOICE[TTSProvider.ELEVENLABS],
                        os.getenv("ELEVENLABS_API_KEY", ""),
                    )
                elif provider == TTSProvider.EDGE:
                    result = _synthesize_edge(text, voice)
                else:  # GTTS
                    result = _synthesize_gtts(text, voice)

                result.latency_ms = round((time.time() - start) * 1000, 1)
                result.duration_est_sec = _estimate_duration_sec(text)
                result.tried = tried + [provider.value]
                logger.info("TTS نجح عبر %s (%.0fms)", provider.value, result.latency_ms)
                return result

            except (urllib.error.URLError, RuntimeError, Exception) as exc:  # noqa: BLE001
                logger.warning("TTS فشل عبر %s: %s", provider.value, exc)
                tried.append(f"{provider.value}(fail)")
                self._mark_failed(provider)
                continue

        return TTSResult(
            audio_bytes=b"", provider=TTSProvider.GTTS,
            error="فشلت كل مزوّدات TTS المتاحة", tried=tried,
        )

    # ── سرد سيناريو كامل (مقاطع متعددة) ──────────────────────────────
    def synthesize_segments(self, narrations: List[str], voice: str = "") -> List[TTSResult]:
        """يحوّل قائمة نصوص (مقاطع/لقطات) إلى قائمة نتائج صوتية منفصلة،
        جاهزة للتركيب لاحقاً في فيديو واحد (كل مقطع = لقطة)."""
        return [self.synthesize(t, voice=voice) for t in narrations]
