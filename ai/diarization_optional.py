"""
فصل المتحدثين (Speaker Diarization) — غلاف اختياري لـ NSM.

يعتمد على pyannote.audio عند التوفر؛ يفشل بهدوء على Streamlit Cloud
أو أي بيئة بلا GPU/نموذج.

المتطلبات الاختيارية:
  pip install pyannote.audio torch torchaudio
  # ورمز Hugging Face للنماذج gated:
  export HF_TOKEN=hf_...
  # أو HUGGINGFACE_TOKEN / PYANNOTE_TOKEN

الاستخدام:
    from ai.diarization_optional import diarize_file, diarize_available

    if diarize_available():
        result = diarize_file("meeting.wav", min_speakers=1, max_speakers=4)
        for seg in result.segments:
            print(seg.speaker, seg.start, seg.end)
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("diarization_optional")

# نماذج شائعة (gated على HF — تحتاج token)
_DEFAULT_PIPELINE = os.environ.get(
    "NSM_DIARIZATION_PIPELINE",
    "pyannote/speaker-diarization-3.1",
)
_ALT_PIPELINE = "pyannote/speaker-diarization-community-1"


@dataclass
class DiarizationSegment:
    speaker: str
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, float(self.end) - float(self.start))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker,
            "start": round(float(self.start), 3),
            "end": round(float(self.end), 3),
            "duration": round(self.duration, 3),
        }


@dataclass
class DiarizationResult:
    ok: bool
    segments: List[DiarizationSegment] = field(default_factory=list)
    num_speakers: int = 0
    pipeline: str = ""
    error: Optional[str] = None
    available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "available": self.available,
            "num_speakers": self.num_speakers,
            "pipeline": self.pipeline,
            "error": self.error,
            "segments": [s.to_dict() for s in self.segments],
        }

    def speakers(self) -> List[str]:
        seen = []
        for s in self.segments:
            if s.speaker not in seen:
                seen.append(s.speaker)
        return seen


def _hf_token() -> Optional[str]:
    for key in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN", "PYANNOTE_TOKEN"):
        v = os.environ.get(key, "").strip()
        if v:
            return v
    return None


def diarize_available() -> bool:
    """هل يمكن استيراد pyannote.audio؟ (لا يتحقق من تحميل الأوزان)."""
    try:
        import pyannote.audio  # noqa: F401
        return True
    except Exception:
        return False


def _load_pipeline(model_id: Optional[str] = None):
    from pyannote.audio import Pipeline
    import torch

    mid = model_id or _DEFAULT_PIPELINE
    token = _hf_token()
    kwargs: Dict[str, Any] = {}
    if token:
        # إصدارات مختلفة من pyannote/huggingface_hub
        kwargs["token"] = token
    try:
        pipeline = Pipeline.from_pretrained(mid, **kwargs)
    except TypeError:
        # واجهات أقدم تستخدم use_auth_token
        kwargs.pop("token", None)
        if token:
            kwargs["use_auth_token"] = token
        pipeline = Pipeline.from_pretrained(mid, **kwargs)

    if torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda"))
        except Exception:
            pass
    return pipeline, mid


def diarize_file(
    audio_path: str,
    *,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    pipeline_id: Optional[str] = None,
) -> DiarizationResult:
    """
    فصل متحدثين من ملف صوتي (wav/mp3 حسب دعم torchaudio/ffmpeg).
    """
    if not diarize_available():
        return DiarizationResult(
            ok=False,
            available=False,
            error=(
                "pyannote.audio غير مثبّت. "
                "ثبّته محلياً: pip install pyannote.audio torch torchaudio "
                "مع HF_TOKEN لنماذج Hugging Face gated."
            ),
        )

    if not audio_path or not os.path.exists(audio_path):
        return DiarizationResult(
            ok=False,
            available=True,
            error=f"الملف غير موجود: {audio_path}",
        )

    try:
        pipeline, mid = _load_pipeline(pipeline_id)
    except Exception as e:
        # محاولة النموذج البديل
        try:
            if not pipeline_id:
                pipeline, mid = _load_pipeline(_ALT_PIPELINE)
            else:
                raise e
        except Exception as e2:
            return DiarizationResult(
                ok=False,
                available=True,
                error=(
                    f"تعذّر تحميل خط أنابيب pyannote: {e2}. "
                    "تأكد من HF_TOKEN وموافقة شروط النموذج على Hugging Face."
                ),
            )

    params: Dict[str, Any] = {}
    if num_speakers is not None:
        params["num_speakers"] = int(num_speakers)
    else:
        if min_speakers is not None:
            params["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            params["max_speakers"] = int(max_speakers)

    try:
        if params:
            annotation = pipeline(audio_path, **params)
        else:
            annotation = pipeline(audio_path)
    except Exception as e:
        return DiarizationResult(
            ok=False,
            available=True,
            pipeline=mid,
            error=f"فشل تشغيل diarization: {e}",
        )

    segments: List[DiarizationSegment] = []
    try:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            segments.append(
                DiarizationSegment(
                    speaker=str(speaker),
                    start=float(turn.start),
                    end=float(turn.end),
                )
            )
    except Exception as e:
        return DiarizationResult(
            ok=False,
            available=True,
            pipeline=mid,
            error=f"فشل قراءة النتائج: {e}",
        )

    speakers = {s.speaker for s in segments}
    return DiarizationResult(
        ok=True,
        available=True,
        segments=segments,
        num_speakers=len(speakers),
        pipeline=mid,
        error=None,
    )


def diarize_bytes(
    audio_bytes: bytes,
    *,
    suffix: str = ".wav",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    num_speakers: Optional[int] = None,
    pipeline_id: Optional[str] = None,
) -> DiarizationResult:
    """يحفظ مؤقتاً ثم يستدعي diarize_file."""
    import tempfile

    if not audio_bytes:
        return DiarizationResult(ok=False, available=diarize_available(), error="لا بيانات صوتية")

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        with open(tmp_path, "wb") as f:
            f.write(audio_bytes)
        return diarize_file(
            tmp_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            num_speakers=num_speakers,
            pipeline_id=pipeline_id,
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def merge_transcript_with_diarization(
    transcript_segments: List[Dict[str, Any]],
    diarization: DiarizationResult,
) -> List[Dict[str, Any]]:
    """
    يدمج مقاطع تفريغ (start/end/text) مع تسميات المتحدثين.
    كل عنصر في transcript_segments: {"start": float, "end": float, "text": str}
    """
    if not diarization.ok or not diarization.segments:
        return [
            {**seg, "speaker": seg.get("speaker") or "SPEAKER_00"}
            for seg in transcript_segments
        ]

    out = []
    for seg in transcript_segments:
        mid = (float(seg.get("start", 0)) + float(seg.get("end", 0))) / 2.0
        speaker = "SPEAKER_00"
        best_overlap = -1.0
        for d in diarization.segments:
            overlap = min(float(seg.get("end", 0)), d.end) - max(float(seg.get("start", 0)), d.start)
            if overlap > best_overlap:
                best_overlap = overlap
                speaker = d.speaker
            # إن كان منتصف المقطع داخل دور المتحدث
            if d.start <= mid <= d.end:
                speaker = d.speaker
                break
        out.append({**seg, "speaker": speaker})
    return out


def format_diarization_text(diarization: DiarizationResult) -> str:
    """نص عربي بسيط لعرض النتائج."""
    if not diarization.ok:
        return diarization.error or "تعذّر فصل المتحدثين."
    if not diarization.segments:
        return "لم يُكتشف كلام متعدد المتحدثين."
    lines = [f"عدد المتحدثين: {diarization.num_speakers}", ""]
    for s in diarization.segments:
        lines.append(f"[{s.start:7.1f}s – {s.end:7.1f}s] {s.speaker}")
    return "\n".join(lines)
