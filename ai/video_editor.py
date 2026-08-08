#!/usr/bin/env python3
"""
Video Editor Toolkit — أدوات تعديل فيديو لـ NSM Shorts
=====================================================
عمليات آمنة ومنظّمة (بدون shell حر من المستخدم):

  • معلومات (probe)
  • قصّ (trim)
  • دمج (concat)
  • كتم / استخراج صوت
  • تحجيم عمودي 9:16 للشورتس
  • تسريع / تبطيء
  • ضغط خفيف
  • إطار مصغّر (thumbnail)

يعتمد على ffmpeg إن وُجد، وإلا moviepy عند الإمكان.
المخرجات تُحفظ تحت artifacts/video_edits/
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger("VideoEditor")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "video_edits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PathLike = Union[str, Path]


class VideoEditorError(RuntimeError):
    pass


def _which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _which_ffprobe() -> Optional[str]:
    return shutil.which("ffprobe")


def _run(cmd: List[str], timeout: int = 300) -> Tuple[int, str, str]:
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout or "", p.stderr or ""
    except subprocess.TimeoutExpired as e:
        raise VideoEditorError(f"انتهت مهلة العملية ({timeout}s)") from e
    except FileNotFoundError as e:
        raise VideoEditorError("ffmpeg/ffprobe غير متوفر في البيئة") from e


def _out_path(suffix: str = ".mp4", prefix: str = "edit") -> Path:
    name = f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}{suffix}"
    return OUT_DIR / name


def _ensure_file(path: PathLike) -> Path:
    p = Path(path)
    if not p.is_file():
        raise VideoEditorError(f"الملف غير موجود: {p}")
    # أمان نسبي: رفض مسارات خارج المشروع أو /tmp المسموح
    try:
        rp = p.resolve()
    except Exception as e:
        raise VideoEditorError(f"مسار غير صالح: {e}") from e
    allowed_roots = [ROOT.resolve(), Path(tempfile.gettempdir()).resolve()]
    if not any(str(rp).startswith(str(a)) for a in allowed_roots):
        # السماح أيضاً إن كان تحت /tmp أو uploads
        if not str(rp).startswith("/tmp"):
            raise VideoEditorError("لأسباب أمنية يُسمح فقط بملفات داخل المشروع أو /tmp")
    return rp


def probe(path: PathLike) -> Dict[str, Any]:
    """معلومات الفيديو عبر ffprobe."""
    p = _ensure_file(path)
    ffprobe = _which_ffprobe()
    if not ffprobe:
        # fallback خفيف عبر moviepy
        try:
            from moviepy import VideoFileClip
            clip = VideoFileClip(str(p))
            info = {
                "path": str(p),
                "duration": float(clip.duration or 0),
                "width": int(clip.w or 0),
                "height": int(clip.h or 0),
                "fps": float(clip.fps or 0),
                "has_audio": clip.audio is not None,
                "backend": "moviepy",
            }
            clip.close()
            return info
        except Exception as e:
            raise VideoEditorError(f"تعذّر فحص الملف: {e}") from e

    code, out, err = _run([
        ffprobe, "-v", "error",
        "-show_format", "-show_streams",
        "-of", "json", str(p),
    ])
    if code != 0:
        raise VideoEditorError(f"ffprobe فشل: {err[-400:]}")
    data = json.loads(out or "{}")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    v = next((s for s in streams if s.get("codec_type") == "video"), {})
    a = next((s for s in streams if s.get("codec_type") == "audio"), {})
    return {
        "path": str(p),
        "duration": float(fmt.get("duration") or 0),
        "size_bytes": int(fmt.get("size") or 0),
        "format": fmt.get("format_name"),
        "width": int(v.get("width") or 0),
        "height": int(v.get("height") or 0),
        "fps": _parse_fps(v.get("r_frame_rate")),
        "video_codec": v.get("codec_name"),
        "audio_codec": a.get("codec_name"),
        "has_audio": bool(a),
        "backend": "ffprobe",
    }


def _parse_fps(rate: Optional[str]) -> float:
    if not rate or rate == "0/0":
        return 0.0
    try:
        if "/" in rate:
            a, b = rate.split("/", 1)
            return float(a) / max(1e-9, float(b))
        return float(rate)
    except Exception:
        return 0.0


def trim(
    path: PathLike,
    start: float = 0.0,
    end: Optional[float] = None,
    reencode: bool = True,
) -> Path:
    """قص مقطع من start إلى end (ثانية)."""
    p = _ensure_file(path)
    start = max(0.0, float(start))
    out = _out_path(prefix="trim")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        return _trim_moviepy(p, start, end, out)

    cmd = [ffmpeg, "-y", "-ss", str(start), "-i", str(p)]
    if end is not None and end > start:
        cmd += ["-to", str(float(end))]
    if reencode:
        cmd += [
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        ]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out))
    code, _, err = _run(cmd)
    if code != 0 or not out.is_file():
        raise VideoEditorError(f"فشل القص: {err[-500:]}")
    return out


def _trim_moviepy(p: Path, start: float, end: Optional[float], out: Path) -> Path:
    try:
        from moviepy import VideoFileClip
    except ImportError as e:
        raise VideoEditorError("moviepy غير متاح وffmpeg غير موجود") from e
    clip = VideoFileClip(str(p))
    end_t = float(end) if end is not None else float(clip.duration or 0)
    sub = clip.subclipped(start, min(end_t, float(clip.duration or end_t)))
    sub.write_videofile(
        str(out), codec="libx264", audio_codec="aac", logger=None,
    )
    sub.close()
    clip.close()
    return out


def concat(paths: Sequence[PathLike]) -> Path:
    """دمج عدة فيديوهات بالترتيب."""
    files = [_ensure_file(p) for p in paths]
    if len(files) < 2:
        raise VideoEditorError("يلزم ملفان على الأقل للدمج")
    out = _out_path(prefix="concat")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        return _concat_moviepy(files, out)

    # filter concat أدق من demuxer عند اختلاف الترميز
    inputs: List[str] = []
    for f in files:
        inputs += ["-i", str(f)]
    n = len(files)
    filt = "".join(f"[{i}:v:0][{i}:a:0]" for i in range(n))
    filt += f"concat=n={n}:v=1:a=1[v][a]"
    cmd = [ffmpeg, "-y", *inputs, "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
           "-c:v", "libx264", "-preset", "fast", "-crf", "20",
           "-c:a", "aac", "-b:a", "192k",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
    code, _, err = _run(cmd, timeout=600)
    if code != 0 or not out.is_file():
        # محاولة بدون صوت
        filt_v = "".join(f"[{i}:v:0]" for i in range(n)) + f"concat=n={n}:v=1:a=0[v]"
        cmd2 = [ffmpeg, "-y", *inputs, "-filter_complex", filt_v, "-map", "[v]",
                "-c:v", "libx264", "-preset", "fast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)]
        code2, _, err2 = _run(cmd2, timeout=600)
        if code2 != 0 or not out.is_file():
            raise VideoEditorError(f"فشل الدمج: {err[-300:]} | {err2[-300:]}")
    return out


def _concat_moviepy(files: List[Path], out: Path) -> Path:
    try:
        from moviepy import VideoFileClip, concatenate_videoclips
    except ImportError as e:
        raise VideoEditorError("moviepy غير متاح") from e
    clips = [VideoFileClip(str(f)) for f in files]
    final = concatenate_videoclips(clips, method="compose")
    final.write_videofile(str(out), codec="libx264", audio_codec="aac", logger=None)
    final.close()
    for c in clips:
        c.close()
    return out


def mute(path: PathLike) -> Path:
    """إزالة الصوت."""
    p = _ensure_file(path)
    out = _out_path(prefix="mute")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise VideoEditorError("ffmpeg مطلوب لكتم الصوت")
    code, _, err = _run([
        ffmpeg, "-y", "-i", str(p), "-c", "copy", "-an", str(out),
    ])
    if code != 0 or not out.is_file():
        raise VideoEditorError(f"فشل كتم الصوت: {err[-400:]}")
    return out


def extract_audio(path: PathLike, fmt: str = "mp3") -> Path:
    """استخراج المسار الصوتي."""
    p = _ensure_file(path)
    fmt = fmt if fmt in ("mp3", "wav", "aac", "m4a") else "mp3"
    out = _out_path(suffix=f".{fmt}", prefix="audio")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise VideoEditorError("ffmpeg مطلوب لاستخراج الصوت")
    cmd = [ffmpeg, "-y", "-i", str(p), "-vn"]
    if fmt == "mp3":
        cmd += ["-codec:a", "libmp3lame", "-q:a", "2"]
    elif fmt == "wav":
        cmd += ["-codec:a", "pcm_s16le"]
    else:
        cmd += ["-codec:a", "aac", "-b:a", "192k"]
    cmd.append(str(out))
    code, _, err = _run(cmd)
    if code != 0 or not out.is_file():
        raise VideoEditorError(f"فشل استخراج الصوت: {err[-400:]}")
    return out


def to_shorts_vertical(path: PathLike, width: int = 1080, height: int = 1920) -> Path:
    """تحجيم/قص مركزي إلى 9:16 للشورتس."""
    p = _ensure_file(path)
    out = _out_path(prefix="shorts9x16")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        return _vertical_moviepy(p, width, height, out)
    # scale to cover then crop center
    vf = (
        f"scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height}"
    )
    code, _, err = _run([
        ffmpeg, "-y", "-i", str(p),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ])
    if code != 0 or not out.is_file():
        raise VideoEditorError(f"فشل التحويل العمودي: {err[-400:]}")
    return out


def _vertical_moviepy(p: Path, w: int, h: int, out: Path) -> Path:
    try:
        from moviepy import VideoFileClip
        from moviepy.video.fx import Resize, Crop
    except ImportError as e:
        raise VideoEditorError("moviepy غير متاح") from e
    clip = VideoFileClip(str(p))
    # cover scale
    scale = max(w / clip.w, h / clip.h)
    resized = clip.resized(scale)
    x1 = max(0, (resized.w - w) // 2)
    y1 = max(0, (resized.h - h) // 2)
    cropped = resized.cropped(x1=x1, y1=y1, width=w, height=h)
    cropped.write_videofile(str(out), codec="libx264", audio_codec="aac", logger=None)
    cropped.close()
    clip.close()
    return out


def change_speed(path: PathLike, factor: float = 1.25) -> Path:
    """تسريع/تبطيء (0.5–2.0)."""
    p = _ensure_file(path)
    factor = max(0.5, min(2.0, float(factor)))
    out = _out_path(prefix=f"speed{factor:.2f}".replace(".", "_"))
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise VideoEditorError("ffmpeg مطلوب لتغيير السرعة")
    # atempo accepts 0.5-2.0
    vf = f"setpts={1.0/factor}*PTS"
    af = f"atempo={factor}"
    code, _, err = _run([
        ffmpeg, "-y", "-i", str(p),
        "-filter_complex", f"[0:v]{vf}[v];[0:a]{af}[a]",
        "-map", "[v]", "-map", "[a]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ])
    if code != 0 or not out.is_file():
        # video only
        code2, _, err2 = _run([
            ffmpeg, "-y", "-i", str(p), "-vf", vf, "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(out),
        ])
        if code2 != 0 or not out.is_file():
            raise VideoEditorError(f"فشل تغيير السرعة: {err[-300:]} | {err2[-300:]}")
    return out


def compress(path: PathLike, crf: int = 28) -> Path:
    """ضغط خفيف للمشاركة."""
    p = _ensure_file(path)
    crf = max(18, min(32, int(crf)))
    out = _out_path(prefix="compress")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise VideoEditorError("ffmpeg مطلوب للضغط")
    code, _, err = _run([
        ffmpeg, "-y", "-i", str(p),
        "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
        "-c:a", "aac", "-b:a", "128k",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(out),
    ])
    if code != 0 or not out.is_file():
        raise VideoEditorError(f"فشل الضغط: {err[-400:]}")
    return out


def thumbnail(path: PathLike, at_seconds: float = 1.0) -> Path:
    """استخراج إطار كصورة JPEG."""
    p = _ensure_file(path)
    out = _out_path(suffix=".jpg", prefix="thumb")
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise VideoEditorError("ffmpeg مطلوب للصورة المصغّرة")
    code, _, err = _run([
        ffmpeg, "-y", "-ss", str(max(0.0, at_seconds)),
        "-i", str(p), "-frames:v", "1", "-q:v", "2", str(out),
    ])
    if code != 0 or not out.is_file():
        raise VideoEditorError(f"فشل استخراج الإطار: {err[-400:]}")
    return out


def available_tools() -> Dict[str, bool]:
    return {
        "ffmpeg": bool(_which_ffmpeg()),
        "ffprobe": bool(_which_ffprobe()),
        "moviepy": _has_moviepy(),
        "out_dir": str(OUT_DIR.relative_to(ROOT)),
    }


def _has_moviepy() -> bool:
    try:
        import moviepy  # noqa: F401
        return True
    except Exception:
        return False


def format_probe_report(info: Dict[str, Any]) -> str:
    lines = [
        "## 🎞️ معلومات الفيديو",
        f"- المسار: `{info.get('path')}`",
        f"- المدة: **{info.get('duration', 0):.2f}** ث",
        f"- الأبعاد: **{info.get('width')}×{info.get('height')}**",
        f"- FPS: {info.get('fps')}",
        f"- فيديو: {info.get('video_codec', '—')} · صوت: {info.get('audio_codec', '—')}",
        f"- حجم: {int(info.get('size_bytes') or 0) / 1024:.1f} KB" if info.get("size_bytes") else "",
        f"- الخلفية: {info.get('backend')}",
    ]
    return "\n".join([ln for ln in lines if ln])
