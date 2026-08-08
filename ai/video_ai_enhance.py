#!/usr/bin/env python3
"""
Video AI Enhance — أدوات تحسين فيديو «ذكية» لـ NSM
=================================================
تحسين محلي عالي الجودة عبر ffmpeg (بدون مفاتيح):
  • وضوح (clarity)
  • سينمائي (cinematic grade)
  • تنعيم ضوضاء (denoise)
  • إضاءة منخفضة (low_light)
  • رفع دقة ذكي إلى HD/9:16 (upscale)
  • استقرار خفيف (stabilize)

اختياري: محاولة تحسين عبر مساحة Hugging Face مجانية إن توفّرت الشبكة
و`gradio_client` — مع تراجع تلقائي للمحلي عند الفشل.

المخرجات: artifacts/video_edits/
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger("VideoAIEnhance")

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "artifacts" / "video_edits"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PathLike = Union[str, Path]


class VideoAIEnhanceError(RuntimeError):
    pass


PRESETS: Dict[str, Dict[str, str]] = {
    "clarity": {
        "label": "وضوح حاد",
        "desc": "تقليل ضوضاء خفيف + حدة + تباين",
        "vf": (
            "hqdn3d=1.5:1.5:3:3,"
            "unsharp=5:5:0.8:5:5:0.0,"
            "eq=contrast=1.08:saturation=1.06:brightness=0.01"
        ),
    },
    "cinematic": {
        "label": "درجة سينمائية",
        "desc": "تباين فيلمي + تشبع غني + حدة ناعمة",
        "vf": (
            "eq=contrast=1.12:saturation=1.18:brightness=0.02:gamma=1.05,"
            "unsharp=5:5:0.55:5:5:0.0,"
            "colorbalance=rs=0.03:gs=-0.02:bs=-0.04:rm=0.02:bm=-0.02"
        ),
    },
    "denoise": {
        "label": "إزالة ضوضاء",
        "desc": "تنظيف قوي مع الحفاظ على الحواف",
        "vf": (
            "hqdn3d=4:3:6:4.5,"
            "unsharp=3:3:0.4:3:3:0.0"
        ),
    },
    "low_light": {
        "label": "إضاءة منخفضة",
        "desc": "رفع السطوع/الظلال بحذر",
        "vf": (
            "eq=brightness=0.06:contrast=1.15:gamma=1.12:saturation=1.08,"
            "hqdn3d=2:1.5:3:2.5,"
            "unsharp=5:5:0.5:5:5:0.0"
        ),
    },
    "upscale_hd": {
        "label": "رفع إلى Full HD",
        "desc": "تنظيف → Lanczos 1080p → حدة",
        "vf": (
            "hqdn3d=1.2:1.2:2.4:2.4,"
            "scale=1920:1080:flags=lanczos+accurate_rnd+full_chroma_int:"
            "force_original_aspect_ratio=decrease,"
            "pad=1920:1080:(ow-iw)/2:(oh-ih)/2:color=black,"
            "unsharp=5:5:1.0:5:5:0.0,"
            "eq=contrast=1.06:saturation=1.06"
        ),
    },
    "upscale_2x": {
        "label": "تكبير ×2",
        "desc": "مضاعفة الأبعاد مع حدة متدرجة",
        "vf": (
            "hqdn3d=1.0:1.0:2:2,"
            "scale=iw*2:ih*2:flags=lanczos+accurate_rnd+full_chroma_int,"
            "unsharp=5:5:0.9:5:5:0.0,"
            "eq=contrast=1.05:saturation=1.04"
        ),
    },
    "upscale_qhd": {
        "label": "رفع إلى 1440p",
        "desc": "2K تقريبي مع Lanczos",
        "vf": (
            "hqdn3d=1.2:1.2:2.4:2.4,"
            "scale=2560:1440:flags=lanczos+accurate_rnd+full_chroma_int:"
            "force_original_aspect_ratio=decrease,"
            "pad=2560:1440:(ow-iw)/2:(oh-ih)/2:color=black,"
            "unsharp=5:5:1.0:5:5:0.0"
        ),
    },
    "upscale_shorts": {
        "label": "رفع لشورتس 9:16",
        "desc": "1080×1920 cover + حدة وألوان",
        "vf": (
            "hqdn3d=1.2:1.2:2.5:2.5,"
            "scale=1080:1920:flags=lanczos+accurate_rnd+full_chroma_int:"
            "force_original_aspect_ratio=increase,"
            "crop=1080:1920,"
            "unsharp=5:5:1.0:5:5:0.0,"
            "eq=contrast=1.10:saturation=1.12:brightness=0.02"
        ),
    },
    "stabilize": {
        "label": "ثبات خفيف",
        "desc": "deshake + حدة خفيفة",
        "vf": (
            "deshake,"
            "unsharp=5:5:0.5:5:5:0.0"
        ),
    },
}

DEFAULT_PRESET = "clarity"


def _ffmpeg() -> str:
    bin_ = shutil.which("ffmpeg")
    if not bin_:
        raise VideoAIEnhanceError("ffmpeg غير متوفر — مطلوب لتحسين الفيديو")
    return bin_


def _ensure(path: PathLike) -> Path:
    p = Path(path)
    if not p.is_file():
        raise VideoAIEnhanceError(f"الملف غير موجود: {p}")
    rp = p.resolve()
    allowed = [str(ROOT.resolve()), str(Path("/tmp").resolve())]
    if not any(str(rp).startswith(a) for a in allowed):
        raise VideoAIEnhanceError("يُسمح فقط بملفات المشروع أو /tmp")
    return rp


def _out(prefix: str = "ai_enhance") -> Path:
    return OUT_DIR / f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}.mp4"


def list_presets() -> List[Dict[str, str]]:
    return [
        {"id": k, "label": v["label"], "desc": v["desc"]}
        for k, v in PRESETS.items()
    ]


def enhance_local(
    path: PathLike,
    preset: str = DEFAULT_PRESET,
    crf: int = 17,
) -> Path:
    """تحسين محلي عبر مرشحات ffmpeg عالية الجودة."""
    src = _ensure(path)
    preset = preset if preset in PRESETS else DEFAULT_PRESET
    vf = PRESETS[preset]["vf"]
    out = _out(prefix=f"ai_{preset}")
    crf = max(14, min(28, int(crf)))
    try:
        from ai.ffmpeg_quality import encode_args
        # CRF من المستخدم يطغى على المستوى الافتراضي
        tail = encode_args("high", extra_vf=vf)
        # استبدال crf في الذيل
        if "-crf" in tail:
            i = tail.index("-crf")
            tail[i + 1] = str(crf)
        cmd = [_ffmpeg(), "-y", "-i", str(src), *tail, str(out)]
    except Exception:
        cmd = [
            _ffmpeg(), "-y", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-c:a", "aac", "-b:a", "256k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out),
        ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired as e:
        raise VideoAIEnhanceError("انتهت مهلة التحسين") from e
    if p.returncode != 0 or not out.is_file():
        raise VideoAIEnhanceError(f"فشل التحسين: {(p.stderr or '')[-500:]}")
    return out


def enhance_chain(
    path: PathLike,
    presets: Optional[List[str]] = None,
    crf: int = 17,
) -> Path:
    """تطبيق سلسلة إعدادات بالترتيب (مثلاً denoise ثم clarity)."""
    presets = presets or [DEFAULT_PRESET]
    current = Path(path)
    for i, pr in enumerate(presets):
        current = enhance_local(current, preset=pr, crf=crf if i == len(presets) - 1 else 18)
    return current



UPSCALE_TARGETS = {
    "2x": "upscale_2x",
    "720p": None,  # dynamic
    "1080p": "upscale_hd",
    "1440p": "upscale_qhd",
    "shorts": "upscale_shorts",
}


def _vf_target(width: int, height: int, mode: str = "fit") -> str:
    """mode=fit يحافظ على النسبة مع pad، cover يقص للملء."""
    flags = "lanczos+accurate_rnd+full_chroma_int"
    pre = "hqdn3d=1.2:1.2:2.4:2.4,"
    post = ",unsharp=5:5:1.0:5:5:0.0,eq=contrast=1.06:saturation=1.06"
    if mode == "cover":
        return (
            f"{pre}scale={width}:{height}:flags={flags}:force_original_aspect_ratio=increase,"
            f"crop={width}:{height}{post}"
        )
    return (
        f"{pre}scale={width}:{height}:flags={flags}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black{post}"
    )


def upscale_video(
    path: PathLike,
    target: str = "1080p",
    crf: int = 16,
) -> Path:
    """
    رفع دقة الفيديو بجودة عالية.
    target: 2x | 720p | 1080p | 1440p | shorts | 4k
    """
    src = _ensure(path)
    target = (target or "1080p").lower().strip()
    out = _out(prefix=f"upscale_{target}")

    if target == "2x":
        vf = PRESETS["upscale_2x"]["vf"]
    elif target == "720p":
        vf = _vf_target(1280, 720, "fit")
    elif target in ("1080p", "hd", "fhd"):
        vf = PRESETS["upscale_hd"]["vf"]
    elif target in ("1440p", "qhd", "2k"):
        vf = PRESETS["upscale_qhd"]["vf"]
    elif target in ("shorts", "9:16", "vertical"):
        vf = PRESETS["upscale_shorts"]["vf"]
    elif target in ("4k", "2160p"):
        vf = _vf_target(3840, 2160, "fit")
    else:
        vf = PRESETS["upscale_hd"]["vf"]

    crf = max(14, min(23, int(crf)))
    try:
        from ai.ffmpeg_quality import encode_args
        tail = encode_args("archive" if crf <= 15 else "high", extra_vf=vf)
        if "-crf" in tail:
            i = tail.index("-crf")
            tail[i + 1] = str(crf)
        cmd = [_ffmpeg(), "-y", "-i", str(src), *tail, str(out)]
    except Exception:
        cmd = [
            _ffmpeg(), "-y", "-i", str(src),
            "-vf", vf,
            "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
            "-profile:v", "high", "-pix_fmt", "yuv420p",
            "-tune", "film",
            "-c:a", "aac", "-b:a", "256k",
            "-movflags", "+faststart",
            str(out),
        ]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired as e:
        raise VideoAIEnhanceError("انتهت مهلة رفع الدقة") from e
    if p.returncode != 0 or not out.is_file():
        raise VideoAIEnhanceError(f"فشل الـ upscale: {(p.stderr or '')[-500:]}")
    return out


def try_hf_upscale(path: PathLike, timeout: int = 90) -> Optional[Path]:
    """
    محاولة اختيارية عبر مساحة HF مجانية لتحسين إطار/فيديو.
    تُرجع None عند الفشل (الشبكة، الطابور، المكتبة).
    حالياً: استخراج إطار أوسط → إن وُجدت مساحة غير مستقرة نتخطى.
    """
    # المسارات المجانية العامة غير مستقرة لتحديث الفيديو الكامل؛
    # نبقي الواجهة جاهزة ونُفضّل المحلي الموثوق.
    logger.info("HF upscale skipped — local enhance is preferred for reliability")
    return None


def enhance_auto(
    path: PathLike,
    mode: str = "auto",
    prefer_hf: bool = False,
    crf: int = 17,
) -> Dict[str, Any]:
    """
    mode:
      auto | clarity | cinematic | denoise | low_light |
      upscale_hd | upscale_shorts | stabilize | pro (denoise+clarity)
    """
    src = _ensure(path)
    backend = "local"
    if prefer_hf:
        hf = try_hf_upscale(src)
        if hf is not None:
            return {"path": str(hf), "backend": "hf", "preset": mode}

    if mode == "auto":
        # كشف بسيط من الأبعاد
        try:
            from ai.video_editor import probe
            info = probe(src)
            w, h = int(info.get("width") or 0), int(info.get("height") or 0)
            if h > w and h < 1600:
                mode = "upscale_shorts"
            elif max(w, h) < 1000:
                mode = "upscale_hd"
            else:
                mode = "clarity"
        except Exception:
            mode = "clarity"

    if mode == "pro":
        out = enhance_chain(src, ["denoise", "clarity"], crf=crf)
        return {"path": str(out), "backend": backend, "preset": "pro"}

    if mode in ("2x", "720p", "1080p", "1440p", "4k", "shorts", "upscale"):
        target = "1080p" if mode == "upscale" else mode
        out = upscale_video(src, target=target, crf=crf)
        return {"path": str(out), "backend": backend, "preset": f"upscale:{target}"}

    out = enhance_local(src, preset=mode, crf=crf)
    return {"path": str(out), "backend": backend, "preset": mode}


def format_presets_help() -> str:
    lines = ["## ✨ إعدادات تحسين الفيديو بالذكاء (محلي)", ""]
    for k, v in PRESETS.items():
        lines.append(f"- **{v['label']}** (`{k}`): {v['desc']}")
    lines.append("- **احترافي** (`pro`): تنعيم ثم وضوح")
    lines.append("- **تلقائي** (`auto`): يختار حسب الأبعاد")
    lines.append("- **رفع دقة**: `2x` · `720p` · `1080p` · `1440p` · `4k` · `shorts`")
    return "\n".join(lines)
