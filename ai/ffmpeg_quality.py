#!/usr/bin/env python3
"""
FFmpeg Quality Presets — إعدادات ترميز موحّدة لجودة أعلى في NSM
===============================================================
تُستخدم من video_engine / video_editor / video_ai_enhance لتجنّب تكرار
إعدادات ضعيفة (preset=fast + crf=20) في كل مسار.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

# مستويات جودة: أرشفة > عالية > متوازنة > سريعة
LEVELS: Dict[str, Dict] = {
    "archive": {
        "preset": "slow",
        "crf": 14,
        "audio_bitrate": "320k",
        "x264_params": "aq-mode=3:ref=5:bframes=3:me=umh:subme=9",
    },
    "high": {
        "preset": "slow",
        "crf": 16,
        "audio_bitrate": "256k",
        "x264_params": "aq-mode=3:ref=4:bframes=3:me=umh:subme=8",
    },
    "balanced": {
        "preset": "medium",
        "crf": 18,
        "audio_bitrate": "192k",
        "x264_params": "aq-mode=3:ref=3:bframes=2",
    },
    "fast": {
        "preset": "veryfast",
        "crf": 22,
        "audio_bitrate": "160k",
        "x264_params": "aq-mode=2:ref=2",
    },
}

# مرشح تحسين بصري خفيف يُلحق قبل الترميز عند الطلب
QUALITY_VF = {
    "none": "",
    "soft": "hqdn3d=1.0:1.0:2:2,unsharp=5:5:0.5:5:5:0.0,eq=contrast=1.04:saturation=1.04",
    "strong": (
        "hqdn3d=1.5:1.5:3:3,"
        "unsharp=5:5:0.75:5:5:0.0,"
        "eq=contrast=1.08:saturation=1.08:brightness=0.01"
    ),
}


def encode_args(level: str = "high", extra_vf: str = "") -> List[str]:
    """وسائط ffmpeg بعد -i لإخراج mp4 عالي الجودة."""
    cfg = LEVELS.get(level, LEVELS["high"])
    args: List[str] = []
    vf = extra_vf.strip()
    if vf:
        args += ["-vf", vf]
    args += [
        "-c:v", "libx264",
        "-preset", cfg["preset"],
        "-crf", str(cfg["crf"]),
        "-profile:v", "high",
        "-level", "4.2",
        "-pix_fmt", "yuv420p",
        "-tune", "film",
        "-x264-params", cfg["x264_params"],
        "-c:a", "aac",
        "-b:a", cfg["audio_bitrate"],
        "-movflags", "+faststart",
    ]
    return args


def write_videofile_kwargs(level: str = "high") -> Dict:
    """kwargs مناسبة لـ moviepy.write_videofile."""
    cfg = LEVELS.get(level, LEVELS["high"])
    return {
        "codec": "libx264",
        "audio_codec": "aac",
        "audio_bitrate": cfg["audio_bitrate"],
        "preset": cfg["preset"],
        "ffmpeg_params": [
            "-crf", str(cfg["crf"]),
            "-profile:v", "high",
            "-level", "4.2",
            "-pix_fmt", "yuv420p",
            "-tune", "film",
            "-x264-params", cfg["x264_params"],
            "-movflags", "+faststart",
        ],
        "logger": None,
    }


def pick_level_for_duration(seconds: float, professional: bool = False) -> str:
    if professional and seconds <= 120:
        return "archive"
    if seconds <= 90:
        return "high"
    if seconds <= 300:
        return "balanced"
    return "fast"
