# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.video_ai_enhance import upscale_video, _AI_UPSCALE_SPACES


def test_ai_spaces_configured():
    assert len(_AI_UPSCALE_SPACES) >= 1
    assert "Real-ESRGAN" in _AI_UPSCALE_SPACES[0]["space"]


def test_upscale_use_ai_falls_back_local(tmp_path):
    import subprocess, shutil
    if not shutil.which("ffmpeg"):
        return
    src = tmp_path / "s.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=160x90:rate=5",
            "-t", "0.4", "-pix_fmt", "yuv420p", str(src),
        ],
        capture_output=True, check=True,
    )
    # use_ai may fail network → must still produce local upscale
    out = upscale_video(src, target="2x", crf=22, use_ai=True)
    assert Path(out).is_file()
