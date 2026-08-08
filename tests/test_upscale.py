# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import pytest
from ai.video_ai_enhance import upscale_video, VideoAIEnhanceError, PRESETS

def test_upscale_presets_exist():
    assert "upscale_2x" in PRESETS
    assert "upscale_hd" in PRESETS

def test_upscale_missing():
    with pytest.raises(VideoAIEnhanceError):
        upscale_video("/tmp/no_video_upscale_xyz.mp4")
