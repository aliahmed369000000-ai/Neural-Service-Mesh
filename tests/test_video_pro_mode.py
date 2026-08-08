# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

def test_video_engine_accepts_professional_flag():
    from ai.video_engine import VideoEngine
    eng = VideoEngine(professional_mode=True, use_background_music=False)
    assert eng._professional_mode is True
    assert eng._use_background_music is False

def test_progress_bar_draw():
    from PIL import Image
    from ai.video_engine import VideoEngine, FRAME_W, FRAME_H
    eng = VideoEngine(professional_mode=True)
    img = Image.new("RGB", (FRAME_W, FRAME_H), (20, 20, 40))
    out = eng._draw_progress_bar(img, 0.4)
    assert out.size == (FRAME_W, FRAME_H)
