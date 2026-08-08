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

def test_hero_strategy_selects_key_shots():
    from ai.video_engine import VideoEngine
    eng = VideoEngine(professional_mode=True, cinematic_strategy="hero")
    eng._use_cinematic_backgrounds = True
    eng._pro_total_segments = 6
    assert eng._should_fetch_cinematic(0) is True
    assert eng._should_fetch_cinematic(5) is True
    assert eng._should_fetch_cinematic(3) is True  # middle
    assert eng._should_fetch_cinematic(1) is False


def test_free_provider_alias():
    from ai.video_engine import VideoEngine
    eng = VideoEngine(cinematic_provider="free")
    assert eng._cinematic_provider == "wan_free"


def test_enhance_free_clip_noop_missing():
    from ai.video_engine import _enhance_free_clip
    assert _enhance_free_clip("/tmp/no_such_clip.mp4", "/tmp/out_x.mp4") == "/tmp/no_such_clip.mp4"


def test_negative_prompt_stronger():
    from ai.video_engine import _wan_free_negative_prompt
    n = _wan_free_negative_prompt()
    assert "watermark" in n
    assert "blurry" in n
