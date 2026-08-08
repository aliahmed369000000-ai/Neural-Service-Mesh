# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.ffmpeg_quality import encode_args, pick_level_for_duration, LEVELS


def test_encode_args_high():
    args = encode_args("high")
    assert "-crf" in args
    assert "libx264" in args
    assert str(LEVELS["high"]["crf"]) in args


def test_pick_level_pro_short():
    assert pick_level_for_duration(60, professional=True) == "archive"
    assert pick_level_for_duration(200, professional=False) == "balanced"
