# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.video_ai_enhance import list_presets, PRESETS, enhance_auto


def test_presets_nonempty():
    assert len(list_presets()) >= 5
    assert "clarity" in PRESETS


def test_enhance_auto_rejects_missing():
    import pytest
    from ai.video_ai_enhance import VideoAIEnhanceError
    with pytest.raises(VideoAIEnhanceError):
        enhance_auto("/tmp/nsm_missing_video_xyz.mp4")
