# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pytest
from ai.video_editor import VideoEditorError, available_tools, probe, _out_path


def test_available_tools_dict():
    t = available_tools()
    assert "ffmpeg" in t
    assert "out_dir" in t


def test_probe_missing_raises():
    with pytest.raises(VideoEditorError):
        probe("/tmp/nsm_does_not_exist_xyz.mp4")


def test_out_path_under_artifacts():
    p = _out_path()
    assert "video_edits" in str(p)
