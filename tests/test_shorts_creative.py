# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.fable_engine import SHORTS_STYLES, DEFAULT_SHORTS_STYLE, FableEngine, ExplainerSegment


def test_styles_present():
    assert "تعليمي" in SHORTS_STYLES
    assert DEFAULT_SHORTS_STYLE in SHORTS_STYLES


def test_offline_short():
    eng = FableEngine.__new__(FableEngine)
    eng.memory = type("M", (), {"save_short": lambda *a, **k: None})()
    script = FableEngine._generate_short_offline(
        eng,
        "القمر يبعد عن الأرض. المد والجزر ظاهرة طبيعية. الضوء يحتاج وقتاً.",
        40,
        5,
        "حقائق سريعة",
    )
    assert len(script.segments) >= 4
    assert script.format == "شورت"
    assert script.provider == "offline-creative"
    assert all(isinstance(s, ExplainerSegment) for s in script.segments)
