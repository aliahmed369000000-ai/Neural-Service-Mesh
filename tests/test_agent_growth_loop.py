# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.agent_growth_loop import (
    decompose_goal,
    handle_growth_command,
    inspect_project,
    run_safe_mission,
    similar_experiences,
    record_experience,
)
from ai.agent_project_bridge import dispatch_agent_message, agent_integration_status


def test_inspect_project():
    info = inspect_project()
    assert info["ai_modules"] > 0
    assert info["has_nsm_agent"] is True


def test_decompose_goal_tests():
    p = decompose_goal("افحص المشروع وشغّل اختبارات")
    assert "run_safe_tests" in p["tools"] or "inspect_project" in p["tools"]
    assert p["dangerous"] is False


def test_reject_dangerous():
    out = run_safe_mission("rm -rf / && git push --force", execute=True)
    assert "رفض" in out or "🚫" in out


def test_plan_only():
    out = handle_growth_command("خطة: افحص المشروع")
    assert out is not None
    assert "خطة" in out


def test_growth_status():
    out = handle_growth_command("حالة نمو الوكيل")
    assert out is not None
    assert "نمو" in out


def test_bridge_dispatches_growth():
    out = dispatch_agent_message("حالة نمو الوكيل")
    assert out is not None
    assert "نمو" in out


def test_integration_includes_growth():
    st = agent_integration_status()
    assert "agent_growth_loop" in st["components"]


def test_record_and_similar():
    record_experience("تشغيل اختبارات آمنة", ["a"], ["run_safe_tests"], True, "ok")
    sims = similar_experiences("اختبارات المشروع")
    assert isinstance(sims, list)
