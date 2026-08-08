# -*- coding: utf-8 -*-
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.system_hub import system_snapshot, format_system_report, handle_system_command
from ai.agent_project_bridge import dispatch_agent_message


def test_snapshot_score():
    snap = system_snapshot()
    assert 0 <= snap["score"] <= 1
    assert snap["path_total"] >= 5


def test_report_text():
    r = format_system_report()
    assert "NSM" in r or "النظام" in r
    assert "MoE" in r


def test_handle_command():
    r = handle_system_command("تقرير النظام")
    assert r is not None
    assert "صحة" in r or "MoE" in r


def test_bridge_system():
    r = dispatch_agent_message("تقرير النظام")
    assert r is not None
    assert len(r) > 80
