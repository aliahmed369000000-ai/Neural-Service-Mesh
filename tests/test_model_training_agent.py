# -*- coding: utf-8 -*-
"""اختبارات دورة التدريب الموحّدة في model_training_agent."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai.model_training_agent import (
    handle_training_command,
    inspect_training_data,
    list_training_runs,
    run_training_mission,
    _safe_resolve_data_path,
)


DEMO = "data/samples/classification_demo.csv"


def test_safe_path_demo_csv():
    p = _safe_resolve_data_path(DEMO)
    assert p.is_file()
    assert "classification_demo.csv" in p.name


def test_reject_outside_path():
    try:
        _safe_resolve_data_path("/etc/passwd")
        assert False, "should reject"
    except (PermissionError, FileNotFoundError):
        pass


def test_inspect_demo():
    info = inspect_training_data(DEMO, target_col="label")
    assert info["task"] == "classification"
    assert info["n_samples"] == 200
    assert info["n_features"] >= 1
    assert info["n_classes"] == 2
    assert info["engine"] in ("sklearn", "torch")


def test_mission_dry_run_only():
    out = run_training_mission(DEMO, target_col="label", execute=False)
    assert "معاينة" in out or "Dry-run" in out
    assert "نفّذ" in out


def test_mission_execute_rf():
    out = run_training_mission(DEMO, target_col="label", prefer="sklearn", execute=True)
    assert "completed" in out.lower() or "Accuracy" in out or "تدريب" in out or "نتائج" in out


def test_handle_mission_preview():
    out = handle_training_command(f"مهمة تدريب {DEMO} الهدف=label")
    assert out is not None
    assert "معاينة" in out or "خطة" in out


def test_handle_mission_execute():
    out = handle_training_command(f"مهمة تدريب {DEMO} الهدف=label نفّذ")
    assert out is not None
    assert "Accuracy" in out or "نتائج" in out or "تدريب" in out


def test_list_runs():
    out = list_training_runs()
    assert "سجل" in out


def test_handle_runs_log():
    out = handle_training_command("سجل مهام التدريب")
    assert out is not None
    assert "سجل" in out


def test_legacy_csv_train_still_works():
    out = handle_training_command(f"درّب على csv {DEMO} الهدف=label")
    assert out is not None
    assert "CSV" in out or "تدريب" in out or "Accuracy" in out


def test_dashboard_contains_runs_section():
    out = handle_training_command("لوحة التحكم")
    assert out is not None
    assert "مهام" in out or "MoE" in out or "تدريب" in out
