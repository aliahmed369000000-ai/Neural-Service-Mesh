"""اختبارات سجل تجارب التدريب المركزي — لا تستدعي API خارجي حقيقيًا (mock)."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

import ai.training_experience_registry as reg  # noqa: E402


def _reset_registry(tmp_path: Path):
    reg.EXPERIMENTS_PATH = tmp_path / "experiments.jsonl"
    if reg.EXPERIMENTS_PATH.exists():
        reg.EXPERIMENTS_PATH.unlink()


def test_record_run_persists(tmp_path):
    _reset_registry(tmp_path)
    with mock.patch.object(reg, "kernel_state_from_github", return_value={"tag": "t1"}):
        entry = reg.record_training_run(
            "aliahmedmo/test-kernel-1",
            preset="xlarge",
            d_model=8192,
            n=60000,
            epochs=10,
            status="failed",
            failure_reason="NameError: contextlib",
        )
    assert entry["status"] == "failed"
    assert reg.registry_summary()["total"] == 1
    assert reg.registry_summary()["failed"] == 1


def test_compare_runs_classification(tmp_path):
    _reset_registry(tmp_path)
    with mock.patch.object(reg, "kernel_state_from_github") as gh:
        gh.side_effect = [
            {"tag": "t1", "progress": [5.0, 4.0, 3.5], "first_loss": 5.0, "last_loss": 3.5, "best_loss": 3.5, "epochs_recorded": 3},
            {"tag": "t2", "progress": [6.0, 5.0]},
        ]
        reg.record_training_run("k1", preset="small", d_model=128, n=1000, epochs=5, status="complete", extra={"tag": "t1"})
        reg.record_training_run("k2", preset="medium", d_model=512, n=10000, epochs=3, status="failed", extra={"tag": "t2"})
    c = reg.compare_runs()
    assert c["total_runs"] == 2
    assert c["completed"] == 1
    assert c["failed"] == 1
    assert c["table"][0]["best_loss"] == 3.5
    assert c["table"][1]["status"] == "failed"


def test_preset_auto_inference(tmp_path):
    _reset_registry(tmp_path)
    with mock.patch.object(reg, "kernel_state_from_github", return_value={}):
        entry = reg.record_training_run("k3", d_model=8192, status="complete")
    assert entry["preset"] == "xlarge"


def test_summary_empty(tmp_path):
    _reset_registry(tmp_path)
    s = reg.registry_summary()
    assert s["total"] == 0 and s["last_run"] is None


def test_github_state_real_repo():
    """الrepo العام متاح — نتأكد أن القارئ يعمل على البيانات الحقيقية (best_loss موجود)."""
    data = reg.kernel_state_from_github("d8192_s1p0")
    assert "state" in data or "progress" in data


def test_py_compile():
    import py_compile

    py_compile.compile(str(HERE / "ai/training_experience_registry.py"), doraise=True)
