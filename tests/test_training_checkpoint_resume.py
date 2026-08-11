# -*- coding: utf-8 -*-
"""اختبارات checkpoint / استئناف بعد الفشل / rollback في model_training_agent.

تستخدم بيانات اصطناعية (numpy) مباشرة بدل CSV لتفادي الاعتماد على Git LFS.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ai.model_training_agent import (  # noqa: E402
    train_torch_on_arrays,
    _checkpoint_dir,
    _load_checkpoint,
    find_resumable_run,
    rollback_to_best,
)


def _demo_arrays(n=200, n_features=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype("float32")
    w = rng.normal(size=(n_features,))
    y = (X @ w > 0).astype("int64")
    return X, y


def test_checkpoint_created_on_train():
    X, y = _demo_arrays()
    run_id = "test_run_ckpt_1"
    out = train_torch_on_arrays(X, y, task="classification", epochs=6, run_id=run_id, checkpoint_every=2)
    assert "✅" in out
    ck = _load_checkpoint(run_id, "latest")
    assert ck is not None
    assert ck["epoch"] == 6
    assert (_checkpoint_dir(run_id) / "meta.json").is_file()


def test_resume_continues_from_last_epoch():
    X, y = _demo_arrays(seed=1)
    run_id = "test_run_ckpt_resume"
    train_torch_on_arrays(X, y, task="classification", epochs=4, run_id=run_id, checkpoint_every=2)
    ck_before = _load_checkpoint(run_id, "latest")
    assert ck_before["epoch"] == 4

    out = train_torch_on_arrays(
        X, y, task="classification", epochs=3, run_id=run_id, resume=True, checkpoint_every=2
    )
    assert "استؤنف من الحقبة 4" in out
    ck_after = _load_checkpoint(run_id, "latest")
    # 4 (سابقة) + 3 (جديدة) = 7
    assert ck_after["epoch"] == 7


def test_find_resumable_run_needs_checkpoint():
    # اسم عشوائي لا يوجد له checkpoint
    assert find_resumable_run(path_str="no/such/path.csv", target_col="zzz") is None


def test_rollback_to_best_produces_file():
    X, y = _demo_arrays(seed=2)
    run_id = "test_run_ckpt_rollback"
    train_torch_on_arrays(X, y, task="classification", epochs=6, run_id=run_id, checkpoint_every=1)
    out = rollback_to_best(run_id=run_id)
    assert "أفضل نسخة" in out
    assert run_id in out
