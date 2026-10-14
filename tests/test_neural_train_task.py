# -*- coding: utf-8 -*-
"""مهمة neural_train الموزَّعة: كل عقدة تُدرِّب شبكتها العصبية الحقيقية (ai.neural_core)
على شريحة بياناتها. الحمولة تأتي من نظير، فالتحقق الصارم من الحدود جزء من العقد."""
from __future__ import annotations

import time

import numpy as np
import pytest

from ai import mesh_task_protocol as mt


def _shard(n, seed, noise=0.02):
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, (n, 2))
    y = np.sin(3 * X[:, 0]) * X[:, 1] + r.normal(0, noise, n)
    return X.tolist(), y.tolist()


def _task(seed=0, n=150, **kw):
    X, y = _shard(n, seed)
    base = {"task_id": f"t{seed}", "x": X, "y": y, "hidden": [16, 8], "epochs": 40,
            "lr": 0.01, "seed": seed}
    base.update(kw)
    return base


# ── تعلّم حقيقي ───────────────────────────────────────────────────────────
def test_regression_learns_far_better_than_baseline():
    r = mt.execute_neural_train(_task())
    assert r["ok"] and r["epochs_run"] == 40 and not r["stopped_early"]
    assert r["improvement"] > 0.8 and r["val_loss"] < r["baseline_val_loss"] * 0.2
    assert r["params"] > 0 and "tanh" in r["network"]


def test_classification_beats_majority_baseline():
    r = np.random.default_rng(3)
    X = r.uniform(-1, 1, (240, 2))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 0.5).astype(float)      # دائرة: غير خطية
    out = mt.execute_neural_train({"x": X.tolist(), "y": y.tolist(), "task": "classification",
                                   "hidden": [16, 8], "epochs": 60, "lr": 0.02, "seed": 1})
    assert out["ok"] and out["val_accuracy"] > 0.85
    assert out["val_accuracy"] > out["baseline_accuracy"]


def test_deterministic_for_same_seed_and_different_for_different_seed():
    a, b = mt.execute_neural_train(_task(seed=5)), mt.execute_neural_train(_task(seed=5))
    c = mt.execute_neural_train(_task(seed=5) | {"seed": 6})   # نفس البيانات، تهيئة/تقسيم مختلفان
    assert a["val_loss"] == b["val_loss"] and a["train_loss"] == b["train_loss"]
    assert a["val_loss"] != c["val_loss"]


def test_predictions_returned_for_requested_rows_only():
    Xp = [[0.1, 0.2], [-0.5, 0.4], [0.9, -0.9]]
    r = mt.execute_neural_train(_task(predict=Xp))
    assert len(r["predictions"]) == 3 and all(isinstance(v, float) for v in r["predictions"])
    assert "predictions" not in mt.execute_neural_train(_task())


# ── توزيع الشرائح ودمج النتائج ───────────────────────────────────────────
def test_three_nodes_ensemble_generalises_to_unseen_data():
    Xt, yt = _shard(60, 999)
    results = [mt.dispatch_task(mt.KIND_NEURAL_TRAIN, _task(seed=n) | {"predict": Xt}) for n in range(3)]
    merged = mt.merge_neural_results(results)
    assert merged["ok"] and merged["nodes_ok"] == 3 and merged["ensemble_size"] == 3
    yt = np.array(yt)
    indiv = [float(np.mean((np.array(r["predictions"]) - yt) ** 2)) for r in results]
    ens = float(np.mean((np.array(merged["ensemble_predictions"]) - yt) ** 2))
    assert ens <= float(np.mean(indiv)) and ens < float(yt.var()) * 0.1
    assert merged["best_val_loss"] == min(r["val_loss"] for r in results)


def test_merge_tolerates_failed_nodes_and_mismatched_predictions():
    ok = mt.execute_neural_train(_task(predict=[[0.0, 0.0]]))
    m = mt.merge_neural_results([ok, {"ok": False, "error": "x"}, None])
    assert m["nodes_ok"] == 1 and m["nodes_failed"] == 2
    ok2 = mt.execute_neural_train(_task(seed=2, predict=[[0.0, 0.0], [1.0, 1.0]]))
    assert "ensemble_predictions" not in mt.merge_neural_results([ok, ok2])   # أطوال مختلفة
    assert mt.merge_neural_results([]) == {"ok": False, "nodes_ok": 0, "nodes_failed": 0}


# ── التحقق الصارم من الحمولة (تأتي من نظير) ───────────────────────────────
@pytest.mark.parametrize("patch,err", [
    ({"x": [[1, 2]] * 5}, "x_rows_out_of_range"),                    # أقل من 8 صفوف
    ({"x": [[0.1, 0.2]] * 401, "y": [0.0] * 401}, "x_rows_out_of_range"),
    ({"x": [[0.1] * 9] * 20, "y": [0.0] * 20}, "x_bad_row"),         # أكثر من 8 أبعاد
    ({"x": [[0.1, 0.2]] * 7 + [[0.1]], "y": [0.0] * 8}, "x_ragged"),
    ({"x": [[0.1, "a"]] * 10, "y": [0.0] * 10}, "x_non_numeric"),
    ({"x": [[0.1, True]] * 10, "y": [0.0] * 10}, "x_non_numeric"),
    ({"x": [[0.1, float("nan")]] * 10, "y": [0.0] * 10}, "x_value_out_of_range"),
    ({"x": [[0.1, 1e9]] * 10, "y": [0.0] * 10}, "x_value_out_of_range"),
    ({"y": [0.0] * 3}, "y_length_mismatch"),
    ({"task": "ranking"}, "task_must_be_regression_or_classification"),
    ({"task": "classification"}, "classification_labels_must_be_0_or_1"),
    ({"hidden": [64]}, "hidden_out_of_range"),
    ({"hidden": [4, 4, 4, 4]}, "hidden_out_of_range"),
    ({"hidden": ["a"]}, "hidden_out_of_range"),
    ({"hidden": []}, None),                                           # فارغ → الافتراضي
    ({"predict": [[1.0]]}, "predict_width_mismatch"),
    ({"predict": [[0.1, 0.2]] * 65}, "predict_rows_out_of_range"),
])
def test_invalid_payloads_return_error_never_raise(patch, err):
    r = mt.execute_neural_train(_task(n=20) | patch)
    if err is None:
        assert r["ok"]
    else:
        assert r["ok"] is False and r["error"] == err and r["task_id"] == "t0"


@pytest.mark.parametrize("junk", [None, {}, {"x": "no"}, {"x": None, "y": None}])
def test_garbage_payloads_do_not_raise(junk):
    assert mt.execute_neural_train(junk or {})["ok"] is False


def test_resource_limits_are_clamped_not_trusted():
    t0 = time.time()
    r = mt.execute_neural_train(_task(n=60, epochs=10 ** 9, lr=1e9))
    assert r["ok"] and r["epochs_run"] == 100                      # الحد الأقصى للدورات
    assert time.time() - t0 < 20


def test_hard_time_budget_stops_training_early(monkeypatch):
    monkeypatch.setattr(mt, "_NEURAL_BUDGET_S", 0.0)
    r = mt.execute_neural_train(_task(epochs=50))
    assert r["ok"] and r["stopped_early"] and r["epochs_run"] < 50


# ── التسجيل في البروتوكول ────────────────────────────────────────────────
def test_registered_in_protocol():
    assert mt.KIND_NEURAL_TRAIN in mt.ALL_TASK_KINDS and mt.KIND_NEURAL_TRAIN_RESULT in mt.ALL_TASK_KINDS
    assert mt.result_kind_for(mt.KIND_NEURAL_TRAIN) == mt.KIND_NEURAL_TRAIN_RESULT
    assert mt.dispatch_task(mt.KIND_NEURAL_TRAIN, _task())["ok"]
