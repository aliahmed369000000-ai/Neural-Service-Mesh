# -*- coding: utf-8 -*-
"""تكليف عُقد بمهمة عصبية عبر POST /v2/control/task (مصادقة + allowlist + dry_run افتراضي):
كل «عقدة» هنا تنفّذ فعلياً الكود نفسه الذي تنفّذه عقدة بعيدة عند استقبال المهمة."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from ai import mesh_task_protocol as mt
from ai.node_launcher import CONTROL_TASK_KINDS, handle_control_task

TOKEN = "test-control-token"
IDS = ["mesh_node_01", "mesh_node_02", "mesh_node_03"]


class FakeMesh:
    node_id = "coordinator"

    def __init__(self, missing_key=()):
        self.calls = []
        self.missing_key = set(missing_key)

    def _load_state(self):
        return {"nodes": {i: {"host": f"{i}.example", "port": 443, "status": "online",
                              **({} if i in self.missing_key else {"public_key": "PK"})}
                          for i in IDS}}

    def _public_key_for_id(self, node_id):
        return None if node_id in self.missing_key else b"PK"

    async def dispatch_mesh_task(self, host, port, kind, payload, target_id=None, timeout=None):
        self.calls.append((target_id, kind, payload, timeout))
        # العقدة البعيدة تنفّذ dispatch_task على الحمولة التي وصلتها
        return {"ok": True, "result": mt.dispatch_task(kind, payload)}


class Req:
    def __init__(self, node, body, token=TOKEN):
        self.app = {"node": node}
        self.headers = {"X-NSM-Control-Token": token} if token else {}
        self._body = body

    async def json(self):
        return self._body


@pytest.fixture(autouse=True)
def _token(monkeypatch):
    monkeypatch.setenv("NSM_CONTROL_TOKEN", TOKEN)


def call(node, body, token=TOKEN):
    resp = asyncio.run(handle_control_task(Req(node, body, token)))
    return resp.status, json.loads(resp.body)


def _data(n=120, seed=0):
    r = np.random.default_rng(seed)
    X = r.uniform(-1, 1, (n, 2))
    y = np.sin(3 * X[:, 0]) * X[:, 1] + r.normal(0, 0.02, n)
    return X.tolist(), y.tolist()


def _body(**over):
    X, y = _data()
    Xp, yp = _data(40, 99)
    b = {"kind": "neural_train_benchmark", "targets": IDS, "dry_run": False,
         "parameters": {"x": X, "y": y, "epochs": 30, "hidden": [16, 8], "predict": Xp, "seed": 10}}
    b["_truth"] = yp
    b.update(over)
    return b


def test_kind_is_in_the_allowlist():
    assert "neural_train_benchmark" in CONTROL_TASK_KINDS


def test_dry_run_is_default_and_sends_nothing():
    mesh = FakeMesh()
    b = _body()
    b.pop("dry_run"); b.pop("_truth")
    status, out = call(mesh, b)
    assert status == 200 and out["mode"] == "dry_run" and mesh.calls == []
    assert [t["id"] for t in out["plan"]["targets"]] == IDS


def test_requires_auth():
    status, out = call(FakeMesh(), {"kind": "neural_train_benchmark", "targets": IDS}, token=None)
    assert status == 401 and out["error"] == "control_auth_required"


def test_executed_run_trains_every_node_and_the_ensemble_generalises():
    mesh = FakeMesh()
    b = _body()
    truth = np.array(b.pop("_truth"))
    status, out = call(mesh, b)
    assert status == 200 and out["ok"] and out["mode"] == "executed"
    agg = out["aggregate"]
    assert agg["method"] == "neural_ensemble_mean" and agg["nodes_ok"] == 3 and agg["ensemble_size"] == 3
    # كل عقدة استلمت بذرة مختلفة (تهيئة مستقلة لشبكتها) ومهلة معقولة
    assert [c[2]["seed"] for c in mesh.calls] == [10, 11, 12]
    assert all(c[1] == mt.KIND_NEURAL_TRAIN and c[3] == 30 for c in mesh.calls)
    ens = np.array(agg["ensemble_predictions"])
    assert float(np.mean((ens - truth) ** 2)) < float(truth.var()) * 0.15
    assert agg["mean_improvement"] > 0.8


@pytest.mark.parametrize("patch,err", [
    ({"x": [[1, 2]] * 3}, "x_rows_out_of_range"),
    ({"y": [0.0] * 3}, "y_length_mismatch"),
    ({"x": [[0.1, "a"]] * 10, "y": [0.0] * 10}, "x_non_numeric"),
    ({"task": "ranking"}, "task_must_be_regression_or_classification"),
    ({"hidden": [64]}, "hidden_out_of_range"),
    ({"epochs": 101}, "epochs_must_be_between_1_and_100"),
    ({"epochs": "many"}, "numeric_parameters_invalid"),
    ({"timeout_s": 61}, "timeout_s_must_be_between_1_and_60"),
    ({"predict": [[1.0]]}, "predict_width_mismatch"),
])
def test_parameters_are_validated_before_anything_is_sent(patch, err):
    mesh = FakeMesh()
    b = _body()
    b.pop("_truth")
    b["parameters"].update(patch)
    status, out = call(mesh, b)
    assert status == 400 and out["error"] == err and mesh.calls == []


def test_peer_public_key_required_like_other_encrypted_tasks():
    mesh = FakeMesh(missing_key={"mesh_node_02"})
    b = _body()
    b.pop("_truth")
    status, out = call(mesh, b)
    assert status == 409 and out["error"] == "peer_public_key_required_for_encrypted_task"
    assert mesh.calls == []
