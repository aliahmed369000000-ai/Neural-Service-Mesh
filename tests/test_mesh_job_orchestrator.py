# -*- coding: utf-8 -*-
"""اختبارات وحدة لـ MeshJobOrchestrator (بدون شبكة حقيقية)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ai.mesh_job_orchestrator import (
    MeshJobOrchestrator,
    STRATEGY_FIRST,
    STRATEGY_MAJORITY,
    STRATEGY_REPUTATION,
    _result_digest,
)


class FakeNode:
    def __init__(self):
        self.node_id = "orchestrator_node"
        self._rep = {}

    def _get_active_peers_list(self, require_capabilities=None):
        return [
            {"id": "w1", "host": "127.0.0.1", "port": 9001, "capabilities": ["text"]},
            {"id": "w2", "host": "127.0.0.1", "port": 9002, "capabilities": ["text"]},
            {"id": "w3", "host": "127.0.0.1", "port": 9003, "capabilities": ["text"]},
        ]

    def get_reputation(self, peer_id):
        return {"score": self._rep.get(peer_id, 0)}

    def update_reputation(self, peer_id, delta=1, reason=""):
        self._rep[peer_id] = self._rep.get(peer_id, 0) + int(delta)
        return {"score": self._rep[peer_id]}

    async def request_from_peer(self, host, port, kind, data, timeout=12.0):
        # يُستبدل في الاختبارات
        raise NotImplementedError


def test_digest_stable():
    r = {"ok": True, "output": "hello", "prompt_preview": "p"}
    assert _result_digest(r) == _result_digest(dict(r))
    print("✅ digest stable")


def test_majority_selection():
    node = FakeNode()
    orch = MeshJobOrchestrator(node)
    attempts = [
        {"ok": True, "worker": "w1", "task_id": "t1", "rtt_ms": 10, "digest": "aaa", "result": {"ok": True, "output": "A"}},
        {"ok": True, "worker": "w2", "task_id": "t2", "rtt_ms": 12, "digest": "aaa", "result": {"ok": True, "output": "A"}},
        {"ok": True, "worker": "w3", "task_id": "t3", "rtt_ms": 8, "digest": "bbb", "result": {"ok": True, "output": "B"}},
    ]
    sel = orch._select_winner(attempts, STRATEGY_MAJORITY)
    assert sel["winner"]["digest"] == "aaa"
    assert sel["cluster_size"] == 2
    assert abs(sel["agreement"] - 2/3) < 0.01
    print("✅ majority selection")


def test_first_success_lowest_rtt():
    node = FakeNode()
    orch = MeshJobOrchestrator(node)
    attempts = [
        {"ok": True, "worker": "w1", "rtt_ms": 20, "digest": "x", "result": {"ok": True}},
        {"ok": True, "worker": "w2", "rtt_ms": 5, "digest": "y", "result": {"ok": True}},
    ]
    sel = orch._select_winner(attempts, STRATEGY_FIRST)
    assert sel["winner"]["worker"] == "w2"
    print("✅ first_success")


def test_submit_job_with_mock_dispatch():
    node = FakeNode()
    orch = MeshJobOrchestrator(node)
    calls = []

    async def fake_dispatch(worker, kind, payload, timeout):
        calls.append(payload["task_id"])
        wid = worker.get("peer_id")
        # w2 يفشل مرة
        if wid == "w2" and not payload.get("retry_of"):
            return {
                "ok": False, "worker": wid, "task_id": payload["task_id"],
                "error": "timeout", "rtt_ms": 100, "result": None, "digest": None,
            }
        return {
            "ok": True, "worker": wid, "task_id": payload["task_id"],
            "acked": True, "rtt_ms": 8, "error": None,
            "result": {"ok": True, "output": "same"},
            "digest": "same_digest",
        }

    orch._dispatch_one = fake_dispatch  # type: ignore

    async def run():
        return await orch.submit_job(
            "inference_request",
            {"prompt": "test"},
            n_workers=3,
            strategy=STRATEGY_MAJORITY,
            retry_failed=1,
            worker_list=[
                {"peer_id": "w1", "host": "h", "port": 1},
                {"peer_id": "w2", "host": "h", "port": 2},
                {"peer_id": "w3", "host": "h", "port": 3},
            ],
        )

    report = asyncio.run(run())
    assert report["ok"] is True
    assert report["success_count"] >= 2
    assert report["winner"] is not None
    assert report["winner"]["digest"] == "same_digest"
    assert len(report["all_results"]) >= 3
    print("✅ submit_job mock", report["job_id"], "success", report["success_count"])


def test_non_retryable_duplicate_not_counted_success():
    node = FakeNode()
    orch = MeshJobOrchestrator(node)
    attempts = [
        {"ok": False, "worker": "w1", "error": "duplicate_rejected", "result": None, "digest": None},
        {"ok": True, "worker": "w2", "rtt_ms": 3, "digest": "z", "result": {"ok": True}},
    ]
    sel = orch._select_winner(attempts, STRATEGY_FIRST)
    assert sel["winner"]["worker"] == "w2"
    print("✅ duplicate excluded from winners")


if __name__ == "__main__":
    test_digest_stable()
    test_majority_selection()
    test_first_success_lowest_rtt()
    test_submit_job_with_mock_dispatch()
    test_non_retryable_duplicate_not_counted_success()
    print("\n🎉 orchestrator tests passed")
