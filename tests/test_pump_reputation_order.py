"""pump_inboxes يرتّب العُقد والمرسلين حسب السمعة — سلوك حقيقي قابل للتحقق."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT, RIGHT


@pytest.fixture()
def bundle_paths():
    tmp = tempfile.mkdtemp(prefix="nsm_pump_rep_")
    try:
        yield tmp, f"{tmp}/mesh.db"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_pump_inboxes_reports_reputation_order(bundle_paths):
    storage_dir, db_path = bundle_paths
    b = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    stats = b.pump_inboxes()
    assert isinstance(stats, dict)
    assert stats.get("order") == "reputation_desc"
    assert "pings_answered" in stats
    assert "deferred" in stats


def test_pick_audit_records_reputation_vs_performance():
    from ai.swarm_coordinator import SwarmCoordinator

    class _A:
        def __init__(self, agent_id, role, performance_score):
            self.agent_id = agent_id
            self.role = role
            self.performance_score = performance_score

    class _F:
        def __init__(self, agents):
            self._agents = agents
        def list_by_capability(self, capability):
            return list(self._agents)
        def spawn(self, role):
            raise RuntimeError("no spawn")

    low_rep_high_perf = _A("a1", "RoleA", 0.99)
    high_rep_low_perf = _A("b1", "RoleB", 0.10)
    factory = _F([low_rep_high_perf, high_rep_low_perf])
    coord = SwarmCoordinator(
        factory,
        role_reputation=lambda role: 0.9 if role == "RoleB" else 0.05,
    )
    picked = coord._pick_agent("x")
    assert picked.agent_id == "b1"
    assert coord._last_pick_audit, "يجب تسجيل قرار الاختيار"
    last = coord._last_pick_audit[-1]
    assert last["chosen_id"] == "b1"
    assert last["reputation_used"] is True
    assert last["chosen_rep"] >= last["candidates"][0]["rep"] - 1e-9
    assert len(last["candidates"]) == 2
