"""تفضيل السمعة عند اختيار وكيل السرب — مسار حقيقي لا وهمي."""
from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

from ai.swarm_coordinator import SwarmCoordinator


class _FakeAgent:
    def __init__(self, agent_id: str, role: str, performance_score: float):
        self.agent_id = agent_id
        self.role = role
        self.performance_score = performance_score


class _FakeFactory:
    def __init__(self, agents):
        self._agents = list(agents)

    def list_by_capability(self, capability: str):
        return list(self._agents)

    def spawn(self, role: str):
        raise RuntimeError("should not spawn in this test")


def test_pick_agent_prefers_higher_reputation_over_performance():
    # A: أداء أعلى لكن سمعة أقل — B: أداء أقل وسمعة أعلى → يُختار B
    a = _FakeAgent("a1", "RoleA", performance_score=0.95)
    b = _FakeAgent("b1", "RoleB", performance_score=0.40)
    factory = _FakeFactory([a, b])
    rep = {"RoleA": 0.1, "RoleB": 0.9}

    coord = SwarmCoordinator(
        factory,
        role_reputation=lambda role: float(rep.get(role, 0.0)),
    )
    picked = coord._pick_agent("any")
    assert picked is not None
    assert picked.agent_id == "b1"


def test_pick_agent_falls_back_to_performance_without_reputation_cb():
    a = _FakeAgent("a1", "RoleA", performance_score=0.95)
    b = _FakeAgent("b1", "RoleB", performance_score=0.40)
    factory = _FakeFactory([a, b])
    coord = SwarmCoordinator(factory)  # بلا role_reputation
    picked = coord._pick_agent("any")
    assert picked is not None
    assert picked.agent_id == "a1"


def test_pick_agent_skips_quarantined_even_if_high_rep():
    a = _FakeAgent("a1", "RoleA", performance_score=0.5)
    b = _FakeAgent("b1", "RoleB", performance_score=0.5)
    factory = _FakeFactory([a, b])
    coord = SwarmCoordinator(
        factory,
        is_role_quarantined=lambda role: role == "RoleB",
        role_reputation=lambda role: 1.0 if role == "RoleB" else 0.0,
    )
    picked = coord._pick_agent("any")
    assert picked is not None
    assert picked.agent_id == "a1"


def test_get_pick_audit_returns_newest_first():
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

    factory = _F([_A("a1", "RoleA", 0.5), _A("b1", "RoleB", 0.8)])
    coord = SwarmCoordinator(
        factory,
        role_reputation=lambda role: 0.2 if role == "RoleA" else 0.9,
    )
    coord._pick_agent("cap1")
    coord._pick_agent("cap2")
    audit = coord.get_pick_audit(limit=5)
    assert len(audit) >= 2
    assert audit[0]["capability"] == "cap2"  # newest first
    assert audit[0]["chosen_role"] == "RoleB"
    assert audit[0]["reputation_used"] is True
