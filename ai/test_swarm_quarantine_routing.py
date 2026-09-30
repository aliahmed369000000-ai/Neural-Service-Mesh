"""الحجر يجب أن يمنع فعلياً توجيه مهام جديدة لدور محجور (SwarmCoordinator).

قبل هذا الإصلاح: الحجر كان يوقف عقدة الـregistry الرمزية فقط، بينما
SwarmCoordinator لا يعرف شيئاً عن السمعة فيستمر بتوجيه العمل لنفس الدور.
"""
from ai.agent_factory import AgentFactory, AGENT_CATALOGUE
from ai.swarm_coordinator import SwarmCoordinator


def _coordinator(quarantined_roles):
    factory = AgentFactory()
    coord = SwarmCoordinator(
        factory, max_agents=5,
        is_role_quarantined=lambda role: role in quarantined_roles,
    )
    coord._store = None  # عزل عن قاعدة بيانات السرب
    return factory, coord


def _roles_with(capability):
    return [r for r, s in AGENT_CATALOGUE.items() if capability in s.get("capabilities", [])]


def test_no_check_behaves_as_before():
    factory = AgentFactory()
    coord = SwarmCoordinator(factory, max_agents=5)
    coord._store = None
    cap = "search"
    agent = coord._pick_agent(cap)
    assert agent is not None and cap in agent.capabilities


def test_existing_quarantined_agent_is_not_picked():
    role = _roles_with("search")[0]
    factory, coord = _coordinator({role})
    factory.spawn(role)  # وكيل موجود أصلاً من دور محجور
    agent = coord._pick_agent("search")
    assert agent is None or agent.role != role


def test_auto_spawn_skips_quarantined_role_and_uses_alternative():
    roles = _roles_with("validate") or _roles_with("search")
    cap = "validate" if _roles_with("validate") else "search"
    roles = _roles_with(cap)
    factory, coord = _coordinator({roles[0]})
    agent = coord._pick_agent(cap)
    if len(roles) > 1:
        assert agent is not None and agent.role != roles[0]
    else:
        assert agent is None  # الدور الوحيد محجور -> لا وكيل، بدل استخدامه


def test_all_roles_quarantined_returns_none():
    cap = "search"
    factory, coord = _coordinator(set(_roles_with(cap)))
    assert coord._pick_agent(cap) is None


def test_non_quarantined_agent_still_preferred_by_score():
    cap = "search"
    roles = _roles_with(cap)
    factory, coord = _coordinator({roles[0]} if len(roles) > 1 else set())
    a = factory.spawn(roles[-1])
    picked = coord._pick_agent(cap)
    assert picked is not None and picked.role != roles[0] or len(roles) == 1


def test_broken_check_function_never_blocks_execution():
    factory = AgentFactory()
    def boom(role):
        raise RuntimeError("x")
    coord = SwarmCoordinator(factory, max_agents=5, is_role_quarantined=boom)
    coord._store = None
    assert coord._pick_agent("search") is not None


def test_mesh_bundle_predicate_reads_reputation_lazily():
    from core.mesh_bundle import MeshBundle
    from ai.reputation_engine import NodeReputationEngine

    class Stub:
        pass
    s = Stub()
    s.role_node_ids = {"ResearchAgent": "n1"}
    s.reputation_engine = NodeReputationEngine()
    assert MeshBundle._is_role_quarantined(s, "ResearchAgent") is False
    s.reputation_engine.ensure_node("n1", "ResearchAgent")
    s.reputation_engine.quarantine("n1")
    assert MeshBundle._is_role_quarantined(s, "ResearchAgent") is True
    assert MeshBundle._is_role_quarantined(s, "Unknown") is False
    assert MeshBundle._is_role_quarantined(s, None) is False
