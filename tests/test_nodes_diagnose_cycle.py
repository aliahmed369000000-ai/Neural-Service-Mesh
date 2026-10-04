"""دورة تشخيص العُقد الدورية + تنبيهات العتبات."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_diag_cycle_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_run_nodes_diagnose_cycle_stores_summary(bundle):
    out = bundle.run_nodes_diagnose_cycle()
    assert out["scanned"] >= 1
    assert "ts" in out
    summary = bundle.get_nodes_diagnose_summary()
    assert summary.get("layer") in ("mesh-nodes-diagnose-cycle-v1", "mesh-nodes-diagnose-cycle-v2")
    assert summary.get("scanned") == out["scanned"]
    assert isinstance(summary.get("nodes"), list)
    assert len(summary["nodes"]) >= 1


def test_summary_includes_nodes_diagnose(bundle):
    bundle.run_nodes_diagnose_cycle()
    s = bundle.summary()
    assert "nodes_diagnose" in s
    assert s["nodes_diagnose"].get("scanned", 0) >= 1


def test_system_hub_check_mesh_nodes():
    from ai.system_hub import check_mesh_nodes
    # may use process singleton — just ensure structure
    r = check_mesh_nodes()
    assert "ok" in r and "detail" in r


def test_peer_compare_and_mesh_diagnose_summary_tools(bundle):
    from core.node_hands import LEFT
    bundle.run_nodes_diagnose_cycle()
    n = bundle.registry.get(list(bundle.role_node_ids.values())[0])
    ms = n.use_hand(LEFT, "mesh_diagnose_summary")
    assert ms.ok and ms.output.get("scanned", 0) >= 1
    pc = n.use_hand(LEFT, "peer_compare")
    assert pc.ok
    assert "my_reputation" in pc.output
    assert "rank_among_peers" in pc.output
    assert "peer_avg_reputation" in pc.output


def test_diagnose_cycle_broadcasts_mesh_diagnose_topic(bundle):
    """بعد الدورة تُبث رسائل topic=mesh_diagnose لصناديق الأدوار."""
    from core.node_hands import LEFT

    bundle.run_nodes_diagnose_cycle()
    role_ids = list(bundle.role_node_ids.values())
    assert len(role_ids) >= 2
    # مرسل = root أو أول دور — المستلمون الآخرون يجب أن يروا الموضوع
    found = 0
    for nid in role_ids:
        node = bundle.registry.get(nid)
        if node is None or node.hands is None:
            continue
        ib = node.use_hand(LEFT, "read_inbox", unread_only=False, topic="mesh_diagnose", limit=5)
        if ib.ok and ib.output:
            found += 1
            assert any(m.get("topic") == "mesh_diagnose" for m in ib.output)
    assert found >= 1


def test_diagnose_cycle_dynamic_threshold_and_collective(bundle):
    from core.node_hands import LEFT
    out = bundle.run_nodes_diagnose_cycle()
    assert out["scanned"] >= 1
    summary = bundle.get_nodes_diagnose_summary()
    assert summary.get("layer") == "mesh-nodes-diagnose-cycle-v2"
    assert "effective_low_rep_threshold" in summary
    assert "avg_reputation" in summary
    assert summary["effective_low_rep_threshold"] >= 0.05
    n = bundle.registry.get(list(bundle.role_node_ids.values())[0])
    d = n.use_hand(LEFT, "self_diagnose")
    assert d.ok
    assert "collective_health" in d.output
    ib = n.use_hand(LEFT, "inbox_summary")
    assert ib.ok
    assert "collective_diagnose" in ib.output
    assert "mesh_diagnose_unread" in ib.output


def test_role_reputation_penalizes_low_diagnose_nodes(bundle):
    """تحت العتبة → ×0.1؛ فوق 1.2× العتبة → تعافٍ كامل رغم القائمة."""
    role = list(bundle.role_node_ids.keys())[0]
    nid = bundle.role_node_ids[role]
    with bundle._lock:
        bundle._node_runtime_meta["__mesh_diagnose_summary__"] = {
            "ts": "test",
            "scanned": 1,
            "effective_low_rep_threshold": 0.15,
            "low_reputation": [{"node_id": nid, "name": role, "reputation": 0.01}],
            "high_unread": [],
            "nodes": [],
        }
    assert bundle._routing_penalty_factor(nid, 0.01) == 0.1
    # إن كانت السمعة الخام عالية جداً يُرفع العامل تلقائياً (تعافٍ)
    assert bundle._routing_penalty_factor(nid, 50.0) == 1.0


def test_routing_penalty_gradual_recovery(bundle):
    role = list(bundle.role_node_ids.keys())[0]
    nid = bundle.role_node_ids[role]
    thr = 0.15
    with bundle._lock:
        bundle._node_runtime_meta["__mesh_diagnose_summary__"] = {
            "ts": "test",
            "scanned": 1,
            "effective_low_rep_threshold": thr,
            "low_reputation": [{"node_id": nid, "name": role, "reputation": 0.01}],
            "high_unread": [],
            "nodes": [],
        }
    assert bundle._routing_penalty_factor(nid, 0.05) == 0.1  # still low
    assert bundle._routing_penalty_factor(nid, thr) == 0.5  # recovering
    assert bundle._routing_penalty_factor(nid, thr * 1.2) == 1.0  # recovered
    info = bundle.get_routing_penalties()
    assert "roles" in info and info["threshold"] == thr


def test_diagnose_cycle_tracks_recovered_nodes(bundle):
    """المعافون يُزالون من low_reputation ويُدرجون في recovered."""
    role = list(bundle.role_node_ids.keys())[0]
    nid = bundle.role_node_ids[role]
    # دورة 1: أجبر القائمة على اعتبار العقدة منخفضة (ملخص يدوي)
    with bundle._lock:
        bundle._node_runtime_meta["__mesh_diagnose_summary__"] = {
            "ts": "prev",
            "scanned": 1,
            "effective_low_rep_threshold": 0.15,
            "low_reputation": [{"node_id": nid, "name": role, "reputation": 0.01}],
            "high_unread": [],
            "nodes": [],
        }
    # دورة حقيقية: السمعة الافتراضية غالباً أعلى من العتبة → recovered
    out = bundle.run_nodes_diagnose_cycle()
    summary = bundle.get_nodes_diagnose_summary()
    assert "recovered" in summary
    # إما تعافت (في recovered وليست في low) أو ما زالت منخفضة
    low_ids = {x.get("node_id") for x in summary.get("low_reputation") or []}
    rec_ids = {x.get("node_id") for x in summary.get("recovered") or []}
    assert nid not in low_ids or nid not in rec_ids  # لا تتعارض
    if nid not in low_ids:
        assert nid in rec_ids


def test_pick_audit_includes_penalty_factor(bundle):
    from ai.swarm_coordinator import SwarmCoordinator

    class _A:
        def __init__(self, agent_id, role, performance_score):
            self.agent_id = agent_id
            self.role = role
            self.performance_score = performance_score

    class _F:
        def list_by_capability(self, capability):
            return [_A("a1", "RoleA", 0.5), _A("b1", "RoleB", 0.8)]
        def spawn(self, role):
            raise RuntimeError("no")

    coord = SwarmCoordinator(
        _F(),
        role_reputation=lambda r: 0.9 if r == "RoleB" else 0.2,
        role_penalty_factor=lambda r: 0.1 if r == "RoleA" else 1.0,
    )
    coord._pick_agent("x")
    audit = coord.get_pick_audit(limit=1)[0]
    assert audit.get("chosen_penalty_factor") == 1.0
    by_role = {c["role"]: c for c in audit["candidates"]}
    assert by_role["RoleA"]["penalty_factor"] == 0.1
    assert by_role["RoleB"]["penalty_factor"] == 1.0
