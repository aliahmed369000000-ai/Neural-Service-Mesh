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
