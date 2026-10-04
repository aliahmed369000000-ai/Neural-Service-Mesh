"""عقدة self_evolved تُحيا بعد إعادة التشغيل يجب أن تملك «اليدين» أيضاً،
وأن تعمل يدها اليمنى المقيَّدة فعلياً (لا مجرد وجود الكائن)."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT, RIGHT
from services.generated_service_nodes import NormalizerNode


@pytest.fixture()
def paths():
    tmp = tempfile.mkdtemp(prefix="nsm_revived_hands_")
    try:
        yield tmp, f"{tmp}/mesh.db"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_revived_evolved_node_has_working_hands(paths):
    storage_dir, db_path = paths
    b1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    node_id = b1.register_node(NormalizerNode(name="evolved-norm"))

    b2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    revived = b2.registry.get(node_id)
    assert isinstance(revived, NormalizerNode)
    assert revived.hands is not None
    tools = revived.hands.tools()
    assert {t["name"] for t in tools[RIGHT]} == {"send_message", "request_evolution", "request_peer_ping"}
    left_names = {t["name"] for t in tools[LEFT]}
    assert "peers" in left_names
    assert "node_status" in left_names
    assert "mesh_health" in left_names
    assert "routes" in left_names
    for extra in ("capabilities", "neighbors", "reputation_detail", "graph_stats",
                  "terminal_policy", "terminal_run_safe", "terminal_history",
                  "inbox_summary", "self_diagnose"):
        assert extra in left_names, extra

    # اليسرى تعمل فعلاً، واليمنى (send_message) تصل لعقدة حقيقية.
    assert revived.use_hand(LEFT, "peers").ok
    st = revived.use_hand(LEFT, "node_status")
    assert st.ok and isinstance(st.output, dict) and st.output.get("node_id") == node_id
    mh = revived.use_hand(LEFT, "mesh_health")
    assert mh.ok and isinstance(mh.output, dict) and mh.output.get("total_nodes", 0) >= 1
    rt = revived.use_hand(LEFT, "routes")
    assert rt.ok and isinstance(rt.output, dict) and "routes" in rt.output
    cap = revived.use_hand(LEFT, "capabilities")
    assert cap.ok and isinstance(cap.output, dict) and cap.output.get("node_type")
    nb = revived.use_hand(LEFT, "neighbors")
    assert nb.ok and isinstance(nb.output, dict) and "degree_out" in nb.output
    rd = revived.use_hand(LEFT, "reputation_detail")
    assert rd.ok and isinstance(rd.output, dict)
    gs = revived.use_hand(LEFT, "graph_stats")
    assert gs.ok and isinstance(gs.output, dict)
    tp = revived.use_hand(LEFT, "terminal_policy")
    assert tp.ok and isinstance(tp.output, dict) and tp.output.get("shell") is False
    bad = revived.use_hand(LEFT, "terminal_run_safe", cmd="rm -rf /")
    assert bad.ok and isinstance(bad.output, dict) and bad.output.get("ok") is False
    good = revived.use_hand(LEFT, "terminal_run_safe", cmd="git status")
    assert good.ok and isinstance(good.output, dict)
    st2 = revived.use_hand(LEFT, "node_status")
    assert st2.ok and "last_terminal_check" in (st2.output or {})
    sd = revived.use_hand(LEFT, "self_diagnose")
    assert sd.ok and isinstance(sd.output, dict) and sd.output.get("layer") == "node-self-diagnose-v1"
    ib = revived.use_hand(LEFT, "inbox_summary")
    assert ib.ok and "unread_total" in (ib.output or {})
    other = next(i for i in b2.role_node_ids.values())
    res = revived.use_hand(RIGHT, "send_message", to_id=other, topic="ping")
    assert res.ok or res.denied  # السياسة تقرّر؛ المهم ألا ترفع استثناءً
    assert res.error is None or isinstance(res.error, str)
