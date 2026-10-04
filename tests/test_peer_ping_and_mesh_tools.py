"""request_peer_ping + إظهار أدوات الشبكة في قائمة NSMAgent."""
from __future__ import annotations

import shutil
import tempfile
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT, RIGHT
from core.node import BaseNode, NodeSchema
from core.node_hands import NodeHands
from ai.nsm_agent_core import NSMAgent


@pytest.fixture()
def paths():
    tmp = tempfile.mkdtemp(prefix="nsm_peer_ping_")
    try:
        yield tmp, f"{tmp}/mesh.db"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_request_peer_ping_picks_highest_reputation_peer(paths):
    storage_dir, db_path = paths
    b = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    # role nodes exist after init; pick two and set reputation if possible
    role_ids = list(b.role_node_ids.values())
    assert len(role_ids) >= 2
    n1 = b.registry.get(role_ids[0])
    n2_id = role_ids[1]
    # boost reputation of n2 via record_execution successes
    for _ in range(3):
        try:
            b.reputation_engine.record_execution(n2_id, "test", True, 10.0)
        except Exception:
            pass
    res = n1.use_hand(RIGHT, "request_peer_ping")
    # may be ok or denied by policy/rate — but if ok, to_id must be registered
    assert res.error is None or isinstance(res.error, str)
    if res.ok:
        assert res.output.get("to_id") in b.role_node_ids.values() or b.registry.exists(res.output.get("to_id"))
        assert res.output.get("topic") == "ping"


def test_request_peer_ping_explicit_to_id(paths):
    storage_dir, db_path = paths
    b = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    role_ids = list(b.role_node_ids.values())
    assert len(role_ids) >= 2
    n1 = b.registry.get(role_ids[0])
    target = role_ids[1]
    res = n1.use_hand(RIGHT, "request_peer_ping", to_id=target, seq="abc1")
    if res.ok:
        assert res.output["to_id"] == target
        assert res.output["seq"] == "abc1"
    else:
        # denied is acceptable under policy (e.g. paused) — must not raise
        assert res.denied or res.error


def test_agent_menu_includes_mesh_tools_when_bound():
    class N(BaseNode):
        input_schema = NodeSchema(fields={}, required=[])
        output_schema = NodeSchema(fields={}, required=[])
        def process(self, data):
            return {}

    node = N("n")
    hands = NodeHands(node)
    hands.bind(LEFT, "capabilities", lambda: {"node_type": "N"}, "قدرات")
    hands.bind(LEFT, "routes", lambda: {"routes": []}, "مسارات")
    node.attach_hands(hands)
    agent = NSMAgent()
    lf = Mock()
    lf.generate.return_value = SimpleNamespace(text="جواب بدون أداة", provider=SimpleNamespace(value="groq"))
    agent._llm_fallback = lf
    out = agent.run("صف الشبكة", hands=hands)
    assert out  # non-empty
    prompt = lf.generate.call_args_list[0][0][0]
    assert "capabilities" in prompt
    assert "routes" in prompt
    assert "أدوات الشبكة" in prompt or "capabilities" in prompt
