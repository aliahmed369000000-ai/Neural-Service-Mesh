# -*- coding: utf-8 -*-
"""اختبارات «اليدين» (core/node_hands.py) وتسليكهما في MeshBundle."""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from core.mesh_bundle import MeshBundle, AGENT_CATALOGUE
from core.node import BaseNode, NodeSchema, NodeState
from core.node_hands import LEFT, RIGHT, HandResult, NodeHands


class _N(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])

    def process(self, data):
        return {}


def _allow_all(node, hand, tool, kwargs):
    return True, "test"


# ── وحدات NodeHands ───────────────────────────────────────────────────────
def test_left_allowed_by_default_right_denied_by_default():
    h = NodeHands(_N("a"))
    h.bind(LEFT, "read", lambda: "r")
    h.bind(RIGHT, "act", lambda: "a")
    assert h.use(LEFT, "read").ok
    res = h.use(RIGHT, "act")
    assert not res.ok and res.denied and "default-deny" in res.error


def test_paused_node_cannot_use_any_hand():
    n = _N("a")
    h = NodeHands(n, policy=_allow_all)
    h.bind(LEFT, "read", lambda: 1)
    h.bind(RIGHT, "act", lambda: 1)
    n.pause()
    assert h.use(LEFT, "read").denied
    assert h.use(RIGHT, "act").denied


def test_policy_error_fails_closed():
    def bad(node, hand, tool, kwargs):
        raise RuntimeError("boom")

    h = NodeHands(_N("a"), policy=bad)
    h.bind(LEFT, "read", lambda: 1)
    res = h.use(LEFT, "read")
    assert res.denied and "fail-closed" in res.error


def test_unknown_tool_and_unknown_hand():
    h = NodeHands(_N("a"))
    assert "not bound" in h.use(LEFT, "nope").error
    assert "unknown hand" in h.use("middle", "x").error


def test_tool_exception_becomes_error_result_not_raise():
    h = NodeHands(_N("a"))
    h.bind(LEFT, "bad", lambda: 1 / 0)
    res = h.use(LEFT, "bad")
    assert not res.ok and "ZeroDivisionError" in res.error


def test_right_hand_rate_limit():
    h = NodeHands(_N("a"), policy=_allow_all, max_calls_per_minute={RIGHT: 2})
    h.bind(RIGHT, "act", lambda: 1)
    assert h.use(RIGHT, "act").ok
    assert h.use(RIGHT, "act").ok
    third = h.use(RIGHT, "act")
    assert third.denied and "rate limit" in third.error


def test_timeout():
    h = NodeHands(_N("a"), timeout_s=0.2)
    h.bind(LEFT, "slow", lambda: time.sleep(2))
    res = h.use(LEFT, "slow")
    assert not res.ok and "timeout" in res.error


def test_long_output_is_clipped():
    h = NodeHands(_N("a"), max_output_chars=50)
    h.bind(LEFT, "big", lambda: "x" * 500)
    out = h.use(LEFT, "big").output
    assert isinstance(out, str) and out.endswith("…[truncated]") and len(out) < 100


def test_audit_logs_denials_and_redacts_secrets(tmp_path):
    audit = tmp_path / "audit.jsonl"
    h = NodeHands(_N("a"), audit_path=audit)
    h.bind(LEFT, "read", lambda **kw: "ok")
    h.bind(RIGHT, "act", lambda **kw: "ok")
    h.use(LEFT, "read", api_token="SUPERSECRET", q="hello")
    h.use(RIGHT, "act")  # مرفوض
    raw = audit.read_text(encoding="utf-8")
    assert "SUPERSECRET" not in raw
    rows = [json.loads(line) for line in raw.splitlines()]
    assert rows[0]["args"]["api_token"] == "***"
    assert rows[1]["denied"] is True


def test_bind_validation():
    h = NodeHands(_N("a"))
    with pytest.raises(ValueError):
        h.bind("middle", "x", lambda: 1)
    with pytest.raises(TypeError):
        h.bind(LEFT, "x", "not callable")


def test_basenode_use_hand_without_hands_is_denied_not_error():
    res = _N("a").use_hand(LEFT, "read")
    assert isinstance(res, HandResult) and res.denied and "no hands" in res.error


def test_hands_not_serialized_in_to_dict():
    n = _N("a")
    n.attach_hands(NodeHands(n))
    assert "hands" not in n.to_dict()


# ── تكامل MeshBundle ──────────────────────────────────────────────────────
@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_hands_test_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db"), tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _two_roles(b):
    ids = list(b.role_node_ids.values())
    return b.registry.get(ids[0]), b.registry.get(ids[1])


def test_every_live_node_has_two_hands(bundle):
    b, _ = bundle
    assert b.registry.count() >= len(AGENT_CATALOGUE)
    for n in b.registry.list_all():
        assert n.hands is not None
        t = n.hands.tools()
        assert {x["name"] for x in t[RIGHT]} == {"send_message", "request_evolution"}
        assert "peers" in {x["name"] for x in t[LEFT]}


def test_left_hand_reads_peers(bundle):
    b, _ = bundle
    a, _other = _two_roles(b)
    res = a.use_hand(LEFT, "peers")
    assert res.ok and any(p["node_id"] == a.node_id for p in res.output)


def test_left_hand_rejects_git_remote_and_bad_glob(bundle):
    b, _ = bundle
    a, _o = _two_roles(b)
    assert "must be one of" in a.use_hand(LEFT, "git_info", what="remote").error
    assert "glob must be" in a.use_hand(LEFT, "search_code", pattern="x", glob="*.json").error


def test_right_hand_send_message_reaches_recipient_inbox(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    res = a.use_hand(RIGHT, "send_message", to_id=other.node_id, topic="hello", payload={"n": 1})
    assert res.ok, res.error
    inbox = b.channel.inbox(other.node_id)
    assert any(m["topic"] == "hello" and m["from_id"] == a.node_id for m in inbox)


def test_send_message_rejects_unknown_recipient_and_huge_payload(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    assert "not a registered node" in a.use_hand(
        RIGHT, "send_message", to_id="ghost", topic="t").error
    assert "exceeds" in a.use_hand(
        RIGHT, "send_message", to_id=other.node_id, topic="t",
        payload={"blob": "x" * 5000}).error


def test_paused_or_failed_node_cannot_act_but_can_resume_and_act(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    a.pause(reason="quarantine")
    assert a.use_hand(RIGHT, "send_message", to_id=other.node_id, topic="t").denied
    a.resume()
    assert a.use_hand(RIGHT, "send_message", to_id=other.node_id, topic="t").ok
    a.mark_failed("x")
    denied = a.use_hand(RIGHT, "send_message", to_id=other.node_id, topic="t")
    assert denied.denied and "cannot act" in denied.error


def test_request_evolution_runs_governed_cycle_then_cools_down(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    first = a.use_hand(RIGHT, "request_evolution")
    assert first.ok, first.error
    second = other.use_hand(RIGHT, "request_evolution")
    assert not second.ok and "cooldown" in second.error


def test_unlisted_right_hand_action_is_rejected_by_policy(bundle):
    b, _ = bundle
    a, _o = _two_roles(b)
    a.hands.bind(RIGHT, "format_disk", lambda: "never")
    res = a.use_hand(RIGHT, "format_disk")
    assert res.denied and "allow-list" in res.error


def test_audit_file_written_in_storage_dir(bundle):
    b, tmp = bundle
    a, other = _two_roles(b)
    a.use_hand(RIGHT, "send_message", to_id=other.node_id, topic="t")
    assert (Path(tmp) / "node_hands_audit.jsonl").exists()


def test_self_evolved_node_registered_later_also_gets_hands(bundle):
    from services.dynamic_node import PassThroughNode

    b, _ = bundle
    born = PassThroughNode(name="born_later", description="d", tags=["self_evolved"])
    b.register_node(born)
    assert born.hands is not None
    assert born.use_hand(LEFT, "peers").ok


# ── نبض ping/pong: العُقد تقرأ بريدها وتردّ بيديها ─────────────────────────
def _ping(b, frm, to, seq=7):
    r = frm.use_hand(RIGHT, "send_message", to_id=to.node_id, topic="ping", payload={"seq": seq})
    assert r.ok, r.error
    return r.output["message_id"]


def test_ping_pong_round_trip_with_reply_to(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    ping_id = _ping(b, a, other, seq=7)
    stats = b.pump_inboxes()
    assert stats["pings_answered"] == 1
    pongs = a.use_hand(LEFT, "read_inbox", topic="pong").output
    assert len(pongs) == 1
    assert pongs[0]["reply_to"] == ping_id and pongs[0]["payload"] == {"echo": 7}
    assert pongs[0]["from_id"] == other.node_id
    assert b.channel.unread_count(other.node_id) == 0 or all(
        m["topic"] != "ping" for m in b.channel.inbox(other.node_id, unread_only=True))


def test_pong_is_never_answered_so_no_loops(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    _ping(b, a, other)
    b.pump_inboxes()
    again = b.pump_inboxes()
    assert again["pings_answered"] == 0
    assert [m["topic"] for m in b.channel.inbox(a.node_id) if m["topic"] == "pong"] == ["pong"]


def test_ping_from_unregistered_sender_or_self_is_dropped_without_reply(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    b.channel.send("ghost", other.node_id, "ping", {})
    b.channel.send(other.node_id, other.node_id, "ping", {})
    stats = b.pump_inboxes()
    assert stats["pings_dropped"] == 2 and stats["pings_answered"] == 0
    assert not [m for m in b.channel.inbox(other.node_id, unread_only=True) if m["topic"] == "ping"]


def test_paused_node_leaves_ping_unread_until_resumed(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    _ping(b, a, other)
    other.pause(reason="quarantine")
    assert b.pump_inboxes()["pings_answered"] == 0
    assert [m for m in b.channel.inbox(other.node_id, unread_only=True) if m["topic"] == "ping"]
    other.resume()
    assert b.pump_inboxes()["pings_answered"] == 1


def test_right_hand_rate_limit_defers_extra_pings_without_losing_them(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    for i in range(7):
        b.channel.send(a.node_id, other.node_id, "ping", {"seq": i})
    stats = b.pump_inboxes()
    assert stats["pings_answered"] == 5 and stats["deferred"] == 1
    left = [m for m in b.channel.inbox(other.node_id, unread_only=True) if m["topic"] == "ping"]
    assert len(left) == 2


def test_informational_messages_stay_unread_for_the_console(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    b.channel.send(a.node_id, other.node_id, "node_failed", {"x": 1})
    b.pump_inboxes()
    assert [m["topic"] for m in b.channel.inbox(other.node_id, unread_only=True)
            if m["topic"] == "node_failed"] == ["node_failed"]


def test_read_inbox_returns_copies_and_only_own_inbox(bundle):
    b, _ = bundle
    a, other = _two_roles(b)
    b.channel.send(a.node_id, other.node_id, "hello", {})
    msgs = other.use_hand(LEFT, "read_inbox", topic="hello").output
    msgs[0]["read"] = True
    assert not [m for m in b.channel.inbox(other.node_id) if m["topic"] == "hello"][0]["read"]
    res = other.use_hand(LEFT, "read_inbox", node_id=a.node_id)
    assert not res.ok and "node_id" in res.error


def test_periodic_hook_pumps_inboxes(bundle, monkeypatch):
    from ai.agent_factory import AgentInstance
    from ai.swarm_coordinator import SwarmTask
    from core.mesh_bundle import EVOLUTION_CYCLE_INTERVAL

    b, _ = bundle
    calls = {"pump": 0}
    monkeypatch.setattr(b, "pump_inboxes", lambda *a, **k: calls.__setitem__("pump", calls["pump"] + 1) or {})
    monkeypatch.setattr(b, "run_evolution_cycle", lambda *a, **k: {})
    role = next(iter(AGENT_CATALOGUE))
    agent = AgentInstance(role, AGENT_CATALOGUE[role])
    b.agent_factory._agents[agent.agent_id] = agent

    class _R:
        pass

    for i in range(EVOLUTION_CYCLE_INTERVAL):
        t = SwarmTask(f"t{i}", "g", "x", {})
        t.assigned_agent_id = agent.agent_id
        t.status = "done"
        t.duration_ms = 1.0
        r = _R()
        r.tasks = [t]
        b.record_swarm_result(r)
    assert calls["pump"] == 1
