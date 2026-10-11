# -*- coding: utf-8 -*-
"""كاشف الأعطال: فحص نشط قبل إعلان الموت (لا إنذارات DEAD كاذبة للعقد الحيّة)."""
import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import ai.living_mesh as lm


def _iso(sec_ago):
    return (datetime.now(timezone.utc) - timedelta(seconds=sec_ago)).isoformat()


@pytest.fixture
def node(tmp_path, monkeypatch):
    d = Path(tmp_path)
    (d / "content").mkdir()
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    n = lm.LivingMeshNode(node_id="me", host="127.0.0.1", port=0)
    return n


def _set_peers(node, peers):
    st = node._load_state()
    for nid, (port, ago, status) in peers.items():
        st["nodes"][nid] = {"host": "127.0.0.1", "port": port, "status": status,
                            "last_seen": _iso(ago)}
    node._save_state(st)


def _stub_ping(node, alive_ports):
    calls = []

    async def ping(host, port, timeout=5.0):
        calls.append(port)
        if port in alive_ports:
            st = node._load_state()
            for info in st["nodes"].values():
                if info.get("port") == port:
                    info["last_seen"] = _iso(0)
                    info["status"] = "online"
            node._save_state(st)
            return {"ok": True}
        return {"ok": False}

    node.ping_peer = ping
    return calls


def test_live_stale_peer_probed_and_not_declared_dead(node):
    _set_peers(node, {"a": (9001, 50, "online")})        # أقدم من 45ث لكنه حيّ
    calls = _stub_ping(node, alive_ports={9001})
    res = asyncio.run(node.probe_stale_peers(stale_after=36))
    assert res == {"suspects": 1, "probed": 1, "alive": 1} and calls == [9001]
    assert node.check_network_health(timeout_seconds=45) == []   # لا إنذار كاذب


def test_truly_dead_peer_still_detected(node):
    _set_peers(node, {"dead": (9002, 50, "online")})
    _stub_ping(node, alive_ports=set())
    res = asyncio.run(node.probe_stale_peers(stale_after=36))
    assert res["alive"] == 0
    assert node.check_network_health(timeout_seconds=45) == ["dead"]


def test_without_probe_old_behaviour_was_false_positive(node):
    _set_peers(node, {"a": (9001, 50, "online")})
    assert node.check_network_health(timeout_seconds=45) == ["a"]  # توثيق الخلل الأصلي


def test_skips_fresh_offline_self_and_incomplete(node):
    _set_peers(node, {"fresh": (9003, 5, "online"), "off": (9004, 99, "offline")})
    st = node._load_state()
    st["nodes"]["noport"] = {"host": "127.0.0.1", "port": None, "status": "online",
                             "last_seen": _iso(99)}
    node._save_state(st)
    calls = _stub_ping(node, alive_ports=set())
    res = asyncio.run(node.probe_stale_peers(stale_after=36))
    assert res["probed"] == 0 and calls == []


def test_oldest_first_and_capped(node):
    _set_peers(node, {f"p{i}": (9100 + i, 40 + i, "online") for i in range(8)})
    calls = _stub_ping(node, alive_ports=set())
    res = asyncio.run(node.probe_stale_peers(stale_after=36, max_probes=3))
    assert res["suspects"] == 8 and res["probed"] == 3
    assert sorted(calls) == [9105, 9106, 9107]   # الأقدم (47..49ث) أولاً


def test_ping_exception_does_not_break(node):
    _set_peers(node, {"a": (9001, 50, "online")})

    async def boom(host, port, timeout=5.0):
        raise RuntimeError("شبكة")

    node.ping_peer = boom
    res = asyncio.run(node.probe_stale_peers(stale_after=36))
    assert res["probed"] == 1 and res["alive"] == 0
