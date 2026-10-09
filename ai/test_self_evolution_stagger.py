# -*- coding: utf-8 -*-
"""أول دورة تطوّر ذاتي مبعثرة عشوائياً (لا دفعات متزامنة من كل العقد)."""
import time
from pathlib import Path


def _node(tmp_path, monkeypatch, name):
    import ai.living_mesh as lm
    d = Path(tmp_path)
    (d / "content").mkdir(exist_ok=True)
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    return lm.LivingMeshNode(node_id=name, host="127.0.0.1", port=0)


def test_first_cycle_is_deferred_not_immediate(tmp_path, monkeypatch):
    n = _node(tmp_path, monkeypatch, "stagger_a")
    n.stop_self_evolution_watch()
    n._last_self_evolution_ts = 0.0
    calls = []
    n._execute_evolution = lambda d: calls.append(d)
    n.start_self_evolution_watch(interval_seconds=600)
    try:
        time.sleep(0.5)
        assert calls == []  # لا تنفيذ فوري عند الإقلاع
        remaining = n._self_evolution_interval - (time.time() - n._last_self_evolution_ts)
        assert 29.0 <= remaining <= n._self_evolution_interval
    finally:
        n.stop_self_evolution_watch()


def test_force_still_runs_immediately(tmp_path, monkeypatch):
    n = _node(tmp_path, monkeypatch, "stagger_b")
    n.stop_self_evolution_watch()
    seen = []

    async def fake(d):
        seen.append(d)

    n._execute_evolution = fake
    assert n.maybe_self_evolve(force=True) is True and len(seen) == 1
