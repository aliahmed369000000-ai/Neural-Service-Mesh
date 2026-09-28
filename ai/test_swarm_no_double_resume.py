"""سرب يعمل فعلاً لا يُعرض ولا يُستأنف ثانيةً (تنفيذ مزدوج)، والاستئناف
التلقائي عند الإقلاع لا يحجز خيط البناء."""
import threading
import time
from pathlib import Path

from ai.agent_factory import AgentFactory
from ai.swarm_coordinator import SwarmCoordinator, SwarmResult, SwarmTask
from ai.swarm_history_store import SwarmHistoryStore


def _coord(tmp_path):
    c = SwarmCoordinator(AgentFactory(), max_agents=2)
    c._store = SwarmHistoryStore(db_path=Path(tmp_path) / "s.db")
    return c


def _checkpoint(c, swarm_id="swarm_x"):
    r = SwarmResult(swarm_id, "goal")
    t = SwarmTask("t1", "sub", "search", {}, 5)
    r.tasks = [t]
    c._store.save_progress(r.to_dict())
    return r


def test_active_swarm_not_listed_and_not_resumable(tmp_path):
    c = _coord(tmp_path)
    _checkpoint(c)
    assert [x["swarm_id"] for x in c.list_resumable()] == ["swarm_x"]
    c._active_swarm_ids.add("swarm_x")   # يعمل الآن
    assert c.list_resumable() == []
    assert c.resume("swarm_x") is None
    assert c._store.get_progress("swarm_x") is not None  # لم تُمسّ نقطة التفتيش


def test_claim_is_exclusive_and_released(tmp_path):
    c = _coord(tmp_path)
    r = _checkpoint(c)
    gate, started, results = threading.Event(), threading.Event(), []

    def slow_inner(result, tasks, **kw):
        started.set(); gate.wait(5)
        return result
    c._execute_tasks_inner = slow_inner

    t = threading.Thread(target=lambda: results.append(c._execute_tasks(r, r.tasks)))
    t.start(); assert started.wait(5)
    assert c._execute_tasks(SwarmResult("swarm_x", "goal"), []) is None  # مكرر مرفوض
    gate.set(); t.join(5)
    assert results[0] is r
    assert "swarm_x" not in c._active_swarm_ids     # حُرّر بعد الانتهاء


def test_claim_released_when_execution_raises(tmp_path):
    c = _coord(tmp_path)
    r = _checkpoint(c)
    def boom(*a, **k):
        raise RuntimeError("x")
    c._execute_tasks_inner = boom
    try:
        c._execute_tasks(r, r.tasks)
    except RuntimeError:
        pass
    assert "swarm_x" not in c._active_swarm_ids


def test_boot_auto_resume_runs_in_background_thread(tmp_path):
    from core.mesh_bundle import MeshBundle
    ran_in = []
    class Coord:
        def list_resumable(self):
            ran_in.append(threading.current_thread().name)
            return []
    stub = type("S", (), {})()
    stub.coordinator = Coord()
    th = threading.Thread(target=lambda: MeshBundle._auto_resume_swarms(stub), name="bg")
    th.start(); th.join(5)
    assert ran_in == ["bg"]


def test_mesh_bundle_init_starts_daemon_thread():
    import inspect
    from core.mesh_bundle import MeshBundle
    src = inspect.getsource(MeshBundle.__init__)
    assert "_auto_resume_swarms" in src and "daemon=True" in src
    assert "self.coordinator.resume(" not in src   # لا استئناف متزامن في __init__
