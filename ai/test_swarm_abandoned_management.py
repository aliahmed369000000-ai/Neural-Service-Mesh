"""إدارة الأسرِبة المتخلّى عنها: عرض، إعادة تفعيل، حذف — بحماية من الخطأ."""
from pathlib import Path

from ai.agent_factory import AgentFactory
from ai.swarm_coordinator import SwarmCoordinator, SwarmResult, SwarmTask
from ai.swarm_history_store import SwarmHistoryStore


def _coord(tmp_path):
    c = SwarmCoordinator(AgentFactory(), max_agents=2)
    c._store = SwarmHistoryStore(db_path=Path(tmp_path) / "s.db")
    return c


def _save(c, sid, status, attempts=0):
    r = SwarmResult(sid, "goal " + sid)
    r.tasks = [SwarmTask("t", "sub", "search", {}, 5)]
    r.resume_attempts = attempts
    d = r.to_dict(); d["status"] = status
    c._store.save_progress(d)


def test_list_abandoned_only_shows_abandoned(tmp_path):
    c = _coord(tmp_path)
    _save(c, "run", "running"); _save(c, "dead", "abandoned", 3)
    ab = c.list_abandoned()
    assert [a["swarm_id"] for a in ab] == ["dead"] and ab[0]["resume_attempts"] == 3
    assert [x["swarm_id"] for x in c.list_resumable()] == ["run"]


def test_reactivate_resets_attempts_and_makes_resumable(tmp_path):
    c = _coord(tmp_path)
    _save(c, "dead", "abandoned", 3)
    assert c.reactivate("dead") is True
    cp = c._store.get_progress("dead")
    assert cp["status"] == "running" and cp["resume_attempts"] == 0
    assert [x["swarm_id"] for x in c.list_resumable()] == ["dead"]
    assert c.list_abandoned() == []


def test_reactivate_and_discard_refuse_non_abandoned(tmp_path):
    c = _coord(tmp_path)
    _save(c, "run", "running", 1)
    assert c.reactivate("run") is False
    assert c.discard("run") is False               # لا يُحذف سرب قابل للاستئناف
    assert c._store.get_progress("run") is not None
    assert c.reactivate("missing") is False and c.discard("missing") is False


def test_discard_removes_abandoned_only(tmp_path):
    c = _coord(tmp_path)
    _save(c, "dead", "abandoned", 3); _save(c, "run", "running")
    assert c.discard("dead") is True
    assert c._store.get_progress("dead") is None
    assert c._store.get_progress("run") is not None


def test_no_store_is_safe():
    c = SwarmCoordinator(AgentFactory(), max_agents=2)
    c._store = None
    assert c.list_abandoned() == [] and c.reactivate("x") is False and c.discard("x") is False
