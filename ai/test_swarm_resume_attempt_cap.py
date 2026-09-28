"""سرب يقتل العملية كل مرة يُستأنف لا يجب أن يُستأنف للأبد (حلقة انهيار)."""
from pathlib import Path

import pytest

from ai.agent_factory import AgentFactory
from ai.swarm_coordinator import SwarmCoordinator, SwarmResult, SwarmTask
from ai.swarm_history_store import SwarmHistoryStore


class ProcessDied(BaseException):
    """يحاكي موت العملية (لا يُلتقط بـ except Exception)."""


def _setup(tmp_path):
    c = SwarmCoordinator(AgentFactory(), max_agents=2)
    c._store = SwarmHistoryStore(db_path=Path(tmp_path) / "s.db")
    r = SwarmResult("swarm_poison", "goal")
    r.tasks = [SwarmTask("t1", "sub", "search", {}, 5)]
    c._store.save_progress(r.to_dict())
    return c


def _crash_on_dispatch(c):
    def die(cap):
        raise ProcessDied()
    c._pick_agent = die


def test_attempts_persist_across_deaths_then_abandoned(tmp_path):
    c = _setup(tmp_path)
    _crash_on_dispatch(c)
    for i in range(1, c.MAX_RESUME_ATTEMPTS + 1):
        with pytest.raises(ProcessDied):
            c.resume("swarm_poison")
        assert c._store.get_progress("swarm_poison")["resume_attempts"] == i
        assert "swarm_poison" not in c._active_swarm_ids   # حُرّر رغم الموت
    assert c.resume("swarm_poison") is None                # المحاولة الرابعة: تخلٍّ
    cp = c._store.get_progress("swarm_poison")
    assert cp["status"] == "abandoned"
    assert c.list_resumable() == []


def test_healthy_resume_completes_and_clears(tmp_path):
    c = _setup(tmp_path)
    res = c.resume("swarm_poison", retry_failed=False)
    assert res is not None and c._store.get_progress("swarm_poison") is None


def test_new_swarm_starts_with_zero_attempts():
    assert SwarmResult("a", "g").to_dict()["resume_attempts"] == 0
    assert SwarmResult.from_checkpoint({"swarm_id": "a", "goal": "g"}).resume_attempts == 0
