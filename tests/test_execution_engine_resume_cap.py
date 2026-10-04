"""عقدة تقتل العملية في كل استئناف عبر ExecutionEngine.resume_interrupted
لا يجب أن تُستأنف للأبد (نفس علّة حلقة الانهيار المُصلَحة للسرب/المهام)."""
from core.engine import ExecutionEngine, MAX_RESUME_ATTEMPTS
from core.node import BaseNode, NodeSchema, NodeState
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from storage.file_storage import FileStorage


class N(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])
    def process(self, data):
        return {}


def _engine(tmp_path):
    storage = FileStorage(str(tmp_path))
    reg = NodeRegistry(storage)
    graph = ServiceGraph()
    eng = ExecutionEngine(reg, graph, storage)
    return reg, graph, eng


def test_node_abandoned_as_failed_after_max_attempts(tmp_path, monkeypatch):
    reg, graph, eng = _engine(tmp_path)
    n = N("n"); reg.register(n); graph.add_node(n.node_id, n.to_dict())
    n.begin_execution({"x": 1})           # RUNNING + pending_input
    n._resume_attempts = MAX_RESUME_ATTEMPTS
    reg.refresh_meta(n.node_id)

    calls = []
    monkeypatch.setattr(eng, "run_path", lambda ids, data: calls.append(ids))
    results = eng.resume_interrupted()

    assert results == [] and calls == []
    assert n.state == NodeState.FAILED
    assert n._pending_input is None
    persisted = reg.get_meta_by_name("n")
    assert persisted["state"] == "failed"


def test_node_resumes_and_increments_counter(tmp_path, monkeypatch):
    reg, graph, eng = _engine(tmp_path)
    n = N("n"); reg.register(n); graph.add_node(n.node_id, n.to_dict())
    n.begin_execution({"x": 1})
    n._resume_attempts = 1
    reg.refresh_meta(n.node_id)

    calls = []
    monkeypatch.setattr(eng, "run_path", lambda ids, data: calls.append((ids, n._resume_attempts)))
    results = eng.resume_interrupted()

    assert calls == [([n.node_id], 2)]
    persisted = reg.get_meta_by_name("n")
    assert persisted["resume_attempts"] == 2   # محفوظ قبل أي عمل فعلي


def test_successful_execute_resets_counter():
    n = N("n")
    n._resume_attempts = 2
    n.execute({})
    assert n._resume_attempts == 0


def test_round_trip_preserves_counter():
    n = N("n")
    n.begin_execution({"x": 1})
    n._resume_attempts = 5
    snap = n.to_dict()
    assert snap["resume_attempts"] == 5
    n2 = N("n", node_id=n.node_id)
    n2.restore_state(snap)
    assert n2._resume_attempts == 5
