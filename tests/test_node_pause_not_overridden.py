"""pause/quarantine لا يُلغى ضمنياً بنتيجة تنفيذ متأخرة (سباق حقيقي: مهمة
بدأت قبل الحجر وتنتهي بعده ثم تستدعي record_execution)."""
from core.node import BaseNode, NodeSchema, NodeState


class N(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])
    def process(self, data):
        self.pause(reason="quarantine")  # يُحجر أثناء التنفيذ
        return {}


def test_late_success_does_not_unpause():
    n = N("n"); n.pause(reason="quarantine")
    n.record_execution(True)
    assert n.state == NodeState.PAUSED and n.pause_reason == "quarantine"
    assert n.to_dict()["execution_count"] == 1


def test_late_failure_keeps_paused_and_records_error():
    n = N("n"); n.pause(reason="quarantine")
    n.record_execution(False, "boom")
    assert n.state == NodeState.PAUSED and n.pause_reason == "quarantine"
    assert n.to_dict()["last_error"] == "boom"


def test_pause_during_process_survives_success():
    n = N("n")
    n.execute({})
    assert n.state == NodeState.PAUSED and n.pause_reason == "quarantine"


def test_normal_paths_unchanged():
    n = N("n"); n.record_execution(False, "x")
    assert n.state == NodeState.FAILED
    n.record_execution(True)
    assert n.state == NodeState.ACTIVE
    n.pause(); n.resume()
    assert n.state == NodeState.ACTIVE and n.pause_reason is None
