from ai.node_health_layer import NodeHealthLayer, TASK_LOG_MAX, TASK_JOURNAL_KEY


class FakeNode:
    def __init__(self):
        self.state = {"nodes": {}, TASK_JOURNAL_KEY: [{"task_id": "old"}]}

    def _load_state(self):
        return dict(self.state)

    def _save_state(self, state):
        self.state = state


def test_task_log_restores_and_persists():
    node = FakeNode()
    layer = NodeHealthLayer(node)
    assert layer.recent_tasks() == [{"task_id": "old"}]
    layer._log_task({"task_id": "new"})
    assert node.state[TASK_JOURNAL_KEY][-1] == {"task_id": "new"}


def test_task_log_is_bounded():
    node = FakeNode()
    layer = NodeHealthLayer(node)
    for i in range(TASK_LOG_MAX + 10):
        layer._log_task({"task_id": str(i)})
    assert len(layer._task_log) == TASK_LOG_MAX
    assert layer._task_log[0]["task_id"] == "10"
