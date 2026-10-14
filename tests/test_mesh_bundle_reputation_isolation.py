"""عزل الأعطال في صيانة السمعة (core/mesh_bundle.py):

1. عقدة معطوبة (استثناء أثناء الحجر) لا توقف حجر بقية العقد.
2. فشل خطوة (الحجر) لا يمنع التعافي والتقاعد (_run_reputation_maintenance).
3. ترتيب الحجر: الإيقاف الفعلي أولاً ثم تسجيل الحجر في السمعة، فلو فشل
   الإيقاف تبقى العقدة غير محجورة في السمعة وتُعاد معالجتها في الدورة التالية.
بلا بناء MeshBundle كامل: كائن بديل خفيف يستعير الدوال الحقيقية.
"""
from types import SimpleNamespace

from core.mesh_bundle import MeshBundle
from core.node import NodeState


class _Node:
    def __init__(self, nid, boom=False):
        self.node_id, self.state, self.pause_reason, self.boom = nid, NodeState.ACTIVE, None, boom

    def pause(self, reason="manual"):
        if self.boom:
            raise RuntimeError("pause-boom")
        self.state, self.pause_reason = NodeState.PAUSED, reason


class _Rep:
    def __init__(self, ids):
        self.ids, self.quarantined = ids, []

    def low_reputation_nodes(self, threshold):
        return [{"node_id": i, "total_runs": 9, "is_quarantined": i in self.quarantined,
                 "reputation_score": 1.0} for i in self.ids]

    def quarantine(self, nid):
        self.quarantined.append(nid)


def _fake(nodes, rep):
    reg = SimpleNamespace(get=lambda nid: nodes.get(nid), refresh_meta=lambda nid: None)
    graph = SimpleNamespace(has_node=lambda nid: False)
    ch = SimpleNamespace(broadcast=lambda **kw: None)
    return SimpleNamespace(reputation_engine=rep, registry=reg, graph=graph, channel=ch)


def test_one_broken_node_does_not_block_other_quarantines():
    nodes = {"bad00000": _Node("bad00000", boom=True), "ok000000": _Node("ok000000")}
    rep = _Rep(["bad00000", "ok000000"])
    MeshBundle._apply_reputation_feedback(_fake(nodes, rep))
    assert nodes["ok000000"].state == NodeState.PAUSED
    assert "ok000000" in rep.quarantined


def test_failed_pause_leaves_node_retriable_not_half_quarantined():
    nodes = {"bad00000": _Node("bad00000", boom=True)}
    rep = _Rep(["bad00000"])
    MeshBundle._apply_reputation_feedback(_fake(nodes, rep))
    assert rep.quarantined == []  # لم يُسجَّل حجر وهمي → تُعاد المحاولة لاحقاً
    nodes["bad00000"].boom = False
    MeshBundle._apply_reputation_feedback(_fake(nodes, rep))
    assert nodes["bad00000"].state == NodeState.PAUSED and rep.quarantined == ["bad00000"]


def test_maintenance_steps_are_independent():
    calls = []
    fake = SimpleNamespace()

    def boom():
        calls.append("feedback")
        raise RuntimeError("x")

    fake._apply_reputation_feedback = boom
    fake._apply_reputation_recovery = lambda: calls.append("recovery")
    fake._apply_node_retirement = lambda: calls.append("retirement")
    MeshBundle._run_reputation_maintenance(fake)
    assert calls == ["feedback", "recovery", "retirement"]
