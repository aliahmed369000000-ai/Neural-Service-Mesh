"""
اختبار: core/node.py — BaseNode.pause()/resume() كانت تستخدم state=PAUSED
واحد بلا أي تمييز لسببه. MeshBundit._apply_reputation_feedback يستدعي
node.pause() للحجر التلقائي بسبب سمعة منخفضة، و_apply_reputation_recovery
يستدعي node.resume() تلقائياً عند التعافي — لو أوقف إنسان نفس العقدة
يدوياً بعد حجرها (لسبب مختلف تماماً)، كان تعافي السمعة سيُعيد تشغيلها
رغماً عنه لأن كلا الحالتين PAUSED متطابقتان تماماً من منظور BaseNode.

هذا الاختبار يتحقق:
1. pause(reason=...) يسجّل السبب في pause_reason.
2. resume(expected_reason=...) يرفض الاستئناف (يرجع False، لا تغيير
   بالحالة) إن كان السبب الفعلي مختلفاً — يحاكي حماية عقدة أوقفها إنسان
   يدوياً من استئناف تلقائي بسبب تعافي السمعة.
3. resume(expected_reason=...) ينجح فعلياً حين يتطابق السبب.
4. resume() بلا expected_reason (الاستدعاء المباشر القديم) يستمر يعمل
   كما كان — لا تراجع بالسلوك السابق.
5. to_dict()/restore_state() يحفظان ويسترجعان pause_reason.
"""
from __future__ import annotations

from core.node import BaseNode, NodeSchema, NodeState


class _EchoNode(BaseNode):
    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={}, required=[])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={}, required=[])

    def process(self, data):
        return data


def test_pause_records_reason():
    node = _EchoNode("n")
    node.pause(reason="quarantine")
    assert node.state == NodeState.PAUSED
    assert node.pause_reason == "quarantine"


def test_resume_expected_reason_mismatch_is_rejected():
    node = _EchoNode("n")
    node.pause(reason="manual")  # إنسان أوقفها يدوياً
    resumed = node.resume(expected_reason="quarantine")  # تعافي سمعة تلقائي
    assert resumed is False
    assert node.state == NodeState.PAUSED
    assert node.pause_reason == "manual"


def test_resume_expected_reason_match_succeeds():
    node = _EchoNode("n")
    node.pause(reason="quarantine")
    resumed = node.resume(expected_reason="quarantine")
    assert resumed is True
    assert node.state == NodeState.ACTIVE
    assert node.pause_reason is None


def test_resume_without_expected_reason_is_unconditional():
    node = _EchoNode("n")
    node.pause()  # سلوك قديم: بلا سبب صريح
    resumed = node.resume()
    assert resumed is True
    assert node.state == NodeState.ACTIVE


def test_pause_reason_round_trips_through_to_dict_and_restore_state():
    node = _EchoNode("n")
    node.pause(reason="manual")
    snapshot = node.to_dict()
    assert snapshot["pause_reason"] == "manual"

    restored = _EchoNode("n", node_id=node.node_id)
    restored.restore_state(snapshot)
    assert restored.state == NodeState.PAUSED
    assert restored.pause_reason == "manual"


def test_pause_reason_cleared_when_restored_state_is_not_paused():
    node = _EchoNode("n")
    node.pause(reason="manual")
    node.resume()
    snapshot = node.to_dict()
    assert snapshot["pause_reason"] is None

    restored = _EchoNode("n", node_id=node.node_id)
    restored.restore_state(snapshot)
    assert restored.state == NodeState.ACTIVE
    assert restored.pause_reason is None
