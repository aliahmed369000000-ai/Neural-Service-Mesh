"""
اختبار: ai/swarm_manager.py — إعادة كتابة خفيفة سابقة (كوميت 9f7d6cc) حذفت
10 دوال وثلاث خصائص بينما ai/agent_loop.py ما زال يستدعيها فعلياً في
الإنتاج (report_result، create_proposal/check_consensus، set_agent_sleep/
awake، share_media، trigger_reflection)، وحذفت فحص IDS من check_permission
فصار أمر مثل "rm -rf /" يمرّ لأي وكيل يملك صلاحية الكتابة.

يتحقق: الدوال المُعادة تعمل، IDS يحجر الأمر التخريبي ويُحجَر الوكيل،
والإقلاع يبقى خفياً (لا IDS/ذاكرة/محرك عاطفي/عقدة شبكة قبل أول استخدام).
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from ai.swarm_manager import SwarmManager


@pytest.fixture()
def mgr():
    d = tempfile.mkdtemp(prefix="nsm_swarm_restored_")
    try:
        m = SwarmManager(storage_dir=d)
        m.register_worker("w1", "worker", trust_score=0.8)
        m.register_worker("w2", "worker", trust_score=0.8)
        yield m
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_construction_stays_lightweight():
    d = tempfile.mkdtemp(prefix="nsm_swarm_lazy_")
    try:
        m = SwarmManager(storage_dir=d)
        assert m._ids is None and m._memory is None and m._emotional_engine is None
        assert m._mesh_node is None
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_report_result_records_and_updates_trust(mgr):
    before = mgr.workers["w1"]["trust_score"]
    mgr.report_result("w1", "t", "ok", success=True)
    assert mgr.results and mgr.results[-1]["agent_id"] == "w1"
    assert mgr.workers["w1"]["trust_score"] > before
    mgr.report_result("w1", "t2", "boom", success=False)
    assert mgr.results[-1]["success"] is False


def test_proposal_vote_and_consensus(mgr):
    pid = mgr.create_proposal("w1", "deploy", {"k": 1})
    assert pid in mgr.proposals
    res = mgr.cast_vote(pid, "w1", True, "yes")
    assert res["ok"] is True
    # صوت نعم واحد من وكيل نشط = 100% ≥ العتبة فيُعتمد المقترح فوراً ويُغلق
    assert res["consensus"]["status"] == "approved"
    assert mgr.proposals[pid].status == "approved"
    assert mgr.cast_vote(pid, "w2", True, "late")["ok"] is False  # مغلق بالفعل
    rej = mgr.create_proposal("w1", "risky", {})
    mgr.cast_vote(rej, "w1", False, "no")
    assert mgr.proposals[rej].status in ("rejected", "pending")


def test_sleep_and_awake(mgr):
    mgr.set_agent_sleep("w1", "/tmp/snap")
    assert mgr.workers["w1"]["status"] == "sleeping" and "w1" in mgr.sleeping_agents
    mgr.set_agent_awake("w1")
    assert mgr.workers["w1"]["status"] == "active" and "w1" not in mgr.sleeping_agents


def test_check_permission_blocks_destructive_command_and_quarantines(mgr):
    assert mgr.check_permission("w1", "write", params={"command": "ls"}) is True
    assert mgr.check_permission("bad", "write", params={"command": "rm -rf /"}) is False
    assert mgr.ids.is_quarantined("bad") is True
    # الوكيل المحجور يبقى ممنوعاً حتى بفعل بريء
    assert mgr.check_permission("bad", "read", params={}) is False
