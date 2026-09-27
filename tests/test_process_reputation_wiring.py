"""
اختبار MeshBundle.record_direct_execution / snapshot_node_states
====================================================================
المشكلة: core.engine.ExecutionEngine.run_path (مسار /process المباشر)
تُنفِّذ عُقداً حقيقية وتُحدِّث BaseNode.state مباشرة (عبر node.execute()
نفسها) — لكن NodeReputationEngine لم يكن يعرف عن أي تنفيذ عبر /process
شيئاً على الإطلاق، لأن التغذية الوحيدة لسمعة العقدة (في
MeshBundle.record_swarm_result) مصدرها حصرياً AgentFactory.run_task عبر
مسار السرب — مسار منفصل تماماً. الأثر العملي: عقدة تُستدعى مباشرة عبر
/process وتفشل مراراً لن تُحجَر (quarantine) أبداً، ولن يُبلَّغ جيرانها
في الرسم البياني (node_failed/node_recovered)، ولا تُطبَّق دورة الحجر/
رفع الحجر التلقائية (_apply_reputation_feedback/_apply_reputation_recovery)
على نتائج /process إطلاقاً.

يتحقق هذا الملف عبر MeshBundle حقيقية (مجلد بيانات مؤقت) وExecutionEngine
حقيقي (لا محاكاة):
1. فشل حقيقي متكرر (5 مرات، use_fallback=False لعزل الاختبار عن fallback)
   لعقدة عبر /process → يُحجَر فعلياً (is_quarantined=True، node.state
   يصبح PAUSED) — لم يكن هذا يحدث إطلاقاً قبل هذا التعديل.
2. بعد الحجر واستئناف يدوي (يحاكي تعافياً حقيقياً مُكتشَفاً بوسيلة أخرى)،
   تراكم نجاحات حقيقية كافية عبر /process يرفع raw_reputation_score فوق
   عتبة التعافي فعلياً ويُلغي الحجر (is_quarantined=False).
3. انتقال حالة حقيقي (نشطة→فاشلة) يُبَث فعلاً كرسالة "node_failed" لجار
   حقيقي في الرسم البياني عبر NodeChannel — لا تكرار بث لنفس الحالة عبر
   استدعاءات متتالية بلا انتقال فعلي (لا إغراق للقناة).
4. عقدة غير موجودة في المسجّل (step.node_id لا يطابق أي عقدة حقيقية) لا
   تُسجَّل في السمعة إطلاقاً (لا معنى لسمعة عقدة لا وجود لها) ولا ترفع
   استثناء.
"""
from __future__ import annotations

import shutil
import tempfile

from core.engine import ExecutionEngine
from core.mesh_bundle import MeshBundle
from core.node import NodeState


def _build_bundle(tmp_dir):
    return MeshBundle(storage_dir=tmp_dir, db_path=f"{tmp_dir}/mesh.db")


def test_repeated_real_failure_via_process_quarantines_node():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_reputation_wiring_test1_")
    try:
        b = _build_bundle(tmp_dir)
        root_id = b._root_node_id
        other_role, other_id = next(
            (r, n) for r, n in b.role_node_ids.items() if n != root_id
        )
        node = b.registry.get(other_id)
        node.process = lambda data: (_ for _ in ()).throw(RuntimeError("فشل متعمَّد للاختبار"))

        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)
        for _ in range(5):
            prev = b.snapshot_node_states()
            result = engine.run_path([other_id], {"task": "x"}, use_fallback=False)
            b.record_direct_execution(result, prev)

        rep = b.reputation_engine.get_reputation(other_id)
        assert rep is not None and rep.total_runs == 5
        assert rep.is_quarantined is True, "كان يجب أن تُحجَر بعد فشل متكرر عبر /process"
        assert node.state == NodeState.PAUSED
        print("OK: repeated real /process failures actually quarantine the node")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_recovery_after_quarantine_via_process_successes():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_reputation_wiring_test2_")
    try:
        b = _build_bundle(tmp_dir)
        root_id = b._root_node_id
        other_role, other_id = next(
            (r, n) for r, n in b.role_node_ids.items() if n != root_id
        )
        node = b.registry.get(other_id)
        node.process = lambda data: (_ for _ in ()).throw(RuntimeError("فشل"))

        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)
        for _ in range(5):
            prev = b.snapshot_node_states()
            result = engine.run_path([other_id], {"task": "x"}, use_fallback=False)
            b.record_direct_execution(result, prev)

        rep = b.reputation_engine.get_reputation(other_id)
        assert rep.is_quarantined is True

        # تعافٍ حقيقي: العقدة تعمل الآن بشكل صحيح، استئناف يدوي يحاكي
        # اكتشاف تعافٍ (كما تفعل _apply_reputation_recovery فعلياً عبر
        # node.resume(expected_reason="quarantine"))
        node.process = lambda data: {"result": "تمّ فعلاً"}
        node.resume(expected_reason="quarantine")

        for _ in range(30):
            prev = b.snapshot_node_states()
            result = engine.run_path([other_id], {"task": "x"}, use_fallback=False)
            b.record_direct_execution(result, prev)
            rep = b.reputation_engine.get_reputation(other_id)
            if not rep.is_quarantined:
                break

        assert rep.is_quarantined is False, "لم يُرفَع الحجر رغم نجاحات كافية عبر /process"
        print("OK: quarantine lifted after enough real /process successes")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_state_transition_broadcasts_node_failed_to_real_neighbor():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_reputation_wiring_test3_")
    try:
        b = _build_bundle(tmp_dir)
        root_id = b._root_node_id
        other_role, other_id = next(
            (r, n) for r, n in b.role_node_ids.items() if n != root_id
        )
        node = b.registry.get(other_id)
        # حافة صادرة حقيقية من العقدة نحو الجذر (كي يكون للجذر جاراً يبثّ
        # له عند فشل هذه العقدة تحديداً — عكس الاتجاه الافتراضي جذر→أدوار)
        b.graph.add_edge(other_id, root_id)
        node.process = lambda data: (_ for _ in ()).throw(RuntimeError("فشل"))

        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)
        prev = b.snapshot_node_states()
        result = engine.run_path([other_id], {"task": "x"}, use_fallback=False)
        assert result.steps[0].node_id == other_id and result.steps[0].status == "error"
        b.record_direct_execution(result, prev)

        msgs = b.channel.recent(10)
        assert len(msgs) == 1, msgs
        assert msgs[0]["topic"] == "node_failed"
        assert msgs[0]["from_id"] == other_id
        assert msgs[0]["to_id"] == root_id

        # فشل ثانٍ بلا انتقال فعلي (تبقى FAILED→FAILED): لا بث إضافي
        prev2 = b.snapshot_node_states()
        result2 = engine.run_path([other_id], {"task": "x"}, use_fallback=False)
        b.record_direct_execution(result2, prev2)
        assert len(b.channel.recent(10)) == 1, "لا يجب بث node_failed مرة أخرى بلا انتقال حالة جديد"
        print("OK: node_failed broadcast fires once on real transition, not on repeat")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_unknown_node_id_is_skipped_safely():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_reputation_wiring_test4_")
    try:
        b = _build_bundle(tmp_dir)
        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)
        prev = b.snapshot_node_states()
        result = engine.run_path(["does-not-exist"], {"task": "x"}, use_fallback=False)
        b.record_direct_execution(result, prev)  # يجب ألا يرفع استثناءً
        assert b.reputation_engine.get_reputation("does-not-exist") is None
        print("OK: unknown node id skipped safely, no phantom reputation entry")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── اختبار HTTP كامل عبر api_server.py::/process الحقيقي (لا استدعاء مباشر
# لـMeshBundle.record_direct_execution) — يثبت أن /process نفسها، لا فقط
# الدالة المعزولة، تربط كل طلب فعلي بنظام السمعة/الحجر. ──────────────────
import pytest


@pytest.fixture()
def isolated_bundle(monkeypatch):
    tmp_dir = tempfile.mkdtemp(prefix="nsm_reputation_wiring_http_test_")
    bundle = _build_bundle(tmp_dir)
    monkeypatch.setattr("core.mesh_bundle.get_mesh_bundle", lambda: bundle)
    yield bundle
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("NSM_API_KEY", "nsm-test-key")
    import api_server
    from fastapi.testclient import TestClient
    return TestClient(api_server.app)


def test_process_endpoint_itself_quarantines_node_after_real_failures(client, isolated_bundle):
    root_id = isolated_bundle._root_node_id
    other_role, other_id = next(
        (r, n) for r, n in isolated_bundle.role_node_ids.items() if n != root_id
    )
    node = isolated_bundle.registry.get(other_id)
    node.process = lambda data: (_ for _ in ()).throw(RuntimeError("فشل عبر HTTP"))

    for _ in range(5):
        r = client.post(
            "/process",
            json={"path": [other_id], "data": {"task": "x"}},
            headers={"x-api-key": "nsm-test-key"},
        )
        assert r.status_code == 200
        assert r.json()["result"]["status"] == "failed"

    rep = isolated_bundle.reputation_engine.get_reputation(other_id)
    assert rep is not None and rep.total_runs == 5
    assert rep.is_quarantined is True, "كان يجب أن يُحجَر بعد فشل متكرر عبر /process الحقيقي"
    print("OK: /process endpoint itself (full HTTP) now quarantines a repeatedly-failing node")


if __name__ == "__main__":
    test_repeated_real_failure_via_process_quarantines_node()
    test_recovery_after_quarantine_via_process_successes()
    test_state_transition_broadcasts_node_failed_to_real_neighbor()
    test_unknown_node_id_is_skipped_safely()
    print("جميع اختبارات ربط /process بنظام السمعة/الحجر نجحت")
