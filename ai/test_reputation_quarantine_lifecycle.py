"""
اختبار دورة حياة الحجر/رفع الحجر الحقيقية:
  1) ai/reputation_engine.py: raw_reputation_score() لا تُصفَّر بالحجر،
     و release_eligible_nodes() تفرّق بين عقدة تعافت فعلياً وأخرى لم تراكم
     تنفيذاً جديداً كافياً أو لم تتحسّن درجتها الحقيقية.
  2) core/mesh_bundle.py: MeshBundle._apply_reputation_feedback يحجر عقدة
     منخفضة السمعة فعلياً على مستوى BaseNode.state (pause حقيقي، وليس فقط
     علم في طبقة السمعة)، و _apply_reputation_recovery يفك الحجر ويستأنف
     العقدة (resume) فعلياً بعد تعافيها.
"""
import tempfile

from ai.reputation_engine import NodeReputationEngine
from core.mesh_bundle import MeshBundle, AgentRoleNode
from core.registry import NodeRegistry
from core.graph import ServiceGraph
from core.node_channel import NodeChannel
from core.node import NodeState
from storage.file_storage import FileStorage


# ── 1) اختبارات وحدة على NodeReputationEngine مباشرة ─────────────────────

def test_raw_score_not_pinned_by_quarantine():
    eng = NodeReputationEngine()
    for _ in range(10):
        eng.record_execution("n1", "Worker", success=True, latency_ms=10.0)
    eng.quarantine("n1")
    rep = eng.get_reputation("n1")
    assert rep.is_quarantined
    assert eng.get_score("n1") == 0.0          # reputation_score مُصفَّرة أثناء الحجر
    assert rep.raw_reputation_score() > 0.0    # لكن الدرجة الحقيقية ما زالت محسوبة


def test_release_eligible_requires_enough_new_runs():
    eng = NodeReputationEngine()
    for _ in range(10):
        eng.record_execution("n1", "Worker", success=False, latency_ms=10.0)
    eng.quarantine("n1")
    # نجاحان فقط بعد الحجر — أقل من الحد الأدنى الافتراضي (5)
    eng.record_execution("n1", "Worker", success=True, latency_ms=10.0)
    eng.record_execution("n1", "Worker", success=True, latency_ms=10.0)
    assert eng.release_eligible_nodes() == []


def test_release_eligible_after_real_recovery():
    eng = NodeReputationEngine()
    for _ in range(10):
        eng.record_execution("n1", "Worker", success=False, latency_ms=10.0)
    eng.quarantine("n1")
    # 20 نجاحاً جديداً كاملاً بزمن استجابة منخفض بعد الحجر
    for _ in range(20):
        eng.record_execution("n1", "Worker", success=True, latency_ms=5.0)
    eligible = eng.release_eligible_nodes(recovery_threshold=60.0, min_new_runs=5)
    assert len(eligible) == 1
    assert eligible[0]["node_id"] == "n1"


def test_unquarantine_clears_runs_at_quarantine():
    eng = NodeReputationEngine()
    eng.record_execution("n1", "Worker", success=True, latency_ms=5.0)
    eng.quarantine("n1")
    assert eng.get_reputation("n1").runs_at_quarantine == 1
    eng.unquarantine("n1")
    assert eng.get_reputation("n1").runs_at_quarantine is None
    assert eng.get_reputation("n1").is_quarantined is False


# ── 2) اختبار تكاملي خفيف على MeshBundle (بدون تركيب MeshBundle كاملة) ──

def _make_mesh_stub(tmp_dir):
    """يبني كائناً بنفس الخصائص التي تستخدمها _apply_reputation_feedback/
    _apply_reputation_recovery فقط (registry, graph, channel,
    reputation_engine) بدل تركيب MeshBundle الكاملة (تحتاج AgentFactory،
    EvolutionEngine، DB... غير ضرورية لهذا الاختبار)."""

    class MeshStub:
        pass

    storage = FileStorage(tmp_dir)
    mesh = MeshStub()
    mesh.registry = NodeRegistry(storage)
    mesh.graph = ServiceGraph()
    mesh.channel = NodeChannel(storage)
    mesh.reputation_engine = NodeReputationEngine()
    return mesh


def test_quarantine_actually_pauses_node_and_recovery_resumes_it():
    with tempfile.TemporaryDirectory() as tmp:
        mesh = _make_mesh_stub(tmp)
        node = AgentRoleNode("ResearchAgent", {"description": "", "tags": [], "capabilities": []})
        node_id = mesh.registry.register(node)
        mesh.graph.add_node(node_id, node.to_dict())

        # 10 فشل متتالٍ -> سمعة منخفضة جداً
        for _ in range(10):
            mesh.reputation_engine.record_execution(node_id, node.name, success=False, latency_ms=10.0)

        MeshBundle._apply_reputation_feedback(mesh)

        rep = mesh.reputation_engine.get_reputation(node_id)
        assert rep.is_quarantined is True
        # التحقق الحقيقي المطلوب: العقدة نفسها أصبحت paused فعلاً، وليس
        # فقط علم "is_quarantined" في طبقة منفصلة
        assert mesh.registry.get(node_id).state == NodeState.PAUSED

        # تعافٍ حقيقي: 20 نجاحاً جديداً بزمن منخفض
        for _ in range(20):
            mesh.reputation_engine.record_execution(node_id, node.name, success=True, latency_ms=5.0)

        MeshBundle._apply_reputation_recovery(mesh)

        rep = mesh.reputation_engine.get_reputation(node_id)
        assert rep.is_quarantined is False
        assert mesh.registry.get(node_id).state == NodeState.ACTIVE


def test_recovery_is_noop_when_no_node_quarantined():
    with tempfile.TemporaryDirectory() as tmp:
        mesh = _make_mesh_stub(tmp)
        # لا عُقد محجورة إطلاقاً -> يجب ألا يفشل شيء ولا يتغيّر شيء
        MeshBundle._apply_reputation_recovery(mesh)
        assert mesh.reputation_engine.quarantined_nodes() == []
