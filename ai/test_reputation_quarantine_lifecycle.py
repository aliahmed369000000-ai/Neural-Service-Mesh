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
from storage.db import SQLiteStorage


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

def _make_mesh_stub(tmp_dir, storage=None):
    """يبني كائناً بنفس الخصائص التي تستخدمها _apply_reputation_feedback/
    _apply_reputation_recovery فقط (registry, graph, channel,
    reputation_engine) بدل تركيب MeshBundle الكاملة (تحتاج AgentFactory،
    EvolutionEngine، DB... غير ضرورية لهذا الاختبار). storage قابل
    للتمرير من الخارج (نفس الكائن) لمحاكاة 'إعادة تشغيل' حقيقية تشارك
    نفس التخزين الدائم بين مثيلَي MeshStub متتاليين."""

    class MeshStub:
        pass

    storage = storage if storage is not None else FileStorage(tmp_dir)
    mesh = MeshStub()
    mesh.registry = NodeRegistry(storage)
    mesh.graph = ServiceGraph()
    mesh.channel = NodeChannel(storage)
    mesh.reputation_engine = NodeReputationEngine(storage=storage)
    mesh.exec_log = SQLiteStorage(db_path=str(tmp_dir) + "/exec_log_stub.db")
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


# ── 3) استمرارية الحجر عبر إعادة التشغيل (توقف مفاجئ) ────────────────────
# قبل هذا: NodeReputationEngine._reputations كان قاموساً في الذاكرة فقط —
# لا شيء يحفظه أو يستعيده. BaseNode.state يُستعاد PAUSED فعلاً بعد إعادة
# التشغيل (core/registry.py)، لكن NodeReputationEngine يبدأ بذاكرة فارغة
# (is_quarantined=False افتراضياً)، فـ_apply_reputation_recovery لا يرى
# العقدة محجورة إطلاقاً ولا يستطيع أبداً استئنافها — عقدة عالقة في PAUSED
# للأبد رغم تحسّن أدائها الفعلي لاحقاً. هذا الاختبار يحاكي التوقف المفاجئ
# فعلياً: يبني NodeReputationEngine جديداً (كأنه بعد إعادة تشغيل العملية)
# بنفس storage، ويتحقق أن الحجر يُستعاد صحيحاً فيمكن رفعه بعد ذلك.

def test_quarantine_survives_simulated_restart():
    with tempfile.TemporaryDirectory() as tmp:
        storage = FileStorage(tmp)
        eng1 = NodeReputationEngine(storage=storage)
        for _ in range(10):
            eng1.record_execution("n1", "Worker", success=False, latency_ms=10.0)
        eng1.quarantine("n1")
        assert eng1.get_reputation("n1").is_quarantined is True

        # 🆕 محاكاة توقف مفاجئ وإعادة تشغيل: محرك جديد تماماً بنفس storage
        eng2 = NodeReputationEngine(storage=storage)
        rep2 = eng2.get_reputation("n1")
        assert rep2 is not None, "الحجر يجب أن يُستعاد من التخزين الدائم بعد إعادة التشغيل"
        assert rep2.is_quarantined is True
        assert rep2.runs_at_quarantine == 10
        assert rep2.total_runs == 10

        # والتعافي يعمل طبيعياً بعد الاستعادة (لم يعد عالقاً للأبد)
        for _ in range(20):
            eng2.record_execution("n1", "Worker", success=True, latency_ms=5.0)
        eligible = eng2.release_eligible_nodes(recovery_threshold=60.0, min_new_runs=5)
        assert len(eligible) == 1 and eligible[0]["node_id"] == "n1"


def test_quarantine_persists_across_mesh_bundle_restart():
    """اختبار تكاملي: pause فعلي على BaseNode + حجر في NodeReputationEngine
    كلاهما يُستعاد بشكل متسق بعد 'إعادة تشغيل' (registry/reputation_engine
    جديدان بنفس storage) — لا عقدة عالقة PAUSED بلا مسار رجوع."""
    with tempfile.TemporaryDirectory() as tmp:
        storage = FileStorage(tmp)
        mesh1 = _make_mesh_stub(tmp, storage=storage)
        node = AgentRoleNode("ResearchAgent", {"description": "", "tags": [], "capabilities": []})
        node_id = mesh1.registry.register(node)
        mesh1.graph.add_node(node_id, node.to_dict())
        for _ in range(10):
            mesh1.reputation_engine.record_execution(node_id, node.name, success=False, latency_ms=10.0)
        MeshBundle._apply_reputation_feedback(mesh1)
        assert mesh1.registry.get(node_id).state == NodeState.PAUSED

        # "إعادة تشغيل": registry وreputation_engine جديدان تماماً، نفس storage
        mesh2 = _make_mesh_stub(tmp, storage=storage)
        persisted = mesh2.registry.get_meta_by_name("ResearchAgent")
        assert persisted is not None
        # نفس النمط الحقيقي في MeshBundle._register_roles: تمرير node_id
        # المحفوظ صراحة عند البناء، وليس تركه يولّد معرّفاً عشوائياً جديداً
        restored_node = AgentRoleNode(
            "ResearchAgent", {"description": "", "tags": [], "capabilities": []},
            node_id=persisted.get("node_id"),
        )
        restored_node.restore_state(persisted)
        restored_id = mesh2.registry.register(restored_node, overwrite=True)
        mesh2.graph.add_node(restored_id, restored_node.to_dict())
        assert restored_node.state == NodeState.PAUSED  # كما كان قبل الاختبار الأصلي

        rep = mesh2.reputation_engine.get_reputation(node_id)
        assert rep is not None and rep.is_quarantined is True  # 🆕 الحجر استُعيد أيضاً

        for _ in range(20):
            mesh2.reputation_engine.record_execution(node_id, node.name, success=True, latency_ms=5.0)
        MeshBundle._apply_reputation_recovery(mesh2)
        assert mesh2.reputation_engine.get_reputation(node_id).is_quarantined is False
        assert mesh2.registry.get(restored_id).state == NodeState.ACTIVE


# ── 4) تقاعد دائم لعُقد ai-generated محجورة بشكل مزمن ─────────────────────

def test_tick_quarantine_checks_only_increments_quarantined_nodes():
    eng = NodeReputationEngine()
    eng.record_execution("n1", "Worker", success=False, latency_ms=10.0)
    eng.record_execution("n2", "Worker", success=True, latency_ms=5.0)
    eng.quarantine("n1")  # n2 يبقى غير محجور
    eng.tick_quarantine_checks()
    eng.tick_quarantine_checks()
    assert eng.get_reputation("n1").quarantine_checks == 2
    assert eng.get_reputation("n2").quarantine_checks == 0


def test_retirement_eligible_requires_min_checks():
    eng = NodeReputationEngine()
    eng.record_execution("n1", "Worker", success=False, latency_ms=10.0)
    eng.quarantine("n1")
    for _ in range(4):
        eng.tick_quarantine_checks()
    assert eng.retirement_eligible_nodes(min_checks=15) == []
    for _ in range(11):
        eng.tick_quarantine_checks()
    eligible = eng.retirement_eligible_nodes(min_checks=15)
    assert len(eligible) == 1 and eligible[0]["node_id"] == "n1"


def test_recovery_resets_quarantine_checks_preventing_retirement():
    """عقدة تتعافى قبل بلوغ عتبة التقاعد يجب ألا تُتقاعد لاحقاً حتى لو
    حُجرت واستمر العدّاد لاحقاً من الصفر مجدداً."""
    eng = NodeReputationEngine()
    eng.record_execution("n1", "Worker", success=False, latency_ms=10.0)
    eng.quarantine("n1")
    for _ in range(10):
        eng.tick_quarantine_checks()
    eng.unquarantine("n1")
    assert eng.get_reputation("n1").quarantine_checks == 0


def _make_ai_generated_node(name="GeneratedWorker"):
    return AgentRoleNode(
        name, {"description": "", "tags": ["ai-generated", "phase5"], "capabilities": []},
    )


def test_apply_node_retirement_removes_chronically_quarantined_generated_node():
    with tempfile.TemporaryDirectory() as tmp:
        mesh = _make_mesh_stub(tmp)
        node = _make_ai_generated_node()
        node_id = mesh.registry.register(node)
        mesh.graph.add_node(node_id, node.to_dict())
        root = AgentRoleNode("RootAgent", {"description": "", "tags": ["agent_role"], "capabilities": []})
        root_id = mesh.registry.register(root)
        mesh.graph.add_node(root_id, root.to_dict())
        mesh.graph.add_edge(root_id, node_id, label="self_evolved")
        mesh.exec_log.upsert_node(node.to_dict())
        mesh.exec_log.upsert_connection(root_id, node_id, label="self_evolved")

        for _ in range(10):
            mesh.reputation_engine.record_execution(node_id, node.name, success=False, latency_ms=10.0)
        MeshBundle._apply_reputation_feedback(mesh)
        assert mesh.registry.get(node_id).state == NodeState.PAUSED

        for _ in range(15):
            MeshBundle._apply_node_retirement(mesh)

        assert mesh.registry.get(node_id) is None, "يجب أن تُحذف العقدة نهائياً من الـregistry"
        assert mesh.graph.has_node(node_id) is False, "ويجب أن تُحذف من الرسم البياني"
        remaining = [c for c in mesh.exec_log.list_connections() if c["target_id"] == node_id]
        assert remaining == [], "ويجب تنظيف الرابط المحفوظ في exec_log"
        assert mesh.exec_log.get_node(node_id) is None, "وسجلّ العقدة نفسه في exec_log"
        # الجذر نفسه لم يُمس
        assert mesh.registry.get(root_id) is not None


def test_apply_node_retirement_never_removes_non_generated_node():
    """عقدة كتالوج أساسية (بلا وسم ai-generated) يجب أن تبقى محجورة (paused)
    مهما طال أمد الحجر — لا تُحذف أبداً."""
    with tempfile.TemporaryDirectory() as tmp:
        mesh = _make_mesh_stub(tmp)
        node = AgentRoleNode("ResearchAgent", {"description": "", "tags": [], "capabilities": []})
        node_id = mesh.registry.register(node)
        mesh.graph.add_node(node_id, node.to_dict())

        for _ in range(10):
            mesh.reputation_engine.record_execution(node_id, node.name, success=False, latency_ms=10.0)
        MeshBundle._apply_reputation_feedback(mesh)

        for _ in range(20):
            MeshBundle._apply_node_retirement(mesh)

        assert mesh.registry.get(node_id) is not None
        assert mesh.registry.get(node_id).state == NodeState.PAUSED
        assert mesh.graph.has_node(node_id) is True


def test_apply_node_retirement_is_noop_before_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        mesh = _make_mesh_stub(tmp)
        node = _make_ai_generated_node()
        node_id = mesh.registry.register(node)
        mesh.graph.add_node(node_id, node.to_dict())

        for _ in range(10):
            mesh.reputation_engine.record_execution(node_id, node.name, success=False, latency_ms=10.0)
        MeshBundle._apply_reputation_feedback(mesh)

        for _ in range(5):  # أقل من العتبة الافتراضية (15)
            MeshBundle._apply_node_retirement(mesh)

        assert mesh.registry.get(node_id) is not None
        assert mesh.graph.has_node(node_id) is True
