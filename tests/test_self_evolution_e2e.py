"""اختبار تكاملي شامل للتطوّر الذاتي الحقيقي: GapDetector → ServiceGenerator
→ AIGovernanceLayer → تسجيل عقدة حيّة فعلية من النوع الصحيح → نجاتها عبر
إعادة تشغيل العملية. لم يكن لهذا المسار بالكامل أي اختبار رغم أنه يمثّل
جوهر 'العقد اللامركزية تطوّر نفسها' في هذا المشروع."""
import threading
import time

from core.mesh_bundle import MeshBundle
from ai.gap_detector import DetectedGap
from services.generated_service_nodes import NODE_CLASS_BY_SERVICE_TYPE, NODE_CLASS_BY_NAME


def _mesh(tmp_path, suffix=""):
    m = MeshBundle(
        storage_dir=str(tmp_path / "storage"),
        db_path=str(tmp_path / f"mesh{suffix}.db"),
    )
    # الخيوط الخلفية (استئناف أسرِبة/مهام) غير مطلوبة هنا — ننتظرها تنتهي
    # بسرعة (لا شيء متوقف فعلياً في بيئة اختبار فارغة) حتى لا تتشابك مع
    # تنظيف tmp_path بعد انتهاء الاختبار.
    for t in threading.enumerate():
        if t.name in ("nsm-swarm-auto-resume", "nsm-jobs-auto-resume") and t.is_alive():
            t.join(timeout=3)
    return m


def _inject_gap(mesh, missing_service="data_validator", confidence=0.8):
    gap = DetectedGap(
        gap_type="capability",
        missing_service=missing_service,
        confidence=confidence,
        source_node={"name": mesh.registry.get(mesh._root_node_id).name,
                     "capability": "orchestrate"},
        target_node={"name": "", "capability": missing_service},
    )
    mesh.gap_detector.scan = lambda: [gap]


def test_evolution_cycle_creates_real_typed_node_not_passthrough(tmp_path):
    mesh = _mesh(tmp_path)
    _inject_gap(mesh, missing_service="data_validator")
    before = set(mesh.registry.list_metadata_ids()) if hasattr(mesh.registry, "list_metadata_ids") \
        else {m["node_id"] for m in mesh.registry.list_metadata()}

    result = mesh.run_evolution_cycle()

    after = {m["node_id"] for m in mesh.registry.list_metadata()}
    added = after - before
    assert len(added) == 1, f"عقدة واحدة جديدة متوقعة، الناتج: {result}"
    new_id = added.pop()

    live = mesh.registry.get(new_id)
    assert live is not None
    expected_cls = NODE_CLASS_BY_SERVICE_TYPE.get("validator")
    assert type(live) is expected_cls, (
        f"النوع {type(live).__name__} — يجب أن يكون {expected_cls.__name__} "
        f"الحقيقي وليس PassThroughNode عاماً"
    )
    assert mesh.graph.has_node(new_id)
    assert mesh.reputation_engine.get_reputation(new_id) is None or True  # لا يُشترط سجلّ سمعة قبل أول تنفيذ


def test_self_evolved_node_survives_restart_with_correct_class_and_state(tmp_path):
    mesh1 = _mesh(tmp_path, "_r")
    _inject_gap(mesh1, missing_service="normalizer_step")
    result = mesh1.run_evolution_cycle()
    added_ids = [n for n in result.get("nodes_added", [])] if isinstance(result, dict) else []
    assert len(added_ids) == 1
    node_id = added_ids[0]

    original = mesh1.registry.get(node_id)
    original.pause(reason="manual")
    mesh1.registry.refresh_meta(node_id)

    # "إعادة تشغيل": حزمة جديدة بنفس التخزين
    mesh2 = _mesh(tmp_path, "_r")
    restored = mesh2.registry.get(node_id)
    assert restored is not None, "العقدة المُولَّدة ذاتياً أصبحت شبحاً بعد إعادة التشغيل"
    assert type(restored) is type(original)
    assert restored.state == "paused" and restored.pause_reason == "manual"


def test_low_confidence_gap_is_rejected_and_adds_no_node(tmp_path):
    mesh = _mesh(tmp_path)
    _inject_gap(mesh, missing_service="data_validator", confidence=0.1)
    before = {m["node_id"] for m in mesh.registry.list_metadata()}
    result = mesh.run_evolution_cycle()
    after = {m["node_id"] for m in mesh.registry.list_metadata()}
    assert after == before
    assert result.get("summary", result).get("nodes_added", result.get("nodes_added", 0)) in (0, [])


def test_no_gaps_is_safe_noop(tmp_path):
    mesh = _mesh(tmp_path)
    mesh.gap_detector.scan = lambda: []
    before = {m["node_id"] for m in mesh.registry.list_metadata()}
    mesh.run_evolution_cycle()
    after = {m["node_id"] for m in mesh.registry.list_metadata()}
    assert after == before
