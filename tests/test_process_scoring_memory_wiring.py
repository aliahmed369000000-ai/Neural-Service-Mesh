"""
اختبار MeshBundle.record_direct_execution — الجزء الخاص بـScoringEngine/
MemoryEngine (امتداد لنفس الدالة التي تربط /process بالسمعة، راجع
tests/test_process_reputation_wiring.py للجزء الخاص بالسمعة/الحجر/البث).

المشكلة: ai/scoring_engine.py::ScoringEngine.record_run وai/memory_engine.py
::MemoryEngine.learn_from_run كلاهما مصمَّم أصلاً ليأخذ run_result بنفس
الشكل الذي يُنتجه core.engine.ExecutionResult.to_dict() بالضبط (path/
steps/status/total_duration_ms) — لكنهما لم يكونا يُستدعَيان إلا من
MeshBundle.record_swarm_result (مسار السرب فقط، عبر run_result مُصنَّع
يدوياً هناك). أي نتيجة /process حقيقية (عبر core.engine.ExecutionEngine
مباشرة) لم تكن تصل لهما إطلاقاً — درجات الحواف (ScoringEngine) وذاكرة
المسارات/العُقد (MemoryEngine) بقيتا عمياوين تماماً تجاه نصف مسارات
التنفيذ في المشروع.

يتحقق هذا الملف عبر MeshBundle+ExecutionEngine حقيقيتين (لا محاكاة):
1. مسار من خطوة واحدة عبر /process (استدعاء مباشر لـrecord_direct_execution،
   نفس ما يفعله api_server.py فعلياً) يزيد MemoryEngine._nodes[node_id]
   .executions فعلاً.
2. مسار من خطوتين (حافة حقيقية src→tgt) يحدّث ScoringEngine.get_score
   لتلك الحافة بالضبط (total_runs+=1)، ويضيف RouteMemory بمفتاح المسار.
3. تشغيل الدالة الفارغة (بلا خطوات، نتيجة "لا مسار") لا يرفع استثناءً —
   run_result بلا path حقيقي يُتجاهَل بأمان (نفس سلوك learn_from_run
   الأصلي لمسار فارغ).
"""
from __future__ import annotations

import shutil
import tempfile

from core.engine import ExecutionEngine
from core.mesh_bundle import MeshBundle


def _build_bundle(tmp_dir):
    return MeshBundle(storage_dir=tmp_dir, db_path=f"{tmp_dir}/mesh.db")


def test_single_hop_process_updates_node_memory():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_scoring_memory_wiring_test1_")
    try:
        b = _build_bundle(tmp_dir)
        root_id = b._root_node_id
        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)

        prev = b.snapshot_node_states()
        result = engine.run_path([root_id], {"task": "x"})
        assert result.status == "success"
        b.record_direct_execution(result, prev)

        node_mem = b.memory_engine._nodes.get(root_id)
        assert node_mem is not None and node_mem.executions == 1
        print("OK: single-hop /process run updates MemoryEngine node stats")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_two_hop_process_updates_edge_score_and_route_memory():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_scoring_memory_wiring_test2_")
    try:
        b = _build_bundle(tmp_dir)
        root_id = b._root_node_id
        other_role, other_id = next(
            (r, n) for r, n in b.role_node_ids.items() if n != root_id
        )
        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)

        prev = b.snapshot_node_states()
        # المسار سيفشل عند الخطوة الثانية (فجوة مخطط بيانات معروفة منفصلة —
        # راجع ملاحظة الكوميت السابق) لكن هذا لا يمنع تسجيل الحافة المقطوعة
        # فعلياً؛ الهدف هنا فقط التأكد أن ScoringEngine/MemoryEngine غُذِّيا
        # بمسار حقيقي من خطوتين، بصرف النظر عن نجاح التنفيذ.
        result = engine.run_path([root_id, other_id], {"task": "x"}, use_fallback=False)
        b.record_direct_execution(result, prev)

        score = b.scoring_engine.get_score(root_id, other_id)
        assert score.total_runs == 1, score.to_dict()

        path_key = f"{root_id[:8]}->{other_id[:8]}"
        assert path_key in b.memory_engine._routes
        assert b.memory_engine._routes[path_key].runs == 1
        print("OK: two-hop /process run updates edge score + route memory")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_no_op_on_pathless_result():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_scoring_memory_wiring_test3_")
    try:
        b = _build_bundle(tmp_dir)
        # عقدتان حقيقيتان مسجَّلتان في الرسم البياني لكن بلا أي حافة تربطهما
        # (root فقط له حواف صادرة، وهذه عُقد أدوار بلا حواف خارجة إطلاقاً
        # — راجع _sync_nodes_to_graph) → run_between تعيد نتيجة "لا مسار"
        # حقيقية بلا أي استثناء، وهو المخرج الذي نختبر أمانه هنا.
        role_ids = [n for n in b.role_node_ids.values() if n != b._root_node_id]
        assert len(role_ids) >= 2
        engine = ExecutionEngine(b.registry, b.graph, b.storage, db=b.exec_log, ai=b.ai_decision)
        prev = b.snapshot_node_states()
        result = engine.run_between(role_ids[0], role_ids[1], {"task": "x"})
        assert result.status == "failed" and result.path == []
        b.record_direct_execution(result, prev)  # يجب ألا يرفع استثناءً
        assert len(b.memory_engine._routes) == 0
        print("OK: pathless (no-route) result is a safe no-op for scoring/memory")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_single_hop_process_updates_node_memory()
    test_two_hop_process_updates_edge_score_and_route_memory()
    test_no_op_on_pathless_result()
    print("جميع اختبارات ربط /process بـScoringEngine/MemoryEngine نجحت")
