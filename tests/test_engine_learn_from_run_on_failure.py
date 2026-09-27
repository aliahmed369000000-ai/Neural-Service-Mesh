"""
اختبار مستقل لـ core/engine.py::ExecutionEngine._finalize (الدالة
المشتركة الجديدة لكل مخارج run_path).

المشكلة التي يتحقق منها: قبل هذا التعديل، 3 من أصل 4 مخارج فشل في
run_path (عقدة غير موجودة بلا بديل، عقدة محجورة بلا بديل، فشل تنفيذ
حقيقي بلا بديل) كانت تُرجِع النتيجة مباشرة دون استدعاء
self._ai.learn_from_run إطلاقاً — فقط مسار النجاح النهائي كان يستدعيها.
الأثر: AIDecisionLayer._path_stats[key]["runs"] لا يزيد إلا عند النجاح،
فـ successes/runs في get_insights() تكون دائماً 1.0 (100%) لأي مسار جرّب
الفشل ولو عشرات المرات، طالما نجح مرة واحدة على الأقل — يُفسِد كامل
الغرض من تصنيف health='critical' عند تدهور الأداء الفعلي.

يتحقق هذا الملف:
1. فشل تنفيذ حقيقي (بلا بديل) يزيد runs في path_stats فعلاً، ولا يزيد
   successes — success_rate تنخفض عن 1.0 كما يجب.
2. عقدة غير موجودة بلا بديل تُسجَّل أيضاً كتشغيلة فاشلة (runs+=1،
   successes لا تزيد)، مع total_duration_ms محسوبة فعلياً (لم تكن تُضبَط
   إطلاقاً في هذا المخرج تحديداً قبل هذا التعديل — تبقى None).
3. عقدة محجورة بلا بديل: نفس التحقق — runs+=1، total_duration_ms محسوبة.
4. تنفيذ ناجح عادي: لا تراجع — runs+=1، successes+=1 كما كان.
"""
from __future__ import annotations

import shutil
import tempfile

from core.engine import ExecutionEngine
from core.graph import ServiceGraph
from core.node import BaseNode, NodeSchema
from core.registry import NodeRegistry
from storage.file_storage import FileStorage


class FlakyNode(BaseNode):
    def __init__(self, name, fail=False, node_id=None):
        super().__init__(name, node_id=node_id)
        self._fail = fail

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={}, required=[])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "str"}, required=[])

    def process(self, data):
        if self._fail:
            raise RuntimeError(f"فشل متعمَّد في {self.name}")
        return {"result": self.name}


class RealAILikeStub:
    """نسخة مبسّطة حقيقية من ai/decision.py::AIDecisionLayer.learn_from_run
    (نفس الحساب بالضبط) بدل استيراد الكلاس الكامل — تكفي لإثبات أن
    run_path يستدعيها فعلاً على كل مخرج، بما فيه الفشل."""

    def __init__(self):
        self._path_stats = {}

    def should_fallback(self, failed_node_id, error):
        return None  # لا بديل في هذا الاختبار: نريد رؤية الفشل يُسجَّل كما هو

    def learn_from_run(self, run_result):
        path = run_result.get("path", [])
        if not path:
            return
        key = "->".join(p[:8] for p in path)
        s = self._path_stats.setdefault(key, {"runs": 0, "successes": 0, "total_ms": 0.0})
        s["runs"] += 1
        if run_result.get("status") == "success":
            s["successes"] += 1
        s["total_ms"] += run_result.get("total_duration_ms") or 0.0


def _build(tmp_dir):
    storage = FileStorage(storage_dir=tmp_dir)
    registry = NodeRegistry(storage)
    graph = ServiceGraph()
    return storage, registry, graph


def test_real_failure_is_counted_in_learn_from_run():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_learn_test1_")
    try:
        storage, registry, graph = _build(tmp_dir)
        node = FlakyNode("primary", fail=True)
        registry.register(node)
        graph.add_node(node.node_id, node.to_dict())

        ai = RealAILikeStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([node.node_id], {"task": "x"})

        assert result.status == "failed"
        assert result.total_duration_ms is not None
        key = node.node_id[:8]
        assert ai._path_stats[key]["runs"] == 1
        assert ai._path_stats[key]["successes"] == 0
        print("OK: real failure counted in learn_from_run (success_rate no longer stuck at 100%)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_missing_node_failure_is_counted_and_has_duration():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_learn_test2_")
    try:
        storage, registry, graph = _build(tmp_dir)
        ai = RealAILikeStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        fake_id = "does-not-exist-0000"
        result = engine.run_path([fake_id], {"task": "x"})

        assert result.status == "failed"
        assert result.total_duration_ms is not None, "كانت تبقى None في هذا المخرج تحديداً"
        key = fake_id[:8]
        assert ai._path_stats[key]["runs"] == 1
        assert ai._path_stats[key]["successes"] == 0
        print("OK: missing-node failure counted + total_duration_ms now set")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paused_node_failure_is_counted_and_has_duration():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_learn_test3_")
    try:
        storage, registry, graph = _build(tmp_dir)
        node = FlakyNode("primary", fail=False)
        registry.register(node)
        graph.add_node(node.node_id, node.to_dict())
        node.pause()

        ai = RealAILikeStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([node.node_id], {"task": "x"})

        assert result.status == "failed"
        assert result.total_duration_ms is not None
        key = node.node_id[:8]
        assert ai._path_stats[key]["runs"] == 1
        assert ai._path_stats[key]["successes"] == 0
        print("OK: paused-node failure counted + total_duration_ms now set")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_success_still_counted_normally():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_learn_test4_")
    try:
        storage, registry, graph = _build(tmp_dir)
        node = FlakyNode("primary", fail=False)
        registry.register(node)
        graph.add_node(node.node_id, node.to_dict())

        ai = RealAILikeStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([node.node_id], {"task": "x"})

        assert result.status == "success"
        key = node.node_id[:8]
        assert ai._path_stats[key]["runs"] == 1
        assert ai._path_stats[key]["successes"] == 1
        print("OK: normal success still counted correctly (no regression)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_real_failure_is_counted_in_learn_from_run()
    test_missing_node_failure_is_counted_and_has_duration()
    test_paused_node_failure_is_counted_and_has_duration()
    test_success_still_counted_normally()
    print("جميع اختبارات core/engine.py (تسجيل الفشل في learn_from_run) نجحت")
