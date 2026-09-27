"""
اختبار مستقل لـ core/engine.py: فشل تنفيذ حقيقي (استثناء من process()
نفسها، أو رفض NodeSchema.validate()) داخل run_path كان يُسقِط المسار
بالكامل فوراً دون أي محاولة بديل — رغم أن self._ai.should_fallback
مصمَّمة أصلاً لهذه الحالة العامة (اسم وسيطتها الأولى "failed_node_id")،
وليس فقط لعقدة غير موجودة في المسجّل أو محجورة (النطاقان اللذان كانا
مدعومَين فقط قبل هذا التعديل).

هذا الملف يتحقق:
1. عقدة تفشل فعلياً أثناء process() ولها بديل صالح عبر should_fallback
   → المسار ينجح فعلياً عبر البديل (is_fallback=True، step.error يحمل
   رسالة الفشل الأصلي).
2. عقدة تفشل ولا يوجد بديل → فشل نظيف (status=failed) كما كان.
3. عقدة تفشل والبديل المقترَح يفشل هو الآخر أثناء التنفيذ → لا محاولة
   ثالثة (مستوى واحد فقط)، فشل نظيف نهائي.
4. عقدة نشطة تنجح من أول محاولة → لا تراجع، is_fallback يبقى False.
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
    """عقدة اختبار: تفشل دائماً إن fail=True، وإلا تُعيد اسمها."""

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
            raise RuntimeError(f"فشل حقيقي متعمَّد في {self.name}")
        return {"result": self.name}


class FakeAI:
    def __init__(self, fallback_id=None):
        self.fallback_id = fallback_id
        self.calls = 0

    def should_fallback(self, failed_node_id, error):
        self.calls += 1
        return self.fallback_id

    def learn_from_run(self, *a, **k):
        pass


def _build(tmp_dir):
    storage = FileStorage(storage_dir=tmp_dir)
    registry = NodeRegistry(storage)
    graph = ServiceGraph()
    return storage, registry, graph


def test_real_execution_failure_uses_fallback():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_realfail_test1_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary = FlakyNode("primary", fail=True)
        backup = FlakyNode("backup", fail=False)
        registry.register(primary)
        registry.register(backup)
        graph.add_node(primary.node_id, primary.to_dict())
        graph.add_node(backup.node_id, backup.to_dict())
        graph.add_edge(primary.node_id, backup.node_id)

        ai = FakeAI(fallback_id=backup.node_id)
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert ai.calls == 1
        assert result.status == "success", result.to_dict()
        step = result.steps[0]
        assert step.is_fallback is True
        assert step.node_id == backup.node_id
        assert "فشل حقيقي متعمَّد" in step.error
        assert result.final_output == {"result": "backup"}
        print("OK: real execution failure -> fallback used")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_real_execution_failure_no_fallback_fails_cleanly():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_realfail_test2_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary = FlakyNode("primary", fail=True)
        registry.register(primary)
        graph.add_node(primary.node_id, primary.to_dict())

        ai = FakeAI(fallback_id=None)
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert ai.calls == 1
        assert result.status == "failed"
        print("OK: no fallback available -> clean failure")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_fallback_that_also_fails_is_not_chained_further():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_realfail_test3_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary = FlakyNode("primary", fail=True)
        backup = FlakyNode("backup", fail=True)  # البديل نفسه يفشل
        registry.register(primary)
        registry.register(backup)
        graph.add_node(primary.node_id, primary.to_dict())
        graph.add_node(backup.node_id, backup.to_dict())
        graph.add_edge(primary.node_id, backup.node_id)

        ai = FakeAI(fallback_id=backup.node_id)
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert ai.calls == 1, "لا يجب طلب بديل ثانٍ لبديل فشل هو الآخر"
        assert result.status == "failed"
        assert "backup" in result.steps[0].error
        print("OK: fallback-of-fallback correctly not attempted")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_normal_success_unaffected():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_realfail_test4_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary = FlakyNode("primary", fail=False)
        registry.register(primary)
        graph.add_node(primary.node_id, primary.to_dict())

        engine = ExecutionEngine(registry, graph, storage)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert result.status == "success"
        assert result.steps[0].is_fallback is False
        print("OK: normal successful execution unaffected")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_real_execution_failure_uses_fallback()
    test_real_execution_failure_no_fallback_fails_cleanly()
    test_fallback_that_also_fails_is_not_chained_further()
    test_normal_success_unaffected()
    print("جميع اختبارات core/engine.py (فشل تنفيذ حقيقي داخل run_path) نجحت")
