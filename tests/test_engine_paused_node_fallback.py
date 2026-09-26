"""
اختبار مستقل (core/engine.py لا يملك ملف اختبار مخصصاً قبل هذا التعديل)
للتحقق من إصلاح: عقدة موجودة فعلياً في المسار لكن محجورة
(NodeState.PAUSED — عادة عبر MeshBundle._apply_reputation_feedback بسبب
سمعة منخفضة) كانت تُسقِط run_path بالكامل فوراً، لأن node.execute() ترفع
RuntimeError يلتقطها الـexcept العام فيُسجَّل كخطأ تنفيذ عادي بلا أي محاولة
بديل — رغم أن نفس run_path يملك بالفعل مسار fallback حقيقي
(self._ai.should_fallback) لعقدة غير موجودة تماماً في المسجّل، لكنه لم
يكن يُستدعى أبداً لعقدة موجودة لكن محجورة.

هذا الملف يبني ExecutionEngine حقيقياً (NodeRegistry+ServiceGraph+
FileStorage حقيقية بمجلد مؤقت، لا سجل زائف) ويتحقق:
1. عقدة محجورة ولها بديل غير محجور عبر should_fallback → المسار ينجح
   فعلياً عبر البديل (is_fallback=True)، وshould_fallback استُدعيت فعلاً.
2. عقدة محجورة بلا أي بديل → فشل نظيف (status=failed) بلا استثناء غير
   متحكَّم به، ورسالة خطأ واضحة تذكر الحجر.
3. البديل المقترَح نفسه محجور أيضاً → يُرفَض ولا يُستخدَم (لا نُسلسل
   fallback فوق fallback).
4. عقدة نشطة عادية (غير محجورة) تعمل تماماً كما كانت — لا تراجع.
"""
from __future__ import annotations

import shutil
import tempfile

from core.engine import ExecutionEngine
from core.graph import ServiceGraph
from core.node import BaseNode, NodeSchema
from core.registry import NodeRegistry
from storage.file_storage import FileStorage


class EchoNode(BaseNode):
    """عقدة اختبار بسيطة: بلا حقول مطلوبة، تُعيد اسمها كنتيجة."""

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={}, required=[])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "str"}, required=[])

    def process(self, data):
        return {"result": self.name}


class FakeAI:
    """يحاكي ai/decision.py::AIDecisionEngine.should_fallback بأبسط شكل
    (بديل محدَّد سلفاً) مع عدّاد استدعاءات، للتأكد أن run_path يستدعيها
    فعلاً لعقدة محجورة — وليس فقط لعقدة غير موجودة كما كان الحال سابقاً."""

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


def test_paused_node_uses_fallback():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_paused_test1_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary, backup = EchoNode("primary"), EchoNode("backup")
        registry.register(primary)
        registry.register(backup)
        graph.add_node(primary.node_id, primary.to_dict())
        graph.add_node(backup.node_id, backup.to_dict())
        graph.add_edge(primary.node_id, backup.node_id)

        primary.pause()  # يحاكي الحجر الفعلي عبر _apply_reputation_feedback

        ai = FakeAI(fallback_id=backup.node_id)
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert ai.calls == 1, "يجب استدعاء should_fallback فعلياً لعقدة محجورة"
        assert result.status == "success", result.to_dict()
        assert result.steps[0].is_fallback is True
        assert result.steps[0].node_id == backup.node_id
        assert result.final_output == {"result": "backup"}
        print("OK: fallback used for paused node")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paused_node_no_fallback_fails_cleanly():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_paused_test2_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary = EchoNode("primary")
        registry.register(primary)
        graph.add_node(primary.node_id, primary.to_dict())
        primary.pause()

        ai = FakeAI(fallback_id=None)
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert ai.calls == 1
        assert result.status == "failed"
        assert result.steps[0].status == "error"
        assert "paused" in result.steps[0].error
        print("OK: no fallback available -> clean failure, no crash")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_paused_fallback_candidate_also_paused_is_rejected():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_paused_test3_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary, backup = EchoNode("primary"), EchoNode("backup")
        registry.register(primary)
        registry.register(backup)
        graph.add_node(primary.node_id, primary.to_dict())
        graph.add_node(backup.node_id, backup.to_dict())
        graph.add_edge(primary.node_id, backup.node_id)
        primary.pause()
        backup.pause()  # كلا العقدتين محجورتان

        ai = FakeAI(fallback_id=backup.node_id)
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert result.status == "failed", "لا يجب استخدام بديل محجور أيضاً"
        print("OK: paused fallback candidate correctly rejected")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_normal_execution_unaffected():
    """عقدة نشطة عادية (غير محجورة) يجب أن تعمل تماماً كما كانت — لا تراجع."""
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_paused_test4_")
    try:
        storage, registry, graph = _build(tmp_dir)
        primary = EchoNode("primary")
        registry.register(primary)
        graph.add_node(primary.node_id, primary.to_dict())

        engine = ExecutionEngine(registry, graph, storage)
        result = engine.run_path([primary.node_id], {"task": "x"})

        assert result.status == "success"
        assert result.steps[0].is_fallback is False
        print("OK: normal (non-paused) execution unaffected")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_paused_node_uses_fallback()
    test_paused_node_no_fallback_fails_cleanly()
    test_paused_fallback_candidate_also_paused_is_rejected()
    test_normal_execution_unaffected()
    print("جميع اختبارات core/engine.py (عقدة محجورة داخل run_path) نجحت")
