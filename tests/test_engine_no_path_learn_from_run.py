"""
اختبار مستقل لـ core/engine.py::ExecutionEngine.run_between و run_full_graph
— مخرجا الفشل "لا يوجد مسار" (run_between) و"لا يوجد ترتيب طوبولوجي"
(run_full_graph، رسم بياني فارغ/فيه دورة).

المشكلة: هذان المخرجان كانا يبنيان ExecutionResult يدوياً (status/finished_at
فقط) ويستدعيان self._persist() مباشرة — بنفس علّة c56be5c بالضبط (3 من 4
مخارج فشل run_path كانت تتخطى self._ai.learn_from_run)، لكن هنا في مخرجين
منفصلين تماماً لم يمسّهما ذلك التعديل لأنهما خارج run_path نفسها:
  1. self._ai.learn_from_run لا يُستدعى إطلاقاً على هذا الفشل.
  2. result.total_duration_ms يبقى None دائماً (لا t_start أصلاً في الدالتين).

يتحقق هذا الملف:
1. run_between بلا مسار (start/end غير متصلين): learn_from_run استُدعيت،
   total_duration_ms محسوبة فعلياً (لا None).
2. run_full_graph برسم بياني فيه دورة (topological_sort يرجع فارغة): نفس
   التحقق.
3. لا تراجع: run_between/run_full_graph بمسار صحيح ما زالا ينجحان تماماً
   كالسابق (نفس نتيجة run_path الداخلية، ai_suggested يُضبط كما كان).
"""
from __future__ import annotations

import shutil
import tempfile

from core.engine import ExecutionEngine
from core.graph import ServiceGraph
from core.node import BaseNode, NodeSchema
from core.registry import NodeRegistry
from storage.file_storage import FileStorage


class SimpleNode(BaseNode):
    def __init__(self, name, node_id=None):
        super().__init__(name, node_id=node_id)

    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={}, required=[])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "str"}, required=[])

    def process(self, data):
        return {"result": self.name}


class CountingAIStub:
    """يعدّ استدعاءات learn_from_run صراحة (بعكس RealAILikeStub في
    test_engine_learn_from_run_on_failure.py اللي تتجاهل مسار فارغ بصمت) —
    هنا نريد إثبات أن *الاستدعاء نفسه* حصل، بغضّ النظر عن معالجته الداخلية
    لمسار فارغ، لأن ذلك بالضبط ما كان مفقوداً قبل هذا الإصلاح."""

    def __init__(self):
        self.learn_calls = []

    def choose_path(self, start_id, end_id):
        return None  # نجبر البحث البديل (find_path_bfs) ليفشل هو الآخر

    def should_fallback(self, failed_node_id, error):
        return None

    def learn_from_run(self, run_result):
        self.learn_calls.append(run_result)


def _build(tmp_dir):
    storage = FileStorage(storage_dir=tmp_dir)
    registry = NodeRegistry(storage)
    graph = ServiceGraph()
    return storage, registry, graph


def test_run_between_no_path_calls_learn_from_run_with_duration():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_no_path_test1_")
    try:
        storage, registry, graph = _build(tmp_dir)
        a = SimpleNode("a")
        b = SimpleNode("b")
        registry.register(a)
        registry.register(b)
        graph.add_node(a.node_id, a.to_dict())
        graph.add_node(b.node_id, b.to_dict())
        # لا حافة بين a وb — find_path_bfs يفشل، choose_path يرجع None عمداً

        ai = CountingAIStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_between(a.node_id, b.node_id, {"task": "x"})

        assert result.status == "failed"
        assert result.total_duration_ms is not None, "كانت تبقى None قبل هذا الإصلاح"
        assert len(ai.learn_calls) == 1, "learn_from_run لم تُستدعَ إطلاقاً قبل هذا الإصلاح"
        assert ai.learn_calls[0]["status"] == "failed"
        print("OK: run_between (no path) -> learn_from_run استُدعيت + total_duration_ms محسوبة")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_full_graph_no_order_calls_learn_from_run_with_duration():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_no_path_test2_")
    try:
        storage, registry, graph = _build(tmp_dir)
        a = SimpleNode("a")
        b = SimpleNode("b")
        registry.register(a)
        registry.register(b)
        graph.add_node(a.node_id, a.to_dict())
        graph.add_node(b.node_id, b.to_dict())
        # دورة a->b->a تُفشل topological_sort عمداً
        graph.add_edge(a.node_id, b.node_id)
        graph.add_edge(b.node_id, a.node_id)

        ai = CountingAIStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_full_graph({"task": "x"})

        assert result.status == "failed"
        assert result.total_duration_ms is not None, "كانت تبقى None قبل هذا الإصلاح"
        assert len(ai.learn_calls) == 1, "learn_from_run لم تُستدعَ إطلاقاً قبل هذا الإصلاح"
        print("OK: run_full_graph (دورة/بلا ترتيب) -> learn_from_run استُدعيت + total_duration_ms محسوبة")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_run_between_success_path_unaffected():
    """لا تراجع: مسار ناجح عادي عبر run_between ما زال يشتغل ويضبط
    ai_suggested تماماً كالسابق — هذا الإصلاح لا يمسّ هذا المسار."""
    tmp_dir = tempfile.mkdtemp(prefix="nsm_engine_no_path_test3_")
    try:
        storage, registry, graph = _build(tmp_dir)
        a = SimpleNode("a")
        b = SimpleNode("b")
        registry.register(a)
        registry.register(b)
        graph.add_node(a.node_id, a.to_dict())
        graph.add_node(b.node_id, b.to_dict())
        graph.add_edge(a.node_id, b.node_id)

        class ChoosePathStub(CountingAIStub):
            def choose_path(self, start_id, end_id):
                return [start_id, end_id]

        ai = ChoosePathStub()
        engine = ExecutionEngine(registry, graph, storage, ai=ai)
        result = engine.run_between(a.node_id, b.node_id, {"task": "x"})

        assert result.status == "success"
        assert result.ai_suggested is True
        assert result.total_duration_ms is not None
        print("OK: run_between (مسار ناجح) بلا تراجع — نفس السلوك السابق تماماً")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    test_run_between_no_path_calls_learn_from_run_with_duration()
    test_run_full_graph_no_order_calls_learn_from_run_with_duration()
    test_run_between_success_path_unaffected()
    print("جميع اختبارات core/engine.py (learn_from_run في مخارج run_between/run_full_graph) نجحت")
