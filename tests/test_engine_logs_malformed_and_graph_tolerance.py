"""
اختبار تحصينين في طبقة العقد الأساسية:

1. core/engine.py::_persist — logs.json قد يحوي runs غير قائمة (تلف)؛ يُعاد
   تهيئتها بدل الانهيار. (لا سقف لحجم السجل — قرار المالك: يكبر بحرية.)
2. core/graph.py::ServiceGraph.from_dict — حافة تشير لعقدة غير موجودة (أو
   عقدة بلا node_id) كانت ترفع KeyError وتُسقط تحميل الرسم كله؛ الآن تُتجاهَل
   بتحذير وبقية الرسم يُحمَّل سليماً.
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
    @property
    def input_schema(self) -> NodeSchema:
        return NodeSchema(fields={}, required=[])

    @property
    def output_schema(self) -> NodeSchema:
        return NodeSchema(fields={"result": "str"}, required=[])

    def process(self, data):
        return {"result": self.name}


def _build(tmp):
    storage = FileStorage(tmp)
    registry = NodeRegistry(storage)
    graph = ServiceGraph()
    node = EchoNode("echo")
    registry.register(node)
    graph.add_node(node.node_id, node.to_dict())
    return ExecutionEngine(registry, graph, storage), node, storage


def test_logs_file_with_malformed_runs_does_not_break_persist():
    tmp = tempfile.mkdtemp()
    try:
        eng, node, storage = _build(tmp)
        storage.save("logs.json", {"runs": "not-a-list"})
        res = eng.run_path([node.node_id], {})
        assert res.status == "success"
        assert [r["run_id"] for r in storage.load("logs.json")["runs"]] == [res.run_id]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_graph_from_dict_skips_dangling_edges_and_bad_nodes():
    g = ServiceGraph()
    g.from_dict({
        "nodes": [
            {"node_id": "a", "name": "A"},
            {"node_id": "b", "name": "B"},
            {"name": "no-id"},
            "garbage",
        ],
        "edges": [
            {"source_id": "a", "target_id": "b", "weight": 2.0},
            {"source_id": "a", "target_id": "ghost"},
            {"source_id": "ghost", "target_id": "b"},
            {"source_id": "a"},
            None,
        ],
    })
    assert g.stats()["node_count"] == 2
    assert g.stats()["edge_count"] == 1
    assert g.find_path_bfs("a", "b") == ["a", "b"]


def test_graph_roundtrip_unchanged():
    g = ServiceGraph()
    g.add_node("x", {"node_id": "x"})
    g.add_node("y", {"node_id": "y"})
    g.add_edge("x", "y", 3.0, "lbl")
    g2 = ServiceGraph()
    g2.from_dict(g.to_dict())
    assert g2.to_dict() == g.to_dict()
