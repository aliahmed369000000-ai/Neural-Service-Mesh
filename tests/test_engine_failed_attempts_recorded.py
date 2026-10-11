# -*- coding: utf-8 -*-
"""فشل عقدة أُنقذ ببديل (fallback) لا يجوز أن يضيع: كانت الخطوة الواحدة تُستبدل
ببيانات البديل فلا يرى التدريب العصبي ولا السمعة ولا بثّ node_failed فشلَ العقدة
الأصلية — فتبدو عقدة تفشل وتُنقَذ دائماً «مثالية» ولا تُحجَر أبداً."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from ai.decision import AIDecisionLayer
from ai.node_neural_advisor import NodeNeuralAdvisor
from core.engine import ExecutionEngine
from core.mesh_bundle import MeshBundle
from core.node import BaseNode, NodeSchema
from core.graph import ServiceGraph
from core.registry import NodeRegistry
from storage.file_storage import FileStorage


class _Ok(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])

    def process(self, data):
        return {}


class _Fail(_Ok):
    def process(self, data):
        raise ValueError("boom")


@pytest.fixture()
def tmp(monkeypatch):
    d = tempfile.mkdtemp(prefix="nsm_fa_test_")
    monkeypatch.chdir(d)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _plain_engine(tmp):
    storage = FileStorage(storage_dir=tmp)
    registry = NodeRegistry(storage)
    graph = ServiceGraph()
    adv = NodeNeuralAdvisor()
    ai = AIDecisionLayer(graph=graph, advisor=adv)
    return registry, graph, adv, ExecutionEngine(registry, graph, storage, ai=ai)


def _add(registry, graph, node):
    registry.register(node)
    graph.add_node(node.node_id, node.to_dict())
    return node


def test_rescued_failure_is_recorded_in_failed_attempts_and_to_dict(tmp):
    registry, graph, _adv, eng = _plain_engine(tmp)
    a, b = _add(registry, graph, _Fail("a")), _add(registry, graph, _Ok("b"))
    graph.add_edge(a.node_id, b.node_id)
    res = eng.run_path([a.node_id], {})
    assert res.status == "success"
    assert len(res.steps) == 1 and res.steps[0].node_id == b.node_id and res.steps[0].is_fallback
    assert len(res.failed_attempts) == 1
    fa = res.failed_attempts[0]
    assert fa["node_id"] == a.node_id and fa["error"] == "boom" and fa["state_before"] == "created"
    assert res.to_dict()["failed_attempts"] == res.failed_attempts


def test_neural_network_learns_the_original_nodes_failure(tmp):
    registry, graph, adv, eng = _plain_engine(tmp)
    a, b = _add(registry, graph, _Fail("a")), _add(registry, graph, _Ok("b"))
    graph.add_edge(a.node_id, b.node_id)
    eng.run_path([a.node_id], {})
    assert list(adv._stats[a.node_id].window) == [0]   # الفشل الأصلي لم يعد يضيع
    assert list(adv._stats[b.node_id].window) == [1]
    assert adv.summary()["observations"] == 2


def test_unrescued_failure_is_not_double_counted(tmp):
    registry, graph, adv, eng = _plain_engine(tmp)
    a = _add(registry, graph, _Fail("a"))
    res = eng.run_path([a.node_id], {})
    assert res.status == "failed" and res.failed_attempts == []
    assert list(adv._stats[a.node_id].window) == [0] and adv.summary()["observations"] == 1


def test_transform_failure_still_records_pre_attempt_state(tmp):
    registry, graph, _adv, eng = _plain_engine(tmp)
    a = _add(registry, graph, _Ok("a"))
    a.pause()
    a.resume()
    res = eng.run_path([a.node_id], {})
    assert res.steps[0].state_before == "active"


def test_bundle_reputation_and_failure_broadcast_see_the_rescued_node(tmp):
    b = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    a, good = _Fail("flaky"), _Ok("rescuer")
    b.register_node(a)
    b.register_node(good)
    b.graph.add_edge(a.node_id, good.node_id)
    prev = b.snapshot_node_states()
    eng = ExecutionEngine(b.registry, b.graph, b.storage, ai=b.ai_decision)
    res = eng.run_path([a.node_id], {})
    assert res.status == "success" and len(res.failed_attempts) == 1
    b.record_direct_execution(res, prev_states=prev)

    rep_a = b.reputation_engine.get_reputation(a.node_id)
    assert rep_a is not None and rep_a.failed_runs == 1 and rep_a.successful_runs == 0
    rep_good = b.reputation_engine.get_reputation(good.node_id)
    assert rep_good.successful_runs == 1
    inbox = b.channel.inbox(good.node_id)
    assert any(m["topic"] == "node_failed" and m["from_id"] == a.node_id for m in inbox)
