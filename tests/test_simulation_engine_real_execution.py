"""ai/simulation_engine.py::SimulationEngine — لم تُبنَ قط في أي مكان
بالمشروع (صفر نتائج لـ"SimulationEngine(" خارج تعريف الكلاس نفسه، ولا
ملف اختبار واحد لها قبل هذا الملف)، وكانت معطوبة بالكامل عملياً:

1. self._mesh.run(inp, out, payload, use_ai=True) — MeshBundle لا تملك
   ولم تملك قط ميثود run() بهذا التوقيع (الواجهة الحقيقية الوحيدة لتشغيل
   مسار هي ExecutionEngine.run_path()). كل تنفيذ واحد كان يرفع
   AttributeError يلتقطه except أدناه صامتاً، فنتيجة الجولة بالكامل كانت
   "فشل" وهمي 100% من الوقت دون أي تنفيذ فعلي حدث إطلاقاً.
2. run_result.get("status", "failed") — حتى لو أُصلح (1)، ExecutionResult
   كائن له سمات (.status/.path) وليس dict، فـ.get() يرفع AttributeError.
3. self._mesh.knowledge/.memory/.scoring — هذه الأسماء غير موجودة على
   MeshBundle إطلاقاً (الأسماء الحقيقية: knowledge_store/memory_engine/
   scoring_engine) — كتلة "Update knowledge layer" بالكامل كانت تفشل من
   أول سطر، يلتقطها except منفصل صامتاً أيضاً.
4. self._mesh.graph.stats().get("total_edges", 0) — ServiceGraph.stats()
   يرجع المفتاح "edge_count" لا "total_edges"؛ كان يُرسَل 0 دائماً.
5. update_graph_statistics(..., total_runs=, success_rate=) — الدالة
   الحقيقية في knowledge/knowledge_store.py لا تقبل هذين المعاملين
   إطلاقاً (TypeError: unexpected keyword argument)، ملتقطة بنفس except.

كل نقطة من الخمس تحقّقتُ منها فعلياً بتشغيل محاكاة حقيقية قبل الإصلاح
(عبر monkeypatch لرفع الاستثناء المطابق لرسالة logger.debug الفعلية)،
وهذا الملف يثبت أن المحاكاة تُنفَّذ فعلياً الآن وتُحدِّث طبقة المعرفة
الحقيقية بأرقام صحيحة، لا نتائج فشل صامتة.
"""
from __future__ import annotations

import json
import logging
import shutil
import tempfile

import pytest

from ai.simulation_engine import SimulationEngine
from core.mesh_bundle import MeshBundle


@pytest.fixture()
def bundle():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_sim_engine_test_")
    try:
        yield MeshBundle(storage_dir=tmp_dir, db_path=f"{tmp_dir}/mesh.db")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_simulation_runs_real_executions_not_silent_failures(bundle):
    """قبل الإصلاح: كل تنفيذ كان AttributeError صامتاً → failures=كل شيء،
    successes=0 دائماً مهما كان السيناريو. الآن: تنفيذ حقيقي عبر
    ExecutionEngine.run_path على مسار 3 عُقد بسيط (نجاح شبه مضمون)."""
    sim = SimulationEngine(bundle)
    summary = sim.run_simulation(
        rounds=2, executions_per_round=3, delay_between_rounds=0, verbose=False,
    )
    assert summary["simulation"]["total_executions"] == 6
    # المسار (Input->Processor->Output) بسيط وحتمي — يجب أن ينجح فعلياً،
    # لا 0% كما كان يحدث بالضرورة قبل الإصلاح.
    assert summary["simulation"]["total_successes"] > 0
    assert summary["simulation"]["overall_success_rate"] > 0.0


def test_knowledge_layer_block_raises_no_exception(bundle, caplog):
    """الكتلة الثلاث (update_graph_statistics/update_node_rankings/
    update_route_rankings/update_connection_scores) يجب ألا ترفع أي
    استثناء ملتقَط صامتاً بعد الإصلاح — نتحقق مباشرة من عدم وجود رسالة
    'Knowledge update error' في السجلّ."""
    with caplog.at_level(logging.DEBUG, logger="ai.simulation_engine"):
        sim = SimulationEngine(bundle)
        sim.run_simulation(rounds=1, executions_per_round=2, delay_between_rounds=0, verbose=False)

    knowledge_errors = [r for r in caplog.records if "Knowledge update error" in r.message]
    assert knowledge_errors == [], f"كتلة تحديث المعرفة ما زالت تفشل صامتاً: {knowledge_errors}"


def test_graph_statistics_reflect_real_registry_and_graph_state(bundle):
    """graph_metrics.json['graph_statistics'] يجب أن يعكس العدد الحقيقي
    لعُقد/حواف self._mesh بعد المحاكاة، لا 0/0 (أثر الأسماء الخاطئة
    والمفتاح الخاطئ total_edges قبل الإصلاح)."""
    sim = SimulationEngine(bundle)
    sim.run_simulation(rounds=1, executions_per_round=1, delay_between_rounds=0, verbose=False)

    gm_path = bundle.knowledge_store._dir / "graph_metrics.json"
    assert gm_path.exists()
    with open(gm_path, encoding="utf-8") as f:
        data = json.load(f)

    gs = data["graph_statistics"]
    assert gs["total_nodes"] == bundle.registry.count()
    assert gs["total_edges"] == bundle.graph.stats()["edge_count"]
    assert gs["total_nodes"] > 0
    assert gs["total_edges"] > 0


def test_setup_simulation_nodes_registers_three_connected_nodes(bundle):
    sim = SimulationEngine(bundle)
    inp, proc, out = sim.setup_simulation_nodes()
    assert bundle.registry.exists(inp)
    assert bundle.registry.exists(proc)
    assert bundle.registry.exists(out)
    assert proc in bundle.graph.get_neighbors(inp)
    assert out in bundle.graph.get_neighbors(proc)
