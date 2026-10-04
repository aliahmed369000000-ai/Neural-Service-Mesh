"""core/mesh_bundle.py: تسليك OptimizationEngine (ai/optimization_engine.py).

ai/optimization_engine.py::OptimizationEngine كانت مكتوبة بالكامل (يحلّل
الرسم البياني فعلياً: وصلات فاشلة/ناجحة، عُقد غير مُستخدَمة، تحديث أوزان،
ويكتب كل هذا إلى knowledge/graph_metrics.json عبر KnowledgeStore) لكن لا
شيء في المشروع كان يبنيها (instantiate) — تحقّقتُ بالبحث عن
"OptimizationEngine(" قبل هذا التعديل: صفر نتائج خارج تعريف الكلاس. هذا
الملف يثبت:

1. MeshBundle تبني نسخة واحدة مربوطة بـself.graph/scoring_engine/
   memory_engine/knowledge_store الحقيقية.
2. run_evolution_cycle() تستدعي analyze() فعلاً (وليس فقط توجد النسخة).
3. knowledge/graph_metrics.json يحتوي فعلاً optimization_metrics بعد دورة
   واحدة — لا يبقى بمخططه الفارغ كما كان.
4. apply_report() عمداً غير مُستدعاة تلقائياً (القرار متروك لطبقة أعلى) —
   لا حذف حواف/عُقد فعلي من run_evolution_cycle نفسها.
5. فشل analyze() (استثناء) لا يُسقط run_evolution_cycle كاملة (best-effort).
"""
from __future__ import annotations

import json
import shutil
import tempfile

import pytest

from ai.optimization_engine import OptimizationEngine
from core.mesh_bundle import MeshBundle


@pytest.fixture()
def bundle():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_opt_wiring_test_")
    try:
        yield MeshBundle(storage_dir=tmp_dir, db_path=f"{tmp_dir}/mesh.db")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_optimization_engine_wired_with_real_components(bundle):
    assert isinstance(bundle.optimization_engine, OptimizationEngine)
    assert bundle.optimization_engine._graph is bundle.graph
    assert bundle.optimization_engine._scoring is bundle.scoring_engine
    assert bundle.optimization_engine._memory is bundle.memory_engine
    assert bundle.optimization_engine._knowledge is bundle.knowledge_store


def test_run_evolution_cycle_calls_analyze(bundle):
    calls = {"n": 0}
    real_analyze = bundle.optimization_engine.analyze

    def _spy_analyze():
        calls["n"] += 1
        return real_analyze()

    bundle.optimization_engine.analyze = _spy_analyze
    bundle.run_evolution_cycle()
    assert calls["n"] == 1


def test_run_evolution_cycle_does_not_auto_apply_report(bundle):
    """التصميم المقصود: analyze() فقط. apply_report() قرار طبقة أعلى."""
    applied = {"n": 0}
    bundle.optimization_engine.apply_report = lambda *a, **k: applied.__setitem__("n", applied["n"] + 1)
    bundle.run_evolution_cycle()
    assert applied["n"] == 0


def test_graph_metrics_populated_after_evolution_cycle(bundle):
    bundle.run_evolution_cycle()
    report = bundle.optimization_engine.last_report()
    assert report is not None
    assert report["total_actions"] >= 0  # قد تكون صفراً لو الرسم البياني فارغ، المهم أنها رُكّبت فعلياً

    gm_path = bundle.knowledge_store._dir / "graph_metrics.json"
    assert gm_path.exists(), "graph_metrics.json لم يُكتب إطلاقاً"
    with open(gm_path, encoding="utf-8") as f:
        km = json.load(f)
    assert km["optimization_metrics"]["total_optimization_runs"] >= 1


def test_analyze_failure_does_not_break_evolution_cycle(bundle):
    def _boom():
        raise RuntimeError("فشل تحليل مقصود للاختبار")

    bundle.optimization_engine.analyze = _boom
    # لا يرفع استثناء إلى الخارج — best-effort، نفس نمط dna.snapshot المجاورة
    result = bundle.run_evolution_cycle()
    assert result is not None
