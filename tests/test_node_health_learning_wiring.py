"""ai/node_health_layer.py::NodeHealthLayer.__init__ — self.learning =
MeshLearningEngine(node) كان يستخدم اسم متغيّر node غير معرَّف إطلاقاً في
الدالة (المعامل الحقيقي اسمه mesh_node)، فيرفع NameError في كل مرة بلا
استثناء، يلتقطه except Exception عام صامتاً فيترك self.learning = None
دائماً. تحقّقتُ من هذا بتشغيل عقدة حقيقية فعلاً عبر
`python3 -m ai.node_launcher` وطباعة الاستثناء المكتوم بأداة تصحيح مؤقتة:
"NameError: name 'node' is not defined". النتيجة العملية على أي عقدة
شُغِّلت على الإطلاق: /v2/learn/status و/v2/learn/cycle كانا يرجعان دائماً
503 {"error": "learning_engine_unavailable"}، ولا مهمة واحدة كانت
تُغذّي MeshLearningEngine.learn_from_task عبر record_task_result (الشرط
self.learning is not None يفشل دائماً) — التعلّم من نتائج المهام كان
معطَّلاً بالكامل على مستوى الوحدة، لا غير مُستخدَم فقط.

هذا الملف (أول اختبار على الإطلاق لهذا التسليك) يثبت أن NodeHealthLayer
تبني MeshLearningEngine فعلياً الآن، وأن مهمة حقيقية تُغذّيها بنجاح.
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile

import pytest

from ai.living_mesh import LivingMeshNode
from ai.mesh_learning_engine import MeshLearningEngine
from ai.node_health_layer import NodeHealthLayer


@pytest.fixture()
def mesh_node():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_node_health_test_")
    try:
        yield LivingMeshNode(node_id="health_test_node", host="127.0.0.1", port=0, data_dir=tmp_dir)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_learning_engine_constructed_not_none(mesh_node):
    health = NodeHealthLayer(mesh_node)
    assert health.learning is not None
    assert isinstance(health.learning, MeshLearningEngine)


def test_learning_init_failure_is_logged_not_silent(mesh_node, caplog, monkeypatch):
    """حتى لو فشل بناء MeshLearningEngine لسبب آخر مستقبلاً، يجب ألا يُكتَم
    الاستثناء صامتاً كما كان — على الأقل تحذير واضح في السجلّ."""
    import ai.node_health_layer as nhl

    def _boom(*a, **k):
        raise RuntimeError("فشل متعمَّد للاختبار")

    monkeypatch.setattr(nhl, "MeshLearningEngine", _boom, raising=False)
    # نحتاج محاكاة الاستيراد المحلي داخل __init__ (from ai.mesh_learning_engine
    # import MeshLearningEngine) — أسهل طريقة: نراقب وحدة ai.mesh_learning_engine نفسها
    import ai.mesh_learning_engine as mle
    monkeypatch.setattr(mle, "MeshLearningEngine", _boom)

    with caplog.at_level(logging.WARNING, logger="NodeHealthLayer"):
        health = nhl.NodeHealthLayer(mesh_node)

    assert health.learning is None
    assert any("تعذّر تفعيل MeshLearningEngine" in r.message for r in caplog.records)


def test_learning_skills_snapshot_updates_after_real_task(mesh_node):
    """مهمة محلية حقيقية عبر submit_verifiable_task(local=True) — نفس
    المسار الذي يخدم POST /v2/task فعلياً على عقدة حيّة — تُغذّي
    MeshLearningEngine فعلاً الآن (كانت self.learning is not None تفشل
    دائماً قبل الإصلاح، فلا سطر تعلّم كان يُنفَّذ قط)."""
    health = NodeHealthLayer(mesh_node)
    before = health.learning.skills_snapshot()["stats"]["lessons"]

    result = asyncio.run(health.submit_verifiable_task(
        kind="map_reduce_map",
        payload={"lines": ["learning wiring test"], "op": "wordcount"},
        local=True,
    ))
    assert result["ok"] is True
    assert "learning" in result  # الحقل الذي يُضاف فقط عندما self.learning ليست None

    after = health.learning.skills_snapshot()["stats"]["lessons"]
    assert after == before + 1
