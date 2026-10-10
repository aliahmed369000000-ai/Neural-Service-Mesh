# -*- coding: utf-8 -*-
"""اختبارات لربط الاستخدام التلقائي للشبكة العصبية المفتوحة المصدر
(ai/node_think.py::neural_refine_evolution_task) داخل
core/mesh_bundle.py::run_evolution_cycle.

قبل هذا: neural_refine_evolution_task كانت مُفعَّلة فعلياً فقط داخل
LivingMeshNode المستقلة (ai/living_mesh.py) — الحزمة في العملية الواحدة
(core/mesh_bundle.py، وهي ما يستخدمه تطبيق Streamlit الفعلي) لم تكن
تستدعيها إطلاقاً رغم وجودها كدالة مشتركة مصمَّمة للاستخدام من كليهما.
"""
from __future__ import annotations

import shutil
import tempfile
from unittest.mock import patch

import pytest

from core.mesh_bundle import MeshBundle


@pytest.fixture()
def bundle():
    tmp = tempfile.mkdtemp(prefix="nsm_evo_neural_test_")
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_evolution_cycle_includes_neural_insight_when_available(bundle):
    """الحالة السعيدة: neural_refine_evolution_task ترجع اقتراحاً حقيقياً
    → يجب أن يظهر كمفتاح neural_insight في نتيجة الدورة."""
    with patch(
        "ai.node_think.neural_refine_evolution_task",
        return_value="ركّز على تقليل زمن استجابة التنفيذ",
    ):
        result = bundle.run_evolution_cycle()

    assert result.get("neural_insight") == "ركّز على تقليل زمن استجابة التنفيذ"


def test_evolution_cycle_omits_neural_insight_when_unavailable(bundle):
    """بلا مفتاح مزوّد مفتوح (الحالة الافتراضية الفعلية بلا توكنات) —
    neural_refine_evolution_task ترجع None — يجب ألا يظهر مفتاح
    neural_insight إطلاقاً في النتيجة (لا تلويث بقيمة None)."""
    with patch("ai.node_think.neural_refine_evolution_task", return_value=None):
        result = bundle.run_evolution_cycle()

    assert "neural_insight" not in result


def test_evolution_cycle_survives_neural_step_failure(bundle):
    """فشل غير متوقع داخل الخطوة العصبية (استثناء) يجب ألا يُسقط دورة
    التطوّر نفسها — اختيارية بالكامل، تماماً كما في LivingMeshNode."""
    with patch(
        "ai.node_think.neural_refine_evolution_task",
        side_effect=RuntimeError("fake neural failure"),
    ):
        result = bundle.run_evolution_cycle()

    assert "neural_insight" not in result
    assert "cycle_number" in result or "summary" in result  # الدورة أكملت فعلياً


def test_evolution_cycle_neural_step_real_end_to_end(bundle, monkeypatch):
    """اختبار تكاملي حقيقي بلا أي mock على node_think نفسها — فقط على طبقة
    HTTP (ai.llm_fallback._post_json)، للتأكد أن المسار الكامل (run_
    evolution_cycle → neural_refine_evolution_task → neural_think →
    LLMFallback(provider_override='hf') → HTTP) يعمل فعلياً من طرف لطرف."""
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf-key")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    def fake_post_json(url, payload, headers, timeout=15):
        return [{"generated_text": "تحسين فهرسة الذاكرة قصيرة المدى"}]

    with patch("ai.llm_fallback._post_json", side_effect=fake_post_json):
        result = bundle.run_evolution_cycle()

    assert result.get("neural_insight") == "تحسين فهرسة الذاكرة قصيرة المدى"


def test_evolution_cycle_neural_insight_reflects_real_cycle_summary(bundle):
    """تحقّق من أن diagnose المُمرَّر لـneural_refine_evolution_task يعكس
    فعلياً ملخّص الدورة الحقيقية (gaps_found/services_approved)، لا قيماً
    وهمية ثابتة."""
    captured = {}

    def fake_neural(node_id, base_task, diagnose=None, **kwargs):
        captured["node_id"] = node_id
        captured["base_task"] = base_task
        captured["diagnose"] = diagnose
        return None

    with patch("ai.node_think.neural_refine_evolution_task", side_effect=fake_neural):
        bundle.run_evolution_cycle()

    assert captured["node_id"] == "mesh_bundle_evolution"
    assert "دورة تطوّر" in captured["base_task"]
    assert "cycle_summary" in captured["diagnose"]
    assert isinstance(captured["diagnose"]["cycle_summary"], dict)
    assert "gaps_found" in captured["diagnose"]["cycle_summary"]
