# -*- coding: utf-8 -*-
"""اختبارات الاستخدام التلقائي للشبكة العصبية داخل دورة التطوّر الذاتي للعقدة
(ai/node_think.py + ai/living_mesh.py::maybe_self_evolve)."""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import patch

import pytest

from ai import node_think
from ai.node_think import neural_refine_evolution_task


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    node_think._auto_state.clear()
    for k in ("HUGGINGFACE_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "NSM_NODE_NEURAL_AUTO"):
        monkeypatch.delenv(k, raising=False)
    yield
    node_think._auto_state.clear()


def _hf(text="ركّز على اكتشاف الأقران"):
    return patch("ai.llm_fallback._post_json", return_value=[{"generated_text": text}])


def test_no_hf_key_returns_none_and_makes_no_call():
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        assert neural_refine_evolution_task("n1", "مهمة") is None


def test_with_hf_key_returns_clean_one_line(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with _hf("  ركّز\nعلى   اكتشاف الأقران  "):
        out = neural_refine_evolution_task("n1", "مهمة", {"status": {"evolution_score": 0.1}})
    assert out == "ركّز على اكتشاف الأقران"


def test_output_truncated_to_200(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with _hf("س" * 500):
        out = neural_refine_evolution_task("n1", "مهمة")
    assert out is not None and len(out) == 200


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    monkeypatch.setenv("NSM_NODE_NEURAL_AUTO", "0")
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        assert neural_refine_evolution_task("n1", "مهمة") is None


def test_rate_limited_per_node(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with _hf() as m:
        assert neural_refine_evolution_task("n1", "مهمة") is not None
        assert neural_refine_evolution_task("n1", "مهمة") is None  # ضمن الفترة
        assert neural_refine_evolution_task("n2", "مهمة") is not None  # عقدة أخرى
        assert m.call_count == 2


def test_failure_never_raises_and_backs_off(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", side_effect=RuntimeError("شبكة")):
        assert neural_refine_evolution_task("n1", "مهمة", min_interval=0.0) is None
    assert node_think._auto_state["n1"]["fails"] >= 1


def test_never_calls_paid_provider(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")

    def boom(*a, **k):
        raise AssertionError("استُدعي مزوّد مدفوع!")

    with patch("ai.llm_fallback.LLMFallback._call_anthropic", side_effect=boom), _hf():
        assert neural_refine_evolution_task("n1", "مهمة") is not None


def test_think_bound_on_living_node_and_auto_in_self_evolve(tmp_path, monkeypatch):
    import ai.living_mesh as lm
    d = Path(tmp_path)
    (d / "content").mkdir(exist_ok=True)
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    node = lm.LivingMeshNode(node_id="neural_auto", host="127.0.0.1", port=0)

    names = {t["name"] for t in node.hands.tools()["left"]}
    assert "think" in names

    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    captured = {}

    async def fake_exec(task_data):
        captured.update(task_data)

    node._execute_evolution = fake_exec
    with _hf("حسّن اكتشاف الأقران"):
        assert node.maybe_self_evolve(force=True) is True
    assert captured["neural_insight"] == "حسّن اكتشاف الأقران"
    assert captured["source"] == "self"


def test_self_evolve_unaffected_without_hf_key(tmp_path, monkeypatch):
    import ai.living_mesh as lm
    d = Path(tmp_path)
    (d / "content").mkdir(exist_ok=True)
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    node = lm.LivingMeshNode(node_id="neural_off", host="127.0.0.1", port=0)
    captured = {}

    async def fake_exec(task_data):
        captured.update(task_data)

    node._execute_evolution = fake_exec
    assert node.maybe_self_evolve(force=True) is True
    assert captured["neural_insight"] is None
