# -*- coding: utf-8 -*-
"""execute_inference: الشبكة العصبية المفتوحة أولاً تلقائياً، وبلا تغيير بدونها."""
from unittest.mock import patch

import pytest

from ai.mesh_task_protocol import execute_inference


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("HUGGINGFACE_API_KEY", "HF_TOKEN", "ANTHROPIC_API_KEY", "NSM_NODE_NEURAL_AUTO",
              "NSM_LLM_PROVIDER_PREF"):
        monkeypatch.delenv(k, raising=False)


def _hf(text="جواب من فالكون"):
    return patch("ai.llm_fallback._post_json", return_value=[{"generated_text": text}])


def test_uses_neural_when_hf_key(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with _hf():
        r = execute_inference({"prompt": "ما عاصمة اليمن؟", "task_id": "t1"})
    assert r["ok"] and r["output"] == "جواب من فالكون"
    assert r["used_real_llm"] is True and r["used_neural_open_source"] is True


def test_no_paid_provider_when_neural_succeeds(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic")
    with patch("ai.llm_fallback.LLMFallback._call_anthropic",
               side_effect=AssertionError("مزوّد مدفوع!")), _hf():
        r = execute_inference({"prompt": "سؤال"})
    assert r["used_neural_open_source"] is True


def test_without_key_old_path_unchanged():
    r = execute_inference({"prompt": "سؤال"})
    assert r["ok"] and r["used_neural_open_source"] is False


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    monkeypatch.setenv("NSM_NODE_NEURAL_AUTO", "0")
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        r = execute_inference({"prompt": "سؤال"})
    assert r["used_neural_open_source"] is False


def test_hf_failure_falls_back_not_raise(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", side_effect=RuntimeError("شبكة")):
        r = execute_inference({"prompt": "سؤال"})
    assert r["ok"] is True and r["used_neural_open_source"] is False


def test_image_modality_untouched(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        r = execute_inference({"prompt": "قطة", "modality": "image_desc"})
    assert r["ok"] and r["used_neural_open_source"] is False
