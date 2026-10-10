# -*- coding: utf-8 -*-
"""Groq (gpt-oss-120b) أولاً للعقد، وHugging Face احتياطاً — عام وبرمجي."""
import urllib.error
from unittest.mock import patch

import pytest

from ai import code_think as ct
from ai import node_think as nt
from ai.mesh_task_protocol import execute_inference

GROQ = "https://api.groq.com/openai/v1/chat/completions"
HF = "https://router.huggingface.co/v1/chat/completions"


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("HUGGINGFACE_API_KEY", "HF_TOKEN", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
              "NSM_NODE_NEURAL_AUTO", "NSM_CODE_MODELS", "NSM_GROQ_CODE_MODELS",
              "NSM_LLM_PROVIDER_PREF", "NSM_HF_CHAT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    ct._reset_state()
    nt._auto_state.clear()
    yield
    ct._reset_state()


def _ok(text="جواب"):
    return {"choices": [{"message": {"content": text}}]}


def _err(code, headers=None):
    return urllib.error.HTTPError("u", code, "x", headers or {}, None)


def _router(groq=None, hf=None, seen=None):
    def fake(url, payload, headers, timeout=15):
        if seen is not None:
            seen.append((url, payload.get("model"), payload.get("reasoning_effort")))
        h = groq if url == GROQ else hf
        if h is None:
            raise AssertionError(f"استدعاء غير متوقع: {url}")
        if isinstance(h, Exception):
            raise h
        return h
    return fake


def test_available_with_only_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    assert nt.neural_available() and ct.code_available()


def test_general_uses_groq_first_not_hf(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_fake")
    seen = []
    with patch("ai.llm_fallback._post_json", side_effect=_router(groq=_ok("صنعاء"), seen=seen)):
        r = nt.neural_first_generate("ما عاصمة اليمن؟")
    assert r and r["text"] == "صنعاء" and r["provider"] == "groq"
    assert [u for u, *_ in seen] == [GROQ]  # لم يُلمس HF


def test_general_falls_back_to_hf_when_groq_fails(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_fake")
    with patch("ai.llm_fallback._post_json", side_effect=_router(groq=_err(500), hf=_ok("من HF"))):
        r = nt.neural_first_generate("سؤال عام")
    assert r and r["text"] == "من HF" and r["provider"] == "huggingface"


def test_general_none_when_all_fail(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    with patch("ai.llm_fallback._post_json", side_effect=_router(groq=_err(500))):
        assert nt.neural_first_generate("سؤال عام") is None  # CKG ليس شبكة عصبية


def test_code_uses_groq_gpt_oss_low_reasoning(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_fake")
    seen = []
    with patch("ai.llm_fallback._post_json", side_effect=_router(groq=_ok("print(1)"), seen=seen)):
        r = ct.code_think("اكتب كود بايثون")
    assert r["ok"] and r["provider"] == "groq" and r["model"] == "openai/gpt-oss-120b"
    assert seen == [(GROQ, "openai/gpt-oss-120b", "low")]


def test_code_groq_429_next_groq_model_then_hf(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_fake")
    seen = []
    with patch("ai.llm_fallback._post_json",
               side_effect=_router(groq=_err(429, {"retry-after": "30"}), hf=_ok("من HF"), seen=seen)):
        r = ct.code_think("python bug")
    models = [(u, m) for u, m, _ in seen]
    assert models[0] == (GROQ, "openai/gpt-oss-120b") and models[1] == (GROQ, "llama-3.3-70b-versatile")
    assert models[2][0] == HF and r["ok"] and r["provider"] == "huggingface-router"
    # التبريد يحترم retry-after (30ث) لا 5 دقائق كاملة
    import time
    left = ct._model_blocked_until["groq:openai/gpt-oss-120b"] - time.time()
    assert 20 < left <= 31


def test_code_groq_bad_key_blocks_groq_only(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_bad")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_fake")
    seen = []
    with patch("ai.llm_fallback._post_json",
               side_effect=_router(groq=_err(401), hf=_ok("x"), seen=seen)):
        r1 = ct.code_think("python bug")
        r2 = ct.code_think("python bug")
    groq_calls = [u for u, *_ in seen if u == GROQ]
    assert len(groq_calls) == 1               # حُظر Groq بعد 401 فلم يُعَد
    assert r1["ok"] and r2["ok"] and r2["provider"] == "huggingface-router"


def test_execute_inference_end_to_end_groq_only(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    with patch("ai.llm_fallback._post_json", side_effect=_router(groq=_ok("def f(): pass"))):
        rc = execute_inference({"prompt": "اكتب كود بايثون لدالة"})
    assert rc["used_code_model"] == "openai/gpt-oss-120b" and rc["used_neural_open_source"] is True
    with patch("ai.llm_fallback._post_json", side_effect=_router(groq=_ok("صنعاء"))):
        rg = execute_inference({"prompt": "ما عاصمة اليمن؟"})
    assert rg["output"] == "صنعاء" and rg["used_code_model"] is None
    assert rg["used_neural_open_source"] is True


def test_no_paid_provider_leak(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    with patch("ai.llm_fallback.LLMFallback._call_anthropic",
               side_effect=AssertionError("مزوّد مدفوع!")), \
         patch("ai.llm_fallback._post_json", side_effect=_router(groq=_ok("ok"))):
        assert nt.neural_first_generate("سؤال") is not None
        assert ct.code_first_generate("اكتب كود بايثون") is not None


def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("NSM_NODE_NEURAL_AUTO", "0")
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        assert nt.neural_first_generate("سؤال") is None
        assert ct.code_first_generate("اكتب كود بايثون") is None
