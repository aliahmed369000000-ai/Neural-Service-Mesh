# -*- coding: utf-8 -*-
"""نموذج البرمجة المفتوح للعقد (ai/code_think.py): تبديل/حظر/رصيد/دمج."""
import urllib.error
from unittest.mock import patch

import pytest

from ai import code_think as ct


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("HUGGINGFACE_API_KEY", "HF_TOKEN", "NSM_CODE_MODELS", "NSM_NODE_NEURAL_AUTO",
              "ANTHROPIC_API_KEY", "NSM_HF_CHAT_MODEL"):
        monkeypatch.delenv(k, raising=False)
    ct._reset_state()
    yield
    ct._reset_state()


def _ok(text="def f():\n    return 1"):
    return {"choices": [{"message": {"content": text}}]}


def _err(code):
    return urllib.error.HTTPError("https://router.huggingface.co", code, "x", {}, None)


def test_no_key_no_call_no_raise():
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        r = ct.code_think("اكتب دالة")
    assert r["ok"] is False and r["reason"] == "no_hf_key"


def test_success_uses_strongest_model_first(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    seen = []

    def fake(url, payload, headers, timeout=15):
        seen.append((url, payload["model"], headers["Authorization"]))
        return _ok()

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        r = ct.code_think("اكتب دالة تجمع رقمين")
    assert r["ok"] and r["model"] == ct.DEFAULT_CODE_MODELS[0]
    assert r["used_open_source_model"] is True
    assert seen[0][0] == "https://router.huggingface.co/v1/chat/completions"
    assert seen[0][2] == "Bearer fake-hf"


def test_unsupported_model_skipped_and_remembered(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    calls = []

    def fake(url, payload, headers, timeout=15):
        calls.append(payload["model"])
        if payload["model"] == ct.DEFAULT_CODE_MODELS[0]:
            raise _err(404)
        return _ok()

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        r1 = ct.code_think("fix this bug")
        r2 = ct.code_think("fix this bug")
    assert r1["model"] == ct.DEFAULT_CODE_MODELS[1]
    assert calls.count(ct.DEFAULT_CODE_MODELS[0]) == 1  # لم يُعَد تجريب المحظور
    assert r2["ok"]


def test_credits_depleted_blocks_everything(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    n = {"c": 0}

    def fake(*a, **k):
        n["c"] += 1
        raise _err(402)

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        r1 = ct.code_think("python bug")
        r2 = ct.code_think("python bug")
    assert r1["ok"] is False and r2["ok"] is False
    assert n["c"] == 1  # توقّف فوراً ولم يستنزف بقية النماذج ولا الطلب الثاني
    assert r2["reason"].startswith("blocked_until_")
    assert ct.code_available() is False


def test_rate_limit_cools_only_that_model(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")

    def fake(url, payload, headers, timeout=15):
        if payload["model"] == ct.DEFAULT_CODE_MODELS[0]:
            raise _err(429)
        return _ok()

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        r = ct.code_think("refactor this")
    assert r["ok"] and r["model"] == ct.DEFAULT_CODE_MODELS[1]
    assert ct.code_available() is True


def test_all_fail_returns_not_ok_never_raises(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", side_effect=RuntimeError("شبكة")):
        r = ct.code_think("python error")
    assert r["ok"] is False and r["text"] == ""


def test_env_override_models(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    monkeypatch.setenv("NSM_CODE_MODELS", "a/b, c/d")
    seen = []

    def fake(url, payload, headers, timeout=15):
        seen.append(payload["model"])
        return _ok()

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        ct.code_think("python")
    assert seen == ["a/b"]


def test_input_validation(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with pytest.raises(ValueError):
        ct.code_think("   ")
    with pytest.raises(ValueError):
        ct.code_think("س" * (ct.MAX_PROMPT_CHARS + 1))


@pytest.mark.parametrize("text,expected", [
    ("اكتب لي دالة بايثون لفرز قائمة", True),
    ("fix this: ```python\nprint(1)\n```", True),
    ("TypeError: 'NoneType' object is not callable", True),
    ("def add(a, b):\n  return a+b  # why slow?", True),
    ("ما عاصمة اليمن؟", False),
    ("let me know what time it is", False),
    ("ما حكم الصلاة في السفر؟", False),
])
def test_is_code_prompt(text, expected):
    assert ct.is_code_prompt(text) is expected


def test_code_first_generate_gates(monkeypatch):
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        assert ct.code_first_generate("ما عاصمة اليمن؟") is None  # ليس برمجياً
        monkeypatch.setenv("NSM_NODE_NEURAL_AUTO", "0")
        assert ct.code_first_generate("اكتب دالة بايثون") is None  # معطَّل


def test_execute_inference_routes_code_to_code_model(monkeypatch):
    from ai.mesh_task_protocol import execute_inference
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", return_value=_ok("print('hi')")):
        r = execute_inference({"prompt": "اكتب كود بايثون يطبع hi"})
    assert r["used_code_model"] == ct.DEFAULT_CODE_MODELS[0]
    assert r["used_neural_open_source"] is True and r["output"] == "print('hi')"


def test_execute_inference_general_question_not_sent_to_code_model(monkeypatch):
    from ai.mesh_task_protocol import execute_inference
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    seen = []

    def fake(url, payload, headers, timeout=15):
        seen.append(payload["model"])
        return _ok("صنعاء")

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        r = execute_inference({"prompt": "ما عاصمة اليمن؟"})
    assert r["used_code_model"] is None
    assert all(m not in ct.DEFAULT_CODE_MODELS for m in seen)


def test_execute_inference_code_credits_out_falls_to_old_path(monkeypatch):
    from ai.mesh_task_protocol import execute_inference
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    with patch("ai.llm_fallback._post_json", side_effect=_err(402)):
        r = execute_inference({"prompt": "اكتب كود بايثون"})
    assert r["ok"] is True and r["used_code_model"] is None


def test_think_code_bound_on_both_node_types(tmp_path, monkeypatch):
    import shutil, tempfile
    from pathlib import Path
    from core.mesh_bundle import MeshBundle
    from core.node_hands import LEFT
    tmp = tempfile.mkdtemp(prefix="nsm_code_")
    try:
        b = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
        node = b.registry.get(list(b.role_node_ids.values())[0])
        assert "think_code" in {t["name"] for t in node.hands.tools()[LEFT]}
        res = node.use_hand(LEFT, "think_code", prompt="python")
        assert res.ok and res.output["ok"] is False  # بلا مفتاح: لا انهيار
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    import ai.living_mesh as lm
    d = Path(tmp_path)
    (d / "content").mkdir()
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    n = lm.LivingMeshNode(node_id="code_node", host="127.0.0.1", port=0)
    assert "think_code" in {t["name"] for t in n.hands.tools()[LEFT]}
