# -*- coding: utf-8 -*-
"""الجسر اليدوي (ai/claude_bridge.py): طابور ملفات غير حاجب."""
import json
from unittest.mock import patch

import pytest

from ai import claude_bridge as cb
from ai.mesh_task_protocol import execute_inference


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for k in ("HUGGINGFACE_API_KEY", "HF_TOKEN", "NSM_CLAUDE_BRIDGE", "GROQ_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("NSM_BRIDGE_DIR", str(tmp_path / "bridge"))
    yield


def test_disabled_does_no_io(tmp_path):
    assert cb.bridge_lookup_or_enqueue("سؤال") is None
    assert not (tmp_path / "bridge").exists()


def test_enqueue_dedupes_and_lists(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    a = cb.enqueue("ما   هو\nالمعنى؟", node_id="n1")
    b = cb.enqueue("ما هو المعنى؟", node_id="n2")
    assert a == b
    pend = cb.list_pending()
    assert len(pend) == 1 and pend[0]["prompt"] == "ما هو المعنى؟"


def test_answer_roundtrip_and_validation(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    pid = cb.enqueue("سؤال؟")
    assert cb.answer("../../etc/passwd", "x") is False   # لا اجتياز مسارات
    assert cb.answer("0" * 16, "x") is False             # غير معلّق
    assert cb.answer(pid, "   ") is False                # فارغ
    assert cb.answer(pid, "جواب") is True
    assert cb.list_pending() == []
    assert cb.lookup("سؤال؟") == "جواب"


def test_limits(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    pid = cb.enqueue("س" * 5000)
    q = json.loads((cb.bridge_dir() / "inbox" / f"{pid}.json").read_text(encoding="utf-8"))
    assert len(q["prompt"]) == cb.MAX_PROMPT
    assert cb.answer(pid, "ج" * 9000)
    assert len(cb.lookup("س" * 5000)) == cb.MAX_ANSWER


def test_pending_cap(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    monkeypatch.setattr(cb, "MAX_PENDING", 3)
    ids = [cb.enqueue(f"q{i}") for i in range(5)]
    assert ids[:3] == [cb.prompt_id(f"q{i}") for i in range(3)] and ids[3:] == [None, None]


def test_inference_enqueues_then_uses_answer(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    r1 = execute_inference({"prompt": "ما عاصمة اليمن؟", "node_id": "node_7"})
    assert r1["ok"] and r1["used_claude_bridge"] is False   # غير حاجب: أكمل بمساره
    pend = cb.list_pending()
    assert len(pend) == 1 and pend[0]["node_id"] == "node_7"
    assert cb.answer(pend[0]["id"], "صنعاء")
    r2 = execute_inference({"prompt": "ما عاصمة اليمن؟"})
    assert r2["output"] == "صنعاء" and r2["used_claude_bridge"] is True
    assert r2["used_neural_open_source"] is False and r2["used_real_llm"] is True


def test_bridge_answer_skips_other_models(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "fake-hf")
    cb.enqueue("اكتب كود بايثون")
    cb.answer(cb.prompt_id("اكتب كود بايثون"), "print(1)")
    with patch("ai.llm_fallback._post_json", side_effect=AssertionError("لا استدعاء")):
        r = execute_inference({"prompt": "اكتب كود بايثون"})
    assert r["output"] == "print(1)" and r["used_claude_bridge"] is True


def test_inference_unchanged_when_disabled(tmp_path):
    r = execute_inference({"prompt": "سؤال عام"})
    assert r["ok"] and r["used_claude_bridge"] is False
    assert not (tmp_path / "bridge").exists()


def test_image_modality_not_bridged(monkeypatch):
    monkeypatch.setenv("NSM_CLAUDE_BRIDGE", "1")
    r = execute_inference({"prompt": "قطة", "modality": "image_desc"})
    assert r["ok"] and r["used_claude_bridge"] is False and cb.list_pending() == []
