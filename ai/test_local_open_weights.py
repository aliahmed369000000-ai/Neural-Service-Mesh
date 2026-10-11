# -*- coding: utf-8 -*-
"""نموذج مفتوح الأوزان محلي (Ollama / OpenAI-compatible) أولاً — عبر HTTP حقيقي
على خادم وهمي محلي (لا نموذج حقيقي في بيئة الاختبار)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

import pytest

from ai import code_think as ct
from ai import llm_fallback as lf
from ai import node_think as nt
from ai.mesh_task_protocol import execute_inference

_real_post = lf._post_json


class _Server:
    def __init__(self, style):
        self.hits = []
        outer = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.hits.append((self.path, body))
                if self.path == "/api/chat":
                    out = {"message": {"role": "assistant", "content": f"[ollama] {body['model']}"}}
                elif self.path == "/v1/chat/completions":
                    out = {"choices": [{"message": {"content": f"[openai] {body['model']}"}}]}
                else:
                    self.send_response(404); self.end_headers(); return
                raw = json.dumps(out).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.port}" + ("/v1" if style == "openai" else "")

    def close(self):
        self.httpd.shutdown()


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("HUGGINGFACE_API_KEY", "HF_TOKEN", "GROQ_API_KEY", "ANTHROPIC_API_KEY",
              "NSM_LOCAL_LLM_URL", "NSM_LOCAL_MODEL", "NSM_LOCAL_CODE_MODEL",
              "NSM_NODE_NEURAL_AUTO", "NSM_LLM_PROVIDER_PREF", "NSM_LOCAL_TIMEOUT_SEC"):
        monkeypatch.delenv(k, raising=False)
    ct._reset_state()
    nt._auto_state.clear()
    yield
    ct._reset_state()


@pytest.fixture(params=["ollama", "openai"])
def server(request):
    s = _Server(request.param)
    yield s
    s.close()


def test_available_with_only_local(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    assert nt.neural_available() and ct.code_available()


def test_general_answer_from_local_model(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("NSM_LOCAL_MODEL", "qwen-test")
    r = nt.neural_first_generate("ما عاصمة اليمن؟")
    assert r and r["provider"] == "local" and "qwen-test" in r["text"]
    assert len(server.hits) == 1


def test_local_first_even_when_groq_hf_keys_exist(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")
    monkeypatch.setenv("HUGGINGFACE_API_KEY", "hf_fake")
    with patch("ai.llm_fallback._post_json", side_effect=lambda url, *a, **k:
               _real_post(url, *a, **k) if url.startswith("http://127.0.0.1")
               else (_ for _ in ()).throw(AssertionError("استُدعي مزوّد سحابي: " + url))):
        r = nt.neural_first_generate("سؤال عام")
    assert r and r["provider"] == "local"


def test_code_uses_local_code_model(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("NSM_LOCAL_MODEL", "general-m")
    monkeypatch.setenv("NSM_LOCAL_CODE_MODEL", "coder-m")
    r = ct.code_think("اكتب كود بايثون")
    assert r["ok"] and r["provider"] == "local" and r["model"] == "coder-m"
    assert "coder-m" in r["text"]
    path, body = server.hits[0]
    assert body["messages"][0]["role"] == "system" and "software engineer" in body["messages"][0]["content"]


def test_code_model_defaults_to_local_model(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("NSM_LOCAL_MODEL", "only-m")
    assert ct.code_think("python bug")["model"] == "only-m"


def test_local_down_falls_to_groq(monkeypatch):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", "http://127.0.0.1:9")   # لا خادم
    monkeypatch.setenv("GROQ_API_KEY", "gsk_fake")

    def fake(url, payload, headers, timeout=15):
        if url.startswith("http://127.0.0.1"):
            return _real_post(url, payload, headers, 3)
        return {"choices": [{"message": {"content": "من Groq"}}]}

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        r = ct.code_think("python bug")
        g = nt.neural_first_generate("سؤال عام")
    assert r["ok"] and r["provider"] == "groq"
    assert g and g["provider"] == "groq"
    # الخادم المحلي المعطّل يُبرَّد فلا يُعاد تجريبه فوراً في الكود
    assert ct._model_blocked_until.get("local:qwen2.5:7b-instruct-q4_K_M", 0) > 0


def test_execute_inference_end_to_end_local(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("NSM_LOCAL_MODEL", "gen-m")
    monkeypatch.setenv("NSM_LOCAL_CODE_MODEL", "code-m")
    rc = execute_inference({"prompt": "اكتب كود بايثون لدالة"})
    assert rc["used_code_model"] == "code-m" and rc["used_neural_open_source"] is True
    rg = execute_inference({"prompt": "ما عاصمة اليمن؟"})
    assert "gen-m" in rg["output"] and rg["used_code_model"] is None
    assert rg["used_neural_open_source"] is True


def test_override_local_is_accepted_and_never_leaks_paid(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    with patch("ai.llm_fallback.LLMFallback._call_anthropic",
               side_effect=AssertionError("مزوّد مدفوع!")):
        res = lf.LLMFallback(provider_override="local", timeout=10).generate("سؤال")
    assert res.provider.value == "local"


def test_disabled_by_env(monkeypatch, server):
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", server.url)
    monkeypatch.setenv("NSM_NODE_NEURAL_AUTO", "0")
    assert nt.neural_first_generate("سؤال") is None and ct.code_first_generate("اكتب كود بايثون") is None
    assert server.hits == []
