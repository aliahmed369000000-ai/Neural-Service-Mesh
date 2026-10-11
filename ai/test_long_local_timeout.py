# -*- coding: utf-8 -*-
"""مهلة النموذج المحلي الأطول (180ث) بأمان: لا تجميد لحلقة الأحداث، ومهل متناسقة."""
import ast
import asyncio
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from ai import task_timeouts as tt


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for k in ("NSM_LOCAL_TIMEOUT_SEC", "NSM_LOCAL_LLM_URL", "NSM_TASK_TIMEOUT_SEC"):
        monkeypatch.delenv(k, raising=False)


def test_default_is_180_not_60():
    assert tt.local_timeout_sec() == 180.0


@pytest.mark.parametrize("val,expected", [("300", 300.0), ("1", 5.0), ("99999", 1800.0),
                                           ("abc", 180.0), ("", 180.0)])
def test_env_override_clamped(monkeypatch, val, expected):
    monkeypatch.setenv("NSM_LOCAL_TIMEOUT_SEC", val)
    assert tt.local_timeout_sec() == expected


def test_remote_timeout_raised_only_for_llm_with_local(monkeypatch):
    assert tt.remote_task_timeout("inference_request", 15.0) == 15.0      # لا نموذج محلي
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", "http://x:11434")
    assert tt.remote_task_timeout("inference_request", 15.0) == 200.0     # 180+20
    assert tt.remote_task_timeout("inference_request", 300.0) == 300.0    # لا يخفض صريحاً أكبر
    assert tt.remote_task_timeout("search_request", 15.0) == 15.0         # غير لغوي
    monkeypatch.setenv("NSM_TASK_TIMEOUT_SEC", "42")
    assert tt.remote_task_timeout("anything", 15.0) == 42.0


def test_node_think_and_code_think_share_same_timeout(monkeypatch):
    from ai import code_think, node_think
    monkeypatch.setenv("NSM_LOCAL_TIMEOUT_SEC", "240")
    assert node_think.local_timeout_sec() == 240.0 and code_think.local_timeout_sec() == 240.0


def test_llmfallback_local_default_path_uses_configured_timeout(monkeypatch):
    from ai import llm_fallback as lf
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", "http://127.0.0.1:1")
    monkeypatch.setenv("NSM_LOCAL_TIMEOUT_SEC", "400")
    seen = {}

    def fake(url, payload, headers, timeout=15):
        seen["t"] = timeout
        return {"message": {"content": "ok"}}

    with patch("ai.llm_fallback._post_json", side_effect=fake):
        llm = lf.LLMFallback(timeout=10)
        llm._model = "m"
        llm._call_local("q", [])
    assert seen["t"] == 400


def test_dispatch_mesh_task_default_timeout_resolved(tmp_path, monkeypatch):
    import ai.living_mesh as lm
    d = Path(tmp_path)
    (d / "content").mkdir()
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    monkeypatch.setenv("NSM_LOCAL_LLM_URL", "http://x:11434")
    node = lm.LivingMeshNode(node_id="t", host="127.0.0.1", port=0)
    seen = {}

    async def fake_req(host, port, kind, data, timeout=15.0, hops=0, expect_result_kind=None):
        seen["t"] = timeout
        return {"ok": True, "result": {}}

    node.request_from_peer = fake_req
    asyncio.run(node.dispatch_mesh_task("127.0.0.1", 1, "inference_request", {}))
    assert seen["t"] == 200.0
    asyncio.run(node.dispatch_mesh_task("127.0.0.1", 1, "inference_request", {}, timeout=7.0))
    assert seen["t"] == 7.0


def test_long_task_does_not_freeze_event_loop(tmp_path, monkeypatch):
    """استدلال متزامن طويل داخل submit_verifiable_task(local=True) لا يجمّد الحلقة."""
    import ai.living_mesh as lm
    from ai import mesh_task_protocol as mt
    from ai.node_health_layer import NodeHealthLayer
    d = Path(tmp_path)
    (d / "content").mkdir()
    monkeypatch.setattr(lm, "LIVING_MESH_DIR", d)
    monkeypatch.setattr(lm, "NETWORK_STATE", d / "network_state.json")
    monkeypatch.setattr(lm, "CONTENT_DIR", d / "content")
    node = lm.LivingMeshNode(node_id="t2", host="127.0.0.1", port=0)
    health = NodeHealthLayer(node)

    def slow(kind, payload):
        time.sleep(1.2)          # يحاكي استدلالاً محلياً بطيئاً (حجب متزامن)
        return {"ok": True, "output": "x"}

    ticks = []
    stop = []

    async def heartbeat():
        while not stop:
            ticks.append(time.time())
            await asyncio.sleep(0.05)

    async def main():
        hb = asyncio.create_task(heartbeat())
        await asyncio.sleep(0.3)          # دع النبضات تبدأ قبل المهمة
        with patch.object(mt, "dispatch_task", side_effect=slow):
            res = await health.submit_verifiable_task("inference_request", {"prompt": "q"}, local=True)
        await asyncio.sleep(0.2)
        stop.append(1)
        await hb
        return res

    res = asyncio.run(main())
    assert res["ok"] is True
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    # مع الحجب المتزامن تظهر فجوة ≈1.2ث؛ مع الخيط تبقى الفجوات ≈50ms
    assert max(gaps) < 0.6, f"الحلقة جُمِّدت: أكبر فجوة {max(gaps):.2f}ث"


def test_no_sync_dispatch_task_inside_async_functions():
    """حارس انحدار: لا استدعاء مباشر لـdispatch_task داخل دالة async إلا عبر to_thread."""
    root = Path(__file__).resolve().parent
    offenders = []
    for name in ("living_mesh.py", "node_health_layer.py", "node_launcher.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for fn in ast.walk(tree):
            if not isinstance(fn, ast.AsyncFunctionDef):
                continue
            for call in ast.walk(fn):
                if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                        and call.func.attr == "dispatch_task"):
                    offenders.append(f"{name}:{call.lineno} in {fn.name}")
    assert offenders == [], offenders
