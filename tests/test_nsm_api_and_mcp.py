"""
اختبارات طبقة API وMCP لمشروع NSM
==================================
تغطي:
1) نقاط REST الجديدة للوكلاء والسرب (api_server.py) عبر FastAPI TestClient:
   - fail-closed: بدون NSM_API_KEY كل نقطة ترد 403
   - بمفتاح خاطئ ترد 403، وبمفتاح صحيح ترد 200 ببنية صحيحة
2) أدوات MCP الجديدة (mcp_server/server.py) باستدعاء مباشر — ثبتنا أنها
   مسجّلة في FastMCP وأن استدعاءها يعمل ببيانات حية (بدون مفاتيح خارجية).

لا تُشغّل هذه الاختبارات خادم Uvicorn فعليًا ولا تتطلب أي مفتاح API.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EVENTS_KEY = "events"


def _emit_fake_events():
    """حقن أحداث مزيفة في ناقل الأحداث (نظيفة — test-scoped events)."""
    from ai import agent_event_bus as bus
    bus.emit_event("agent_run_start", agent_id="tester-agent",
                   title="اختبار")
    bus.emit_event("agent_run_end", agent_id="tester-agent",
                   title="اختبار", status="done", detail="نجح")


def _client_with_key(key_env: str = ""):
    """استيراد api_server نظيفًا بمفتاح NSM_API_KEY مضبوط (أو فارغ)."""
    import api_server  # noqa: F401  (استيراد متكرر آمن: نقاط مسجلة مسبقًا)
    from fastapi.testclient import TestClient
    old = os.environ.get("NSM_API_KEY")
    if key_env:
        os.environ["NSM_API_KEY"] = key_env
    elif old is not None:
        del os.environ["NSM_API_KEY"]
    client = TestClient(api_server.app)
    yield client
    if old is None:
        os.environ.pop("NSM_API_KEY", None)
    else:
        os.environ["NSM_API_KEY"] = old


def _get(client, path, key=None):
    headers = {"x-api-key": key} if key else {}
    return client.get(path, headers=headers)


# ── 1) Fail-closed: بدون مفتاح، كل النقاط الجديدة ترد 403 ─────────────

class TestApiKeyFailClosed:
    """عند غياب NSM_API_KEY من البيئة تُرفض كل نقاط الوكلاء/السرب بـ403
    (مبدأ fail-closed المتّبع في /process)."""
    _PATHS = ["/agents/states", "/agents/events", "/swarm/dashboard",
              "/swarm/alerts", "/swarm/long-horizon", "/mesh/history",
              "/performance/system"]

    @pytest.fixture(autouse=True)
    def _clear_key(self, monkeypatch):
        monkeypatch.delenv("NSM_API_KEY", raising=False)

    def test_all_nsm_paths_reject_without_key(self):
        import api_server
        from fastapi.testclient import TestClient
        c = TestClient(api_server.app)
        for path in self._PATHS:
            r = c.get(path)
            assert r.status_code == 403, path

    def test_wrong_key_rejected(self, monkeypatch):
        import api_server
        from fastapi.testclient import TestClient
        monkeypatch.setenv("NSM_API_KEY", "real-secret-123")
        c = TestClient(api_server.app)
        for path in self._PATHS:
            r = c.get(path, headers={"x-api-key": "wrong"})
            assert r.status_code == 403, path


# ── 2) النقاط ترد 200 ببنية صحيحة مع مفتاح صحيح ─────────────────────

class TestNsmApiEndpoints:
    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch):
        monkeypatch.setenv("NSM_API_KEY", "nsm-test-key")
        _emit_fake_events()

    @pytest.fixture()
    def client(self, monkeypatch):
        # إعادة ضبط المفتاح لكل test لضمان العزل
        monkeypatch.setenv("NSM_API_KEY", "nsm-test-key")
        import api_server
        from fastapi.testclient import TestClient
        return TestClient(api_server.app)

    def test_agents_states_ok(self, client):
        r = _get(client, "/agents/states", "nsm-test-key")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert isinstance(data["agents"], dict)
        assert "tester-agent" in data["agents"]

    def test_agents_events_default_limit(self, client):
        r = _get(client, "/agents/events", "nsm-test-key")
        assert r.status_code == 200
        events = r.json()["events"]
        assert isinstance(events, list)
        assert len(events) >= 2

    def test_agents_events_clamped_limit(self, client):
        import api_server
        from fastapi.testclient import TestClient
        c = TestClient(api_server.app)
        r = c.get("/agents/events?limit=1",
                  headers={"x-api-key": "nsm-test-key"})
        assert r.status_code == 200
        assert len(r.json()["events"]) <= 1

    def test_swarm_dashboard_ok(self, client):
        r = _get(client, "/swarm/dashboard", "nsm-test-key")
        assert r.status_code == 200
        snap = r.json()["dashboard"]
        assert set(snap) >= {"agents", "swarm", "long_horizon",
                             "performance", "alerts", "alert_rules",
                             "auto_actions"}
        assert snap["agents"]["counts"]["done"] >= 1

    def test_swarm_alerts_ok(self, client):
        r = _get(client, "/swarm/alerts", "nsm-test-key")
        assert r.status_code == 200
        assert isinstance(r.json()["alerts"], list)

    def test_swarm_apply_actions_post_ok(self, client):
        import api_server
        from fastapi.testclient import TestClient
        c = TestClient(api_server.app)
        r = c.post("/swarm/apply-actions",
                   headers={"x-api-key": "nsm-test-key"})
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        assert isinstance(data["applied"], list)

    def test_long_horizon_ok(self, client):
        r = _get(client, "/swarm/long-horizon", "nsm-test-key")
        assert r.status_code == 200
        data = r.json()["long_horizon"]
        assert "tasks" in data and "counts" in data

    def test_mesh_history_ok(self, client):
        r = _get(client, "/mesh/history", "nsm-test-key")
        assert r.status_code == 200
        data = r.json()
        assert data["ok"] is True
        summary = data["summary"]
        assert isinstance(summary, dict)
        # MeshBundle يعيد ملخصًا مجمعًا: تقييم/ذاكرة/سمعة/exec_log —
        # لا قائمة سجلات فردية (verify من core/mesh_bundle.py نفسه)
        assert "scoring" in summary
        assert "memory" in summary
        assert "reputation" in summary
        assert "exec_log" in summary

    def test_performance_system_ok(self, client):
        r = _get(client, "/performance/system", "nsm-test-key")
        assert r.status_code == 200
        sys_perf = r.json()["system"]
        assert "memory_used_mb" in sys_perf
        assert "peak_rss_mb" in sys_perf


# ── 3) أدوات MCP الجديدة مسجّلة وقابلة للاستدعاء ─────────────────────

class TestMcpTools:
    @pytest.fixture(autouse=True)
    def _setup(self):
        _emit_fake_events()

    def _call(self, name, **kwargs):
        import json
        import mcp_server.server as ms
        fn = ms.mcp._tool_manager._tools[name].fn
        return json.loads(fn(**kwargs))

    def test_nsm_agent_states_registered_and_working(self):
        data = self._call("nsm_agent_states")
        assert "tester-agent" in data["agents"]
        assert data["agents"]["tester-agent"]["status"] == "done"

    def test_nsm_swarm_dashboard_registered_and_working(self):
        data = self._call("nsm_swarm_dashboard")
        assert isinstance(data, dict)
        assert set(data) >= {"agents", "swarm", "alerts", "performance"}

    def test_nsm_alerts_evaluate_registered_and_working(self):
        data = self._call("nsm_alerts_evaluate")
        assert isinstance(data["alerts"], list)
        assert isinstance(data["applied_actions"], list)

    def test_nsm_long_horizon_tasks_with_filter(self):
        data = self._call("nsm_long_horizon_tasks",
                          status_filter="done")
        assert isinstance(data["tasks"], list)
        for task in data["tasks"]:
            assert (task.get("status") or "").lower() == "done"


class TestAgentContract:
    def test_health_aliases_are_available(self):
        import api_server
        from fastapi.testclient import TestClient
        c = TestClient(api_server.app)
        assert c.get("/health").status_code == 200
        assert c.get("/healthz").status_code == 200

    def test_agent_task_requires_authentication(self, monkeypatch):
        import api_server
        from fastapi.testclient import TestClient
        monkeypatch.setenv("NSM_API_KEY", "contract-test-key")
        c = TestClient(api_server.app)
        response = c.post("/api/agent/tasks", json={"task": "اختبار", "url": "https://example.com"})
        assert response.status_code == 403
        assert "error" in response.json()

    def test_agent_task_rejects_invalid_public_url(self, monkeypatch):
        import api_server
        from fastapi.testclient import TestClient
        monkeypatch.setenv("NSM_API_KEY", "contract-test-key")
        c = TestClient(api_server.app)
        response = c.post(
            "/api/agent/tasks",
            headers={"x-api-key": "contract-test-key"},
            json={"task": "اختبار", "url": "http://127.0.0.1:5000"},
        )
        assert response.status_code == 400

    def test_chat_rejects_empty_message(self, monkeypatch):
        import api_server
        from fastapi.testclient import TestClient
        monkeypatch.setenv("NSM_API_KEY", "contract-test-key")
        c = TestClient(api_server.app)
        response = c.post("/agents/chat", headers={"x-api-key": "contract-test-key"}, json={})
        assert response.status_code == 400
        assert response.json()["ok"] is False
