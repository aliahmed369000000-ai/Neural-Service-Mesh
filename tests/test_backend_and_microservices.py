"""
NSM Backend Layer + External Connectors + Microservices — اختبارات
===================================================================
- طبقة Backend Layer (ai/backend_layer.py): CRUD كامل للوكلاء والمهام
  والذاكرة والرسائل ومخزن KV على قاعدة SQLite معزولة (tmp_path).
- الموصلات الخارجية (connectors/external_services.py): المغلف الموحد
  (envelope)، OTP بصلاحية زمنية، التوليد العشوائي للـ OTP (عزل RNG)،
  والموصلات غير المسجّلة.
- الخدمات المصغرة (ai/microservices.py): عقد الطلب/الاستجابة الثابت
  (nsm-ms/1.0)، الاكتشاف، استدعاء handlers الافتراضية، والخدمات
  المجهولة.
- نقاط REST API الجديدة في api_server.py: fail-closed (403 بدون
  NSM_API_KEY أو بمفتاح خاطئ) ثم CRUD فعلي بالمفتاح الصحيح.

لا مفاتيح API حقيقية مطلوبة — كل شيء معزول محليًا.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest import mock

import pytest

_ROOT = __import__("pathlib").Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def _data_dirs(monkeypatch):
    """عزل مسارات بيانات Backend Layer والموصلات والمicroservices (SQLite
    وملفات JSON) في tmp_path — لا يلمس .nsm_data الخاصة بالحقيقة."""
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "nsm_test.db")
    monkeypatch.setattr("ai.backend_layer.db_path", lambda: db)
    # الموصلات الخارجية تحفظ سجلاتها في _DATA_DIR
    import connectors.external_services as ext
    monkeypatch.setattr(ext, "_DATA_DIR", tmp, raising=False)
    # خدمات الميكرو (معلومات connectors محفوظة في JSON)
    import ai.microservices as ms
    monkeypatch.setattr(ms, "_DATA_DIR", tmp, raising=False)
    yield tmp


@pytest.fixture(autouse=True)
def _isolate_event_bus(monkeypatch):
    """لا نريدها تطلق أحداثًا حقيقية في ناقل events المشترك."""
    from ai import microservices
    monkeypatch.setattr(microservices, "_emit_service_event",
                        lambda *a, **kw: None)


# ═══════════════════════════════════════════════════════════════════════════
# طبقة Backend Layer (SQLite)
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendStore:
    def test_kv_roundtrip(self, _data_dirs):
        from ai import backend_layer as bl
        assert bl.kv_set("k1", {"v": 2}, "swarm")["ok"]
        assert bl.kv_get("k1", "swarm") == {"v": 2}
        assert bl.kv_get("missing", "swarm", default=7) == 7
        assert bl.kv_delete("k1", "swarm")["ok"]
        assert bl.kv_get("k1", "swarm") is None

    def test_kv_domain_isolation_and_list(self, _data_dirs):
        from ai import backend_layer as bl
        bl.kv_set("a", 1, "d1")
        bl.kv_set("a", 2, "d2")
        assert bl.kv_get("a", "d1") == 1
        assert bl.kv_get("a", "d2") == 2
        names = [r["key"] for r in bl.kv_list("d1")]
        assert "a" in names

    def test_agents_crud(self, _data_dirs):
        from ai import backend_layer as bl
        assert bl.agent_register("agent-x", "researcher")["ok"]
        agents = bl.agent_list()
        assert any(a["id"] == "agent-x" for a in agents)
        assert bl.agent_update("agent-x", {"role": "coder"})["ok"]
        assert bl.agent_get("agent-x")["role"] == "coder"
        assert bl.agent_unregister("agent-x")["ok"]
        assert bl.agent_get("agent-x") is None

    def test_tasks_create_update_list(self, _data_dirs):
        from ai import backend_layer as bl
        res = bl.task_create("جمع بيانات", "collection", {"src": "f"})
        assert res["ok"] and res["id"].startswith("task_")
        tid = res["id"]
        assert bl.task_update(tid, {"status": "running"})["ok"]
        tasks = [t for t in bl.task_list() if t["id"] == tid]
        assert tasks and tasks[0]["title"] == "جمع بيانات"

    def test_memory_search(self, _data_dirs):
        from ai import backend_layer as bl
        bl.memory_add("surah_chain", "شبكة قرآنية d_model=128", ["ai"], 0.9)
        hits = bl.memory_search("d_model")
        assert hits and "d_model=128" in hits[0]["content"]

    def test_messages_inbox(self, _data_dirs):
        from ai import backend_layer as bl
        r = bl.message_send("agent-1", "main", "تقرير", "جاهز", None)
        assert r["ok"] and r["id"].startswith("msg_")
        inbox = bl.message_inbox("main")
        assert any(m["subject"] == "تقرير" for m in inbox)
        assert bl.message_mark_read(inbox[0]["id"])["ok"]

    def test_backend_counts(self, _data_dirs):
        from ai import backend_layer as bl
        c = bl.backend_counts()
        assert isinstance(c, dict) and "agents" in c
        assert all(isinstance(v, int) for v in c.values())
        assert c.get("agents", 0) >= 0


# ═══════════════════════════════════════════════════════════════════════════
# الموصلات الخارجية (دفع / خرائط / رسائل)
# ═══════════════════════════════════════════════════════════════════════════

class TestExternalConnectors:
    def test_connector_list_and_describe(self, _data_dirs):
        from connectors.external_services import list_connectors, describe_connector
        names = {c["name"] for c in list_connectors()}
        assert {"payment", "maps", "sms"} <= names
        for name in names:
            d = describe_connector(name)
            assert d["ok"] and d["actions"]

    def test_payment_create_checkout_envelope(self, _data_dirs):
        from connectors.external_services import call_connector
        r = call_connector("payment", "create_payment",
                           {"amount": 100, "currency": "USD"})
        assert r["ok"] and r["service"] == "payment" and r["simulated"]
        assert r["latency_ms"] >= 0 and "request_id" in r
        assert r["payload_raw"]["amount"] == 100

    def test_maps_geocode(self, _data_dirs):
        from connectors.external_services import call_connector
        r = call_connector("maps", "geocode", {"address": "الرياض"})
        assert r["ok"] and r["simulated"]

    def test_sms_send_otp_ttl(self, _data_dirs):
        from connectors.external_services import call_connector
        r = call_connector("sms", "send_otp", {"to": "+966500000000"})
        assert r["ok"] and "code" in r["result"]
        res = r["result"]
        r2 = call_connector("sms", "verify_otp",
                            {"to": res["to"], "code": res["code"]})
        assert r2["ok"]
        # رمز خاطئ
        r3 = call_connector("sms", "verify_otp",
                            {"to": res["to"], "code": "0000"})
        assert not r3["ok"]
        # إرسال SMS نصي عادي
        r4 = call_connector("sms", "send_sms",
                            {"to": "+1", "message": "نص تجريبي"})
        assert r4["ok"]

    def test_unknown_connector_fails_open_false(self, _data_dirs):
        from connectors.external_services import call_connector
        r = call_connector("unknown-svc", "ping", {})
        assert not r["ok"] and "غير مسجل" in r["error"]

    def test_otp_randomness_seeded(self, _data_dirs):
        """التوليد العشوائي لـ OTP قابل للعزل (توقعات اختبارية)."""
        from connectors.external_services import call_connector
        with mock.patch("connectors.external_services.random.randint",
                        return_value=123456):
            r = call_connector("sms", "send_otp", {"to": "+1"})
            assert r["ok"] and r["result"]["code"] == "123456"


# ═══════════════════════════════════════════════════════════════════════════
# الخدمات المصغرة (عقد الطلب/الاستجابة الثابت)
# ═══════════════════════════════════════════════════════════════════════════

class TestMicroserviceBus:
    def test_default_services_registered(self, _data_dirs):
        from ai.microservices import list_services
        svcs = list_services()
        for s in ("meta", "harm", "ckg", "dashboard", "connectors",
                  "backend"):
            assert s in svcs, s

    def test_response_contract(self, _data_dirs):
        from ai.microservices import call_service
        r = call_service("backend", "counts")
        assert r["ok"]
        assert r["schema_version"] == "nsm-ms/1.0"
        assert r["service"] == "backend" and r["action"] == "counts"
        assert isinstance(r["result"], dict)

    def test_harm_classify(self, _data_dirs):
        from ai.microservices import call_service
        r = call_service("harm", "classify",
                         {"text": "رسالة عادية آمنة"})
        assert r["ok"] and "result" in r

    def test_dashboard_snapshot(self, _data_dirs):
        from ai.microservices import call_service
        r = call_service("dashboard", "snapshot")
        assert r["service"] == "dashboard" and r["action"] == "snapshot"
        assert r["schema_version"] == "nsm-ms/1.0"
        # اللقطة قد تفشل داخليًا خارج سياق Streamlit — العقد فقط مطلوب

    def test_connectors_list_via_service(self, _data_dirs):
        from ai.microservices import call_service
        r = call_service("connectors", "list")
        assert r["ok"] and "payment" in (
            {c["name"] for c in r["result"]["connectors"]})

    def test_unknown_service_error(self, _data_dirs):
        from ai.microservices import call_service
        r = call_service("nonexistent", "ping")
        assert not r["ok"] and "schema_version" in r

    def test_unknown_action_error(self, _data_dirs):
        from ai.microservices import call_service
        r = call_service("meta", "bogus_action_x")
        assert not r["ok"]

    def test_backend_task_flow(self, _data_dirs):
        from ai.microservices import call_service
        c = call_service("backend", "task_create",
                         {"title": "اختبار", "type": "general"})
        assert c["ok"]
        s = call_service("backend", "kv_set",
                         {"key": "ms_k", "value": 7})
        assert s["ok"]
        g = call_service("backend", "kv_get", {"key": "ms_k"})
        assert g["ok"] and g["result"]["value"] == 7


# ═══════════════════════════════════════════════════════════════════════════
# نقاط REST API الجديدة (fail-closed + CRUD حي)
# ═══════════════════════════════════════════════════════════════════════════

class TestBackendAndServicesApi:
    @pytest.fixture(autouse=True)
    def _api_env(self, _data_dirs, monkeypatch):
        monkeypatch.setenv("NSM_API_KEY", "test-secret-key")

    def _client(self):
        from fastapi.testclient import TestClient
        import api_server
        return TestClient(api_server.app)

    def test_fail_closed_no_key(self):
        r = self._client().get("/services/list")
        assert r.status_code == 403

    def test_fail_closed_wrong_key(self):
        r = self._client().get(
            "/services/list", headers={"X-API-Key": "wrong"})
        assert r.status_code == 403

    def test_services_list_ok(self):
        r = self._client().get(
            "/services/list", headers={"X-API-Key": "test-secret-key"})
        assert r.status_code == 200
        assert "meta" in r.json()["services"]

    def test_services_call_contract(self):
        r = self._client().post(
            "/services/call",
            headers={"X-API-Key": "test-secret-key"},
            json={"service": "meta", "action": "list_services"})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] and body["schema_version"] == "nsm-ms/1.0"

    def test_connectors_list_and_call(self):
        c = self._client()
        h = {"X-API-Key": "test-secret-key"}
        r = c.get("/connectors/list", headers=h)
        assert r.status_code == 200 and r.json()["ok"]
        r = c.post("/connectors/call", headers=h,
                   json={"service": "sms", "action": "send_otp",
                         "payload": {"to": "+1"}})
        assert r.status_code == 200 and r.json()["ok"]

    def test_backend_kv_crud(self):
        c = self._client()
        h = {"X-API-Key": "test-secret-key"}
        r = c.post("/backend/kv", headers=h,
                   json={"key": "api_k", "value": 42, "domain": "test"})
        assert r.status_code == 200 and r.json()["ok"]
        r = c.get("/backend/kv", headers=h,
                  params={"key": "api_k", "domain": "test"})
        assert r.status_code == 200 and r.json()["value"] == 42
        r = c.delete("/backend/kv", headers=h,
                     params={"key": "api_k", "domain": "test"})
        assert r.status_code == 200 and r.json()["ok"]

    def test_backend_agents_endpoint(self):
        c = self._client()
        h = {"X-API-Key": "test-secret-key"}
        r = c.post("/backend/agents", headers=h,
                   json={"id": "api-agent", "role": "tester"})
        assert r.status_code == 200 and r.json()["ok"]
        r = c.get("/backend/agents", headers=h)
        assert r.status_code == 200
        data = r.json()
        rows = data["agents"] if isinstance(data, dict) and "agents" in data else data
        assert any(a["id"] == "api-agent" for a in rows)

    def test_backend_tasks_endpoint(self):
        c = self._client()
        h = {"X-API-Key": "test-secret-key"}
        r = c.post("/backend/tasks", headers=h,
                   json={"title": "api-task", "type": "general"})
        assert r.status_code == 200 and r.json()["ok"]
        r = c.get("/backend/tasks", headers=h)
        assert r.status_code == 200
        assert any(t["title"] == "api-task" for t in r.json()["tasks"]) if isinstance(r.json(), dict) and "tasks" in r.json() else any(t["title"] == "api-task" for t in r.json())

    def test_backend_memories_endpoint(self):
        c = self._client()
        h = {"X-API-Key": "test-secret-key"}
        r = c.post("/backend/memories", headers=h,
                   json={"subject": "م", "content": "ذكرى اختبارية"})
        assert r.status_code == 200 and r.json()["ok"]
        r = c.get("/backend/memories", headers=h)
        assert r.status_code == 200
        data = r.json()
        rows = data.get("memories", data) if isinstance(data, dict) else data
        assert any("ذكرى اختبارية" in (m.get("content", "") or "")
                   for m in rows)

    def test_backend_messages_endpoint(self):
        c = self._client()
        h = {"X-API-Key": "test-secret-key"}
        r = c.post("/backend/messages", headers=h,
                   json={"sender": "s", "receiver": "r",
                         "subject": "hi", "body": "ok"})
        assert r.status_code == 200 and r.json()["ok"]
        r = c.get("/backend/messages", headers=h,
                  params={"receiver": "r"})
        assert r.status_code == 200
        data = r.json()
        rows = data.get("messages", data) if isinstance(data, dict) else data
        assert any(mg.get("subject") == "hi" for mg in rows)
