"""واجهة REST للعقد (api_nodes.py) على MeshBundle حقيقية عبر TestClient:
المصادقة fail-closed، القائمة/التفاصيل، تشغيل عقدة واحدة فعلياً (وتأثيرها على
عدّاد التنفيذ والشبكة العصبية)، الإيقاف/الاستئناف، السجل، وترتيب الشبكة."""
import tempfile

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import api_nodes
from core.mesh_bundle import MeshBundle
from core.node import BaseNode, NodeSchema

KEY = "k" * 24
H = {"X-API-Key": KEY}


class EchoNode(BaseNode):
    input_schema = NodeSchema(fields={"text": "str"}, required=["text"])
    output_schema = NodeSchema(fields={"out": "str"}, required=[])

    def process(self, data):
        return {"out": data["text"].strip()}


@pytest.fixture()
def env(monkeypatch):
    monkeypatch.setenv("NSM_API_KEY", KEY)
    monkeypatch.delenv("NSM_ADMIN_KEY", raising=False)
    tmp = tempfile.mkdtemp()
    bundle = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    node = EchoNode("echo_api_test")
    bundle.registry.register(node)
    bundle.graph.add_node(node.node_id, node.to_dict())
    app = FastAPI()
    app.include_router(api_nodes.router)
    app.dependency_overrides[api_nodes.get_bundle] = lambda: bundle
    return TestClient(app), bundle, node


def test_auth_fail_closed(env, monkeypatch):
    c, _, node = env
    for method, path in (("get", "/nodes"), ("get", f"/nodes/{node.node_id}"),
                         ("post", f"/nodes/{node.node_id}/execute"),
                         ("get", "/nodes/neural/ranking")):
        assert getattr(c, method)(path).status_code == 403
        assert getattr(c, method)(path, headers={"X-API-Key": "wrong" * 5}).status_code == 403
    assert c.get("/nodes", headers={"X-API-Key": "مفتاح".encode("utf-8")}).status_code == 403  # غير ASCII لا يسبّب 500
    monkeypatch.delenv("NSM_API_KEY")
    assert c.get("/nodes", headers=H).status_code == 403  # لا مفتاح مضبوط = مغلق
    monkeypatch.setenv("NSM_ADMIN_KEY", "a" * 20)
    assert c.get("/nodes", headers={"X-API-Key": "a" * 20}).status_code == 200


def test_list_filter_and_detail(env):
    c, _, node = env
    r = c.get("/nodes", headers=H).json()
    assert r["count"] == len(r["nodes"]) and any(n["node_id"] == node.node_id for n in r["nodes"])
    assert [n["node_id"] for n in c.get("/nodes?state=paused", headers=H).json()["nodes"]] == []
    d = c.get(f"/nodes/{node.node_id}", headers=H).json()
    assert d["name"] == "echo_api_test" and "input_schema" in d
    assert 0.0 <= d["neural"]["predicted_success"] <= 1.0 and d["is_quarantined"] is False
    assert c.get("/nodes/nope", headers=H).status_code == 404


def test_execute_real_node_updates_counters_and_neural(env):
    c, bundle, node = env
    obs0 = bundle.neural_advisor.summary()["observations"]
    r = c.post(f"/nodes/{node.node_id}/execute", headers=H, json={"data": {"text": "  مرحبا  "}})
    body = r.json()
    assert r.status_code == 200 and body["status"] == "success"
    assert body["result"]["final_output"]["out"] == "مرحبا"
    assert node.to_dict()["execution_count"] == 1
    assert bundle.neural_advisor.summary()["observations"] > obs0  # الشبكة تعلّمت من التشغيل
    run_id = body["result"]["run_id"]
    assert c.get(f"/nodes/runs/{run_id}", headers=H).json()["run_id"] == run_id
    assert c.get(f"/nodes/{node.node_id}/runs", headers=H).json()["count"] >= 1
    assert c.get("/nodes/runs/missing", headers=H).status_code == 404


def test_execute_bad_input_reports_failure_not_500(env):
    c, _, node = env
    r = c.post(f"/nodes/{node.node_id}/execute", headers=H, json={"data": {}})
    assert r.status_code == 200 and r.json()["status"] != "success"
    assert c.post(f"/nodes/{node.node_id}/execute", headers=H, json={"data": "x"}).status_code == 422
    assert c.post("/nodes/nope/execute", headers=H, json={}).status_code == 404


def test_pause_resume_cycle(env):
    c, _, node = env
    p = c.post(f"/nodes/{node.node_id}/pause", headers=H, json={"reason": "maintenance"}).json()
    assert p["state"] == "paused" and p["pause_reason"] == "maintenance"
    assert [n["node_id"] for n in c.get("/nodes?state=paused", headers=H).json()["nodes"]] == [node.node_id]
    assert c.post(f"/nodes/{node.node_id}/pause", headers=H, json={"reason": ""}).status_code == 422
    r = c.post(f"/nodes/{node.node_id}/resume", headers=H).json()
    assert r["resumed"] is True and r["state"] != "paused" and r["reputation_quarantined"] is False


def test_neural_ranking_shape_and_limit(env):
    c, _, _ = env
    r = c.get("/nodes/neural/ranking?limit=3", headers=H).json()
    assert len(r["ranking"]) <= 3 and "observations" in r["advisor"]
    scores = [x["predicted_success"] for x in r["ranking"]]
    assert scores == sorted(scores, reverse=True)
    assert c.get("/nodes/neural/ranking?limit=0", headers=H).status_code == 422
