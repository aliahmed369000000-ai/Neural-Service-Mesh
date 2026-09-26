"""
اختبارات /process في api_server.py
=======================================
كان هذا الـendpoint معطّلاً بالكامل عملياً بخطأين متتاليين:
  1) `Engine()` بلا وسائط ترفع TypeError فوراً (registry/graph/storage
     إلزامية بلا قيمة افتراضية في core.engine.ExecutionEngine.__init__).
  2) حتى لو نجحت، `ExecutionEngine` لا تملك دالة `process()` أصلاً — فقط
     run_path/run_between/run_full_graph (تحقّقت بالبحث في core/engine.py).

هذا الملف يتحقق أن `/process` بعد الإصلاح:
  - يستخدم فعلياً registry/graph/storage المشتركة من core.mesh_bundle
    (عبر MeshBundle معزولة بمجلد بيانات مؤقت، حتى لا يلوّث الاختبار
    ملفات data/*.json الحقيقية المتتبَّعة في git).
  - يُنفّذ مساراً حقيقياً عبر run_path وينجح فعلياً على عقدة دور وكيل
    حقيقية مسجّلة في الـregistry.
  - يرفض الطلبات بلا مفتاح (fail-closed) وبلا حقل تنفيذ صالح (400)،
    ويتعامل مع node_id غير موجود بفشل نظيف (200 مع status=failed) بدل
    استثناء غير مُتحكَّم به.
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def isolated_bundle(monkeypatch):
    """MeshBundle حقيقية لكن بمجلد بيانات مؤقت — لا تلمس data/*.json
    المتتبَّعة في git، ولا الـsingleton الحقيقي (lru_cache) المشترك مع
    بقية التطبيق."""
    from core.mesh_bundle import MeshBundle

    tmp_dir = tempfile.mkdtemp(prefix="nsm_process_endpoint_test_")
    bundle = MeshBundle(storage_dir=tmp_dir, db_path=f"{tmp_dir}/mesh.db")
    monkeypatch.setattr("core.mesh_bundle.get_mesh_bundle", lambda: bundle)
    yield bundle
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("NSM_API_KEY", "nsm-test-key")
    import api_server
    from fastapi.testclient import TestClient
    return TestClient(api_server.app)


class TestProcessEndpoint:
    def test_fail_closed_without_key(self, client, isolated_bundle, monkeypatch):
        monkeypatch.delenv("NSM_API_KEY", raising=False)
        r = client.post("/process", json={"path": ["x"], "data": {}})
        assert r.status_code == 403

    def test_wrong_key_rejected(self, client, isolated_bundle):
        r = client.post(
            "/process",
            json={"path": ["x"], "data": {}},
            headers={"x-api-key": "wrong"},
        )
        assert r.status_code == 403

    def test_missing_execution_fields_returns_400(self, client, isolated_bundle):
        r = client.post("/process", json={"data": {}}, headers={"x-api-key": "nsm-test-key"})
        assert r.status_code == 400

    def test_run_path_executes_real_registered_node(self, client, isolated_bundle):
        # عقدة دور وكيل حقيقية مسجّلة فعلاً في الـregistry المعزولة —
        # لا تعتمد على أي مفتاح API خارجي أو حزمة mcp.
        role, node_id = next(iter(isolated_bundle.role_node_ids.items()))

        r = client.post(
            "/process",
            json={"path": [node_id], "data": {"task": "افحص العقد"}},
            headers={"x-api-key": "nsm-test-key"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        result = body["result"]
        assert result["status"] == "success"
        assert len(result["steps"]) == 1
        assert result["steps"][0]["node_id"] == node_id
        assert result["steps"][0]["status"] == "success"
        assert result["final_output"] == {"result": ""}

    def test_run_path_unknown_node_fails_cleanly(self, client, isolated_bundle):
        r = client.post(
            "/process",
            json={"path": ["no-such-node-id"], "data": {}},
            headers={"x-api-key": "nsm-test-key"},
        )
        # لا استثناء غير مُتحكَّم به (500) — فشل نظيف ومُبلَّغ به في النتيجة.
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["status"] == "failed"

    def test_run_between_uses_shared_graph(self, client, isolated_bundle):
        # الجذر متصل بكل الأدوار عبر _sync_nodes_to_graph — تحقّق أن
        # run_between يجد المسار فعلياً عبر نفس الرسم البياني المشترك.
        role, node_id = next(iter(isolated_bundle.role_node_ids.items()))
        root_id = isolated_bundle._root_node_id
        if node_id == root_id:
            pytest.skip("أول دور مسجّل هو نفسه الجذر — لا مسار مفيد للاختبار")

        r = client.post(
            "/process",
            json={"start_id": root_id, "end_id": node_id, "data": {"task": "t"}},
            headers={"x-api-key": "nsm-test-key"},
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["status"] == "success"
        assert result["path"][0] == root_id
        assert result["path"][-1] == node_id


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
