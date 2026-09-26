"""
اختبار ربط AIDecisionLayer (ai/decision.py) بمسار الإنتاج الحقيقي
====================================================================
كانت AIDecisionLayer مكتوبة بالكامل (اختيار مسار مُقيَّم، اقتراح بديل
لعقدة فاشلة/محجورة، تعلّم بسيط من تاريخ التنفيذ) لكن لا يوجد أي مكان في
المشروع يبنيها فعلياً (تحقّقت بالبحث عن "AIDecisionLayer(" في كل الملفات
قبل هذا التعديل — لا نتيجة سوى تعريف الكلاس نفسه). api_server.py كان
يبني core.engine.ExecutionEngine بلا `ai=` إطلاقاً، فيبقى self._ai=None
دائماً في أي طلب حقيقي — ما يعني أن منطق fallback في run_path (بما فيه
إصلاح العقدة المحجورة في الكوميت السابق) لم يكن يعمل أبداً في الإنتاج،
وrun_between كان يستخدم BFS البسيط دائماً بدل اختيار مسار مُقيَّم.

هذا الملف يتحقق، عبر MeshBundle حقيقية (بمجلد بيانات مؤقت) وTestClient
حقيقي لـapi_server.app:
1. bundle.ai_decision موجودة فعلاً، من نوع AIDecisionLayer، ومربوطة
   بنفس bundle.graph الحيّ (لا نسخة منفصلة فارغة).
2. عقدة الجذر (وهي العقدة الوحيدة ذات حواف خارجة فعلياً — بقية أدوار
   العُقد بلا حواف خارجة إطلاقاً حسب _sync_nodes_to_graph) عند حجرها
   (node.pause()، تماماً كما تفعل MeshBundle._apply_reputation_feedback
   فعلياً) وطلبها عبر /process الحقيقي (HTTP كامل، لا استدعاء مباشر
   لـExecutionEngine) → تُستخدَم عقدة بديلة تلقائياً (is_fallback=True)
   بدل فشل الطلب بالكامل — هذا يثبت أن إصلاح الكوميت السابق أصبح فعلياً
   قابلاً للوصول من طرف إلى طرف، وليس فقط عبر استدعاء مباشر لـEngine في
   اختبار معزول.
3. run_between عبر /process الحقيقي يُعيد ai_suggested=True الآن (كان
   دائماً False قبل هذا التعديل لأن self._ai كان None).
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def isolated_bundle(monkeypatch):
    from core.mesh_bundle import MeshBundle

    tmp_dir = tempfile.mkdtemp(prefix="nsm_ai_decision_wiring_test_")
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


class TestAIDecisionWiring:
    def test_bundle_exposes_real_ai_decision_bound_to_live_graph(self, isolated_bundle):
        from ai.decision import AIDecisionLayer

        assert isinstance(isolated_bundle.ai_decision, AIDecisionLayer)
        assert isolated_bundle.ai_decision._graph is isolated_bundle.graph

    def test_paused_root_node_falls_back_end_to_end_via_http(self, client, isolated_bundle):
        # عقدة الجذر هي الوحيدة المضمون أن لها حواف خارجة (نحو كل الأدوار
        # الأخرى) حسب _sync_nodes_to_graph — أدوار غير الجذر بلا حواف
        # خارجة إطلاقاً، فلن يكون لديها أي بديل ممكن لاختبار الحالة الإيجابية.
        root_id = isolated_bundle._root_node_id
        assert len(isolated_bundle.role_node_ids) >= 2, "يلزم أكثر من دور واحد لاختبار بديل حقيقي"

        root_node = isolated_bundle.registry.get(root_id)
        root_node.pause()  # يحاكي حجراً حقيقياً بالسمعة المنخفضة

        r = client.post(
            "/process",
            json={"path": [root_id], "data": {"task": "افحص العقد"}},
            headers={"x-api-key": "nsm-test-key"},
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["status"] == "success", result
        step = result["steps"][0]
        assert step["is_fallback"] is True
        assert step["node_id"] != root_id

    def test_run_between_reports_ai_suggested_via_http(self, client, isolated_bundle):
        root_id = isolated_bundle._root_node_id
        other_role, other_id = next(
            (r, n) for r, n in isolated_bundle.role_node_ids.items() if n != root_id
        )
        r = client.post(
            "/process",
            json={"start_id": root_id, "end_id": other_id, "data": {"task": "ت"}},
            headers={"x-api-key": "nsm-test-key"},
        )
        assert r.status_code == 200
        result = r.json()["result"]
        assert result["ai_suggested"] is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
