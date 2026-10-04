"""
اختبار: core/mesh_bundle.py — MemoryEngine كانت تُبنى بلا استدعاء
set_knowledge_store() إطلاقاً (تحقّقت: لا مكان آخر في المشروع يستدعيها
لها). MemoryEngine تعمل صحيحة بدونه (تخزينها الأساسي عبر SQLite)، لكن
تصديرها التكميلي الحقيقي إلى knowledge/route_memory.json عبر
learn_from_run() (upsert_route/append_route_execution — كلاهما مكتوب
بالكامل في ai/memory_engine.py) كان يبقى no-op دائماً لأن self._knowledge
تبقى None. النتيجة: ai/routing_engine.py المصمَّم صراحة ليقرأ من نفس
الملف (بحسب docstring الملف نفسه) لا يجد فيه أي بيانات مسارات حقيقية
إطلاقاً مهما تراكم من تنفيذات فعلية.

هذا الاختبار يتحقق أن MeshBundle الحقيقية تسلك memory_engine بـ
knowledge_store فعلياً، وأن learn_from_run() يكتب مساراً حقيقياً إلى
knowledge/route_memory.json (لا يبقى بمخططه الافتراضي الفارغ).
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle


@pytest.fixture()
def storage_paths():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_memory_knowledge_test_")
    try:
        yield tmp_dir, f"{tmp_dir}/mesh.db"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_memory_engine_is_wired_to_the_real_knowledge_store(storage_paths):
    storage_dir, db_path = storage_paths
    bundle = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    assert bundle.memory_engine._knowledge is bundle.knowledge_store, (
        "MemoryEngine._knowledge ما زالت غير مربوطة بـMeshBundle.knowledge_store"
    )


def test_learn_from_run_writes_a_real_route_into_knowledge_store(storage_paths):
    storage_dir, db_path = storage_paths
    bundle = MeshBundle(storage_dir=storage_dir, db_path=db_path)

    run_result = {
        "run_id": "test-run-1",
        "status": "success",
        "total_duration_ms": 42.0,
        "path": ["node-a", "node-b"],
        "steps": [
            {"node_id": "node-a", "node_name": "A", "status": "success", "duration_ms": 10.0},
            {"node_id": "node-b", "node_name": "B", "status": "success", "duration_ms": 32.0},
        ],
    }
    bundle.memory_engine.learn_from_run(run_result)

    route_memory = bundle.knowledge_store._read_raw(bundle.knowledge_store.ROUTE_MEMORY)
    routes = route_memory.get("routes", {})
    assert routes, "route_memory.json ما زال فارغاً بعد learn_from_run() فعلي"
    matching = [r for key, r in routes.items() if key == "node-a->node-b"]
    assert matching, f"لم أجد المسار المتوقَّع في {list(routes.keys())}"
    assert matching[0]["runs"] >= 1
