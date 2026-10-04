"""
اختبار: DiscoveryEngine كانت مكتوبة بالكامل ولا تُبنى في أي مكان، فكان
knowledge/node_profiles.json فارغاً دائماً، وKnowledgeStore.update_node_execution_stats()
(التي تستدعيها MemoryEngine بعد كل تنفيذ) no-op لأنها تتطلب profile مسجَّلاً.

يتحقق: (1) MeshBundle تبني DiscoveryEngine مربوطة بنفس KnowledgeStore،
(2) كل عقدة مسجَّلة تُعلَن وتظهر في node_profiles.json،
(3) learn_from_run() يحدّث execution_stats فعلياً لعقدة معلَنة،
(4) عقدة self_evolved تُعلَن عبر register_node،
(5) إعادة التشغيل لا تُعيد إعلان المُعلَنة (announced_at ثابت).
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from services.dynamic_node import PassThroughNode


@pytest.fixture()
def paths():
    d = tempfile.mkdtemp(prefix="nsm_discovery_wiring_")
    try:
        yield d, f"{d}/mesh.db"
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _profiles(bundle):
    return bundle.knowledge_store._read_raw(bundle.knowledge_store.NODE_PROFILES).get("nodes", {})


def test_discovery_engine_is_wired(paths):
    d, db = paths
    b = MeshBundle(storage_dir=d, db_path=db)
    assert b.discovery_engine._knowledge is b.knowledge_store


def test_registered_nodes_get_profiles(paths):
    d, db = paths
    b = MeshBundle(storage_dir=d, db_path=db)
    profiles = _profiles(b)
    ids = {n.node_id for n in b.registry.list_all()}
    assert ids and ids <= set(profiles), "عُقد مسجَّلة بلا profile في node_profiles.json"


def test_execution_stats_reach_node_profile(paths):
    d, db = paths
    b = MeshBundle(storage_dir=d, db_path=db)
    node_id = next(iter(b.registry.list_all())).node_id
    b.memory_engine.learn_from_run({
        "run_id": "r1", "status": "success", "total_duration_ms": 5.0,
        "path": [node_id],
        "steps": [{"node_id": node_id, "node_name": "n", "status": "success", "duration_ms": 5.0}],
    })
    stats = _profiles(b)[node_id].get("execution_stats", {})
    assert stats.get("total_executions", 0) >= 1


def test_self_evolved_node_is_announced(paths):
    d, db = paths
    b = MeshBundle(storage_dir=d, db_path=db)
    nid = b.register_node(PassThroughNode("announce-me"))
    assert b.discovery_engine.get_announcement(nid) is not None
    assert nid in _profiles(b)


def test_restart_keeps_original_announced_at(paths):
    d, db = paths
    b1 = MeshBundle(storage_dir=d, db_path=db)
    nid = next(iter(b1.registry.list_all())).node_id
    first = b1.discovery_engine.get_announcement(nid).announced_at
    b2 = MeshBundle(storage_dir=d, db_path=db)
    assert b2.discovery_engine.get_announcement(nid).announced_at == first
