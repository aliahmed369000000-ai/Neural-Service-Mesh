"""
اختبار: core/registry.py — NodeRegistry._load() كان يملأ meta_cache
(بيانات وصفية) فقط، وليس self._nodes (كائنات Python حيّة). عُقد الكتالوج
الثابت (أدوار/أدوات MCP) تُعاد بناؤها بنوعها الصحيح دائماً عند إقلاع
MeshBundle، لكن أي عقدة أخرى — خصوصاً عُقد self_evolved التي ينشئها
EvolutionEngine أثناء دورة تطوّر ذاتي حقيقية — كانت تبقى "شبحاً" بعد أي
إعادة تشغيل: موجودة في meta_cache/exec_log، غائبة تماماً عن
registry.list_all()/get_by_state()، فلا تظهر في ServiceGraph، ولا يقدر
أي كود حجر/رفع حجر بالسمعة (self.registry.get(node_id)) أن يجدها.

هذا الاختبار يتحقق مباشرة من الإصلاح (NodeRegistry.orphaned_metadata +
MeshBundle._restore_dynamic_nodes):
1. عقدة self_evolved مسجَّلة في عملية أولى تختفي من self._nodes في عملية
   جديدة قبل إحيائها — orphaned_metadata() يلتقطها بالضبط.
2. بعد بناء MeshBundle ثانية بنفس التخزين، العقدة تظهر فعلياً في
   registry.get(node_id) و registry.list_all() و registry.get_by_state()
   بنفس node_id ونفس الحالة/عدد مرات التنفيذ المحفوظة.
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node import NodeState
from services.dynamic_node import PassThroughNode


@pytest.fixture()
def storage_paths():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_dynamic_node_restore_test_")
    try:
        yield tmp_dir, f"{tmp_dir}/mesh.db"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_self_evolved_node_is_a_ghost_before_restore_fix_is_visible_in_metadata(storage_paths):
    """يوثّق نفسه فعلياً: قبل الإصلاح كانت meta_cache تحتوي العقدة بينما
    self._nodes (وبالتالي list_all/get_by_state) لا يعرفانها بعد
    إعادة تشغيل — orphaned_metadata() تكشف بالضبط هذا الفرق."""
    storage_dir, db_path = storage_paths
    bundle1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    node = PassThroughNode("ghost-check-node")
    node.record_execution(success=True)  # تاريخ حقيقي يجب أن يُستعاد لاحقاً
    node_id = bundle1.registry.register(node)

    # عملية ثانية بنفس التخزين لكن *بدون* استدعاء _restore_dynamic_nodes
    # (محاكاة الحالة قبل الإصلاح): بناء registry نيء مباشرة.
    from core.registry import NodeRegistry
    from storage.file_storage import FileStorage
    raw_registry = NodeRegistry(FileStorage(storage_dir=storage_dir))
    assert raw_registry.get(node_id) is None, "قبل الإحياء: العقدة غائبة عن self._nodes"
    orphans = raw_registry.orphaned_metadata()
    assert any(m.get("node_id") == node_id for m in orphans), (
        "orphaned_metadata() لم تلتقط العقدة المحفوظة غير الحيّة"
    )


def test_self_evolved_node_is_revived_after_mesh_bundle_restart(storage_paths):
    storage_dir, db_path = storage_paths

    bundle1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    node = PassThroughNode("evolved-revival-node")
    node.record_execution(success=True)
    node.record_execution(success=True)
    node_id = bundle1.registry.register(node)
    assert bundle1.registry.get(node_id) is not None

    bundle2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)

    revived = bundle2.registry.get(node_id)
    assert revived is not None, "العقدة self_evolved لم تُحيَ بعد إعادة تشغيل MeshBundle"
    assert revived.node_id == node_id
    assert revived.name == "evolved-revival-node"
    assert revived.state == NodeState.ACTIVE
    assert revived._execution_count == 2

    assert any(n.node_id == node_id for n in bundle2.registry.list_all())
    assert any(n.node_id == node_id for n in bundle2.registry.get_by_state(NodeState.ACTIVE))
    # لا orphans متبقية بعد الإحياء
    assert not any(m.get("node_id") == node_id for m in bundle2.registry.orphaned_metadata())
