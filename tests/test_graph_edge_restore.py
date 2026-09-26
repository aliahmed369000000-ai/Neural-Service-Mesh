"""
اختبار: core/mesh_bundle.py — _sync_nodes_to_graph() كانت تبني
ServiceGraph فقط من طوبولوجيا الكتالوج الثابتة ("mesh_member": الجذر →
كل دور/أداة)، بينما register_node() (عُقد self_evolved) وrecord_swarm_result()
(روابط نتائج السرب) كانا يحفظان الروابط الديناميكية فعلياً عبر
exec_log.upsert_connection() (storage/db.py) — لكن exec_log.list_connections()
المقابلة لم تكن تُستدعى من أي مكان لإعادة بناء الرسم البياني منها.

النتيجة العملية قبل الإصلاح: أي إعادة تشغيل للعملية (كما يحدث فعلياً
على Streamlit Cloud) كانت تُفرغ ServiceGraph من كل رابط ديناميكي مكتسَب
قبل إعادة التشغيل مباشرة، رغم بقاء الرابط محفوظاً فعلياً في SQLite.

هذا الاختبار يبني MeshBundle حقيقياً، يسجّل عقدة self_evolved متصلة
بالجذر (register_node بلا connect_to)، ثم يبني MeshBundle *ثانية* بنفس
مجلد التخزين/قاعدة البيانات (محاكاة إعادة تشغيل حقيقية) ويتحقق أن
الرابط الديناميكي مُستعاد في الرسم البياني الجديد بلا أي استدعاء إضافي
لـregister_node.
"""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from services.dynamic_node import PassThroughNode


@pytest.fixture()
def storage_paths():
    tmp_dir = tempfile.mkdtemp(prefix="nsm_graph_restore_test_")
    try:
        yield tmp_dir, f"{tmp_dir}/mesh.db"
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def test_self_evolved_edge_persists_across_mesh_bundle_restart(storage_paths):
    storage_dir, db_path = storage_paths

    bundle1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    root_id = bundle1._root_node_id
    assert root_id, "لازم يوجد عقدة جذر بعد تسجيل الأدوار الأولي"

    new_node = PassThroughNode("evolved-test-node")
    new_node_id = bundle1.register_node(new_node)  # connect_to=None → يتصل بالجذر

    # قبل إعادة التشغيل: الرابط موجود في نفس الرسم البياني الحي.
    assert new_node_id in bundle1.graph.get_neighbors(root_id)
    # ويُفترض أنه محفوظ فعلياً في SQLite (وليس فقط في الذاكرة).
    persisted = bundle1.exec_log.list_connections()
    assert any(
        c["source_id"] == root_id and c["target_id"] == new_node_id
        for c in persisted
    ), "الرابط self_evolved لم يُحفظ فعلياً عبر exec_log.upsert_connection"

    # محاكاة إعادة تشغيل حقيقية: MeshBundle جديدة تماماً بنفس التخزين.
    bundle2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    assert bundle2._root_node_id == root_id, "معرّف الجذر يجب أن يبقى ثابتاً بين العمليتين"
    assert bundle2.graph.has_node(new_node_id), "العقدة نفسها يجب أن تُستعاد من registry"

    restored_neighbors = bundle2.graph.get_neighbors(root_id)
    assert new_node_id in restored_neighbors, (
        "الرابط الديناميكي (self_evolved) لم يُستعَد في الرسم البياني "
        "بعد إعادة التشغيل رغم بقائه محفوظاً في exec_log"
    )
