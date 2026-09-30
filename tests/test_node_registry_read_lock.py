"""core/registry.py: تكملة قفل التزامن (181fce4 قفل الكتابة فقط).

181fce4 أضاف threading.RLock لـNodeRegistry لكن قصره على مسارات الكتابة
(register/unregister/refresh_meta/_save) — دوال القراءة (get/get_by_name/
get_by_tag/get_by_state/list_all/list_metadata/orphaned_metadata/count/
exists/get_interrupted) بقيت تكرّر مباشرة على self._nodes/self._meta_cache
بلا أي قفل، رغم أنها تُستدعى فعلياً من dev_console بينما register() قد
يعمل في خيط آخر (EvolutionEngine/AutoRuntime في الخلفية). تحقّقتُ أن هذا
يرفع فعلياً RuntimeError: dictionary changed size during iteration قبل
هذا التعديل (2000 تسجيل متزامن مع قارئ في خيط منفصل → أكثر من 1400
استثناء متكرر). هذا الملف يثبت أن القراءات الآن آمنة أيضاً، بدون تكرار
تغطية tests/test_storage_corruption_recovery.py (تلف الملف/الكتابة
المتزامنة) الموجودة أصلاً.
"""
from __future__ import annotations

import threading

from core.node import BaseNode, NodeSchema
from core.registry import NodeRegistry
from storage.file_storage import FileStorage


class _N(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])

    def process(self, data):
        return {}


def test_concurrent_reads_during_writes_raise_nothing(tmp_path):
    registry = NodeRegistry(FileStorage(str(tmp_path)))
    stop = threading.Event()
    errors = []

    def reader():
        while not stop.is_set():
            try:
                registry.list_all()
                registry.list_metadata()
                registry.orphaned_metadata()
                registry.get_by_state("active")
                registry.get_interrupted()
            except Exception as e:  # noqa: BLE001
                errors.append(e)

    t = threading.Thread(target=reader)
    t.start()
    try:
        for i in range(500):
            registry.register(_N(f"n{i}"))
    finally:
        stop.set()
        t.join()

    assert errors == [], f"قراءات متزامنة رفعت استثناءات: {errors}"
    assert registry.count() == 500


def test_orphaned_metadata_still_excludes_live_nodes_after_lock():
    """لا تراجع في سلوك orphaned_metadata نفسه بعد إضافة القفل حوله."""
    registry = NodeRegistry(FileStorage(str(__import__("tempfile").mkdtemp())))
    node = _N("ghost", node_id="ghost-id")
    registry.register(node)
    del registry._nodes["ghost-id"]
    orphaned = registry.orphaned_metadata()
    assert any(m["node_id"] == "ghost-id" for m in orphaned)
    assert registry.get("ghost-id") is None
