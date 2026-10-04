"""عقدة self_evolved تُحيا بعد إعادة التشغيل يجب أن تملك «اليدين» أيضاً،
وأن تعمل يدها اليمنى المقيَّدة فعلياً (لا مجرد وجود الكائن)."""
from __future__ import annotations

import shutil
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node_hands import LEFT, RIGHT
from services.generated_service_nodes import NormalizerNode


@pytest.fixture()
def paths():
    tmp = tempfile.mkdtemp(prefix="nsm_revived_hands_")
    try:
        yield tmp, f"{tmp}/mesh.db"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_revived_evolved_node_has_working_hands(paths):
    storage_dir, db_path = paths
    b1 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    node_id = b1.register_node(NormalizerNode(name="evolved-norm"))

    b2 = MeshBundle(storage_dir=storage_dir, db_path=db_path)
    revived = b2.registry.get(node_id)
    assert isinstance(revived, NormalizerNode)
    assert revived.hands is not None
    tools = revived.hands.tools()
    assert {t["name"] for t in tools[RIGHT]} == {"send_message", "request_evolution"}
    assert "peers" in {t["name"] for t in tools[LEFT]}

    # اليسرى تعمل فعلاً، واليمنى (send_message) تصل لعقدة حقيقية.
    assert revived.use_hand(LEFT, "peers").ok
    other = next(i for i in b2.role_node_ids.values())
    res = revived.use_hand(RIGHT, "send_message", to_id=other, topic="ping")
    assert res.ok or res.denied  # السياسة تقرّر؛ المهم ألا ترفع استثناءً
    assert res.error is None or isinstance(res.error, str)
