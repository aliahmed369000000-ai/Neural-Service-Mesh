"""إقلاع MeshBundle مع سجلات عقد تالفة في nodes.json:

كان سجل واحد تالف (node_id رقم، node_type قائمة) يرفع استثناءً من
_restore_dynamic_nodes داخل __init__ فيفشل إقلاع الحزمة كلها؛ وسجلات أخرى
تُحيا بقيم فاسدة (state غير معروفة، execution_count نصي ينهار عند أول تنفيذ،
tags نص). الآن: السجل التالف يُتخطّى بتحذير، والقيم القابلة للإنقاذ تُعقَّم.
"""
import json
import tempfile

import pytest

from core.mesh_bundle import MeshBundle
from core.node import NodeState


def _boot(metas):
    tmp = tempfile.mkdtemp()
    with open(f"{tmp}/nodes.json", "w", encoding="utf-8") as f:
        json.dump({"nodes": metas}, f)
    return MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")


def _find(bundle, name):
    return next((n for n in bundle.registry.list_all() if n.name == name), None)


@pytest.mark.parametrize("meta", [
    {"node_id": 123, "name": "bad_int_id", "node_type": "PassThroughNode"},
    {"node_id": "abcdef123456", "name": "bad_type_list", "node_type": ["a"]},
    "not-a-dict",
    {"name": "no_id"},
])
def test_corrupt_record_does_not_break_boot(meta):
    healthy = {"node_id": "eeeeeeee5555", "name": "healthy_node", "node_type": "PassThroughNode",
               "state": "active", "execution_count": 7}
    b = _boot([meta, healthy])
    h = _find(b, "healthy_node")
    assert h is not None and h._execution_count == 7  # السجل السليم أُحيي رغم التالف قبله


def test_corrupt_field_values_are_sanitized():
    b = _boot([
        {"node_id": "bbbbbbbb1111", "name": "bad_state", "node_type": "PassThroughNode", "state": "banana"},
        {"node_id": "cccccccc2222", "name": "bad_count", "node_type": "PassThroughNode",
         "execution_count": "abc", "resume_attempts": -3, "pending_input": "oops"},
        {"node_id": "dddddddd3333", "name": "bad_tags", "node_type": "PassThroughNode", "tags": "dynamic"},
    ])
    assert _find(b, "bad_state").state == NodeState.CREATED
    c = _find(b, "bad_count")
    assert c._execution_count == 0 and c._resume_attempts == 0 and c._pending_input is None
    c._execution_count += 1  # لم يعد ينهار
    assert _find(b, "bad_tags").tags == ["dynamic", "passthrough", "restored"]


def test_valid_snapshot_roundtrip_unchanged():
    from core.node import BaseNode, NodeSchema

    class N(BaseNode):
        input_schema = NodeSchema(fields={}, required=[])
        output_schema = NodeSchema(fields={}, required=[])

        def process(self, data):
            return data

    a = N("a")
    a.pause(reason="quarantine")
    a._execution_count = 4
    b = N("a", node_id=a.node_id)
    b.restore_state(a.to_dict())
    assert (b.state, b.pause_reason, b._execution_count) == (NodeState.PAUSED, "quarantine", 4)
