# -*- coding: utf-8 -*-
"""اختبارات إصلاحات #14 منع تكرار الرسالة، #15 حظر الكود غير الموثوق، #16 إظهار أخطاء العمال."""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ai.living_mesh import LivingMeshNode, PROTOCOL_VERSION
import ai.living_mesh as lm
from ai.toolbox import nsm_toolbox, generate_custom_tool


def _node(base: Path, nid: str) -> LivingMeshNode:
    root = base / nid
    root.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = root
    lm.NETWORK_STATE = root / "network_state.json"
    (root / "keys").mkdir(exist_ok=True)
    return LivingMeshNode(node_id=nid, host="127.0.0.1", port=19200)


def test_14_duplicate_message_rejected():
    with tempfile.TemporaryDirectory() as td:
        node = _node(Path(td), "sec14")
        payload = {
            "protocol_version": PROTOCOL_VERSION,
            "id": "msg_fixed_1",
            "request_id": "req_fixed_1",
            "nonce": "nonce_fixed_1",
            "kind": "sovereign_gossip",
            "data": {"x": 1},
            "from": "peer",
            "ts_unix": int(time.time()),
        }
        assert node._validate_protocol_fields(payload) is None
        assert node._validate_protocol_fields(payload) == "replay"
        # بدون معرّف — يُرفض للأنواع غير الاكتشاف
        bare = {"kind": "sovereign_gossip", "data": {}, "from": "p", "ts_unix": int(time.time())}
        assert node._validate_protocol_fields(bare) == "missing_message_id"
        print("✅ #14 anti-duplicate + missing id")


def test_15_block_untrusted_code():
    # generate_custom_tool must raise
    try:
        generate_custom_tool("evil", "def evil():\n  return 1\n", "bad")
        assert False, "should have raised"
    except PermissionError as e:
        assert "not allowed" in str(e).lower() or "forbidden" in str(e).lower() or "disabled" in str(e).lower()
    # execute_tool blocks tool_generator and code kwargs
    try:
        nsm_toolbox.execute_tool("tool_generator", code="print(1)")
        assert False, "should block"
    except PermissionError:
        pass
    try:
        nsm_toolbox.execute_tool("code_analyzer", code="eval('1')")
        assert False, "should block code kwarg"
    except PermissionError:
        pass
    print("✅ #15 untrusted code blocked")


def test_16_tool_errors_surfaced():
    with tempfile.TemporaryDirectory() as td:
        node = _node(Path(td), "sec16")
        loop = asyncio.new_event_loop()
        # أداة غير مسموحة
        res = loop.run_until_complete(
            node._handle_tool_request({"tool_name": "tool_generator", "args": {"code": "x"}, "task_id": "t1"})
        )
        assert res["ok"] is False
        assert res["error"]
        assert "not allowed" in res["error"].lower() or "Permission" in res["error"]
        # أداة غير موجودة
        res2 = loop.run_until_complete(
            node._handle_tool_request({"tool_name": "no_such_tool_xyz", "args": {}, "task_id": "t2"})
        )
        assert res2["ok"] is False
        assert res2["error"]
        loop.close()
        print("✅ #16 tool errors returned explicitly:", res["error"][:80])


def test_16_mesh_task_unknown_returns_error_dict():
    with tempfile.TemporaryDirectory() as td:
        node = _node(Path(td), "sec16b")
        node.node_info = {"id": node.node_id, "capabilities": ["text", "CPU", "tf_engine"]}
        loop = asyncio.new_event_loop()
        # نوع غير معروف — يجب أن يسجّل فشل وليس صمتاً
        loop.run_until_complete(
            node._handle_mesh_task("totally_unknown_kind", {"task_id": "u1"}, sender_id="x")
        )
        st = node.get_task_status("u1")
        assert st is not None
        assert st["status"] in ("failed", "completed")
        loop.close()
        print("✅ #16 unknown mesh task not silent")


if __name__ == "__main__":
    test_14_duplicate_message_rejected()
    test_15_block_untrusted_code()
    test_16_tool_errors_surfaced()
    test_16_mesh_task_unknown_returns_error_dict()
    print("\n🎉 #14 #15 #16 tests passed")
