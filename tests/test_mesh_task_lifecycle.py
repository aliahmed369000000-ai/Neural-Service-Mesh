# -*- coding: utf-8 -*-
"""
اختبارات دورة حياة المهام الموزّعة — سجل محلي، منع التكرار، إلغاء، حالات.
"""
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

from ai import mesh_task_protocol as mt
from ai.living_mesh import LivingMeshNode
import ai.living_mesh as lm


def _isolated_node(base: Path, node_id: str, port: int = 19100) -> LivingMeshNode:
    """عقدة معزولة بمجلد مفاتيح/حالة خاص."""
    root = base / node_id
    root.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = root
    lm.NETWORK_STATE = root / "network_state.json"
    (root / "keys").mkdir(exist_ok=True)
    (root / "content").mkdir(exist_ok=True)
    return LivingMeshNode(node_id=node_id, host="127.0.0.1", port=port)


def test_register_and_status():
    with tempfile.TemporaryDirectory() as td:
        node = _isolated_node(Path(td), "life-a")
        tid = f"task_{uuid.uuid4().hex[:8]}"
        node._register_task(tid, mt.KIND_INFERENCE, mt.TASK_STATUS_PENDING)
        st = node.get_task_status(tid)
        assert st is not None
        assert st["status"] == mt.TASK_STATUS_PENDING
        node._register_task(tid, mt.KIND_INFERENCE, mt.TASK_STATUS_RUNNING)
        assert node.get_task_status(tid)["status"] == mt.TASK_STATUS_RUNNING
        print("✅ register + status")


def test_cancel_before_complete():
    with tempfile.TemporaryDirectory() as td:
        node = _isolated_node(Path(td), "life-b")
        tid = f"task_{uuid.uuid4().hex[:8]}"
        node._register_task(tid, mt.KIND_MAP, mt.TASK_STATUS_RUNNING)
        res = node.cancel_local_task(tid)
        assert res["ok"] is True
        assert node.get_task_status(tid)["status"] == mt.TASK_STATUS_CANCELLED
        # إلغاء مجدد يفشل بأدب
        res2 = node.cancel_local_task(tid)
        assert res2["ok"] is False
        print("✅ cancel")


def test_duplicate_execution_rejected():
    with tempfile.TemporaryDirectory() as td:
        node = _isolated_node(Path(td), "life-c")
        tid = f"task_{uuid.uuid4().hex[:8]}"
        # محاكاة تنفيذ أول
        node._register_task(tid, mt.KIND_INFERENCE, mt.TASK_STATUS_COMPLETED)
        # استدعاء المعالج بنتيجة مكررة
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            node._handle_mesh_task(
                mt.KIND_INFERENCE_RESULT,
                {"task_id": tid, "ok": True, "output": "dup"},
                sender_id="peer-x",
            )
        )
        loop.close()
        assert node._metrics["tasks_duplicate_rejected"] >= 1
        print("✅ duplicate result rejected")


def test_list_tasks_and_metrics():
    with tempfile.TemporaryDirectory() as td:
        node = _isolated_node(Path(td), "life-d")
        for i, st in enumerate([mt.TASK_STATUS_PENDING, mt.TASK_STATUS_RUNNING, mt.TASK_STATUS_COMPLETED]):
            node._register_task(f"t{i}", mt.KIND_SIM, st)
        listed = node.list_tasks(limit=10)
        assert len(listed) == 3
        m = node.get_mesh_metrics()
        assert m["task_registry_size"] == 3
        assert "tasks_by_status" in m
        print("✅ list_tasks + metrics", m["tasks_by_status"])


def test_local_dispatch_updates_registry():
    with tempfile.TemporaryDirectory() as td:
        node = _isolated_node(Path(td), "life-e")
        # تأكد أن node_info موجود للقدرات
        node.node_info = {
            "id": node.node_id,
            "capabilities": ["text", "tf_engine", "CPU", "GPU_LOW"],
        }
        tid = f"task_{uuid.uuid4().hex[:8]}"
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            node._handle_mesh_task(
                mt.KIND_INFERENCE,
                {"task_id": tid, "prompt": "مرحبا", "max_tokens": 32},
                sender_id="client",
            )
        )
        loop.close()
        st = node.get_task_status(tid)
        assert st is not None
        assert st["status"] in (mt.TASK_STATUS_COMPLETED, mt.TASK_STATUS_FAILED)
        assert node._metrics["tasks_executed"] >= 1
        print("✅ local dispatch updates registry →", st["status"])



def test_duplicate_sends_explicit_error_reply():
    """# رفض التكرار يجب أن يعيد نتيجة ok=false وليس صمتاً."""
    with tempfile.TemporaryDirectory() as td:
        node = _isolated_node(Path(td), "life-dup")
        node.node_info = {"id": node.node_id, "capabilities": ["text", "tf_engine", "CPU"]}
        tid = f"task_{uuid.uuid4().hex[:8]}"
        # أول تنفيذ
        loop = asyncio.new_event_loop()
        loop.run_until_complete(
            node._handle_mesh_task(mt.KIND_INFERENCE, {"task_id": tid, "prompt": "a", "max_tokens": 8}, sender_id="c")
        )
        # محاكاة websocket يجمع الرسائل
        sent = []
        class FakeWS:
            async def send_str(self, m):
                sent.append(m)
            async def send(self, m):
                sent.append(m)
        loop.run_until_complete(
            node._handle_mesh_task(
                mt.KIND_INFERENCE,
                {"task_id": tid, "prompt": "a", "max_tokens": 8},
                sender_id="c",
                websocket=FakeWS(),
            )
        )
        loop.close()
        assert sent, "expected explicit reply on duplicate"
        import json
        payload = json.loads(sent[-1]).get("payload") or {}
        data = payload.get("data") or {}
        assert data.get("ok") is False
        assert data.get("error") == "duplicate_rejected"
        assert data.get("task_id") == tid
        print("✅ explicit duplicate_rejected reply")

if __name__ == "__main__":
    test_register_and_status()
    test_cancel_before_complete()
    test_duplicate_execution_rejected()
    test_list_tasks_and_metrics()
    test_local_dispatch_updates_registry()
    test_duplicate_sends_explicit_error_reply()
    print("\n🎉 All task lifecycle tests passed")
