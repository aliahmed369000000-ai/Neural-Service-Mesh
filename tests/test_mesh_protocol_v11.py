# -*- coding: utf-8 -*-
"""
اختبارات بروتوكول الرسائل v1.1 — مكافحة Replay، حدود الحجم، الطابع الزمني، الإصدار.
بدون شبكة حقيقية (وحدة فقط).
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from ai.living_mesh import (
    LivingMeshNode,
    PROTOCOL_VERSION,
    MAX_MESSAGE_BYTES,
    MAX_TIMESTAMP_SKEW_SEC,
)


def _make_node(tmp_keys: Path, node_id: str = "test-node-a") -> LivingMeshNode:
    """ينشئ عقدة باختبار معزول (مجلد مفاتيح خاص)."""
    # نعيد توجيه مجلد المفاتيح عبر monkeypatch بسيط على الثابت
    import ai.living_mesh as lm
    original = lm.LIVING_MESH_DIR
    lm.LIVING_MESH_DIR = tmp_keys
    lm.NETWORK_STATE = tmp_keys / "network_state.json"
    (tmp_keys / "keys").mkdir(parents=True, exist_ok=True)
    try:
        node = LivingMeshNode(node_id=node_id, host="127.0.0.1", port=18901)
        return node
    finally:
        # لا نعيد الأصل هنا لأن العقدة تحتفظ بالمراجع
        pass


def test_build_signed_payload_has_v11_fields(tmp_path):
    node = _make_node(tmp_path / "n1", "node-a")
    raw = node._build_signed_payload("ping_request", {"client_ts": time.time()})
    msg = json.loads(raw)
    payload = msg["payload"]
    assert "signature" in msg
    assert payload.get("protocol_version") == PROTOCOL_VERSION
    assert payload.get("request_id")
    assert payload.get("nonce")
    assert payload.get("ts_unix")
    assert payload.get("timestamp")
    assert payload.get("kind") == "ping_request"
    assert payload.get("from") == "node-a"
    print("✅ build_signed_payload v1.1 fields present")


def test_anti_replay_rejects_duplicate(tmp_path):
    node = _make_node(tmp_path / "n2", "node-b")
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "id": "msg_abc",
        "request_id": "req_unique_123",
        "nonce": "nonce_unique_456",
        "kind": "ping_request",
        "data": {},
        "from": "peer-x",
        "p2p_hops": 0,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "ts_unix": int(time.time()),
    }
    # أول مرة: مقبول
    assert node._validate_protocol_fields(payload) is None
    # ثاني مرة بنفس request_id/nonce: رفض
    assert node._validate_protocol_fields(payload) == "replay"
    print("✅ anti-replay rejects duplicate request_id/nonce")


def test_timestamp_skew_rejected(tmp_path):
    node = _make_node(tmp_path / "n3", "node-c")
    old_ts = int(time.time()) - (MAX_TIMESTAMP_SKEW_SEC + 60)
    payload = {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": f"req_{uuid.uuid4().hex}",
        "nonce": uuid.uuid4().hex,
        "kind": "ping_request",
        "data": {},
        "from": "peer-y",
        "ts_unix": old_ts,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(old_ts)),
    }
    assert node._validate_protocol_fields(payload) == "timestamp_skew"
    print("✅ old timestamp rejected")


def test_oversized_message_rejected(tmp_path):
    node = _make_node(tmp_path / "n4", "node-d")
    huge = "x" * (MAX_MESSAGE_BYTES + 100)
    # نستدعي المعالجة مباشرة
    async def _run():
        await node._process_secure_message(huge)
    asyncio.get_event_loop().run_until_complete(_run()) if False else None
    # نفحص العداد بعد استدعاء متزامن عبر حلقة جديدة
    loop = asyncio.new_event_loop()
    loop.run_until_complete(node._process_secure_message(huge))
    loop.close()
    assert node._metrics["messages_rejected_size"] >= 1
    print("✅ oversized message rejected")


def test_metrics_exposed(tmp_path):
    node = _make_node(tmp_path / "n5", "node-e")
    m = node.get_mesh_metrics()
    assert m["protocol_version"] == PROTOCOL_VERSION
    assert "metrics" in m
    assert "messages_received" in m["metrics"]
    print("✅ get_mesh_metrics works")


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        test_build_signed_payload_has_v11_fields(p)
        test_anti_replay_rejects_duplicate(p)
        test_timestamp_skew_rejected(p)
        test_oversized_message_rejected(p)
        test_metrics_exposed(p)
    print("\n🎉 All protocol v1.1 unit tests passed")
