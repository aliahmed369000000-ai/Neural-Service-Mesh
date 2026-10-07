# -*- coding: utf-8 -*-
"""
انحدار: نتيجة RPC الحقيقية كانت تُسقَط دائماً بـ"bad signature".

LivingMeshNode.request_from_peer كانت تتحقق من توقيع الرد صحيحاً قبل فك التشفير
الطرفي، ثم تستبدل payload["data"] بالنص المفكوك، ثم تعيد مراجعة قديمة (سابقة
للتشفير الطرفي) على الحمولة المعدَّلة — فيختلف json.dumps عن المُوقَّع وتفشل
المراجعة دائماً. النتيجة العملية: العامل ينفّذ ويردّ ok=True والعميل يرمي الرد
ويُبلَّغ "timeout". اكتُشفت بتشغيل شبكة محلية من 3 عقد فعلياً وإرسال مهمة لها.

هنا عقدتان حقيقيتان (عامل + عميل) بمجلدَي بيانات معزولين وخادم WebSocket حقيقي.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from aiohttp import web, WSMsgType

from ai import mesh_task_protocol as mt
import ai.living_mesh as lm
from ai.living_mesh import LivingMeshNode

HOST = "127.0.0.1"
PORT = 8931


async def _scenario(tmp: str) -> tuple[dict, dict]:
    # network_state عامّ على مستوى الوحدة — اعزله في مجلد مؤقت كباقي اختبارات الشبكة
    old_dir, old_state = lm.LIVING_MESH_DIR, lm.NETWORK_STATE
    lm.LIVING_MESH_DIR = Path(tmp)
    lm.NETWORK_STATE = Path(tmp) / "network_state.json"
    worker = LivingMeshNode(node_id="rpc_worker", host=HOST, port=PORT, data_dir=f"{tmp}/worker")
    client = LivingMeshNode(node_id="rpc_client", host=HOST, port=PORT + 1, data_dir=f"{tmp}/client")
    worker.join_network()
    caps = set(worker.node_info.get("capabilities") or [])
    caps.update(["text", "tf_engine", "CPU", "GPU_LOW"])
    worker.node_info["capabilities"] = sorted(caps)

    async def ws_handler(request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        async for msg in ws:
            if msg.type == WSMsgType.TEXT:
                await worker._handle_aiohttp_ws_msg(ws, json.loads(msg.data))
        return ws

    app = web.Application()
    app.router.add_get("/ws", ws_handler)
    runner = web.AppRunner(app)
    await runner.setup()
    await web.TCPSite(runner, HOST, PORT).start()
    try:
        # مصافحة مفاتيح مباشرة مع العامل (شرط التحقق من التوقيع والتشفير الطرفي)
        assert await client.request_peers(HOST, PORT, retries=2)
        tid = f"rpc_{uuid.uuid4().hex[:6]}"
        payload = {"task_id": tid, "prompt": "مرحبا", "max_tokens": 8}
        first = await client.request_from_peer(HOST, PORT, mt.KIND_INFERENCE, dict(payload), timeout=10.0)
        dup = await client.request_from_peer(HOST, PORT, mt.KIND_INFERENCE, dict(payload), timeout=10.0)
        return first, dup
    finally:
        await runner.cleanup()
        lm.LIVING_MESH_DIR, lm.NETWORK_STATE = old_dir, old_state


def test_rpc_result_not_dropped_as_bad_signature():
    with tempfile.TemporaryDirectory() as tmp:
        first, dup = asyncio.run(_scenario(tmp))
    assert first["ok"] is True, f"النتيجة سُقطت: {first}"
    assert first["result"], "لا نتيجة فعلية رغم تنفيذ العامل"
    # رفض التكرار يعود صريحاً ومشفّراً، لا timeout صامتاً
    assert (dup.get("result") or {}).get("error") == "duplicate_rejected", dup
