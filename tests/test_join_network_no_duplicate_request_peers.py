"""وُجد بتشغيل شبكة محلية حقيقية من 3 عقد (seed + عاملين، عبر
ai/node_launcher.py): كل عقدة عاملة كانت تتصل بالبذرة مرتين فعلياً —
مرة عبر asyncio.create_task() غير مُنتظَرة داخل join_network()، ومرة
أخرى صراحةً في node_launcher.py بعدها مباشرة. الأخطر: join_network()
دالة sync عادية، وasyncio.create_task() يتطلب حلقة أحداث نشطة بالفعل —
فاستدعاؤها بـseed_nodes من أي سياق sync (كما يحدث فعلياً في
ai/gradient_mesh.py، وكما يفعل أي اختبار يستدعيها مباشرة) كان يرفع
RuntimeError فوراً."""
import asyncio
from unittest.mock import AsyncMock, patch

import pytest


def _isolate(tmp_path, node_id):
    import ai.living_mesh as lm
    from pathlib import Path
    d = Path(tmp_path) / node_id
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    return lm


def test_join_network_does_not_crash_from_sync_context_with_seeds(tmp_path):
    lm = _isolate(tmp_path, "n1")
    n = lm.LivingMeshNode(node_id="n1", host="127.0.0.1", port=0)
    try:
        n.join_network(seed_nodes=[{"id": "seed", "host": "127.0.0.1", "port": 9999}])
    finally:
        n.mark_offline()


def test_join_network_no_longer_calls_request_peers_itself(tmp_path):
    """المستدعي (node_launcher.py) هو المسؤول الوحيد الآن عن طلب الأقران
    — join_network() لا تستدعي request_peers داخلياً إطلاقاً بعد اليوم،
    فلا ازدواج فعلي لاتصال WebSocket بنفس البذرة."""
    lm = _isolate(tmp_path, "n2")
    n = lm.LivingMeshNode(node_id="n2", host="127.0.0.1", port=0)
    try:
        with patch.object(n, "request_peers", new=AsyncMock()) as rp:
            n.join_network(seed_nodes=[{"id": "seed", "host": "127.0.0.1", "port": 9999}])
            rp.assert_not_called()
    finally:
        n.mark_offline()


def test_join_network_safe_inside_running_loop_too(tmp_path):
    """نفس الضمان من داخل حلقة أحداث نشطة فعلاً (حالة node_launcher.py) —
    لا تغيير سلوك هناك أيضاً."""
    lm = _isolate(tmp_path, "n3")
    n = lm.LivingMeshNode(node_id="n3", host="127.0.0.1", port=0)

    async def _run():
        with patch.object(n, "request_peers", new=AsyncMock()) as rp:
            n.join_network(seed_nodes=[{"id": "seed", "host": "127.0.0.1", "port": 9999}])
            await asyncio.sleep(0)  # يسمح لأي task مجدولة خطأً أن تُشغَّل لو بقيت
            rp.assert_not_called()

    try:
        asyncio.run(_run())
    finally:
        n.mark_offline()


def test_gradient_protocol_seed_init_does_not_disable_p2p_outside_event_loop(tmp_path, monkeypatch):
    """قبل الإصلاح: نفس RuntimeError كان يُلتقَط بصمت في
    GradientExchangeProtocol.__init__ فيُعطَّل p2p_ready/mesh_node كلياً
    بلا سبب واضح في السجلّ، كلما وُجدت بذرة وبلا حلقة أحداث نشطة."""
    gm = pytest.importorskip("ai.gradient_mesh", reason="torch غير مثبَّت في هذه البيئة")
    monkeypatch.setattr(gm, "_parse_seed_from_alpha_url",
                         lambda url: {"host": "127.0.0.1", "port": 9999})
    lm = _isolate(tmp_path, "n4")

    proto = gm.GradientExchangeProtocol(
        node_id="n4", alpha_url="ws://127.0.0.1:9999", host="127.0.0.1", port=0,
    )
    try:
        assert proto.mesh_node is not None
        assert proto.p2p_ready is True
    finally:
        if proto.mesh_node is not None:
            proto.mesh_node.mark_offline()
