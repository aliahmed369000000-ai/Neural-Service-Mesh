"""
اختبارات وحدة خفيفة لـ: قياس صحة الأقران + Relay API + Multi-sig
لا تحتاج شبكة حقيقية — تتحقق من الواجهة والتوقيعات المحلية.
"""
import asyncio
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def _isolate(tmp: str, node_id: str):
    import ai.living_mesh as lm
    d = Path(tmp) / node_id
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    return lm


def test_sign_verify_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "n1")
        node = lm.LivingMeshNode(node_id="n1", host="127.0.0.1", port=0)
        msg = "hello-multisig"
        sig = node.sign_message(msg)
        assert node.verify_signature(node._pub_pem().encode(), msg, sig)
        assert not node.verify_signature(node._pub_pem().encode(), msg, "AAAA")
        print("✅ sign/verify OK")


def test_multisig_local_threshold():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "n1")
        node = lm.LivingMeshNode(node_id="n1", host="127.0.0.1", port=0)
        node.join_network()
        # required=1 حتى تُنفَّذ محلياً بدون أقران
        async def run():
            aid = await node.propose_multisig(
                {"reward": 10, "to": "wallet_x", "reason": "test"},
                required_signatures=1,
            )
            state = node._load_state()
            entry = state["multisig"][aid]
            assert entry["executed"] is True
            assert entry["status"] == "approved"
            assert "n1" in entry["signatures"]
            print("✅ multisig local threshold OK", aid)
        asyncio.run(run())


def test_health_api_empty_peers():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "n1")
        node = lm.LivingMeshNode(node_id="n1", host="127.0.0.1", port=0)
        node.join_network()
        async def run():
            results = await node.measure_peers_health()
            assert isinstance(results, list)
            print("✅ measure_peers_health empty OK", results)
        asyncio.run(run())


def test_relay_api_no_peers():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "n1")
        node = lm.LivingMeshNode(node_id="n1", host="127.0.0.1", port=0)
        node.join_network()
        async def run():
            # سيفشل المباشر والـrelay لعدم وجود أقران — يجب أن يُرجع ok=False بوضوح
            r = await node.send_to_peer_with_relay("127.0.0.1", 59999, "sovereign_gossip", {"x": 1})
            assert r["ok"] is False
            assert r["mode"] == "failed"
            print("✅ relay failure explicit OK", r)
        asyncio.run(run())


if __name__ == "__main__":
    test_sign_verify_roundtrip()
    test_multisig_local_threshold()
    test_health_api_empty_peers()
    test_relay_api_no_peers()
    print("🏆 all light tests passed")
