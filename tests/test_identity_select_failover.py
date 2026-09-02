"""اختبارات #5 الهوية الدائمة و#6 اختيار العامل و#7 failover (بدون شبكة)."""
import asyncio
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def isolate(tmp, name):
    import ai.living_mesh as lm
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    # keys under LIVING_MESH_DIR/keys via node
    return lm


def test_identity_survives_restart():
    with tempfile.TemporaryDirectory() as tmp:
        lm = isolate(tmp, "id")
        n1 = lm.LivingMeshNode(node_id=None, host="127.0.0.1", port=1)
        nid = n1.node_id
        fp1 = n1.identity_info()["public_key_fingerprint"]
        pem1 = n1._pub_pem()
        # "restart"
        n2 = lm.LivingMeshNode(node_id=None, host="127.0.0.1", port=1)
        assert n2.node_id == nid, (n2.node_id, nid)
        assert n2._pub_pem() == pem1
        assert n2.identity_info()["public_key_fingerprint"] == fp1
        print("✅ #5 identity survives restart", nid[:12], fp1[:12])


def test_rank_workers_by_latency_reputation():
    with tempfile.TemporaryDirectory() as tmp:
        lm = isolate(tmp, "rk")
        n = lm.LivingMeshNode(node_id="ranker", host="127.0.0.1", port=0)
        n.join_network()
        from ai.node_health_layer import NodeHealthLayer
        h = NodeHealthLayer(n)
        # seed route cache
        h._route_cache = {
            "slow": {"ok": True, "rtt_ms": 80.0, "path": "direct"},
            "fast": {"ok": True, "rtt_ms": 5.0, "path": "direct"},
            "down": {"ok": False, "rtt_ms": None, "path": "down"},
        }
        state = n._load_state()
        state["nodes"] = {
            "slow": {"id": "slow", "host": "1.1.1.1", "port": 1, "status": "online", "capabilities": ["GPU_HIGH"]},
            "fast": {"id": "fast", "host": "1.1.1.2", "port": 2, "status": "online", "capabilities": ["GPU_HIGH"]},
            "down": {"id": "down", "host": "1.1.1.3", "port": 3, "status": "online", "capabilities": ["GPU_HIGH"]},
            "ranker": n.node_info if hasattr(n, "node_info") else {"id": "ranker", "status": "online"},
        }
        n._save_state(state)
        n.update_reputation("fast", delta=10, reason="test")
        ranked = h.rank_workers(require_capabilities=["GPU_HIGH"])
        assert ranked[0]["peer_id"] == "fast", ranked
        print("✅ #6 rank_workers prefers fast+reputable", [r["peer_id"] for r in ranked])


def test_failover_local_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        lm = isolate(tmp, "fo")
        n = lm.LivingMeshNode(node_id="fo_node", host="127.0.0.1", port=0)
        n.join_network()
        from ai.node_health_layer import NodeHealthLayer
        from ai import mesh_task_protocol as mt
        h = NodeHealthLayer(n)
        # no workers reachable -> local fallback
        out = asyncio.run(h.submit_with_failover(
            mt.KIND_MAP,
            {"lines": ["failover a", "failover b"], "op": "wordcount"},
            max_attempts=2,
            prefer_local_fallback=True,
        ))
        assert out["ok"] is True
        assert out["mode"] == "failover_local"
        assert out.get("receipt")
        ver = h.verify_receipt(out["receipt"], out["result"])
        assert ver["ok"] is True
        print("✅ #7 failover local fallback + verifiable receipt")


if __name__ == "__main__":
    test_identity_survives_restart()
    test_rank_workers_by_latency_reputation()
    test_failover_local_fallback()
    print("🏆 #5 #6 #7 tests passed")
