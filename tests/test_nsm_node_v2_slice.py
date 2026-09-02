"""
NSM Node 2.0 Vertical Slice — اختبارات وحدة للشريحة الرأسية
تغطي: هوية دائمة، سمعة، إيصال، FL quorum محلي، صحة الشبكة.
(بدون شبكة حقيقية — مناسب لـ CI)
"""
import asyncio
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
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    return lm


def test_permanent_identity():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "idnode")
        n1 = lm.LivingMeshNode(node_id="idnode", host="127.0.0.1", port=0)
        pem1 = n1._pub_pem()
        priv_path = n1.keys_dir / "idnode.pem"
        assert priv_path.exists()
        n2 = lm.LivingMeshNode(node_id="idnode", host="127.0.0.1", port=0)
        assert n2._pub_pem() == pem1
        print("✅ permanent identity")


def test_receipt_and_reputation():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "rnode")
        n = lm.LivingMeshNode(node_id="rnode", host="127.0.0.1", port=0)
        n.join_network()
        result = {"ok": True, "value": 42}
        receipt = n.issue_execution_receipt("t1", "demo", result)
        assert receipt.get("signature")
        assert n.verify_signature(n._pub_pem().encode(), 
            __import__("json").dumps({k: receipt[k] for k in receipt if k != "signature"}, sort_keys=True),
            receipt["signature"])
        rep = n.get_reputation("rnode")
        assert rep["score"] >= 1
        print("✅ receipt + reputation", rep["score"])


def test_federated_quorum_local():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "flnode")
        n = lm.LivingMeshNode(node_id="flnode", host="127.0.0.1", port=0)
        n.join_network()
        out = asyncio.run(
            n.federated_round(worker_peers=[], steps=2, quorum=1)
        )
        assert out["ok"] is True
        assert out["merged"]["layers_count"] >= 1
        print("✅ federated local quorum", out["round_id"])


def test_health_snapshot():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "hnode")
        n = lm.LivingMeshNode(node_id="hnode", host="127.0.0.1", port=9100)
        n.join_network()
        snap = n.network_health_snapshot()
        assert snap["node_id"] == "hnode"
        assert "identity_pub_fingerprint" in snap
        print("✅ health snapshot", snap["identity_pub_fingerprint"])


def test_unified_task_envelope():
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "tnode")
        n = lm.LivingMeshNode(node_id="tnode", host="127.0.0.1", port=0)
        env = n.build_unified_task("inference_request", {"prompt": "hi"}, ttl=3)
        assert env["ttl"] == 3 and env["signature"]
        print("✅ unified task")


def test_five_nodes_identity_distinct():
    """خمس هويات دائمة متمايزة — أساس اختبار الخمس عقد."""
    with tempfile.TemporaryDirectory() as tmp:
        lm = _isolate(tmp, "shared")
        pubs = []
        for i in range(5):
            nid = f"node_{i}"
            # isolate each node storage
            d = Path(tmp) / nid
            d.mkdir(parents=True, exist_ok=True)
            lm.LIVING_MESH_DIR = d
            lm.NETWORK_STATE = d / "network_state.json"
            lm.CONTENT_DIR = d / "content"
            lm.CONTENT_DIR.mkdir(exist_ok=True)
            n = lm.LivingMeshNode(node_id=nid, host="127.0.0.1", port=9200 + i)
            pubs.append(n._pub_pem())
        assert len(set(pubs)) == 5
        print("✅ five distinct permanent identities")


if __name__ == "__main__":
    test_permanent_identity()
    test_receipt_and_reputation()
    test_federated_quorum_local()
    test_health_snapshot()
    test_unified_task_envelope()
    test_five_nodes_identity_distinct()
    print("🏆 NSM Node 2.0 vertical slice unit tests passed")
