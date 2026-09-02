"""اختبارات شبكة التنفيذ المعرفي القابلة للتحقق — توقيع + Hash + مدقق مستقل + Quorum."""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def mk_node(tmp, name, port=0):
    import ai.living_mesh as lm
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    return lm.LivingMeshNode(node_id=name, host="127.0.0.1", port=port)


def test_reject_without_signature_hash_quorum():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    with tempfile.TemporaryDirectory() as tmp:
        n = mk_node(tmp, "exec1")
        vcen = VerifiableCognitiveNet(n, quorum=2, require_independent=True)
        # مطالبة ناقصة
        bad = {
            "claim_id": "x",
            "result_hash": None,
            "executor_id": "exec1",
            "executor_signature": None,
            "attestations": [],
            "result": {"ok": True},
        }
        v = vcen.evaluate_acceptance(bad)
        assert v["accepted"] is False
        assert "no_executor_signature" in v["reasons_reject"]
        assert any("quorum_not_met" in r for r in v["reasons_reject"])
        print("✅ rejects missing signature/hash/quorum")


def test_reject_hash_mismatch():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet, canonical_hash
    with tempfile.TemporaryDirectory() as tmp:
        n = mk_node(tmp, "exec2")
        vcen = VerifiableCognitiveNet(n, quorum=1, require_independent=False)
        result = {"ok": True, "v": 1}
        claim = vcen.build_claim("map_reduce_map", result)
        claim["result"] = {"ok": True, "v": 999}  # تلاعب
        v = vcen.evaluate_acceptance(claim)
        assert v["accepted"] is False
        assert "result_hash_mismatch" in v["reasons_reject"]
        print("✅ rejects hash mismatch")


def test_reject_without_independent_verifier():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai import mesh_task_protocol as mt
    with tempfile.TemporaryDirectory() as tmp:
        n = mk_node(tmp, "solo")
        vcen = VerifiableCognitiveNet(n, quorum=1, require_independent=True)
        out = vcen.execute_and_claim(mt.KIND_MAP, {"lines": ["a b"], "op": "wordcount"})
        claim = out["claim"]
        # مصادقة من نفس المنفّذ فقط
        att = vcen.attest_as_verifier(claim)
        claim["attestations"] = [att]
        v = vcen.evaluate_acceptance(claim)
        assert v["accepted"] is False
        assert "no_independent_verifier" in v["reasons_reject"]
        print("✅ rejects without independent verifier")


def test_accept_with_signature_hash_independent_quorum():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai import mesh_task_protocol as mt
    with tempfile.TemporaryDirectory() as tmp:
        exec_n = mk_node(tmp, "executor")
        ver_n = mk_node(tmp, "verifier")
        exec_v = VerifiableCognitiveNet(exec_n, quorum=1, require_independent=True)
        ver_v = VerifiableCognitiveNet(ver_n, quorum=1, require_independent=True)
        path = exec_v.simulate_quorum_path(
            mt.KIND_MAP,
            {"lines": ["alpha beta", "beta gamma"], "op": "wordcount"},
            verifier_nodes=[ver_v],
        )
        assert path["ok"] is True, path
        assert path["verdict"]["accepted"] is True
        assert path["verdict"]["independent_attestations"] >= 1
        assert "executor_signature_valid" in path["verdict"]["reasons_ok"]
        assert "result_hash_valid" in path["verdict"]["reasons_ok"]
        print("✅ accepts with sig+hash+independent+quorum", path["claim_id"][:16])


def test_model_update_requires_same_policy():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    with tempfile.TemporaryDirectory() as tmp:
        exec_n = mk_node(tmp, "trainer")
        ver_n = mk_node(tmp, "auditor")
        ev = VerifiableCognitiveNet(exec_n, quorum=1, require_independent=True)
        vv = VerifiableCognitiveNet(ver_n, quorum=1, require_independent=True)
        result = {"partial_weights": [0.1, 0.2], "final_loss": 0.05, "ok": True}
        claim = ev.build_claim("submodel_train", result, claim_type="model_update")
        # تبادل مفاتيح
        ver_n.keys_dir.joinpath("trainer.pub").write_text(exec_n._pub_pem())
        exec_n.keys_dir.joinpath("auditor.pub").write_text(ver_n._pub_pem())
        att = vv.attest_as_verifier(claim)
        claim["attestations"] = [att]
        rec = ev.accept_model_update(claim)
        assert rec["accepted"] is True
        assert claim["claim_id"] in ev._accepted
        print("✅ model_update accepted under full policy")


def test_forged_attestation_rejected():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai import mesh_task_protocol as mt
    with tempfile.TemporaryDirectory() as tmp:
        exec_n = mk_node(tmp, "ex")
        ver_n = mk_node(tmp, "vr")
        ev = VerifiableCognitiveNet(exec_n, quorum=1, require_independent=True)
        out = ev.execute_and_claim(mt.KIND_INFERENCE, {"prompt": "hi"})
        claim = out["claim"]
        ver_n.keys_dir.joinpath("ex.pub").write_text(exec_n._pub_pem())
        vv = VerifiableCognitiveNet(ver_n, quorum=1)
        att = vv.attest_as_verifier(claim)
        att["verifier_signature"] = "FORGED"
        claim["attestations"] = [att]
        exec_n.keys_dir.joinpath("vr.pub").write_text(ver_n._pub_pem())
        v = ev.evaluate_acceptance(claim)
        assert v["accepted"] is False
        assert any("bad_attestation" in r for r in v["reasons_reject"])
        print("✅ forged attestation rejected")


if __name__ == "__main__":
    test_reject_without_signature_hash_quorum()
    test_reject_hash_mismatch()
    test_reject_without_independent_verifier()
    test_accept_with_signature_hash_independent_quorum()
    test_model_update_requires_same_policy()
    test_forged_attestation_rejected()
    print("🏆 VCEN policy tests passed")
