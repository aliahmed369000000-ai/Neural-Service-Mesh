"""اختبارات الذاكرة والنماذج والقرارات الجماعية القابلة للتدقيق."""
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def mk(tmp, name):
    import ai.living_mesh as lm
    d = Path(tmp) / name
    d.mkdir(parents=True, exist_ok=True)
    lm.LIVING_MESH_DIR = d
    lm.NETWORK_STATE = d / "network_state.json"
    lm.CONTENT_DIR = d / "content"
    lm.CONTENT_DIR.mkdir(exist_ok=True)
    return lm.LivingMeshNode(node_id=name, host="127.0.0.1", port=0)


def test_collective_memory_from_vcen():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    from ai import mesh_task_protocol as mt
    with tempfile.TemporaryDirectory() as tmp:
        ex, vr = mk(tmp, "ex"), mk(tmp, "vr")
        ev = VerifiableCognitiveNet(ex, quorum=1, require_independent=True)
        vv = VerifiableCognitiveNet(vr, quorum=1, require_independent=True)
        ccl = CollectiveCognitiveLedger(ex, vcen=ev, quorum=1)
        path = ev.simulate_quorum_path(mt.KIND_MAP, {"lines": ["a b c"], "op": "wordcount"}, [vv])
        assert path["ok"]
        claim = ev._claims[path["claim_id"]]
        mem = ccl.ingest_accepted_claim(claim)
        assert mem["ok"]
        snap = ccl.memory_snapshot()
        assert snap["count"] == 1
        assert ccl.verify_memory_integrity()["ok"]
        print("✅ collective memory ingest", snap["chain_tip"][:12])


def test_reject_memory_without_vcen():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "solo")
        ccl = CollectiveCognitiveLedger(n, quorum=2, require_independent=True)
        bad = {
            "claim_id": "x", "claim_type": "task_result", "kind": "x",
            "result_hash": "abc", "executor_id": "solo",
            "executor_signature": None, "result": {}, "attestations": [],
        }
        mem = ccl.ingest_accepted_claim(bad)
        assert mem["ok"] is False
        assert ccl.memory_snapshot()["count"] == 0
        print("✅ memory rejects non-VCEN claims")


def test_collective_model_update():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    with tempfile.TemporaryDirectory() as tmp:
        tr, au = mk(tmp, "trainer"), mk(tmp, "auditor")
        tv = VerifiableCognitiveNet(tr, quorum=1, require_independent=True)
        av = VerifiableCognitiveNet(au, quorum=1, require_independent=True)
        ccl = CollectiveCognitiveLedger(tr, vcen=tv, quorum=1)
        claim = tv.build_claim(
            "submodel_train",
            {"ok": True, "partial_weights": [0.2, 0.4, 0.6], "final_loss": 0.1},
            claim_type="model_update",
        )
        au.keys_dir.joinpath("trainer.pub").write_text(tr._pub_pem())
        tr.keys_dir.joinpath("auditor.pub").write_text(au._pub_pem())
        claim["attestations"] = [av.attest_as_verifier(claim)]
        out = ccl.apply_model_update_claim(claim)
        assert out["ok"] is True
        assert out["model"]["version"] == 1
        assert out["model"]["model_hash"]
        assert ccl.memory_snapshot()["count"] == 1
        print("✅ collective model v", out["model"]["version"], out["model"]["model_hash"][:12])


def test_collective_decision_quorum():
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    with tempfile.TemporaryDirectory() as tmp:
        a, b = mk(tmp, "nodeA"), mk(tmp, "nodeB")
        ccl_a = CollectiveCognitiveLedger(a, quorum=2)
        ccl_b = CollectiveCognitiveLedger(b, quorum=2)
        # تبادل مفاتيح
        b.keys_dir.joinpath("nodeA.pub").write_text(a._pub_pem())
        a.keys_dir.joinpath("nodeB.pub").write_text(b._pub_pem())

        dec = ccl_a.propose_decision("اعتماد جولة FL", {"round": 1, "accept": True}, threshold=2)
        ccl_a.vote_decision(dec["decision_id"], approve=True)
        # نسخ القرار للعقدة B والتصويت
        ccl_b.decisions[dec["decision_id"]] = dec
        ccl_b.vote_decision(dec["decision_id"], approve=True)
        # إعادة الأصوات لـ A
        dec["votes"] = ccl_b.decisions[dec["decision_id"]]["votes"]
        # تأكد من وجود صوت A أيضاً
        ccl_a.decisions[dec["decision_id"]] = dec
        if not any(v.get("voter_id") == "nodeA" for v in dec["votes"]):
            ccl_a.vote_decision(dec["decision_id"], approve=True)

        fin = ccl_a.finalize_decision(dec["decision_id"])
        assert fin["ok"] is True, fin
        assert fin["decision"]["status"] == "accepted"
        assert ccl_a.memory_snapshot()["count"] >= 1
        audit = ccl_a.full_audit_export()
        assert audit["memory_integrity"]["ok"]
        assert audit["decisions_accepted"] >= 1
        print("✅ collective decision finalized", fin["decision"]["decision_hash"][:12])


def test_full_audit_export():
    from ai.verifiable_cognitive_net import VerifiableCognitiveNet
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    from ai import mesh_task_protocol as mt
    with tempfile.TemporaryDirectory() as tmp:
        ex, vr = mk(tmp, "e2"), mk(tmp, "v2")
        ev = VerifiableCognitiveNet(ex, quorum=1, require_independent=True)
        vv = VerifiableCognitiveNet(vr, quorum=1, require_independent=True)
        ccl = CollectiveCognitiveLedger(ex, vcen=ev, quorum=1)
        path = ev.simulate_quorum_path(mt.KIND_INFERENCE, {"prompt": "تدقيق"}, [vv])
        claim = ev._claims[path["claim_id"]]
        ccl.ingest_accepted_claim(claim)
        exp = ccl.full_audit_export()
        assert exp["memory"]["count"] == 1
        assert exp["audit_events"] >= 1
        print("✅ full audit export events", exp["audit_events"])


if __name__ == "__main__":
    test_collective_memory_from_vcen()
    test_reject_memory_without_vcen()
    test_collective_model_update()
    test_collective_decision_quorum()
    test_full_audit_export()
    print("🏆 Collective Cognitive Ledger tests passed")
