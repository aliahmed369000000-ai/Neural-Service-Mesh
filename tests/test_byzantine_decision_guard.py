"""اختبارات حماية القرارات من الهجمات وانقسام الشبكة."""
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


def test_reject_outsider_vote():
    from ai.byzantine_decision_guard import ByzantineDecisionGuard
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "core")
        g = ByzantineDecisionGuard(n)
        g.set_roster(["core", "a", "b"])
        r = g.validate_vote("attacker", term=1, candidate_id="core", signature_ok=True)
        assert r["ok"] is False and r["reason"] == "voter_not_in_roster"
        print("✅ reject outsider vote")


def test_reject_stale_term():
    from ai.byzantine_decision_guard import ByzantineDecisionGuard
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "core")
        g = ByzantineDecisionGuard(n)
        g.set_roster(["core", "a", "b"])
        g.state["last_seen_term"] = 5
        g._save()
        r = g.validate_vote("a", term=3, candidate_id="core", signature_ok=True)
        assert r["ok"] is False and r["reason"] == "stale_term"
        print("✅ reject stale term")


def test_split_brain_detection():
    from ai.byzantine_decision_guard import ByzantineDecisionGuard
    with tempfile.TemporaryDirectory() as tmp:
        n = mk(tmp, "core")
        g = ByzantineDecisionGuard(n)
        g.set_roster(["core", "a", "b", "c"])
        # قبول قائد term 1
        ok = g.validate_leader_claim(term=1, leader_id="a", vote_count=3)
        assert ok["ok"]
        # قائد آخر لنفس term
        bad = g.validate_leader_claim(term=1, leader_id="b", vote_count=3)
        assert bad["ok"] is False and bad["reason"] == "split_brain_conflict"
        sb = g.detect_split_brain([
            {"term": 1, "leader_id": "a"},
            {"term": 1, "leader_id": "b"},
        ])
        assert sb["split_brain"] is True
        print("✅ split-brain conflict detected")


def test_decision_rejects_minority_partition():
    from ai.byzantine_decision_guard import ByzantineDecisionGuard
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c, d = mk(tmp, "A"), mk(tmp, "B"), mk(tmp, "C"), mk(tmp, "D")
        # اتحاد 4 عقد — majority = 3
        g = ByzantineDecisionGuard(a)
        g.set_roster(["A", "B", "C", "D"])
        ccl = CollectiveCognitiveLedger(a, quorum=2)
        ccl.guard = g
        # تبادل مفاتيح
        for x in (a, b):
            for y in (a, b):
                if x is y: continue
                x.keys_dir.joinpath(f"{y.node_id}.pub").write_text(y._pub_pem())

        dec = ccl.propose_decision("قرار قسم صغير", {"x": 1}, threshold=2)
        ccl.vote_decision(dec["decision_id"], True)
        # محاكاة صوت B فقط (قسم من عقدتين) — أقل من majority الاتحاد (3)
        ccl_b = CollectiveCognitiveLedger(b, quorum=2)
        ccl_b.decisions[dec["decision_id"]] = dec
        ccl_b.vote_decision(dec["decision_id"], True)
        dec["votes"] = ccl_b.decisions[dec["decision_id"]]["votes"]
        if not any(v["voter_id"] == "A" for v in dec["votes"]):
            ccl.vote_decision(dec["decision_id"], True)
            dec = ccl.decisions[dec["decision_id"]]

        fin = ccl.finalize_decision(dec["decision_id"])
        # أصوات A+B = 2 < majority 3 → رفض الحارس
        assert fin["ok"] is False, fin
        assert fin.get("error") in (
            "decision_quorum_not_met", "quorum_not_met", "threshold_below_federation_majority"
        ) or "guard" in fin
        print("✅ minority partition decision rejected", fin.get("error") or fin.get("guard", {}).get("reason"))


def test_safe_decision_with_full_majority():
    from ai.byzantine_decision_guard import ByzantineDecisionGuard
    from ai.collective_cognitive_ledger import CollectiveCognitiveLedger
    with tempfile.TemporaryDirectory() as tmp:
        nodes = [mk(tmp, n) for n in ("A", "B", "C")]
        for x in nodes:
            for y in nodes:
                if x is y: continue
                x.keys_dir.joinpath(f"{y.node_id}.pub").write_text(y._pub_pem())
        g = ByzantineDecisionGuard(nodes[0])
        g.set_roster(["A", "B", "C"])  # majority=2
        ccl = CollectiveCognitiveLedger(nodes[0], quorum=2)
        ccl.guard = g
        dec = ccl.propose_decision("قرار آمن", {"go": True}, threshold=2)
        for n, node in zip(("A", "B", "C"), nodes):
            if n == "A":
                ccl.vote_decision(dec["decision_id"], True)
            else:
                c2 = CollectiveCognitiveLedger(node, quorum=2)
                c2.decisions[dec["decision_id"]] = ccl.decisions[dec["decision_id"]]
                c2.vote_decision(dec["decision_id"], True)
                # دمج الأصوات
                existing = {v["voter_id"]: v for v in ccl.decisions[dec["decision_id"]]["votes"]}
                for v in c2.decisions[dec["decision_id"]]["votes"]:
                    existing[v["voter_id"]] = v
                ccl.decisions[dec["decision_id"]]["votes"] = list(existing.values())

        fin = ccl.finalize_decision(dec["decision_id"])
        assert fin["ok"] is True, fin
        print("✅ full majority decision accepted")


def test_leader_election_guard_blocks_low_votes():
    from ai.byzantine_decision_guard import ByzantineDecisionGuard
    from ai.leader_election import LeaderElection
    with tempfile.TemporaryDirectory() as tmp:
        a = mk(tmp, "A")
        le = LeaderElection(a, lease_seconds=30)
        g = ByzantineDecisionGuard(a)
        g.set_roster(["A", "B", "C", "D", "E"])  # majority=3
        le.guard = g
        elec = le.start_election()
        # صوت واحد فقط (A) — أقل من 3
        tally = le.tally_votes(elec["term"], elec["votes"], cluster_size=5)
        assert tally["won"] is False
        print("✅ leader election blocked below federation majority")


if __name__ == "__main__":
    test_reject_outsider_vote()
    test_reject_stale_term()
    test_split_brain_detection()
    test_decision_rejects_minority_partition()
    test_safe_decision_with_full_majority()
    test_leader_election_guard_blocks_low_votes()
    print("🏆 Byzantine decision guard tests passed")
