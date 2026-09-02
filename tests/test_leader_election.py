"""اختبارات انتخاب القائد، التسليم، واستكمال الجولة بعد الفشل."""
import sys
import tempfile
import time
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


def share_keys(*nodes):
    for a in nodes:
        for b in nodes:
            if a is b:
                continue
            a.keys_dir.joinpath(f"{b.node_id}.pub").write_text(b._pub_pem())


def test_election_majority():
    from ai.leader_election import LeaderElection
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = mk(tmp, "A"), mk(tmp, "B"), mk(tmp, "C")
        share_keys(a, b, c)
        la, lb, lc = LeaderElection(a, lease_seconds=30), LeaderElection(b, lease_seconds=30), LeaderElection(c, lease_seconds=30)
        elec = la.start_election()
        votes = list(elec["votes"])
        votes.append(lb.handle_vote_request(elec["vote_request"], candidate_pub=a._pub_pem().encode()))
        votes.append(lc.handle_vote_request(elec["vote_request"], candidate_pub=a._pub_pem().encode()))
        pubs = {n.node_id: n._pub_pem().encode() for n in (a, b, c)}
        # إصلاح candidate_id في التوقيع — handle_vote_response already includes it
        tally = la.tally_votes(elec["term"], votes, cluster_size=3, voter_pubs=pubs)
        assert tally["won"] is True, tally
        assert la.current_leader() == "A"
        assert la.state["role"] == "leader"
        print("✅ election majority term", tally["term"], "votes", tally["count"])


def test_handoff():
    from ai.leader_election import LeaderElection
    with tempfile.TemporaryDirectory() as tmp:
        a, b = mk(tmp, "L1"), mk(tmp, "L2")
        share_keys(a, b)
        la, lb = LeaderElection(a, lease_seconds=30), LeaderElection(b, lease_seconds=30)
        la.become_leader(1)
        assert la.is_leader_alive()
        ho = la.handoff("L2")
        assert ho["ok"]
        assert not la.is_leader_alive()
        acc = lb.accept_handoff(ho["handoff"], from_pub=a._pub_pem().encode())
        assert acc.get("leader_id") == "L2" or lb.current_leader() == "L2"
        assert lb.state["role"] == "leader"
        print("✅ handoff L1 → L2 term", lb.state["term"])


def test_round_continue_after_leader_failure():
    from ai.leader_election import LeaderElection
    with tempfile.TemporaryDirectory() as tmp:
        a, b, c = mk(tmp, "n0"), mk(tmp, "n1"), mk(tmp, "n2")
        share_keys(a, b, c)
        la = LeaderElection(a, lease_seconds=30)
        lb = LeaderElection(b, lease_seconds=30)
        lc = LeaderElection(c, lease_seconds=30)
        la.become_leader(1)
        opened = la.open_round("map_reduce", {
            "shards": ["s0", "s1", "s2"],
            "workers": ["n0", "n1", "n2"],
        })
        assert opened["ok"]
        rid = opened["round"]["round_id"]
        la.report_shard_result(rid, "s0", {"ok": True, "counts": {"x": 1}})
        # محاكاة فشل القائد A
        la.state["lease_until"] = 0
        la._save_state()

        # B يكتشف الفشل وينتخب بمساعدة أصوات C و A-state
        def peer_vote(req):
            # C و A (السابق) يمنحان الصوت لـ B
            # نحتاج أن يبدأ B الانتخاب أولاً داخل continue...
            return None

        # انتخاب B يدوياً بأسلوب continue
        # انسخ journal إلى B
        lb.rounds[rid] = dict(la.rounds[rid])
        import copy
        lb.rounds[rid] = copy.deepcopy(la.rounds[rid])

        elec = lb.start_election()
        votes = list(elec["votes"])
        votes.append(lc.handle_vote_request(elec["vote_request"], candidate_pub=b._pub_pem().encode()))
        votes.append(la.handle_vote_request(elec["vote_request"], candidate_pub=b._pub_pem().encode()))
        pubs = {n.node_id: n._pub_pem().encode() for n in (a, b, c)}
        # vote responses use candidate_id from request
        for v in votes:
            if "candidate_id" not in v:
                v["candidate_id"] = "n1"
        tally = lb.tally_votes(elec["term"], votes, cluster_size=3, voter_pubs=pubs)
        assert tally["won"], tally
        la.mark_leader_failed_on_round(rid)
        lb.rounds[rid] = copy.deepcopy(la.rounds[rid])
        resumed = lb.resume_round_as_leader(rid)
        assert resumed["ok"]
        assert "s1" in resumed["resume_shards"] and "s2" in resumed["resume_shards"]
        assert "s0" in resumed["completed"]
        # أكمل المتبقي
        lb.report_shard_result(rid, "s1", {"ok": True})
        lb.report_shard_result(rid, "s2", {"ok": True})
        assert lb.rounds[rid]["status"] == "completed"
        print("✅ round resumed after leader failure", rid, "pending was", resumed["resume_shards"])


def test_continue_after_leader_failure_helper():
    from ai.leader_election import LeaderElection
    import copy
    with tempfile.TemporaryDirectory() as tmp:
        a, b = mk(tmp, "x0"), mk(tmp, "x1")
        share_keys(a, b)
        la, lb = LeaderElection(a, lease_seconds=5), LeaderElection(b, lease_seconds=5)
        la.become_leader(1)
        rid = la.open_round("fl", {"shards": ["w0", "w1"]})["round"]["round_id"]
        la.report_shard_result(rid, "w0", {"ok": True})
        # فشل
        la.mark_leader_failed_on_round(rid)
        lb.rounds[rid] = copy.deepcopy(la.rounds[rid])

        def peer_vote(req):
            return la.handle_vote_request(req, candidate_pub=b._pub_pem().encode())

        out = lb.continue_after_leader_failure(rid, cluster_size=2, peer_vote_fn=peer_vote)
        assert out["ok"], out
        assert out["stage"] == "resumed"
        assert "w1" in out["resume"]["resume_shards"]
        print("✅ continue_after_leader_failure helper OK")


if __name__ == "__main__":
    test_election_majority()
    test_handoff()
    test_round_continue_after_leader_failure()
    test_continue_after_leader_failure_helper()
    print("🏆 leader election tests passed")
