"""تحصين LivingMeshNode: حالة ذرّية، غوسيب لا يُحيي الموتى، العقدة لا تقتل نفسها،
تعافٍ ذاتي عبر البذور، وربط الشبكة العصبية باختيار الأقران والتعلّم من نتائج المهام.
بلا شبكة ولا مفاتيح API."""
import asyncio
import threading
import time
from datetime import datetime, timedelta, timezone

import pytest

from ai.living_mesh import LivingMeshNode


def _iso(delta_sec=0):
    return (datetime.now(timezone.utc) + timedelta(seconds=delta_sec)).isoformat()


@pytest.fixture
def node(tmp_path, monkeypatch):
    monkeypatch.setenv("NSM_PEER_STALE_SEC", "45")
    return LivingMeshNode(node_id="A", host="127.0.0.1", port=9001, data_dir=tmp_path)


def test_state_never_torn_under_concurrent_read_write(node):
    node._save_state({"nodes": {f"n{i}": {"id": f"n{i}", "status": "online", "pad": "x" * 3000}
                                for i in range(40)}, "global_experience": []})
    stop, bad = threading.Event(), []

    def writer():
        while not stop.is_set():
            with node._state_txn():
                s = node._load_state()
                s["nodes"]["n0"]["status"] = "online"
                node._save_state(s)

    def reader():
        while not stop.is_set():
            if len(node._load_state()["nodes"]) != 40:
                bad.append(1)

    ts = [threading.Thread(target=writer) for _ in range(3)] + [threading.Thread(target=reader) for _ in range(2)]
    [t.start() for t in ts]
    time.sleep(1.5)
    stop.set()
    [t.join() for t in ts]
    assert bad == []


def test_txn_prevents_lost_updates(node):
    node._save_state({"nodes": {}, "global_experience": []})

    def bump(i):
        for _ in range(25):
            with node._state_txn():
                s = node._load_state()
                s["nodes"][f"k{i}"] = s["nodes"].get(f"k{i}", 0) + 1
                node._save_state(s)

    ts = [threading.Thread(target=bump, args=(i,)) for i in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert sorted(node._load_state()["nodes"].values()) == [25] * 6


def test_gossip_does_not_resurrect_dead_peer(node):
    node._save_state({"nodes": {"X": {"id": "X", "status": "offline", "last_seen": _iso(-7200),
                                      "host": "10.0.0.9", "port": 1, "public_key": "PEM-X"}}})
    st = node._load_state()
    # قرين ثالث يروي X بتاريخ قديم + محاولة تغيير host/public_key
    node._merge_gossiped_peer(st, {"id": "X", "status": "online", "last_seen": _iso(-3600),
                                   "host": "6.6.6.6", "port": 2, "public_key": "EVIL"})
    x = st["nodes"]["X"]
    assert x["status"] == "offline" and x["host"] == "10.0.0.9" and x["public_key"] == "PEM-X"


def test_gossip_fresher_evidence_revives_but_never_forges_now(node):
    node._save_state({"nodes": {"X": {"id": "X", "status": "offline", "last_seen": _iso(-600)}}})
    st = node._load_state()
    fresh = _iso(-5)
    assert node._merge_gossiped_peer(st, {"id": "X", "last_seen": fresh})
    assert st["nodes"]["X"]["status"] == "online" and st["nodes"]["X"]["last_seen"] == fresh


def test_gossip_future_timestamp_is_clamped(node):
    st = {"nodes": {}}
    node._merge_gossiped_peer(st, {"id": "F", "host": "h", "port": 5, "last_seen": _iso(10 ** 9)})
    assert (datetime.fromisoformat(st["nodes"]["F"]["last_seen"]) - datetime.now(timezone.utc)).total_seconds() < 2


def test_gossip_new_stale_peer_added_offline(node):
    st = {"nodes": {}}
    node._merge_gossiped_peer(st, {"id": "S", "host": "h", "port": 5, "last_seen": _iso(-999)})
    assert st["nodes"]["S"]["status"] == "offline"


@pytest.mark.parametrize("bad", ["../../evil", "a/b", "", "x" * 200, "..", "a\\b"])
def test_gossip_rejects_unsafe_peer_ids(node, bad):
    st = {"nodes": {}}
    assert node._merge_gossiped_peer(st, {"id": bad, "host": "h", "port": 5}) is False
    assert st["nodes"] == {}


@pytest.mark.parametrize("port", [0, 70000, "abc", -1])
def test_gossip_rejects_bad_ports(node, port):
    assert node._merge_gossiped_peer({"nodes": {}}, {"id": "P", "host": "h", "port": port}) is False


def test_node_does_not_mark_itself_dead(node):
    node._save_state({"nodes": {"A": {"id": "A", "status": "online", "last_seen": _iso(-500)},
                                "B": {"id": "B", "status": "online", "last_seen": _iso(-500)}}})
    dead = node.check_network_health(timeout_seconds=45)
    st = node._load_state()["nodes"]
    assert dead == ["B"] and st["A"]["status"] == "online" and st["B"]["status"] == "offline"


def test_resync_falls_back_to_seeds_when_no_active_peers(node, monkeypatch):
    node._seed_nodes = [{"host": "seed.local", "port": 7000}]
    node._save_state({"nodes": {}})
    calls = []

    async def fake_request_peers(host, port, retries=3, **kw):
        calls.append((host, port))
        return True

    monkeypatch.setattr(node, "request_peers", fake_request_peers)
    assert node.resync_peers() == 1 and calls == [("seed.local", 7000)]


def test_resync_probes_one_offline_peer_so_it_can_return(node, monkeypatch):
    node._save_state({"nodes": {
        "B": {"id": "B", "status": "online", "host": "b", "port": 1, "last_seen": _iso()},
        "D": {"id": "D", "status": "offline", "host": "d", "port": 2, "last_seen": _iso(-999)}}})
    calls = []

    async def fake(host, port, retries=3, **kw):
        calls.append(host)
        return True

    monkeypatch.setattr(node, "request_peers", fake)
    node.resync_peers(sample_size=2)
    assert sorted(calls) == ["b", "d"]


def test_dispatch_trains_neural_advisor_and_ranking_prefers_reliable_peer(node, monkeypatch):
    monkeypatch.setenv("NSM_PEER_NEURAL_RANK", "1")
    node._save_state({"nodes": {
        "good": {"id": "good", "status": "online", "host": "g", "port": 1, "last_seen": _iso()},
        "bad": {"id": "bad", "status": "online", "host": "b", "port": 2, "last_seen": _iso()}}})

    async def fake_raw(host, port, kind, data, target_id=None, **kw):
        return {"ok": True, "result": 1} if host == "g" else {"ok": False, "error": "boom"}

    monkeypatch.setattr(node, "_dispatch_mesh_task_raw", fake_raw)

    async def run():
        for _ in range(12):
            await node.dispatch_mesh_task("g", 1, "predict", {})
            await node.dispatch_mesh_task("b", 2, "predict", {})

    asyncio.run(run())
    assert node._advisor().summary()["observations"] == 24
    peers = node._get_active_peers_list()
    assert [p["id"] for p in peers][0] == "good"
    assert peers[0]["neural_score"] > peers[1]["neural_score"]


def test_neural_rank_can_be_disabled(node, monkeypatch):
    monkeypatch.setenv("NSM_PEER_NEURAL_RANK", "0")
    node._save_state({"nodes": {"p": {"id": "p", "status": "online", "host": "h", "port": 1}}})
    assert "neural_score" not in node._get_active_peers_list()[0]


def test_state_txn_is_reentrant_no_deadlock(node):
    done = []

    def run():
        with node._state_txn():
            with node._state_txn():
                node._save_state({"nodes": {}, "global_experience": []})
                node._task_inbox()          # يفتح _load/_save داخل نفس القفل
        done.append(1)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(5)
    assert done == [1], "deadlock: nested _state_txn did not return"


def test_sync_experience_not_lost_to_background_writer(node):
    """انحدار: كاتب خلفي (heartbeat/capability) يكتب نسخة قديمة فوق خبرة جديدة."""
    node.join_network()
    node.stop_capability_watch(); node.stop_self_evolution_watch(); node.stop_peer_sync_watch()
    stop = threading.Event()

    def background():
        while not stop.is_set():
            node.send_heartbeat()

    bg = threading.Thread(target=background, daemon=True)
    bg.start()
    try:
        for i in range(30):
            node.sync_experience("evolution_sync", {"i": i})
    finally:
        stop.set(); bg.join(5)
    got = [e["data"]["i"] for e in node.get_evolutionary_updates()]
    assert sorted(got) == list(range(30))


def test_reputation_updates_are_atomic_across_threads(node):
    node.update_reputation("p", 0)

    def bump():
        for _ in range(20):
            node.update_reputation("p", 1)

    ts = [threading.Thread(target=bump) for _ in range(5)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert node.get_reputation("p")["score"] == 100
