"""ربط الشبكة العصبية بترتيب العمال/الـfailover: تطبيع السمعة، تعلّم المنسّق من نتائجه،
ودخول احتمال النجاح في rank_workers. بلا شبكة ولا مفاتيح API."""
import asyncio
from datetime import datetime, timezone

import pytest

from ai.living_mesh import LivingMeshNode
from ai.mesh_job_orchestrator import MeshJobOrchestrator
from ai.node_health_layer import NodeHealthLayer


def _now():
    return datetime.now(timezone.utc).isoformat()


@pytest.fixture
def node(tmp_path, monkeypatch):
    monkeypatch.setenv("NSM_PEER_NEURAL_RANK", "1")
    monkeypatch.setenv("NSM_NEURAL_RANK_WEIGHT", "300")
    n = LivingMeshNode(node_id="me", host="127.0.0.1", port=9001, data_dir=tmp_path)
    n._save_state({"nodes": {
        "good": {"id": "good", "status": "online", "host": "g", "port": 1, "last_seen": _now()},
        "bad": {"id": "bad", "status": "online", "host": "b", "port": 2, "last_seen": _now()}}})
    return n


def _train(node, n=40):
    """حركة متداخلة واقعية (لا كتلة نجاحات ثم كتلة فشل: الأخيرة تُسبّب نسياناً كارثياً)."""
    for _ in range(n):
        node.record_dispatch_outcome("g", 1, True, 20.0, target_id="good")
        node.record_dispatch_outcome("b", 2, False, 900.0, target_id="bad")


def _health(node):
    h = NodeHealthLayer(node)
    # نفس الكمون ونفس الوصولية للعاملين: الفارق الوحيد = الشبكة العصبية
    h._route_cache = {"good": {"ok": True, "rtt_ms": 50.0, "path": "direct"},
                      "bad": {"ok": True, "rtt_ms": 50.0, "path": "direct"}}
    return h


# ── تطبيع السمعة ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("entry,expected", [
    (None, None), ({"score": 0, "events": []}, None), ({"score": 50, "events": [1]}, 0.5),
    ({"score": 150, "events": [1]}, 1.0), ({"score": -7, "events": [1]}, 0.0),
    ({"score": "x", "events": [1]}, None),
])
def test_reputation_normalization(entry, expected):
    assert LivingMeshNode._rep01(entry) == expected


# ── التعلّم من النتائج ───────────────────────────────────────────────────
def test_record_dispatch_outcome_trains_and_scores(node):
    _train(node)
    assert node._advisor().summary()["observations"] == 80
    assert node.neural_score("good") > 0.7 > 0.4 > node.neural_score("bad")


def test_neural_score_none_when_disabled(node, monkeypatch):
    monkeypatch.setenv("NSM_PEER_NEURAL_RANK", "0")
    assert node.neural_score("good") is None
    node.record_dispatch_outcome("g", 1, True, 1.0, target_id="good")   # لا استثناء


# ── rank_workers ─────────────────────────────────────────────────────────
def test_rank_workers_prefers_reliable_peer_when_all_else_equal(node):
    _train(node)
    ranked = _health(node).rank_workers()
    assert [w["peer_id"] for w in ranked] == ["good", "bad"]
    assert ranked[0]["selection_score"] > ranked[1]["selection_score"]
    assert ranked[0]["neural_score"] > ranked[1]["neural_score"]


def test_rank_workers_failover_order_flips_after_good_peer_starts_failing(node):
    _train(node)
    h = _health(node)
    assert h.rank_workers()[0]["peer_id"] == "good"
    for _ in range(150):                                  # good ينهار، bad يتعافى
        node.record_dispatch_outcome("g", 1, False, 900.0, target_id="good")
        node.record_dispatch_outcome("b", 2, True, 20.0, target_id="bad")
    assert h.rank_workers()[0]["peer_id"] == "bad"


def test_rank_workers_weight_zero_disables_neural_term(node, monkeypatch):
    _train(node)
    monkeypatch.setenv("NSM_NEURAL_RANK_WEIGHT", "0")
    ranked = _health(node).rank_workers()
    assert "neural_score" not in ranked[0]
    assert ranked[0]["selection_score"] == ranked[1]["selection_score"]


def test_rank_workers_ignores_non_numeric_neural_score(node):
    class Stub:
        node_id = "me"
        def __getattr__(self, name):          # يحاكي MagicMock: كل شيء موجود وغير رقمي
            return lambda *a, **k: {} if name == "get_reputation" else object()
    h = NodeHealthLayer(node)
    h.node = Stub()
    h.node._get_active_peers_list = lambda **k: []
    h.rank_workers()                           # لا استثناء


# ── المنسّق يتعلّم من حمله ───────────────────────────────────────────────
def _orch(node):
    return MeshJobOrchestrator(node, _health(node))


def test_orchestrator_dispatch_one_trains_advisor(node, monkeypatch):
    async def fake_request(host, port, kind, data, timeout=15.0, **kw):
        if host == "g":
            return {"ok": True, "result": {"ok": True}, "acked": True}
        raise ConnectionError("down")
    monkeypatch.setattr(node, "request_from_peer", fake_request)
    orch = _orch(node)

    async def run():
        for _ in range(40):
            await orch._dispatch_one({"peer_id": "good", "host": "g", "port": 1}, "predict", {}, 1.0)
            await orch._dispatch_one({"peer_id": "bad", "host": "b", "port": 2}, "predict", {}, 1.0)
    asyncio.run(run())
    assert node._advisor().summary()["observations"] == 80
    assert node.neural_score("good") > node.neural_score("bad")
    assert [w["peer_id"] for w in orch._rank_workers()] == ["good", "bad"]


def test_orchestrator_does_not_blame_worker_for_non_retryable_errors(node, monkeypatch):
    async def fake_request(host, port, kind, data, timeout=15.0, **kw):
        return {"ok": False, "error": "duplicate_rejected", "result": None}
    monkeypatch.setattr(node, "request_from_peer", fake_request)
    orch = _orch(node)
    out = asyncio.run(orch._dispatch_one({"peer_id": "good", "host": "g", "port": 1}, "predict", {}, 1.0))
    assert out["ok"] is False and out["error"] == "duplicate_rejected"
    adv = node._advisor()
    assert adv is None or adv.summary()["observations"] == 0


def test_mark_offline_persists_advisor(node, tmp_path):
    node.join_network()
    node.stop_capability_watch(); node.stop_self_evolution_watch(); node.stop_peer_sync_watch()
    node.record_dispatch_outcome("g", 1, True, 5.0, target_id="good")   # أقل من save_every
    node.mark_offline()
    from ai.node_neural_advisor import NodeNeuralAdvisor
    reloaded = NodeNeuralAdvisor(node.data_dir / "peer_neural_advisor.json")
    assert reloaded.summary()["observations"] == 1


# ── جودة الترتيب نفسها (انحدار على العيب الذي كشفه القياس) ─────────────────
def _trial(seed, n, p_good, p_bad, lat_good, lat_bad):
    import random
    from ai.node_neural_advisor import NodeNeuralAdvisor
    rnd = random.Random(seed)
    adv = NodeNeuralAdvisor(None)
    for _ in range(n):
        adv.observe("good", rnd.random() < p_good, lat_good, "active", None)
        adv.observe("bad", rnd.random() < p_bad, lat_bad, "active", None)
    return adv


@pytest.mark.parametrize("p_good,p_bad,lat_g,lat_b,min_correct", [
    (0.95, 0.40, 30, 300, 58),   # فارق كبير
    (0.90, 0.70, 50, 50, 50),    # كمون متساوٍ — الشبكة وحدها كانت 23/60 (أسوأ من العملة)
])
def test_ranking_score_orders_peers_correctly_where_nn_alone_failed(p_good, p_bad, lat_g, lat_b, min_correct):
    correct = 0
    for seed in range(60):
        adv = _trial(seed, 100, p_good, p_bad, lat_g, lat_b)
        correct += adv.ranking_score("good", "active", None) > adv.ranking_score("bad", "active", None)
    assert correct >= min_correct, f"{correct}/60"


def test_ranking_score_weight_extremes():
    adv = _trial(3, 100, 0.95, 0.40, 30, 300)
    assert adv.ranking_score("good", nn_weight=0) == pytest.approx(adv.success_rate("good"))
    assert adv.ranking_score("good", nn_weight=1) == pytest.approx(adv.predict("good", None, None))
    assert 0.0 <= adv.ranking_score("unseen") <= 1.0


def test_neural_blend_env_controls_weight(node, monkeypatch):
    _train(node)
    adv = node._advisor()
    monkeypatch.setenv("NSM_NEURAL_BLEND", "0")
    assert node.neural_score("good") == pytest.approx(adv.success_rate("good"))
    monkeypatch.setenv("NSM_NEURAL_BLEND", "not-a-number")
    assert 0.0 <= node.neural_score("good") <= 1.0
