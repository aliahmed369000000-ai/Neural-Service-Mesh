# -*- coding: utf-8 -*-
"""الشبكة العصبية داخل قرارات العُقد (ai/node_neural_advisor.py + ai/decision.py +
MeshBundle): تتدرّب آلياً من النتائج الفعلية وتختار البديل عند الفشل."""
from __future__ import annotations

import json
import random
import shutil
import tempfile
import threading

import pytest

from ai.decision import AIDecisionLayer
from ai.node_neural_advisor import FEATURE_NAMES, NodeNeuralAdvisor
from core.engine import ExecutionEngine
from core.mesh_bundle import AGENT_CATALOGUE, MeshBundle
from core.node import BaseNode, NodeSchema
from core.node_hands import LEFT


def _train_good_bad(adv, n=300, seed=1):
    rnd = random.Random(seed)
    for i in range(n):
        adv.observe("good", rnd.random() < 0.95, 100, "active")
        adv.observe("bad", rnd.random() < 0.15, 3000, "failed" if i % 3 else "active")


# ── المستشار وحده ─────────────────────────────────────────────────────────
def test_cold_start_is_neutral_and_untrained():
    adv = NodeNeuralAdvisor()
    assert adv.predict("x") == pytest.approx(0.5) and not adv.is_trained()


def test_learns_to_separate_healthy_from_failing_node():
    adv = NodeNeuralAdvisor()
    _train_good_bad(adv)
    assert adv.is_trained()
    good, bad = adv.predict("good", "active"), adv.predict("bad", "failed")
    assert good > 0.7 and bad < 0.35 and good - bad > 0.4
    assert [n for n, _ in adv.rank(["bad", "good"])] == ["good", "bad"]


def test_rank_ties_keep_original_order_and_predict_is_clamped():
    adv = NodeNeuralAdvisor()
    assert [n for n, _ in adv.rank(["c", "a", "b"])] == ["c", "a", "b"]
    for state in ("active", "failed", "paused", None):
        assert 0.0 <= adv.predict("z", state, reputation=5.0) <= 1.0


def test_persistence_roundtrip_identical_predictions(tmp_path):
    path = tmp_path / "adv.json"
    adv = NodeNeuralAdvisor(path, save_every=0)
    _train_good_bad(adv, n=120)
    assert adv.save() and path.exists() and path.with_suffix(".net.json").exists()
    adv2 = NodeNeuralAdvisor(path)
    assert adv2.summary()["observations"] == adv.summary()["observations"]
    assert adv2.predict("good", "active") == pytest.approx(adv.predict("good", "active"))


def test_autosave_every_n_observations(tmp_path):
    path = tmp_path / "adv.json"
    adv = NodeNeuralAdvisor(path, save_every=10)
    for _ in range(10):
        adv.observe("n", True)
    assert path.exists()


@pytest.mark.parametrize("break_file", ["state", "net", "schema"])
def test_corrupt_or_stale_saved_state_starts_fresh_without_raising(tmp_path, break_file):
    path = tmp_path / "adv.json"
    adv = NodeNeuralAdvisor(path, save_every=0)
    _train_good_bad(adv, n=40)
    adv.save()
    if break_file == "state":
        path.write_text("{not json", encoding="utf-8")
    elif break_file == "net":
        path.with_suffix(".net.json").write_text("garbage", encoding="utf-8")
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["features"] = list(FEATURE_NAMES)[:-1]
        path.write_text(json.dumps(data), encoding="utf-8")
    fresh = NodeNeuralAdvisor(path)
    assert fresh.summary()["observations"] == 0 and fresh.predict("good") == pytest.approx(0.5)


def test_thread_safety_counts_every_observation():
    adv = NodeNeuralAdvisor()
    errors = []

    def work(k):
        try:
            for i in range(60):
                adv.observe(f"n{k % 3}", i % 2 == 0, 10, "active")
                adv.predict(f"n{k % 3}", "active")
        except Exception as e:  # pragma: no cover
            errors.append(e)

    ts = [threading.Thread(target=work, args=(k,)) for k in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors and adv.summary()["observations"] == 360


# ── AIDecisionLayer ───────────────────────────────────────────────────────
class _G:
    def __init__(self, nb):
        self.nb = nb

    def get_neighbors(self, nid):
        if nid not in self.nb:
            raise KeyError(nid)
        return list(self.nb[nid])


def test_fallback_without_advisor_is_legacy_first_neighbor():
    ai = AIDecisionLayer(graph=_G({"A": ["B", "C"]}))
    assert ai.should_fallback("A", "err") == "B"


def test_fallback_with_trained_advisor_picks_most_likely_to_succeed():
    adv = NodeNeuralAdvisor()
    for _ in range(60):
        adv.observe("B", False, 2000, "active")
        adv.observe("C", True, 50, "active")
    ai = AIDecisionLayer(graph=_G({"A": ["B", "C"]}), advisor=adv)
    assert ai.should_fallback("A", "err") == "C"


def test_fallback_skips_paused_and_self_and_returns_none_if_no_eligible():
    adv = NodeNeuralAdvisor()
    states = {"B": "paused", "C": "active", "A": "failed"}
    ai = AIDecisionLayer(graph=_G({"A": ["A", "B", "C"], "X": ["B"]}), advisor=adv,
                         state_lookup=states.get)
    assert ai.should_fallback("A", "e") == "C"
    assert ai.should_fallback("X", "e") is None


def test_learn_from_run_trains_automatically_with_state_before():
    class Rec:
        def __init__(self):
            self.calls = []

        def observe(self, *a):
            self.calls.append(a)

    rec = Rec()
    ai = AIDecisionLayer(advisor=rec, state_lookup=lambda n: "WRONG-POST-STATE")
    ai.learn_from_run({"path": [], "status": "failed", "steps": [
        {"node_id": "n1", "status": "success", "duration_ms": 5.0, "state_before": "active"},
        {"node_id": "n2", "status": "error", "duration_ms": 9.0, "state_before": "failed"},
        {"node_id": "n3", "status": "running"},
    ]})
    assert [c[0] for c in rec.calls] == ["n1", "n2"]
    assert rec.calls[0][1] is True and rec.calls[1][1] is False
    assert [c[3] for c in rec.calls] == ["active", "failed"]  # لا تسريب من الحالة اللاحقة


def test_advisor_errors_never_break_decisions():
    class Boom:
        def predict(self, *a):
            raise RuntimeError("x")

        def observe(self, *a):
            raise RuntimeError("x")

        def is_trained(self):
            return True

        def rank(self, *a):
            raise RuntimeError("x")

        def summary(self):
            return {}

    ai = AIDecisionLayer(graph=_G({"A": ["B"]}), advisor=Boom())
    ai.observe_outcome("A", True)
    ai.learn_from_run({"path": ["A"], "status": "success", "steps": [
        {"node_id": "A", "status": "success"}]})
    assert ai._neural_p("A") is None


def test_path_scores_unchanged_until_trained_then_neural_bonus_applies():
    g = _G({})
    base = AIDecisionLayer(graph=g)._score_path(["aaaaaaaa", "bbbbbbbb"]).score
    adv = NodeNeuralAdvisor()
    ai = AIDecisionLayer(graph=g, advisor=adv)
    assert ai._score_path(["aaaaaaaa", "bbbbbbbb"]).score == pytest.approx(base)
    for _ in range(60):
        adv.observe("aaaaaaaa", True, 10, "active")
        adv.observe("bbbbbbbb", True, 10, "active")
    scored = ai._score_path(["aaaaaaaa", "bbbbbbbb"])
    assert scored.score > base and "nn=" in scored.reason
    assert "neural" in ai.get_insights()


# ── تكامل MeshBundle ──────────────────────────────────────────────────────
class _Ok(BaseNode):
    input_schema = NodeSchema(fields={}, required=[])
    output_schema = NodeSchema(fields={}, required=[])

    def process(self, data):
        return {}


class _Fail(_Ok):
    def process(self, data):
        raise ValueError("boom")


@pytest.fixture()
def bundle(monkeypatch):
    tmp = tempfile.mkdtemp(prefix="nsm_neural_test_")
    monkeypatch.chdir(tmp)  # يمنع تلويث مجلدات المستودع النسبية (artifacts/ memory/)
    try:
        yield MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db"), tmp
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_bundle_wires_advisor_into_decision_layer(bundle):
    b, _ = bundle
    assert b.ai_decision._advisor is b.neural_advisor


def test_engine_runs_train_the_network_automatically(bundle):
    b, _ = bundle
    ok, bad = _Ok("ok"), _Fail("bad")
    b.register_node(ok)
    b.register_node(bad)
    eng = ExecutionEngine(b.registry, b.graph, b.storage, ai=b.ai_decision)
    before = b.neural_advisor.summary()["observations"]
    res = eng.run_path([ok.node_id], {})
    assert res.steps[0].state_before in ("created", "active")
    eng.run_path([bad.node_id], {})
    assert b.neural_advisor.summary()["observations"] == before + 2
    assert b.neural_advisor.predict(bad.node_id, "failed") < b.neural_advisor.predict(ok.node_id, "active")


def test_swarm_results_train_the_network_with_pre_outcome_state(bundle):
    from ai.agent_factory import AgentInstance
    from ai.swarm_coordinator import SwarmTask

    b, _ = bundle
    role = next(iter(AGENT_CATALOGUE))
    agent = AgentInstance(role, AGENT_CATALOGUE[role])
    b.agent_factory._agents[agent.agent_id] = agent
    node_id = b.role_node_ids[role]
    before = b.neural_advisor.summary()["observations"]

    class R:
        pass

    t = SwarmTask("t1", "g", "x", {})
    t.assigned_agent_id, t.status, t.error, t.duration_ms = agent.agent_id, "failed", "e", 5.0
    r = R()
    r.tasks = [t]
    b.record_swarm_result(r)
    assert b.neural_advisor.summary()["observations"] == before + 1
    assert list(b.neural_advisor._stats[node_id].window) == [0]


def test_neural_advice_hand_tool(bundle):
    b, _ = bundle
    node = b.registry.get(next(iter(b.role_node_ids.values())))
    res = node.use_hand(LEFT, "neural_advice")
    assert res.ok and 0.0 <= res.output["p_success"] <= 1.0 and res.output["trained"] is False
    assert not node.use_hand(LEFT, "neural_advice", node_id="ghost").ok


def test_advisor_state_survives_bundle_restart(bundle):
    b, tmp = bundle
    for _ in range(40):
        b.neural_advisor.observe("persisted-node", True, 5, "active")
    b.neural_advisor.save()
    b2 = MeshBundle(storage_dir=tmp, db_path=f"{tmp}/mesh.db")
    assert b2.neural_advisor.summary()["observations"] >= 40
    assert b2.neural_advisor.is_trained()


def test_end_to_end_fallback_uses_the_network_choice(bundle):
    b, _ = bundle
    a, bad, good = _Fail("a"), _Ok("bad_peer"), _Ok("good_peer")
    for n in (a, bad, good):
        b.register_node(n)
    b.graph.add_edge(a.node_id, bad.node_id)
    b.graph.add_edge(a.node_id, good.node_id)
    for _ in range(60):
        b.neural_advisor.observe(bad.node_id, False, 3000, "active")
        b.neural_advisor.observe(good.node_id, True, 20, "active")
    eng = ExecutionEngine(b.registry, b.graph, b.storage, ai=b.ai_decision)
    res = eng.run_path([a.node_id], {})
    assert res.status == "success"
    assert res.steps[-1].node_id == good.node_id and res.steps[-1].is_fallback
