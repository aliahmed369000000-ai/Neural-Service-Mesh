"""الاختيار الاستباقي بالشبكة العصبية (ai/decision.py::suggest_preemptive +
core/engine.py::run_path): بديل مكافئ أرجح نجاحاً يُنفَّذ بدل عقدة سيئة *قبل*
تشغيلها. الضمانات: لا استباق بشبكة غير مدرَّبة، ولا لبديل غير مكافئ (جيران الرسم
عادة عقد لاحقة مختلفة)، استكشاف دوري للأصلي، ومفتاح إيقاف NSM_PREEMPTIVE_ROUTING=0."""
import random
import shutil
import tempfile

import pytest

from ai.decision import AIDecisionLayer
from ai.node_neural_advisor import NodeNeuralAdvisor
from core.engine import ExecutionEngine
from core.graph import ServiceGraph
from core.registry import NodeRegistry
from services.generated_service_nodes import AnalyzerNode, NormalizerNode
from storage.file_storage import FileStorage


class Flaky(NormalizerNode):
    def __init__(self, name, p, seed):
        super().__init__(name=name)
        self._p, self._rnd = p, random.Random(seed)

    def process(self, data):
        if self._rnd.random() < self._p:
            raise RuntimeError("فشل عابر")
        return super().process(data)


def _world(min_obs=30):
    tmp = tempfile.mkdtemp()
    st = FileStorage(tmp)
    reg, g = NodeRegistry(st), ServiceGraph()
    P, B, G = Flaky("p", 0.5, 1), Flaky("b", 0.85, 2), Flaky("g", 0.0, 3)
    nodes = [P, B, G]
    for n in nodes:
        reg.register(n)
        g.add_node(n.node_id, n.to_dict())
    g.add_edge(P.node_id, B.node_id)
    g.add_edge(P.node_id, G.node_id)
    adv = NodeNeuralAdvisor(min_observations=min_obs)
    ai = AIDecisionLayer(graph=g, advisor=adv, state_lookup=lambda i: getattr(reg.get(i), "state", None))
    return tmp, reg, g, adv, ai, ExecutionEngine(reg, g, st, ai=ai), (P, B, G)


def _warm(eng, P, n=120):
    for i in range(n):
        eng.run_path([P.node_id], {"raw_data": f"نص {i}"})


def _late_failures(eng, P, n=100):
    fails = pre = 0
    for i in range(n):
        r = eng.run_path([P.node_id], {"raw_data": f"نص {i}"})
        fails += len(r.failed_attempts) + (r.status != "success")
        pre += bool(r.steps[0].to_dict().get("preempted_from"))
    return fails, pre


def test_preempt_reduces_wasted_failures_and_keeps_success(monkeypatch):
    monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "0")
    tmp, *_, eng, (P, B, G) = _world()
    try:
        _warm(eng, P)
        base_fails, base_pre = _late_failures(eng, P)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert base_pre == 0

    monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "1")
    tmp, *_, eng, (P, B, G) = _world()
    try:
        _warm(eng, P)
        fails, pre = _late_failures(eng, P)
        r = eng.run_path([P.node_id], {"raw_data": "اختبار"})
        assert r.status == "success"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert pre > 0 and fails < base_fails


def test_preempted_step_records_origin_and_good_replica(monkeypatch):
    monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "1")
    tmp, reg, g, adv, ai, eng, (P, B, G) = _world()
    try:
        _warm(eng, P)
        seen = [eng.run_path([P.node_id], {"raw_data": f"س{i}"}).steps[0].to_dict() for i in range(40)]
        pre = [s for s in seen if s.get("preempted_from")]
        assert pre and all(s["preempted_from"] == P.node_id and s["is_fallback"] for s in pre)
        assert sum(s["node_id"] == G.node_id for s in pre) > sum(s["node_id"] == B.node_id for s in pre)
        assert all("preempted_from" not in s for s in seen if not s.get("preempted_from"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_non_replica_downstream_neighbor_is_never_substituted(monkeypatch):
    """جار لاحق مختلف الصنف بسجل ممتاز (وهو الجار الوحيد) لا تُستبدل به العقدة."""
    monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "1")
    tmp = tempfile.mkdtemp()
    try:
        st = FileStorage(tmp)
        reg, g = NodeRegistry(st), ServiceGraph()
        P, ana = Flaky("p", 0.5, 1), AnalyzerNode("analyzer")
        for n in (P, ana):
            reg.register(n)
            g.add_node(n.node_id, n.to_dict())
        g.add_edge(P.node_id, ana.node_id)
        adv = NodeNeuralAdvisor(min_observations=30)
        ai = AIDecisionLayer(graph=g, advisor=adv, state_lookup=lambda i: getattr(reg.get(i), "state", None))
        eng = ExecutionEngine(reg, g, st, ai=ai)
        for _ in range(60):
            adv.observe(P.node_id, False)
            adv.observe(ana.node_id, True)
        assert adv.is_trained()
        assert not eng._is_replica(P, ana.node_id)
        # بدون فحص المكافئ كان سيُقترح (الدليل أن الحاجز هو التوافق لا الشبكة)
        assert ai.suggest_preemptive(P.node_id) == ana.node_id
        assert ai.suggest_preemptive(P.node_id, is_compatible=lambda c: eng._is_replica(P, c)) is None
        for i in range(30):
            r = eng.run_path([P.node_id], {"raw_data": f"x{i}"})
            assert r.steps[0].to_dict().get("preempted_from") is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_untrained_advisor_never_preempts(monkeypatch):
    monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "1")
    tmp, reg, g, adv, ai, eng, (P, B, G) = _world(min_obs=10_000)
    try:
        for _ in range(60):
            adv.observe(P.node_id, False)
            adv.observe(G.node_id, True)
        assert not adv.is_trained()
        assert ai.suggest_preemptive(P.node_id) is None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_kill_switch_and_periodic_exploration(monkeypatch):
    tmp, reg, g, adv, ai, eng, (P, B, G) = _world()
    try:
        for _ in range(80):
            adv.observe(P.node_id, False)
            adv.observe(G.node_id, True)
        assert adv.is_trained()
        monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "0")
        assert ai.suggest_preemptive(P.node_id) is None
        monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "1")
        picks = [ai.suggest_preemptive(P.node_id) for _ in range(20)]
        # كل قرار عاشر استكشاف (None) ليُكتشف تعافي الأصلي؛ الباقي هو البديل السليم
        assert [i for i, p in enumerate(picks, 1) if p is None] == [10, 20]
        assert {p for p in picks if p} == {G.node_id}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_step_dict_shape_unchanged_when_not_preempted(monkeypatch):
    monkeypatch.setenv("NSM_PREEMPTIVE_ROUTING", "0")
    tmp, *_, eng, (P, B, G) = _world()
    try:
        d = eng.run_path([G.node_id], {"raw_data": "x"}).steps[0].to_dict()
        assert "preempted_from" not in d
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_ai_objects_without_or_with_broken_preempt_do_not_break_runs():
    """واجهة ai اختيارية: كائن بلا suggest_preemptive، أو يرفع استثناءً، لا يُسقط التشغيل."""
    class NoPre:
        def should_fallback(self, *a):
            return None

        def learn_from_run(self, *a):
            pass

    class Boom(NoPre):
        def suggest_preemptive(self, *a, **k):
            raise RuntimeError("boom")

    for ai in (NoPre(), Boom()):
        tmp = tempfile.mkdtemp()
        try:
            st = FileStorage(tmp)
            reg, g = NodeRegistry(st), ServiceGraph()
            n = Flaky("ok", 0.0, 1)
            reg.register(n)
            g.add_node(n.node_id, n.to_dict())
            r = ExecutionEngine(reg, g, st, ai=ai).run_path([n.node_id], {"raw_data": "x"})
            assert r.status == "success"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
