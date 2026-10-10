# -*- coding: utf-8 -*-
"""المشرف: إعادة تشغيل ما يخرج/يتجمّد بتراجع أسّي، بلا عاصفة، وإيقاف نظيف."""
from ai.mesh_supervisor import NodeSpec, Supervisor


class FakeProc:
    def __init__(self):
        self.rc = None
        self.killed = False
        self.terminated = False

    def poll(self):
        return self.rc

    def kill(self):
        self.killed = True
        self.rc = -9

    def terminate(self):
        self.terminated = True
        self.rc = 0

    def wait(self, timeout=None):
        return self.rc


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _sup(n=2, probe=None, **kw):
    clock, procs = Clock(), []

    def popen(argv, cwd=None, env=None):
        p = FakeProc()
        procs.append(p)
        return p

    specs = [NodeSpec(f"n{i}", ["x"], {}, ".", 7000 + i) for i in range(n)]
    logs = []
    sup = Supervisor(specs, popen=popen, now=clock, probe=probe, log=logs.append, **kw)
    sup.start_all()
    return sup, clock, procs, logs


def test_restarts_crashed_node_after_backoff():
    sup, clock, procs, _ = _sup()
    procs[0].rc = 1
    sup.tick()                       # اكتشف الموت وجدول الإعادة (+2ث)
    assert sup.alive_count() == 1 and len(procs) == 2
    clock.t += 1.0
    sup.tick()
    assert len(procs) == 2           # لم يحن الموعد بعد
    clock.t += 1.5
    sup.tick()
    assert len(procs) == 3 and sup.alive_count() == 2 and sup.states[0].restarts == 1


def test_exponential_backoff_capped_no_storm():
    sup, clock, procs, _ = _sup(n=1, max_backoff=20.0)
    waits = []
    for _ in range(6):
        procs[-1].rc = 1
        sup.tick()
        waits.append(sup.states[0].restart_at - clock.t)
        clock.t = sup.states[0].restart_at
        sup.tick()
    assert waits == [2.0, 4.0, 8.0, 16.0, 20.0, 20.0]


def test_backoff_resets_after_stable_run():
    sup, clock, procs, _ = _sup(n=1, stable_after=60.0)
    procs[-1].rc = 1
    sup.tick(); clock.t = sup.states[0].restart_at; sup.tick()   # backoff → 4
    clock.t += 120.0                                              # عملت مستقرة
    procs[-1].rc = 1
    sup.tick()
    assert sup.states[0].restart_at - clock.t == 2.0


def test_hung_node_killed_and_restarted_after_grace():
    state = {"ok": True}
    sup, clock, procs, logs = _sup(n=1, probe=lambda spec: state["ok"],
                                   probe_every=20.0, startup_grace=90.0, health_fails=3)
    state["ok"] = False
    for _ in range(3):               # داخل فترة السماح: لا قتل
        clock.t += 20.0
        sup.tick()
    assert not procs[0].killed
    for _ in range(4):               # أول فحص (80ث) ما زال في السماح؛ ثم 3 إخفاقات = تجمّد
        clock.t += 20.0
        sup.tick()
    assert procs[0].killed and any("تجمّد" in l for l in logs)
    clock.t += 5.0
    sup.tick()
    assert len(procs) == 2


def test_single_probe_failure_does_not_kill():
    state = {"ok": True}
    sup, clock, procs, _ = _sup(n=1, probe=lambda spec: state["ok"], startup_grace=0.0)
    for ok in (False, True, False, True, False):
        state["ok"] = ok
        clock.t += 20.0
        sup.tick()
    assert not procs[0].killed


def test_stop_all_terminates():
    sup, _, procs, _ = _sup(n=3)
    sup.stop_all()
    assert all(p.terminated for p in procs)
