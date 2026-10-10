# -*- coding: utf-8 -*-
"""ai/mesh_supervisor.py — مشرف يُبقي عقد الشبكة تعمل بلا توقف.

كان scripts/run_local_mesh.py يشغّل العقد ثم يكتفي بالانتظار: أي عقدة تنهار تبقى
ميتة للأبد، ولا يعالج أي تجمّد (عملية حيّة لكنها لا ترد). المشرف:
  • يعيد تشغيل كل عملية تخرج، بتراجع أسّي (2ث → حتى 120ث) كي لا تدور عاصفة
    إعادة تشغيل؛ ويُصفَّر التراجع بعد عمل مستقر ≥60ث.
  • يفحص /status لكل عقدة (اختياري): 3 إخفاقات متتالية = تجمّد → قتل وإعادة تشغيل،
    مع فترة سماح بعد كل إقلاع كي لا تُقتل عقدة ما زالت تقلع.
  • لا يستهلك CPU: نبضة كل ثانيتين، وفحص الصحة كل 20ث.
لا يستطيع إنقاذ العقد إن أُوقفت الحاوية/الجهاز نفسه؛ ذلك دور restart: always في
Docker/systemd (انظر docker-compose.mesh.yml).
"""
from __future__ import annotations

import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class NodeSpec:
    name: str
    argv: List[str]
    env: Dict[str, str]
    cwd: str
    port: int = 0


@dataclass
class _State:
    spec: NodeSpec
    proc: Optional[object] = None
    started_at: float = 0.0
    backoff: float = 2.0
    restart_at: float = 0.0
    health_fails: int = 0
    restarts: int = 0


def http_probe(spec: NodeSpec, timeout: float = 3.0) -> bool:
    if not spec.port:
        return True
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{spec.port}/status", timeout=timeout) as r:
            return 200 <= r.status < 500
    except Exception:
        return False


class Supervisor:
    def __init__(
        self,
        specs: List[NodeSpec],
        popen: Callable = subprocess.Popen,
        now: Callable[[], float] = time.monotonic,
        probe: Optional[Callable[[NodeSpec], bool]] = None,
        *,
        base_backoff: float = 2.0,
        max_backoff: float = 120.0,
        stable_after: float = 60.0,
        health_fails: int = 3,
        probe_every: float = 20.0,
        startup_grace: float = 90.0,
        log: Callable[[str], None] = print,
    ):
        self.states = [_State(s, backoff=base_backoff) for s in specs]
        self._popen, self._now, self._probe = popen, now, probe
        self.base_backoff, self.max_backoff = base_backoff, max_backoff
        self.stable_after, self.health_fails = stable_after, health_fails
        self.probe_every, self.startup_grace = probe_every, startup_grace
        self._last_probe = 0.0
        self._log = log

    def _spawn(self, st: _State) -> None:
        st.proc = self._popen(st.spec.argv, cwd=st.spec.cwd, env=st.spec.env)
        st.started_at = self._now()
        st.health_fails = 0

    def start_all(self, stagger: float = 0.0, sleep: Callable[[float], None] = time.sleep) -> None:
        for st in self.states:
            self._spawn(st)
            if stagger:
                sleep(stagger)

    def _mark_dead(self, st: _State, reason: str) -> None:
        now = self._now()
        if now - st.started_at >= self.stable_after:
            st.backoff = self.base_backoff  # كانت مستقرة: ابدأ التراجع من جديد
        st.restart_at = now + st.backoff
        self._log(f"⚠️ {st.spec.name}: {reason} — إعادة تشغيل بعد {st.backoff:.0f}ث")
        st.backoff = min(st.backoff * 2, self.max_backoff)
        st.proc = None

    def tick(self) -> None:
        now = self._now()
        do_probe = self._probe is not None and (now - self._last_probe) >= self.probe_every
        if do_probe:
            self._last_probe = now
        for st in self.states:
            if st.proc is None:
                if now >= st.restart_at:
                    self._spawn(st)
                    st.restarts += 1
                    self._log(f"🔄 {st.spec.name}: أُعيد تشغيلها (المرة {st.restarts})")
                continue
            if st.proc.poll() is not None:
                self._mark_dead(st, f"خرجت بالرمز {st.proc.poll()}")
                continue
            if do_probe and (now - st.started_at) >= self.startup_grace:
                if self._probe(st.spec):
                    st.health_fails = 0
                else:
                    st.health_fails += 1
                    if st.health_fails >= self.health_fails:
                        try:
                            st.proc.kill()
                            st.proc.wait(timeout=5)
                        except Exception:
                            pass
                        self._mark_dead(st, "لا ترد على /status (تجمّد)")

    def alive_count(self) -> int:
        return sum(1 for s in self.states if s.proc is not None and s.proc.poll() is None)

    def stop_all(self) -> None:
        for st in self.states:
            if st.proc is not None:
                try:
                    st.proc.terminate()
                except Exception:
                    pass
        deadline = self._now() + 5
        for st in self.states:
            if st.proc is not None:
                try:
                    st.proc.wait(timeout=max(0.1, deadline - self._now()))
                except Exception:
                    try:
                        st.proc.kill()
                    except Exception:
                        pass
