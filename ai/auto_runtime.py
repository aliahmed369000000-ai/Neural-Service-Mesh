"""
NSM Auto Runtime — تشغيل تلقائي مستمر بدون انتظار أمر
=======================================================
يجمع ويشغّل في الخلفية:
  1) AutonomousWill — بحث + تعلّم + اقتراحات تطوير
  2) فحوصات مشروع آمنة عبر NSM Terminal (git/compile)
  3) دورات تعلّم ذاتي خفيفة
  4) نبض صحة دوري

الفلسفة: الوكلاء والنظام يعملون من تلقاء أنفسهم؛ الأوامر البشرية اختيارية.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent
_STATE = ROOT / "memory" / "auto_runtime_state.json"
_LOG = ROOT / "memory" / "auto_runtime_actions.jsonl"

# أسرع من الإرادة وحدها — نبض شامل
INTERVAL_S = 45.0
HEALTH_EVERY_N = 3          # كل 3 نبضات: فحص مشروع
LEARN_EVERY_N = 2           # كل نبضتين: تعلّم مستمر
MAX_ACTIONS_HOUR = 30


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load() -> dict:
    if _STATE.is_file():
        try:
            return json.loads(_STATE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {
        "enabled": True,
        "ticks": 0,
        "started_at": None,
        "last_tick_at": None,
        "hour_bucket": "",
        "hour_actions": 0,
        "last_health": None,
        "last_learn": None,
    }


def _save(st: dict) -> None:
    _STATE.parent.mkdir(parents=True, exist_ok=True)
    _STATE.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")


def _log(entry: dict) -> None:
    try:
        _LOG.parent.mkdir(parents=True, exist_ok=True)
        with _LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


class AutoRuntime:
    """منسّق التشغيل التلقائي للنظام."""

    def __init__(self, interval_s: float = INTERVAL_S):
        self.interval_s = float(interval_s)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.state = _load()

    def enable(self, v: bool = True) -> None:
        self.state["enabled"] = bool(v)
        _save(self.state)

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self.state["started_at"] = self.state.get("started_at") or _now()
        self.state["enabled"] = True
        _save(self.state)
        # تأكد من تشغيل الإرادة + تفعيل التعلّم المستمر
        try:
            from ai.continuous_training_agent import enable_continuous_learning, is_continuous_enabled
            if not is_continuous_enabled():
                enable_continuous_learning(True)
            else:
                enable_continuous_learning(True)  # ثبّت العلم والـ config
        except Exception as e:
            logger.debug("continuous enable: %s", e)
        try:
            from ai.autonomous_will import get_autonomous_will
            get_autonomous_will(start=True)
        except Exception as e:
            logger.debug("will start: %s", e)
        self._thread = threading.Thread(
            target=self._loop, name="NSM-AutoRuntime", daemon=True
        )
        self._thread.start()
        logger.info("[AutoRuntime] started interval=%ss", self.interval_s)

    def stop(self) -> None:
        self._running = False
        self.state["enabled"] = False
        _save(self.state)

    def _loop(self) -> None:
        time.sleep(5)
        while self._running:
            try:
                if self.state.get("enabled", True):
                    self.tick()
            except Exception as e:
                logger.warning("[AutoRuntime] tick error: %s", e)
            time.sleep(self.interval_s)

    def _rate_ok(self) -> bool:
        hour = datetime.now(timezone.utc).strftime("%Y%m%d%H")
        if self.state.get("hour_bucket") != hour:
            self.state["hour_bucket"] = hour
            self.state["hour_actions"] = 0
        return int(self.state.get("hour_actions", 0)) < MAX_ACTIONS_HOUR

    def tick(self) -> dict:
        with self._lock:
            result: Dict[str, Any] = {"ts": _now(), "phases": {}}
            if not self._rate_ok():
                result["skipped"] = "rate_limit"
                return result

            self.state["ticks"] = int(self.state.get("ticks", 0)) + 1
            ticks = self.state["ticks"]
            self.state["last_tick_at"] = _now()

            # 1) دائماً: نبضة إرادة (قد لا تفعل شيئاً إن كانت الرغبة منخفضة)
            try:
                from ai.autonomous_will import get_autonomous_will
                will = get_autonomous_will(start=True)
                # ارفع الرغبة قليلاً كل نبضة لتجنب الخمول الدائم
                d = will._state.setdefault("desire", {})
                for k in ("curiosity", "growth", "hunger"):
                    d[k] = min(1.0, float(d.get(k, 0.3)) + 0.04)
                will_res = will.tick()
                result["phases"]["will"] = {
                    "acted": will_res is not None,
                    "summary": (will_res or {}).get("topic") if will_res else None,
                    "motive": (will_res or {}).get("motive") if will_res else None,
                }
                if will_res:
                    self.state["hour_actions"] = int(self.state.get("hour_actions", 0)) + 1
            except Exception as e:
                result["phases"]["will"] = {"error": str(e)}

            # 2) فحص صحة المشروع تلقائياً
            if ticks % HEALTH_EVERY_N == 0:
                health = self._auto_health()
                result["phases"]["health"] = health
                self.state["last_health"] = health
                self.state["hour_actions"] = int(self.state.get("hour_actions", 0)) + 1

            # 3) دورة تعلّم دورية
            if ticks % LEARN_EVERY_N == 0:
                learn = self._auto_learn()
                result["phases"]["learn"] = learn
                self.state["last_learn"] = {
                    "ok": learn.get("ok"),
                    "ts": _now(),
                }
                if learn.get("ok"):
                    self.state["hour_actions"] = int(self.state.get("hour_actions", 0)) + 1

            _save(self.state)
            _log(result)
            return result

    def _auto_health(self) -> dict:
        """فحوصات آمنة عبر الطرفيه — بدون تدخل بشري."""
        out: Dict[str, Any] = {"ok": True, "checks": []}
        try:
            from ai.nsm_terminal import get_terminal
            term = get_terminal()
            for name in ("status", "compile_ai", "python"):
                try:
                    r = term.quick(name, mode="safe")
                    out["checks"].append({
                        "name": name,
                        "ok": r.ok,
                        "exit": r.exit_code,
                        "ms": r.duration_ms,
                        "tail": ((r.stdout or r.stderr or r.error or "")[-200:]),
                    })
                    if not r.ok and name == "compile_ai":
                        out["ok"] = False
                except Exception as e:
                    out["checks"].append({"name": name, "ok": False, "error": str(e)})
                    out["ok"] = False
        except Exception as e:
            return {"ok": False, "error": str(e)}
        return out

    def _auto_learn(self) -> dict:
        """تعلّم مستمر كامل إن كان مفعّلاً."""
        try:
            from ai.continuous_training_agent import (
                is_continuous_enabled,
                run_continuous_learning_pulse,
            )
            if not is_continuous_enabled():
                return {"ok": False, "msg": "continuous learning disabled"}
            return run_continuous_learning_pulse()
        except Exception as e:
            try:
                from ai.self_feed_learner import self_learn_cycle
                return self_learn_cycle(limit=2)
            except Exception as e2:
                return {"ok": False, "error": f"{e} | {e2}"}

    def status(self) -> dict:
        st = {
            "running": self._running,
            "enabled": self.state.get("enabled", True),
            "interval_s": self.interval_s,
            "ticks": self.state.get("ticks"),
            "last_tick_at": self.state.get("last_tick_at"),
            "last_health_ok": (self.state.get("last_health") or {}).get("ok"),
            "last_learn": self.state.get("last_learn"),
            "hour_actions": self.state.get("hour_actions"),
            "continuous_learning": None,
        }
        try:
            from ai.continuous_training_agent import is_continuous_enabled
            st["continuous_learning"] = is_continuous_enabled()
        except Exception:
            pass
        return st


_rt: Optional[AutoRuntime] = None
_rt_lock = threading.Lock()


def get_auto_runtime(start: bool = True) -> AutoRuntime:
    global _rt
    with _rt_lock:
        if _rt is None:
            _rt = AutoRuntime()
            if start:
                _rt.start()
        elif start and not _rt._running:
            _rt.start()
        return _rt


def handle_auto_command(user_input: str) -> Optional[str]:
    import re
    t = (user_input or "").strip()
    if not t:
        return None
    if not re.search(
        r"(تشغيل\s*تلقائي|auto\s*runtime|حالة\s*التشغيل|أتمتة|"
        r"أوقف\s*التشغيل\s*التلقائي|شغ[ّل]ل\s*التشغيل\s*التلقائي|نبضة\s*تلقائية)",
        t,
        re.I,
    ):
        return None
    rt = get_auto_runtime(start=True)
    low = t.lower()
    if re.search(r"أوقف\s*التشغيل\s*التلقائي|stop\s*auto", low, re.I):
        rt.enable(False)
        return "## ⏸ التشغيل التلقائي\nتم التعليق."
    if re.search(r"شغ[ّل]ل\s*التشغيل\s*التلقائي|start\s*auto", low, re.I):
        rt.enable(True)
        rt.start()
        return "## ▶ التشغيل التلقائي\nمفعّال ويعمل في الخلفية."
    if re.search(r"نبضة\s*تلقائية|auto\s*tick", low, re.I):
        res = rt.tick()
        return "## ⚡ نبضة تلقائية\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"
    return (
        "## 🤖 التشغيل التلقائي NSM\n"
        "يعمل **بدون أوامر**: إرادة + فحص مشروع + تعلّم دوري.\n\n"
        "```json\n"
        + json.dumps(rt.status(), ensure_ascii=False, indent=2)
        + "\n```"
    )
