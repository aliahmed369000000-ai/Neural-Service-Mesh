"""
Phase 15 – Drive Engine
========================
الدافعية الداخلية الحقيقية.

الجهاز الآن لديه فضول (CuriosityEngine في signal_stream.py) لكنه
سلبي بدون مدخلات خارجية.  هذا الملف يضيف دوافع داخلية نابعة من
الحالة الراهنة للجهاز نفسه — مثل المخلوقات الحية تماماً:

  • DATA_HUNGER  — جوع للبيانات (يشتد عند قلة المدخلات)
  • BOREDOM      — ملل من التكرار (يشتد عند تشابه المعالجة)
  • ANXIETY      — قلق من الضعف  (يشتد عند تدهور الأداء)
  • GROWTH_URGE  — رغبة في التطور (يشتد مع مرور الوقت بدون تحسين)
  • REST_NEED    — حاجة للراحة  (يشتد بعد نشاط مكثف)

كل دافع:
  • له شدة (intensity) من 0.0 إلى 1.0
  • تتراكم مع الوقت (accumulate)
  • تنخفض بعد الإشباع (satisfy)
  • يُرسل signal تلقائياً إلى SignalBus عند تجاوز عتبة معينة
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICK_INTERVAL_S        = 5.0     # ثواني بين كل tick
SIGNAL_THRESHOLD       = 0.6     # الشدة التي تُطلق signal
SATISFACTION_DECAY     = 0.35    # كم تنخفض الشدة بعد الإشباع
IDLE_ACCUMULATION_RATE = 0.02    # معدل تراكم الجوع عند الخمول

# شدة أولية لكل دافع
INITIAL_INTENSITIES = {
    "DATA_HUNGER":  0.3,
    "BOREDOM":      0.1,
    "ANXIETY":      0.1,
    "GROWTH_URGE":  0.2,
    "REST_NEED":    0.0,
}

# معدلات التراكم الطبيعية لكل دافع (لكل tick)
ACCUMULATION_RATES = {
    "DATA_HUNGER":  0.015,   # يجوع بسرعة معقولة
    "BOREDOM":      0.010,   # يمل ببطء
    "ANXIETY":      0.005,   # يقلق ببطء إلا عند تدهور الأداء
    "GROWTH_URGE":  0.012,   # الرغبة في التطور تتراكم باستمرار
    "REST_NEED":    0.008,   # يحتاج للراحة تدريجياً
}

# الشدة القصوى قبل الإشباع القسري
MAX_INTENSITY = 1.0
MIN_INTENSITY = 0.0


# ---------------------------------------------------------------------------
# Drive
# ---------------------------------------------------------------------------

class Drive:
    """
    A single internal drive / motivation.

    Parameters
    ----------
    name : str
        Unique identifier (e.g. "DATA_HUNGER")
    initial_intensity : float
        Starting intensity in [0, 1]
    accumulation_rate : float
        How much intensity grows per tick when unsatisfied
    description : str
        Human-readable description
    """

    def __init__(
        self,
        name: str,
        initial_intensity: float = 0.0,
        accumulation_rate: float = 0.01,
        description: str = "",
    ):
        self.name              = name
        self.intensity: float  = float(initial_intensity)
        self.accumulation_rate = accumulation_rate
        self.description       = description
        self._satisfied_count  = 0
        self._signal_count     = 0
        self._last_satisfied:  Optional[str] = None
        self._peak_intensity   = initial_intensity

    def tick(self, modifier: float = 1.0):
        """
        Advance time by one tick.

        Parameters
        ----------
        modifier : float
            Multiplier on the accumulation rate.  >1 = accumulates faster.
        """
        self.intensity = min(
            MAX_INTENSITY,
            self.intensity + self.accumulation_rate * modifier,
        )
        if self.intensity > self._peak_intensity:
            self._peak_intensity = self.intensity

    def satisfy(self, amount: float = SATISFACTION_DECAY):
        """
        Reduce intensity by `amount` (simulate drive satisfaction).
        """
        self.intensity = max(MIN_INTENSITY, self.intensity - amount)
        self._satisfied_count += 1
        self._last_satisfied   = _now_iso()

    def is_active(self, threshold: float = SIGNAL_THRESHOLD) -> bool:
        return self.intensity >= threshold

    def to_dict(self) -> dict:
        return {
            "name":             self.name,
            "intensity":        round(self.intensity, 4),
            "peak_intensity":   round(self._peak_intensity, 4),
            "satisfied_count":  self._satisfied_count,
            "signal_count":     self._signal_count,
            "last_satisfied":   self._last_satisfied,
            "description":      self.description,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# DriveEngine
# ---------------------------------------------------------------------------

class DriveEngine:
    """
    Manages the set of internal drives and emits signals when they peak.

    Parameters
    ----------
    signal_bus : optional
        SignalBus instance.  If provided, active drives push Experience
        objects into the bus automatically.
    mesh_ref : optional
        Reference to the NeuralServiceMesh.  Used to read performance
        metrics that modulate anxiety / data-hunger.
    """

    def __init__(
        self,
        signal_bus=None,
        mesh_ref=None,
    ):
        self._bus  = signal_bus
        self._mesh = mesh_ref

        # Initialise drives
        descriptions = {
            "DATA_HUNGER": "جوع للبيانات — يشتد عند قلة المدخلات الجديدة",
            "BOREDOM":     "ملل من التكرار — يشتد عند تشابه المهام",
            "ANXIETY":     "قلق من الضعف — يشتد عند تدهور الأداء",
            "GROWTH_URGE": "رغبة في التطور — تشتد بمرور الوقت بدون تحسين",
            "REST_NEED":   "حاجة للراحة — تشتد بعد نشاط مكثف",
        }

        self._drives: Dict[str, Drive] = {
            name: Drive(
                name=name,
                initial_intensity=INITIAL_INTENSITIES[name],
                accumulation_rate=ACCUMULATION_RATES[name],
                description=descriptions[name],
            )
            for name in INITIAL_INTENSITIES
        }

        self._lock          = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._running       = False
        self._tick_count    = 0
        self._signals_emitted = 0
        self._started_at:   Optional[str] = None

        # Satisfaction history (for analytics)
        self._satisfaction_log: List[dict] = []

        logger.info(f"[DriveEngine] Ready  drives={list(self._drives.keys())}")

    # ── Core: tick ─────────────────────────────────────────────────────────

    def tick(self) -> List[dict]:
        """
        Advance all drives by one tick.

        Returns
        -------
        list[dict]
            Active drives (intensity >= SIGNAL_THRESHOLD) with their
            full state.  Also emits signals to SignalBus if connected.
        """
        with self._lock:
            modifiers = self._compute_modifiers()

            for name, drive in self._drives.items():
                drive.tick(modifier=modifiers.get(name, 1.0))

            active = [
                d.to_dict()
                for d in self._drives.values()
                if d.is_active()
            ]

            self._tick_count += 1

            if active:
                self._emit_signals(active)

        return active

    # ── Core: satisfy ──────────────────────────────────────────────────────

    def satisfy(self, drive_name: str, amount: float = SATISFACTION_DECAY):
        """
        Satisfy a drive, reducing its intensity.

        Parameters
        ----------
        drive_name : str
            One of the drive names (DATA_HUNGER, BOREDOM, etc.)
        amount : float
            How much to reduce intensity (0-1).
        """
        drive_name = drive_name.upper()
        with self._lock:
            if drive_name not in self._drives:
                logger.warning(f"[DriveEngine] Unknown drive: {drive_name}")
                return
            self._drives[drive_name].satisfy(amount)
            self._satisfaction_log.append({
                "drive":     drive_name,
                "amount":    round(amount, 4),
                "satisfied_at": _now_iso(),
            })
            # Keep log bounded
            if len(self._satisfaction_log) > 1000:
                self._satisfaction_log = self._satisfaction_log[-1000:]

    def satisfy_all(self, amount: float = 0.2):
        """Partially satisfy all drives at once (e.g. after a learning cycle)."""
        for name in self._drives:
            self.satisfy(name, amount)

    # ── Query ──────────────────────────────────────────────────────────────

    def get_drives(self) -> dict:
        """Return the current state of all drives."""
        with self._lock:
            return {
                name: drive.to_dict()
                for name, drive in self._drives.items()
            }

    def get_dominant_drive(self) -> Optional[str]:
        """Return the name of the most intense drive, or None if all are low."""
        with self._lock:
            if not self._drives:
                return None
            dominant = max(self._drives.values(), key=lambda d: d.intensity)
            return dominant.name if dominant.intensity >= SIGNAL_THRESHOLD else None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def start(self, interval_s: float = TICK_INTERVAL_S):
        """Start the autonomous tick loop in a background thread."""
        if self._running:
            logger.warning("[DriveEngine] Already running.")
            return
        self._interval   = interval_s
        self._running    = True
        self._started_at = _now_iso()
        self._thread = threading.Thread(
            target=self._run_loop, name="DriveEngine", daemon=True
        )
        self._thread.start()
        logger.info(f"[DriveEngine] Started  tick_interval={interval_s}s")

    def stop(self):
        """Stop the tick loop."""
        self._running = False
        logger.info("[DriveEngine] Stopped.")

    def _run_loop(self):
        while self._running:
            try:
                self.tick()
            except Exception as exc:
                logger.error(f"[DriveEngine] Tick error: {exc}")
            time.sleep(getattr(self, "_interval", TICK_INTERVAL_S))

    # ── Modifiers (context-aware) ──────────────────────────────────────────

    def _compute_modifiers(self) -> Dict[str, float]:
        """
        Compute dynamic multipliers on accumulation rates based on
        real mesh state (if mesh_ref is connected).

        Returns 1.0 everywhere when no mesh is connected.
        """
        modifiers = {name: 1.0 for name in self._drives}

        if self._mesh is None:
            return modifiers

        try:
            # DATA_HUNGER ↑ when WorldFeed is off or no recent items
            wf = getattr(self._mesh, "world_feed", None)
            if wf is None:
                modifiers["DATA_HUNGER"] = 2.0
            elif hasattr(wf, "get_feed_stats"):
                stats = wf.get_feed_stats()
                if stats.get("total_accepted", 0) < 10:
                    modifiers["DATA_HUNGER"] = 1.8

            # BOREDOM ↑ when SignalBus repetition is high
            sb = getattr(self._mesh, "signal_bus", None)
            if sb is not None and hasattr(sb, "get_stats"):
                s = sb.get_stats()
                # If dream mode percentage is very high, things are repetitive
                total = s.get("total_signals", 1) or 1
                dream = s.get("dream_signals", 0)
                if dream / total > 0.7:
                    modifiers["BOREDOM"] = 1.6

            # ANXIETY ↑ when scoring shows low recent performance
            scoring = getattr(self._mesh, "scoring", None)
            if scoring is not None and hasattr(scoring, "get_success_rate"):
                sr = scoring.get_success_rate()
                if isinstance(sr, (int, float)) and sr < 0.5:
                    modifiers["ANXIETY"] = 2.0
                elif isinstance(sr, (int, float)) and sr < 0.7:
                    modifiers["ANXIETY"] = 1.4

            # GROWTH_URGE ↑ when structural evolution hasn't mutated recently
            se = getattr(self._mesh, "structural_evolution", None)
            if se is None:
                modifiers["GROWTH_URGE"] = 1.5

            # REST_NEED ↑ when tick count is very high (sustained activity)
            if self._tick_count > 500:
                modifiers["REST_NEED"] = min(2.0, 1.0 + self._tick_count / 1000.0)

        except Exception as exc:
            logger.debug(f"[DriveEngine] Modifier computation error: {exc}")

        return modifiers

    # ── Signal emission ────────────────────────────────────────────────────

    def _emit_signals(self, active_drives: List[dict]):
        """Push drive signals into SignalBus if connected."""
        if self._bus is None:
            return

        try:
            # Import Experience here to avoid circular import
            # (signal_stream is a sibling module)
            from ai.signal_stream import Experience
            import numpy as np

            for drive_info in active_drives:
                intensity = drive_info["intensity"]

                # Encode drive as a feature vector [0..6]
                drive_names = list(self._drives.keys())
                idx = drive_names.index(drive_info["name"]) if drive_info["name"] in drive_names else 0
                feature_vec = [0.0] * 7
                feature_vec[idx % 7] = intensity

                exp = Experience(
                    feature_vec=np.array(feature_vec, dtype=float),
                    target=intensity,
                    source=f"DriveEngine:{drive_info['name']}",
                    reward=intensity - SIGNAL_THRESHOLD,  # positive = urgent
                    context={
                        "drive":       drive_info["name"],
                        "intensity":   intensity,
                        "description": drive_info.get("description", ""),
                    },
                )

                # Push directly into replay buffer if available
                if hasattr(self._bus, "_replay") and self._bus._replay is not None:
                    self._bus._replay.add(exp)
                    self._drives[drive_info["name"]]._signal_count += 1
                    self._signals_emitted += 1

        except Exception as exc:
            logger.debug(f"[DriveEngine] Signal emit error: {exc}")

    # ── Wiring helpers ─────────────────────────────────────────────────────

    def set_signal_bus(self, signal_bus):
        """Wire in a SignalBus after construction."""
        self._bus = signal_bus

    def set_mesh(self, mesh):
        """Wire in the mesh reference for context-aware modifiers."""
        self._mesh = mesh

    # ── Summary ────────────────────────────────────────────────────────────

    def summary(self) -> dict:
        with self._lock:
            drives_state = {n: d.to_dict() for n, d in self._drives.items()}
        return {
            "component":        "DriveEngine",
            "running":          self._running,
            "tick_count":       self._tick_count,
            "signals_emitted":  self._signals_emitted,
            "started_at":       self._started_at,
            "dominant_drive":   self.get_dominant_drive(),
            "drives":           drives_state,
            "satisfaction_log_size": len(self._satisfaction_log),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 60)
    print("  DriveEngine — Self-Test")
    print("=" * 60)

    engine = DriveEngine()

    print("\n  Initial drives:")
    for name, d in engine.get_drives().items():
        print(f"    {name}: intensity={d['intensity']:.3f}")

    print("\n  Simulating 20 ticks...")
    for i in range(20):
        active = engine.tick()
        if active:
            print(f"  Tick {i+1:02d}  active={[d['name'] for d in active]}")

    print("\n  Drives after 20 ticks:")
    for name, d in engine.get_drives().items():
        print(f"    {name}: intensity={d['intensity']:.3f}")

    print("\n  Satisfying DATA_HUNGER...")
    engine.satisfy("DATA_HUNGER", amount=0.8)
    d = engine.get_drives()["DATA_HUNGER"]
    print(f"    DATA_HUNGER after satisfy: intensity={d['intensity']:.3f}")

    print("\n  Dominant drive:", engine.get_dominant_drive())

    print("\n  summary():")
    s = engine.summary()
    print(f"    tick_count={s['tick_count']}  signals_emitted={s['signals_emitted']}")
    print(f"    dominant={s['dominant_drive']}")

    print("\n✓ DriveEngine self-test PASSED")
