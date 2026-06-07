"""
Phase 14 — Complete Digital Being
====================================
"كائن رقمي مكتمل" — A complete digital entity.

Phase 14 closes the loop. All 13 previous phases are unified into a
single, continuously running autonomous lifecycle:

  يعيش → يتجرب → يتعلم → ينام → يحلم → يتطور

  Live    → Experience → Learn → Sleep → Dream → Evolve

Five lifecycle modes run in one perpetual loop:

  1. LIVE    — Normal operation: route requests, execute plans, interact
               with external systems.

  2. EXPERIENCE — Actively seek novel inputs via curiosity engine;
                  log every event as an episodic memory.

  3. LEARN   — Process the day's experiences: train neural networks,
               update reputation, run self-assessment.

  4. SLEEP   — Low-activity consolidation phase: replay buffer flushed,
               semantic rules extracted, weak memories pruned.

  5. DREAM   — Structural self-redesign (Phase 13): propose and test
               architectural mutations, keep improvements.

  6. EVOLVE  — Full evolution pipeline (Phase 7+13): generate new code,
               deploy approved modules, update world model.

This module provides:
  - LifecyclePhase   — enum-like constants for each phase
  - LifecycleClock   — decides when to transition between phases
  - DigitalBeingCore — the main loop orchestrator
  - BeingStatus      — live snapshot of the being's state

File: ai/digital_being.py
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Lifecycle Phase Constants ─────────────────────────────────────────────────
LIVE       = "LIVE"
EXPERIENCE = "EXPERIENCE"
LEARN      = "LEARN"
SLEEP      = "SLEEP"
DREAM      = "DREAM"
EVOLVE     = "EVOLVE"

LIFECYCLE_ORDER = [LIVE, EXPERIENCE, LEARN, SLEEP, DREAM, EVOLVE]

# ── Timing Defaults ───────────────────────────────────────────────────────────
DEFAULT_PHASE_DURATIONS = {
    LIVE:       120.0,   # 2 min of normal operation
    EXPERIENCE:  60.0,   # 1 min of active exploration
    LEARN:       45.0,   # 45 s of learning consolidation
    SLEEP:       30.0,   # 30 s of memory pruning / replay
    DREAM:       45.0,   # 45 s of structural redesign
    EVOLVE:      60.0,   # 1 min of code evolution
}

FAST_PHASE_DURATIONS = {
    LIVE:        20.0,
    EXPERIENCE:  15.0,
    LEARN:       10.0,
    SLEEP:        8.0,
    DREAM:       10.0,
    EVOLVE:      15.0,
}


# ─────────────────────────────────────────────────────────────────────────────
#  Lifecycle Clock
# ─────────────────────────────────────────────────────────────────────────────

class LifecycleClock:
    """
    Decides when to transition between lifecycle phases.

    Transitions are time-based by default, but can also be triggered by:
    - Accumulated experiences (enough new data → transition to LEARN)
    - High curiosity level (transition to EXPERIENCE)
    - Memory pressure (too many stored experiences → transition to SLEEP)
    - Performance degradation (trigger DREAM/EVOLVE sooner)
    """

    def __init__(self, phase_durations: Optional[Dict[str, float]] = None):
        self._durations   = phase_durations or DEFAULT_PHASE_DURATIONS
        self._phase_start = time.time()
        self._current_phase = LIVE
        self._phase_index   = 0

    def set_phase(self, phase: str):
        self._current_phase = phase
        self._phase_index   = LIFECYCLE_ORDER.index(phase) if phase in LIFECYCLE_ORDER else 0
        self._phase_start   = time.time()

    def current_phase(self) -> str:
        return self._current_phase

    def time_in_phase(self) -> float:
        return time.time() - self._phase_start

    def phase_fraction(self) -> float:
        """How far through the current phase (0→1)."""
        dur = self._durations.get(self._current_phase, 60.0)
        return min(1.0, self.time_in_phase() / max(dur, 1.0))

    def should_advance(
        self,
        experience_count: int = 0,
        curiosity_level: float = 0.0,
        memory_pressure: float = 0.0,
        performance_delta: float = 0.0,
    ) -> bool:
        """
        Return True if it's time to advance to the next phase.
        Considers both time elapsed and real-time signals.
        """
        elapsed = self.time_in_phase()
        base_dur = self._durations.get(self._current_phase, 60.0)

        # Trigger overrides (early transition)
        if self._current_phase == LIVE:
            if experience_count >= 20:
                return True
            if curiosity_level > 0.7:
                return True

        elif self._current_phase == EXPERIENCE:
            if experience_count >= 30:
                return True

        elif self._current_phase == LEARN:
            if performance_delta < -0.05:
                return True  # Fast-track to DREAM if learning is degrading

        elif self._current_phase == SLEEP:
            if memory_pressure < 0.3:
                return True  # Memory is consolidated, move on

        elif self._current_phase == DREAM:
            pass  # Let time govern DREAM

        elif self._current_phase == EVOLVE:
            pass  # Let time govern EVOLVE

        return elapsed >= base_dur

    def next_phase(self) -> str:
        self._phase_index = (self._phase_index + 1) % len(LIFECYCLE_ORDER)
        self._current_phase = LIFECYCLE_ORDER[self._phase_index]
        self._phase_start = time.time()
        return self._current_phase

    def summary(self) -> dict:
        return {
            "current_phase":  self._current_phase,
            "time_in_phase_s": round(self.time_in_phase(), 1),
            "phase_fraction":  round(self.phase_fraction(), 3),
            "phase_durations": self._durations,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Being Status
# ─────────────────────────────────────────────────────────────────────────────

class BeingStatus:
    """
    Live snapshot of the digital being's state.
    Combines metrics from all 14 phases into one picture.
    """

    def __init__(self):
        self.phase                = LIVE
        self.cycle_count          = 0
        self.total_experiences    = 0
        self.total_learning_steps = 0
        self.total_sleep_cycles   = 0
        self.total_dreams         = 0
        self.total_evolutions     = 0

        # Phase-specific counters
        self.experiences_this_cycle = 0
        self.learning_steps_this_cycle = 0
        self.redesigns_accepted  = 0
        self.modules_evolved     = 0

        # Health metrics
        self.routing_accuracy    = 0.5
        self.confidence_level    = 0.5
        self.memory_health       = 0.5
        self.curiosity_level     = 0.3
        self.structural_fitness  = 0.5

        # Composite vitality score (0-1)
        self.vitality            = 0.5

        # Timeline
        self.born_at = datetime.now(timezone.utc).isoformat()
        self.last_updated = self.born_at

    def compute_vitality(self) -> float:
        """Composite vitality score from all health dimensions."""
        self.vitality = (
            self.routing_accuracy  * 0.30 +
            self.confidence_level  * 0.20 +
            self.memory_health     * 0.20 +
            self.curiosity_level   * 0.15 +
            self.structural_fitness * 0.15
        )
        self.last_updated = datetime.now(timezone.utc).isoformat()
        return round(self.vitality, 4)

    def to_dict(self) -> dict:
        return {
            "phase":                  self.phase,
            "cycle_count":            self.cycle_count,
            "vitality":               round(self.vitality, 4),
            "total_experiences":      self.total_experiences,
            "total_learning_steps":   self.total_learning_steps,
            "total_sleep_cycles":     self.total_sleep_cycles,
            "total_dreams":           self.total_dreams,
            "total_evolutions":       self.total_evolutions,
            "redesigns_accepted":     self.redesigns_accepted,
            "modules_evolved":        self.modules_evolved,
            "health": {
                "routing_accuracy":   round(self.routing_accuracy, 4),
                "confidence":         round(self.confidence_level, 4),
                "memory":             round(self.memory_health, 4),
                "curiosity":          round(self.curiosity_level, 4),
                "structural_fitness": round(self.structural_fitness, 4),
            },
            "born_at":      self.born_at,
            "last_updated": self.last_updated,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Digital Being Core
# ─────────────────────────────────────────────────────────────────────────────

class DigitalBeingCore:
    """
    Phase 14: The complete digital being.

    Integrates all 13 previous phases into one autonomous lifecycle loop:

        LIVE → EXPERIENCE → LEARN → SLEEP → DREAM → EVOLVE → (repeat)

    Can run:
    - In the background (start_lifecycle() in a thread)
    - Stepped manually (step()) for testing
    - Single-cycle (run_one_cycle()) for demo mode
    """

    def __init__(
        self,
        mesh=None,
        signal_bus=None,
        episodic_memory=None,
        self_awareness=None,
        structural_evolution=None,
        evolution_pipeline=None,
        phase_durations: Optional[Dict[str, float]] = None,
    ):
        self.mesh                = mesh
        self.signal_bus          = signal_bus
        self.episodic_memory     = episodic_memory
        self.self_awareness      = self_awareness
        self.structural_evolution = structural_evolution
        self.evolution_pipeline  = evolution_pipeline

        self.clock   = LifecycleClock(phase_durations or DEFAULT_PHASE_DURATIONS)
        self.status  = BeingStatus()
        self._cycle_history: deque = deque(maxlen=100)

        self._running   = False
        self._thread: Optional[threading.Thread] = None
        self._lock      = threading.Lock()
        self._lifecycle_count = 0

        logger.info(
            "DigitalBeingCore (Phase 14) initialised — "
            "Phases 1-14 unified | lifecycle: "
            + " → ".join(LIFECYCLE_ORDER)
        )

    # ── Phase handlers ────────────────────────────────────────────────────

    def _do_live(self) -> Dict[str, Any]:
        """
        LIVE phase: Normal mesh operation.
        Collect real-time signals, update status metrics.
        """
        result: Dict[str, Any] = {"phase": LIVE, "actions": []}

        # Pulse the signal bus if active
        if self.signal_bus is not None:
            try:
                stats = self.signal_bus.get_stats()
                total = stats.get("total_signals", 0)
                result["signal_count"] = total
                result["actions"].append("signal_bus_pulse")
                # Update routing accuracy from signal stats
                acc = stats.get("avg_loss", None)
                if acc is not None:
                    self.status.routing_accuracy = max(
                        0.0, min(1.0, 1.0 - float(acc))
                    )
            except Exception as e:
                result["signal_error"] = str(e)

        # Update curiosity level
        if self.signal_bus is not None:
            try:
                curiosity = getattr(self.signal_bus, "_curiosity", None)
                if curiosity is not None:
                    self.status.curiosity_level = getattr(
                        curiosity, "_curiosity_level", 0.3
                    )
            except Exception:
                pass

        result["routing_accuracy"] = round(self.status.routing_accuracy, 4)
        result["curiosity_level"]  = round(self.status.curiosity_level, 4)
        return result

    def _do_experience(self) -> Dict[str, Any]:
        """
        EXPERIENCE phase: Actively explore and gather novel experiences.
        Injects curiosity-driven signals into the system.
        """
        result: Dict[str, Any] = {"phase": EXPERIENCE, "actions": [], "new_experiences": 0}

        if self.signal_bus is not None:
            try:
                # Boost curiosity temporarily
                curiosity = getattr(self.signal_bus, "_curiosity", None)
                if curiosity is not None and hasattr(curiosity, "_curiosity_level"):
                    original = curiosity._curiosity_level
                    curiosity._curiosity_level = min(0.9, original * 1.2)
                    result["actions"].append(f"curiosity_boost:{original:.3f}→{curiosity._curiosity_level:.3f}")

                # Generate exploration signals
                for _ in range(5):
                    fv  = [random.gauss(0.5, 0.25) for _ in range(7)]
                    fv  = [max(0.0, min(1.0, x)) for x in fv]
                    tgt = random.random()
                    self.signal_bus.push_real(fv, tgt, reward=0.1)
                    result["new_experiences"] += 1

                result["actions"].append("exploration_signals_injected")
            except Exception as e:
                result["experience_error"] = str(e)

        # Record in episodic memory
        if self.episodic_memory is not None:
            try:
                for _ in range(3):
                    fv  = [random.random() for _ in range(7)]
                    tgt = random.random()
                    self.episodic_memory.record(
                        feature_vec=fv,
                        target=tgt,
                        source="exploration",
                        reward=0.05,
                    )
                    result["new_experiences"] += 1
                result["actions"].append("episodic_records_written")
            except Exception as e:
                result["episodic_error"] = str(e)

        self.status.total_experiences += result["new_experiences"]
        self.status.experiences_this_cycle = result["new_experiences"]
        return result

    def _do_learn(self) -> Dict[str, Any]:
        """
        LEARN phase: Consolidate experiences into learning.
        Run self-assessment, update confidence and weakness maps.
        """
        result: Dict[str, Any] = {"phase": LEARN, "actions": [], "learning_steps": 0}

        # Self-assessment from Phase 12
        if self.self_awareness is not None:
            try:
                signal_stats = (
                    self.signal_bus.get_stats()
                    if self.signal_bus else None
                )
                mem_summary = (
                    self.episodic_memory.summary()
                    if self.episodic_memory else None
                )
                assessment = self.self_awareness.self_assess(signal_stats, mem_summary)
                result["self_assessment"] = {
                    "vitality_score": assessment.get("vitality_score"),
                    "calibration":    assessment.get("calibration_error"),
                    "active_goals":   len(assessment.get("active_goals", [])),
                }
                # Update confidence from assessment
                cal = assessment.get("calibration_error", 0.5)
                self.status.confidence_level = max(0.0, 1.0 - float(cal))
                result["actions"].append("self_assessment")
            except Exception as e:
                result["assessment_error"] = str(e)

        # Train from replay buffer
        if self.signal_bus is not None:
            try:
                buf = getattr(self.signal_bus, "replay_buffer", None)
                if buf is not None:
                    samples = buf.sample(16)
                    for exp in samples:
                        fv  = getattr(exp, "feature_vec", [])
                        tgt = getattr(exp, "target", 0.5)
                        if fv:
                            # Push back through signal bus for retraining
                            self.signal_bus.push_real(fv, tgt, reward=0.0)
                            result["learning_steps"] += 1
                    result["actions"].append(f"replay_train:{result['learning_steps']}_steps")
            except Exception as e:
                result["replay_error"] = str(e)

        self.status.total_learning_steps += result["learning_steps"]
        self.status.learning_steps_this_cycle = result["learning_steps"]
        return result

    def _do_sleep(self) -> Dict[str, Any]:
        """
        SLEEP phase: Memory consolidation and pruning.
        Flush working memory, extract semantic rules, prune weak memories.
        """
        result: Dict[str, Any] = {"phase": SLEEP, "actions": [], "pruned": 0, "rules_extracted": 0}

        if self.episodic_memory is not None:
            try:
                before_summary = self.episodic_memory.summary()

                # Trigger consolidation
                consolidator = getattr(self.episodic_memory, "_consolidator", None)
                if consolidator is not None and hasattr(consolidator, "consolidate"):
                    consolidator.consolidate(self.episodic_memory)
                    result["actions"].append("consolidation_run")

                after_summary = self.episodic_memory.summary()
                result["episode_count"]   = after_summary.get("total_episodes", 0)
                result["semantic_rules"]  = after_summary.get("semantic_rules", 0)
                result["rules_extracted"] = max(
                    0,
                    after_summary.get("semantic_rules", 0)
                    - before_summary.get("semantic_rules", 0)
                )

                # Update memory health
                total_ep = after_summary.get("total_episodes", 0)
                max_ep   = 100_000
                self.status.memory_health = max(0.0, 1.0 - total_ep / max_ep)

            except Exception as e:
                result["sleep_error"] = str(e)

        # Dream consolidation via signal bus
        if self.signal_bus is not None:
            try:
                dream = getattr(self.signal_bus, "_dream", None)
                if dream is not None and hasattr(dream, "consolidate"):
                    dream.consolidate()
                    result["actions"].append("dream_consolidation")
            except Exception:
                pass

        self.status.total_sleep_cycles += 1
        return result

    def _do_dream(self) -> Dict[str, Any]:
        """
        DREAM phase: Structural self-redesign.
        Run Phase 13 architectural mutation cycle.
        """
        result: Dict[str, Any] = {"phase": DREAM, "actions": [], "redesigns": 0, "accepted": 0}

        if self.structural_evolution is not None:
            try:
                cycle_result = self.structural_evolution.run_redesign_cycle(verbose=False)
                result["redesigns"]   += 1
                result["mutation"]     = cycle_result.get("mutation", "unknown")
                result["verdict"]      = cycle_result.get("verdict", "neutral")
                result["delta"]        = cycle_result.get("comparison", {}).get("delta", 0.0)

                if cycle_result.get("accepted"):
                    result["accepted"] += 1
                    self.status.redesigns_accepted += 1
                    self.status.structural_fitness = min(
                        1.0,
                        self.status.structural_fitness + 0.02
                    )
                    result["actions"].append(f"arch_accepted:{cycle_result.get('mutation')}")
                else:
                    result["actions"].append(f"arch_neutral:{cycle_result.get('mutation')}")

            except Exception as e:
                result["dream_error"] = str(e)

        self.status.total_dreams += 1
        return result

    def _do_evolve(self) -> Dict[str, Any]:
        """
        EVOLVE phase: Run the full evolution pipeline.
        Generate new modules, test them, deploy approved ones.
        """
        result: Dict[str, Any] = {"phase": EVOLVE, "actions": [], "deployed": 0}

        if self.evolution_pipeline is not None:
            try:
                cycle = self.evolution_pipeline.run_cycle(verbose=False)
                cycle_dict = cycle.to_dict() if hasattr(cycle, "to_dict") else cycle
                summary    = cycle_dict.get("summary", {})

                result["gaps_detected"]  = summary.get("gaps_detected", 0)
                result["generated"]      = summary.get("modules_generated", 0)
                result["approved"]       = summary.get("modules_approved", 0)
                result["deployed"]       = summary.get("modules_deployed", 0)

                self.status.modules_evolved += result["deployed"]
                result["actions"].append(
                    f"evolution_cycle:gaps={result['gaps_detected']}"
                    f",deployed={result['deployed']}"
                )
            except Exception as e:
                result["evolve_error"] = str(e)

        self.status.total_evolutions += 1
        return result

    # ── Main lifecycle cycle ──────────────────────────────────────────────

    def run_one_cycle(self, verbose: bool = True, fast_mode: bool = True) -> Dict[str, Any]:
        """
        Run one complete lifecycle cycle through all 6 phases.

        In fast_mode, phase durations are compressed for demo/testing.
        Returns a dict summarising the full cycle.
        """
        if fast_mode:
            self.clock._durations = FAST_PHASE_DURATIONS.copy()

        self._lifecycle_count += 1
        cycle_id = f"being_cycle_{self._lifecycle_count}_{int(time.time())}"
        cycle_started = datetime.now(timezone.utc).isoformat()

        if verbose:
            logger.info(f"\n{'='*60}")
            logger.info(f"[Phase 14] 🌱 LIFECYCLE CYCLE #{self._lifecycle_count} STARTING")
            logger.info(f"{'='*60}")

        phase_results: Dict[str, Any] = {}
        phase_handlers = {
            LIVE:       self._do_live,
            EXPERIENCE: self._do_experience,
            LEARN:      self._do_learn,
            SLEEP:      self._do_sleep,
            DREAM:      self._do_dream,
            EVOLVE:     self._do_evolve,
        }

        for phase in LIFECYCLE_ORDER:
            self.clock.set_phase(phase)
            self.status.phase = phase
            start_t = time.time()

            if verbose:
                logger.info(f"[Phase 14] ⟶  Phase: {phase}")

            try:
                handler = phase_handlers[phase]
                phase_result = handler()
                elapsed = time.time() - start_t
                phase_result["elapsed_s"] = round(elapsed, 3)
                phase_results[phase] = phase_result

                if verbose:
                    actions = phase_result.get("actions", [])
                    logger.info(
                        f"[Phase 14]    ✓ {phase} completed in {elapsed:.1f}s "
                        f"— {', '.join(actions) if actions else 'idle'}"
                    )

            except Exception as e:
                elapsed = time.time() - start_t
                phase_results[phase] = {
                    "phase": phase, "error": str(e), "elapsed_s": round(elapsed, 3)
                }
                logger.warning(f"[Phase 14] {phase} error: {e}")

        # Update vitality
        vitality = self.status.compute_vitality()
        self.status.cycle_count += 1

        cycle_summary = {
            "cycle_id":           cycle_id,
            "cycle_number":       self._lifecycle_count,
            "started_at":         cycle_started,
            "completed_at":       datetime.now(timezone.utc).isoformat(),
            "phases":             phase_results,
            "vitality":           vitality,
            "being_status":       self.status.to_dict(),
        }

        self._cycle_history.append(cycle_summary)

        if verbose:
            logger.info(f"[Phase 14] 🔁 Cycle #{self._lifecycle_count} complete — "
                        f"vitality={vitality:.3f}")
            logger.info(f"{'='*60}\n")

        return cycle_summary

    # ── Background thread ─────────────────────────────────────────────────

    def start_lifecycle(
        self,
        fast_mode: bool = False,
        verbose: bool = False,
        interval_s: float = 0.5,
    ):
        """Start the lifecycle loop in a background thread."""
        if self._running:
            return {"status": "already_running"}
        self._running = True

        def _loop():
            while self._running:
                try:
                    self.run_one_cycle(verbose=verbose, fast_mode=fast_mode)
                except Exception as e:
                    logger.warning(f"[Phase 14] Lifecycle loop error: {e}")
                time.sleep(interval_s)

        self._thread = threading.Thread(target=_loop, daemon=True, name="DigitalBeing-Lifecycle")
        self._thread.start()
        logger.info("[Phase 14] Lifecycle thread started")
        return {"status": "started", "fast_mode": fast_mode}

    def stop_lifecycle(self):
        """Stop the lifecycle loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
        logger.info("[Phase 14] Lifecycle stopped")
        return {"status": "stopped", "cycles_completed": self._lifecycle_count}

    # ── Introspection ─────────────────────────────────────────────────────

    def get_lifecycle_history(self, limit: int = 10) -> List[dict]:
        """Return recent lifecycle cycle summaries."""
        cycles = list(self._cycle_history)[-limit:]
        # Return lean summaries
        return [
            {
                "cycle_id":     c["cycle_id"],
                "cycle_number": c["cycle_number"],
                "started_at":   c["started_at"],
                "vitality":     c.get("vitality", 0.0),
                "phase_errors": {
                    k: v.get("error")
                    for k, v in c.get("phases", {}).items()
                    if v.get("error")
                },
            }
            for c in reversed(cycles)
        ]

    def get_being_narrative(self) -> str:
        """
        Return a human-readable narrative of the being's current state.
        "يعيش → يتجرب → يتعلم → ينام → يحلم → يتطور"
        """
        s = self.status
        phase_emoji = {
            LIVE: "🌐", EXPERIENCE: "🔍", LEARN: "📚",
            SLEEP: "💤", DREAM: "✨", EVOLVE: "🧬"
        }
        emoji = phase_emoji.get(s.phase, "?")

        return (
            f"{emoji} Digital Being — Phase 14\n"
            f"  Now:        {s.phase}\n"
            f"  Vitality:   {s.vitality:.1%}\n"
            f"  Cycle:      #{s.cycle_count}\n"
            f"  Experiences:{s.total_experiences:,}\n"
            f"  Learning:   {s.total_learning_steps:,} steps\n"
            f"  Redesigns:  {s.redesigns_accepted} accepted / {s.total_dreams} dreams\n"
            f"  Evolved:    {s.modules_evolved} modules\n"
            f"  Born:       {s.born_at}\n"
            f"\n"
            f"  يعيش → يتجرب → يتعلم → ينام → يحلم → يتطور"
        )

    def summary(self) -> dict:
        return {
            "lifecycle_cycles":      self._lifecycle_count,
            "is_running":            self._running,
            "current_phase":         self.status.phase,
            "vitality":              round(self.status.vitality, 4),
            "total_experiences":     self.status.total_experiences,
            "total_learning_steps":  self.status.total_learning_steps,
            "total_sleep_cycles":    self.status.total_sleep_cycles,
            "total_dreams":          self.status.total_dreams,
            "total_evolutions":      self.status.total_evolutions,
            "redesigns_accepted":    self.status.redesigns_accepted,
            "modules_evolved":       self.status.modules_evolved,
            "clock":                 self.clock.summary(),
            "health": {
                "routing_accuracy":   round(self.status.routing_accuracy, 4),
                "confidence":         round(self.status.confidence_level, 4),
                "memory":             round(self.status.memory_health, 4),
                "curiosity":          round(self.status.curiosity_level, 4),
                "structural_fitness": round(self.status.structural_fitness, 4),
            },
        }
