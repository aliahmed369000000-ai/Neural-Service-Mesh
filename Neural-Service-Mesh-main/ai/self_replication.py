"""
Phase 15 – Self-Replication Engine
Intelligent self-improvement: analyse current performance, generate a
mutated child configuration, validate it in an isolated sandbox, then
promote or rollback based on measured outcomes.

This is *not* blind copying — the child is a parameter-mutated variant
that must outperform the parent on three axes before it is promoted:
  • Success Rate  (higher is better)
  • Latency       (lower is better)
  • Resource Usage (lower is better)
"""
from __future__ import annotations

import copy
import hashlib
import logging
import random
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum improvement on the composite fitness score to promote a child
PROMOTION_THRESHOLD = 0.02   # 2 % relative improvement

# Maximum mutations applied per generation
MAX_MUTATIONS = 5

# Sandbox run count (simulated executions to estimate child performance)
SANDBOX_TRIALS = 50

# Mutation magnitude as a fraction of the current parameter value
MUTATION_SCALE = 0.15        # ±15 %


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fitness(metrics: dict) -> float:
    """
    Composite fitness score in [0, 1].
    Higher is better.

    Formula:
        fitness = 0.5 * success_rate
                + 0.3 * (1 - latency_norm)
                + 0.2 * (1 - resource_norm)

    Where latency_norm and resource_norm are clamped to [0, 1] relative
    to a reference ceiling (5 000 ms latency, 100 % resource usage).
    """
    sr  = float(metrics.get("success_rate", 0.5))
    lat = float(metrics.get("latency_ms",   500.0))
    res = float(metrics.get("resource_pct", 50.0))

    lat_norm = min(1.0, lat  / 5_000.0)
    res_norm = min(1.0, res  / 100.0)

    return round(0.5 * sr + 0.3 * (1.0 - lat_norm) + 0.2 * (1.0 - res_norm), 6)


# ---------------------------------------------------------------------------
# SelfReplicationEngine
# ---------------------------------------------------------------------------

class SelfReplicationEngine:
    """
    Manages an evolving population of one: the current (parent) configuration.

    Parameters
    ----------
    knowledge_store : optional
        Object with .get(key) / .set(key, value) used to persist the active
        configuration and historical performance records.
    """

    def __init__(self, knowledge_store=None):
        self._ks = knowledge_store

        # Current (parent) configuration – mutable numeric parameters
        self._parent_config: Dict[str, float] = {
            "learning_rate":       0.01,
            "decay_factor":        0.95,
            "confidence_threshold": 0.70,
            "exploration_rate":    0.10,
            "memory_retention":    0.85,
            "routing_weight":      1.00,
            "quality_cutoff":      60.0,
        }

        # Simulated parent performance (updated after each promote/rollback)
        self._parent_metrics: Dict[str, float] = {
            "success_rate":  0.75,
            "latency_ms":    320.0,
            "resource_pct":  55.0,
        }

        # Active children: child_id -> child dict
        self._children: Dict[str, dict] = {}

        # History of all replication attempts
        self._history: List[dict] = []

        self._generation: int = 0
        logger.info("SelfReplicationEngine initialised (generation 0)")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze_self(self) -> dict:
        """
        Measure the current (parent) configuration's performance.

        Returns a snapshot with:
            generation, config, metrics, fitness, timestamp
        """
        # Pull live metrics from knowledge_store if available
        if self._ks is not None:
            try:
                stored = self._ks.get("performance_metrics")
                if stored and isinstance(stored, dict):
                    self._parent_metrics.update(stored)
            except Exception:
                pass

        fitness = _fitness(self._parent_metrics)
        snapshot = {
            "generation":  self._generation,
            "config":      copy.deepcopy(self._parent_config),
            "metrics":     copy.deepcopy(self._parent_metrics),
            "fitness":     fitness,
            "timestamp":   _now_iso(),
        }
        logger.info("analyze_self → generation=%d fitness=%.4f", self._generation, fitness)
        return snapshot

    def generate_child(self) -> dict:
        """
        Create a mutated child configuration and run it through the sandbox.

        Returns a child descriptor dict with:
            child_id, parent_generation, config, sandbox_metrics,
            fitness, status ('pending'), timestamp
        """
        parent_analysis = self.analyze_self()
        child_config    = self._mutate(copy.deepcopy(self._parent_config))
        child_id        = f"child_{self._generation}_{uuid.uuid4().hex[:8]}"

        sandbox_metrics = self._run_sandbox(child_config, parent_analysis["metrics"])

        child = {
            "child_id":           child_id,
            "parent_generation":  self._generation,
            "parent_fitness":     parent_analysis["fitness"],
            "config":             child_config,
            "sandbox_metrics":    sandbox_metrics,
            "fitness":            _fitness(sandbox_metrics),
            "status":             "pending",
            "timestamp":          _now_iso(),
        }
        self._children[child_id] = child
        logger.info(
            "generate_child → %s  fitness=%.4f (parent=%.4f)",
            child_id, child["fitness"], child["parent_fitness"],
        )
        return child

    def promote_or_kill(self, child_id: str) -> str:
        """
        Evaluate the child against its parent and decide fate.

        Returns 'promoted' or 'rolled_back'.
        """
        if child_id not in self._children:
            raise KeyError(f"Unknown child_id: {child_id!r}")

        child          = self._children[child_id]
        parent_fitness = child["parent_fitness"]
        child_fitness  = child["fitness"]
        delta          = child_fitness - parent_fitness

        if delta >= PROMOTION_THRESHOLD * parent_fitness:
            # ---- PROMOTE ------------------------------------------------
            old_config  = copy.deepcopy(self._parent_config)
            old_metrics = copy.deepcopy(self._parent_metrics)

            self._parent_config  = child["config"]
            self._parent_metrics = child["sandbox_metrics"]
            self._generation    += 1

            child["status"] = "promoted"
            outcome = "promoted"

            event = {
                "event":        "promote",
                "child_id":     child_id,
                "generation":   self._generation,
                "delta_fitness": round(delta, 6),
                "old_fitness":  parent_fitness,
                "new_fitness":  child_fitness,
                "timestamp":    _now_iso(),
            }
            logger.info(
                "PROMOTE %s  generation %d→%d  fitness %.4f→%.4f",
                child_id, self._generation - 1, self._generation,
                parent_fitness, child_fitness,
            )
        else:
            # ---- ROLLBACK -----------------------------------------------
            child["status"] = "rolled_back"
            outcome = "rolled_back"

            event = {
                "event":        "rollback",
                "child_id":     child_id,
                "generation":   self._generation,
                "delta_fitness": round(delta, 6),
                "reason":       "insufficient_improvement",
                "timestamp":    _now_iso(),
            }
            logger.info(
                "ROLLBACK %s  delta_fitness=%.4f (threshold=%.4f)",
                child_id, delta, PROMOTION_THRESHOLD * parent_fitness,
            )

        self._history.append(event)

        # Persist to knowledge_store if available
        if self._ks is not None:
            try:
                self._ks.set("active_config",    self._parent_config)
                self._ks.set("active_generation", self._generation)
            except Exception:
                pass

        # Clean up processed child
        del self._children[child_id]
        return outcome

    def summary(self) -> dict:
        promotions  = sum(1 for e in self._history if e["event"] == "promote")
        rollbacks   = sum(1 for e in self._history if e["event"] == "rollback")
        return {
            "engine":            "SelfReplicationEngine",
            "current_generation": self._generation,
            "current_fitness":    _fitness(self._parent_metrics),
            "current_metrics":    copy.deepcopy(self._parent_metrics),
            "pending_children":   len(self._children),
            "total_promotions":   promotions,
            "total_rollbacks":    rollbacks,
            "history_length":     len(self._history),
            "mutation_scale":     MUTATION_SCALE,
            "promotion_threshold": PROMOTION_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _mutate(self, config: dict) -> dict:
        """Apply random parameter mutations (±MUTATION_SCALE) to a config copy."""
        keys     = list(config.keys())
        n_mutate = random.randint(1, min(MAX_MUTATIONS, len(keys)))
        targets  = random.sample(keys, n_mutate)

        for key in targets:
            original = config[key]
            factor   = 1.0 + random.uniform(-MUTATION_SCALE, MUTATION_SCALE)
            mutated  = original * factor

            # Clamp to sensible ranges per parameter
            if "rate" in key or "factor" in key or "threshold" in key or "retention" in key:
                mutated = max(0.001, min(1.0, mutated))
            elif "cutoff" in key:
                mutated = max(0.0, min(100.0, mutated))
            else:
                mutated = max(0.0, mutated)

            config[key] = round(mutated, 6)
            logger.debug("mutate %s: %.4f → %.4f", key, original, config[key])

        return config

    def _run_sandbox(self, child_config: dict, parent_metrics: dict) -> dict:
        """
        Simulate SANDBOX_TRIALS executions of the child configuration.

        The simulation is a stochastic model: each config parameter nudges
        the probability distribution of success/latency/resource.  In a real
        deployment this would execute actual tasks in an isolated environment.
        """
        rng = random.Random(int(hashlib.md5(str(child_config).encode()).hexdigest(), 16))

        # Derive base performance modifiers from config deltas
        lr_ratio   = child_config.get("learning_rate",      0.01) / 0.01
        cr_ratio   = child_config.get("confidence_threshold", 0.70) / 0.70
        mem_ratio  = child_config.get("memory_retention",   0.85) / 0.85

        successes  = 0
        latencies  = []
        resources  = []

        base_sr  = parent_metrics.get("success_rate", 0.75)
        base_lat = parent_metrics.get("latency_ms",   320.0)
        base_res = parent_metrics.get("resource_pct", 55.0)

        for _ in range(SANDBOX_TRIALS):
            # Success probability nudged by confidence and memory retention
            p_success = min(0.999, max(0.001,
                base_sr * (0.8 + 0.1 * cr_ratio + 0.1 * mem_ratio)
                + rng.gauss(0, 0.05)
            ))
            if rng.random() < p_success:
                successes += 1

            # Latency nudged by learning rate (higher lr → faster convergence)
            lat = max(1.0, base_lat / max(0.5, lr_ratio) + rng.gauss(0, base_lat * 0.1))
            latencies.append(lat)

            # Resource usage nudged inversely by learning rate
            res = max(1.0, min(100.0,
                base_res * (1.0 / max(0.5, lr_ratio)) + rng.gauss(0, base_res * 0.08)
            ))
            resources.append(res)

        return {
            "success_rate":  round(successes / SANDBOX_TRIALS, 4),
            "latency_ms":    round(sum(latencies) / len(latencies), 2),
            "resource_pct":  round(sum(resources)  / len(resources),  2),
            "sandbox_trials": SANDBOX_TRIALS,
        }


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = SelfReplicationEngine()

    print("=== SelfReplicationEngine Demo ===")
    print("\n--- Step 1: Analyse self ---")
    analysis = engine.analyze_self()
    print(f"  Generation : {analysis['generation']}")
    print(f"  Fitness    : {analysis['fitness']:.4f}")
    print(f"  Metrics    : {analysis['metrics']}")

    print("\n--- Step 2: Generate child ---")
    child = engine.generate_child()
    print(f"  Child ID   : {child['child_id']}")
    print(f"  Parent fit : {child['parent_fitness']:.4f}")
    print(f"  Child fit  : {child['fitness']:.4f}")
    print(f"  Sandbox    : {child['sandbox_metrics']}")

    print("\n--- Step 3: Promote or rollback ---")
    outcome = engine.promote_or_kill(child["child_id"])
    print(f"  Outcome    : {outcome.upper()}")

    print("\n--- Summary ---")
    for k, v in engine.summary().items():
        print(f"  {k}: {v}")

    print("\n--- Running 5 generations ---")
    for i in range(5):
        c = engine.generate_child()
        o = engine.promote_or_kill(c["child_id"])
        s = engine.summary()
        print(f"  Gen {s['current_generation']:2d}  fitness={s['current_fitness']:.4f}  outcome={o}")
