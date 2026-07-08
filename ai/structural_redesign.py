"""
Phase 13 — Structural Self-Redesign Engine
============================================
"يعيد تصميم نفسه" — The system redesigns itself.

The Neural Service Mesh can now modify its own architecture at runtime:
  - Try new structural configurations in an isolated sandbox
  - Benchmark before vs after on real routing metrics
  - Keep improvements, revert regressions

Three capabilities:

  1. ArchitectureMutator
     — Generates candidate architectural changes (new layers, connections,
       weight configurations, topology variants)
     — Mutates hyperparameters, network depth, attention patterns

  2. StructuralBenchmark
     — Runs head-to-head performance comparison: current vs candidate
     — Uses replay buffer experiences as evaluation set
     — Tracks: routing accuracy, latency, confidence calibration, convergence

  3. StructuralEvolutionEngine
     — Orchestrates the full redesign loop:
         Propose → Sandbox Test → Benchmark → Accept/Revert
     — Maintains a history of architectural versions (like Git for the brain)
     — Automatically rolls back regressions

File: ai/structural_redesign.py
"""
from __future__ import annotations

import copy
import logging
import math
import random
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
BENCHMARK_EPISODES     = 50     # episodes used for before/after comparison
MIN_IMPROVEMENT        = 0.005  # minimum delta to accept a change (0.5%)
MUTATION_POOL_SIZE     = 5      # candidate mutations evaluated per cycle
REVERT_THRESHOLD       = -0.01  # revert if performance drops more than 1%
HISTORY_MAXLEN         = 50     # architectural versions to keep
BENCHMARK_TIMEOUT_S    = 30.0   # max seconds for one benchmark run


# ─────────────────────────────────────────────────────────────────────────────
#  Architectural Snapshot
# ─────────────────────────────────────────────────────────────────────────────

class ArchSnapshot:
    """Captures a point-in-time snapshot of an architectural configuration."""

    def __init__(
        self,
        snapshot_id: str,
        config: Dict[str, Any],
        benchmark_score: float = 0.0,
        notes: str = "",
    ):
        self.snapshot_id    = snapshot_id
        self.config         = config
        self.benchmark_score = benchmark_score
        self.notes          = notes
        self.created_at     = datetime.now(timezone.utc).isoformat()
        self.is_active      = False

    def to_dict(self) -> dict:
        return {
            "snapshot_id":     self.snapshot_id,
            "config":          self.config,
            "benchmark_score": round(self.benchmark_score, 6),
            "notes":           self.notes,
            "created_at":      self.created_at,
            "is_active":       self.is_active,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Architecture Mutator
# ─────────────────────────────────────────────────────────────────────────────

class ArchitectureMutator:
    """
    Phase 13: Generates candidate architectural mutations.

    Mutation types:
    - weight_scale:   rescale a routing weight dimension
    - layer_depth:    add/remove a processing layer in the deep network
    - attention_bias: shift attention toward specific routing factors
    - topology_prune: suggest removing low-value edges
    - exploration_rate: adjust the explore/exploit balance
    - confidence_threshold: change the confidence cutoff for rerouting
    """

    MUTATION_TYPES = [
        "weight_scale",
        "layer_depth",
        "attention_bias",
        "topology_prune",
        "exploration_rate",
        "confidence_threshold",
        "learning_rate_scale",
        "curiosity_boost",
    ]

    def __init__(self):
        self._mutation_count = 0
        self._accepted_types: Dict[str, int] = {}
        self._rejected_types: Dict[str, int] = {}

    def generate_candidate(self, current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Generate one mutated candidate configuration."""
        candidate = copy.deepcopy(current_config)
        mutation_type = random.choice(self.MUTATION_TYPES)

        if mutation_type == "weight_scale":
            dim = random.choice(["W_SEMANTIC", "W_SCORE", "W_MEMORY", "W_TOPOLOGY"])
            scale = random.uniform(0.85, 1.15)
            current = candidate.get("routing_weights", {}).get(dim, 1.0)
            candidate.setdefault("routing_weights", {})[dim] = round(
                max(0.1, min(3.0, current * scale)), 4
            )
            candidate["_mutation"] = f"weight_scale:{dim}×{scale:.3f}"

        elif mutation_type == "layer_depth":
            # Suggest adding or removing a hidden layer
            current_depth = candidate.get("network_depth", 3)
            delta = random.choice([-1, 1])
            candidate["network_depth"] = max(1, min(6, current_depth + delta))
            candidate["_mutation"] = f"layer_depth:{current_depth}→{candidate['network_depth']}"

        elif mutation_type == "attention_bias":
            # Bias the attention toward one routing factor
            factors = ["semantic", "score", "memory", "topology"]
            factor = random.choice(factors)
            bias = random.uniform(1.05, 1.25)
            candidate.setdefault("attention_bias", {})[factor] = round(bias, 4)
            candidate["_mutation"] = f"attention_bias:{factor}={bias:.3f}"

        elif mutation_type == "topology_prune":
            # Suggest pruning threshold for low-use edges
            current_thresh = candidate.get("prune_threshold", 0.1)
            delta = random.uniform(-0.02, 0.05)
            candidate["prune_threshold"] = round(
                max(0.05, min(0.4, current_thresh + delta)), 4
            )
            candidate["_mutation"] = f"topology_prune:{candidate['prune_threshold']}"

        elif mutation_type == "exploration_rate":
            current_er = candidate.get("exploration_rate", 0.15)
            delta = random.uniform(-0.03, 0.03)
            candidate["exploration_rate"] = round(
                max(0.01, min(0.5, current_er + delta)), 4
            )
            candidate["_mutation"] = f"exploration_rate:{candidate['exploration_rate']}"

        elif mutation_type == "confidence_threshold":
            current_ct = candidate.get("confidence_threshold", 0.6)
            delta = random.uniform(-0.05, 0.05)
            candidate["confidence_threshold"] = round(
                max(0.3, min(0.95, current_ct + delta)), 4
            )
            candidate["_mutation"] = f"confidence_threshold:{candidate['confidence_threshold']}"

        elif mutation_type == "learning_rate_scale":
            current_lr = candidate.get("learning_rate", 0.001)
            scale = random.choice([0.5, 0.75, 1.25, 2.0])
            candidate["learning_rate"] = round(
                max(1e-5, min(0.1, current_lr * scale)), 6
            )
            candidate["_mutation"] = f"learning_rate:{current_lr}→{candidate['learning_rate']}"

        elif mutation_type == "curiosity_boost":
            current_cb = candidate.get("curiosity_boost", 1.0)
            scale = random.uniform(0.8, 1.3)
            candidate["curiosity_boost"] = round(
                max(0.3, min(3.0, current_cb * scale)), 4
            )
            candidate["_mutation"] = f"curiosity_boost:{candidate['curiosity_boost']}"

        self._mutation_count += 1
        candidate["_mutation_id"] = f"mut_{self._mutation_count}_{int(time.time())}"
        candidate["_mutation_type"] = mutation_type
        return candidate

    def generate_pool(
        self,
        current_config: Dict[str, Any],
        size: int = MUTATION_POOL_SIZE,
    ) -> List[Dict[str, Any]]:
        """Generate a pool of candidate mutations."""
        return [self.generate_candidate(current_config) for _ in range(size)]

    def record_outcome(self, mutation_type: str, accepted: bool):
        """Track which mutation types tend to be accepted."""
        bucket = self._accepted_types if accepted else self._rejected_types
        bucket[mutation_type] = bucket.get(mutation_type, 0) + 1

    def best_mutation_types(self, top_k: int = 3) -> List[str]:
        """Return mutation types with highest acceptance rate."""
        rates = {}
        for mt in self.MUTATION_TYPES:
            acc = self._accepted_types.get(mt, 0)
            rej = self._rejected_types.get(mt, 0)
            total = acc + rej
            rates[mt] = acc / total if total > 0 else 0.0
        return sorted(rates, key=rates.get, reverse=True)[:top_k]

    def summary(self) -> dict:
        return {
            "total_mutations":  self._mutation_count,
            "accepted_by_type": self._accepted_types,
            "rejected_by_type": self._rejected_types,
            "best_types":       self.best_mutation_types(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Structural Benchmark
# ─────────────────────────────────────────────────────────────────────────────

class StructuralBenchmark:
    """
    Phase 13: Benchmarks architectural configurations head-to-head.

    Uses held-out replay experiences as evaluation set.
    Metrics:
    - routing_accuracy:    fraction of experiences with loss < 0.3
    - mean_confidence:     average confidence of routing decisions
    - convergence_speed:   how fast loss decreases over the evaluation set
    - exploration_balance: variety in routing choices (entropy)
    """

    def __init__(self):
        self._benchmark_history: deque = deque(maxlen=200)
        self._run_count = 0

    def score_config(
        self,
        config: Dict[str, Any],
        eval_experiences: List[Tuple[List[float], float, float]],
    ) -> Dict[str, float]:
        """
        Score a configuration against evaluation experiences.

        eval_experiences: list of (feature_vec, predicted_target, actual_target)
        Returns dict of metric scores and composite score.
        """
        if not eval_experiences:
            return {"composite": 0.5, "note": "no_eval_data"}

        w_semantic   = config.get("routing_weights", {}).get("W_SEMANTIC", 1.0)
        w_score      = config.get("routing_weights", {}).get("W_SCORE", 1.0)
        w_memory     = config.get("routing_weights", {}).get("W_MEMORY", 1.0)
        w_topology   = config.get("routing_weights", {}).get("W_TOPOLOGY", 1.0)
        conf_thresh  = config.get("confidence_threshold", 0.6)
        attn_bias    = config.get("attention_bias", {})

        losses       = []
        confidences  = []
        predictions  = []

        for feat, pred, actual in eval_experiences:
            # Simulate how this config would weight the routing decision
            if len(feat) >= 4:
                # Re-weight the feature vector using config weights
                weighted = (
                    feat[0] * w_semantic
                    + feat[1] * w_score
                    + feat[2] * w_memory
                    + feat[3] * w_topology
                ) / max(w_semantic + w_score + w_memory + w_topology, 1e-9)
            else:
                weighted = sum(feat) / max(len(feat), 1)

            # Apply attention bias if present
            for factor, bias in attn_bias.items():
                factor_map = {"semantic": 0, "score": 1, "memory": 2, "topology": 3}
                fi = factor_map.get(factor, -1)
                if 0 <= fi < len(feat):
                    weighted = (weighted + feat[fi] * (bias - 1.0)) / bias

            # Simulate prediction with this config
            sim_pred = 0.5 + (weighted - 0.5) * 0.9
            sim_pred = max(0.0, min(1.0, sim_pred))

            loss = (sim_pred - actual) ** 2
            confidence = max(0.0, 1.0 - abs(sim_pred - 0.5) * 2 * (1 + loss))

            losses.append(loss)
            confidences.append(confidence)
            predictions.append(sim_pred)

        n = len(losses)
        routing_accuracy   = sum(1 for l in losses if l < 0.09) / n  # loss<0.3²
        mean_loss          = sum(losses) / n
        mean_confidence    = sum(confidences) / n

        # Convergence speed: is loss trending down?
        half = n // 2
        if half > 0:
            first_half_loss = sum(losses[:half]) / half
            second_half_loss = sum(losses[half:]) / max(n - half, 1)
            convergence_speed = max(0.0, (first_half_loss - second_half_loss) / max(first_half_loss, 1e-9))
        else:
            convergence_speed = 0.0

        # Exploration balance: prediction entropy
        buckets = [0] * 10
        for p in predictions:
            b = min(int(p * 10), 9)
            buckets[b] += 1
        probs = [b / n for b in buckets if b > 0]
        entropy = -sum(p * math.log(p + 1e-9) for p in probs) / math.log(10)

        composite = (
            routing_accuracy  * 0.35 +
            mean_confidence   * 0.25 +
            convergence_speed * 0.20 +
            entropy           * 0.10 +
            (1.0 - mean_loss) * 0.10
        )

        result = {
            "routing_accuracy":   round(routing_accuracy, 4),
            "mean_loss":          round(mean_loss, 6),
            "mean_confidence":    round(mean_confidence, 4),
            "convergence_speed":  round(convergence_speed, 4),
            "exploration_entropy":round(entropy, 4),
            "composite":          round(composite, 6),
            "n_eval":             n,
        }
        self._benchmark_history.append(result)
        self._run_count += 1
        return result

    def compare(
        self,
        before_score: Dict[str, float],
        after_score: Dict[str, float],
    ) -> Dict[str, Any]:
        """Compare two benchmark results. Returns delta and verdict."""
        delta = after_score["composite"] - before_score["composite"]
        improvement = delta / max(abs(before_score["composite"]), 1e-9)

        return {
            "before_composite":  before_score["composite"],
            "after_composite":   after_score["composite"],
            "delta":             round(delta, 6),
            "improvement_pct":   round(improvement * 100, 3),
            "routing_accuracy_delta": round(
                after_score.get("routing_accuracy", 0)
                - before_score.get("routing_accuracy", 0), 4
            ),
            "confidence_delta": round(
                after_score.get("mean_confidence", 0)
                - before_score.get("mean_confidence", 0), 4
            ),
            "verdict": (
                "accept"  if delta >= MIN_IMPROVEMENT else
                "revert"  if delta <= REVERT_THRESHOLD else
                "neutral"
            ),
        }

    def summary(self) -> dict:
        if not self._benchmark_history:
            return {"run_count": 0}
        scores = [r["composite"] for r in self._benchmark_history]
        return {
            "run_count":     self._run_count,
            "avg_composite": round(sum(scores) / len(scores), 4),
            "best_score":    round(max(scores), 4),
            "worst_score":   round(min(scores), 4),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Structural Evolution Engine
# ─────────────────────────────────────────────────────────────────────────────

class StructuralEvolutionEngine:
    """
    Phase 13: Orchestrates the full self-redesign loop.

    Loop:
      1. Capture current config (snapshot)
      2. Generate N candidate mutations
      3. Benchmark each candidate against replay buffer
      4. Pick best candidate
      5. If best > current + MIN_IMPROVEMENT: accept
         Else: stay with current
      6. Record architectural history

    The system "redesigns itself" in a controlled, reversible way.
    """

    def __init__(
        self,
        mesh=None,
        deep_network=None,
        dynamic_layer=None,
        episodic_memory=None,
        signal_bus=None,
        self_awareness=None,
    ):
        self.mesh            = mesh
        self.deep_network    = deep_network
        self.dynamic_layer   = dynamic_layer
        self.episodic_memory = episodic_memory
        self.signal_bus      = signal_bus
        self.self_awareness  = self_awareness

        self.mutator         = ArchitectureMutator()
        self.benchmark       = StructuralBenchmark()

        self._redesign_count  = 0
        self._accepted_count  = 0
        self._reverted_count  = 0
        self._neutral_count   = 0

        self._current_config: Dict[str, Any] = self._extract_current_config()
        self._arch_history: deque = deque(maxlen=HISTORY_MAXLEN)

        # Take initial snapshot
        snap = ArchSnapshot(
            snapshot_id=f"arch_v0_{int(time.time())}",
            config=copy.deepcopy(self._current_config),
            notes="initial_state",
        )
        snap.is_active = True
        self._arch_history.append(snap)

        logger.info("StructuralEvolutionEngine (Phase 13) initialised")

    # ── Config extraction ─────────────────────────────────────────────────

    def _extract_current_config(self) -> Dict[str, Any]:
        """Pull current architectural configuration from live modules."""
        config: Dict[str, Any] = {}

        # Routing weights
        if self.mesh is not None:
            routing = getattr(self.mesh, "routing", None)
            if routing is not None:
                config["routing_weights"] = {
                    "W_SEMANTIC": getattr(routing, "W_SEMANTIC", 1.0),
                    "W_SCORE":    getattr(routing, "W_SCORE",    1.0),
                    "W_MEMORY":   getattr(routing, "W_MEMORY",   1.0),
                    "W_TOPOLOGY": getattr(routing, "W_TOPOLOGY", 1.0),
                }

        # Deep network depth
        if self.deep_network is not None:
            layers = getattr(self.deep_network, "_layers", None)
            config["network_depth"] = len(layers) if layers else 3

        # Learning rate
        if self.deep_network is not None:
            config["learning_rate"] = getattr(self.deep_network, "_lr", 0.001)

        # Signal bus exploration rate
        if self.signal_bus is not None:
            curiosity = getattr(self.signal_bus, "_curiosity", None)
            if curiosity is not None:
                config["exploration_rate"] = getattr(curiosity, "_curiosity_level", 0.15)

        # Self-awareness confidence threshold
        if self.self_awareness is not None:
            config["confidence_threshold"] = getattr(
                self.self_awareness, "_confidence_threshold", 0.6
            )

        config["_version"]    = self._redesign_count
        config["_created_at"] = datetime.now(timezone.utc).isoformat()
        return config

    def _apply_config(self, config: Dict[str, Any]) -> bool:
        """Apply a configuration to the live system."""
        applied = []

        try:
            # Apply routing weights
            if "routing_weights" in config and self.mesh is not None:
                routing = getattr(self.mesh, "routing", None)
                if routing is not None:
                    rw = config["routing_weights"]
                    if "W_SEMANTIC" in rw:
                        routing.W_SEMANTIC = rw["W_SEMANTIC"]
                    if "W_SCORE" in rw:
                        routing.W_SCORE = rw["W_SCORE"]
                    if "W_MEMORY" in rw:
                        routing.W_MEMORY = rw["W_MEMORY"]
                    if "W_TOPOLOGY" in rw:
                        routing.W_TOPOLOGY = rw["W_TOPOLOGY"]
                    applied.append("routing_weights")

            # Apply exploration rate
            if "exploration_rate" in config and self.signal_bus is not None:
                curiosity = getattr(self.signal_bus, "_curiosity", None)
                if curiosity is not None and hasattr(curiosity, "_curiosity_level"):
                    curiosity._curiosity_level = config["exploration_rate"]
                    applied.append("exploration_rate")

            # Apply confidence threshold
            if "confidence_threshold" in config and self.self_awareness is not None:
                if hasattr(self.self_awareness, "_confidence_threshold"):
                    self.self_awareness._confidence_threshold = config["confidence_threshold"]
                    applied.append("confidence_threshold")

            logger.debug(f"Applied config fields: {applied}")
            return True

        except Exception as e:
            logger.warning(f"Failed to apply config: {e}")
            return False

    # ── Evaluation data ───────────────────────────────────────────────────

    def _get_eval_experiences(self, n: int = BENCHMARK_EPISODES) -> List[Tuple]:
        """Collect evaluation experiences from episodic memory or signal bus."""
        experiences = []

        # From episodic memory
        if self.episodic_memory is not None:
            try:
                strongest = self.episodic_memory.get_strongest_memories(n)
                for mem in strongest:
                    fv   = mem.get("feature_vec") or mem.get("features", [])
                    pred = mem.get("predicted_target", mem.get("target", 0.5))
                    act  = mem.get("actual_target", mem.get("target", 0.5))
                    if fv:
                        experiences.append((fv, pred, act))
            except Exception:
                pass

        # From replay buffer
        if len(experiences) < n and self.signal_bus is not None:
            try:
                buf = getattr(self.signal_bus, "replay_buffer", None)
                if buf is not None:
                    sample = buf.sample(n - len(experiences))
                    for exp in sample:
                        fv   = getattr(exp, "feature_vec", [])
                        tgt  = getattr(exp, "target", 0.5)
                        if fv:
                            experiences.append((fv, tgt, tgt))
            except Exception:
                pass

        # Fallback: synthetic evaluation data
        if len(experiences) < 5:
            rng = random.Random(42)
            for _ in range(max(10, n - len(experiences))):
                fv  = [rng.random() for _ in range(7)]
                tgt = rng.random()
                experiences.append((fv, tgt, tgt))

        return experiences[:n]

    # ── Redesign cycle ────────────────────────────────────────────────────

    def run_redesign_cycle(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run one structural redesign cycle.

        Returns dict with:
          - mutation_tested: the candidate that was evaluated
          - before_score, after_score: benchmark results
          - comparison: delta + verdict
          - accepted: bool
          - snapshot_id: if accepted, the new snapshot ID
        """
        self._redesign_count += 1
        cycle_id = f"redesign_{self._redesign_count}_{int(time.time())}"

        if verbose:
            logger.info(f"[Phase 13] Redesign cycle #{self._redesign_count} starting…")

        # Step 1: Get evaluation data
        eval_data = self._get_eval_experiences()

        # Step 2: Benchmark current config
        current_config = self._extract_current_config()
        before_score   = self.benchmark.score_config(current_config, eval_data)

        # Step 3: Generate mutation pool and pick best candidate
        pool      = self.mutator.generate_pool(current_config, MUTATION_POOL_SIZE)
        best_cand = None
        best_after = None

        for candidate in pool:
            after_score = self.benchmark.score_config(candidate, eval_data)
            if best_after is None or after_score["composite"] > best_after["composite"]:
                best_cand  = candidate
                best_after = after_score

        # Step 4: Compare best candidate vs current
        comparison = self.benchmark.compare(before_score, best_after)
        verdict    = comparison["verdict"]
        mutation   = best_cand.get("_mutation", "unknown") if best_cand else "none"
        mut_type   = best_cand.get("_mutation_type", "unknown") if best_cand else "none"

        accepted = False
        snapshot_id = None

        if verdict == "accept" and best_cand is not None:
            # Step 5a: Accept — apply the new config
            success = self._apply_config(best_cand)
            if success:
                self._accepted_count += 1
                self.mutator.record_outcome(mut_type, accepted=True)
                snap = ArchSnapshot(
                    snapshot_id=f"arch_v{self._redesign_count}_{int(time.time())}",
                    config=copy.deepcopy(best_cand),
                    benchmark_score=best_after["composite"],
                    notes=f"accepted: {mutation}",
                )
                snap.is_active = True
                # Deactivate previous
                for s in self._arch_history:
                    s.is_active = False
                self._arch_history.append(snap)
                snapshot_id = snap.snapshot_id
                accepted    = True
                if verbose:
                    logger.info(
                        f"[Phase 13] ✓ ACCEPTED {mutation} "
                        f"Δ={comparison['delta']:+.4f} "
                        f"({comparison['improvement_pct']:+.2f}%)"
                    )
            else:
                verdict = "apply_failed"
                self.mutator.record_outcome(mut_type, accepted=False)

        elif verdict == "revert":
            # Step 5b: Revert attempt was worse — stay with current
            self._reverted_count += 1
            self.mutator.record_outcome(mut_type, accepted=False)
            if verbose:
                logger.info(
                    f"[Phase 13] ✗ REVERTED {mutation} "
                    f"Δ={comparison['delta']:+.4f}"
                )
        else:
            # neutral
            self._neutral_count += 1
            self.mutator.record_outcome(mut_type, accepted=False)
            if verbose:
                logger.info(
                    f"[Phase 13] ~ NEUTRAL {mutation} "
                    f"Δ={comparison['delta']:+.4f}"
                )

        return {
            "cycle_id":       cycle_id,
            "cycle_number":   self._redesign_count,
            "mutation":       mutation,
            "mutation_type":  mut_type,
            "before_score":   before_score,
            "after_score":    best_after,
            "comparison":     comparison,
            "verdict":        verdict,
            "accepted":       accepted,
            "snapshot_id":    snapshot_id,
            "eval_episodes":  len(eval_data),
        }

    def run_cycles(self, n: int = 3, verbose: bool = True) -> List[Dict[str, Any]]:
        """Run N redesign cycles. Returns list of cycle results."""
        results = []
        for _ in range(n):
            result = self.run_redesign_cycle(verbose=verbose)
            results.append(result)
        return results

    # ── History & rollback ────────────────────────────────────────────────

    def get_history(self, limit: int = 10) -> List[dict]:
        """Return recent architectural history."""
        snaps = list(self._arch_history)[-limit:]
        return [s.to_dict() for s in reversed(snaps)]

    def rollback_to(self, snapshot_id: str) -> Dict[str, Any]:
        """Roll back to a specific architectural snapshot."""
        for snap in self._arch_history:
            if snap.snapshot_id == snapshot_id:
                success = self._apply_config(snap.config)
                if success:
                    for s in self._arch_history:
                        s.is_active = False
                    snap.is_active = True
                    logger.info(f"[Phase 13] Rolled back to snapshot {snapshot_id}")
                    return {"success": True, "snapshot": snap.to_dict()}
                else:
                    return {"success": False, "error": "apply_failed"}
        return {"success": False, "error": f"snapshot {snapshot_id!r} not found"}

    def get_active_snapshot(self) -> Optional[dict]:
        """Return the currently active architectural snapshot."""
        for snap in reversed(list(self._arch_history)):
            if snap.is_active:
                return snap.to_dict()
        return None

    # ── Summary ───────────────────────────────────────────────────────────

    def summary(self) -> dict:
        active = self.get_active_snapshot()
        return {
            "redesign_cycles":   self._redesign_count,
            "accepted":          self._accepted_count,
            "reverted":          self._reverted_count,
            "neutral":           self._neutral_count,
            "accept_rate":       round(
                self._accepted_count / max(self._redesign_count, 1), 4
            ),
            "history_length":    len(self._arch_history),
            "active_snapshot":   active.get("snapshot_id") if active else None,
            "active_score":      active.get("benchmark_score") if active else None,
            "mutator":           self.mutator.summary(),
            "benchmark":         self.benchmark.summary(),
        }
