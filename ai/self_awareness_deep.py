"""
Phase 12 — Deep Self-Awareness & Confidence Engine
====================================================
The system's inner eye — knows what it knows, knows what it doesn't.

Three capabilities:

  1. ConfidenceEstimator
     — Before every routing decision, estimates how confident it is
     — Tracks calibration: does 80% confidence actually mean 80% correct?
     — Detects regions of uncertainty and flags them

  2. WeaknessDetector
     — Continuously scans the network's performance landscape
     — Identifies specific feature-space regions where it fails
     — Generates targeted training signals to fix weaknesses

  3. MetaCognitionEngine
     — "Thinking about thinking"
     — Monitors its own learning rate and improvement trajectory
     — Decides when to explore vs exploit
     — Generates self-improvement goals and reports its own state

Together they form the system's "prefrontal cortex" — executive control
over its own learning and decision-making.
"""
from __future__ import annotations

import logging
import math
import random
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CALIBRATION_WINDOW    = 200   # decisions used for calibration check
WEAKNESS_SCAN_BINS    = 8     # bins per feature dimension for weakness map
WEAKNESS_SCAN_DIMS    = 7     # feature vector dimensions
MIN_WEAKNESS_SAMPLES  = 5     # min samples needed to declare a weakness
WEAKNESS_THRESHOLD    = 0.35  # avg target below this = weak region
META_WINDOW           = 100   # steps for metacognition tracking


# ─────────────────────────────────────────────────────────────────────────────
#  Confidence Estimator
# ─────────────────────────────────────────────────────────────────────────────

class ConfidenceEstimator:
    """
    Phase 12: Estimates prediction confidence and tracks calibration.

    Calibration = when the model says "I'm 80% confident",
                  it's actually correct 80% of the time.

    Uses:
    - Variance of outputs across the deep network layers as uncertainty proxy
    - Historical accuracy in similar feature regions
    - Surprise history (high surprise → low confidence)
    """

    def __init__(self):
        # Calibration bins: confidence 0-1 in 10 buckets
        self._cal_bins: List[List[float]] = [[] for _ in range(10)]
        self._decision_history: deque = deque(maxlen=CALIBRATION_WINDOW)
        self._total_decisions  = 0
        self._surprises: deque = deque(maxlen=100)

    def estimate(
        self,
        feature_vec: List[float],
        predicted_target: float,
        deep_network=None,
    ) -> Tuple[float, dict]:
        """
        Estimate confidence for a prediction.

        Returns
        -------
        confidence : float in [0, 1]
        details    : dict with breakdown
        """
        confidence_signals = []

        # Signal 1: Distance from extremes (0.5 = max uncertainty)
        target_certainty = abs(predicted_target - 0.5) * 2  # 0→1
        confidence_signals.append(("target_certainty", target_certainty, 0.3))

        # Signal 2: Feature vector quality (are features well-defined?)
        vec = np.array(feature_vec)
        # Avoid features clustered near 0.5 (high uncertainty zone)
        feature_clarity = float(np.mean(np.abs(vec - 0.5)) * 2)
        confidence_signals.append(("feature_clarity", feature_clarity, 0.25))

        # Signal 3: Recent surprise rate (high surprise = low confidence)
        if self._surprises:
            avg_surprise   = sum(self._surprises) / len(self._surprises)
            surprise_conf  = max(0.0, 1.0 - avg_surprise * 2)
        else:
            surprise_conf  = 0.7  # neutral default
        confidence_signals.append(("surprise_history", surprise_conf, 0.25))

        # Signal 4: Deep network output spread (if available)
        if deep_network is not None:
            try:
                out = deep_network.forward(feature_vec)
                spread = float(np.std(out))  # low spread = high confidence
                net_conf = max(0.0, 1.0 - spread * 5)
                confidence_signals.append(("network_spread", net_conf, 0.2))
            except Exception:
                confidence_signals.append(("network_spread", 0.5, 0.2))
        else:
            confidence_signals.append(("network_spread", 0.6, 0.2))

        # Weighted combination
        total_w    = sum(w for _, _, w in confidence_signals)
        confidence = sum(v * w for _, v, w in confidence_signals) / total_w
        confidence = float(np.clip(confidence, 0.0, 1.0))

        self._total_decisions += 1

        return confidence, {
            "confidence":  round(confidence, 4),
            "signals":     {n: round(v, 4) for n, v, _ in confidence_signals},
            "level":       self._confidence_level(confidence),
        }

    def record_outcome(self, confidence: float, actual_target: float,
                       predicted_target: float) -> None:
        """Record outcome for calibration tracking."""
        surprise = abs(actual_target - predicted_target)
        self._surprises.append(surprise)

        # Calibration bin
        bin_idx = min(int(confidence * 10), 9)
        correct = 1.0 if surprise < 0.2 else 0.0
        self._cal_bins[bin_idx].append(correct)
        # Keep bins manageable
        if len(self._cal_bins[bin_idx]) > 50:
            self._cal_bins[bin_idx] = self._cal_bins[bin_idx][-50:]

        self._decision_history.append({
            "confidence":  confidence,
            "surprise":    surprise,
            "correct":     correct,
        })

    def calibration_report(self) -> dict:
        """
        How well-calibrated is the confidence estimator?
        Perfect calibration: 80% confidence bucket has 80% accuracy.
        """
        report = {}
        for i, bucket in enumerate(self._cal_bins):
            if not bucket:
                continue
            conf_range = f"{i*10}-{(i+1)*10}%"
            accuracy   = sum(bucket) / len(bucket)
            expected   = (i + 0.5) / 10
            calibration_error = abs(accuracy - expected)
            report[conf_range] = {
                "samples":           len(bucket),
                "accuracy":          round(accuracy, 3),
                "expected":          round(expected, 3),
                "calibration_error": round(calibration_error, 3),
            }
        return report

    def _confidence_level(self, c: float) -> str:
        if c >= 0.85: return "very_high"
        if c >= 0.70: return "high"
        if c >= 0.55: return "moderate"
        if c >= 0.40: return "low"
        return "very_low"

    def summary(self) -> dict:
        avg_surprise = (sum(self._surprises) / len(self._surprises)
                        if self._surprises else 0.0)
        return {
            "total_decisions":  self._total_decisions,
            "avg_surprise":     round(avg_surprise, 4),
            "calibration":      self.calibration_report(),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Weakness Detector
# ─────────────────────────────────────────────────────────────────────────────

class WeaknessDetector:
    """
    Phase 12: Scans the feature space for regions where the network fails.

    Maintains a coarse 7-dimensional grid (8 bins per dimension).
    Each cell tracks avg_target and sample count.

    A cell is "weak" if:
    - Has >= MIN_WEAKNESS_SAMPLES samples
    - avg_target < WEAKNESS_THRESHOLD

    Weak cells generate targeted training signals to fix them.
    """

    def __init__(self):
        # grid: cell_key → [sum_target, count, sum_loss]
        self._grid: Dict[Tuple, List[float]] = {}
        self._total_observations = 0
        self._weakness_history: deque = deque(maxlen=50)

    def _cell_key(self, vec: List[float]) -> Tuple:
        """Discretise feature vector to grid cell."""
        return tuple(min(int(v * WEAKNESS_SCAN_BINS), WEAKNESS_SCAN_BINS - 1)
                     for v in vec)

    def observe(self, feature_vec: List[float], target: float,
                loss: float = 0.0) -> None:
        """Record one observation into the weakness grid."""
        key = self._cell_key(feature_vec)
        if key not in self._grid:
            self._grid[key] = [0.0, 0, 0.0]
        self._grid[key][0] += target
        self._grid[key][1] += 1
        self._grid[key][2] += loss
        self._total_observations += 1

    def scan(self) -> List[dict]:
        """
        Identify all weak regions in the feature space.
        Returns list of weakness descriptors sorted by severity.
        """
        weaknesses = []
        for key, (sum_t, count, sum_loss) in self._grid.items():
            if count < MIN_WEAKNESS_SAMPLES:
                continue
            avg_target = sum_t / count
            avg_loss   = sum_loss / count
            if avg_target < WEAKNESS_THRESHOLD:
                # Reconstruct prototype vector for this cell
                proto_vec = [
                    (k + 0.5) / WEAKNESS_SCAN_BINS for k in key
                ]
                weaknesses.append({
                    "cell":         list(key),
                    "prototype_vec": [round(v, 3) for v in proto_vec],
                    "avg_target":   round(avg_target, 4),
                    "avg_loss":     round(avg_loss, 4),
                    "sample_count": count,
                    "severity":     round(WEAKNESS_THRESHOLD - avg_target, 4),
                })

        weaknesses.sort(key=lambda x: x["severity"], reverse=True)
        if weaknesses:
            self._weakness_history.append({
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "count":         len(weaknesses),
                "worst_severity": weaknesses[0]["severity"] if weaknesses else 0,
            })
        return weaknesses

    def generate_remedial_signals(
        self, n: int = 10
    ) -> List[Tuple[List[float], float]]:
        """
        Generate (feature_vec, target) pairs targeting weak regions.
        Used by SignalBus to produce focused training on failures.
        """
        weaknesses = self.scan()
        if not weaknesses:
            return []

        signals = []
        # Weight by severity — worse weaknesses get more signals
        weights = [w["severity"] for w in weaknesses]
        total_w = sum(weights) or 1.0

        for _ in range(n):
            # Pick a weak region proportionally to severity
            r = random.random() * total_w
            cumulative = 0.0
            chosen = weaknesses[0]
            for w in weaknesses:
                cumulative += w["severity"]
                if r <= cumulative:
                    chosen = w
                    break

            # Perturb the prototype vector slightly
            proto = chosen["prototype_vec"]
            noisy = [max(0.0, min(1.0, v + random.gauss(0, 0.08)))
                     for v in proto]

            # Target: pull the weak region toward success
            target = min(0.75, chosen["avg_target"] + 0.3)
            signals.append((noisy, target))

        return signals

    def coverage_pct(self) -> float:
        """What fraction of the feature space has been observed?"""
        total_cells = WEAKNESS_SCAN_BINS ** WEAKNESS_SCAN_DIMS
        return round(len(self._grid) / total_cells * 100, 3)

    def summary(self) -> dict:
        weaknesses = self.scan()
        return {
            "total_observations": self._total_observations,
            "cells_observed":     len(self._grid),
            "coverage_pct":       self.coverage_pct(),
            "weak_regions":       len(weaknesses),
            "worst_weakness":     weaknesses[0] if weaknesses else None,
            "top_3_weaknesses":   weaknesses[:3],
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Meta-Cognition Engine
# ─────────────────────────────────────────────────────────────────────────────

class MetaCognitionEngine:
    """
    Phase 12: "Thinking about thinking."

    The executive layer that:
    1. Monitors learning progress (is the network actually improving?)
    2. Detects learning stagnation (stuck in local minima?)
    3. Controls exploration rate (when to try new things vs exploit known good)
    4. Generates self-improvement goals
    5. Produces periodic self-assessment reports

    This is the highest-level control loop in the system.
    """

    def __init__(
        self,
        confidence_estimator: ConfidenceEstimator,
        weakness_detector: WeaknessDetector,
    ):
        self._confidence   = confidence_estimator
        self._weakness     = weakness_detector

        # Learning progress tracking
        self._loss_log:    deque = deque(maxlen=META_WINDOW * 3)
        self._reward_log:  deque = deque(maxlen=META_WINDOW)
        self._step_count   = 0
        self._last_assessment_ts = 0.0

        # Exploration control
        self._explore_rate = 0.3    # start with 30% exploration
        self._min_explore  = 0.05
        self._max_explore  = 0.6

        # Self-improvement goals
        self._active_goals: List[dict] = []
        self._completed_goals: List[dict] = []

        # Assessment history
        self._assessments: deque = deque(maxlen=20)

        logger.info("MetaCognitionEngine (Phase 12) initialised")

    # ── Learning monitoring ───────────────────────────────────────────────

    def record_step(self, loss: float, reward: float = 0.0) -> None:
        """Record one training step."""
        self._loss_log.append(loss)
        self._reward_log.append(reward)
        self._step_count += 1

        # Adjust exploration rate based on learning progress
        self._adjust_exploration()

    def _learning_velocity(self) -> float:
        """
        How fast is the network improving?
        Positive = improving, negative = degrading, 0 = stable.
        """
        log = list(self._loss_log)
        if len(log) < META_WINDOW:
            return 0.0
        first  = sum(log[:META_WINDOW//2]) / (META_WINDOW//2)
        second = sum(log[META_WINDOW//2:META_WINDOW]) / (META_WINDOW//2)
        if first <= 0:
            return 0.0
        return (first - second) / first   # positive = loss decreasing = good

    def _is_stagnating(self) -> bool:
        """Detect learning stagnation (velocity near 0)."""
        v = self._learning_velocity()
        return abs(v) < 0.005 and len(self._loss_log) >= META_WINDOW

    def _adjust_exploration(self) -> None:
        """
        Adaptive exploration: explore more when stagnating,
        exploit more when learning is progressing well.
        """
        v = self._learning_velocity()
        if self._is_stagnating():
            # Stagnating → increase exploration
            self._explore_rate = min(
                self._max_explore,
                self._explore_rate * 1.02
            )
        elif v > 0.05:
            # Learning well → reduce exploration (exploit)
            self._explore_rate = max(
                self._min_explore,
                self._explore_rate * 0.98
            )

    def should_explore(self) -> bool:
        """Should the next decision be exploratory?"""
        return random.random() < self._explore_rate

    # ── Goal generation ───────────────────────────────────────────────────

    def generate_goals(self) -> List[dict]:
        """
        Generate self-improvement goals based on current state.
        Goals are concrete targets the system will work toward.
        """
        new_goals = []
        weaknesses = self._weakness.scan()

        # Goal: fix top weakness
        if weaknesses:
            worst = weaknesses[0]
            goal = {
                "id":        f"goal_fix_weakness_{int(time.time())}",
                "type":      "fix_weakness",
                "target_region": worst["prototype_vec"],
                "current_performance": worst["avg_target"],
                "target_performance":  min(0.70, worst["avg_target"] + 0.25),
                "priority":  "high" if worst["severity"] > 0.2 else "medium",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status":    "active",
            }
            new_goals.append(goal)

        # Goal: improve calibration if poorly calibrated
        cal = self._confidence.calibration_report()
        max_cal_error = 0.0
        for bucket_data in cal.values():
            max_cal_error = max(max_cal_error, bucket_data.get("calibration_error", 0))
        if max_cal_error > 0.25:
            goal = {
                "id":       f"goal_calibration_{int(time.time())}",
                "type":     "improve_calibration",
                "current_error": round(max_cal_error, 4),
                "target_error":  0.15,
                "priority": "medium",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status":   "active",
            }
            new_goals.append(goal)

        # Goal: increase exploration if stagnating
        if self._is_stagnating():
            goal = {
                "id":       f"goal_explore_{int(time.time())}",
                "type":     "break_stagnation",
                "current_velocity": round(self._learning_velocity(), 6),
                "target_velocity":  0.01,
                "priority": "high",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "status":   "active",
            }
            new_goals.append(goal)

        # Add new goals (deduplicate by type)
        existing_types = {g["type"] for g in self._active_goals}
        for g in new_goals:
            if g["type"] not in existing_types:
                self._active_goals.append(g)

        return new_goals

    def update_goals(self, current_metrics: dict) -> None:
        """Check if any active goals have been achieved."""
        still_active = []
        for goal in self._active_goals:
            achieved = False

            if goal["type"] == "fix_weakness":
                # Check if the target region improved
                weaknesses = self._weakness.scan()
                weak_vecs = [w["prototype_vec"] for w in weaknesses]
                region    = goal["target_region"]
                still_weak = any(
                    sum(abs(a-b) for a,b in zip(region, w)) < 0.3
                    for w in weak_vecs
                )
                achieved = not still_weak

            elif goal["type"] == "improve_calibration":
                cal = self._confidence.calibration_report()
                if cal:
                    max_err = max(
                        v.get("calibration_error", 0) for v in cal.values()
                    )
                    achieved = max_err <= goal.get("target_error", 0.15)

            elif goal["type"] == "break_stagnation":
                achieved = self._learning_velocity() > 0.01

            if achieved:
                goal["status"]       = "completed"
                goal["completed_at"] = datetime.now(timezone.utc).isoformat()
                self._completed_goals.append(goal)
                logger.info(f"MetaCognition: GOAL ACHIEVED → {goal['type']}")
            else:
                still_active.append(goal)

        self._active_goals = still_active

    # ── Self-assessment ───────────────────────────────────────────────────

    def assess(self, signal_bus_stats: Optional[dict] = None,
               memory_summary: Optional[dict] = None) -> dict:
        """
        Full self-assessment report.
        Called periodically — the system's mirror.
        """
        velocity    = self._learning_velocity()
        stagnating  = self._is_stagnating()
        weaknesses  = self._weakness.scan()

        avg_loss    = (sum(list(self._loss_log)[-50:]) / 50
                       if len(self._loss_log) >= 50 else
                       sum(self._loss_log) / len(self._loss_log)
                       if self._loss_log else 0.0)
        avg_reward  = (sum(self._reward_log) / len(self._reward_log)
                       if self._reward_log else 0.0)

        # Generate new goals based on current state
        new_goals = self.generate_goals()

        # Insight generation
        insights = []
        if stagnating:
            insights.append("⚠ Learning stagnation detected — increasing exploration")
        if velocity > 0.05:
            insights.append("✓ Learning velocity is strong — network is improving")
        if weaknesses:
            insights.append(
                f"⚠ {len(weaknesses)} weak feature regions detected — "
                f"worst severity: {weaknesses[0]['severity']:.3f}"
            )
        if self._explore_rate > 0.4:
            insights.append("→ High exploration mode active")
        elif self._explore_rate < 0.1:
            insights.append("→ High exploitation mode active")
        if avg_reward > 0.3:
            insights.append("✓ Positive reward signal — decisions are improving")

        assessment = {
            "timestamp":            datetime.now(timezone.utc).isoformat(),
            "step_count":           self._step_count,
            "learning_velocity":    round(velocity, 6),
            "is_stagnating":        stagnating,
            "avg_loss_recent":      round(avg_loss, 6),
            "avg_reward_recent":    round(avg_reward, 4),
            "explore_rate":         round(self._explore_rate, 4),
            "weak_regions":         len(weaknesses),
            "active_goals":         len(self._active_goals),
            "completed_goals":      len(self._completed_goals),
            "new_goals_generated":  len(new_goals),
            "insights":             insights,
            "confidence_summary":   self._confidence.summary(),
            "weakness_summary":     self._weakness.summary(),
        }

        if signal_bus_stats:
            assessment["signal_stream"] = {
                "total_signals":     signal_bus_stats.get("total_signals", 0),
                "signal_rate":       signal_bus_stats.get("signal_rate_per_min", 0),
                "loss_trend":        signal_bus_stats.get("loss_trend", "unknown"),
            }

        if memory_summary:
            assessment["memory"] = {
                "episodic_size":  memory_summary.get("episodic_store_size", 0),
                "semantic_rules": memory_summary.get("semantic_rules", 0),
                "dreams":         memory_summary.get("dream_cycles", 0),
            }

        self._assessments.append(assessment)
        self._last_assessment_ts = time.time()
        return assessment

    def summary(self) -> dict:
        return {
            "step_count":          self._step_count,
            "learning_velocity":   round(self._learning_velocity(), 6),
            "is_stagnating":       self._is_stagnating(),
            "explore_rate":        round(self._explore_rate, 4),
            "active_goals":        self._active_goals,
            "completed_goals_count": len(self._completed_goals),
            "last_assessment":     self._assessments[-1] if self._assessments else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Deep Self-Awareness Engine — integrates all three Phase 12 components
# ─────────────────────────────────────────────────────────────────────────────

class DeepSelfAwareness:
    """
    Phase 12: Unified self-awareness system.

    Integrates ConfidenceEstimator + WeaknessDetector + MetaCognitionEngine
    into a single API used by the SignalBus and main mesh.

    The system can now:
    - Know when it's confident vs uncertain
    - Know where it's weak and generate fixes
    - Monitor its own learning and generate goals
    - Produce human-readable self-assessment reports
    """

    def __init__(self):
        self.confidence  = ConfidenceEstimator()
        self.weakness    = WeaknessDetector()
        self.metacog     = MetaCognitionEngine(self.confidence, self.weakness)
        self._step_count = 0
        logger.info("DeepSelfAwareness (Phase 12) initialised — all 3 modules active")

    def before_decision(
        self,
        feature_vec: List[float],
        predicted_target: float,
        deep_network=None,
    ) -> dict:
        """
        Called BEFORE a routing decision.
        Returns confidence estimate and whether to explore.
        """
        conf, conf_details = self.confidence.estimate(
            feature_vec, predicted_target, deep_network
        )
        explore = self.metacog.should_explore()
        return {
            "confidence":     conf,
            "conf_details":   conf_details,
            "should_explore": explore,
            "explore_rate":   round(self.metacog._explore_rate, 4),
        }

    def after_decision(
        self,
        feature_vec: List[float],
        predicted_target: float,
        actual_target: float,
        confidence: float,
        loss: float = 0.0,
        reward: float = 0.0,
    ) -> None:
        """
        Called AFTER a routing decision with the actual outcome.
        Updates all internal models.
        """
        self.confidence.record_outcome(confidence, actual_target, predicted_target)
        self.weakness.observe(feature_vec, actual_target, loss)
        self.metacog.record_step(loss, reward)
        self._step_count += 1

        # Periodic goal updates
        if self._step_count % 50 == 0:
            self.metacog.update_goals({})

    def get_remedial_signals(self, n: int = 5) -> List[Tuple[List[float], float]]:
        """Get targeted training signals for weak regions."""
        return self.weakness.generate_remedial_signals(n)

    def self_assess(
        self,
        signal_bus_stats: Optional[dict] = None,
        memory_summary: Optional[dict] = None,
    ) -> dict:
        """Full self-assessment — the system's mirror."""
        return self.metacog.assess(signal_bus_stats, memory_summary)

    def summary(self) -> dict:
        return {
            "step_count":   self._step_count,
            "confidence":   self.confidence.summary(),
            "weakness":     self.weakness.summary(),
            "metacognition": self.metacog.summary(),
        }
