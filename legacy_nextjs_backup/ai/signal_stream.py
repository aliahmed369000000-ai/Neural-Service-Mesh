"""
Phase 10 — Continuous Signal Stream Engine
===========================================
Transforms the Neural Service Mesh from a passive learner into an
**always-on living system** that generates, receives, and processes a
continuous flow of signals — even with zero external traffic.

Three signal sources feed the network 24/7:

  1. SelfStimulator   — generates synthetic experiences internally
                        (like a brain that thinks even in silence)

  2. CuriosityEngine  — actively seeks out unknown territory,
                        chooses the paths it understands least

  3. DreamConsolidator — during idle periods, replays past experiences
                         and extracts deeper patterns (sleep learning)

  4. ReplayBuffer     — stores thousands of past decisions for
                        batch re-training (experience replay)

  5. SignalBus        — central hub that routes all signals to the
                        neural layers and tracks signal statistics

Architecture
------------
  SignalBus
    ├── SelfStimulator      (synthetic signal generation)
    ├── CuriosityEngine     (exploration-driven signals)
    ├── DreamConsolidator   (idle-time replay + consolidation)
    └── ReplayBuffer        (experience storage + sampling)

All components write to the same ReplayBuffer.
SignalBus trains the DeepRoutingNetwork + DynamicWeightLayer
on every signal batch.
"""
from __future__ import annotations

import logging
import math
import random
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
REPLAY_BUFFER_SIZE   = 50_000   # max stored experiences
BATCH_SIZE           = 32       # experiences per training batch
SIGNAL_INTERVAL_S    = 2.0      # seconds between signal pulses
DREAM_IDLE_THRESHOLD = 30.0     # seconds idle before dream mode activates
CURIOSITY_DECAY      = 0.995    # how fast curiosity fades after exploring
MIN_CURIOSITY        = 0.05     # minimum curiosity floor
SELF_STIM_SCENARIOS  = 12       # number of synthetic scenario types


# ─────────────────────────────────────────────────────────────────────────────
#  Experience: the atomic unit of learning
# ─────────────────────────────────────────────────────────────────────────────

class Experience:
    """
    One learning experience — the atom of memory.

    Fields
    ------
    feature_vec  : 7-element normalised input vector
    target       : float in [0,1] — quality of the outcome
    source       : where this experience came from
    reward       : reinforcement signal (positive = good decision)
    timestamp    : UTC ISO string
    context      : arbitrary metadata dict
    """
    __slots__ = ("feature_vec", "target", "source", "reward",
                 "timestamp", "context", "weight")

    def __init__(
        self,
        feature_vec: List[float],
        target: float,
        source: str = "unknown",
        reward: float = 0.0,
        context: Optional[Dict] = None,
    ):
        self.feature_vec = feature_vec
        self.target      = float(np.clip(target, 0.0, 1.0))
        self.source      = source
        self.reward      = reward
        self.timestamp   = datetime.now(timezone.utc).isoformat()
        self.context     = context or {}
        self.weight      = 1.0   # priority weight for sampling

    def to_dict(self) -> dict:
        return {
            "feature_vec": self.feature_vec,
            "target":      self.target,
            "source":      self.source,
            "reward":      self.reward,
            "timestamp":   self.timestamp,
            "context":     self.context,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Replay Buffer — experience storage
# ─────────────────────────────────────────────────────────────────────────────

class ReplayBuffer:
    """
    Circular buffer of Experiences with prioritised sampling.

    Recent experiences and high-reward experiences are sampled more
    often than old / low-reward ones.
    """

    def __init__(self, maxlen: int = REPLAY_BUFFER_SIZE):
        self._buf: deque = deque(maxlen=maxlen)
        self._maxlen = maxlen
        self._total_added = 0
        self._source_counts: Dict[str, int] = {}
        self._lock = threading.Lock()

    def add(self, exp: Experience) -> None:
        with self._lock:
            self._buf.append(exp)
            self._total_added += 1
            self._source_counts[exp.source] = (
                self._source_counts.get(exp.source, 0) + 1
            )

    def sample(self, n: int = BATCH_SIZE) -> List[Experience]:
        """
        Sample n experiences with priority weighting.
        Recent + high-reward experiences get higher probability.
        """
        with self._lock:
            buf = list(self._buf)
        if not buf:
            return []
        n = min(n, len(buf))

        # Compute weights: recency (index/len) * reward_boost
        weights = []
        total = len(buf)
        for i, exp in enumerate(buf):
            recency = (i + 1) / total           # 0→1 (newer = higher)
            reward_boost = 1.0 + max(exp.reward, 0.0)
            weights.append(recency * reward_boost * exp.weight)

        # Normalise
        w_arr = np.array(weights, dtype=np.float64)
        w_arr /= w_arr.sum()

        indices = np.random.choice(len(buf), size=n, replace=False, p=w_arr)
        return [buf[i] for i in indices]

    def sample_by_source(self, source: str, n: int = 16) -> List[Experience]:
        """Sample only from a specific source (for targeted replay)."""
        with self._lock:
            filtered = [e for e in self._buf if e.source == source]
        if not filtered:
            return []
        return random.sample(filtered, min(n, len(filtered)))

    @property
    def size(self) -> int:
        return len(self._buf)

    def summary(self) -> dict:
        return {
            "size":          self.size,
            "capacity":      self._maxlen,
            "total_added":   self._total_added,
            "fill_pct":      round(self.size / self._maxlen * 100, 1),
            "source_counts": dict(self._source_counts),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Self-Stimulator — synthetic experience generator
# ─────────────────────────────────────────────────────────────────────────────

class SelfStimulator:
    """
    Phase 10: Generates synthetic experiences internally.

    Like a brain that runs simulations even in silence — it imagines
    scenarios and learns from them before they happen in reality.

    12 scenario archetypes cover the full problem space:
      0  perfect_route         — ideal conditions, high reward
      1  cascade_failure       — everything failing at once
      2  high_load_degradation — system under pressure
      3  recovery_sequence     — bouncing back from failure
      4  semantic_mismatch     — wrong node for the job
      5  latency_spike         — sudden slowdown
      6  memory_advantage      — known route outperforming
      7  topology_shortcut     — shorter path wins
      8  time_peak_stress      — peak hours behaviour
      9  drift_instability     — semantics shifting
      10 exploration_bonus     — reward for trying new paths
      11 steady_state          — normal operation baseline
    """

    SCENARIOS = [
        # name, base_target, base_reward, feature_fn
        ("perfect_route",        0.95, +1.0),
        ("cascade_failure",      0.05, -1.0),
        ("high_load_degradation",0.35, -0.3),
        ("recovery_sequence",    0.65, +0.5),
        ("semantic_mismatch",    0.20, -0.5),
        ("latency_spike",        0.40, -0.2),
        ("memory_advantage",     0.80, +0.6),
        ("topology_shortcut",    0.75, +0.4),
        ("time_peak_stress",     0.45, -0.1),
        ("drift_instability",    0.30, -0.4),
        ("exploration_bonus",    0.60, +0.7),
        ("steady_state",         0.70, +0.2),
    ]

    def __init__(self):
        self._scenario_counts = [0] * len(self.SCENARIOS)
        self._total_generated = 0

    def _scenario_vector(self, scenario_idx: int) -> List[float]:
        """Generate a realistic feature vector for a given scenario."""
        idx = scenario_idx
        noise = lambda s=0.05: random.gauss(0, s)

        if idx == 0:   # perfect_route
            return [0.9+noise(), 0.95+noise(), 0.88+noise(), 0.85+noise(),
                    0.92+noise(), 0.85+noise(), 0.80+noise()]
        elif idx == 1: # cascade_failure
            return [0.1+noise(), 0.05+noise(), 0.08+noise(), 0.12+noise(),
                    0.09+noise(), 0.01+noise(), 0.01+noise()]
        elif idx == 2: # high_load_degradation
            return [0.6+noise(), 0.5+noise(), 0.55+noise(), 0.3+noise(),
                    0.49+noise(), 0.3+noise(), 0.17+noise()]
        elif idx == 3: # recovery_sequence
            return [0.5+noise(), 0.6+noise(0.1), 0.55+noise(), 0.7+noise(),
                    0.59+noise(), 0.3+noise(), 0.39+noise()]
        elif idx == 4: # semantic_mismatch
            return [0.1+noise(), 0.7+noise(), 0.6+noise(), 0.5+noise(),
                    0.48+noise(), 0.07+noise(), 0.30+noise()]
        elif idx == 5: # latency_spike
            return [0.8+noise(), 0.4+noise(), 0.75+noise(), 0.2+noise(),
                    0.54+noise(), 0.32+noise(), 0.15+noise()]
        elif idx == 6: # memory_advantage
            return [0.7+noise(), 0.65+noise(), 0.95+noise(), 0.7+noise(),
                    0.75+noise(), 0.46+noise(), 0.67+noise()]
        elif idx == 7: # topology_shortcut
            return [0.6+noise(), 0.7+noise(), 0.6+noise(), 0.95+noise(),
                    0.71+noise(), 0.42+noise(), 0.57+noise()]
        elif idx == 8: # time_peak_stress
            return [0.7+noise(), 0.5+noise(), 0.6+noise(), 0.4+noise(),
                    0.55+noise(), 0.35+noise(), 0.24+noise()]
        elif idx == 9: # drift_instability
            return [0.3+noise(0.1), 0.5+noise(), 0.4+noise(), 0.5+noise(),
                    0.43+noise(), 0.15+noise(), 0.20+noise()]
        elif idx == 10: # exploration_bonus
            return [random.random() for _ in range(7)]  # fully random
        else:          # steady_state
            return [0.7+noise(), 0.72+noise(), 0.68+noise(), 0.65+noise(),
                    0.69+noise(), 0.50+noise(), 0.44+noise()]

    def _clip_vec(self, v: List[float]) -> List[float]:
        return [max(0.0, min(1.0, x)) for x in v]

    def generate(self, n: int = 1) -> List[Experience]:
        """Generate n synthetic experiences across all scenario types."""
        experiences = []
        for _ in range(n):
            # Weighted scenario selection — less-seen scenarios get priority
            counts = self._scenario_counts
            max_c  = max(counts) + 1
            weights = [max_c - c for c in counts]
            idx = random.choices(range(len(self.SCENARIOS)), weights=weights, k=1)[0]

            name, base_target, base_reward = self.SCENARIOS[idx]
            vec = self._clip_vec(self._scenario_vector(idx))
            noise_target = float(np.clip(base_target + random.gauss(0, 0.05), 0.0, 1.0))

            exp = Experience(
                feature_vec=vec,
                target=noise_target,
                source=f"self_stim:{name}",
                reward=base_reward + random.gauss(0, 0.1),
                context={"scenario": name, "scenario_idx": idx},
            )
            experiences.append(exp)
            self._scenario_counts[idx] += 1
            self._total_generated += 1

        return experiences

    def summary(self) -> dict:
        return {
            "total_generated": self._total_generated,
            "scenario_counts": {
                s[0]: self._scenario_counts[i]
                for i, s in enumerate(self.SCENARIOS)
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Curiosity Engine — exploration-driven signal generation
# ─────────────────────────────────────────────────────────────────────────────

class CuriosityEngine:
    """
    Phase 10: Drives exploration of unknown territory.

    Maintains a curiosity map across the 7-dimensional feature space.
    Regions that have been seen less get higher curiosity scores.
    The engine generates experiences that deliberately probe
    under-explored regions.

    Curiosity decays after visiting a region (satiation).
    """

    GRID_SIZE = 5   # 5^7 = 78,125 cells — coarse but manageable

    def __init__(self):
        # visit count per grid cell (sparse dict)
        self._visit_counts: Dict[Tuple, int] = {}
        self._total_explorations = 0
        self._curiosity_level = 1.0  # global curiosity 0-1
        self._peak_curiosity_regions: List[Tuple] = []

    def _to_cell(self, vec: List[float]) -> Tuple:
        """Discretise a feature vector to a grid cell."""
        return tuple(int(v * (self.GRID_SIZE - 1)) for v in vec)

    def _curiosity_score(self, cell: Tuple) -> float:
        """Higher score = less visited = more curious."""
        visits = self._visit_counts.get(cell, 0)
        return 1.0 / (1.0 + visits)

    def _sample_curious_vec(self) -> List[float]:
        """
        Sample a feature vector from an under-explored region.
        Uses rejection sampling — keeps trying until it finds
        a low-visit-count region.
        """
        best_vec = None
        best_score = -1.0
        for _ in range(20):   # 20 candidate vectors
            candidate = [random.random() for _ in range(7)]
            cell = self._to_cell(candidate)
            score = self._curiosity_score(cell)
            if score > best_score:
                best_score = score
                best_vec   = candidate
        return best_vec

    def record_visit(self, vec: List[float]) -> None:
        """Mark a region as visited — reduces its curiosity."""
        cell = self._to_cell(vec)
        self._visit_counts[cell] = self._visit_counts.get(cell, 0) + 1
        # Decay global curiosity
        self._curiosity_level = max(
            MIN_CURIOSITY,
            self._curiosity_level * CURIOSITY_DECAY
        )

    def generate(self, n: int = 1) -> List[Experience]:
        """
        Generate n curiosity-driven experiences.
        Target is unknown (0.5) — the network will learn from the outcome.
        """
        experiences = []
        for _ in range(n):
            vec = self._sample_curious_vec()
            self.record_visit(vec)

            # Target is uncertain — we explore without knowing the answer
            target = 0.5 + random.gauss(0, 0.2)
            target = float(np.clip(target, 0.0, 1.0))

            exp = Experience(
                feature_vec=vec,
                target=target,
                source="curiosity",
                reward=self._curiosity_level * 0.3,  # reward for exploring
                context={
                    "curiosity_level": round(self._curiosity_level, 4),
                    "region_visits":   self._visit_counts.get(self._to_cell(vec), 0),
                },
            )
            experiences.append(exp)
            self._total_explorations += 1

        return experiences

    def most_curious_regions(self, top_k: int = 5) -> List[dict]:
        """Return top-k least-visited regions."""
        if not self._visit_counts:
            return []
        sorted_cells = sorted(
            self._visit_counts.items(), key=lambda x: x[1]
        )[:top_k]
        return [{"cell": list(c), "visits": v} for c, v in sorted_cells]

    def summary(self) -> dict:
        return {
            "total_explorations":   self._total_explorations,
            "curiosity_level":      round(self._curiosity_level, 4),
            "unique_regions_visited": len(self._visit_counts),
            "most_curious_regions": self.most_curious_regions(3),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Dream Consolidator — idle-time experience replay
# ─────────────────────────────────────────────────────────────────────────────

class DreamConsolidator:
    """
    Phase 10: Replays past experiences during idle periods.

    Inspired by sleep consolidation in biological brains:
    - During idle periods (no real signals), the system replays
      its strongest memories
    - It focuses on high-reward AND high-loss experiences
      (the memorable events)
    - It extracts 'consolidated' patterns — abstract rules that
      generalise across many specific memories

    Dream cycles run automatically when signal_bus detects idleness.
    """

    def __init__(self, replay_buffer: ReplayBuffer):
        self._buffer       = replay_buffer
        self._dream_cycles = 0
        self._last_dream   = 0.0
        self._consolidated_patterns: List[dict] = []
        self._total_replayed = 0

    def should_dream(self, last_signal_ts: float) -> bool:
        """Return True if enough idle time has passed."""
        idle_secs = time.time() - last_signal_ts
        return idle_secs >= DREAM_IDLE_THRESHOLD

    def dream(self, n_batches: int = 5) -> List[Experience]:
        """
        Run one dream cycle: sample from buffer, weight by importance.

        Returns the list of replayed experiences (for training).
        Important = high |reward| or high loss (surprising events).
        """
        if self._buffer.size < BATCH_SIZE:
            return []

        replayed = []

        # Sample high-reward experiences (positive memories)
        pos = self._buffer.sample_by_source("real", n=BATCH_SIZE // 2)
        if not pos:
            pos = self._buffer.sample(n=BATCH_SIZE // 2)

        # Sample self-stim experiences for variety
        stim = self._buffer.sample_by_source("self_stim:cascade_failure",
                                             n=BATCH_SIZE // 4)
        curious = self._buffer.sample_by_source("curiosity", n=BATCH_SIZE // 4)

        batch = pos + stim + curious
        if not batch:
            batch = self._buffer.sample(BATCH_SIZE)

        # Tag as dream replay
        for exp in batch:
            dream_exp = Experience(
                feature_vec=exp.feature_vec,
                target=exp.target,
                source="dream",
                reward=exp.reward * 0.8,   # slightly discounted
                context={**exp.context, "original_source": exp.source},
            )
            replayed.append(dream_exp)
            self._total_replayed += 1

        self._dream_cycles += 1
        self._last_dream = time.time()

        # Extract consolidated pattern from this dream batch
        if len(replayed) >= 4:
            targets = [e.target for e in replayed]
            vecs    = [e.feature_vec for e in replayed]
            pattern = {
                "cycle":       self._dream_cycles,
                "timestamp":   datetime.now(timezone.utc).isoformat(),
                "avg_target":  round(sum(targets) / len(targets), 4),
                "avg_vec":     [round(sum(v[i] for v in vecs) / len(vecs), 4)
                                for i in range(7)],
                "n_replayed":  len(replayed),
            }
            self._consolidated_patterns.append(pattern)
            if len(self._consolidated_patterns) > 100:
                self._consolidated_patterns = self._consolidated_patterns[-100:]

        logger.info(
            f"DreamConsolidator: cycle {self._dream_cycles}  "
            f"replayed {len(replayed)} experiences"
        )
        return replayed

    def summary(self) -> dict:
        return {
            "dream_cycles":           self._dream_cycles,
            "total_replayed":         self._total_replayed,
            "consolidated_patterns":  len(self._consolidated_patterns),
            "last_pattern":           (self._consolidated_patterns[-1]
                                       if self._consolidated_patterns else None),
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Signal Bus — central hub connecting all signal sources to neural layers
# ─────────────────────────────────────────────────────────────────────────────

class SignalBus:
    """
    Phase 10: The nervous system's central signal router.

    Orchestrates all signal sources and routes their output to the
    neural layers for continuous training.

    Signal flow:
      [SelfStimulator]      ──┐
      [CuriosityEngine]     ──┤→ ReplayBuffer → batch → DeepNetwork
      [DreamConsolidator]   ──┤                       → DynamicLayer
      [External/Real]       ──┘                       → Phase8 Layer

    Runs in a background thread — completely autonomous.

    Usage
    -----
        bus = SignalBus(deep_network, dynamic_layer, neural_layer)
        bus.start()          # begins the continuous signal loop
        bus.push_real(vec, target)  # inject a real routing event
        bus.stop()
    """

    def __init__(
        self,
        deep_network=None,
        dynamic_layer=None,
        neural_layer=None,
        rich_data_collector=None,
    ):
        self.replay_buffer    = ReplayBuffer()
        self.stimulator       = SelfStimulator()
        self.curiosity        = CuriosityEngine()
        self.dream            = DreamConsolidator(self.replay_buffer)

        self._deep_network    = deep_network
        self._dynamic_layer   = dynamic_layer
        self._neural_layer    = neural_layer
        self._rich_data       = rich_data_collector

        self._running         = False
        self._thread: Optional[threading.Thread] = None
        self._last_signal_ts  = time.time()
        self._lock            = threading.Lock()

        # Statistics
        self._stats = {
            "total_signals":       0,
            "real_signals":        0,
            "synthetic_signals":   0,
            "dream_signals":       0,
            "total_train_steps":   0,
            "avg_loss":            0.0,
            "loss_history":        deque(maxlen=500),
            "signal_rate_per_min": 0.0,
            "started_at":          None,
        }
        self._callbacks: List[Callable] = []

        logger.info("SignalBus (Phase 10) initialised — all signal sources ready")

    # ── Public API ────────────────────────────────────────────────────────

    def push_real(
        self,
        feature_vec: List[float],
        target: float,
        reward: float = 0.0,
        context: Optional[dict] = None,
    ) -> None:
        """
        Inject a real routing experience into the signal stream.
        Called by RoutingEngine after each route decision.
        """
        exp = Experience(
            feature_vec=feature_vec,
            target=target,
            source="real",
            reward=reward,
            context=context or {},
        )
        self.replay_buffer.add(exp)
        self._train_on_batch([exp])
        self._last_signal_ts = time.time()
        with self._lock:
            self._stats["real_signals"] += 1
            self._stats["total_signals"] += 1

    def start(self, interval_s: float = SIGNAL_INTERVAL_S) -> None:
        """Start the continuous signal loop in a background thread."""
        if self._running:
            return
        self._running = True
        self._stats["started_at"] = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(
            target=self._signal_loop,
            args=(interval_s,),
            daemon=True,
            name="SignalBus-Phase10",
        )
        self._thread.start()
        logger.info(f"SignalBus started — pulse every {interval_s}s")

    def stop(self) -> None:
        """Stop the signal loop gracefully."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        logger.info("SignalBus stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def register_callback(self, fn: Callable) -> None:
        """Register a function called after each training step."""
        self._callbacks.append(fn)

    # ── Internal signal loop ──────────────────────────────────────────────

    def _signal_loop(self, interval_s: float) -> None:
        """Main loop — runs in background thread forever."""
        pulse_count = 0
        loop_start  = time.time()

        while self._running:
            try:
                pulse_count += 1
                t0 = time.time()

                # 1. Generate synthetic signals
                stim_exps = self.stimulator.generate(n=3)
                for e in stim_exps:
                    self.replay_buffer.add(e)

                # 2. Generate curiosity-driven signals
                cur_exps = self.curiosity.generate(n=2)
                for e in cur_exps:
                    self.replay_buffer.add(e)

                synthetic = stim_exps + cur_exps
                with self._lock:
                    self._stats["synthetic_signals"] += len(synthetic)
                    self._stats["total_signals"] += len(synthetic)

                # 3. Train on a mixed batch from replay buffer
                if self.replay_buffer.size >= BATCH_SIZE:
                    batch = self.replay_buffer.sample(BATCH_SIZE)
                    self._train_on_batch(batch)

                # 4. Dream mode — activate when idle
                if self.dream.should_dream(self._last_signal_ts):
                    dream_exps = self.dream.dream(n_batches=3)
                    for e in dream_exps:
                        self.replay_buffer.add(e)
                    if dream_exps:
                        self._train_on_batch(dream_exps)
                    with self._lock:
                        self._stats["dream_signals"] += len(dream_exps)

                # 5. Signal rate calculation
                elapsed = time.time() - loop_start
                if elapsed > 0:
                    rate = self._stats["total_signals"] / elapsed * 60
                    with self._lock:
                        self._stats["signal_rate_per_min"] = round(rate, 1)

                # 6. Fire callbacks
                for cb in self._callbacks:
                    try:
                        cb({"pulse": pulse_count, "stats": self.get_stats()})
                    except Exception:
                        pass

                # Sleep for the interval minus processing time
                processing_time = time.time() - t0
                sleep_time = max(0.0, interval_s - processing_time)
                time.sleep(sleep_time)

            except Exception as e:
                logger.error(f"SignalBus loop error: {e}")
                time.sleep(interval_s)

    def _train_on_batch(self, batch: List[Experience]) -> None:
        """Train all neural layers on a batch of experiences."""
        if not batch:
            return

        losses = []

        for exp in batch:
            x      = exp.feature_vec
            target = exp.target

            # Phase 9 Axis-3: Deep network (primary learner)
            if self._deep_network is not None:
                try:
                    loss = self._deep_network.train_step(x, target)
                    losses.append(loss)
                except Exception as e:
                    logger.debug(f"DeepNetwork train error: {e}")

            # Phase 9 Axis-2: Dynamic growing layer
            if self._dynamic_layer is not None:
                try:
                    self._dynamic_layer.train_step(x, target)
                except Exception as e:
                    logger.debug(f"DynamicLayer train error: {e}")

            # Phase 8: Original neural layer
            if self._neural_layer is not None:
                try:
                    self._neural_layer.train_step(x, target)
                except Exception as e:
                    logger.debug(f"NeuralLayer train error: {e}")

        # Update stats
        with self._lock:
            self._stats["total_train_steps"] += len(batch)
            if losses:
                avg = sum(losses) / len(losses)
                self._stats["loss_history"].append(avg)
                # Exponential moving average
                old = self._stats["avg_loss"]
                self._stats["avg_loss"] = round(
                    0.95 * old + 0.05 * avg if old > 0 else avg, 6
                )

    # ── Status ────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
        recent_losses = list(stats.pop("loss_history", []))
        return {
            **stats,
            "is_running":       self._running,
            "buffer":           self.replay_buffer.summary(),
            "stimulator":       self.stimulator.summary(),
            "curiosity":        self.curiosity.summary(),
            "dream":            self.dream.summary(),
            "recent_avg_loss":  round(
                sum(recent_losses[-20:]) / len(recent_losses[-20:]), 6
            ) if recent_losses else None,
            "loss_trend":       self._loss_trend(recent_losses),
        }

    def _loss_trend(self, losses: List[float]) -> str:
        if len(losses) < 10:
            return "insufficient_data"
        first_half = sum(losses[:len(losses)//2]) / (len(losses)//2)
        second_half = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)
        if second_half < first_half * 0.95:
            return "improving"
        if second_half > first_half * 1.05:
            return "degrading"
        return "stable"
