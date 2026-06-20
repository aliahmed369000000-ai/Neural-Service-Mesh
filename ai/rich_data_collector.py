"""
Phase 9 — Axis 1: Rich Data Sources
=====================================
Replaces the simple routing_events-only feed with 7 rich data sources
that provide the NeuralWeightLayer with far deeper signal:

    1. routing_events      — success/failure of each route (existing)
    2. node_performance    — per-node latency, error-rate, throughput
    3. time_patterns       — hour-of-day / day-of-week execution patterns
    4. failure_patterns    — sequences of failures (cascade detection)
    5. load_patterns       — queue depth & concurrent requests per node
    6. semantic_drift      — how much node semantics shift over time
    7. external_signals    — environment-model events (CPU, memory, anomalies)

Each source produces a normalised 7-element feature sub-vector.
The collector merges them into a single rich feature vector used for
training and inference in the Phase 9 neural weight layer.

Backward-compatible: the existing `_build_feature_vector(breakdown)`
path in RoutingEngine continues to work unchanged.  The new enriched
vector is only used when RichDataCollector is wired in.
"""
from __future__ import annotations

import logging
import math
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Feature-vector length produced by this collector ─────────────────────────
RICH_FEATURE_DIM = 784  # 49 قيمة (7 sources × 7) + 735 TF-IDF hash (يطابق neural_core.py INPUT_DIM=784)


class NodePerformanceTracker:
    """
    Axis-1 sub-component: tracks per-node statistics.

    Metrics kept in a rolling window of the last N executions:
      - success_rate   (0-1)
      - avg_latency_ms
      - error_rate     (0-1)
      - throughput     (requests / total time window in seconds)
    """

    WINDOW = 200  # rolling window size

    def __init__(self):
        # node_id → deque of (success: bool, latency_ms: float, ts: float)
        self._history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.WINDOW))

    def record(self, node_id: str, success: bool, latency_ms: float) -> None:
        ts = datetime.now(timezone.utc).timestamp()
        self._history[node_id].append((success, latency_ms, ts))

    def stats(self, node_id: str) -> Dict[str, float]:
        h = list(self._history.get(node_id, []))
        if not h:
            return {"success_rate": 0.5, "avg_latency_ms": 0.0,
                    "error_rate": 0.5, "throughput": 0.0}
        n = len(h)
        successes = sum(1 for s, _, __ in h if s)
        avg_lat = sum(lat for _, lat, __ in h) / n
        if n > 1:
            span = h[-1][2] - h[0][2]
            throughput = n / max(span, 1.0)
        else:
            throughput = 0.0
        return {
            "success_rate": successes / n,
            "avg_latency_ms": avg_lat,
            "error_rate": (n - successes) / n,
            "throughput": throughput,
        }

    def feature_vector(self, node_id: str) -> List[float]:
        """Return a 7-element normalised feature vector for `node_id`."""
        s = self.stats(node_id)
        sr = s["success_rate"]
        # latency: map 0-5000 ms → 1-0  (lower latency = higher score)
        lat_norm = max(0.0, 1.0 - s["avg_latency_ms"] / 5000.0)
        er = s["error_rate"]
        # throughput: log-scale, cap at 100 req/s
        tp = min(math.log(s["throughput"] + 1) / math.log(101), 1.0)
        avg = (sr + lat_norm + (1 - er) + tp) / 4.0
        return [sr, lat_norm, er, tp, avg, sr * lat_norm, (1 - er) * tp]


class TimePatternTracker:
    """
    Axis-1 sub-component: captures time-of-day / day-of-week patterns.

    Records which hours and days have historically high/low success rates
    so the model learns temporal routing preferences.
    """

    def __init__(self):
        # (hour, weekday) → [success_count, total_count]
        self._hourly: Dict[Tuple[int, int], List[int]] = defaultdict(lambda: [0, 0])
        self._total_events = 0

    def record(self, success: bool) -> None:
        now = datetime.now(timezone.utc)
        key = (now.hour, now.weekday())
        self._hourly[key][1] += 1
        if success:
            self._hourly[key][0] += 1
        self._total_events += 1

    def feature_vector(self) -> List[float]:
        """7-element vector encoding current temporal context."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        weekday = now.weekday()  # 0=Monday … 6=Sunday

        # Cyclical encoding of hour (sin/cos)
        hour_sin = (math.sin(2 * math.pi * hour / 24) + 1) / 2
        hour_cos = (math.cos(2 * math.pi * hour / 24) + 1) / 2
        # Cyclical encoding of weekday
        day_sin = (math.sin(2 * math.pi * weekday / 7) + 1) / 2
        day_cos = (math.cos(2 * math.pi * weekday / 7) + 1) / 2

        # Historical success rate for this (hour, weekday) slot
        key = (hour, weekday)
        counts = self._hourly.get(key, [0, 0])
        hist_sr = counts[0] / counts[1] if counts[1] > 0 else 0.5

        # Density: how often this slot is used relative to average
        avg_count = self._total_events / max(len(self._hourly), 1)
        density = min(counts[1] / max(avg_count, 1), 2.0) / 2.0

        # Is it peak hours? (9-18 weekdays)
        is_peak = 1.0 if (0 <= weekday <= 4 and 9 <= hour <= 18) else 0.0

        return [hour_sin, hour_cos, day_sin, day_cos, hist_sr, density, is_peak]


class FailurePatternDetector:
    """
    Axis-1 sub-component: detects failure cascades.

    A cascade = 3+ consecutive failures within a short window.
    Features capture how "healthy" the current execution stream is.
    """

    CASCADE_WINDOW = 20   # look at last 20 events
    CASCADE_THRESHOLD = 3  # 3+ failures in a row = cascade

    def __init__(self):
        self._events: deque = deque(maxlen=100)  # (ts, success, node_id)
        self._cascade_count = 0

    def record(self, success: bool, node_id: str = "") -> None:
        ts = datetime.now(timezone.utc).timestamp()
        self._events.append((ts, success, node_id))

    def _recent(self, n: int) -> List[bool]:
        events = list(self._events)[-n:]
        return [e[1] for e in events]

    def feature_vector(self) -> List[float]:
        recent = self._recent(self.CASCADE_WINDOW)
        if not recent:
            return [1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5]

        n = len(recent)
        successes = sum(recent)
        recent_sr = successes / n

        # Consecutive failures at the tail
        consecutive_failures = 0
        for ok in reversed(recent):
            if not ok:
                consecutive_failures += 1
            else:
                break
        cascade_active = 1.0 if consecutive_failures >= self.CASCADE_THRESHOLD else 0.0
        cascade_severity = min(consecutive_failures / 10.0, 1.0)

        # Trend: compare first-half vs second-half success rates
        mid = n // 2
        first_half_sr = sum(recent[:mid]) / mid if mid > 0 else 0.5
        second_half_sr = sum(recent[mid:]) / (n - mid) if (n - mid) > 0 else 0.5
        trend = (second_half_sr - first_half_sr + 1.0) / 2.0  # normalised to [0,1]

        # Volatility (std-dev of success as 0/1 sequence)
        mean = recent_sr
        variance = sum((int(x) - mean) ** 2 for x in recent) / n
        volatility = math.sqrt(variance)

        return [
            recent_sr,          # overall recent success rate
            cascade_active,     # 1 if cascade currently active
            cascade_severity,   # how severe the cascade is
            trend,              # improvement trend (0=degrading, 1=improving)
            volatility,         # instability measure
            first_half_sr,      # baseline success rate
            second_half_sr,     # current success rate
        ]


class LoadPatternTracker:
    """
    Axis-1 sub-component: estimates concurrent load per node.
    """

    def __init__(self):
        # node_id → current inflight count
        self._inflight: Dict[str, int] = defaultdict(int)
        self._peak: Dict[str, int] = defaultdict(int)
        self._total_requests: int = 0
        self._history: deque = deque(maxlen=500)  # (ts, load)

    def enter(self, node_id: str) -> None:
        self._inflight[node_id] += 1
        self._peak[node_id] = max(self._peak[node_id], self._inflight[node_id])
        self._total_requests += 1
        ts = datetime.now(timezone.utc).timestamp()
        total_load = sum(self._inflight.values())
        self._history.append((ts, total_load))

    def exit(self, node_id: str) -> None:
        if self._inflight[node_id] > 0:
            self._inflight[node_id] -= 1

    def feature_vector(self) -> List[float]:
        total_inflight = sum(self._inflight.values())
        node_count = max(len(self._inflight), 1)
        avg_inflight = total_inflight / node_count

        # Load normalised to a "healthy" ceiling of 10 concurrent per node
        load_norm = min(avg_inflight / 10.0, 1.0)

        # Peak load pressure
        avg_peak = sum(self._peak.values()) / node_count
        peak_norm = min(avg_peak / 20.0, 1.0)

        # Recent load trend (last 10 vs last 50 entries)
        history = list(self._history)
        if len(history) >= 10:
            recent_10 = sum(v for _, v in history[-10:]) / 10
            recent_50 = sum(v for _, v in history[-50:]) / len(history[-50:])
            load_trend = (recent_10 - recent_50 + 10) / 20  # normalised
        else:
            load_trend = 0.5

        # Total requests (log-normalised)
        total_norm = min(math.log(self._total_requests + 1) / math.log(10001), 1.0)

        # Imbalance: max inflight / avg inflight
        if self._inflight:
            max_inf = max(self._inflight.values())
            imbalance = min(max_inf / max(avg_inflight, 0.1), 5.0) / 5.0
        else:
            imbalance = 0.0

        return [
            load_norm,      # current average load
            peak_norm,      # historical peak pressure
            load_trend,     # load increasing (>0.5) or decreasing (<0.5)
            total_norm,     # total throughput experience
            imbalance,      # load imbalance between nodes
            1.0 - load_norm,  # inverse load (capacity available)
            (load_norm + peak_norm) / 2,  # overall load score
        ]


class SemanticDriftDetector:
    """
    Axis-1 sub-component: detects how much node descriptions/capabilities drift.

    When a node changes its capabilities frequently, routing decisions based on
    old semantic matching become less reliable.  This component tracks
    semantic stability as a feature signal.
    """

    def __init__(self):
        # node_id → list of (ts, capability_hash)
        self._snapshots: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        self._drift_scores: Dict[str, float] = {}

    def record_snapshot(self, node_id: str, capabilities: Any) -> None:
        """Record a capability snapshot for drift detection."""
        cap_hash = hash(str(sorted(capabilities) if isinstance(capabilities, (list, set))
                           else str(capabilities))) % (10 ** 9)
        ts = datetime.now(timezone.utc).timestamp()
        history = self._snapshots[node_id]
        history.append((ts, cap_hash))

        # Compute drift: fraction of unique hashes in window
        if len(history) >= 2:
            hashes = [h for _, h in history]
            unique_ratio = len(set(hashes)) / len(hashes)
            self._drift_scores[node_id] = unique_ratio

    def global_drift(self) -> float:
        """Average drift score across all nodes (0=stable, 1=chaotic)."""
        if not self._drift_scores:
            return 0.1  # assume mostly stable
        return sum(self._drift_scores.values()) / len(self._drift_scores)

    def feature_vector(self) -> List[float]:
        drift = self.global_drift()
        stability = 1.0 - drift

        # How many nodes have we tracked?
        node_coverage = min(len(self._snapshots) / 20.0, 1.0)

        # Recent drift vs overall
        recent_drift = drift  # simplified — could weight recent snapshots
        long_term_drift = drift * 0.8  # smoothed approximation

        # Stability trend
        drift_trend = 0.5  # neutral — could track drift over time

        return [
            stability,          # semantic stability (higher = more reliable)
            drift,              # drift magnitude
            node_coverage,      # how many nodes we have data for
            recent_drift,       # recent semantic changes
            long_term_drift,    # long-term semantic drift
            drift_trend,        # improving or worsening
            (stability + node_coverage) / 2,  # composite signal quality
        ]


class ExternalSignalReader:
    """
    Axis-1 sub-component: converts EnvironmentModel events into features.

    Reads from the world model (Phase 7) when available, otherwise
    returns neutral defaults.
    """

    def __init__(self, env_model=None):
        self._env = env_model
        self._last_signals: Dict[str, float] = {}
        self._anomaly_count = 0
        self._event_history: deque = deque(maxlen=100)

    def set_env_model(self, env_model) -> None:
        self._env = env_model

    def record_event(self, event_type: str, severity: float = 0.5) -> None:
        """Register an external event (called by SensorHub integration)."""
        self._event_history.append((
            datetime.now(timezone.utc).timestamp(),
            event_type,
            severity,
        ))
        if severity > 0.7:
            self._anomaly_count += 1

    def feature_vector(self) -> List[float]:
        """7-element vector from environment/external signals."""
        try:
            if self._env is not None:
                state = self._env.snapshot() if hasattr(self._env, "snapshot") else {}
            else:
                state = {}
        except Exception:
            state = {}

        cpu = min(state.get("cpu_usage", 0.3), 1.0)
        mem = min(state.get("memory_usage", 0.3), 1.0)
        err_rate = min(state.get("error_rate", 0.05), 1.0)
        latency = min(state.get("p99_latency_ms", 100) / 5000.0, 1.0)

        # Recent anomaly density
        history = list(self._event_history)
        if history:
            recent_50 = history[-50:]
            anomaly_density = sum(1 for _, _, sev in recent_50 if sev > 0.7) / len(recent_50)
        else:
            anomaly_density = 0.0

        # System health composite
        health = max(0.0, 1.0 - (cpu * 0.3 + mem * 0.3 + err_rate * 0.4))

        return [
            1.0 - cpu,          # inverse CPU (higher = more headroom)
            1.0 - mem,          # inverse memory pressure
            1.0 - err_rate,     # system error cleanliness
            1.0 - latency,      # system latency health
            1.0 - anomaly_density,  # anomaly-free score
            health,             # composite system health
            0.5,                # reserved (future: external API health)
        ]


# ─────────────────────────────────────────────────────────────────────────────
#  Main collector: merges all 7 sources into a single rich feature vector
# ─────────────────────────────────────────────────────────────────────────────

class RichDataCollector:
    """
    Phase 9 Axis-1: Aggregates all 7 data sources.

    The `collect(route_breakdown, node_id)` method returns a rich
    feature vector suitable for the Phase 9 multi-layer network.

    The vector is always of length RICH_FEATURE_DIM (784), computed as
        49 weighted source values (7 sources × 7) + 735 TF-IDF hash values,
    the *weighted average* of the 7 source sub-vectors so it remains
    compatible with the existing NeuralWeightLayer input contract.

    Source weights (equal by default; can be tuned):
        routing_events:    0.20
        node_performance:  0.20
        time_patterns:     0.10
        failure_patterns:  0.15
        load_patterns:     0.10
        semantic_drift:    0.10
        external_signals:  0.15
    """

    SOURCE_WEIGHTS = {
        "routing_events":   0.20,
        "node_performance": 0.20,
        "time_patterns":    0.10,
        "failure_patterns": 0.15,
        "load_patterns":    0.10,
        "semantic_drift":   0.10,
        "external_signals": 0.15,
    }

    def __init__(self, env_model=None):
        self.node_perf = NodePerformanceTracker()
        self.time_pat = TimePatternTracker()
        self.failure_det = FailurePatternDetector()
        self.load_track = LoadPatternTracker()
        self.semantic_drift = SemanticDriftDetector()
        self.ext_signals = ExternalSignalReader(env_model)
        self._total_collections = 0
        logger.info("RichDataCollector (Phase 9 Axis-1) initialised — 7 data sources active")

    def set_env_model(self, env_model) -> None:
        self.ext_signals.set_env_model(env_model)

    # ── Recording helpers (called by RoutingEngine / ExecutionEngine hooks) ──

    def record_routing_event(self, node_id: str, success: bool,
                             latency_ms: float = 0.0) -> None:
        """Record one routing execution result across all relevant trackers."""
        self.node_perf.record(node_id, success, latency_ms)
        self.time_pat.record(success)
        self.failure_det.record(success, node_id)
        self.load_track.exit(node_id)

    def record_request_start(self, node_id: str) -> None:
        self.load_track.enter(node_id)

    def record_external_event(self, event_type: str, severity: float = 0.5) -> None:
        self.ext_signals.record_event(event_type, severity)

    def record_capability_snapshot(self, node_id: str, capabilities) -> None:
        self.semantic_drift.record_snapshot(node_id, capabilities)

    # ── Feature vector construction ────────────────────────────────────────

    def _routing_events_vector(self, breakdown: dict) -> List[float]:
        """
        Source 1: the existing routing breakdown dict.
        Mirrors the original `_build_feature_vector` logic.
        """
        sem   = breakdown.get("semantic", 50.0) / 100.0
        score = breakdown.get("score",    50.0) / 100.0
        mem   = breakdown.get("memory",   50.0) / 100.0
        topo  = breakdown.get("topology", 50.0) / 100.0
        avg         = (sem + score + mem + topo) / 4.0
        sem_x_score = sem * score
        mem_x_topo  = mem * topo
        return [sem, score, mem, topo, avg, sem_x_score, mem_x_topo]

    def collect(
        self,
        breakdown: dict,
        node_id: str = "",
    ) -> List[float]:
        """
        Build and return the merged 7-element rich feature vector.

        Parameters
        ----------
        breakdown : dict
            Route score breakdown from RoutingEngine
            (keys: semantic, score, memory, topology).
        node_id : str
            Primary node being evaluated (for per-node stats).

        Returns
        -------
        list[float]  length 7, all values in [0, 1]
        """
        sources: Dict[str, List[float]] = {
            "routing_events":   self._routing_events_vector(breakdown),
            "node_performance": self.node_perf.feature_vector(node_id) if node_id else [0.5]*7,
            "time_patterns":    self.time_pat.feature_vector(),
            "failure_patterns": self.failure_det.feature_vector(),
            "load_patterns":    self.load_track.feature_vector(),
            "semantic_drift":   self.semantic_drift.feature_vector(),
            "external_signals": self.ext_signals.feature_vector(),
        }

        # Weighted average across sources → BASE_DIM values (n_sources × 7)
        src_list = list(sources.items())
        BASE_DIM = len(src_list) * 7  # 7 sources × 7 = 49
        merged_base = [0.0] * BASE_DIM
        for si, (source_name, vec) in enumerate(src_list):
            w = self.SOURCE_WEIGHTS[source_name]
            base_idx = si * 7
            for i in range(min(7, len(vec))):
                merged_base[base_idx + i] = max(0.0, min(1.0, w * vec[i]))

        # توسيع إلى 784 بإضافة 735 قيمة TF-IDF hash
        import math
        n_hash = max(1, RICH_FEATURE_DIM - BASE_DIM)  # متغير حسب عدد المصادر
        hash_vec = [0.0] * n_hash
        text_key = (node_id or "") + "|" + "|".join(f"{v:.2f}" for v in merged_base[:7])
        for i in range(len(text_key) - 1):
            h = abs(hash(text_key[i:i+2])) % n_hash
            hash_vec[h] += 1.0
        total_h = sum(hash_vec)
        if total_h > 0:
            hash_vec = [math.log1p(v * 10.0 / total_h) / math.log1p(10.0) for v in hash_vec]
        merged = merged_base + hash_vec  # len=784

        self._total_collections += 1
        logger.debug(
            f"RichDataCollector.collect: node={node_id!r}  "
            f"merged_base={[round(v, 3) for v in merged_base[:7]]}  "
            f"total={self._total_collections}  dim={len(merged)}"
        )
        return merged  # len=784

    def summary(self) -> dict:
        """Serialisable summary for status reporting."""
        return {
            "total_collections": self._total_collections,
            "source_weights":    self.SOURCE_WEIGHTS,
            "node_count_tracked": len(self.node_perf._history),
            "failure_cascade_active": self.failure_det.feature_vector()[1] > 0.5,
            "global_semantic_drift": round(self.semantic_drift.global_drift(), 4),
            "load_feature_snapshot": [round(v, 3) for v in self.load_track.feature_vector()],
            "time_feature_snapshot": [round(v, 3) for v in self.time_pat.feature_vector()],
        }
