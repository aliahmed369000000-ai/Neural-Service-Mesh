"""
Phase 6 – Self Optimizer
==========================
Monitors node performance in real-time and autonomously re-routes
traffic away from slow or failing nodes.

Lifecycle per cycle:
  1. Collect latency / success-rate metrics for every active node
  2. Flag nodes whose score drops below threshold
  3. Find or spawn a replacement node
  4. Re-route all paths through the replacement
  5. Emit an OptimizationEvent to the audit trail

Example:
  optimizer = SelfOptimizer(registry, graph, scoring_engine)
  report = optimizer.run_cycle()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Thresholds
SLOW_LATENCY_MS   = 500.0   # flag node if avg latency > this
LOW_SUCCESS_RATE  = 0.70    # flag node if success rate < this
MIN_OBSERVATIONS  = 3       # need at least this many runs before judging


class NodeHealthSnapshot:
    """Performance snapshot for a single node."""

    def __init__(self, node_id: str, node_name: str):
        self.node_id = node_id
        self.node_name = node_name
        self.avg_latency_ms: float = 0.0
        self.success_rate: float = 1.0
        self.total_runs: int = 0
        self.health_score: float = 1.0  # composite 0–1
        self.flagged: bool = False
        self.flag_reason: Optional[str] = None

    def compute_health(self):
        """Compute a composite health score (higher = healthier)."""
        latency_factor = max(0.0, 1.0 - (self.avg_latency_ms / 2000.0))
        self.health_score = 0.6 * self.success_rate + 0.4 * latency_factor

        if self.total_runs < MIN_OBSERVATIONS:
            return

        if self.avg_latency_ms > SLOW_LATENCY_MS:
            self.flagged = True
            self.flag_reason = (
                f"High latency ({self.avg_latency_ms:.0f} ms > {SLOW_LATENCY_MS} ms)"
            )
        elif self.success_rate < LOW_SUCCESS_RATE:
            self.flagged = True
            self.flag_reason = (
                f"Low success rate ({self.success_rate:.1%} < {LOW_SUCCESS_RATE:.1%})"
            )

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": round(self.success_rate, 4),
            "total_runs": self.total_runs,
            "health_score": round(self.health_score, 4),
            "flagged": self.flagged,
            "flag_reason": self.flag_reason,
        }


class OptimizationEvent:
    """Records a single self-optimization action."""

    def __init__(
        self,
        event_type: str,
        node_id: str,
        node_name: str,
        reason: str,
        action_taken: str,
        replacement_id: Optional[str] = None,
    ):
        self.event_id = f"opt_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self.event_type = event_type   # replace / reroute / retire
        self.node_id = node_id
        self.node_name = node_name
        self.reason = reason
        self.action_taken = action_taken
        self.replacement_id = replacement_id
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "node_id": self.node_id,
            "node_name": self.node_name,
            "reason": self.reason,
            "action_taken": self.action_taken,
            "replacement_id": self.replacement_id,
            "timestamp": self.timestamp,
        }


class SelfOptimizerReport:
    """Full report from one optimization cycle."""

    def __init__(self, cycle: int):
        self.cycle = cycle
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.snapshots: List[NodeHealthSnapshot] = []
        self.flagged_nodes: List[str] = []
        self.events: List[OptimizationEvent] = []
        self.paths_rerouted: int = 0

    def complete(self):
        self.finished_at = datetime.now(timezone.utc).isoformat()

    @property
    def summary(self) -> dict:
        return {
            "cycle": self.cycle,
            "nodes_checked": len(self.snapshots),
            "nodes_flagged": len(self.flagged_nodes),
            "optimizations": len(self.events),
            "paths_rerouted": self.paths_rerouted,
        }

    def to_dict(self) -> dict:
        return {
            **self.summary,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "snapshots": [s.to_dict() for s in self.snapshots],
            "events": [e.to_dict() for e in self.events],
        }


class SelfOptimizer:
    """
    Phase 6: Autonomous Self-Optimizer.

    Continuously monitors the mesh and re-routes traffic away from
    underperforming nodes without human intervention.
    """

    def __init__(
        self,
        registry,
        graph,
        scoring_engine=None,
        memory_engine=None,
        knowledge_store=None,
        agent_factory=None,
    ):
        self._registry = registry
        self._graph = graph
        self._scoring = scoring_engine
        self._memory = memory_engine
        self._knowledge = knowledge_store
        self._factory = agent_factory
        self._cycle_count = 0
        self._history: List[SelfOptimizerReport] = []
        self._dna: dict = {}         # current system DNA (populated by SystemDNA)
        logger.info("SelfOptimizer initialised (Phase 6)")

    # ── Main cycle ────────────────────────────────────────────────────────

    def run_cycle(self) -> SelfOptimizerReport:
        self._cycle_count += 1
        report = SelfOptimizerReport(self._cycle_count)

        nodes = self._registry.list_all() if self._registry else []
        if not nodes:
            logger.warning("SelfOptimizer: no nodes in registry, skipping cycle")
            report.complete()
            self._history.append(report)
            return report

        # 1. Snapshot health of each node
        for node in nodes:
            snap = self._snapshot(node)
            report.snapshots.append(snap)
            if snap.flagged:
                report.flagged_nodes.append(snap.node_id)

        # 2. Act on flagged nodes
        for snap in report.snapshots:
            if not snap.flagged:
                continue
            event = self._handle_flagged(snap)
            if event:
                report.events.append(event)
                if event.event_type == "reroute":
                    report.paths_rerouted += 1

        report.complete()
        self._history.append(report)

        logger.info(
            f"SelfOptimizer cycle {self._cycle_count}: "
            f"{len(report.flagged_nodes)} flagged, "
            f"{len(report.events)} actions taken"
        )
        return report

    # ── Health snapshot ───────────────────────────────────────────────────

    def _snapshot(self, node) -> NodeHealthSnapshot:
        snap = NodeHealthSnapshot(node.node_id, node.name)

        if self._scoring:
            try:
                scores = self._scoring.list_scores()
                related = [
                    s for s in scores
                    if s.get("source_id") == node.node_id
                    or s.get("target_id") == node.node_id
                ]
                if related:
                    snap.avg_latency_ms = sum(
                        s.get("avg_latency_ms", 0) for s in related
                    ) / len(related)
                    snap.success_rate = sum(
                        s.get("success_rate", 1.0) for s in related
                    ) / len(related)
                    snap.total_runs = sum(s.get("total_runs", 0) for s in related)
            except Exception as exc:
                logger.debug(f"Could not get scores for {node.node_id}: {exc}")

        snap.compute_health()
        return snap

    # ── Handle flagged node ───────────────────────────────────────────────

    def _handle_flagged(self, snap: NodeHealthSnapshot) -> Optional[OptimizationEvent]:
        node = self._registry.get(snap.node_id) if self._registry else None
        if not node:
            return None

        # Attempt to find an alternative node in the graph
        replacement_id = self._find_replacement(snap.node_id)

        if replacement_id:
            self._reroute_paths(snap.node_id, replacement_id)
            event = OptimizationEvent(
                event_type="reroute",
                node_id=snap.node_id,
                node_name=snap.node_name,
                reason=snap.flag_reason or "Performance degraded",
                action_taken=f"Re-routed traffic to node {replacement_id[:8]}",
                replacement_id=replacement_id,
            )
        else:
            event = OptimizationEvent(
                event_type="flagged",
                node_id=snap.node_id,
                node_name=snap.node_name,
                reason=snap.flag_reason or "Performance degraded",
                action_taken="Flagged for review — no replacement found",
            )

        logger.warning(
            f"Node {snap.node_name} flagged: {snap.flag_reason}. "
            f"Action: {event.action_taken}"
        )
        return event

    def _find_replacement(self, node_id: str) -> Optional[str]:
        """Find an alternative node that can substitute the flagged one."""
        if not self._graph:
            return None
        try:
            all_nodes = self._registry.list_all()
            flagged_node = self._registry.get(node_id)
            if not flagged_node:
                return None

            for candidate in all_nodes:
                if candidate.node_id == node_id:
                    continue
                # Simple heuristic: same class or overlapping tags
                if (
                    type(candidate).__name__ == type(flagged_node).__name__
                    or set(candidate.tags) & set(flagged_node.tags)
                ):
                    return candidate.node_id
        except Exception as exc:
            logger.debug(f"Replacement search error: {exc}")
        return None

    def _reroute_paths(self, old_id: str, new_id: str):
        """Update graph edges to bypass the flagged node."""
        if not self._graph:
            return
        try:
            if hasattr(self._graph, "reroute"):
                self._graph.reroute(old_id, new_id)
            else:
                # Fallback: add edge from predecessors to new node
                preds = list(self._graph._graph.predecessors(old_id)) \
                    if hasattr(self._graph, "_graph") else []
                for pred in preds:
                    self._graph.connect(pred, new_id)
        except Exception as exc:
            logger.debug(f"Reroute error: {exc}")

    # ── DNA integration ───────────────────────────────────────────────────

    def update_dna(self, dna: dict):
        """Called by SystemDNA to inject the current DNA snapshot."""
        self._dna = dna

    # ── History & summary ─────────────────────────────────────────────────

    def history(self, limit: int = 10) -> List[dict]:
        return [r.to_dict() for r in self._history[-limit:]]

    def summary(self) -> dict:
        total_events = sum(len(r.events) for r in self._history)
        total_flagged = sum(len(r.flagged_nodes) for r in self._history)
        return {
            "cycles_run": self._cycle_count,
            "total_flagged_nodes": total_flagged,
            "total_optimizations": total_events,
            "total_paths_rerouted": sum(r.paths_rerouted for r in self._history),
            "thresholds": {
                "slow_latency_ms": SLOW_LATENCY_MS,
                "low_success_rate": LOW_SUCCESS_RATE,
                "min_observations": MIN_OBSERVATIONS,
            },
        }
