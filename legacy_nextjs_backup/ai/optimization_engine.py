"""
Phase 3 – Optimization Engine
Periodically analyses the service graph and:
  (Knowledge Layer: writes all analysis results to graph_metrics.json)
  - Removes consistently failing connections
  - Promotes consistently successful connections (lower weight)
  - Recommends new connections based on semantic compatibility
  - Suggests nodes to remove if they are never used
  - Updates edge weights to reflect real performance
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# Thresholds
MIN_RUNS_TO_JUDGE = 5          # Don't act on connections with < N runs
CRITICAL_SUCCESS_RATE = 0.3    # Below this → prune candidate
EXCELLENT_SUCCESS_RATE = 0.85  # Above this → promote candidate
UNUSED_NODE_RUNS = 0           # Nodes with 0 runs are unused candidates
SEMANTIC_SUGGEST_THRESHOLD = 0.20  # Semantic score threshold for new connections


class OptimizationAction:
    """Represents a single optimization recommendation."""

    PRUNE_EDGE = "prune_edge"
    PROMOTE_EDGE = "promote_edge"
    SUGGEST_EDGE = "suggest_edge"
    REMOVE_NODE = "remove_node"
    UPDATE_WEIGHT = "update_weight"

    def __init__(self, action_type: str, **kwargs):
        self.action_type = action_type
        self.data = kwargs
        self.created_at = datetime.utcnow().isoformat()
        self.applied: bool = False

    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "data": self.data,
            "created_at": self.created_at,
            "applied": self.applied,
        }


class OptimizationReport:
    """Result of a single optimization cycle."""

    def __init__(self):
        self.actions: List[OptimizationAction] = []
        self.generated_at: str = datetime.utcnow().isoformat()
        self.applied_count: int = 0
        self.summary: dict = {}

    def add(self, action: OptimizationAction):
        self.actions.append(action)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "total_actions": len(self.actions),
            "applied_count": self.applied_count,
            "summary": self.summary,
            "actions": [a.to_dict() for a in self.actions],
        }


class OptimizationEngine:
    """
    Phase 3 Optimization Engine.
    Analyses the graph periodically and returns an OptimizationReport
    with concrete actions to improve topology.
    Actions are generated but NOT auto-applied — the mesh or user decides.
    Call `apply_report(report, mesh)` to execute them.
    """

    def __init__(self, graph=None, scoring_engine=None,
                 memory_engine=None, semantic_matcher=None):
        self._graph = graph
        self._scoring = scoring_engine
        self._memory = memory_engine
        self._semantic = semantic_matcher
        self._last_report: Optional[OptimizationReport] = None
        self._run_count: int = 0
        self._knowledge = None   # KnowledgeStore — injected via set_knowledge_store()
        logger.info("OptimizationEngine initialised (Phase 3)")

    def set_knowledge_store(self, ks) -> None:
        """Inject the KnowledgeStore to persist graph metrics and optimization history."""
        self._knowledge = ks
        logger.info("OptimizationEngine: KnowledgeStore connected")

    def set_components(self, graph=None, scoring=None, memory=None, semantic=None):
        if graph:
            self._graph = graph
        if scoring:
            self._scoring = scoring
        if memory:
            self._memory = memory
        if semantic:
            self._semantic = semantic

    # ── Main analysis API ──────────────────────────────────────────────────

    def analyze(self) -> OptimizationReport:
        """
        Run a full optimization analysis and return the report.
        Does NOT modify anything.
        """
        self._run_count += 1
        report = OptimizationReport()
        logger.info(f"OptimizationEngine: running analysis #{self._run_count}")

        # 1. Analyse connection quality → prune / promote
        self._analyze_connections(report)

        # 2. Recommend new connections via semantic compatibility
        self._suggest_new_connections(report)

        # 3. Identify unused nodes
        self._identify_unused_nodes(report)

        # 4. Update edge weights
        self._suggest_weight_updates(report)

        # Summary
        action_counts: Dict[str, int] = {}
        for a in report.actions:
            action_counts[a.action_type] = action_counts.get(a.action_type, 0) + 1
        report.summary = {
            "analysis_run": self._run_count,
            "action_counts": action_counts,
            "total_recommendations": len(report.actions),
        }

        self._last_report = report
        logger.info(
            f"OptimizationEngine: analysis complete, "
            f"{len(report.actions)} actions recommended"
        )

        # ── Knowledge layer: persist metrics to JSON files ──────────────
        if self._knowledge:
            try:
                # Graph statistics
                if self._graph:
                    adj = self._graph._adjacency
                    n_nodes = len(adj)
                    n_edges = sum(len(e) for e in adj.values())
                    avg_deg = n_edges / max(n_nodes, 1)
                    max_edges = n_nodes * (n_nodes - 1) if n_nodes > 1 else 1
                    density = n_edges / max_edges if max_edges > 0 else 0.0
                    self._knowledge.update_graph_statistics(
                        total_nodes=n_nodes, total_edges=n_edges,
                        avg_degree=avg_deg, density=density,
                    )

                # Node + route rankings
                if self._memory:
                    self._knowledge.update_node_rankings(self._memory)
                    self._knowledge.update_route_rankings(self._memory)

                # Connection score snapshot
                if self._scoring:
                    self._knowledge.update_connection_scores(self._scoring)

                # Record this optimization run
                self._knowledge.record_optimization_run(report.to_dict())

                # Health trend snapshot
                avg_score = 0.0
                if self._scoring:
                    sc_summary = self._scoring.summary()
                    avg_score = sc_summary.get("avg_connection_score", 0.0)
                n_nodes = len(self._graph._adjacency) if self._graph else 0
                n_edges = sum(len(e) for e in self._graph._adjacency.values()) if self._graph else 0
                self._knowledge.append_health_snapshot(avg_score, n_nodes, n_edges)

            except Exception as ke:
                logger.warning(f"OptimizationEngine: knowledge write failed: {ke}")

        return report

    def apply_report(self, report: OptimizationReport, mesh) -> int:
        """
        Apply the actions in a report to the mesh.
        Returns count of successfully applied actions.
        """
        applied = 0
        for action in report.actions:
            try:
                if self._apply_action(action, mesh):
                    action.applied = True
                    applied += 1
            except Exception as e:
                logger.error(f"OptimizationEngine: failed to apply {action.action_type}: {e}")
        report.applied_count = applied
        logger.info(f"OptimizationEngine: applied {applied}/{len(report.actions)} actions")
        return applied

    def _apply_action(self, action: OptimizationAction, mesh) -> bool:
        t = action.action_type
        d = action.data

        if t == OptimizationAction.PRUNE_EDGE:
            ok = mesh.graph.remove_edge(d["source_id"], d["target_id"])
            if ok and mesh.db:
                mesh.db.delete_connection(d["source_id"], d["target_id"])
            logger.info(f"Pruned edge {d['source_id'][:8]}->{d['target_id'][:8]}")
            return ok

        elif t == OptimizationAction.SUGGEST_EDGE:
            try:
                edge = mesh.graph.add_edge(
                    d["source_id"], d["target_id"],
                    weight=d.get("weight", 1.0),
                    label=d.get("label", "ai_suggested"),
                )
                if mesh.db:
                    mesh.db.upsert_connection(
                        d["source_id"], d["target_id"],
                        d.get("weight", 1.0), "ai_suggested"
                    )
                logger.info(f"Added AI-suggested edge {d['source_id'][:8]}->{d['target_id'][:8]}")
                return edge is not None
            except Exception:
                return False

        elif t == OptimizationAction.UPDATE_WEIGHT:
            for edge in mesh.graph._adjacency.get(d["source_id"], []):
                if edge.target_id == d["target_id"]:
                    edge.weight = d["new_weight"]
                    if mesh.db:
                        mesh.db.upsert_connection(
                            d["source_id"], d["target_id"], d["new_weight"]
                        )
                    return True
            return False

        elif t == OptimizationAction.PROMOTE_EDGE:
            # Promote by reducing weight (makes BFS prefer it)
            for edge in mesh.graph._adjacency.get(d["source_id"], []):
                if edge.target_id == d["target_id"]:
                    edge.weight = max(0.1, edge.weight * 0.7)
                    return True
            return False

        return False

    # ── Analysis steps ─────────────────────────────────────────────────────

    def _analyze_connections(self, report: OptimizationReport):
        if not self._scoring or not self._graph:
            return

        all_scores = self._scoring.list_scores()
        for s in all_scores:
            src = s["source_id"]
            tgt = s["target_id"]
            runs = s["total_runs"]
            sr = s["success_rate"]

            if runs < MIN_RUNS_TO_JUDGE:
                continue

            if sr < CRITICAL_SUCCESS_RATE:
                report.add(OptimizationAction(
                    OptimizationAction.PRUNE_EDGE,
                    source_id=src,
                    target_id=tgt,
                    reason=f"success_rate={sr:.2%} below threshold",
                    success_rate=sr,
                    runs=runs,
                ))
                logger.debug(f"Optimization: prune candidate {src[:8]}->{tgt[:8]} sr={sr:.2%}")

            elif sr >= EXCELLENT_SUCCESS_RATE:
                report.add(OptimizationAction(
                    OptimizationAction.PROMOTE_EDGE,
                    source_id=src,
                    target_id=tgt,
                    reason=f"success_rate={sr:.2%} excellent",
                    success_rate=sr,
                    runs=runs,
                ))

    def _suggest_new_connections(self, report: OptimizationReport):
        if not self._semantic or not self._graph:
            return

        # Get existing edges
        existing = [
            (src, e.target_id)
            for src, edges in self._graph._adjacency.items()
            for e in edges
        ]

        suggestions = self._semantic.suggest_new_connections(
            existing, threshold=SEMANTIC_SUGGEST_THRESHOLD
        )

        for s in suggestions[:5]:  # Limit to 5 suggestions per cycle
            report.add(OptimizationAction(
                OptimizationAction.SUGGEST_EDGE,
                source_id=s["source_id"],
                target_id=s["target_id"],
                weight=max(0.5, 1.5 - s["semantic_score"]),  # Better match = lower weight
                label="ai_semantic_suggestion",
                semantic_score=s["semantic_score"],
                reason=s["reason"],
            ))

    def _identify_unused_nodes(self, report: OptimizationReport):
        if not self._memory or not self._graph:
            return

        node_ids = list(self._graph._adjacency.keys())
        for node_id in node_ids:
            nm = self._memory.get_node_memory(node_id)
            if nm is None or nm.executions == 0:
                report.add(OptimizationAction(
                    OptimizationAction.REMOVE_NODE,
                    node_id=node_id,
                    reason="Node has never been executed",
                    executions=0,
                ))

    def _suggest_weight_updates(self, report: OptimizationReport):
        """Align edge weights with actual performance data."""
        if not self._scoring or not self._graph:
            return

        for src, edges in self._graph._adjacency.items():
            for edge in edges:
                tgt = edge.target_id
                cs = self._scoring.get_score(src, tgt)
                if cs.total_runs < MIN_RUNS_TO_JUDGE:
                    continue
                # Performance-based weight: excellent → 0.5, terrible → 3.0
                ideal_weight = round(3.5 - (cs.connection_score / 100.0) * 3.0, 2)
                if abs(ideal_weight - edge.weight) > 0.3:
                    report.add(OptimizationAction(
                        OptimizationAction.UPDATE_WEIGHT,
                        source_id=src,
                        target_id=tgt,
                        old_weight=edge.weight,
                        new_weight=ideal_weight,
                        reason=f"connection_score={cs.connection_score:.1f}",
                    ))

    # ── Reporting ──────────────────────────────────────────────────────────

    def last_report(self) -> Optional[dict]:
        return self._last_report.to_dict() if self._last_report else None

    def quick_health_check(self) -> dict:
        """Fast health summary without full analysis."""
        result = {
            "graph_nodes": len(self._graph._adjacency) if self._graph else 0,
            "graph_edges": sum(len(e) for e in self._graph._adjacency.values()) if self._graph else 0,
            "optimization_runs": self._run_count,
        }
        if self._scoring:
            result["scoring_summary"] = self._scoring.summary()
        if self._memory:
            result["memory_summary"] = self._memory.summary()
        return result

    def __repr__(self):
        return f"<OptimizationEngine runs={self._run_count}>"
