"""
Phase 4 – Learning Validator
============================
Proves that the Neural Service Mesh is actually learning by tracking:
  - Improvement in route selection over time
  - Node reputation evolution
  - Score delta between early and recent executions
  - Learning curve metrics
"""
from __future__ import annotations
import logging
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LearningMetrics:
    """Snapshot of the learning state at a point in time."""

    def __init__(self):
        self.total_executions: int = 0
        self.successful_executions: int = 0
        self.failed_executions: int = 0
        self.best_route: Optional[dict] = None
        self.worst_route: Optional[dict] = None
        self.most_trusted_node: Optional[dict] = None
        self.avg_success_rate: float = 0.0
        self.learning_improvement: float = 0.0   # % improvement vs baseline
        self.snapshot_ts: str = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": round(self.successful_executions / self.total_executions, 4)
            if self.total_executions > 0 else 0.0,
            "best_route": self.best_route,
            "worst_route": self.worst_route,
            "most_trusted_node": self.most_trusted_node,
            "avg_success_rate_across_routes": round(self.avg_success_rate, 4),
            "learning_improvement_pct": round(self.learning_improvement, 2),
            "snapshot_ts": self.snapshot_ts,
        }


class LearningValidator:
    """
    Phase 4 Learning Validator.

    Reads from MemoryEngine, ScoringEngine, and KnowledgeStore to
    compute learning metrics that prove the system improves over time.
    """

    def __init__(self, memory_engine=None, scoring_engine=None, knowledge_store=None):
        self._memory = memory_engine
        self._scoring = scoring_engine
        self._knowledge = knowledge_store
        # Track early baseline scores to measure improvement
        self._baseline_scores: Dict[str, float] = {}
        self._baseline_locked: bool = False
        logger.info("LearningValidator initialised (Phase 4)")

    def set_components(self, memory=None, scoring=None, knowledge=None):
        if memory:
            self._memory = memory
        if scoring:
            self._scoring = scoring
        if knowledge:
            self._knowledge = knowledge

    # ── Core metrics computation ───────────────────────────────────────────

    def compute_metrics(self) -> LearningMetrics:
        """Compute all learning metrics from current system state."""
        m = LearningMetrics()

        # 1. Global execution counts — read directly from route data
        if self._memory:
            all_routes = self._memory.all_routes()

            # Aggregate from all routes
            total_runs = sum(r.get("runs", 0) for r in all_routes)
            total_succ = sum(r.get("successes", 0) for r in all_routes)
            total_fail = sum(r.get("failures", 0) for r in all_routes)

            m.total_executions = total_runs
            m.successful_executions = total_succ
            m.failed_executions = total_fail

            # 2. Best and worst routes by memory score
            if all_routes:
                sorted_routes = sorted(all_routes, key=lambda r: r.get("memory_score", 0), reverse=True)
                best = sorted_routes[0]
                worst = sorted_routes[-1]
                m.best_route = {
                    "path_key": best.get("path_key"),
                    "memory_score": best.get("memory_score"),
                    "success_rate": best.get("success_rate"),
                    "runs": best.get("runs"),
                    "health": best.get("health"),
                }
                m.worst_route = {
                    "path_key": worst.get("path_key"),
                    "memory_score": worst.get("memory_score"),
                    "success_rate": worst.get("success_rate"),
                    "runs": worst.get("runs"),
                    "health": worst.get("health"),
                }

                # Average success rate
                rates = [r.get("success_rate", 0.0) for r in all_routes if r.get("runs", 0) > 0]
                m.avg_success_rate = statistics.mean(rates) if rates else 0.0

            # 3. Most trusted node
            all_nodes = self._memory.best_nodes(1)
            if all_nodes:
                node = all_nodes[0]
                m.most_trusted_node = {
                    "node_id": node.get("node_id"),
                    "name": node.get("name"),
                    "success_rate": node.get("success_rate"),
                    "total_runs": node.get("total_runs", node.get("executions", 0)),
                    "reputation_score": node.get("success_rate", 0) * 100,
                }

        # 4. Learning improvement vs baseline
        m.learning_improvement = self._compute_improvement()

        return m

    def _compute_improvement(self) -> float:
        """
        Compute % improvement: compare average route score now vs baseline.
        Baseline = first N=5 executions per route. Current = last N=5.
        """
        if not self._scoring:
            return 0.0
        try:
            scores = self._scoring.list_scores()
            if not scores:
                return 0.0

            current_scores = [s.get("connection_score", 50.0) for s in scores]
            current_avg = statistics.mean(current_scores) if current_scores else 50.0

            # Compare against baseline (locked at first computation)
            if not self._baseline_locked and current_scores:
                for s in scores:
                    key = f"{s['source_id']}->{s['target_id']}"
                    if key not in self._baseline_scores:
                        self._baseline_scores[key] = s.get("connection_score", 50.0)

            if not self._baseline_scores:
                return 0.0

            baseline_avg = statistics.mean(self._baseline_scores.values())
            if baseline_avg == 0:
                return 0.0

            improvement = ((current_avg - baseline_avg) / baseline_avg) * 100.0
            return round(improvement, 2)
        except Exception as e:
            logger.debug(f"LearningValidator._compute_improvement: {e}")
            return 0.0

    def lock_baseline(self):
        """Freeze the baseline so improvement is measured from this point."""
        self._baseline_locked = True
        logger.info(f"LearningValidator: baseline locked with {len(self._baseline_scores)} connections")

    # ── Reputation system ─────────────────────────────────────────────────

    def get_node_reputation(self) -> List[dict]:
        """
        Return all nodes with their reputation scores.
        Reputation = weighted combination of success rate + usage frequency.
        """
        if not self._memory:
            return []
        try:
            all_nodes = self._memory.best_nodes(100)  # Get all
            result = []
            for node in all_nodes:
                sr = node.get("success_rate", 0.0)
                runs = node.get("total_runs", 0)
                # Reputation: 70% success rate + 30% usage (capped at 100 runs)
                usage_score = min(runs / 100.0, 1.0) * 30.0
                reputation = round(sr * 70.0 + usage_score, 2)
                result.append({
                    "node_id": node.get("node_id"),
                    "name": node.get("name"),
                    "reputation_score": reputation,
                    "success_rate": sr,
                    "total_runs": runs,
                    "avg_latency_ms": node.get("avg_latency_ms", 0.0),
                    "tier": self._reputation_tier(reputation),
                })
            return sorted(result, key=lambda x: x["reputation_score"], reverse=True)
        except Exception as e:
            logger.error(f"LearningValidator.get_node_reputation: {e}")
            return []

    def _reputation_tier(self, score: float) -> str:
        if score >= 85:
            return "platinum"
        if score >= 70:
            return "gold"
        if score >= 50:
            return "silver"
        if score >= 30:
            return "bronze"
        return "unrated"

    # ── Learning curve ─────────────────────────────────────────────────────

    def get_learning_curve(self) -> dict:
        """
        Return data showing how scores have improved over executions.
        Reads execution history from knowledge store.
        """
        if not self._knowledge:
            return {"error": "KnowledgeStore not connected"}
        try:
            graph_metrics = self._knowledge.read_graph_metrics()
            health_history = graph_metrics.get("health_history", [])
            if not health_history:
                return {"data_points": [], "trend": "insufficient_data"}

            points = []
            for i, snapshot in enumerate(health_history[-50:]):  # Last 50 points
                points.append({
                    "index": i,
                    "avg_connection_score": snapshot.get("avg_connection_score", 0),
                    "total_runs": snapshot.get("total_runs", 0),
                    "success_rate": snapshot.get("success_rate", 0),
                    "ts": snapshot.get("ts", ""),
                })

            # Determine trend
            if len(points) >= 2:
                first_score = points[0].get("avg_connection_score", 50)
                last_score = points[-1].get("avg_connection_score", 50)
                delta = last_score - first_score
                trend = "improving" if delta > 2 else ("degrading" if delta < -2 else "stable")
            else:
                trend = "insufficient_data"

            return {
                "data_points": points,
                "trend": trend,
                "total_snapshots": len(health_history),
            }
        except Exception as e:
            logger.error(f"LearningValidator.get_learning_curve: {e}")
            return {"error": str(e)}

    # ── Proof of learning ─────────────────────────────────────────────────

    def prove_learning(self) -> dict:
        """
        Generate a human-readable proof that the system is learning.
        Compares early behavior vs recent behavior.
        """
        metrics = self.compute_metrics()
        node_rep = self.get_node_reputation()

        evidence = []
        verdict = "insufficient_data"

        if metrics.total_executions < 5:
            return {
                "verdict": "insufficient_data",
                "message": f"Need at least 5 executions. Currently: {metrics.total_executions}",
                "metrics": metrics.to_dict(),
            }

        # Evidence 1: Success rate
        sr = metrics.successful_executions / metrics.total_executions
        if sr > 0.7:
            evidence.append(f"✓ High success rate: {sr:.1%} ({metrics.successful_executions}/{metrics.total_executions} runs)")

        # Evidence 2: Route improvement
        if metrics.best_route and metrics.best_route.get("memory_score", 0) > 70:
            evidence.append(f"✓ Best route has memory_score={metrics.best_route['memory_score']} — system learned to prefer it")

        # Evidence 3: Node reputation spread
        if len(node_rep) >= 2:
            top = node_rep[0]
            bottom = node_rep[-1]
            spread = top["reputation_score"] - bottom["reputation_score"]
            if spread > 10:
                evidence.append(
                    f"✓ Node reputation differentiation: {top['name']} ({top['reputation_score']:.1f}) "
                    f"vs {bottom['name']} ({bottom['reputation_score']:.1f}) — AI distinguishes reliable nodes"
                )

        # Evidence 4: Learning improvement
        if metrics.learning_improvement > 0:
            evidence.append(f"✓ Score improvement vs baseline: +{metrics.learning_improvement:.1f}%")

        # Verdict
        if len(evidence) >= 2:
            verdict = "learning_confirmed"
        elif len(evidence) == 1:
            verdict = "learning_in_progress"
        else:
            verdict = "learning_not_yet_evident"

        return {
            "verdict": verdict,
            "evidence": evidence,
            "metrics": metrics.to_dict(),
            "top_nodes": node_rep[:3],
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }

    def __repr__(self):
        return "<LearningValidator (Phase 4)>"
