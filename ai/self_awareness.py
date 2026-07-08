"""
Phase 7 – Self-Awareness Engine
==================================
The system's introspective layer — answers:
  • How many nodes do I have?
  • What is my weakest point?
  • What fails most often?
  • What succeeds most?
  • What are my current objectives?

File: ai/self_awareness.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SystemAwarenessReport:
    """Full self-awareness snapshot of the current system state."""

    def __init__(self):
        self.generated_at = datetime.now(timezone.utc).isoformat()
        self.node_count: int = 0
        self.edge_count: int = 0
        self.active_agents: int = 0
        self.weakest_node: Optional[dict] = None
        self.strongest_node: Optional[dict] = None
        self.most_failing_transition: Optional[dict] = None
        self.most_successful_transition: Optional[dict] = None
        self.current_objectives: List[str] = []
        self.known_capabilities: List[str] = []
        self.known_failures: List[dict] = []
        self.system_health_score: float = 0.0
        self.phase7_readiness: float = 0.0
        self.insights: List[str] = []

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "active_agents": self.active_agents,
            "weakest_node": self.weakest_node,
            "strongest_node": self.strongest_node,
            "most_failing_transition": self.most_failing_transition,
            "most_successful_transition": self.most_successful_transition,
            "current_objectives": self.current_objectives,
            "known_capabilities": self.known_capabilities,
            "known_failures": self.known_failures,
            "system_health_score": round(self.system_health_score, 3),
            "phase7_readiness": round(self.phase7_readiness, 3),
            "insights": self.insights,
        }


class SelfAwarenessEngine:
    """
    Phase 7: Provides the system with introspective awareness of its own state.

    Reads from:
      - NodeRegistry          (node count, types)
      - ScoringEngine         (weak/strong connections)
      - MemoryEngine          (failure patterns)
      - AgentFactory          (active agents)
      - EnvironmentModel      (world model)
      - ObjectivesEngine      (current goals)
    """

    def __init__(
        self,
        registry=None,
        graph=None,
        scoring_engine=None,
        memory_engine=None,
        agent_factory=None,
        knowledge_store=None,
        environment_model=None,
        objectives_engine=None,
    ):
        self._registry = registry
        self._graph = graph
        self._scoring = scoring_engine
        self._memory = memory_engine
        self._agent_factory = agent_factory
        self._knowledge = knowledge_store
        self._env_model = environment_model
        self._objectives = objectives_engine
        self._report_count = 0
        self._last_report: Optional[SystemAwarenessReport] = None

    def introspect(self) -> SystemAwarenessReport:
        """Generate a full self-awareness report."""
        report = SystemAwarenessReport()

        # ── Node & graph stats ─────────────────────────────────────────────
        try:
            report.node_count = self._registry.count() if self._registry else 0
        except Exception:
            report.node_count = 0

        try:
            stats = self._graph.stats() if self._graph else {}
            report.edge_count = stats.get("edges", 0)
        except Exception:
            report.edge_count = 0

        # ── Agent count ────────────────────────────────────────────────────
        try:
            if self._agent_factory:
                af_summary = self._agent_factory.summary()
                report.active_agents = af_summary.get("active_agents", 0)
        except Exception:
            pass

        # ── Weakest / strongest nodes (via scoring) ────────────────────────
        try:
            if self._scoring:
                scores = self._scoring.list_scores()
                if scores:
                    worst = min(scores, key=lambda s: s.get("connection_score", 50))
                    best = max(scores, key=lambda s: s.get("connection_score", 50))
                    report.weakest_node = {
                        "connection": f"{worst.get('source_id','?')[:8]} → {worst.get('target_id','?')[:8]}",
                        "score": worst.get("connection_score"),
                        "success_rate": worst.get("success_rate"),
                        "total_runs": worst.get("total_runs"),
                    }
                    report.strongest_node = {
                        "connection": f"{best.get('source_id','?')[:8]} → {best.get('target_id','?')[:8]}",
                        "score": best.get("connection_score"),
                        "success_rate": best.get("success_rate"),
                        "total_runs": best.get("total_runs"),
                    }
        except Exception as exc:
            logger.debug(f"[SelfAwareness] scoring read error: {exc}")

        # ── Most failing / succeeding routes (via memory) ──────────────────
        try:
            if self._memory:
                routes = self._memory.all_routes()
                if routes:
                    routes_with_runs = [r for r in routes if r.get("runs", 0) > 0]
                    if routes_with_runs:
                        worst_route = min(routes_with_runs, key=lambda r: r.get("success_rate", 1.0))
                        best_route = max(routes_with_runs, key=lambda r: r.get("success_rate", 0.0))
                        report.most_failing_transition = {
                            "path": worst_route.get("path_key"),
                            "success_rate": worst_route.get("success_rate"),
                            "runs": worst_route.get("runs"),
                            "health": worst_route.get("health"),
                        }
                        report.most_successful_transition = {
                            "path": best_route.get("path_key"),
                            "success_rate": best_route.get("success_rate"),
                            "runs": best_route.get("runs"),
                            "health": best_route.get("health"),
                        }
        except Exception as exc:
            logger.debug(f"[SelfAwareness] memory read error: {exc}")

        # ── Current objectives ─────────────────────────────────────────────
        try:
            if self._objectives:
                report.current_objectives = self._objectives.active_goals()
        except Exception:
            report.current_objectives = []

        # ── Known capabilities from world model ────────────────────────────
        try:
            if self._env_model:
                report.known_capabilities = self._env_model.get_known_capabilities()[:20]
                failures = self._env_model.get_known_failures()
                report.known_failures = [
                    {"key": k, **v}
                    for k, v in sorted(failures.items(),
                                       key=lambda x: x[1].get("count", 0), reverse=True)
                ][:10]
        except Exception:
            pass

        # ── System health score ────────────────────────────────────────────
        report.system_health_score = self._compute_health(report)
        report.phase7_readiness = self._compute_readiness(report)

        # ── Insights ───────────────────────────────────────────────────────
        report.insights = self._generate_insights(report)

        self._last_report = report
        self._report_count += 1
        logger.info(f"[SelfAwareness] introspection #{self._report_count} complete "
                    f"(health={report.system_health_score:.2f})")
        return report

    def _compute_health(self, report: SystemAwarenessReport) -> float:
        score = 0.5  # baseline
        if report.node_count > 0:
            score += 0.1
        if report.node_count > 5:
            score += 0.1
        if report.weakest_node:
            sr = report.weakest_node.get("success_rate") or 0.5
            score += 0.15 * sr
        if report.strongest_node:
            sr = report.strongest_node.get("success_rate") or 0.5
            score += 0.15 * sr
        if report.active_agents > 0:
            score += 0.05
        failed = len([f for f in report.known_failures if f.get("severity") == "critical"])
        score -= 0.05 * min(failed, 3)
        return max(0.0, min(1.0, score))

    def _compute_readiness(self, report: SystemAwarenessReport) -> float:
        """How ready is the system for autonomous Phase 7 evolution?"""
        checks = [
            report.node_count >= 3,
            report.edge_count >= 2,
            len(report.current_objectives) > 0,
            report.system_health_score > 0.4,
            len(report.known_capabilities) > 0,
        ]
        return sum(checks) / len(checks)

    def _generate_insights(self, report: SystemAwarenessReport) -> List[str]:
        insights = []
        if report.node_count == 0:
            insights.append("No nodes registered — system is empty")
        elif report.node_count < 3:
            insights.append(f"Only {report.node_count} nodes — mesh is sparse")
        if report.weakest_node:
            sr = report.weakest_node.get("success_rate") or 1.0
            if sr < 0.5:
                insights.append(
                    f"Weak link detected: {report.weakest_node['connection']} "
                    f"(success rate {sr:.0%})"
                )
        if report.most_failing_transition:
            h = report.most_failing_transition.get("health", "")
            if h in ("critical", "poor"):
                insights.append(
                    f"Critical route failure: {report.most_failing_transition['path']} ({h})"
                )
        if report.active_agents == 0:
            insights.append("No active agents — consider spawning monitor/optimizer agents")
        if len(report.known_failures) > 5:
            insights.append(
                f"{len(report.known_failures)} recurring failure patterns detected"
            )
        if not insights:
            insights.append("System appears healthy — ready for evolution cycle")
        return insights

    def get_last_report(self) -> Optional[dict]:
        return self._last_report.to_dict() if self._last_report else None

    def summary(self) -> dict:
        return {
            "report_count": self._report_count,
            "last_report_at": self._last_report.generated_at if self._last_report else None,
            "last_health_score": self._last_report.system_health_score if self._last_report else None,
            "last_phase7_readiness": self._last_report.phase7_readiness if self._last_report else None,
        }
