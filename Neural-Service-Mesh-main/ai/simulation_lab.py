"""
Phase 6 – Simulation Lab
==========================
Runs large-scale scenario simulations to identify the optimal execution
plan before committing to a real run.

Workflow:
  1. Generate N candidate plans for a goal
  2. Simulate each plan against synthetic workloads
  3. Score by success rate, latency, cost, and quality
  4. Return the best plan + full comparison report

Usage:
  lab = SimulationLab(registry, graph, routing_engine)
  report = lab.run(goal="translate document", n_plans=1000, data={...})
  best = report.best_plan
"""
from __future__ import annotations

import random
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SimulatedPlan:
    """One candidate execution plan with its simulation results."""

    def __init__(self, plan_id: str, path: List[str], node_names: List[str]):
        self.plan_id = plan_id
        self.path = path
        self.node_names = node_names
        # Simulation metrics (set after run)
        self.simulated_latency_ms: float = 0.0
        self.simulated_success_rate: float = 0.0
        self.simulated_cost: float = 0.0
        self.simulated_quality: float = 0.0
        self.composite_score: float = 0.0
        self.simulation_runs: int = 0

    def compute_composite(
        self,
        w_latency: float = 0.25,
        w_success: float = 0.35,
        w_cost: float = 0.15,
        w_quality: float = 0.25,
    ):
        latency_norm = max(0.0, 1.0 - self.simulated_latency_ms / 3000.0)
        self.composite_score = (
            w_latency * latency_norm
            + w_success * self.simulated_success_rate
            + w_cost * (1.0 - min(1.0, self.simulated_cost / 100.0))
            + w_quality * self.simulated_quality
        )

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "path": self.path,
            "node_names": self.node_names,
            "simulation_runs": self.simulation_runs,
            "simulated_latency_ms": round(self.simulated_latency_ms, 2),
            "simulated_success_rate": round(self.simulated_success_rate, 4),
            "simulated_cost": round(self.simulated_cost, 4),
            "simulated_quality": round(self.simulated_quality, 4),
            "composite_score": round(self.composite_score, 4),
        }


class SimulationReport:
    """Full report from a SimulationLab run."""

    def __init__(self, lab_id: str, goal: str, n_plans: int):
        self.lab_id = lab_id
        self.goal = goal
        self.n_plans_requested = n_plans
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.plans: List[SimulatedPlan] = []
        self.best_plan: Optional[SimulatedPlan] = None

    def complete(self):
        self.finished_at = datetime.now(timezone.utc).isoformat()
        if self.plans:
            self.best_plan = max(self.plans, key=lambda p: p.composite_score)

    def to_dict(self) -> dict:
        return {
            "lab_id": self.lab_id,
            "goal": self.goal,
            "n_plans_requested": self.n_plans_requested,
            "n_plans_simulated": len(self.plans),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "best_plan": self.best_plan.to_dict() if self.best_plan else None,
            "all_plans": [p.to_dict() for p in self.plans],
        }


class SimulationLab:
    """
    Phase 6: Large-Scale Simulation Laboratory.

    Generates and evaluates thousands of execution plans, returning the
    statistically best one for real execution.
    """

    def __init__(
        self,
        registry=None,
        graph=None,
        routing_engine=None,
        scoring_engine=None,
        memory_engine=None,
        knowledge_store=None,
    ):
        self._registry = registry
        self._graph = graph
        self._routing = routing_engine
        self._scoring = scoring_engine
        self._memory = memory_engine
        self._knowledge = knowledge_store
        self._run_history: List[SimulationReport] = []
        logger.info("SimulationLab initialised (Phase 6)")

    # ── Main API ──────────────────────────────────────────────────────────

    def run(
        self,
        goal: str,
        data: dict,
        n_plans: int = 1000,
        weights: Optional[dict] = None,
    ) -> SimulationReport:
        """
        Simulate `n_plans` candidate plans and return the best one.

        Args:
            goal:    Natural-language goal.
            data:    Input data for the pipeline.
            n_plans: Number of candidate plans to generate and simulate.
            weights: Score weights {"latency", "success", "cost", "quality"}.
        """
        lab_id = f"lab_{str(uuid.uuid4())[:8]}"
        report = SimulationReport(lab_id, goal, n_plans)

        weights = weights or {}
        w_lat  = weights.get("latency", 0.25)
        w_suc  = weights.get("success", 0.35)
        w_cost = weights.get("cost", 0.15)
        w_qual = weights.get("quality", 0.25)

        # 2. Generate candidate paths
        candidate_paths = self._generate_candidates(goal, n_plans)

        # 3. Simulate each candidate
        for path, names in candidate_paths:
            plan = SimulatedPlan(
                plan_id=f"plan_{str(uuid.uuid4())[:6]}",
                path=path,
                node_names=names,
            )
            self._simulate_plan(plan, data)
            plan.compute_composite(w_lat, w_suc, w_cost, w_qual)
            report.plans.append(plan)

        report.complete()
        self._run_history.append(report)

        logger.info(
            f"SimulationLab {lab_id}: simulated {len(report.plans)} plans, "
            f"best score={report.best_plan.composite_score:.4f if report.best_plan else 'n/a'}"
        )
        return report

    # ── Candidate generation ──────────────────────────────────────────────

    def _generate_candidates(
        self, goal: str, n_plans: int
    ) -> List[tuple]:
        """
        Generate candidate (path_ids, path_names) tuples.

        In production this would query the RoutingEngine for alternatives.
        Here we produce realistic synthetic variants for demonstration.
        """
        # Try to get real paths from the routing engine
        real_paths: List[tuple] = []
        if self._routing and self._registry:
            try:
                nodes = self._registry.list_all()
                if len(nodes) >= 2:
                    for _ in range(min(10, n_plans)):
                        sample = random.sample(nodes, min(3, len(nodes)))
                        ids = [n.node_id for n in sample]
                        names = [n.name for n in sample]
                        real_paths.append((ids, names))
            except Exception:
                pass

        # Pad with synthetic paths up to n_plans
        synthetic_roles = [
            "InputNode", "ProcessorNode", "TranslatorNode",
            "ValidatorNode", "OutputNode", "FilterNode",
        ]
        while len(real_paths) < n_plans:
            length = random.randint(2, 5)
            names = random.choices(synthetic_roles, k=length)
            ids = [f"synthetic_{str(uuid.uuid4())[:8]}" for _ in names]
            real_paths.append((ids, names))

        return real_paths[:n_plans]

    # ── Per-plan simulation ───────────────────────────────────────────────

    def _simulate_plan(self, plan: SimulatedPlan, data: dict, runs: int = 10):
        """
        Monte-Carlo simulate a single plan over `runs` trials.
        Incorporates historical scoring data when available.
        """
        latencies: List[float] = []
        successes: List[float] = []
        costs: List[float] = []
        qualities: List[float] = []

        # Base metrics from scoring engine if available
        base_sr    = self._get_base_success_rate(plan.path)
        base_lat   = self._get_base_latency(plan.path)
        path_len   = len(plan.path)

        for _ in range(runs):
            # Stochastic variation
            lat  = max(10.0, base_lat * random.uniform(0.6, 1.6) + path_len * random.uniform(5, 30))
            suc  = min(1.0, max(0.0, base_sr + random.gauss(0, 0.05)))
            cost = path_len * random.uniform(0.5, 3.0)
            qual = min(1.0, max(0.0, suc * random.uniform(0.8, 1.1)))

            latencies.append(lat)
            successes.append(1.0 if random.random() < suc else 0.0)
            costs.append(cost)
            qualities.append(qual)

        plan.simulated_latency_ms  = sum(latencies) / runs
        plan.simulated_success_rate = sum(successes) / runs
        plan.simulated_cost         = sum(costs) / runs
        plan.simulated_quality      = sum(qualities) / runs
        plan.simulation_runs        = runs

    def _get_base_success_rate(self, path: List[str]) -> float:
        if not self._scoring or not path:
            return 0.85
        try:
            scores = self._scoring.list_scores()
            relevant = [
                s for s in scores
                if s.get("source_id") in path or s.get("target_id") in path
            ]
            if relevant:
                return sum(s.get("success_rate", 0.85) for s in relevant) / len(relevant)
        except Exception:
            pass
        return 0.85

    def _get_base_latency(self, path: List[str]) -> float:
        if not self._scoring or not path:
            return 100.0
        try:
            scores = self._scoring.list_scores()
            relevant = [
                s for s in scores
                if s.get("source_id") in path or s.get("target_id") in path
            ]
            if relevant:
                return sum(s.get("avg_latency_ms", 100) for s in relevant) / len(relevant)
        except Exception:
            pass
        return 100.0

    # ── History & summary ─────────────────────────────────────────────────

    def history(self, limit: int = 10) -> List[dict]:
        return [r.to_dict() for r in self._run_history[-limit:]]

    def summary(self) -> dict:
        total = len(self._run_history)
        return {
            "total_lab_runs": total,
            "total_plans_simulated": sum(len(r.plans) for r in self._run_history),
            "avg_best_score": (
                sum(r.best_plan.composite_score for r in self._run_history if r.best_plan)
                / max(1, total)
            ),
        }
