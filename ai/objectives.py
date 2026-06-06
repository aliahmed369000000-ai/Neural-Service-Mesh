"""
Phase 7 – Strategic Objectives Engine
=======================================
Manages long-term system objectives that guide autonomous decisions.

File: ai/objectives.py

Goals example:
  - Increase success rate
  - Reduce latency
  - Expand capabilities
  - Reduce failures

The engine:
  - Stores and prioritises goals
  - Tracks progress toward each goal
  - Provides goal-aligned recommendations to the EvolutionPipeline
  - Makes the system act toward objectives rather than just reacting
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Built-in default objectives ────────────────────────────────────────────

_DEFAULT_OBJECTIVES = [
    {
        "name": "Increase success rate",
        "metric": "success_rate",
        "direction": "maximize",
        "target": 0.95,
        "priority": 1,
        "description": "Ensure >95% of executions succeed",
    },
    {
        "name": "Reduce latency",
        "metric": "avg_latency_ms",
        "direction": "minimize",
        "target": 200.0,
        "priority": 2,
        "description": "Keep average execution latency below 200ms",
    },
    {
        "name": "Expand capabilities",
        "metric": "capability_count",
        "direction": "maximize",
        "target": 20,
        "priority": 3,
        "description": "Grow the number of distinct capabilities in the mesh",
    },
    {
        "name": "Reduce failures",
        "metric": "failure_count",
        "direction": "minimize",
        "target": 0,
        "priority": 4,
        "description": "Eliminate recurring failure patterns",
    },
]


class Objective:
    """A single strategic goal."""

    def __init__(
        self,
        name: str,
        metric: str,
        direction: str,       # "maximize" or "minimize"
        target: float,
        priority: int = 5,
        description: str = "",
    ):
        self.objective_id = str(uuid.uuid4())[:12]
        self.name = name
        self.metric = metric
        self.direction = direction
        self.target = target
        self.priority = priority
        self.description = description
        self.active = True
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.progress_history: List[dict] = []
        self.current_value: Optional[float] = None
        self.achieved = False

    @property
    def progress_pct(self) -> float:
        """How far along toward the target (0-100%)."""
        if self.current_value is None:
            return 0.0
        if self.direction == "maximize":
            if self.target == 0:
                return 100.0
            return min(100.0, (self.current_value / self.target) * 100)
        else:  # minimize
            if self.current_value <= self.target:
                return 100.0
            # Progress = how much we've reduced toward target from an assumed 2x baseline
            baseline = self.target * 2 or 1.0
            progress = max(0.0, (baseline - self.current_value) / (baseline - self.target))
            return min(100.0, progress * 100)

    def update(self, current_value: float):
        self.current_value = current_value
        if self.direction == "maximize":
            self.achieved = current_value >= self.target
        else:
            self.achieved = current_value <= self.target
        self.progress_history.append({
            "value": round(current_value, 4),
            "achieved": self.achieved,
            "ts": datetime.now(timezone.utc).isoformat(),
        })
        self.progress_history = self.progress_history[-50:]

    def to_dict(self) -> dict:
        return {
            "objective_id": self.objective_id,
            "name": self.name,
            "metric": self.metric,
            "direction": self.direction,
            "target": self.target,
            "current_value": self.current_value,
            "progress_pct": round(self.progress_pct, 1),
            "achieved": self.achieved,
            "active": self.active,
            "priority": self.priority,
            "description": self.description,
        }


class ObjectivesEngine:
    """
    Phase 7: Strategic Goals Engine.

    Manages long-term objectives and provides goal-aligned recommendations.
    """

    def __init__(self, knowledge_store=None):
        self._knowledge = knowledge_store
        self._objectives: Dict[str, Objective] = {}
        self._recommendation_log: List[dict] = []
        self._measurement_count = 0

        # Load defaults
        for obj_def in _DEFAULT_OBJECTIVES:
            self.add_objective(**obj_def)

    # ── Objective management ───────────────────────────────────────────────

    def add_objective(
        self,
        name: str,
        metric: str,
        direction: str = "maximize",
        target: float = 1.0,
        priority: int = 5,
        description: str = "",
    ) -> Objective:
        obj = Objective(name=name, metric=metric, direction=direction,
                        target=target, priority=priority, description=description)
        self._objectives[obj.objective_id] = obj
        logger.info(f"[Objectives] added '{name}' ({direction} {metric} → {target})")
        return obj

    def active_goals(self) -> List[str]:
        """Return list of active goal names."""
        return [
            obj.name for obj in sorted(self._objectives.values(), key=lambda o: o.priority)
            if obj.active
        ]

    # ── Measurement ────────────────────────────────────────────────────────

    def measure_from_mesh(self, mesh) -> dict:
        """
        Pull current metrics from the mesh and update all objectives.
        Returns a measurement snapshot.
        """
        measurements: Dict[str, float] = {}

        try:
            # success_rate from memory
            routes = mesh.memory.all_routes() if mesh.memory else []
            if routes:
                rates = [r.get("success_rate", 0.0) for r in routes if r.get("runs", 0) > 0]
                measurements["success_rate"] = sum(rates) / len(rates) if rates else 0.0
        except Exception:
            pass

        try:
            # avg_latency_ms from scoring
            scores = mesh.scoring.list_scores() if mesh.scoring else []
            if scores:
                lats = [s.get("avg_latency_ms", 0.0) for s in scores if s.get("total_runs", 0) > 0]
                measurements["avg_latency_ms"] = sum(lats) / len(lats) if lats else 0.0
        except Exception:
            pass

        try:
            # capability_count from marketplace
            caps = mesh.marketplace.list_capabilities() if mesh.marketplace else []
            measurements["capability_count"] = float(len(caps))
        except Exception:
            pass

        try:
            # failure_count
            routes = mesh.memory.all_routes() if mesh.memory else []
            failure_count = sum(1 for r in routes if r.get("health") in ("critical", "poor"))
            measurements["failure_count"] = float(failure_count)
        except Exception:
            pass

        # Update all objectives
        for obj in self._objectives.values():
            if obj.metric in measurements:
                obj.update(measurements[obj.metric])

        self._measurement_count += 1
        return measurements

    # ── Recommendations ────────────────────────────────────────────────────

    def get_recommendations(self) -> List[dict]:
        """
        Return goal-aligned recommendations for the EvolutionPipeline.
        """
        recs = []
        for obj in sorted(self._objectives.values(), key=lambda o: o.priority):
            if not obj.active or obj.achieved:
                continue
            rec = self._recommendation_for(obj)
            if rec:
                recs.append(rec)
        self._recommendation_log.extend(recs)
        self._recommendation_log = self._recommendation_log[-200:]
        return recs

    def _recommendation_for(self, obj: Objective) -> Optional[dict]:
        if obj.metric == "success_rate":
            if obj.current_value is not None and obj.current_value < obj.target:
                return {
                    "objective": obj.name,
                    "action": "investigate_failures",
                    "priority": obj.priority,
                    "reason": f"Current success rate {obj.current_value:.1%} < target {obj.target:.1%}",
                    "suggested_modes": ["scan_gaps", "evolve", "self_optimize"],
                }
        elif obj.metric == "avg_latency_ms":
            if obj.current_value is not None and obj.current_value > obj.target:
                return {
                    "objective": obj.name,
                    "action": "optimize_routes",
                    "priority": obj.priority,
                    "reason": f"Latency {obj.current_value:.0f}ms > target {obj.target:.0f}ms",
                    "suggested_modes": ["self_optimize", "optimize"],
                }
        elif obj.metric == "capability_count":
            if obj.current_value is not None and obj.current_value < obj.target:
                return {
                    "objective": obj.name,
                    "action": "expand_capabilities",
                    "priority": obj.priority,
                    "reason": f"Only {int(obj.current_value or 0)} capabilities < target {int(obj.target)}",
                    "suggested_modes": ["evolve", "scan_gaps"],
                }
        elif obj.metric == "failure_count":
            if obj.current_value and obj.current_value > 0:
                return {
                    "objective": obj.name,
                    "action": "fix_failures",
                    "priority": obj.priority,
                    "reason": f"{int(obj.current_value)} active failure patterns detected",
                    "suggested_modes": ["scan_gaps", "evolve"],
                }
        return None

    # ── Summary ────────────────────────────────────────────────────────────

    def get_all_objectives(self) -> List[dict]:
        return [
            obj.to_dict()
            for obj in sorted(self._objectives.values(), key=lambda o: o.priority)
        ]

    def summary(self) -> dict:
        total = len(self._objectives)
        achieved = sum(1 for o in self._objectives.values() if o.achieved)
        return {
            "total_objectives": total,
            "achieved": achieved,
            "pending": total - achieved,
            "measurement_count": self._measurement_count,
            "active_goals": self.active_goals(),
        }
