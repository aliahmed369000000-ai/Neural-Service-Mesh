"""
Phase 5 – Multi-Goal Planner
==============================
Extends Phase 3 GoalPlanner to handle complex goals with multiple sub-goals.

Phase 3: Goal → Route
Phase 5: Goals → Plans → Routes

Example:
  Goal: "Analyze reviews"
  Subgoals:
    - Clean text
    - Translate
    - Analyze sentiment
    - Generate report

Each sub-goal is planned independently and chained into a composite plan.
Uses CapabilityMarketplace to find providers by capability, not name.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class SubGoal:
    """A single step within a multi-goal execution plan."""

    def __init__(
        self,
        name: str,
        capability: str,
        input_keys: Optional[List[str]] = None,
        output_keys: Optional[List[str]] = None,
        required: bool = True,
        description: str = "",
    ):
        self.name = name
        self.capability = capability
        self.input_keys = input_keys or ["data"]
        self.output_keys = output_keys or ["result"]
        self.required = required
        self.description = description
        self.resolved_node_id: Optional[str] = None
        self.resolved_node_name: Optional[str] = None
        self.confidence: float = 0.0
        self.status: str = "pending"  # pending / resolved / failed / skipped

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "capability": self.capability,
            "input_keys": self.input_keys,
            "output_keys": self.output_keys,
            "required": self.required,
            "description": self.description,
            "resolved_node_id": self.resolved_node_id,
            "resolved_node_name": self.resolved_node_name,
            "confidence": round(self.confidence, 4),
            "status": self.status,
        }


class MultiGoalPlan:
    """
    A composite execution plan with ordered sub-goals.
    Each sub-goal resolves to a specific node via the CapabilityMarketplace.
    """

    def __init__(
        self,
        primary_goal: str,
        sub_goals: List[SubGoal],
        total_confidence: float = 0.0,
    ):
        self.plan_id = f"mgp_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')[:18]}"
        self.primary_goal = primary_goal
        self.sub_goals = sub_goals
        self.total_confidence = total_confidence
        self.resolved_path: List[str] = []
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.status = "pending"
        self.result: Optional[dict] = None

    @property
    def is_fully_resolved(self) -> bool:
        return all(
            sg.status in ("resolved", "skipped")
            for sg in self.sub_goals
        )

    def to_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "primary_goal": self.primary_goal,
            "sub_goals": [sg.to_dict() for sg in self.sub_goals],
            "total_confidence": round(self.total_confidence, 4),
            "resolved_path": self.resolved_path,
            "is_fully_resolved": self.is_fully_resolved,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
        }


class MultiGoalPlanner:
    """
    Phase 5: Decomposes complex goals into ordered sub-goals and builds
    composite execution plans using the CapabilityMarketplace.

    Upgrade from Phase 3 GoalPlanner:
      - Phase 3: Single goal → single path
      - Phase 5: Complex goal → sub-goals → parallel/sequential paths
    """

    # Known goal decompositions (vocabulary for common tasks)
    _GOAL_TEMPLATES: Dict[str, List[dict]] = {
        "analyze_reviews": [
            {"name": "Clean Text", "capability": "clean", "required": True},
            {"name": "Translate", "capability": "translate", "required": False},
            {"name": "Analyze Sentiment", "capability": "sentiment", "required": True},
            {"name": "Generate Report", "capability": "format", "required": True},
        ],
        "process_data": [
            {"name": "Validate Input", "capability": "validate", "required": True},
            {"name": "Transform Data", "capability": "transform", "required": True},
            {"name": "Normalize", "capability": "normalize", "required": False},
            {"name": "Output Results", "capability": "process", "required": True},
        ],
        "translate_content": [
            {"name": "Detect Language", "capability": "analyze", "required": True},
            {"name": "Translate Text", "capability": "translate", "required": True},
            {"name": "Format Output", "capability": "format", "required": False},
        ],
        "enrich_data": [
            {"name": "Clean Data", "capability": "clean", "required": True},
            {"name": "Enrich", "capability": "enrich", "required": True},
            {"name": "Aggregate Results", "capability": "aggregate", "required": False},
        ],
        "summarize": [
            {"name": "Process Text", "capability": "process", "required": True},
            {"name": "Analyze", "capability": "analyze", "required": False},
            {"name": "Summarize", "capability": "summarize", "required": True},
        ],
    }

    # Keywords that map to decomposition templates
    _GOAL_KEYWORDS: Dict[str, str] = {
        "review": "analyze_reviews",
        "sentiment": "analyze_reviews",
        "translate": "translate_content",
        "translation": "translate_content",
        "enrich": "enrich_data",
        "summarize": "summarize",
        "summary": "summarize",
        "process": "process_data",
        "transform": "process_data",
        "analyze": "analyze_reviews",
    }

    def __init__(
        self,
        capability_marketplace=None,
        goal_planner=None,  # Phase 3 planner (fallback)
        memory_engine=None,
        knowledge_store=None,
    ):
        self._marketplace = capability_marketplace
        self._phase3_planner = goal_planner
        self._memory = memory_engine
        self._knowledge = knowledge_store
        self._plan_history: List[MultiGoalPlan] = []
        logger.info("MultiGoalPlanner initialised (Phase 5)")

    def set_capability_marketplace(self, marketplace):
        self._marketplace = marketplace

    def set_phase3_planner(self, planner):
        self._phase3_planner = planner

    # ── Core planning ──────────────────────────────────────────────────────

    def plan(
        self,
        goal: str,
        context: Optional[dict] = None,
        max_sub_goals: int = 8,
    ) -> MultiGoalPlan:
        """
        Decompose a complex goal into sub-goals and resolve each to a node.

        Returns a MultiGoalPlan with resolved_path ready for execution.
        """
        sub_goal_specs = self._decompose_goal(goal, max_sub_goals)
        sub_goals = [SubGoal(**spec) for spec in sub_goal_specs]

        # Resolve each sub-goal to a node via marketplace
        resolved_path = []
        confidences = []

        for sg in sub_goals:
            node_id, node_name, confidence = self._resolve_sub_goal(sg)
            if node_id:
                sg.resolved_node_id = node_id
                sg.resolved_node_name = node_name
                sg.confidence = confidence
                sg.status = "resolved"
                # Only add to path if not already there (avoid duplicates)
                if not resolved_path or resolved_path[-1] != node_id:
                    resolved_path.append(node_id)
                    confidences.append(confidence)
            else:
                if sg.required:
                    sg.status = "failed"
                    logger.warning(f"Could not resolve required sub-goal: '{sg.name}' (capability='{sg.capability}')")
                else:
                    sg.status = "skipped"

        total_confidence = (
            sum(confidences) / len(confidences) if confidences else 0.0
        )

        plan = MultiGoalPlan(
            primary_goal=goal,
            sub_goals=sub_goals,
            total_confidence=total_confidence,
        )
        plan.resolved_path = resolved_path

        if len(resolved_path) >= 2:
            plan.status = "ready"
        elif resolved_path:
            plan.status = "partial"
        else:
            plan.status = "failed"

        self._plan_history.append(plan)
        logger.info(
            f"MultiGoalPlan: '{goal}' → {len(resolved_path)} nodes, "
            f"confidence={total_confidence:.2f}, status={plan.status}"
        )
        return plan

    def _decompose_goal(self, goal: str, max_sub_goals: int) -> List[dict]:
        """
        Decompose a goal string into ordered sub-goal specs.
        First tries template matching, then falls back to single-step.
        """
        goal_lower = goal.lower().replace("-", " ").replace("_", " ")

        # Match to a known template
        for keyword, template_key in self._GOAL_KEYWORDS.items():
            if keyword in goal_lower:
                template = self._GOAL_TEMPLATES.get(template_key, [])
                if template:
                    return template[:max_sub_goals]

        # No template match: create a single generic sub-goal
        return [
            {
                "name": "Execute Goal",
                "capability": self._goal_to_capability(goal),
                "required": True,
                "description": goal,
            }
        ]

    def _goal_to_capability(self, goal: str) -> str:
        """Extract a capability token from a free-form goal string."""
        goal_lower = goal.lower()
        verbs = [
            "translate", "analyze", "process", "transform", "validate",
            "enrich", "filter", "normalize", "summarize", "classify",
            "clean", "aggregate", "format", "route",
        ]
        for verb in verbs:
            if verb in goal_lower:
                return verb
        # Return first meaningful word
        words = [w for w in goal_lower.split() if len(w) > 3]
        return words[0] if words else "process"

    def _resolve_sub_goal(
        self,
        sub_goal: SubGoal,
    ) -> tuple:  # (node_id, node_name, confidence)
        """
        Resolve a sub-goal to a concrete node using CapabilityMarketplace.
        Falls back to Phase 3 planner if marketplace has no providers.
        """
        if self._marketplace:
            ad = self._marketplace.best_provider(sub_goal.capability)
            if ad:
                return ad.node_id, ad.node_name, ad.composite_score

        # Fallback: use Phase 3 planner to find relevant nodes
        if self._phase3_planner:
            try:
                plan = self._phase3_planner.plan(
                    goal=sub_goal.capability,
                    max_hops=3,
                )
                if plan and plan.path:
                    return plan.path[0], "", plan.confidence
            except Exception:
                pass

        return None, None, 0.0

    # ── Execution ──────────────────────────────────────────────────────────

    def execute_plan(self, plan: MultiGoalPlan, engine, data: dict) -> dict:
        """
        Execute a MultiGoalPlan using the given execution engine.
        Chains sub-goal outputs as inputs for the next sub-goal.
        """
        if plan.status not in ("ready", "partial"):
            return {
                "status": "failed",
                "error": f"Plan is not executable (status={plan.status})",
                "plan": plan.to_dict(),
            }

        if len(plan.resolved_path) < 2:
            return {
                "status": "failed",
                "error": "Plan requires at least 2 nodes in resolved_path",
                "plan": plan.to_dict(),
            }

        plan.status = "running"
        try:
            result = engine.run_path(plan.resolved_path, data)
            result_dict = result.to_dict()
            plan.status = result_dict.get("status", "completed")
            plan.result = result_dict
            result_dict["multi_goal_plan"] = plan.to_dict()
            return result_dict
        except Exception as e:
            plan.status = "failed"
            return {
                "status": "failed",
                "error": str(e),
                "plan": plan.to_dict(),
            }

    # ── History & summary ──────────────────────────────────────────────────

    def recent_plans(self, limit: int = 10) -> List[dict]:
        return [p.to_dict() for p in self._plan_history[-limit:]]

    def summary(self) -> dict:
        statuses: Dict[str, int] = {}
        for p in self._plan_history:
            statuses[p.status] = statuses.get(p.status, 0) + 1
        return {
            "total_plans": len(self._plan_history),
            "by_status": statuses,
            "marketplace_connected": self._marketplace is not None,
        }
