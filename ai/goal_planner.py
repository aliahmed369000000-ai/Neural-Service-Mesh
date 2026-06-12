"""
Phase 3 – Goal Planner
Instead of "run node A → B → C", the user says:
  Goal: "Summarize customer reviews"
The planner builds an execution plan automatically.
"""
from __future__ import annotations
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ExecutionPlan:
    """
    A fully resolved execution plan built from a goal description.
    Contains the selected path, confidence, and explanation.
    """

    def __init__(self, goal: str, path: List[str],
                 confidence: float, reasoning: List[str]):
        self.goal = goal
        self.path = path
        self.confidence = confidence
        self.reasoning = reasoning
        self.created_at = datetime.utcnow().isoformat()
        self.status: str = "pending"  # pending / running / completed / failed
        self.result: Optional[dict] = None

    def to_dict(self) -> dict:
        return {
            "goal": self.goal,
            "path": self.path,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "created_at": self.created_at,
            "status": self.status,
            "result": self.result,
        }


class GoalPlanner:
    """
    Phase 3 Goal Planner.

    Given a high-level goal description, it:
    1. Finds the most relevant entry nodes (via SemanticMatcher).
    2. Finds the most relevant exit/output nodes.
    3. Uses RoutingEngine to find the best path.
    4. Returns an ExecutionPlan.

    The planner can also accept constraints:
      - required_tags: path must include a node with these tags
      - max_hops: maximum path length
      - preferred_start: force a specific start node
      - preferred_end:   force a specific end node
    """

    def __init__(self, semantic_matcher=None, routing_engine=None,
                 memory_engine=None, registry=None):
        self._semantic = semantic_matcher
        self._routing = routing_engine
        self._memory = memory_engine
        self._registry = registry
        logger.info("GoalPlanner initialised (Phase 3)")

    def set_components(self, semantic=None, routing=None, memory=None, registry=None):
        if semantic:
            self._semantic = semantic
        if routing:
            self._routing = routing
        if memory:
            self._memory = memory
        if registry:
            self._registry = registry

    # ── Main planning API ──────────────────────────────────────────────────

    def plan(self, goal: str,
             preferred_start: Optional[str] = None,
             preferred_end: Optional[str] = None,
             required_tags: Optional[List[str]] = None,
             max_hops: int = 10) -> Optional[ExecutionPlan]:
        """
        Build an ExecutionPlan for the given goal.
        Returns None if no viable plan can be constructed.
        """
        reasoning: List[str] = []
        reasoning.append(f"Planning for goal: '{goal}'")

        # 1. Determine candidate start nodes
        start_id = preferred_start
        if not start_id:
            start_id = self._find_best_input_node(goal, required_tags)
            if start_id:
                reasoning.append(f"Selected start node via semantic matching: {start_id[:8]}")
            else:
                reasoning.append("Could not identify a suitable start node.")
                return None

        # 2. Determine candidate end nodes
        end_id = preferred_end
        if not end_id:
            end_id = self._find_best_output_node(goal, start_id, required_tags)
            if end_id:
                reasoning.append(f"Selected end node via semantic matching: {end_id[:8]}")
            else:
                reasoning.append("Could not identify a suitable end node.")
                return None

        if start_id == end_id:
            reasoning.append("Start and end are the same node — trivial plan.")
            return ExecutionPlan(goal, [start_id], 0.7, reasoning)

        # 3. Find route
        if self._routing:
            candidates = self._routing.rank_routes(start_id, end_id, max_candidates=5)
        else:
            candidates = []

        if not candidates:
            reasoning.append("No route found between selected nodes.")
            return None

        # 4. Filter by max_hops and required_tags
        valid = [c for c in candidates if (len(c.path) - 1) <= max_hops]
        if required_tags:
            valid = [c for c in valid if self._path_has_tags(c.path, required_tags)]
            if not valid:
                reasoning.append(f"No routes satisfy required_tags={required_tags}")
                return None

        if not valid:
            reasoning.append(f"No routes within max_hops={max_hops}")
            return None

        best = valid[0]
        confidence = min(1.0, best.score / 100.0)
        reasoning.append(
            f"Selected route: {' → '.join(n[:8] for n in best.path)} "
            f"(score={best.score:.1f}, source={best.source})"
        )

        return ExecutionPlan(goal, best.path, confidence, reasoning)

    def plan_multi_step(self, goals: List[str],
                        initial_start: Optional[str] = None) -> List[ExecutionPlan]:
        """
        Build sequential plans for multiple goals.
        Each plan's end node feeds into the next plan's start.
        """
        plans: List[ExecutionPlan] = []
        current_start = initial_start

        for goal in goals:
            plan = self.plan(goal, preferred_start=current_start)
            if plan:
                plans.append(plan)
                # Next goal starts from this plan's end node
                current_start = plan.path[-1] if plan.path else None
            else:
                logger.warning(f"GoalPlanner: could not plan for goal '{goal}'")
                break

        return plans

    # ── Node selection helpers ─────────────────────────────────────────────

    def _find_best_input_node(self, goal: str,
                               required_tags: Optional[List[str]] = None) -> Optional[str]:
        """Find the node best suited as an entry point for this goal."""
        if not self._semantic:
            return self._fallback_first_node()

        candidates = self._semantic.find_nodes_for_goal(goal, top_k=10)
        # Prefer nodes with no incoming edges (source nodes) if possible
        for node_id, score in candidates:
            if score < 0.01:
                continue
            if required_tags and not self._node_has_tags(node_id, required_tags):
                continue
            return node_id

        return None

    def _find_best_output_node(self, goal: str, start_id: str,
                                required_tags: Optional[List[str]] = None) -> Optional[str]:
        """Find the node best suited as the output/terminal node for this goal."""
        if not self._semantic:
            return self._fallback_last_node(start_id)

        # Search for nodes that produce outputs matching the goal
        output_keywords = self._extract_output_keywords(goal)
        candidates = self._semantic.find_nodes_for_goal(output_keywords, top_k=10)

        for node_id, score in candidates:
            if node_id == start_id:
                continue
            if score < 0.01:
                continue
            if required_tags and not self._node_has_tags(node_id, required_tags):
                continue
            return node_id

        return None

    def _extract_output_keywords(self, goal: str) -> str:
        """
        Derive likely output type from goal text.
        e.g. "Summarize reviews" → "summary output"
        """
        goal_lower = goal.lower()
        mappings = [
            (["summarize", "summary", "abstract"], "summary output report"),
            (["analyze", "analysis", "analyse"], "analysis result output"),
            (["classify", "categorize", "label"], "classification label output"),
            (["generate", "create", "produce"], "generate output result"),
            (["process", "transform", "convert"], "process transform output"),
            (["extract", "parse", "read"], "extract parse output data"),
            (["score", "rate", "rank"], "score rating result"),
            (["translate", "language"], "translated text output"),
            (["detect", "identify", "find"], "detection result output"),
            (["store", "save", "persist"], "storage output"),
        ]
        for keywords, output_hint in mappings:
            if any(kw in goal_lower for kw in keywords):
                return output_hint

        return f"{goal} output result"

    def _path_has_tags(self, path: List[str], required_tags: List[str]) -> bool:
        if not self._registry:
            return True
        for node_id in path:
            node = self._registry.get(node_id)
            if node and any(t in node.tags for t in required_tags):
                return True
        return False

    def _node_has_tags(self, node_id: str, required_tags: List[str]) -> bool:
        if not self._registry:
            return True
        node = self._registry.get(node_id)
        if not node:
            return False
        return any(t in node.tags for t in required_tags)

    def _fallback_first_node(self) -> Optional[str]:
        if not self._registry:
            return None
        nodes = self._registry.list_all()
        return nodes[0].node_id if nodes else None

    def _fallback_last_node(self, exclude_id: str) -> Optional[str]:
        if not self._registry:
            return None
        nodes = [n for n in self._registry.list_all() if n.node_id != exclude_id]
        return nodes[-1].node_id if nodes else None

    def suggest_goals(self) -> List[str]:
        """
        Return example goals based on currently registered node capabilities.
        """
        if not self._semantic:
            return []
        profiles = self._semantic.all_profiles()
        goals = []
        for p in profiles:
            cap = p.capability or p.description
            if cap:
                goals.append(f"Use {p.name} to {cap.lower().rstrip('.')}")
        return goals[:10]

    def __repr__(self):
        return "<GoalPlanner (semantic+routing+memory)>"
