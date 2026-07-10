"""
Phase 5 – AI Governance Layer
================================
Governs autonomous AI behaviour to prevent:
  - Routing loops
  - Bad routes (consistently failing)
  - Uncontrolled growth (infinite service generation)

The Governance Layer wraps all Phase 5 autonomous actions and
validates them before execution. It maintains a policy ruleset
and an audit log of every AI-driven change.

File: ai/governor.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

logger = logging.getLogger(__name__)


class GovernanceDecision:
    """Result of a governance evaluation."""

    def __init__(
        self,
        allowed: bool,
        action: str,
        reason: str,
        risk_score: float = 0.0,
        metadata: Optional[dict] = None,
    ):
        self.allowed = allowed
        self.action = action
        self.reason = reason
        self.risk_score = risk_score
        self.metadata = metadata or {}
        self.evaluated_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "action": self.action,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 4),
            "metadata": self.metadata,
            "evaluated_at": self.evaluated_at,
        }


class GovernancePolicy:
    """
    A single governance rule.
    rule_fn(context) -> (allowed: bool, reason: str, risk: float)
    """

    def __init__(self, name: str, rule_fn: Callable, priority: int = 50):
        self.name = name
        self.rule_fn = rule_fn
        self.priority = priority
        self.triggered_count = 0

    def evaluate(self, context: dict) -> tuple:
        try:
            result = self.rule_fn(context)
            if result[0] is False:
                self.triggered_count += 1
            return result
        except Exception as e:
            logger.warning(f"Policy '{self.name}' error: {e}")
            return True, f"Policy error (allowed by default): {e}", 0.0


class AIGovernanceLayer:
    """
    Phase 5: Central governance authority for all autonomous AI actions.

    Evaluates every proposed autonomous action against a policy set
    and maintains a complete audit trail.

    Prevents:
      1. LOOPS: route cycles that would cause infinite execution
      2. BAD ROUTES: connections to quarantined or failing nodes
      3. UNCONTROLLED GROWTH: too many AI-generated services
      4. RAPID GENERATION: service generation rate limiting
    """

    # Hard limits
    MAX_GENERATED_SERVICES = 50
    MAX_PATH_LENGTH = 15
    MIN_NODE_REPUTATION = 20.0  # Below this, node needs governance review
    MAX_GENERATION_RATE = 10    # Per scan cycle
    MAX_FAILURE_RATE = 0.8      # Reject actions involving nodes with >80% failure

    def __init__(
        self,
        graph=None,
        reputation_engine=None,
        knowledge_store=None,
    ):
        self._graph = graph
        self._reputation = reputation_engine
        self._knowledge = knowledge_store
        self._policies: List[GovernancePolicy] = []
        self._audit_log: List[dict] = []
        self._blocked_count = 0
        self._approved_count = 0
        self._generation_count_this_cycle = 0

        # Install default policies
        self._install_default_policies()
        logger.info("AIGovernanceLayer initialised (Phase 5)")

    def set_graph(self, g):
        self._graph = g

    def set_reputation_engine(self, r):
        self._reputation = r

    def set_knowledge_store(self, ks):
        self._knowledge = ks

    # ── Policy installation ────────────────────────────────────────────────

    def _install_default_policies(self):
        """Install the built-in governance rules."""

        # Rule 1: Loop detection
        def loop_policy(ctx):
            path = ctx.get("path", [])
            if len(path) != len(set(path)):
                return False, f"Loop detected in path: {path}", 1.0
            return True, "No loops", 0.0

        # Rule 2: Path length limit
        def path_length_policy(ctx):
            path = ctx.get("path", [])
            if len(path) > self.MAX_PATH_LENGTH:
                return False, f"Path too long: {len(path)} > {self.MAX_PATH_LENGTH}", 0.7
            return True, "Path length OK", 0.0

        # Rule 3: Quarantined node check
        def quarantine_policy(ctx):
            path = ctx.get("path", [])
            if not self._reputation:
                return True, "Reputation engine unavailable", 0.0
            for node_id in path:
                rep = self._reputation.get_reputation(node_id)
                if rep and rep.is_quarantined:
                    name = rep.name or node_id[:8]
                    return False, f"Quarantined node in path: '{name}'", 0.9
            return True, "No quarantined nodes", 0.0

        # Rule 4: Generation rate limit
        def generation_rate_policy(ctx):
            if ctx.get("action") != "generate_service":
                return True, "Not a generation action", 0.0
            if self._generation_count_this_cycle >= self.MAX_GENERATION_RATE:
                return False, f"Generation rate limit hit ({self.MAX_GENERATION_RATE}/cycle)", 0.6
            return True, "Generation rate OK", 0.0

        # Rule 5: Total generated services cap
        def service_cap_policy(ctx):
            if ctx.get("action") != "generate_service":
                return True, "Not a generation action", 0.0
            current = ctx.get("total_generated", 0)
            if current >= self.MAX_GENERATED_SERVICES:
                return False, f"Max services cap reached ({self.MAX_GENERATED_SERVICES})", 0.5
            return True, "Service count OK", 0.0

        # Rule 6: Low confidence service rejection
        def confidence_policy(ctx):
            confidence = ctx.get("confidence", 1.0)
            if confidence < 0.4:
                return False, f"Confidence too low: {confidence:.2f} < 0.40", 0.4
            return True, f"Confidence OK ({confidence:.2f})", 0.0

        # Rule 7: Bad route detection (node failure rate)
        def bad_route_policy(ctx):
            path = ctx.get("path", [])
            if not self._reputation:
                return True, "Reputation unavailable", 0.0
            for node_id in path:
                rep = self._reputation.get_reputation(node_id)
                if rep and rep.total_runs > 5:
                    if rep.success_rate < (1 - self.MAX_FAILURE_RATE):
                        name = rep.name or node_id[:8]
                        return False, (
                            f"Node '{name}' has high failure rate: "
                            f"{1-rep.success_rate:.0%}"
                        ), 0.7
            return True, "No bad routes detected", 0.0

        self.add_policy(GovernancePolicy("loop_detection", loop_policy, priority=100))
        self.add_policy(GovernancePolicy("path_length", path_length_policy, priority=90))
        self.add_policy(GovernancePolicy("quarantine_check", quarantine_policy, priority=95))
        self.add_policy(GovernancePolicy("generation_rate", generation_rate_policy, priority=80))
        self.add_policy(GovernancePolicy("service_cap", service_cap_policy, priority=75))
        self.add_policy(GovernancePolicy("confidence_check", confidence_policy, priority=70))
        self.add_policy(GovernancePolicy("bad_route", bad_route_policy, priority=85))

    def add_policy(self, policy: GovernancePolicy):
        """Add a custom governance policy."""
        self._policies.append(policy)
        self._policies.sort(key=lambda p: p.priority, reverse=True)
        logger.info(f"Governance: policy added '{policy.name}' (priority={policy.priority})")

    # ── Evaluation ─────────────────────────────────────────────────────────

    def evaluate(self, action: str, context: dict) -> GovernanceDecision:
        """
        Evaluate a proposed AI action against all policies.

        action examples:
          "generate_service", "add_route", "remove_node", "promote_route"

        context:
          {
            "path": [...],
            "confidence": 0.85,
            "total_generated": 12,
            "action": "generate_service",
            ...
          }
        """
        ctx = dict(context)
        ctx["action"] = action

        max_risk = 0.0
        blocking_reason = None

        for policy in self._policies:
            allowed, reason, risk = policy.evaluate(ctx)
            max_risk = max(max_risk, risk)
            if not allowed:
                blocking_reason = f"[{policy.name}] {reason}"
                break  # First blocking policy wins

        if blocking_reason:
            decision = GovernanceDecision(
                allowed=False,
                action=action,
                reason=blocking_reason,
                risk_score=max_risk,
                metadata=ctx,
            )
            self._blocked_count += 1
        else:
            decision = GovernanceDecision(
                allowed=True,
                action=action,
                reason="All policies passed",
                risk_score=max_risk,
                metadata=ctx,
            )
            self._approved_count += 1
            if action == "generate_service":
                self._generation_count_this_cycle += 1

        self._audit(decision)
        return decision

    def evaluate_path(self, path: List[str], confidence: float = 1.0) -> GovernanceDecision:
        """Convenience: evaluate a route path."""
        return self.evaluate(
            action="add_route",
            context={"path": path, "confidence": confidence},
        )

    def evaluate_generation(
        self,
        spec_dict: dict,
        total_generated: int,
    ) -> GovernanceDecision:
        """Convenience: evaluate a service generation request."""
        return self.evaluate(
            action="generate_service",
            context={
                "confidence": spec_dict.get("confidence", 0.5),
                "total_generated": total_generated,
                "spec_name": spec_dict.get("name", ""),
            },
        )

    def reset_cycle(self):
        """Call at the start of each evolution cycle to reset rate counters."""
        self._generation_count_this_cycle = 0

    # ── Audit ──────────────────────────────────────────────────────────────

    def _audit(self, decision: GovernanceDecision):
        """Record every governance decision in the audit log."""
        entry = decision.to_dict()
        self._audit_log.append(entry)
        # Keep last 1000 entries
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-1000:]

        # Persist to knowledge store
        if self._knowledge:
            try:
                self._knowledge.write_custom("governance_audit", {
                    "last_100": self._audit_log[-100:],
                    "blocked_total": self._blocked_count,
                    "approved_total": self._approved_count,
                })
            except Exception:
                pass

    def audit_log(self, limit: int = 50) -> List[dict]:
        """Return the most recent audit entries."""
        return self._audit_log[-limit:]

    def policy_stats(self) -> List[dict]:
        """Return trigger counts for each policy."""
        return [
            {"name": p.name, "priority": p.priority, "triggered": p.triggered_count}
            for p in self._policies
        ]

    def summary(self) -> dict:
        return {
            "total_evaluated": self._approved_count + self._blocked_count,
            "approved": self._approved_count,
            "blocked": self._blocked_count,
            "block_rate": (
                self._blocked_count / (self._approved_count + self._blocked_count)
                if (self._approved_count + self._blocked_count) > 0 else 0.0
            ),
            "policies": len(self._policies),
            "generation_this_cycle": self._generation_count_this_cycle,
            "policy_stats": self.policy_stats(),
        }
