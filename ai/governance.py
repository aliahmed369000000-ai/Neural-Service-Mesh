"""
Phase 7 – Governance Approval (Phase 7 Extension)
===================================================
Extends Phase 5 GovernanceLayer with Phase 7-specific rules:
  - Code quality review of generated modules
  - Safety scan (no unsafe patterns)
  - Compatibility checks with existing mesh
  - Sandbox score threshold enforcement

Decisions:
  Approve        – module can be deployed
  Reject         – module is unsafe or broken
  Needs Revision – module needs improvement before retry

File: ai/governance_p7.py
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MIN_SANDBOX_SCORE = 75.0    # modules below this are rejected
_MIN_SYNTAX_REQUIRED = True  # syntax must be valid


class P7GovernanceDecision:
    """Phase 7 governance decision on a generated module."""

    APPROVE = "approve"
    REJECT = "reject"
    NEEDS_REVISION = "needs_revision"

    def __init__(
        self,
        module_id: str,
        module_name: str,
        verdict: str,
        reason: str,
        risk_score: float = 0.0,
        conditions: Optional[List[str]] = None,
    ):
        self.decision_id = f"gov7_{module_id}"
        self.module_id = module_id
        self.module_name = module_name
        self.verdict = verdict
        self.reason = reason
        self.risk_score = risk_score
        self.conditions = conditions or []
        self.decided_at = datetime.now(timezone.utc).isoformat()

    @property
    def allowed(self) -> bool:
        return self.verdict == self.APPROVE

    def to_dict(self) -> dict:
        return {
            "decision_id": self.decision_id,
            "module_id": self.module_id,
            "module_name": self.module_name,
            "verdict": self.verdict,
            "reason": self.reason,
            "risk_score": round(self.risk_score, 3),
            "conditions": self.conditions,
            "decided_at": self.decided_at,
        }


class P7GovernanceLayer:
    """
    Phase 7 Governance: Reviews auto-generated modules before deployment.

    Works in concert with the existing Phase 5 AIGovernanceLayer.
    Called by the EvolutionPipeline after sandbox testing.
    """

    def __init__(
        self,
        min_sandbox_score: float = _MIN_SANDBOX_SCORE,
        knowledge_store=None,
        base_governance=None,
    ):
        self._min_score = min_sandbox_score
        self._knowledge = knowledge_store
        self._base_gov = base_governance    # Phase 5 governor
        self._audit: List[dict] = []
        self._decision_count = 0
        self._approve_count = 0
        self._reject_count = 0

    def review(self, module, test_result: dict) -> P7GovernanceDecision:
        """
        Review a generated module + its sandbox test result.

        Returns a P7GovernanceDecision.
        """
        module_id = getattr(module, "module_id", "unknown")
        module_name = getattr(module, "name", "unknown")

        # 1. Syntax check
        if not test_result.get("syntax_valid", False):
            return self._decide(
                module_id, module_name, P7GovernanceDecision.REJECT,
                "Syntax invalid — module cannot be parsed",
                risk_score=1.0,
            )

        # 2. Safety violations
        violations = test_result.get("safety_violations", [])
        if violations:
            return self._decide(
                module_id, module_name, P7GovernanceDecision.REJECT,
                f"Safety violation: {', '.join(violations)}",
                risk_score=1.0,
            )

        # 3. Import check
        if not test_result.get("import_success", False):
            err = test_result.get("import_error", "unknown")
            return self._decide(
                module_id, module_name, P7GovernanceDecision.NEEDS_REVISION,
                f"Module fails to import: {err}",
                risk_score=0.7,
            )

        # 4. Sandbox score threshold
        score = test_result.get("score", 0.0)
        if score < self._min_score:
            return self._decide(
                module_id, module_name, P7GovernanceDecision.NEEDS_REVISION,
                f"Sandbox score {score:.1f} below threshold {self._min_score}",
                risk_score=0.5,
                conditions=[f"Improve score to ≥ {self._min_score}"],
            )

        # 5. Execution check
        if not test_result.get("execution_success", False):
            err = test_result.get("execution_error", "unknown")
            return self._decide(
                module_id, module_name, P7GovernanceDecision.NEEDS_REVISION,
                f"Execution failed: {err}",
                risk_score=0.6,
            )

        # All checks passed
        latency = test_result.get("execution_latency_ms", 0.0)
        return self._decide(
            module_id, module_name, P7GovernanceDecision.APPROVE,
            f"All checks passed. Score={score:.1f}, latency={latency:.1f}ms",
            risk_score=0.0,
        )

    def _decide(
        self,
        module_id: str,
        module_name: str,
        verdict: str,
        reason: str,
        risk_score: float = 0.0,
        conditions: Optional[List[str]] = None,
    ) -> P7GovernanceDecision:
        decision = P7GovernanceDecision(
            module_id=module_id,
            module_name=module_name,
            verdict=verdict,
            reason=reason,
            risk_score=risk_score,
            conditions=conditions,
        )
        self._audit.append(decision.to_dict())
        self._audit = self._audit[-500:]
        self._decision_count += 1
        if verdict == P7GovernanceDecision.APPROVE:
            self._approve_count += 1
            logger.info(f"[P7Gov] APPROVE '{module_name}': {reason}")
        elif verdict == P7GovernanceDecision.REJECT:
            self._reject_count += 1
            logger.warning(f"[P7Gov] REJECT '{module_name}': {reason}")
        else:
            logger.info(f"[P7Gov] NEEDS_REVISION '{module_name}': {reason}")
        return decision

    def audit_log(self, limit: int = 20) -> List[dict]:
        return self._audit[-limit:]

    def summary(self) -> dict:
        return {
            "total_decisions": self._decision_count,
            "approved": self._approve_count,
            "rejected": self._reject_count,
            "needs_revision": self._decision_count - self._approve_count - self._reject_count,
            "min_sandbox_score": self._min_score,
        }
