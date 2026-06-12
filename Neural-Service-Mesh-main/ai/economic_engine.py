"""
Phase 6 – Economic Engine
===========================
Internal economy for the capability marketplace.

Each node (and agent) earns performance points:
  • Capability Score  – how useful / requested is this node?
  • Trust Score       – how reliable is this node over time?
  • Cost Score        – how expensive is this node to run?

These three scores drive routing, governance approvals, and
self-optimizer decisions.

Usage:
  economy = EconomicEngine(scoring_engine, reputation_engine, knowledge_store)
  snapshot = economy.evaluate_node(node_id)
  leaderboard = economy.leaderboard()
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

# Scoring weights (must sum to 1.0)
W_CAPABILITY = 0.40
W_TRUST      = 0.35
W_COST       = 0.25


class NodeEconomicProfile:
    """Three-dimensional economic profile for a single node."""

    def __init__(self, node_id: str, node_name: str):
        self.node_id = node_id
        self.node_name = node_name
        self.capability_score: float = 0.5
        self.trust_score: float = 0.5
        self.cost_score: float = 0.5   # 1.0 = cheapest, 0.0 = most expensive
        self.composite_score: float = 0.5
        self.usage_count: int = 0
        self.last_evaluated: Optional[str] = None
        self.rank: Optional[int] = None

    def compute_composite(self):
        self.composite_score = (
            W_CAPABILITY * self.capability_score
            + W_TRUST * self.trust_score
            + W_COST * self.cost_score
        )
        self.last_evaluated = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "capability_score": round(self.capability_score, 4),
            "trust_score": round(self.trust_score, 4),
            "cost_score": round(self.cost_score, 4),
            "composite_score": round(self.composite_score, 4),
            "usage_count": self.usage_count,
            "rank": self.rank,
            "last_evaluated": self.last_evaluated,
        }


class EconomicTransaction:
    """Records a change in a node's economic standing."""

    def __init__(
        self,
        node_id: str,
        tx_type: str,       # reward / penalty / usage / depreciation
        delta_capability: float = 0.0,
        delta_trust: float = 0.0,
        delta_cost: float = 0.0,
        reason: str = "",
    ):
        self.tx_id = f"tx_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
        self.node_id = node_id
        self.tx_type = tx_type
        self.delta_capability = delta_capability
        self.delta_trust = delta_trust
        self.delta_cost = delta_cost
        self.reason = reason
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return {
            "tx_id": self.tx_id,
            "node_id": self.node_id,
            "tx_type": self.tx_type,
            "delta_capability": round(self.delta_capability, 4),
            "delta_trust": round(self.delta_trust, 4),
            "delta_cost": round(self.delta_cost, 4),
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


class EconomicEngine:
    """
    Phase 6: Internal Economic Engine.

    Maintains economic profiles for all nodes, processes transactions,
    and exposes a leaderboard for routing decisions.
    """

    TRANSACTION_RULES = {
        "success": dict(d_cap=+0.02, d_trust=+0.03, d_cost=-0.01),
        "failure": dict(d_cap=-0.01, d_trust=-0.05, d_cost=+0.02),
        "usage":   dict(d_cap=+0.01, d_trust=+0.00, d_cost=-0.00),
        "timeout": dict(d_cap=-0.02, d_trust=-0.03, d_cost=+0.03),
        "gap_fill": dict(d_cap=+0.05, d_trust=+0.02, d_cost=+0.00),
        "idle":    dict(d_cap=-0.005, d_trust=+0.00, d_cost=-0.005),
    }

    def __init__(
        self,
        scoring_engine=None,
        reputation_engine=None,
        knowledge_store=None,
        registry=None,
    ):
        self._scoring = scoring_engine
        self._reputation = reputation_engine
        self._knowledge = knowledge_store
        self._registry = registry
        self._profiles: Dict[str, NodeEconomicProfile] = {}
        self._ledger: List[EconomicTransaction] = []
        logger.info("EconomicEngine initialised (Phase 6)")

    # ── Profile management ────────────────────────────────────────────────

    def get_or_create(self, node_id: str, node_name: str = "") -> NodeEconomicProfile:
        if node_id not in self._profiles:
            profile = NodeEconomicProfile(node_id, node_name or node_id[:8])
            # Seed from existing engines if available
            self._seed_profile(profile)
            self._profiles[node_id] = profile
        return self._profiles[node_id]

    def _seed_profile(self, profile: NodeEconomicProfile):
        """Initialise a new profile from existing scoring / reputation data."""
        if self._scoring:
            try:
                scores = self._scoring.list_scores()
                related = [
                    s for s in scores
                    if s.get("source_id") == profile.node_id
                    or s.get("target_id") == profile.node_id
                ]
                if related:
                    avg_sr = sum(s.get("success_rate", 0.5) for s in related) / len(related)
                    profile.trust_score = avg_sr
                    profile.capability_score = min(0.9, avg_sr + 0.1)
                    profile.usage_count = sum(s.get("total_runs", 0) for s in related)
            except Exception:
                pass

        if self._reputation:
            try:
                rep = self._reputation.get_reputation(profile.node_id)
                if rep:
                    profile.trust_score = max(
                        profile.trust_score,
                        rep.get("trust_score", profile.trust_score),
                    )
            except Exception:
                pass

        profile.compute_composite()

    # ── Transactions ──────────────────────────────────────────────────────

    def record_event(
        self,
        node_id: str,
        event: str,
        node_name: str = "",
        reason: str = "",
    ) -> Optional[EconomicTransaction]:
        """Apply a predefined event (success/failure/usage/etc.) to a node."""
        rule = self.TRANSACTION_RULES.get(event)
        if not rule:
            logger.warning(f"EconomicEngine: unknown event '{event}'")
            return None

        profile = self.get_or_create(node_id, node_name)
        tx = EconomicTransaction(
            node_id=node_id,
            tx_type=event,
            delta_capability=rule["d_cap"],
            delta_trust=rule["d_trust"],
            delta_cost=rule["d_cost"],
            reason=reason or event,
        )
        self._apply(profile, tx)
        self._ledger.append(tx)
        return tx

    def _apply(self, profile: NodeEconomicProfile, tx: EconomicTransaction):
        profile.capability_score = _clamp(profile.capability_score + tx.delta_capability)
        profile.trust_score      = _clamp(profile.trust_score      + tx.delta_trust)
        profile.cost_score       = _clamp(profile.cost_score       + tx.delta_cost)
        if tx.tx_type in ("success", "usage", "gap_fill"):
            profile.usage_count += 1
        profile.compute_composite()

    # ── Evaluation ────────────────────────────────────────────────────────

    def evaluate_node(self, node_id: str, node_name: str = "") -> NodeEconomicProfile:
        """Force a full re-evaluation of a node's profile."""
        profile = self.get_or_create(node_id, node_name)
        self._seed_profile(profile)
        return profile

    def evaluate_all(self) -> List[NodeEconomicProfile]:
        """Re-evaluate every registered node and update rankings."""
        if self._registry:
            for node in self._registry.list_all():
                self.evaluate_node(node.node_id, node.name)

        # Rank by composite score
        ranked = sorted(
            self._profiles.values(),
            key=lambda p: p.composite_score,
            reverse=True,
        )
        for rank, profile in enumerate(ranked, start=1):
            profile.rank = rank

        return ranked

    # ── Leaderboard ───────────────────────────────────────────────────────

    def leaderboard(self, top_n: int = 10) -> List[dict]:
        ranked = self.evaluate_all()
        return [p.to_dict() for p in ranked[:top_n]]

    def get_profile(self, node_id: str) -> Optional[dict]:
        profile = self._profiles.get(node_id)
        return profile.to_dict() if profile else None

    # ── Ledger ────────────────────────────────────────────────────────────

    def ledger(self, node_id: Optional[str] = None, limit: int = 50) -> List[dict]:
        txs = self._ledger
        if node_id:
            txs = [t for t in txs if t.node_id == node_id]
        return [t.to_dict() for t in txs[-limit:]]

    def summary(self) -> dict:
        profiles = list(self._profiles.values())
        return {
            "tracked_nodes": len(profiles),
            "total_transactions": len(self._ledger),
            "avg_composite_score": (
                sum(p.composite_score for p in profiles) / len(profiles)
                if profiles else 0.0
            ),
            "top_node": (
                max(profiles, key=lambda p: p.composite_score).to_dict()
                if profiles else None
            ),
            "scoring_weights": {
                "capability": W_CAPABILITY,
                "trust": W_TRUST,
                "cost": W_COST,
            },
        }


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))
