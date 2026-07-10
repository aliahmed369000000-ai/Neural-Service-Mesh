"""
Phase 4 – Node Reputation Engine
==================================
Every node in the mesh receives a dynamic reputation score based on:
  - Execution success rate
  - Latency performance
  - Usage frequency
  - Promotion/demotion by optimizer

Reputation directly affects route planning:
  - High-reputation nodes are preferred by RoutingEngine
  - Low-reputation nodes are candidates for removal by OptimizationEngine
"""
from __future__ import annotations
import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class NodeReputation:
    """
    Reputation record for a single node.
    Score range: 0 (untrusted) → 100 (platinum).
    """

    TIER_THRESHOLDS = {
        "platinum": 85.0,
        "gold":     70.0,
        "silver":   50.0,
        "bronze":   30.0,
        "unrated":  0.0,
    }

    def __init__(self, node_id: str, name: str = ""):
        self.node_id = node_id
        self.name = name
        self.total_runs: int = 0
        self.successful_runs: int = 0
        self.failed_runs: int = 0
        self.total_latency_ms: float = 0.0
        self.manual_boost: float = 0.0      # Added by operator via API
        self.is_quarantined: bool = False    # Quarantined nodes skipped in routing
        self.first_seen: str = datetime.now(timezone.utc).isoformat()
        self.last_active: str = self.first_seen

    # ── Computed properties ───────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        return self.successful_runs / self.total_runs if self.total_runs > 0 else 0.5

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_runs if self.total_runs > 0 else 0.0

    @property
    def reputation_score(self) -> float:
        """
        Composite reputation score [0, 100].
          60% — success rate
          20% — latency penalty (lower is better)
          10% — usage (more usage = more data = more reliable score)
          10% — manual boost/penalty
        """
        if self.is_quarantined:
            return 0.0

        # Success component (0-60)
        sr_component = self.success_rate * 60.0

        # Latency component (0-20): 20 pts at 0ms, 0 pts at 5000ms
        lat_penalty = min(self.avg_latency_ms / 5000.0, 1.0)
        lat_component = (1.0 - lat_penalty) * 20.0

        # Usage component (0-10): saturates at 50 runs
        usage_component = min(self.total_runs / 50.0, 1.0) * 10.0

        # Manual boost (0-10)
        boost = max(-10.0, min(10.0, self.manual_boost))

        score = sr_component + lat_component + usage_component + boost
        return round(min(100.0, max(0.0, score)), 2)

    @property
    def tier(self) -> str:
        score = self.reputation_score
        for tier_name, threshold in self.TIER_THRESHOLDS.items():
            if score >= threshold:
                return tier_name
        return "unrated"

    # ── Record execution ──────────────────────────────────────────────────

    def record(self, success: bool, latency_ms: float = 0.0):
        self.total_runs += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        self.last_active = datetime.now(timezone.utc).isoformat()

    # ── Serialization ─────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "reputation_score": self.reputation_score,
            "tier": self.tier,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "is_quarantined": self.is_quarantined,
            "manual_boost": self.manual_boost,
            "first_seen": self.first_seen,
            "last_active": self.last_active,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NodeReputation":
        r = cls(data["node_id"], data.get("name", ""))
        r.total_runs = data.get("total_runs", 0)
        r.successful_runs = data.get("successful_runs", 0)
        r.failed_runs = data.get("failed_runs", 0)
        r.total_latency_ms = data.get("avg_latency_ms", 0.0) * r.total_runs
        r.manual_boost = data.get("manual_boost", 0.0)
        r.is_quarantined = data.get("is_quarantined", False)
        r.first_seen = data.get("first_seen", r.first_seen)
        r.last_active = data.get("last_active", r.last_active)
        return r


class NodeReputationEngine:
    """
    Phase 4 Node Reputation Engine.

    Maintains reputation scores for all nodes. Integrates with:
      - MemoryEngine: reads node execution stats
      - RoutingEngine: provides reputation weights for route scoring
      - OptimizationEngine: flags low-reputation nodes for review
    """

    def __init__(self, knowledge_store=None, memory_engine=None):
        self._reputations: Dict[str, NodeReputation] = {}
        self._knowledge = knowledge_store
        self._memory = memory_engine
        logger.info("NodeReputationEngine initialised (Phase 4)")

    def set_components(self, knowledge=None, memory=None):
        if knowledge:
            self._knowledge = knowledge
        if memory:
            self._memory = memory

    # ── Core operations ───────────────────────────────────────────────────

    def get_reputation(self, node_id: str) -> Optional[NodeReputation]:
        return self._reputations.get(node_id)

    def get_score(self, node_id: str) -> float:
        """Return reputation score (0-100). Defaults to neutral 50 if unknown."""
        rep = self._reputations.get(node_id)
        return rep.reputation_score if rep else 50.0

    def ensure_node(self, node_id: str, name: str = "") -> NodeReputation:
        """Get or create reputation for a node."""
        if node_id not in self._reputations:
            self._reputations[node_id] = NodeReputation(node_id, name)
        return self._reputations[node_id]

    def record_execution(self, node_id: str, name: str,
                          success: bool, latency_ms: float = 0.0):
        """Record a single execution result for a node."""
        rep = self.ensure_node(node_id, name)
        rep.record(success, latency_ms)
        logger.debug(f"NodeReputation: {name} [{node_id[:8]}] score={rep.reputation_score}")

    def update_from_memory(self):
        """Sync reputation scores from MemoryEngine (batch update)."""
        if not self._memory:
            return
        try:
            all_nodes = self._memory.best_nodes(200) + self._memory.worst_nodes(200)
            seen_ids = set()
            for node_data in all_nodes:
                nid = node_data.get("node_id", "")
                if nid in seen_ids:
                    continue
                seen_ids.add(nid)
                rep = self.ensure_node(nid, node_data.get("name", ""))
                # NodeMemory uses "executions", not "total_runs"
                executions = node_data.get("executions", node_data.get("total_runs", 0))
                rep.total_runs = executions
                sr = node_data.get("success_rate", 0.5)
                if sr == 0.5 and executions == 0:
                    # neutral prior — no real data yet
                    rep.successful_runs = 0
                    rep.failed_runs = 0
                else:
                    rep.successful_runs = int(sr * executions)
                    rep.failed_runs = executions - rep.successful_runs
                rep.total_latency_ms = node_data.get("avg_latency_ms", 0.0) * executions
        except Exception as e:
            logger.error(f"NodeReputationEngine.update_from_memory: {e}")

    # ── Quarantine / boost ────────────────────────────────────────────────

    def quarantine(self, node_id: str):
        """Quarantine a node — it will be excluded from routing."""
        rep = self.ensure_node(node_id)
        rep.is_quarantined = True
        logger.warning(f"NodeReputationEngine: node {node_id[:8]} quarantined")

    def unquarantine(self, node_id: str):
        rep = self.ensure_node(node_id)
        rep.is_quarantined = False

    def boost(self, node_id: str, amount: float = 10.0):
        """Manually boost a node's reputation (max +10)."""
        rep = self.ensure_node(node_id)
        rep.manual_boost = min(10.0, rep.manual_boost + amount)

    def penalise(self, node_id: str, amount: float = 10.0):
        """Manually penalise a node's reputation (max -10)."""
        rep = self.ensure_node(node_id)
        rep.manual_boost = max(-10.0, rep.manual_boost - amount)

    # ── Query ─────────────────────────────────────────────────────────────

    def all_reputations(self) -> List[dict]:
        """Return all node reputations sorted by score descending."""
        return sorted(
            [r.to_dict() for r in self._reputations.values()],
            key=lambda x: x["reputation_score"],
            reverse=True,
        )

    def top_nodes(self, n: int = 10) -> List[dict]:
        return self.all_reputations()[:n]

    def quarantined_nodes(self) -> List[dict]:
        return [r.to_dict() for r in self._reputations.values() if r.is_quarantined]

    def low_reputation_nodes(self, threshold: float = 30.0) -> List[dict]:
        return [
            r.to_dict() for r in self._reputations.values()
            if r.reputation_score < threshold and r.total_runs >= 3
        ]

    def summary(self) -> dict:
        reps = list(self._reputations.values())
        if not reps:
            return {"total_nodes": 0}
        scores = [r.reputation_score for r in reps]
        tier_counts: Dict[str, int] = {}
        for r in reps:
            t = r.tier
            tier_counts[t] = tier_counts.get(t, 0) + 1
        return {
            "total_nodes": len(reps),
            "avg_reputation": round(sum(scores) / len(scores), 2),
            "max_reputation": round(max(scores), 2),
            "min_reputation": round(min(scores), 2),
            "tier_distribution": tier_counts,
            "quarantined": sum(1 for r in reps if r.is_quarantined),
        }

    def __repr__(self):
        return f"<NodeReputationEngine nodes={len(self._reputations)}>"
