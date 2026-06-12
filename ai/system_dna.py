"""
Phase 6 – System DNA
======================
Tracks the evolving "genetic makeup" of the service mesh.

DNA encodes:
  • Active node versions and their performance scores
  • Current routing preferences (which planner/translator/validator)
  • Evolution history (what changed, why, and with what result)

The DNA can be:
  • Snapshotted at any time
  • Compared across versions (diff)
  • Replaced / rolled back to a prior snapshot

This enables the SelfOptimizer to perform true genetic-algorithm-style
evolution rather than just rule-based heuristics.

Usage:
  dna = SystemDNA(knowledge_store)
  snapshot = dna.snapshot(registry, scoring_engine)
  diff = dna.diff(snapshot_v1, snapshot_v2)
  dna.apply(snapshot_id)          # roll back to earlier state
"""
from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DNASnapshot:
    """A point-in-time capture of the system's configuration and performance."""

    def __init__(self, version: str, snapshot_id: str):
        self.snapshot_id = snapshot_id
        self.version = version
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.nodes: Dict[str, dict] = {}       # node_id → {name, version, score}
        self.routes: Dict[str, dict] = {}      # path_key → {success_rate, avg_latency}
        self.capabilities: List[str] = []
        self.composite_health: float = 0.0
        self.active: bool = False
        self.notes: str = ""

    def compute_health(self):
        if not self.nodes:
            self.composite_health = 0.0
            return
        scores = [n.get("score", 0.5) for n in self.nodes.values()]
        self.composite_health = sum(scores) / len(scores)

    def to_dict(self) -> dict:
        return {
            "snapshot_id": self.snapshot_id,
            "version": self.version,
            "created_at": self.created_at,
            "active": self.active,
            "composite_health": round(self.composite_health, 4),
            "node_count": len(self.nodes),
            "route_count": len(self.routes),
            "capabilities": self.capabilities,
            "nodes": self.nodes,
            "routes": self.routes,
            "notes": self.notes,
        }


class SystemDNA:
    """
    Phase 6: System DNA Manager.

    Creates, diffs, and applies snapshots of the system's evolving
    configuration and performance profile.
    """

    _VERSION_PREFIX = "6."

    def __init__(self, knowledge_store=None):
        self._knowledge = knowledge_store
        self._snapshots: List[DNASnapshot] = []
        self._current_snapshot_id: Optional[str] = None
        self._version_counter = 0
        logger.info("SystemDNA initialised (Phase 6)")

    # ── Snapshot ──────────────────────────────────────────────────────────

    def snapshot(
        self,
        registry=None,
        scoring_engine=None,
        memory_engine=None,
        notes: str = "",
    ) -> DNASnapshot:
        """Capture the current system state as a DNA snapshot."""
        self._version_counter += 1
        version = f"{self._VERSION_PREFIX}{self._version_counter}"
        snap_id = f"dna_{str(uuid.uuid4())[:8]}"
        snap = DNASnapshot(version, snap_id)
        snap.notes = notes

        # Capture node information
        if registry:
            try:
                for node in registry.list_all():
                    snap.nodes[node.node_id] = {
                        "name": node.name,
                        "version": getattr(node, "version", "1.0.0"),
                        "score": 0.5,
                        "tags": node.tags,
                    }
            except Exception as exc:
                logger.debug(f"DNA snapshot node error: {exc}")

        # Capture scoring
        if scoring_engine:
            try:
                scores = scoring_engine.list_scores()
                for node_id in snap.nodes:
                    related = [
                        s for s in scores
                        if s.get("source_id") == node_id or s.get("target_id") == node_id
                    ]
                    if related:
                        avg = sum(s.get("connection_score", 0.5) for s in related) / len(related)
                        snap.nodes[node_id]["score"] = avg
            except Exception as exc:
                logger.debug(f"DNA scoring error: {exc}")

        # Capture route memory
        if memory_engine:
            try:
                for route in memory_engine.all_routes():
                    snap.routes[route.get("path_key", "")] = {
                        "success_rate": route.get("success_rate", 0.0),
                        "avg_latency_ms": route.get("avg_latency_ms", 0.0),
                        "runs": route.get("runs", 0),
                        "health": route.get("health", "unknown"),
                    }
            except Exception as exc:
                logger.debug(f"DNA route memory error: {exc}")

        snap.compute_health()

        # Mark previous active → inactive
        for s in self._snapshots:
            s.active = False
        snap.active = True
        self._current_snapshot_id = snap_id
        self._snapshots.append(snap)

        logger.info(f"SystemDNA snapshot {version} ({snap_id}): health={snap.composite_health:.3f}")
        return snap

    # ── Diff ──────────────────────────────────────────────────────────────

    def diff(self, id_a: str, id_b: str) -> dict:
        """
        Compute the difference between two snapshots.

        Returns added/removed nodes and routes, and health delta.
        """
        snap_a = self._find(id_a)
        snap_b = self._find(id_b)
        if not snap_a or not snap_b:
            return {"error": "One or both snapshot IDs not found"}

        nodes_a = set(snap_a.nodes.keys())
        nodes_b = set(snap_b.nodes.keys())

        routes_a = set(snap_a.routes.keys())
        routes_b = set(snap_b.routes.keys())

        # Node score changes
        changed_nodes: List[dict] = []
        for nid in nodes_a & nodes_b:
            score_a = snap_a.nodes[nid].get("score", 0.5)
            score_b = snap_b.nodes[nid].get("score", 0.5)
            delta = score_b - score_a
            if abs(delta) > 0.01:
                changed_nodes.append({
                    "node_id": nid,
                    "name": snap_b.nodes[nid].get("name", "?"),
                    "score_before": round(score_a, 4),
                    "score_after": round(score_b, 4),
                    "delta": round(delta, 4),
                })

        return {
            "from": {"snapshot_id": id_a, "version": snap_a.version},
            "to":   {"snapshot_id": id_b, "version": snap_b.version},
            "health_delta": round(snap_b.composite_health - snap_a.composite_health, 4),
            "nodes_added": list(nodes_b - nodes_a),
            "nodes_removed": list(nodes_a - nodes_b),
            "nodes_changed": sorted(changed_nodes, key=lambda c: abs(c["delta"]), reverse=True),
            "routes_added": list(routes_b - routes_a),
            "routes_removed": list(routes_a - routes_b),
        }

    # ── Apply / rollback ──────────────────────────────────────────────────

    def apply(self, snapshot_id: str) -> bool:
        """
        Activate a prior snapshot (logical rollback marker).

        In production, this would trigger the SelfOptimizer to replace
        nodes back to the versions in the chosen snapshot. Here we mark
        the snapshot as active and log the decision.
        """
        snap = self._find(snapshot_id)
        if not snap:
            return False
        for s in self._snapshots:
            s.active = False
        snap.active = True
        self._current_snapshot_id = snapshot_id
        logger.info(
            f"SystemDNA: activated snapshot {snap.version} ({snapshot_id}), "
            f"health={snap.composite_health:.3f}"
        )
        return True

    # ── Query ─────────────────────────────────────────────────────────────

    def current(self) -> Optional[dict]:
        snap = self._find(self._current_snapshot_id) if self._current_snapshot_id else None
        return snap.to_dict() if snap else None

    def history(self, limit: int = 10) -> List[dict]:
        return [s.to_dict() for s in self._snapshots[-limit:]]

    def get(self, snapshot_id: str) -> Optional[dict]:
        snap = self._find(snapshot_id)
        return snap.to_dict() if snap else None

    def _find(self, snapshot_id: Optional[str]) -> Optional[DNASnapshot]:
        if not snapshot_id:
            return None
        return next((s for s in self._snapshots if s.snapshot_id == snapshot_id), None)

    def summary(self) -> dict:
        current = self._find(self._current_snapshot_id)
        return {
            "total_snapshots": len(self._snapshots),
            "current_version": current.version if current else None,
            "current_health": current.composite_health if current else None,
            "current_snapshot_id": self._current_snapshot_id,
        }
