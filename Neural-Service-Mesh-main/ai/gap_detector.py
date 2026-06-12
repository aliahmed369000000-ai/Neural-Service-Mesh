"""
Phase 5 – Gap Detection Engine
=================================
Scans the service graph for structural and semantic gaps.

Detects three types of gaps:
  1. ROUTING_GAP:     Node A → ??? → Node B  (no path between nodes)
  2. CAPABILITY_GAP:  Repeated failures suggest a missing transformer
  3. SEMANTIC_GAP:    High semantic distance between connected nodes

Output example:
  {
    "missing_service": "TextTranslator",
    "confidence": 0.91,
    "gap_type": "semantic",
    "source_node": {...},
    "target_node": {...}
  }
"""
from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class DetectedGap:
    """Represents a single detected gap in the service mesh."""

    GAP_TYPES = ("routing", "capability", "semantic", "repeated_failure")

    def __init__(
        self,
        gap_type: str,
        missing_service: str,
        confidence: float,
        source_node: dict,
        target_node: dict,
        evidence: Optional[List[str]] = None,
    ):
        self.gap_id = f"gap_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}_{gap_type[:3]}"
        self.gap_type = gap_type
        self.missing_service = missing_service
        self.confidence = confidence
        self.source_node = source_node
        self.target_node = target_node
        self.evidence = evidence or []
        self.detected_at = datetime.now(timezone.utc).isoformat()
        self.resolved = False

    def to_dict(self) -> dict:
        return {
            "gap_id": self.gap_id,
            "gap_type": self.gap_type,
            "missing_service": self.missing_service,
            "confidence": round(self.confidence, 4),
            "source_node": self.source_node,
            "target_node": self.target_node,
            "evidence": self.evidence,
            "detected_at": self.detected_at,
            "resolved": self.resolved,
        }


class GapDetectionEngine:
    """
    Phase 5: Scans the neural service mesh for structural and semantic gaps.

    Works in three passes:
      Pass 1 – Routing gaps: nodes reachable in the global registry but not
               connected to any path.
      Pass 2 – Capability gaps: execution history reveals repeated failures
               at the same node-to-node transition.
      Pass 3 – Semantic gaps: high semantic distance between directly connected
               nodes suggests a missing intermediary.
    """

    # Failure threshold before we raise a capability gap
    FAILURE_THRESHOLD = 3
    # Semantic distance above this triggers a semantic gap
    SEMANTIC_GAP_THRESHOLD = 0.6
    # Min confidence to report a gap
    MIN_CONFIDENCE = 0.5

    def __init__(
        self,
        graph=None,
        memory_engine=None,
        semantic_matcher=None,
        knowledge_store=None,
        scoring_engine=None,
    ):
        self._graph = graph
        self._memory = memory_engine
        self._semantic = semantic_matcher
        self._knowledge = knowledge_store
        self._scoring = scoring_engine
        self._detected_gaps: List[DetectedGap] = []
        self._scan_count = 0
        logger.info("GapDetectionEngine initialised (Phase 5)")

    def set_graph(self, g):
        self._graph = g

    def set_memory_engine(self, m):
        self._memory = m

    def set_semantic_matcher(self, sm):
        self._semantic = sm

    def set_knowledge_store(self, ks):
        self._knowledge = ks

    def set_scoring_engine(self, se):
        self._scoring = se

    # ── Main scan ──────────────────────────────────────────────────────────

    def scan(self) -> List[DetectedGap]:
        """
        Run a full gap scan. Returns all new gaps detected.
        """
        self._scan_count += 1
        new_gaps: List[DetectedGap] = []

        new_gaps.extend(self._detect_routing_gaps())
        new_gaps.extend(self._detect_capability_gaps())
        new_gaps.extend(self._detect_semantic_gaps())

        # Deduplicate against previously found gaps
        existing_keys = {
            (g.source_node.get("node_id", ""), g.target_node.get("node_id", ""), g.gap_type)
            for g in self._detected_gaps
            if not g.resolved
        }
        novel = [
            g for g in new_gaps
            if (g.source_node.get("node_id", ""), g.target_node.get("node_id", ""), g.gap_type)
            not in existing_keys
        ]

        self._detected_gaps.extend(novel)
        logger.info(f"Gap scan #{self._scan_count}: {len(novel)} new gap(s) detected (total={len(self._detected_gaps)})")

        # Persist to knowledge store
        self._persist_gaps(novel)

        return novel

    # ── Pass 1: Routing gaps ───────────────────────────────────────────────

    def _detect_routing_gaps(self) -> List[DetectedGap]:
        """Find nodes with no outgoing or incoming connections (isolated nodes)."""
        gaps = []
        if not self._graph:
            return gaps

        node_list = self._graph.list_nodes()
        if len(node_list) < 2:
            return gaps

        # Find nodes with no outgoing connections (sinks that aren't outputs)
        for node_meta in node_list:
            nid = node_meta.get("node_id", "")
            neighbors = self._graph.get_neighbors(nid)

            # Check if this node has downstream nodes that don't connect back
            for other_meta in node_list:
                oid = other_meta.get("node_id", "")
                if oid == nid:
                    continue

                # Check if no path exists between them
                try:
                    path = self._graph.find_path_bfs(nid, oid)
                    if path is None and len(node_list) > 3:
                        # Large enough graph, missing path is a real gap
                        missing = self._suggest_missing_service(node_meta, other_meta, "routing")
                        gap = DetectedGap(
                            gap_type="routing",
                            missing_service=missing,
                            confidence=0.6,
                            source_node=node_meta,
                            target_node=other_meta,
                            evidence=[
                                f"No path found: {node_meta.get('name', nid[:8])} → {other_meta.get('name', oid[:8])}",
                                f"Graph has {len(node_list)} nodes",
                            ],
                        )
                        gaps.append(gap)
                        if len(gaps) >= 5:  # Cap to avoid explosion
                            return gaps
                except Exception:
                    pass
        return gaps

    # ── Pass 2: Capability gaps ────────────────────────────────────────────

    def _detect_capability_gaps(self) -> List[DetectedGap]:
        """Detect node-to-node transitions with high failure rates."""
        gaps = []
        if not self._scoring:
            return gaps

        try:
            scores = self._scoring.list_scores()
        except Exception:
            return gaps

        for score_entry in scores:
            total = score_entry.get("total_runs", 0)
            failures = score_entry.get("failure_count", 0)
            sr = score_entry.get("success_rate", 1.0)

            if total >= self.FAILURE_THRESHOLD and sr < 0.4:
                src_id = score_entry.get("source_id", "")
                tgt_id = score_entry.get("target_id", "")
                src_meta = self._get_node_meta(src_id)
                tgt_meta = self._get_node_meta(tgt_id)

                missing = self._suggest_missing_service(src_meta, tgt_meta, "capability")
                confidence = min(0.95, 0.5 + (failures / max(total, 1)) * 0.5)

                gap = DetectedGap(
                    gap_type="capability",
                    missing_service=missing,
                    confidence=confidence,
                    source_node=src_meta,
                    target_node=tgt_meta,
                    evidence=[
                        f"Success rate: {sr:.1%} over {total} runs",
                        f"Failures: {failures}",
                        f"Connection score: {score_entry.get('connection_score', 0):.2f}",
                    ],
                )
                gaps.append(gap)

        return gaps

    # ── Pass 3: Semantic gaps ──────────────────────────────────────────────

    def _detect_semantic_gaps(self) -> List[DetectedGap]:
        """Find directly connected nodes with low semantic compatibility."""
        gaps = []
        if not self._semantic or not self._graph:
            return gaps

        node_list = self._graph.list_nodes()
        for node_meta in node_list:
            nid = node_meta.get("node_id", "")
            try:
                neighbors = self._graph.get_neighbors(nid)
            except Exception:
                continue

            for neighbor_id in neighbors:
                try:
                    score = self._semantic.compatibility_score(nid, neighbor_id)
                    # Low compatibility on a direct edge = semantic gap
                    if score < (1.0 - self.SEMANTIC_GAP_THRESHOLD):
                        tgt_meta = self._get_node_meta(neighbor_id)
                        missing = self._suggest_missing_service(node_meta, tgt_meta, "semantic")
                        confidence = min(0.9, 0.5 + (self.SEMANTIC_GAP_THRESHOLD - score) * 0.8)

                        if confidence >= self.MIN_CONFIDENCE:
                            gap = DetectedGap(
                                gap_type="semantic",
                                missing_service=missing,
                                confidence=confidence,
                                source_node=node_meta,
                                target_node=tgt_meta,
                                evidence=[
                                    f"Semantic compatibility: {score:.3f} (threshold: {1.0 - self.SEMANTIC_GAP_THRESHOLD:.2f})",
                                    f"Direct edge exists but semantic mismatch detected",
                                ],
                            )
                            gaps.append(gap)
                except Exception:
                    pass
        return gaps

    # ── Helpers ────────────────────────────────────────────────────────────

    def _get_node_meta(self, node_id: str) -> dict:
        """Get node metadata from graph or knowledge store."""
        if self._graph and self._graph.has_node(node_id):
            nodes = self._graph.list_nodes()
            for n in nodes:
                if n.get("node_id") == node_id:
                    return n
        return {"node_id": node_id, "name": node_id[:8], "capability": ""}

    def _suggest_missing_service(self, source: dict, target: dict, gap_type: str) -> str:
        """Heuristically suggest a name for the missing service."""
        src_name = source.get("name", "Source").replace("Node", "").replace("Service", "")
        tgt_name = target.get("name", "Target").replace("Node", "").replace("Service", "")
        src_tags = source.get("tags", [])
        tgt_tags = target.get("tags", [])

        # Look for clues in names and tags
        all_tags = set(src_tags + tgt_tags)

        if "text" in str(src_name).lower() and "output" in str(tgt_name).lower():
            return "TextNormalizer"
        if "input" in str(src_name).lower() and "process" in str(tgt_name).lower():
            return "DataPreprocessor"
        if "process" in str(src_name).lower() and "output" in str(tgt_name).lower():
            return "ResultFormatter"

        keyword_map = {
            "translate": "TextTranslator",
            "sentiment": "SentimentAnalyzer",
            "clean": "DataCleaner",
            "transform": "DataTransformer",
            "validate": "DataValidator",
            "aggregate": "DataAggregator",
            "enrich": "DataEnricher",
            "filter": "DataFilter",
            "normalize": "DataNormalizer",
            "format": "DataFormatter",
        }
        for kw, name in keyword_map.items():
            if kw in str(source).lower() or kw in str(target).lower():
                return name

        return f"{src_name}To{tgt_name}Adapter"

    def _persist_gaps(self, gaps: List[DetectedGap]):
        """Persist detected gaps to knowledge store."""
        if not self._knowledge or not gaps:
            return
        try:
            existing = {}
            try:
                existing = self._knowledge.read_custom("detected_gaps") or {}
            except Exception:
                pass
            for gap in gaps:
                existing[gap.gap_id] = gap.to_dict()
            self._knowledge.write_custom("detected_gaps", existing)
        except Exception as e:
            logger.warning(f"Could not persist gaps: {e}")

    # ── Public API ─────────────────────────────────────────────────────────

    def mark_resolved(self, gap_id: str) -> bool:
        for gap in self._detected_gaps:
            if gap.gap_id == gap_id:
                gap.resolved = True
                return True
        return False

    def all_gaps(self, include_resolved: bool = False) -> List[dict]:
        return [
            g.to_dict()
            for g in self._detected_gaps
            if include_resolved or not g.resolved
        ]

    def summary(self) -> dict:
        active = [g for g in self._detected_gaps if not g.resolved]
        by_type: Dict[str, int] = {}
        for g in active:
            by_type[g.gap_type] = by_type.get(g.gap_type, 0) + 1
        return {
            "scan_count": self._scan_count,
            "total_detected": len(self._detected_gaps),
            "active_gaps": len(active),
            "resolved_gaps": len(self._detected_gaps) - len(active),
            "by_type": by_type,
        }
