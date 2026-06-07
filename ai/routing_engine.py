"""
Phase 3 – Routing Engine
AI-driven dynamic route selection combining:
  - Semantic compatibility scores
  - Historical execution scores (ScoringEngine)
  - Remembered routes (MemoryEngine)
  - Graph topology

Knowledge Layer Integration (Phase 3 completion):
  Reads best routes from knowledge/route_memory.json via KnowledgeStore.
  Reads node profiles from knowledge/node_profiles.json for routing hints.
  Uses knowledge-backed data as additional route candidates.

Phase 8 — Real Neural Weights:
  Integrates NeuralWeightLayer (ai/neural_weights.py) so that the four
  routing scalars (W_SEMANTIC, W_SCORE, W_MEMORY, W_TOPOLOGY) are derived
  from a real 10×7 numpy weight matrix rather than hard-coded constants.
  Weights are learned incrementally via train_step() on every scored route,
  persisted to models/classifiers/routing_weights.npy, and reloaded on
  startup automatically.
"""
from __future__ import annotations
import logging
import math
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

# ── Phase 8: import neural weight utilities ──────────────────────────────────
try:
    from ai.neural_weights import NeuralWeightLayer, extract_routing_weights, get_default_layer
    _NEURAL_WEIGHTS_AVAILABLE = True
except ImportError:
    _NEURAL_WEIGHTS_AVAILABLE = False
    logger.warning("NeuralWeightLayer not found — RoutingEngine will use static weights")


class RouteCandidate:
    """A single route candidate with its composite score."""

    def __init__(self, path: List[str], score: float, breakdown: dict, source: str):
        self.path = path
        self.score = score
        self.breakdown = breakdown   # {semantic, memory, scoring, topology}
        self.source = source         # "memory", "semantic", "graph_bfs", "ai_dfs"

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "score": round(self.score, 2),
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "source": self.source,
            "hops": len(self.path) - 1,
        }


class RoutingEngine:
    """
    Phase 3 + Phase 8 Routing Engine.

    Route selection algorithm — weights derived from NeuralWeightLayer (Phase 8).
    Fall-back static defaults (used when neural weights are unavailable):
      W_semantic  = 0.30  – how well nodes connect semantically
      W_score     = 0.35  – historical connection performance
      W_memory    = 0.25  – remembered route performance
      W_topology  = 0.10  – graph topology (shorter is better)

    Phase 8 Neural Weights:
      A 10×7 numpy weight matrix stored in NeuralWeightLayer supplies dynamic,
      learned routing scalars.  The layer is trained incrementally after each
      scored route and saved to disk for persistence across restarts.
    """

    # ── Static fallback weights (Phase 3 defaults) ──────────────────────
    W_SEMANTIC = 0.30
    W_SCORE    = 0.35
    W_MEMORY   = 0.25
    W_TOPOLOGY = 0.10

    # Path used to persist the neural weight matrix
    _WEIGHTS_PATH = "models/classifiers/routing_weights.npy"

    def __init__(self, graph=None, semantic_matcher=None,
                 scoring_engine=None, memory_engine=None):
        self._graph    = graph
        self._semantic = semantic_matcher
        self._scoring  = scoring_engine
        self._memory   = memory_engine
        self._knowledge = None   # KnowledgeStore — injected via set_knowledge_store()

        # ── Phase 8: Initialise neural weight layer ──────────────────────
        self._neural_layer: Optional["NeuralWeightLayer"] = None
        if _NEURAL_WEIGHTS_AVAILABLE:
            self._neural_layer = get_default_layer(self._WEIGHTS_PATH)
            self._sync_weights_from_layer()
            logger.info(
                f"RoutingEngine (Phase 8): NeuralWeightLayer active — "
                f"W_SEMANTIC={self.W_SEMANTIC:.4f}  W_SCORE={self.W_SCORE:.4f}  "
                f"W_MEMORY={self.W_MEMORY:.4f}  W_TOPOLOGY={self.W_TOPOLOGY:.4f}"
            )
        else:
            logger.info("RoutingEngine initialised (Phase 3 — static weights)")

    # ── Phase 8 helpers ───────────────────────────────────────────────────

    def _sync_weights_from_layer(self) -> None:
        """Pull the 4 routing scalars out of the neural layer (normalised)."""
        if self._neural_layer is None:
            return
        w = extract_routing_weights(self._neural_layer)
        self.W_SEMANTIC = w["W_SEMANTIC"]
        self.W_SCORE    = w["W_SCORE"]
        self.W_MEMORY   = w["W_MEMORY"]
        self.W_TOPOLOGY = w["W_TOPOLOGY"]

    def _build_feature_vector(self, breakdown: dict) -> list:
        """
        Construct a 7-element feature vector from a route score breakdown.
        Used as input to NeuralWeightLayer.forward() / train_step().
        """
        sem   = breakdown.get("semantic", 50.0) / 100.0
        score = breakdown.get("score",    50.0) / 100.0
        mem   = breakdown.get("memory",   50.0) / 100.0
        topo  = breakdown.get("topology", 50.0) / 100.0
        # Additional engineered features
        avg         = (sem + score + mem + topo) / 4.0
        sem_x_score = sem * score
        mem_x_topo  = mem * topo
        return [sem, score, mem, topo, avg, sem_x_score, mem_x_topo]

    def _neural_train_on_route(self, breakdown: dict, composite: float) -> None:
        """Train the neural layer on one scored route (online learning)."""
        if self._neural_layer is None:
            return
        x      = self._build_feature_vector(breakdown)
        target = composite / 100.0          # normalise composite score to [0,1]
        try:
            loss = self._neural_layer.train_step(x, target)
            # Re-sync routing weights after every N steps to avoid thrashing
            if self._neural_layer._train_steps % 10 == 0:
                self._sync_weights_from_layer()
                self._persist_neural_weights()
            logger.debug(f"Phase8 neural train_step loss={loss:.6f}")
        except Exception as e:
            logger.warning(f"Phase8 neural train_step failed: {e}")

    def _persist_neural_weights(self) -> None:
        """Save the neural layer weights to disk."""
        if self._neural_layer is None:
            return
        try:
            self._neural_layer.save(self._WEIGHTS_PATH)
        except Exception as e:
            logger.warning(f"Phase8 weight save failed: {e}")

    def get_neural_layer(self) -> Optional["NeuralWeightLayer"]:
        """Return the NeuralWeightLayer instance (Phase 8 API)."""
        return self._neural_layer

    def neural_weights_summary(self) -> dict:
        """Return a dict summary of the current neural weight state."""
        if self._neural_layer is None:
            return {"enabled": False, "reason": "NeuralWeightLayer not available"}
        return {
            "enabled": True,
            "layer": self._neural_layer.summary(),
            "routing_scalars": {
                "W_SEMANTIC": self.W_SEMANTIC,
                "W_SCORE":    self.W_SCORE,
                "W_MEMORY":   self.W_MEMORY,
                "W_TOPOLOGY": self.W_TOPOLOGY,
            },
        }

    def set_knowledge_store(self, ks) -> None:
        """Inject the KnowledgeStore to enable knowledge-backed route discovery."""
        self._knowledge = ks
        logger.info("RoutingEngine: KnowledgeStore connected")

    def set_graph(self, graph):
        self._graph = graph

    def set_components(self, semantic=None, scoring=None, memory=None):
        if semantic:
            self._semantic = semantic
        if scoring:
            self._scoring = scoring
        if memory:
            self._memory = memory

    # ── Main routing API ───────────────────────────────────────────────────

    def choose_route(self, start_id: str, end_id: str,
                     max_candidates: int = 8) -> Optional[List[str]]:
        """
        Return the best path from start to end.
        Returns None if no path exists.
        """
        candidates = self.rank_routes(start_id, end_id, max_candidates)
        if not candidates:
            return None
        best = candidates[0]
        logger.info(
            f"RoutingEngine chose path len={len(best.path)} "
            f"score={best.score:.2f} source={best.source}"
        )
        return best.path

    def rank_routes(self, start_id: str, end_id: str,
                    max_candidates: int = 8) -> List[RouteCandidate]:
        """Return all candidates ranked by composite score."""
        raw_paths = self._discover_paths(start_id, end_id, max_candidates)
        if not raw_paths:
            return []

        scored: List[RouteCandidate] = []
        for path, source in raw_paths:
            candidate = self._score_candidate(path, source)
            scored.append(candidate)

        # Deduplicate (same path from different discovery methods)
        seen = set()
        unique = []
        for c in sorted(scored, key=lambda x: x.score, reverse=True):
            key = "->".join(c.path)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        return unique

    # ── Path discovery ─────────────────────────────────────────────────────

    def _discover_paths(self, start_id: str, end_id: str,
                        max_paths: int) -> List[Tuple[List[str], str]]:
        """Collect candidate paths from multiple sources."""
        results: List[Tuple[List[str], str]] = []

        # 1. Memory: previously successful routes
        if self._memory:
            remembered = self._memory.recall_best_routes(start_id, end_id, top_k=3)
            for rm in remembered:
                if rm.path:
                    results.append((rm.path, "memory"))

        # 2. BFS shortest path
        if self._graph:
            try:
                bfs = self._graph.find_path_bfs(start_id, end_id)
                if bfs:
                    results.append((bfs, "graph_bfs"))
            except Exception:
                pass

        # 3. DFS multiple paths
        if self._graph:
            dfs_paths = self._dfs_paths(start_id, end_id, max_paths=5)
            for p in dfs_paths:
                results.append((p, "ai_dfs"))

        # 4. Knowledge-backed routes from JSON layer
        if self._knowledge:
            try:
                best_from_ks = self._knowledge.get_best_routes(top_k=3)
                for kr in best_from_ks:
                    path = kr.get("path", [])
                    if path and path[0] == start_id and path[-1] == end_id:
                        results.append((path, "knowledge_json"))
            except Exception as ke:
                logger.debug(f"RoutingEngine: knowledge read error: {ke}")

        # 5. Semantic chain: build path guided by semantic compatibility
        if self._semantic and self._graph:
            sem_path = self._semantic_guided_path(start_id, end_id)
            if sem_path:
                results.append((sem_path, "semantic"))

        return results[:max_paths]

    def _dfs_paths(self, start: str, end: str,
                   max_paths: int = 5) -> List[List[str]]:
        if not self._graph:
            return []
        all_paths: List[List[str]] = []
        stack = [(start, [start])]
        while stack and len(all_paths) < max_paths:
            node, path = stack.pop()
            try:
                neighbours = self._graph.get_neighbors(node)
            except KeyError:
                continue
            for nb in neighbours:
                if nb == end:
                    all_paths.append(path + [end])
                elif nb not in path and len(path) < 15:
                    stack.append((nb, path + [nb]))
        return all_paths

    def _semantic_guided_path(self, start: str, end: str,
                               max_hops: int = 10) -> Optional[List[str]]:
        """
        Build a path by greedily picking the next node with the highest
        semantic compatibility toward the end node.
        """
        if not self._graph or not self._semantic:
            return None
        path = [start]
        visited = {start}
        current = start

        for _ in range(max_hops):
            if current == end:
                return path
            try:
                neighbours = self._graph.get_neighbors(current)
            except KeyError:
                break
            neighbours = [n for n in neighbours if n not in visited]
            if not neighbours:
                break
            if end in neighbours:
                path.append(end)
                return path
            # Pick neighbour with best semantic compatibility toward end
            best_nb, best_score = None, -1.0
            for nb in neighbours:
                score = self._semantic.compatibility_score(nb, end)
                if score > best_score:
                    best_score, best_nb = score, nb
            if best_nb is None:
                best_nb = neighbours[0]
            path.append(best_nb)
            visited.add(best_nb)
            current = best_nb

        return None  # Couldn't reach end semantically

    # ── Scoring ────────────────────────────────────────────────────────────

    def _score_candidate(self, path: List[str], source: str) -> RouteCandidate:
        breakdown = {
            "semantic": self._semantic_score(path),
            "score":    self._history_score(path),
            "memory":   self._memory_score(path),
            "topology": self._topology_score(path),
        }
        composite = (
            breakdown["semantic"] * self.W_SEMANTIC +
            breakdown["score"]    * self.W_SCORE    +
            breakdown["memory"]   * self.W_MEMORY   +
            breakdown["topology"] * self.W_TOPOLOGY
        )
        # Phase 8: train the neural layer on this route's features + score
        self._neural_train_on_route(breakdown, composite)
        return RouteCandidate(path, composite, breakdown, source)

    def _semantic_score(self, path: List[str]) -> float:
        """Average pairwise semantic compatibility along the path (0-100)."""
        if not self._semantic or len(path) < 2:
            return 50.0
        scores = []
        for i in range(len(path) - 1):
            s = self._semantic.compatibility_score(path[i], path[i + 1])
            scores.append(s * 100.0)   # 0-100
        return sum(scores) / len(scores) if scores else 50.0

    def _history_score(self, path: List[str]) -> float:
        """Aggregate historical connection score (0-100)."""
        if not self._scoring or len(path) < 2:
            return 50.0
        return self._scoring.get_path_score(path)

    def _memory_score(self, path: List[str]) -> float:
        """Score from route memory (0-100). Neutral 50 if not remembered."""
        if not self._memory:
            return 50.0
        rm = self._memory.recall_route(path)
        if rm is None:
            return 50.0
        return rm.memory_score

    def _topology_score(self, path: List[str]) -> float:
        """Shorter paths score higher. Max 100 for 1-hop, decays with length."""
        if not path:
            return 0.0
        hops = max(len(path) - 1, 1)
        # Score: 100 / (1 + 0.3 * (hops - 1))
        return round(100.0 / (1.0 + 0.3 * (hops - 1)), 2)

    # ── Utility ────────────────────────────────────────────────────────────

    def explain_route(self, path: List[str]) -> dict:
        """Return a detailed explanation of a given route."""
        candidate = self._score_candidate(path, "manual")
        explanation = {
            "path": path,
            "hops": len(path) - 1,
            "composite_score": round(candidate.score, 2),
            "breakdown": candidate.breakdown,
            "weights": {
                "semantic": self.W_SEMANTIC,
                "score": self.W_SCORE,
                "memory": self.W_MEMORY,
                "topology": self.W_TOPOLOGY,
            },
        }
        if self._semantic and len(path) >= 2:
            edge_details = []
            for i in range(len(path) - 1):
                s = path[i]; t = path[i + 1]
                edge_details.append({
                    "edge": f"{s[:8]}->{t[:8]}",
                    "semantic_score": round(self._semantic.compatibility_score(s, t) * 100, 2),
                    "connection_score": self._scoring.get_score(s, t).connection_score
                    if self._scoring else None,
                })
            explanation["edge_details"] = edge_details
        return explanation

    def __repr__(self):
        neural = "neural-weights" if self._neural_layer is not None else "static-weights"
        return f"<RoutingEngine (semantic+scoring+memory+topology | {neural})>"
