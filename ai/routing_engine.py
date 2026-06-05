"""
Phase 3 – Routing Engine
AI-driven dynamic route selection combining:
  - Semantic compatibility scores
  - Historical execution scores (ScoringEngine)
  - Remembered routes (MemoryEngine)
  - Graph topology
"""
from __future__ import annotations
import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


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
    Phase 3 Routing Engine.

    Route selection algorithm (weights sum to 1.0):
      W_semantic  = 0.30  – how well nodes connect semantically
      W_score     = 0.35  – historical connection performance
      W_memory    = 0.25  – remembered route performance
      W_topology  = 0.10  – graph topology (shorter is better)
    """

    W_SEMANTIC = 0.30
    W_SCORE = 0.35
    W_MEMORY = 0.25
    W_TOPOLOGY = 0.10

    def __init__(self, graph=None, semantic_matcher=None,
                 scoring_engine=None, memory_engine=None):
        self._graph = graph
        self._semantic = semantic_matcher
        self._scoring = scoring_engine
        self._memory = memory_engine
        logger.info("RoutingEngine initialised (Phase 3)")

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

        # 4. Semantic chain: build path guided by semantic compatibility
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
            "score": self._history_score(path),
            "memory": self._memory_score(path),
            "topology": self._topology_score(path),
        }
        composite = (
            breakdown["semantic"] * self.W_SEMANTIC +
            breakdown["score"] * self.W_SCORE +
            breakdown["memory"] * self.W_MEMORY +
            breakdown["topology"] * self.W_TOPOLOGY
        )
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
        return "<RoutingEngine (semantic+scoring+memory+topology)>"
