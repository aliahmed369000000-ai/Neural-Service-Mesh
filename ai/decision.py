from __future__ import annotations
import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class PathScore:
    """Represents a scored path candidate."""
    def __init__(self, path: List[str], score: float, reason: str):
        self.path = path
        self.score = score
        self.reason = reason

    def to_dict(self):
        return {"path": self.path, "score": round(self.score, 4), "reason": self.reason}


class AIDecisionLayer:
    """
    Phase 2 AI Decision Layer.
    Uses rules + heuristics (pluggable for ML in Phase 3).
    Responsibilities:
      - Choose optimal path between nodes
      - Rank multiple path candidates
      - Suggest next node during execution
      - Learn from execution history (simple frequency-based)
    """

    def __init__(self, graph=None, db=None, routing_engine=None, advisor=None,
                 state_lookup=None, reputation_lookup=None):
        self._graph = graph
        self._db = db        # SQLiteStorage for history-based learning
        # routing_engine اختياري (ai/routing_engine.py::RoutingEngine): يجمع
        # مسارات مرشّحة من ذاكرة المسارات المحفوظة (SQLite) + BFS + DFS +
        # knowledge JSON ويقيّمها بدرجات الاتصال التاريخية — بعكس
        # self._path_stats أدناه التي تبدأ فارغة عند كل إعادة تشغيل. إن
        # وُجد يُجرَّب أولاً في choose_path() ثم يُرجَع للمنطق الأصلي عند
        # غياب نتيجة أو أي استثناء، فلا يمكن أن يجعل الاختيار أسوأ من قبل.
        self._routing = routing_engine
        self._path_stats: Dict[str, Dict] = {}   # path_key -> {runs, successes, avg_ms}
        # الشبكة العصبية (ai/node_neural_advisor.py): اختيارية — بدونها يبقى السلوك
        # قواعد/heuristics كما كان بالضبط. state_lookup(node_id)->str|None وreputation_lookup
        # (node_id)->0..1|None يقدّمهما المنسِّق (MeshBundle) من السجل/محرك السمعة.
        self._advisor = advisor
        self._state_lookup = state_lookup
        self._reputation_lookup = reputation_lookup
        logger.info("AIDecisionLayer initialized (rules + heuristics mode)")

    def set_advisor(self, advisor, state_lookup=None, reputation_lookup=None):
        self._advisor = advisor
        if state_lookup is not None:
            self._state_lookup = state_lookup
        if reputation_lookup is not None:
            self._reputation_lookup = reputation_lookup

    # ── الشبكة العصبية: أدوات داخلية آمنة (لا ترفع أبداً) ──────────────────
    def _state_of(self, node_id: str) -> Optional[str]:
        try:
            return self._state_lookup(node_id) if self._state_lookup else None
        except Exception:
            return None

    def _reputation_of(self, node_id: str) -> Optional[float]:
        try:
            return self._reputation_lookup(node_id) if self._reputation_lookup else None
        except Exception:
            return None

    def _neural_p(self, node_id: str) -> Optional[float]:
        """احتمال نجاح عقدة، أو None إن لم تتوفر شبكة."""
        if self._advisor is None:
            return None
        try:
            return self._advisor.predict(node_id, self._state_of(node_id),
                                         self._reputation_of(node_id))
        except Exception as e:
            logger.warning("neural advisor predict failed: %s", e)
            return None

    def observe_outcome(self, node_id: str, success: bool, latency_ms=None,
                        state_before: Optional[str] = None, reputation_before=None) -> None:
        """تدريب آلي على نتيجة تنفيذ فعلية. state_before/reputation_before هما
        قيمتا ما قبل النتيجة (تفادياً لتسريب التسمية للشبكة)."""
        if self._advisor is None:
            return
        try:
            self._advisor.observe(
                node_id, success, latency_ms,
                state_before if state_before is not None else self._state_of(node_id),
                reputation_before if reputation_before is not None else self._reputation_of(node_id),
            )
        except Exception as e:
            logger.warning("neural advisor observe failed: %s", e)

    def set_graph(self, graph):
        self._graph = graph

    def set_db(self, db):
        self._db = db

    # ── Main Decision API ──────────────────────────────────────────────────

    def choose_path(self, start_id: str, end_id: str) -> Optional[List[str]]:
        """Return the best path from start to end using heuristics."""
        if self._routing is not None:
            try:
                routed = self._routing.choose_route(start_id, end_id)
                if routed:
                    logger.info(f"AI: RoutingEngine chose path len={len(routed)}")
                    return routed
            except Exception as e:
                logger.warning(f"AI: RoutingEngine failed, falling back to heuristics: {e}")
        candidates = self._find_all_paths(start_id, end_id, max_paths=5)
        if not candidates:
            logger.warning(f"AI: no paths found from {start_id[:8]} to {end_id[:8]}")
            return None

        scored = [self._score_path(p) for p in candidates]
        scored.sort(key=lambda s: s.score, reverse=True)

        best = scored[0]
        logger.info(f"AI chose path len={len(best.path)} score={best.score:.3f} reason='{best.reason}'")
        return best.path

    def rank_paths(self, start_id: str, end_id: str) -> List[dict]:
        """Return all candidate paths ranked by score."""
        candidates = self._find_all_paths(start_id, end_id, max_paths=10)
        scored = [self._score_path(p) for p in candidates]
        scored.sort(key=lambda s: s.score, reverse=True)
        return [s.to_dict() for s in scored]

    def suggest_next(self, current_node_id: str, context: dict = None) -> Optional[str]:
        """Suggest the next node to execute after current_node_id."""
        if not self._graph:
            return None
        try:
            neighbors = self._graph.get_neighbors(current_node_id)
        except KeyError:
            return None

        if not neighbors:
            return None
        if len(neighbors) == 1:
            return neighbors[0]

        # Score each neighbor
        best_nb, best_score = None, -math.inf
        for nb in neighbors:
            score = self._score_single_node(nb, context or {})
            if score > best_score:
                best_score, best_nb = score, nb

        logger.info(f"AI suggests next: {best_nb[:8] if best_nb else None}")
        return best_nb

    def should_fallback(self, failed_node_id: str, error: str) -> Optional[str]:
        """
        Given a failed node, suggest a fallback neighbor if available.
        Returns a node_id to try next, or None.
        """
        if not self._graph:
            return None
        try:
            # Look for nodes that point to the same targets the failed node points to
            neighbors = self._graph.get_neighbors(failed_node_id)
            if neighbors and self._advisor is not None:
                # الشبكة العصبية تختار الجار الأرجح نجاحاً، وتتخطى المحجور (PAUSED) —
                # الترتيب الأصلي يكسر التعادل فيبقى السلوك القديم لعُقد بلا تاريخ.
                eligible = [n for n in neighbors
                            if n != failed_node_id and self._state_of(n) != "paused"]
                if not eligible:
                    return None
                ranked = self._advisor.rank(eligible, self._state_lookup, self._reputation_lookup)
                logger.info("AI fallback (neural): %s p=%.3f", ranked[0][0][:8], ranked[0][1])
                return ranked[0][0]
            if neighbors:
                logger.info(f"AI fallback: suggesting first available neighbor of failed node")
                return neighbors[0]
        except KeyError:
            pass
        return None

    def learn_from_run(self, run_result: dict):
        """Update internal stats from a completed run (simple heuristic learning)."""
        # تدريب عصبي آلي من كل خطوة فعلية (نجاح/خطأ) — يعمل حتى لو لم يكن هناك path
        for st in (run_result.get("steps") or []):
            if st.get("status") in ("success", "error") and st.get("node_id"):
                self.observe_outcome(st["node_id"], st["status"] == "success",
                                     st.get("duration_ms"), st.get("state_before"))
        path = run_result.get("path", [])
        if not path:
            return
        key = "->".join(p[:8] for p in path)
        if key not in self._path_stats:
            self._path_stats[key] = {"runs": 0, "successes": 0, "total_ms": 0.0}
        s = self._path_stats[key]
        s["runs"] += 1
        if run_result.get("status") == "success":
            s["successes"] += 1
        s["total_ms"] += run_result.get("total_duration_ms") or 0.0
        logger.debug(f"AI learned from run key={key} stats={s}")

    def get_insights(self) -> dict:
        """Return AI-derived insights about the mesh performance."""
        neural = self._advisor.summary() if self._advisor is not None else None
        if not self._path_stats:
            out = {"message": "No execution data yet. Run some pipelines first."}
            if neural:
                out["neural"] = neural
            return out

        insights = []
        for key, s in self._path_stats.items():
            success_rate = s["successes"] / s["runs"] if s["runs"] > 0 else 0
            avg_ms = s["total_ms"] / s["runs"] if s["runs"] > 0 else 0
            insights.append({
                "path_key": key,
                "runs": s["runs"],
                "success_rate": round(success_rate, 3),
                "avg_duration_ms": round(avg_ms, 2),
                "health": "good" if success_rate > 0.8 else "degraded" if success_rate > 0.5 else "critical",
            })

        insights.sort(key=lambda x: x["success_rate"], reverse=True)
        return {
            "total_paths_tracked": len(insights),
            "paths": insights,
            "recommendation": self._overall_recommendation(insights),
            **({"neural": neural} if neural else {}),
        }

    # ── Internal Heuristics ────────────────────────────────────────────────

    def _find_all_paths(self, start: str, end: str, max_paths: int = 5) -> List[List[str]]:
        """DFS to find multiple paths (not just shortest)."""
        if not self._graph:
            return []

        all_paths = []
        stack = [(start, [start])]

        while stack and len(all_paths) < max_paths:
            node, path = stack.pop()
            try:
                neighbors = self._graph.get_neighbors(node)
            except KeyError:
                continue
            for nb in neighbors:
                if nb == end:
                    all_paths.append(path + [end])
                elif nb not in path and len(path) < 20:
                    stack.append((nb, path + [nb]))

        # Also add BFS shortest path if not already found
        try:
            bfs = self._graph.find_path_bfs(start, end)
            if bfs and bfs not in all_paths:
                all_paths.insert(0, bfs)
        except Exception:
            pass

        return all_paths[:max_paths]

    def _score_path(self, path: List[str]) -> PathScore:
        """
        Score a path using multiple heuristics:
        - Shorter paths score higher (length penalty)
        - Historically successful paths score higher
        - Lighter edges (weights) score higher
        """
        if not path:
            return PathScore(path, 0.0, "empty path")

        score = 100.0
        reasons = []

        # 1. Length penalty: prefer shorter paths
        length_penalty = (len(path) - 1) * 8.0
        score -= length_penalty
        reasons.append(f"len={len(path)}")

        # 2. Historical success bonus
        key = "->".join(p[:8] for p in path)
        if key in self._path_stats:
            s = self._path_stats[key]
            success_rate = s["successes"] / s["runs"] if s["runs"] > 0 else 0.5
            history_bonus = success_rate * 20.0
            score += history_bonus
            reasons.append(f"hist_success={success_rate:.2f}")

        # 3. Edge weight scoring (lower cumulative weight = better routing)
        if self._graph:
            total_weight = 0.0
            for i in range(len(path) - 1):
                try:
                    edges = self._graph._adjacency.get(path[i], [])
                    for e in edges:
                        if e.target_id == path[i + 1]:
                            total_weight += e.weight
                            break
                except Exception:
                    total_weight += 1.0
            weight_penalty = max(0, (total_weight - len(path) + 1) * 5.0)
            score -= weight_penalty
            if total_weight > 0:
                reasons.append(f"weight={total_weight:.1f}")

        # الشبكة العصبية (بعد اكتمال تدريبها فقط): المتوسط الهندسي لاحتمالات نجاح عُقد المسار
        if self._advisor is not None and self._advisor.is_trained():
            ps = [self._neural_p(n) for n in path]
            ps = [p for p in ps if p is not None]
            if ps:
                gm = math.exp(sum(math.log(max(p, 1e-6)) for p in ps) / len(ps))
                score += 40.0 * (gm - 0.5)
                reasons.append(f"nn={gm:.2f}")

        return PathScore(path, max(0.0, score), ", ".join(reasons))

    def _score_single_node(self, node_id: str, context: dict) -> float:
        """Score a single candidate node for next-step suggestion."""
        score = 50.0

        # Prefer nodes with matching tags from context
        hints = context.get("prefer_tags", [])
        if hints and self._graph:
            meta = self._graph._node_meta.get(node_id, {})
            tags = meta.get("tags", [])
            matches = len(set(hints) & set(tags))
            score += matches * 10.0

        # Penalize nodes with many outgoing edges (busier nodes)
        if self._graph:
            try:
                out_degree = len(self._graph.get_neighbors(node_id))
                score -= out_degree * 2.0
            except Exception:
                pass

        # الشبكة العصبية (بعد اكتمال تدريبها فقط): ±20 نقطة حسب احتمال النجاح
        if self._advisor is not None and self._advisor.is_trained():
            p = self._neural_p(node_id)
            if p is not None:
                score += 40.0 * (p - 0.5)

        return score

    def _overall_recommendation(self, insights: List[dict]) -> str:
        if not insights:
            return "No data available."
        avg_success = sum(i["success_rate"] for i in insights) / len(insights)
        if avg_success > 0.9:
            return "Mesh is healthy. All paths performing well."
        elif avg_success > 0.7:
            return "Mesh is mostly healthy. Consider investigating degraded paths."
        else:
            return "Mesh health is critical. Multiple paths are failing. Review node implementations."
