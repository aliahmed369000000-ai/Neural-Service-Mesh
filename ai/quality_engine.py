"""
Phase 15 – Quality Engine
Evaluates every piece of information on a 0-100 scale using five
weighted factors before it is admitted to the knowledge store.

Weights:
  trusted_source          : 30
  cross_source_repetition : 20
  recency                 : 15
  goal_relevance          : 20
  memory_track_record     : 15
"""
from __future__ import annotations

import hashlib
import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Weight table (must sum to 100)
# ---------------------------------------------------------------------------
WEIGHTS: Dict[str, float] = {
    "trusted_source":          30.0,
    "cross_source_repetition": 20.0,
    "recency":                 15.0,
    "goal_relevance":          20.0,
    "memory_track_record":     15.0,
}

# Built-in trusted source list (extend at runtime via QualityEngine.trust())
_DEFAULT_TRUSTED: set = {
    "system",
    "core",
    "verified_feed",
    "internal_sensor",
    "bootstrap",
}


class QualityEngine:
    """
    Rates incoming data items on a 0-100 quality scale.

    Parameters
    ----------
    knowledge_store : optional
        If supplied, used to look up historical track records and
        current active goals.  Any object with .get() / .keys() works.
    """

    def __init__(self, knowledge_store=None):
        self._ks = knowledge_store
        self._trusted_sources: set = set(_DEFAULT_TRUSTED)
        # Cache: content_hash -> list[float]  (past scores)
        self._history: Dict[str, List[float]] = {}
        # Repetition ledger: content_hash -> set of source names
        self._seen_from: Dict[str, set] = {}
        self._total_evaluated: int = 0
        self._score_sum: float = 0.0
        logger.info("QualityEngine initialised (weights=%s)", WEIGHTS)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trust(self, source: str) -> None:
        """Mark a source name as trusted."""
        self._trusted_sources.add(source.lower())

    def distrust(self, source: str) -> None:
        """Remove a source from the trusted set."""
        self._trusted_sources.discard(source.lower())

    def evaluate(self, data: dict) -> float:
        """
        Score a single data item.

        Expected keys in *data*:
            source   (str)  – origin of the data
            content  (str)  – the actual payload / claim
            timestamp(str)  – ISO-8601 creation time (optional)
            tags     (list) – topic tags for goal matching (optional)

        Returns a float in [0.0, 100.0].
        """
        source    = str(data.get("source", "unknown")).lower()
        content   = str(data.get("content", ""))
        timestamp = data.get("timestamp", "")
        tags      = data.get("tags") or []

        content_hash = self._hash(content)

        # --- factor 1: trusted source -----------------------------------
        f_trusted = 100.0 if source in self._trusted_sources else 0.0

        # --- factor 2: cross-source repetition -------------------------
        sources_for_hash = self._seen_from.setdefault(content_hash, set())
        sources_for_hash.add(source)
        unique_sources = len(sources_for_hash)
        # 1 source → 0, 2 → 50, 4+ → 100  (logarithmic)
        f_repetition = min(100.0, math.log(unique_sources, 2) * 50.0) if unique_sources > 1 else 0.0

        # --- factor 3: recency ------------------------------------------
        f_recency = self._score_recency(timestamp)

        # --- factor 4: goal relevance -----------------------------------
        f_goal = self._score_goal_relevance(tags)

        # --- factor 5: memory track record ------------------------------
        past = self._history.get(content_hash, [])
        if past:
            f_memory = min(100.0, sum(past) / len(past))
        else:
            f_memory = 50.0  # neutral prior for unseen content

        raw = (
            WEIGHTS["trusted_source"]          * f_trusted    / 100.0 +
            WEIGHTS["cross_source_repetition"] * f_repetition / 100.0 +
            WEIGHTS["recency"]                 * f_recency    / 100.0 +
            WEIGHTS["goal_relevance"]          * f_goal       / 100.0 +
            WEIGHTS["memory_track_record"]     * f_memory     / 100.0
        )
        score = round(min(100.0, max(0.0, raw)), 4)

        # Persist to history for future evaluations of same content
        self._history.setdefault(content_hash, []).append(score)

        self._total_evaluated += 1
        self._score_sum += score

        logger.debug(
            "evaluate source=%s score=%.2f "
            "(trusted=%.1f rep=%.1f rec=%.1f goal=%.1f mem=%.1f)",
            source, score, f_trusted, f_repetition, f_recency, f_goal, f_memory,
        )
        return score

    def batch_evaluate(self, items: list) -> list:
        """
        Score a list of data items.

        Returns a list of dicts, each containing the original item plus:
            score (float) – quality score 0-100
            grade (str)   – 'A' / 'B' / 'C' / 'D' / 'F'
        """
        results = []
        for item in items:
            score = self.evaluate(item)
            results.append({**item, "score": score, "grade": self._grade(score)})
        return results

    def summary(self) -> dict:
        """Return aggregated statistics for this engine instance."""
        avg = self._score_sum / self._total_evaluated if self._total_evaluated else 0.0
        return {
            "engine": "QualityEngine",
            "total_evaluated": self._total_evaluated,
            "average_score": round(avg, 4),
            "unique_content_seen": len(self._history),
            "trusted_sources_count": len(self._trusted_sources),
            "weights": WEIGHTS,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 85:  return "A"
        if score >= 70:  return "B"
        if score >= 55:  return "C"
        if score >= 40:  return "D"
        return "F"

    def _score_recency(self, timestamp: str) -> float:
        """Returns 0-100; 100 = just now, decays over 30 days."""
        if not timestamp:
            return 40.0  # unknown age → below-average recency
        try:
            dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            age_hours = max(0.0, (now - dt).total_seconds() / 3600.0)
            # Exponential decay: half-life = 72 h
            return round(100.0 * math.exp(-age_hours / (72.0 / math.log(2))), 4)
        except (ValueError, TypeError):
            return 40.0

    def _score_goal_relevance(self, tags: list) -> float:
        """
        Check how many of the data tags match active goals.
        Uses knowledge_store if available; otherwise returns neutral 50.
        """
        if not tags:
            return 50.0
        if self._ks is None:
            return 50.0
        try:
            active_goals = set(self._ks.get("active_goals") or [])
            if not active_goals:
                return 50.0
            tag_set = {str(t).lower() for t in tags}
            overlap = len(tag_set & {g.lower() for g in active_goals})
            return min(100.0, (overlap / len(active_goals)) * 100.0)
        except Exception:
            return 50.0


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from datetime import datetime, timezone, timedelta

    engine = QualityEngine()

    sample_items = [
        {
            "source": "verified_feed",
            "content": "Neural Service Mesh reached v14.0.0 with 14 capability layers.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tags": ["neural", "mesh", "version"],
        },
        {
            "source": "unknown_blog",
            "content": "Some random claim from an unverified source.",
            "timestamp": (datetime.now(timezone.utc) - timedelta(days=10)).isoformat(),
            "tags": [],
        },
        {
            "source": "system",
            "content": "Bootstrap configuration loaded successfully.",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tags": ["system", "boot"],
        },
    ]

    print("=== QualityEngine Demo ===")
    results = engine.batch_evaluate(sample_items)
    for r in results:
        print(f"  [{r['grade']}] {r['score']:6.2f}  source={r['source']}")

    print("\n--- Summary ---")
    for k, v in engine.summary().items():
        print(f"  {k}: {v}")
