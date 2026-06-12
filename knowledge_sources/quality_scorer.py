"""
Knowledge Sources — Quality Scoring Layer
==========================================
Assigns a quality score (0–100) to every KnowledgeItem before it
is written to the knowledge store.

Score Components
----------------
  source_trust    (25 pts) — inherited trust from registered source
  relevance       (20 pts) — topical relevance to system's world model
  novelty         (20 pts) — how new/unique the item is vs existing knowledge
  consistency     (20 pts) — internal coherence and factual stability
  usefulness      (15 pts) — actionability / concept density

The scorer is pluggable: each component is a separate method so
future ML-based components can replace the heuristics incrementally.
"""
from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Set

from knowledge_sources.source_metadata import (
    KnowledgeItem, SourceMetadata, SourceType
)

logger = logging.getLogger(__name__)


# ── Score Breakdown ────────────────────────────────────────────────────────

@dataclass
class QualityScore:
    total:          float = 0.0     # 0–100, final score
    source_trust:   float = 0.0     # 0–25
    relevance:      float = 0.0     # 0–20
    novelty:        float = 0.0     # 0–20
    consistency:    float = 0.0     # 0–20
    usefulness:     float = 0.0     # 0–15
    grade:          str   = "F"     # A/B/C/D/F
    rationale:      str   = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def is_acceptable(self) -> bool:
        return self.total >= 40.0


def _grade(score: float) -> str:
    if score >= 85: return "A"
    if score >= 70: return "B"
    if score >= 55: return "C"
    if score >= 40: return "D"
    return "F"


# ── Quality Scorer ─────────────────────────────────────────────────────────

class QualityScorer:
    """
    Pluggable quality scoring engine.

    Usage:
        scorer = QualityScorer()
        qs = scorer.score(item, source=source_meta, existing_ids=known_ids)
        if qs.is_acceptable:
            ingest(item)
    """

    # Weights must sum to 100
    _WEIGHTS = {
        "source_trust":  25,
        "relevance":     20,
        "novelty":       20,
        "consistency":   20,
        "usefulness":    15,
    }

    # Topic keywords that boost relevance for this system
    _RELEVANCE_KEYWORDS: Set[str] = {
        "knowledge", "concept", "definition", "principle", "law", "rule",
        "science", "history", "geography", "language", "mathematics",
        "verse", "ayah", "surah", "quran", "allah", "prophet",
        "data", "information", "fact", "evidence", "source",
        "معرفة", "علم", "حكم", "آية", "سورة", "الله", "قرآن",
    }

    def __init__(
        self,
        relevance_keywords: Optional[Set[str]] = None,
        min_useful_length: int = 30,
    ):
        self._keywords       = relevance_keywords or self._RELEVANCE_KEYWORDS
        self._min_useful_len = min_useful_length
        self._scored_count   = 0
        self._score_history: List[float] = []   # last 1000 scores

    # ── Main API ───────────────────────────────────────────────────────────

    def score(
        self,
        item: KnowledgeItem,
        source: Optional[SourceMetadata] = None,
        existing_references: Optional[Set[str]] = None,
    ) -> QualityScore:
        """Compute a full QualityScore for a KnowledgeItem."""

        trust_pts  = self._score_trust(item, source)
        rel_pts    = self._score_relevance(item)
        nov_pts    = self._score_novelty(item, existing_references)
        cons_pts   = self._score_consistency(item)
        use_pts    = self._score_usefulness(item)

        total = round(trust_pts + rel_pts + nov_pts + cons_pts + use_pts, 2)
        total = min(100.0, max(0.0, total))

        rationale = (
            f"trust={trust_pts:.1f}/25  "
            f"relevance={rel_pts:.1f}/20  "
            f"novelty={nov_pts:.1f}/20  "
            f"consistency={cons_pts:.1f}/20  "
            f"usefulness={use_pts:.1f}/15"
        )

        qs = QualityScore(
            total        = total,
            source_trust = trust_pts,
            relevance    = rel_pts,
            novelty      = nov_pts,
            consistency  = cons_pts,
            usefulness   = use_pts,
            grade        = _grade(total),
            rationale    = rationale,
        )

        self._scored_count += 1
        self._score_history.append(total)
        if len(self._score_history) > 1000:
            self._score_history.pop(0)

        return qs

    def score_batch(
        self,
        items: List[KnowledgeItem],
        source: Optional[SourceMetadata] = None,
        existing_references: Optional[Set[str]] = None,
    ) -> List[QualityScore]:
        return [self.score(item, source, existing_references) for item in items]

    # ── Component Scorers ──────────────────────────────────────────────────

    def _score_trust(
        self,
        item: KnowledgeItem,
        source: Optional[SourceMetadata],
    ) -> float:
        """Up to 25 pts. Scripture gets the maximum."""
        if item.source_type == SourceType.SCRIPTURE:
            return 25.0

        effective = item.trust_score
        if source:
            effective = max(effective, source.trust_score)
        return round(effective * 25.0, 2)

    def _score_relevance(self, item: KnowledgeItem) -> float:
        """
        Up to 20 pts.
        Keyword overlap between item content and system relevance vocabulary.
        """
        text   = (item.raw_content + " " + item.raw_reference).lower()
        tokens = set(re.split(r"\W+", text))
        overlap = tokens & {k.lower() for k in self._keywords}

        if not tokens:
            return 0.0

        # log scale so a few keywords still give decent score
        raw   = math.log1p(len(overlap)) / math.log1p(len(self._keywords))
        pts   = round(raw * 20.0, 2)

        # Bonus: if item has derived concepts, it's more relevant
        if item.derived_concepts:
            pts = min(20.0, pts + 3.0)

        return pts

    def _score_novelty(
        self,
        item: KnowledgeItem,
        existing_references: Optional[Set[str]],
    ) -> float:
        """
        Up to 20 pts.
        Full score if the reference is new. Partial if unknown set.
        """
        if existing_references is None:
            return 12.0   # neutral when we can't check

        if item.raw_reference and item.raw_reference in existing_references:
            return 0.0    # exact duplicate reference

        # Content-based novelty (approximate): unique words ratio
        words  = re.split(r"\W+", item.raw_content.lower())
        unique = len(set(words))
        ratio  = unique / max(len(words), 1)
        return round(ratio * 20.0, 2)

    def _score_consistency(self, item: KnowledgeItem) -> float:
        """
        Up to 20 pts.
        Heuristic: checks internal structure, encoding, language markers.
        """
        content = item.raw_content or ""
        pts     = 20.0

        # Deduct for very short items
        if len(content) < self._min_useful_len:
            pts -= 8.0

        # Deduct for excessive punctuation (possible corruption)
        punct_ratio = sum(1 for c in content if not c.isalnum() and not c.isspace())
        punct_ratio /= max(len(content), 1)
        if punct_ratio > 0.4:
            pts -= 6.0

        # Deduct for URL-only content
        if re.fullmatch(r"https?://\S+", content.strip()):
            pts -= 10.0

        # Bonus: has a clear reference (surah:ayah, URL, doi, etc.)
        if item.raw_reference:
            pts = min(20.0, pts + 2.0)

        return max(0.0, round(pts, 2))

    def _score_usefulness(self, item: KnowledgeItem) -> float:
        """
        Up to 15 pts.
        Based on: derived concepts available, content density, tags.
        """
        pts = 0.0

        # Has derived concepts
        if item.derived_concepts:
            pts += min(5.0, len(item.derived_concepts) * 1.0)

        # Has tags
        if item.derived_tags:
            pts += min(3.0, len(item.derived_tags) * 0.5)

        # Content density (information per character)
        content = item.raw_content or ""
        words   = [w for w in re.split(r"\W+", content) if w]
        if words:
            avg_len = sum(len(w) for w in words) / len(words)
            density = min(1.0, avg_len / 8.0)
            pts    += density * 5.0

        # Has summary
        if item.derived_summary:
            pts += 2.0

        return min(15.0, round(pts, 2))

    # ── Admin ──────────────────────────────────────────────────────────────

    def add_relevance_keywords(self, keywords: List[str]) -> None:
        self._keywords.update(k.lower() for k in keywords)

    def summary(self) -> Dict[str, Any]:
        scores = self._score_history
        return {
            "total_scored":  self._scored_count,
            "avg_score":     round(sum(scores) / max(len(scores), 1), 2),
            "max_score":     max(scores, default=0.0),
            "min_score":     min(scores, default=0.0),
            "weights":       self._WEIGHTS,
        }
