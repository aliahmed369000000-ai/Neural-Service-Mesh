"""
Phase 11 — Episodic Memory Engine
===================================
Transforms short-term routing experiences into long-term wisdom.

Biological inspiration:
  - Hippocampus  → EpisodicStore   (raw experiences, fast write)
  - Neocortex    → SemanticMemory  (abstract patterns, slow consolidation)
  - Sleep        → Consolidator    (transfers hippocampus → neocortex)
  - Forgetting   → EbbinghausCurve (weak memories fade, strong ones persist)

Three memory tiers:
  1. Working Memory   — last 50 experiences, instant access
  2. Episodic Store   — thousands of tagged episodes, SQLite-backed
  3. Semantic Memory  — generalised rules extracted from many episodes

Memory lifecycle:
  Experience → Working Memory → (consolidation) → Episodic Store
                                               → (abstraction) → Semantic Memory
                                               → (forgetting) → deleted
"""
from __future__ import annotations

import json
import logging
import math
import random
import sqlite3
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
WORKING_MEMORY_SIZE    = 50
EPISODIC_DB_PATH       = "memory/episodic.db"
SEMANTIC_DB_PATH       = "memory/semantic.db"
CONSOLIDATION_INTERVAL = 60      # seconds between consolidation runs
FORGETTING_BASE_RATE   = 0.01    # fraction of weak memories pruned per cycle
MIN_STRENGTH_TO_KEEP   = 0.15    # memories below this are candidates for deletion
ABSTRACTION_THRESHOLD  = 10      # min episodes needed to form a semantic rule
MAX_EPISODIC_SIZE      = 100_000 # hard cap on episodic store


# ─────────────────────────────────────────────────────────────────────────────
#  Ebbinghaus Forgetting Curve
# ─────────────────────────────────────────────────────────────────────────────

class EbbinghausCurve:
    """
    Models memory strength decay over time.

    Formula: S(t) = S0 * e^(-t / (stability * R))
      S0        = initial strength (based on reward + surprise)
      t         = hours since last recall
      stability = how well-consolidated the memory is (0-1)
      R         = base retention constant (24h by default)

    Each time a memory is recalled/replayed, stability increases
    (spaced repetition effect).
    """

    R = 24.0   # base retention in hours

    @staticmethod
    def strength(
        initial_strength: float,
        hours_since_recall: float,
        stability: float,
    ) -> float:
        """Current memory strength [0, 1]."""
        if hours_since_recall <= 0:
            return initial_strength
        decay = math.exp(-hours_since_recall / (stability * EbbinghausCurve.R + 1e-6))
        return float(np.clip(initial_strength * decay, 0.0, 1.0))

    @staticmethod
    def initial_strength(reward: float, surprise: float) -> float:
        """
        Strong memories form from high reward OR high surprise.
        surprise = how different this outcome was from expectation.
        """
        return float(np.clip(0.4 + 0.4 * abs(reward) + 0.2 * surprise, 0.1, 1.0))

    @staticmethod
    def stability_after_recall(current_stability: float) -> float:
        """Spaced repetition: each recall strengthens the memory."""
        return float(np.clip(current_stability + 0.1 * (1 - current_stability), 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Episode: a complete remembered event
# ─────────────────────────────────────────────────────────────────────────────

class Episode:
    """
    A single remembered experience with full metadata.

    Unlike a raw Experience (from signal_stream), an Episode has:
    - memory_strength  : how vivid the memory is (decays over time)
    - stability        : how consolidated it is (grows with recall)
    - recall_count     : how many times it was replayed
    - surprise         : how unexpected the outcome was
    - tags             : semantic labels (e.g. "failure", "recovery")
    """

    def __init__(
        self,
        feature_vec: List[float],
        target: float,
        outcome: float,
        source: str,
        reward: float = 0.0,
        context: Optional[dict] = None,
    ):
        self.id              = f"ep_{int(time.time()*1000)}_{random.randint(0,9999):04d}"
        self.feature_vec     = feature_vec
        self.target          = target
        self.outcome         = outcome        # actual result (may differ from target)
        self.source          = source
        self.reward          = reward
        self.context         = context or {}
        self.timestamp       = datetime.now(timezone.utc).isoformat()
        self.last_recall_ts  = time.time()

        # Memory properties
        surprise             = abs(outcome - target)
        self.surprise        = surprise
        self.initial_strength = EbbinghausCurve.initial_strength(reward, surprise)
        self.memory_strength = self.initial_strength
        self.stability       = 0.1            # low until consolidated
        self.recall_count    = 0

        # Tags — semantic labels for retrieval
        self.tags: List[str] = self._auto_tag()

    def _auto_tag(self) -> List[str]:
        tags = []
        if self.reward > 0.5:    tags.append("high_reward")
        if self.reward < -0.5:   tags.append("penalised")
        if self.surprise > 0.4:  tags.append("surprising")
        if self.target > 0.8:    tags.append("excellent")
        if self.target < 0.2:    tags.append("failure")
        if "cascade" in self.source: tags.append("cascade")
        if "dream" in self.source:   tags.append("dream")
        if "real" in self.source:    tags.append("real")
        return tags

    def update_strength(self) -> float:
        """Recompute current memory strength based on elapsed time."""
        hours = (time.time() - self.last_recall_ts) / 3600.0
        self.memory_strength = EbbinghausCurve.strength(
            self.initial_strength, hours, self.stability
        )
        return self.memory_strength

    def recall(self) -> "Episode":
        """Mark this episode as recalled — strengthens it."""
        self.recall_count   += 1
        self.last_recall_ts  = time.time()
        self.stability       = EbbinghausCurve.stability_after_recall(self.stability)
        self.memory_strength = min(1.0, self.memory_strength * 1.1)
        return self

    def to_dict(self) -> dict:
        return {
            "id":               self.id,
            "feature_vec":      self.feature_vec,
            "target":           self.target,
            "outcome":          self.outcome,
            "source":           self.source,
            "reward":           self.reward,
            "surprise":         round(self.surprise, 4),
            "memory_strength":  round(self.memory_strength, 4),
            "stability":        round(self.stability, 4),
            "recall_count":     self.recall_count,
            "tags":             self.tags,
            "timestamp":        self.timestamp,
        }

    def to_db_row(self) -> tuple:
        return (
            self.id,
            json.dumps(self.feature_vec),
            self.target,
            self.outcome,
            self.source,
            self.reward,
            self.surprise,
            self.memory_strength,
            self.stability,
            self.recall_count,
            json.dumps(self.tags),
            self.timestamp,
            self.last_recall_ts,
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Semantic Memory — abstract rules extracted from many episodes
# ─────────────────────────────────────────────────────────────────────────────

class SemanticRule:
    """
    An abstract rule generalised from multiple episodes.

    Example: "When semantic=high AND memory=high → target ≈ 0.85"
    Represented as: avg feature vector + avg target + confidence
    """

    def __init__(
        self,
        tag: str,
        avg_vec: List[float],
        avg_target: float,
        episode_count: int,
        confidence: float,
    ):
        self.id            = f"rule_{tag}_{int(time.time())}"
        self.tag           = tag
        self.avg_vec       = avg_vec
        self.avg_target    = avg_target
        self.episode_count = episode_count
        self.confidence    = confidence
        self.created_at    = datetime.now(timezone.utc).isoformat()
        self.use_count     = 0

    def predict(self, query_vec: List[float]) -> Tuple[float, float]:
        """
        Predict target for a query vector.
        Returns (prediction, relevance_score).
        Relevance = cosine similarity to the rule's prototype vector.
        """
        a = np.array(self.avg_vec)
        b = np.array(query_vec)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm < 1e-8:
            return self.avg_target, 0.0
        cosine = float(np.dot(a, b) / norm)
        relevance = (cosine + 1) / 2  # map [-1,1] → [0,1]
        return self.avg_target, relevance * self.confidence

    def to_dict(self) -> dict:
        return {
            "id":            self.id,
            "tag":           self.tag,
            "avg_vec":       [round(v, 4) for v in self.avg_vec],
            "avg_target":    round(self.avg_target, 4),
            "confidence":    round(self.confidence, 4),
            "episode_count": self.episode_count,
            "use_count":     self.use_count,
            "created_at":    self.created_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
#  Episodic Memory Engine
# ─────────────────────────────────────────────────────────────────────────────

class EpisodicMemoryEngine:
    """
    Phase 11: Full three-tier memory system.

    Tier 1 — Working Memory (in-RAM deque, last 50 episodes)
    Tier 2 — Episodic Store (SQLite, thousands of episodes)
    Tier 3 — Semantic Memory (in-RAM rules, extracted from episodes)

    Background thread handles:
    - Periodic consolidation (working → episodic)
    - Forgetting (remove weak memories)
    - Abstraction (episodic → semantic rules)
    """

    def __init__(
        self,
        db_path: str = EPISODIC_DB_PATH,
        consolidation_interval: float = CONSOLIDATION_INTERVAL,
    ):
        self._db_path       = db_path
        self._interval      = consolidation_interval
        self._lock          = threading.Lock()

        # Tier 1: Working memory
        self.working_memory: deque = deque(maxlen=WORKING_MEMORY_SIZE)

        # Tier 3: Semantic rules (in RAM)
        self.semantic_rules: List[SemanticRule] = []

        # Statistics
        self._stats = {
            "episodes_recorded":    0,
            "episodes_forgotten":   0,
            "consolidations":       0,
            "semantic_rules":       0,
            "total_recalls":        0,
        }

        # Setup DB
        self._init_db()

        # Start background consolidation thread
        self._running = False
        self._thread: Optional[threading.Thread] = None

        logger.info(f"EpisodicMemoryEngine (Phase 11) initialised — db: {db_path}")

    # ── DB setup ──────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS episodes (
                    id              TEXT PRIMARY KEY,
                    feature_vec     TEXT NOT NULL,
                    target          REAL NOT NULL,
                    outcome         REAL NOT NULL,
                    source          TEXT NOT NULL,
                    reward          REAL DEFAULT 0,
                    surprise        REAL DEFAULT 0,
                    memory_strength REAL DEFAULT 1.0,
                    stability       REAL DEFAULT 0.1,
                    recall_count    INTEGER DEFAULT 0,
                    tags            TEXT DEFAULT '[]',
                    timestamp       TEXT NOT NULL,
                    last_recall_ts  REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_strength
                ON episodes(memory_strength)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source
                ON episodes(source)
            """)
            conn.commit()

    # ── Public API ────────────────────────────────────────────────────────

    def record(
        self,
        feature_vec: List[float],
        target: float,
        outcome: Optional[float] = None,
        source: str = "unknown",
        reward: float = 0.0,
        context: Optional[dict] = None,
    ) -> Episode:
        """
        Record a new experience into working memory.
        High-strength episodes are immediately written to DB.
        """
        ep = Episode(
            feature_vec=feature_vec,
            target=target,
            outcome=outcome if outcome is not None else target,
            source=source,
            reward=reward,
            context=context or {},
        )

        with self._lock:
            self.working_memory.append(ep)
            self._stats["episodes_recorded"] += 1

        # Strong memories bypass working memory → write to DB immediately
        if ep.initial_strength > 0.7:
            self._write_episode_to_db(ep)

        return ep

    def recall_similar(
        self,
        query_vec: List[float],
        top_k: int = 5,
        tag_filter: Optional[str] = None,
    ) -> List[Episode]:
        """
        Recall the top-k most similar episodes from working memory.
        Uses cosine similarity on feature vectors.
        """
        q = np.array(query_vec)

        # Search working memory first
        candidates = list(self.working_memory)

        # Add from DB if needed
        if len(candidates) < top_k:
            db_eps = self._load_recent_from_db(n=50, tag=tag_filter)
            candidates.extend(db_eps)

        if not candidates:
            return []

        # Score by similarity × memory_strength
        scored = []
        for ep in candidates:
            ep.update_strength()
            if ep.memory_strength < MIN_STRENGTH_TO_KEEP:
                continue
            v = np.array(ep.feature_vec)
            norm = np.linalg.norm(q) * np.linalg.norm(v)
            if norm < 1e-8:
                sim = 0.0
            else:
                sim = float(np.dot(q, v) / norm)
            score = (sim + 1) / 2 * ep.memory_strength
            scored.append((score, ep))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = []
        for _, ep in scored[:top_k]:
            ep.recall()
            self._stats["total_recalls"] += 1
            results.append(ep)

        return results

    def predict_from_memory(self, query_vec: List[float]) -> Optional[float]:
        """
        Use semantic rules + recalled episodes to predict target.
        Returns None if insufficient memory.
        """
        predictions = []
        weights     = []

        # From semantic rules
        for rule in self.semantic_rules:
            pred, relevance = rule.predict(query_vec)
            if relevance > 0.3:
                predictions.append(pred)
                weights.append(relevance * rule.confidence)
                rule.use_count += 1

        # From recalled similar episodes
        similar = self.recall_similar(query_vec, top_k=3)
        for ep in similar:
            predictions.append(ep.target)
            weights.append(ep.memory_strength * 0.5)

        if not predictions:
            return None

        total_w = sum(weights)
        if total_w <= 0:
            return sum(predictions) / len(predictions)
        return float(sum(p * w for p, w in zip(predictions, weights)) / total_w)

    # ── Consolidation (background) ────────────────────────────────────────

    def start(self) -> None:
        """Start background consolidation thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._consolidation_loop,
            daemon=True,
            name="EpisodicMemory-Phase11",
        )
        self._thread.start()
        logger.info("EpisodicMemoryEngine background consolidation started")

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _consolidation_loop(self) -> None:
        while self._running:
            try:
                time.sleep(self._interval)
                self._consolidate()
                self._forget()
                self._abstract_semantic_rules()
                with self._lock:
                    self._stats["consolidations"] += 1
            except Exception as e:
                logger.error(f"EpisodicMemory consolidation error: {e}")

    def _consolidate(self) -> None:
        """Move working memory episodes to DB."""
        with self._lock:
            episodes = list(self.working_memory)
        written = 0
        for ep in episodes:
            try:
                self._write_episode_to_db(ep)
                written += 1
            except Exception:
                pass
        if written:
            logger.debug(f"EpisodicMemory consolidated {written} episodes to DB")

    def _write_episode_to_db(self, ep: Episode) -> None:
        try:
            with sqlite3.connect(self._db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO episodes
                    (id, feature_vec, target, outcome, source, reward,
                     surprise, memory_strength, stability, recall_count,
                     tags, timestamp, last_recall_ts)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, ep.to_db_row())
                conn.commit()
        except Exception as e:
            logger.debug(f"EpisodicMemory DB write error: {e}")

    def _forget(self) -> None:
        """Apply Ebbinghaus forgetting — remove weakest memories."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                # Count total episodes
                count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
                if count <= 1000:
                    return  # keep at least 1000 memories

                # Update strengths based on elapsed time
                rows = conn.execute("""
                    SELECT id, memory_strength, stability, last_recall_ts
                    FROM episodes
                    WHERE memory_strength < 0.5
                    ORDER BY memory_strength ASC
                    LIMIT 500
                """).fetchall()

                to_delete = []
                to_update = []
                for row_id, strength, stability, last_recall in rows:
                    hours = (time.time() - last_recall) / 3600.0
                    new_strength = EbbinghausCurve.strength(strength, hours, stability)
                    if new_strength < MIN_STRENGTH_TO_KEEP and count > 1000:
                        to_delete.append(row_id)
                        count -= 1
                    else:
                        to_update.append((new_strength, row_id))

                if to_delete:
                    conn.executemany(
                        "DELETE FROM episodes WHERE id=?",
                        [(i,) for i in to_delete]
                    )
                    with self._lock:
                        self._stats["episodes_forgotten"] += len(to_delete)
                    logger.debug(f"EpisodicMemory forgot {len(to_delete)} weak memories")

                if to_update:
                    conn.executemany(
                        "UPDATE episodes SET memory_strength=? WHERE id=?",
                        to_update,
                    )
                conn.commit()
        except Exception as e:
            logger.debug(f"EpisodicMemory forgetting error: {e}")

    def _abstract_semantic_rules(self) -> None:
        """
        Extract semantic rules from episodic store.
        Groups episodes by tag and computes prototype vectors.
        """
        tags_to_process = [
            "high_reward", "penalised", "surprising", "excellent",
            "failure", "cascade", "real",
        ]
        new_rules = []

        for tag in tags_to_process:
            try:
                with sqlite3.connect(self._db_path) as conn:
                    rows = conn.execute("""
                        SELECT feature_vec, target, memory_strength
                        FROM episodes
                        WHERE tags LIKE ? AND memory_strength > 0.3
                        ORDER BY memory_strength DESC
                        LIMIT 200
                    """, (f'%"{tag}"%',)).fetchall()

                if len(rows) < ABSTRACTION_THRESHOLD:
                    continue

                vecs    = [json.loads(r[0]) for r in rows]
                targets = [r[1] for r in rows]
                weights = [r[2] for r in rows]
                total_w = sum(weights)

                if total_w <= 0:
                    continue

                avg_vec    = [
                    sum(v[i] * w for v, w in zip(vecs, weights)) / total_w
                    for i in range(7)
                ]
                avg_target = sum(t * w for t, w in zip(targets, weights)) / total_w
                confidence = min(len(rows) / 100.0, 1.0)

                rule = SemanticRule(
                    tag=tag,
                    avg_vec=avg_vec,
                    avg_target=avg_target,
                    episode_count=len(rows),
                    confidence=confidence,
                )
                new_rules.append(rule)

            except Exception as e:
                logger.debug(f"Semantic abstraction error for tag {tag}: {e}")

        if new_rules:
            with self._lock:
                # Replace existing rules for the same tags
                existing_tags = {r.tag for r in new_rules}
                self.semantic_rules = [
                    r for r in self.semantic_rules if r.tag not in existing_tags
                ]
                self.semantic_rules.extend(new_rules)
                self._stats["semantic_rules"] = len(self.semantic_rules)
            logger.debug(
                f"EpisodicMemory abstracted {len(new_rules)} semantic rules"
            )

    # ── Queries ───────────────────────────────────────────────────────────

    def _load_recent_from_db(
        self, n: int = 50, tag: Optional[str] = None
    ) -> List[Episode]:
        """Load recent episodes from DB into Episode objects."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                if tag:
                    rows = conn.execute("""
                        SELECT feature_vec, target, outcome, source, reward,
                               surprise, memory_strength, stability, recall_count, tags
                        FROM episodes
                        WHERE tags LIKE ?
                        ORDER BY last_recall_ts DESC LIMIT ?
                    """, (f'%"{tag}"%', n)).fetchall()
                else:
                    rows = conn.execute("""
                        SELECT feature_vec, target, outcome, source, reward,
                               surprise, memory_strength, stability, recall_count, tags
                        FROM episodes
                        ORDER BY last_recall_ts DESC LIMIT ?
                    """, (n,)).fetchall()
        except Exception:
            return []

        episodes = []
        for row in rows:
            try:
                ep = Episode(
                    feature_vec=json.loads(row[0]),
                    target=row[1],
                    outcome=row[2],
                    source=row[3],
                    reward=row[4],
                )
                ep.surprise        = row[5]
                ep.memory_strength = row[6]
                ep.stability       = row[7]
                ep.recall_count    = row[8]
                ep.tags            = json.loads(row[9])
                episodes.append(ep)
            except Exception:
                pass
        return episodes

    def get_strongest_memories(self, n: int = 10) -> List[dict]:
        """Return the n strongest memories."""
        try:
            with sqlite3.connect(self._db_path) as conn:
                rows = conn.execute("""
                    SELECT id, source, target, memory_strength, recall_count, tags
                    FROM episodes
                    ORDER BY memory_strength DESC LIMIT ?
                """, (n,)).fetchall()
            return [
                {
                    "id": r[0], "source": r[1], "target": round(r[2], 4),
                    "strength": round(r[3], 4), "recalls": r[4],
                    "tags": json.loads(r[5]),
                }
                for r in rows
            ]
        except Exception:
            return []

    def episode_count(self) -> int:
        try:
            with sqlite3.connect(self._db_path) as conn:
                return conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        except Exception:
            return 0

    def summary(self) -> dict:
        with self._lock:
            stats = dict(self._stats)
        stats["semantic_rules"] = len(self.semantic_rules)
        return {
            **stats,
            "working_memory_size":   len(self.working_memory),
            "episodic_store_size":   self.episode_count(),
            "semantic_rules_list":   [r.to_dict() for r in self.semantic_rules[:5]],
            "strongest_memories":    self.get_strongest_memories(5),
            "is_running":            self._running,
        }
