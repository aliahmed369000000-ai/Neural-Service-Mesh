"""
Phase 3 – Scoring Engine
Every connection gets a dynamic score based on execution history.
Scores are persisted to SQLite and loaded on startup.
"""
from __future__ import annotations
import sqlite3
import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ConnectionScore:
    """
    Live score for a single directed connection (source_id → target_id).
    Attributes are updated after every execution that crosses this edge.
    """
    def __init__(self, source_id: str, target_id: str):
        self.source_id = source_id
        self.target_id = target_id
        self.total_runs: int = 0
        self.successful_runs: int = 0
        self.failed_runs: int = 0
        self.total_latency_ms: float = 0.0
        self.last_updated: str = datetime.utcnow().isoformat()
        # Exponentially-weighted moving average latency
        self._ema_latency: Optional[float] = None

    @property
    def success_rate(self) -> float:
        if self.total_runs == 0:
            return 0.5   # Neutral prior
        return self.successful_runs / self.total_runs

    @property
    def avg_latency_ms(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.total_latency_ms / self.total_runs

    @property
    def ema_latency_ms(self) -> float:
        return self._ema_latency or self.avg_latency_ms

    @property
    def error_rate(self) -> float:
        if self.total_runs == 0:
            return 0.0
        return self.failed_runs / self.total_runs

    @property
    def connection_score(self) -> float:
        """
        Composite score in [0, 100].
        Higher is better.
        Formula:
          base   = success_rate * 60            (0-60)
          usage  = min(log(runs+1)/log(101), 1) * 20   (0-20, rewards experience)
          speed  = max(0, 1 - ema / 5000) * 20  (0-20, penalises latency > 5 s)
        """
        base = self.success_rate * 60.0
        usage = (math.log(self.total_runs + 1) / math.log(101)) * 20.0
        ema = self.ema_latency_ms if self._ema_latency else self.avg_latency_ms
        speed = max(0.0, 1.0 - ema / 5000.0) * 20.0
        return round(min(100.0, base + usage + speed), 2)

    def record_execution(self, success: bool, latency_ms: float):
        self.total_runs += 1
        self.total_latency_ms += latency_ms
        if success:
            self.successful_runs += 1
        else:
            self.failed_runs += 1
        # EMA with α = 0.3
        alpha = 0.3
        if self._ema_latency is None:
            self._ema_latency = latency_ms
        else:
            self._ema_latency = alpha * latency_ms + (1 - alpha) * self._ema_latency
        self.last_updated = datetime.utcnow().isoformat()

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "total_runs": self.total_runs,
            "successful_runs": self.successful_runs,
            "failed_runs": self.failed_runs,
            "success_rate": round(self.success_rate, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "ema_latency_ms": round(self.ema_latency_ms, 2),
            "error_rate": round(self.error_rate, 4),
            "connection_score": self.connection_score,
            "last_updated": self.last_updated,
        }


class ScoringEngine:
    """
    Phase 3 Scoring Engine.
    Manages ConnectionScore objects and persists them to SQLite.
    Called by the ExecutionEngine after every run to update scores.
    """

    def __init__(self, db_path: str = "./data/mesh.db"):
        self._db_path = Path(db_path)
        self._scores: Dict[Tuple[str, str], ConnectionScore] = {}
        self._init_schema()
        self._load_from_db()
        logger.info("ScoringEngine initialised (Phase 3)")

    # ── Schema / persistence ───────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS connection_scores (
                    source_id       TEXT NOT NULL,
                    target_id       TEXT NOT NULL,
                    total_runs      INTEGER DEFAULT 0,
                    successful_runs INTEGER DEFAULT 0,
                    failed_runs     INTEGER DEFAULT 0,
                    total_latency_ms REAL DEFAULT 0.0,
                    ema_latency_ms  REAL,
                    last_updated    TEXT,
                    PRIMARY KEY (source_id, target_id)
                )
            """)

    def _load_from_db(self):
        try:
            with self._conn() as conn:
                rows = conn.execute("SELECT * FROM connection_scores").fetchall()
            for row in rows:
                key = (row["source_id"], row["target_id"])
                cs = ConnectionScore(row["source_id"], row["target_id"])
                cs.total_runs = row["total_runs"]
                cs.successful_runs = row["successful_runs"]
                cs.failed_runs = row["failed_runs"]
                cs.total_latency_ms = row["total_latency_ms"]
                cs._ema_latency = row["ema_latency_ms"]
                cs.last_updated = row["last_updated"] or cs.last_updated
                self._scores[key] = cs
            logger.info(f"ScoringEngine loaded {len(self._scores)} connection scores")
        except Exception as e:
            logger.warning(f"ScoringEngine load warning: {e}")

    def _persist_score(self, cs: ConnectionScore):
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO connection_scores
                        (source_id, target_id, total_runs, successful_runs,
                         failed_runs, total_latency_ms, ema_latency_ms, last_updated)
                    VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(source_id, target_id) DO UPDATE SET
                        total_runs=excluded.total_runs,
                        successful_runs=excluded.successful_runs,
                        failed_runs=excluded.failed_runs,
                        total_latency_ms=excluded.total_latency_ms,
                        ema_latency_ms=excluded.ema_latency_ms,
                        last_updated=excluded.last_updated
                """, (
                    cs.source_id, cs.target_id,
                    cs.total_runs, cs.successful_runs, cs.failed_runs,
                    cs.total_latency_ms, cs._ema_latency, cs.last_updated,
                ))
        except Exception as e:
            logger.error(f"ScoringEngine persist error: {e}")

    # ── Core API ───────────────────────────────────────────────────────────

    def get_score(self, source_id: str, target_id: str) -> ConnectionScore:
        key = (source_id, target_id)
        if key not in self._scores:
            self._scores[key] = ConnectionScore(source_id, target_id)
        return self._scores[key]

    def record_run(self, run_result: dict):
        """
        Called after every execution run.
        Updates scores for each edge traversed in the path.
        """
        path = run_result.get("path", [])
        steps = run_result.get("steps", [])
        run_success = run_result.get("status") == "success"

        if len(path) < 2:
            return

        # Build latency map: node_id → duration_ms
        latency_map: Dict[str, float] = {}
        for step in steps:
            nid = step.get("node_id")
            ms = step.get("duration_ms") or 0.0
            step_ok = step.get("status") == "success"
            if nid:
                latency_map[nid] = (ms, step_ok)

        for i in range(len(path) - 1):
            src, tgt = path[i], path[i + 1]
            tgt_info = latency_map.get(tgt, (0.0, run_success))
            latency = tgt_info[0]
            step_ok = tgt_info[1]

            cs = self.get_score(src, tgt)
            cs.record_execution(step_ok, latency)
            self._persist_score(cs)

        logger.debug(f"ScoringEngine updated {len(path)-1} connections for run {run_result.get('run_id','?')[:8]}")

    def get_path_score(self, path: List[str]) -> float:
        """
        Aggregate score for an entire path.
        Geometric mean of individual connection scores (so one bad link kills the path).
        """
        if len(path) < 2:
            return 0.0
        scores = []
        for i in range(len(path) - 1):
            cs = self.get_score(path[i], path[i + 1])
            scores.append(cs.connection_score)
        if not scores:
            return 0.0
        # Geometric mean
        log_sum = sum(math.log(max(s, 0.01)) for s in scores)
        return round(math.exp(log_sum / len(scores)), 2)

    def list_scores(self) -> List[dict]:
        return [cs.to_dict() for cs in self._scores.values()]

    def top_connections(self, n: int = 10) -> List[dict]:
        return sorted(self.list_scores(), key=lambda x: x["connection_score"], reverse=True)[:n]

    def worst_connections(self, n: int = 10) -> List[dict]:
        return sorted(self.list_scores(), key=lambda x: x["connection_score"])[:n]

    def summary(self) -> dict:
        scores = self.list_scores()
        if not scores:
            return {"total_connections_tracked": 0}
        avg = sum(s["connection_score"] for s in scores) / len(scores)
        return {
            "total_connections_tracked": len(scores),
            "avg_connection_score": round(avg, 2),
            "top_connection": self.top_connections(1)[0] if scores else None,
            "worst_connection": self.worst_connections(1)[0] if scores else None,
        }

    def __repr__(self):
        return f"<ScoringEngine connections={len(self._scores)}>"
