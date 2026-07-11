from __future__ import annotations
import sqlite3
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class SQLiteStorage:
    """Phase 2: SQLite-backed persistent storage for nodes, connections, and execution logs."""

    def __init__(self, db_path: str = "./data/mesh.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        logger.info(f"SQLiteStorage ready at {self.db_path}")

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS nodes (
                    node_id     TEXT PRIMARY KEY,
                    name        TEXT NOT NULL,
                    node_type   TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    tags        TEXT DEFAULT '[]',
                    version     TEXT DEFAULT '1.0.0',
                    created_at  TEXT NOT NULL,
                    meta_json   TEXT DEFAULT '{}'
                );

                CREATE TABLE IF NOT EXISTS connections (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id   TEXT NOT NULL,
                    target_id   TEXT NOT NULL,
                    weight      REAL DEFAULT 1.0,
                    label       TEXT DEFAULT '',
                    created_at  TEXT NOT NULL,
                    UNIQUE(source_id, target_id)
                );

                CREATE TABLE IF NOT EXISTS execution_logs (
                    run_id          TEXT PRIMARY KEY,
                    status          TEXT NOT NULL,
                    path_json       TEXT NOT NULL,
                    started_at      TEXT NOT NULL,
                    finished_at     TEXT,
                    total_duration_ms REAL,
                    final_output    TEXT,
                    steps_json      TEXT DEFAULT '[]'
                );

                CREATE INDEX IF NOT EXISTS idx_logs_status ON execution_logs(status);
                CREATE INDEX IF NOT EXISTS idx_logs_started ON execution_logs(started_at);
                CREATE INDEX IF NOT EXISTS idx_conn_source ON connections(source_id);
            """)

    # ── Nodes ──────────────────────────────────────────────────────────────

    def upsert_node(self, node_data: dict) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO nodes (node_id, name, node_type, description, tags, version, created_at, meta_json)
                    VALUES (:node_id,:name,:node_type,:description,:tags,:version,:created_at,:meta_json)
                    ON CONFLICT(node_id) DO UPDATE SET
                        name=excluded.name, description=excluded.description,
                        tags=excluded.tags, meta_json=excluded.meta_json
                """, {
                    "node_id": node_data["node_id"],
                    "name": node_data["name"],
                    "node_type": node_data.get("node_type", "unknown"),
                    "description": node_data.get("description", ""),
                    "tags": json.dumps(node_data.get("tags", [])),
                    "version": node_data.get("version", "1.0.0"),
                    "created_at": node_data.get("created_at", datetime.utcnow().isoformat()),
                    "meta_json": json.dumps(node_data),
                })
            return True
        except Exception as e:
            logger.error(f"upsert_node failed: {e}")
            return False

    def get_node(self, node_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT meta_json FROM nodes WHERE node_id=?", (node_id,)).fetchone()
        return json.loads(row["meta_json"]) if row else None

    def list_nodes(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT meta_json FROM nodes ORDER BY created_at").fetchall()
        return [json.loads(r["meta_json"]) for r in rows]

    def delete_node(self, node_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM nodes WHERE node_id=?", (node_id,))
        return cur.rowcount > 0

    def count_nodes(self) -> int:
        with self._conn() as conn:
            return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]

    # ── Connections ────────────────────────────────────────────────────────

    def upsert_connection(self, source_id: str, target_id: str,
                          weight: float = 1.0, label: str = "") -> bool:
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO connections (source_id, target_id, weight, label, created_at)
                    VALUES (?,?,?,?,?)
                    ON CONFLICT(source_id, target_id) DO UPDATE SET weight=excluded.weight, label=excluded.label
                """, (source_id, target_id, weight, label, datetime.utcnow().isoformat()))
            return True
        except Exception as e:
            logger.error(f"upsert_connection failed: {e}")
            return False

    def list_connections(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT source_id, target_id, weight, label FROM connections").fetchall()
        return [dict(r) for r in rows]

    def delete_connection(self, source_id: str, target_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM connections WHERE source_id=? AND target_id=?",
                               (source_id, target_id))
        return cur.rowcount > 0

    # ── Execution Logs ─────────────────────────────────────────────────────

    def save_run(self, result: dict) -> bool:
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO execution_logs
                        (run_id, status, path_json, started_at, finished_at, total_duration_ms, final_output, steps_json)
                    VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        status=excluded.status, finished_at=excluded.finished_at,
                        total_duration_ms=excluded.total_duration_ms,
                        final_output=excluded.final_output, steps_json=excluded.steps_json
                """, (
                    result["run_id"],
                    result["status"],
                    json.dumps(result.get("path", [])),
                    result["started_at"],
                    result.get("finished_at"),
                    result.get("total_duration_ms"),
                    json.dumps(result.get("final_output")),
                    json.dumps(result.get("steps", [])),
                ))
            return True
        except Exception as e:
            logger.error(f"save_run failed: {e}")
            return False

    def get_run(self, run_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM execution_logs WHERE run_id=?", (run_id,)).fetchone()
        return self._deserialize_run(dict(row)) if row else None

    def list_runs(self, limit: int = 50, status: str = None) -> List[dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM execution_logs WHERE status=? ORDER BY started_at DESC LIMIT ?",
                    (status, limit)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM execution_logs ORDER BY started_at DESC LIMIT ?",
                    (limit,)).fetchall()
        return [self._deserialize_run(dict(r)) for r in rows]

    def stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) FROM execution_logs").fetchone()[0]
            success = conn.execute("SELECT COUNT(*) FROM execution_logs WHERE status='success'").fetchone()[0]
            avg_dur = conn.execute(
                "SELECT AVG(total_duration_ms) FROM execution_logs WHERE status='success'").fetchone()[0]
        return {
            "total_runs": total,
            "successful_runs": success,
            "failed_runs": total - success,
            "avg_duration_ms": round(avg_dur or 0, 2),
        }

    def _deserialize_run(self, row: dict) -> dict:
        row["path"] = json.loads(row.pop("path_json", "[]"))
        row["steps"] = json.loads(row.pop("steps_json", "[]"))
        row["final_output"] = json.loads(row.get("final_output") or "null")
        return row

    def db_stats(self) -> dict:
        return {
            "db_path": str(self.db_path),
            "size_bytes": self.db_path.stat().st_size if self.db_path.exists() else 0,
            "nodes": self.count_nodes(),
            "connections": len(self.list_connections()),
            **self.stats(),
        }
