"""سجل تاريخي لدورات تشخيص عُقد الشبكة (penalized / recovered).

يُخزَّن في SQLite بجانب بيانات الـmesh حتى يبقى بعد إعادة التشغيل
ويمكن رسم اتجاه عدد المعاقَبين والمعافين عبر الزمن.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_DB = Path("memory/nodes_diagnose_history.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class NodesDiagnoseStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = Path(db_path or DEFAULT_DB).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS diagnose_cycles (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              TEXT    NOT NULL,
                    scanned         INTEGER NOT NULL DEFAULT 0,
                    errors          INTEGER NOT NULL DEFAULT 0,
                    avg_reputation  REAL,
                    effective_thr   REAL,
                    low_rep_count   INTEGER NOT NULL DEFAULT 0,
                    high_unread_count INTEGER NOT NULL DEFAULT 0,
                    recovered_count INTEGER NOT NULL DEFAULT 0,
                    full_summary    TEXT    NOT NULL,
                    logged_at       TEXT    NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_diagnose_cycles_ts "
                "ON diagnose_cycles(ts)"
            )
            conn.commit()

    def log_cycle(self, summary: Dict[str, Any]) -> int:
        """يسجّل ملخص دورة تشخيص واحدة. لا يرفع استثناء."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cur = conn.execute(
                    """
                    INSERT INTO diagnose_cycles (
                        ts, scanned, errors, avg_reputation, effective_thr,
                        low_rep_count, high_unread_count, recovered_count,
                        full_summary, logged_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        str(summary.get("ts") or _now()),
                        int(summary.get("scanned") or 0),
                        int(summary.get("errors") or 0),
                        float(summary.get("avg_reputation") or 0.0),
                        float(
                            summary.get("effective_low_rep_threshold")
                            or summary.get("effective_thr")
                            or 0.0
                        ),
                        len(summary.get("low_reputation") or []),
                        len(summary.get("high_unread") or []),
                        len(summary.get("recovered") or []),
                        json.dumps(summary, ensure_ascii=False),
                        _now(),
                    ),
                )
                conn.commit()
                return int(cur.lastrowid or -1)
        except Exception as e:
            logger.warning("NodesDiagnoseStore.log_cycle failed: %s", e)
            return -1

    def get_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        """آخر الدورات، الأحدث أولاً — للرسوم البيانية والواجهة."""
        limit = max(1, min(int(limit), 500))
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, ts, scanned, errors, avg_reputation, effective_thr,
                           low_rep_count, high_unread_count, recovered_count, logged_at
                    FROM diagnose_cycles
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as e:
            logger.warning("NodesDiagnoseStore.get_recent failed: %s", e)
            return []

    def trend(self, limit: int = 30) -> Dict[str, Any]:
        """سلسلة زمنية مبسطة لعدد المعاقَبين والمعافين."""
        rows = list(reversed(self.get_recent(limit=limit)))  # أقدم → أحدث
        return {
            "points": len(rows),
            "ts": [r.get("ts") for r in rows],
            "low_rep_count": [int(r.get("low_rep_count") or 0) for r in rows],
            "recovered_count": [int(r.get("recovered_count") or 0) for r in rows],
            "avg_reputation": [float(r.get("avg_reputation") or 0) for r in rows],
            "effective_thr": [float(r.get("effective_thr") or 0) for r in rows],
        }

    def summary(self) -> Dict[str, Any]:
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    """
                    SELECT COUNT(*) AS n,
                           AVG(low_rep_count) AS avg_low,
                           AVG(recovered_count) AS avg_rec,
                           MAX(ts) AS last_ts
                    FROM diagnose_cycles
                    """
                ).fetchone()
            if not row or not row[0]:
                return {"cycles": 0}
            return {
                "cycles": int(row[0]),
                "avg_low_rep": round(float(row[1] or 0), 3),
                "avg_recovered": round(float(row[2] or 0), 3),
                "last_ts": row[3],
            }
        except Exception as e:
            return {"cycles": 0, "error": str(e)}


_STORE: Optional[NodesDiagnoseStore] = None


def get_nodes_diagnose_store(db_path: Optional[Path] = None) -> NodesDiagnoseStore:
    global _STORE
    if db_path is not None:
        return NodesDiagnoseStore(db_path=db_path)
    if _STORE is None:
        _STORE = NodesDiagnoseStore()
    return _STORE
