"""
ai/swarm_history_store.py — Swarm Execution History (Persistent)
================================================================
سجلّ نتائج SwarmCoordinator.execute() بشكل دائم.

المشكلة قبل هذا الملف: SwarmCoordinator._history كانت قائمة Python
عادية في الذاكرة فقط (self._history: List[SwarmResult] = []). أي
إعادة تشغيل للحاوية (شائعة على Streamlit Community Cloud عند الخمول
أو النشر أو الأخطاء) تمسح كل تاريخ تنفيذ السرب بالكامل — بلا أي أثر
لأي swarm سابق، حتى لو نجح ونتج عنه مخرجات مفيدة.

نفس نمط ai/agent_audit.py بالضبط (SQLite + log/get/summary)، مخصّص
لتنفيذات السرب (SwarmResult) بدل تفاعلات الوكلاء الفردية.

يُخزَّن في: memory/swarm_history.db (SQLite)
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path("memory/swarm_history.db")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SwarmHistoryStore:
    """
    يسجّل ويسترجع نتائج تنفيذ السرب (SwarmResult.to_dict()).

    الاستخدام:
        store = get_default_swarm_store()
        store.log_result(result.to_dict())
        recent = store.get_recent(20)
    """

    def __init__(self, db_path: Path = DB_PATH):
        # مسار مطلق فوراً — نفس السبب الموثّق في core_history.py/agent_audit.py:
        # أي os.chdir() لاحق في نفس العملية لا يجب أن يغيّر مكان قاعدة البيانات.
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS swarm_history (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    swarm_id        TEXT    NOT NULL,
                    goal            TEXT    NOT NULL,
                    status          TEXT    NOT NULL,
                    started_at      TEXT    NOT NULL,
                    finished_at     TEXT,
                    total_tasks     INTEGER NOT NULL DEFAULT 0,
                    success_count   INTEGER NOT NULL DEFAULT 0,
                    failed_count    INTEGER NOT NULL DEFAULT 0,
                    logged_at       TEXT    NOT NULL,
                    full_result     TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_swarm_history_swarm_id "
                "ON swarm_history(swarm_id)"
            )
            conn.commit()

    def log_result(self, result_dict: Dict[str, Any]) -> int:
        """يسجّل نتيجة swarm واحدة (خرج SwarmResult.to_dict()). لا يرفع
        استثناء عند فشل الكتابة (يُسجَّل تحذير فقط) — التدقيق لا يجب أن
        يُعطّل تنفيذ السرب أبداً."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    INSERT INTO swarm_history
                        (swarm_id, goal, status, started_at, finished_at,
                         total_tasks, success_count, failed_count,
                         logged_at, full_result)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_dict.get("swarm_id", ""),
                    result_dict.get("goal", ""),
                    result_dict.get("status", "unknown"),
                    result_dict.get("started_at", ""),
                    result_dict.get("finished_at"),
                    result_dict.get("total_tasks", 0),
                    result_dict.get("success_count", 0),
                    result_dict.get("failed_count", 0),
                    _now(),
                    json.dumps(result_dict, ensure_ascii=False),
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.log_result: فشل تسجيل النتيجة: {e}")
            return -1

    def get_recent(self, limit: int = 20) -> List[dict]:
        """يرجع آخر نتائج تنفيذ السرب كاملة (الأحدث أولاً)."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT full_result FROM swarm_history
                    ORDER BY id DESC LIMIT ?
                """, (limit,)).fetchall()
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.get_recent: {e}")
            return []

        result = []
        for row in rows:
            try:
                result.append(json.loads(row["full_result"]))
            except Exception:
                continue
        return result

    def summary(self) -> dict:
        """ملخص إحصائي دائم: إجمالي عمليات السرب، حسب الحالة."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM swarm_history"
                ).fetchone()[0]
                done = conn.execute(
                    "SELECT COUNT(*) FROM swarm_history WHERE status = 'done'"
                ).fetchone()[0]
                partial = conn.execute(
                    "SELECT COUNT(*) FROM swarm_history WHERE status = 'partial'"
                ).fetchone()[0]
                failed = conn.execute(
                    "SELECT COUNT(*) FROM swarm_history WHERE status = 'failed'"
                ).fetchone()[0]
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.summary: {e}")
            return {
                "total_swarms": 0, "done": 0, "partial": 0, "failed": 0,
                "db_path": str(self.db_path),
            }

        return {
            "total_swarms": total,
            "done": done,
            "partial": partial,
            "failed": failed,
            "db_path": str(self.db_path),
        }


# ── Singleton ─────────────────────────────────────────────────────────────
_default_swarm_store: Optional[SwarmHistoryStore] = None


def get_default_swarm_store(db_path: Path = DB_PATH) -> SwarmHistoryStore:
    global _default_swarm_store
    if _default_swarm_store is None:
        _default_swarm_store = SwarmHistoryStore(db_path)
    return _default_swarm_store
