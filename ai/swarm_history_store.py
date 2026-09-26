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
            # 🆕 جدول منفصل لنقاط تفتيش (checkpoints) السرب أثناء التنفيذ —
            # منفصل عمداً عن swarm_history (الذي يُسجَّل فيه فقط عند
            # اكتمال السرب). لو انهارت العملية (توقف مفاجئ: إعادة تشغيل
            # الحاوية على Streamlit Cloud، نفاد الذاكرة...) في منتصف
            # execute()، swarm_history لا يحتوي أي أثر لهذا التشغيل إطلاقاً
            # (self._persist_result لا تُستدعى إلا في النهاية). هذا الجدول
            # يُحدَّث دورياً أثناء التنفيذ (swarm_id هو المفتاح الأساسي،
            # فالتحديث المتكرر لنفس swarm_id يستبدل السجل بدل تكديسه) حتى
            # يقدر SwarmCoordinator.resume() يقرأ آخر حالة معروفة ويكمل من
            # المهام غير المكتملة فقط، بدل إعادة السرب كاملاً من الصفر.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS swarm_progress (
                    swarm_id     TEXT PRIMARY KEY,
                    goal         TEXT    NOT NULL,
                    status       TEXT    NOT NULL,
                    started_at   TEXT    NOT NULL,
                    updated_at   TEXT    NOT NULL,
                    full_result  TEXT    NOT NULL
                )
            """)
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

    # ── نقاط تفتيش السرب أثناء التنفيذ (للاستئناف بعد توقف مفاجئ) ───────────

    def save_progress(self, result_dict: Dict[str, Any]) -> bool:
        """يحفظ/يحدّث آخر حالة معروفة لسرب لا يزال قيد التنفيذ. يُستدعى
        عدة مرات خلال نفس التشغيل (بعد كل مهمة فرعية تكتمل) — INSERT OR
        REPLACE على swarm_id يضمن سجلاً واحداً محدَّثاً بدل التكديس.
        لا يرفع استثناء عند الفشل (التدقيق لا يجب أن يُعطّل تنفيذ السرب)."""
        swarm_id = result_dict.get("swarm_id", "")
        if not swarm_id:
            return False
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("""
                    INSERT INTO swarm_progress
                        (swarm_id, goal, status, started_at, updated_at, full_result)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(swarm_id) DO UPDATE SET
                        status=excluded.status,
                        updated_at=excluded.updated_at,
                        full_result=excluded.full_result
                """, (
                    swarm_id,
                    result_dict.get("goal", ""),
                    result_dict.get("status", "running"),
                    result_dict.get("started_at", _now()),
                    _now(),
                    json.dumps(result_dict, ensure_ascii=False),
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.save_progress: فشل حفظ نقطة التفتيش: {e}")
            return False

    def get_progress(self, swarm_id: str) -> Optional[dict]:
        """يرجع آخر نقطة تفتيش محفوظة لهذا swarm_id، أو None إن لم توجد."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT full_result FROM swarm_progress WHERE swarm_id=?",
                    (swarm_id,),
                ).fetchone()
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.get_progress: {e}")
            return None
        if not row:
            return None
        try:
            return json.loads(row["full_result"])
        except Exception:
            return None

    def clear_progress(self, swarm_id: str) -> bool:
        """يحذف نقطة تفتيش سرب بعد اكتماله فعلياً (نجاحاً أو فشلاً) — النتيجة
        النهائية مسجّلة أصلاً في swarm_history، فلا داعي لإبقاء نسخة مؤقتة."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("DELETE FROM swarm_progress WHERE swarm_id=?", (swarm_id,))
                conn.commit()
                return True
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.clear_progress: {e}")
            return False

    def list_incomplete(self, limit: int = 20) -> List[dict]:
        """يرجع أسرِبة توقفت في منتصف التنفيذ (status='running') — إما لا
        تزال تعمل فعلاً في عملية حيّة، أو انهارت العملية قبل أن تُكمل.
        الأحدث تحديثاً أولاً."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute("""
                    SELECT full_result FROM swarm_progress
                    WHERE status = 'running'
                    ORDER BY updated_at DESC LIMIT ?
                """, (limit,)).fetchall()
        except Exception as e:
            logger.warning(f"SwarmHistoryStore.list_incomplete: {e}")
            return []
        result = []
        for row in rows:
            try:
                result.append(json.loads(row["full_result"]))
            except Exception:
                continue
        return result

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
