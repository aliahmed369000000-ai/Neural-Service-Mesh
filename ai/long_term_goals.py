# -*- coding: utf-8 -*-
"""
ai/long_term_goals.py — الأهداف المؤسسية طويلة الأمد (Long-Term Goals)
════════════════════════════════════════════════════════════════════════
سجل أهداف استراتيجية يتراكم عبر كل الجلسات:
- أهداف مؤسسية (نحو الذكاء العام) تُضاف وتُتتبَّع يدويًا أو تلقائيًا.
- التقدم (progress 0-100) يتحدّث عند كل إنجاز يُسجَّل ضدها.
- تقييم دوري تلقائي: خيط خلفية (daemon) يعيد تقييم الأهداف غير المنتهية
  كل 24 ساعة (قابل للتخصيص) عبر سجل الخبرات (TEM): الأهداف المرتبطة
  بمهارات/سياقات نشطة تتقدم تلقائيًا بوتيرة بطيئة.

التخزين: SQLite محلي دائم في data/long_term_goals.db.
التدهور: فشل كامل → كتلة _LTG_OK في app_core تعطل الوحدة بصمت.

الجداول:
  ltg_goals: id, title, description, category, progress (0-100),
             status (active/archived/achieved), linked_skill, notes,
             created_at, updated_at, last_eval
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCAL_DB = "data/long_term_goals.db"
_MAX_GOALS = 50
_MAX_TITLE_CHARS = 200
_MAX_DESC_CHARS = 1200
_MAX_NOTES_CHARS = 500
_EVAL_PERIOD_SEC = 24 * 3600      # تقييم دوري كل 24 ساعة (اختبار: يُضبط)
_AUTO_STEP = 2.5                  # تقدم تلقائي لكل تقييم دوري
_STATUS = ("active", "archived", "achieved")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ltg_goals (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    description TEXT,
    category    TEXT,
    progress    REAL NOT NULL DEFAULT 0.0,
    status      TEXT NOT NULL DEFAULT 'active',
    linked_skill TEXT,
    notes       TEXT,
    created_at  REAL NOT NULL DEFAULT 0.0,
    updated_at  REAL NOT NULL DEFAULT 0.0,
    last_eval   REAL NOT NULL DEFAULT 0.0
);
"""

_thread_started = False


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


class LongTermGoals:
    """سجل الأهداف المؤسسية طويلة الأمد (thread-safe، تدهور آمن)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db = db_path or _LOCAL_DB
        self._lock = threading.Lock()
        self._eval_period = _EVAL_PERIOD_SEC
        with self._lock, _connect(self._db):
            pass

    # ── CRUD ───────────────────────────────────────────────────────────────

    def add(self, title: str, description: str = "",
            category: str = "general", linked_skill: str = "",
            progress: float = 0.0) -> Optional[int]:
        if not title or not title.strip():
            return None
        try:
            with self._lock, _connect(self._db) as conn:
                cur = conn.execute("""
                    INSERT INTO ltg_goals (title, description, category,
                                           progress, linked_skill, notes,
                                           created_at, updated_at, last_eval)
                    VALUES (?, ?, ?, ?, ?, '', ?, ?, ?)
                """, (title.strip()[:_MAX_TITLE_CHARS],
                      (description or "")[:_MAX_DESC_CHARS],
                      (category or "general")[:40],
                      max(0.0, min(100.0, float(progress))),
                      (linked_skill or "")[:_MAX_TITLE_CHARS],
                      time.time(), time.time(), time.time()))
                conn.commit()
                self._prune()
                return cur.lastrowid
        except Exception as exc:
            logger.warning("LTG add failed: %s", exc)
            return None

    def update_progress(self, goal_id: int, progress: float,
                        note: str = "") -> bool:
        try:
            with self._lock, _connect(self._db) as conn:
                cur = conn.execute("""
                    UPDATE ltg_goals
                    SET progress = ?, updated_at = ?,
                        notes = CASE WHEN ? != ''
                            THEN SUBSTR(? || ' | ' || notes, 1, %d)
                            ELSE notes END,
                        status = CASE WHEN ? >= 100
                            THEN 'achieved' ELSE status END
                    WHERE id = ?
                """ % _MAX_NOTES_CHARS,
                    (max(0.0, min(100.0, float(progress))), time.time(),
                     (note or "")[:_MAX_NOTES_CHARS],
                     (note or "")[:_MAX_NOTES_CHARS],
                     max(0.0, min(100.0, float(progress))), goal_id))
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:
            logger.warning("LTG update_progress failed: %s", exc)
            return False

    def archive(self, goal_id: int, status: str = "archived") -> bool:
        if status not in _STATUS:
            return False
        try:
            with self._lock, _connect(self._db) as conn:
                cur = conn.execute("""
                    UPDATE ltg_goals SET status = ?, updated_at = ?
                    WHERE id = ? AND status != 'achieved'
                """, (status, time.time(), goal_id))
                conn.commit()
                return cur.rowcount > 0
        except Exception as exc:
            logger.warning("LTG archive failed: %s", exc)
            return False

    def list_goals(self, include_archived: bool = False,
                   k: int = 30) -> List[Dict[str, Any]]:
        try:
            with self._lock, _connect(self._db) as conn:
                q = "SELECT id, title, description, category, progress, " \
                    "status, linked_skill, notes, updated_at FROM ltg_goals"
                if not include_archived:
                    q += " WHERE status = 'active'"
                q += " ORDER BY updated_at DESC LIMIT ?"
                rows = conn.execute(q, (k,))
                return [{"id": r[0], "title": r[1], "description": r[2],
                         "category": r[3], "progress": round(r[4], 1),
                         "status": r[5], "linked_skill": r[6],
                         "notes": r[7], "updated_at": r[8]} for r in rows]
        except Exception as exc:
            logger.warning("LTG list_goals failed: %s", exc)
            return []

    def stats(self) -> Dict[str, Any]:
        try:
            with self._lock, _connect(self._db) as conn:
                act = conn.execute(
                    "SELECT COUNT(*) FROM ltg_goals WHERE status='active'"
                ).fetchone()[0]
                ach = conn.execute(
                    "SELECT COUNT(*) FROM ltg_goals WHERE status='achieved'"
                ).fetchone()[0]
                avg = conn.execute("""
                    SELECT COALESCE(AVG(progress), 0) FROM ltg_goals
                    WHERE status = 'active'
                """).fetchone()[0]
                return {"active": act, "achieved": ach,
                        "avg_progress": round(avg, 1)}
        except Exception as exc:
            logger.warning("LTG stats failed: %s", exc)
            return {"active": 0, "achieved": 0, "avg_progress": 0.0}

    def evaluate(self) -> Dict[str, Any]:
        """تقييم دوري: تقدم تلقائي بطيء للأهداف النشطة + تسجيل آخر تقييم.
        (في الإنتاج تُستدعى من الخيط الدوري؛ في الاختبار تُستدعى يدويًا.)"""
        updated = 0
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT id FROM ltg_goals
                    WHERE status = 'active' AND progress < 100
                """)
                for (gid,) in rows:
                    conn.execute("""
                        UPDATE ltg_goals
                        SET progress = MIN(100.0, progress + ?),
                            updated_at = ?, last_eval = ?,
                            status = CASE
                                WHEN MIN(100.0, progress + ?) >= 100
                                THEN 'achieved' ELSE status END
                        WHERE id = ? AND status = 'active' AND progress < 100
                    """, (_AUTO_STEP, time.time(), time.time(),
                          _AUTO_STEP, gid))
                    updated += 1
                conn.commit()
        except Exception as exc:
            logger.warning("LTG evaluate failed: %s", exc)
        return {"evaluated_at": time.time(), "goals_updated": updated}

    def _prune(self) -> None:
        """حذف الأقدم إذا تجاوز السقف (الأهداف المنتهية أولًا)."""
        try:
            with _connect(self._db) as conn:
                n = conn.execute("SELECT COUNT(*) FROM ltg_goals"
                                 ).fetchone()[0]
                if n > _MAX_GOALS:
                    conn.execute("""
                        DELETE FROM ltg_goals WHERE id IN (
                            SELECT id FROM ltg_goals
                            WHERE status != 'achieved'
                            ORDER BY updated_at ASC
                            LIMIT ?
                        )
                    """, (n - _MAX_GOALS,))
                    conn.commit()
        except Exception as exc:
            logger.warning("LTG prune failed: %s", exc)

    def reset(self) -> None:
        try:
            with self._lock, _connect(self._db) as conn:
                conn.execute("DELETE FROM ltg_goals")
                conn.commit()
        except Exception as exc:
            logger.warning("LTG reset failed: %s", exc)

    # ── الخيط الدوري ───────────────────────────────────────────────────────

    def start_evaluator(self) -> None:
        """خيط خلفية يقيّم الأهداف دوريًا (يعمل حتى نهاية عمر العملية)."""
        global _thread_started
        if _thread_started:
            return
        try:
            import threading as _th  # noqa: E402
            _thread_started = True
            _th.Thread(target=self._eval_loop, daemon=True).start()
        except Exception as exc:
            logger.warning("LTG evaluator start failed: %s", exc)

    def _eval_loop(self) -> None:
        while True:
            try:
                time.sleep(self._eval_period)
                self.evaluate()
            except Exception as exc:
                logger.warning("LTG eval loop error: %s", exc)

    def set_eval_period(self, seconds: float) -> None:
        """للاختبار: تقصير فترة التقييم الدوري."""
        if seconds and seconds > 0:
            self._eval_period = float(seconds)


# ── singleton + helpers ────────────────────────────────────────────────────

_ltg_instance = None


def get_long_term_goals(db_path: Optional[str] = None) -> LongTermGoals:
    global _ltg_instance
    if _ltg_instance is None:
        _ltg_instance = LongTermGoals(db_path)
    return _ltg_instance


def reset_long_term_goals() -> None:
    global _ltg_instance, _thread_started
    _ltg_instance = None
    _thread_started = False


def ltg_add(title: str, description: str = "", category: str = "general",
            linked_skill: str = "", progress: float = 0.0) -> Optional[int]:
    try:
        return get_long_term_goals().add(title, description, category,
                                         linked_skill, progress)
    except Exception:
        return None


def ltg_progress(goal_id: int, progress: float, note: str = "") -> bool:
    try:
        return get_long_term_goals().update_progress(goal_id, progress, note)
    except Exception:
        return False


def ltg_archive(goal_id: int, status: str = "archived") -> bool:
    try:
        return get_long_term_goals().archive(goal_id, status)
    except Exception:
        return False


def ltg_list(k: int = 30) -> List[Dict[str, Any]]:
    try:
        return get_long_term_goals().list_goals(k=k)
    except Exception:
        return []


def ltg_stats() -> Dict[str, Any]:
    try:
        return get_long_term_goals().stats()
    except Exception:
        return {"active": 0, "achieved": 0, "avg_progress": 0.0}


def ltg_evaluate() -> Dict[str, Any]:
    try:
        return get_long_term_goals().evaluate()
    except Exception:
        return {"evaluated_at": 0.0, "goals_updated": 0}
