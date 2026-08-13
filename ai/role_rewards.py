# -*- coding: utf-8 -*-
"""
ai/role_rewards.py — نظام المكافآت الذاتية للأدوار (Role Rewards / XP System)
═════════════════════════════════════════════════════════════════════════════
سجل متراكم من نقاط الخبرة والمهارات لكل دور (وكيل) عبر المهام:
- كل دور يكتسب نقاط خبرة (XP) مع كل مهمة ناجحة، ويفقدها جزئيًا عند الفشل.
- كل دور يملك ملف مهارات (skill → score) يترقّى كلما مارس المهارة بنجاح.
- الاستحضار: عند بدء مهمة جديدة، يمكن معرفة «أفضل دور للمهارة» و
  «أفضل مهارات دور» — فيوجَّه اختيار الأدوار تلقائيًا نحو الأنسب.
- الصيانة: سقف 200 دور في السجل (LRU بالأقدم) + pruning للمهارات الصفرية.

التدهور: فشل كامل → كتلة _RR_OK في app_core تعطل الوحدة بصمت (فشل آمن).
التخزين: SQLite محلي دائم في data/role_rewards.db (منفصل عن كل قواعد البيانات
الأخرى). لا يرفع أي استدعاء استثناء خارجًا — كل شيء محمي بـ try/except.

الجداول:
  rr_roles:        دور (role_id, role_type, xp, tasks_n, successes_n, failures_n,
                   avg_score, last_seen, created_at)
  rr_skills:       مهارة دور (role_id, skill, score, hits, last_used, created_at)

التكامل: استيراد اختياري (_RR_OK) في app_core مع دوال late import من الوحدات
الفرعية لمنع circular import (نفس نمط _TEM_OK/_SKB_OK).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── الثوابت ────────────────────────────────────────────────────────────────
_LOCAL_DB = "data/role_rewards.db"
_MAX_ROLES = 200                     # سقف الأدوار المحفوظة (LRU بالأقدم)
_MAX_SKILLS_PER_ROLE = 30            # سقف مهارات الدور الواحد
_MAX_SKILL_CHARS = 80                # سقف طول اسم المهارة
_MAX_ROLE_CHARS = 120                # سقف طول اسم الدور

_XP_SUCCESS = 10.0                   # نقاط نجاح المهمة
_XP_FAILURE = -6.0                   # نقاط فشل المهمة
_XP_BASE_TASK = 3.0                  # نقاط أي مهمة (حتى بدون نتيجة)
_SKILL_SUCCESS = 1.0                 # ترقية المهارة عند نجاح
_SKILL_FAILURE = -0.6                # خفض المهارة عند فشل
_MIN_SKILL_SCORE = 0.0               # أدنى درجة مهارة (تُحذف عند الصفر)
_MAX_SKILL_SCORE = 10.0              # سقف درجة المهارة
_ROLE_XP_DECAY = 0.99                # تدهور بطيء لنقاط الدور عند كل تحديث

_SCHEMA = """
CREATE TABLE IF NOT EXISTS rr_roles (
    role_id     TEXT PRIMARY KEY,
    role_type   TEXT,
    xp          REAL NOT NULL DEFAULT 0.0,
    tasks_n     INTEGER NOT NULL DEFAULT 0,
    successes_n INTEGER NOT NULL DEFAULT 0,
    failures_n  INTEGER NOT NULL DEFAULT 0,
    avg_score   REAL NOT NULL DEFAULT 0.0,
    last_seen   REAL NOT NULL DEFAULT 0.0,
    created_at  REAL NOT NULL DEFAULT 0.0
);
CREATE TABLE IF NOT EXISTS rr_skills (
    role_id    TEXT NOT NULL,
    skill      TEXT NOT NULL,
    score      REAL NOT NULL DEFAULT 1.0,
    score_delta REAL NOT NULL DEFAULT 0.0,
    hits       INTEGER NOT NULL DEFAULT 0,
    last_used  REAL NOT NULL DEFAULT 0.0,
    created_at REAL NOT NULL DEFAULT 0.0,
    PRIMARY KEY (role_id, skill)
);
CREATE INDEX IF NOT EXISTS idx_rr_skills_score ON rr_skills (skill, score DESC);
"""

# ── الاتصال الآمن ──────────────────────────────────────────────────────────

def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    return conn


class RoleRewards:
    """سجل نقاط الخبرة والمهارات للأدوار (thread-safe، تدهور آمن)."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db = db_path or _LOCAL_DB
        self._lock = threading.RLock()
        with self._lock, _connect(self._db):
            pass  # التأكد من إنشاء الجداول

    # ── مكافأة/عقاب الدور ──────────────────────────────────────────────────

    def award(self, role_id: str, outcome: Optional[str] = None,
              role_type: Optional[str] = None, task_id: Optional[str] = None,
              skills: Optional[List[str]] = None) -> bool:
        """
        منح/خصم نقاط خبرة لدور بعد مهمة.
        outcome: "success" | "failure" | None (مهمة لم تُحسم بعد).
        skills: مهارات مارسها الدور في هذه المهمة تُحدَّث درجاتها.
        """
        if not role_id:
            return False
        role_id = role_id.strip()[:_MAX_ROLE_CHARS]
        outcome = (outcome or "").lower().strip()
        try:
            with self._lock, _connect(self._db) as conn:
                if outcome == "success":
                    delta = _XP_SUCCESS
                elif outcome == "failure":
                    delta = _XP_FAILURE
                else:
                    delta = _XP_BASE_TASK
                conn.execute("""
                    INSERT INTO rr_roles (role_id, role_type, xp, tasks_n,
                                          successes_n, failures_n, avg_score,
                                          last_seen, created_at)
                    VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                    ON CONFLICT (role_id) DO UPDATE SET
                        role_type = COALESCE(excluded.role_type, rr_roles.role_type),
                        xp = MAX(-50.0, (rr_roles.xp * %f) + excluded.xp),
                        tasks_n = rr_roles.tasks_n + 1,
                        successes_n = rr_roles.successes_n + excluded.successes_n,
                        failures_n = rr_roles.failures_n + excluded.failures_n,
                        avg_score = CASE WHEN rr_roles.tasks_n + 1 > 0
                            THEN (rr_roles.avg_score * rr_roles.tasks_n +
                                  CASE WHEN excluded.successes_n > 0 THEN 1.0
                                       WHEN excluded.failures_n > 0 THEN 0.0
                                       ELSE rr_roles.avg_score END)
                                 / (rr_roles.tasks_n + 1)
                            ELSE rr_roles.avg_score END,
                        last_seen = excluded.last_seen
                """ % _ROLE_XP_DECAY, (
                    role_id, (role_type or "")[:60], delta,
                    1 if outcome == "success" else 0,
                    1 if outcome == "failure" else 0, delta,
                    time.time(), time.time()))
                if skills:
                    for skill in skills:
                        skill = skill.strip()[:_MAX_SKILL_CHARS]
                        if not skill:
                            continue
                        delta_s = _SKILL_SUCCESS if outcome == "success" \
                            else (_SKILL_FAILURE if outcome == "failure" else 0.2)
                        conn.execute("""
                            INSERT INTO rr_skills (role_id, skill, score,
                                                   score_delta, hits,
                                                   last_used, created_at)
                            VALUES (?, ?, 1.0, ?, 1, ?, ?)
                            ON CONFLICT (role_id, skill) DO UPDATE SET
                                score = MIN(%f, MAX(%f,
                                    rr_skills.score + excluded.score_delta)),
                                hits = rr_skills.hits + 1,
                                last_used = excluded.last_used
                        """ % (_MAX_SKILL_SCORE, _MIN_SKILL_SCORE), (
                            role_id, skill, delta_s, time.time(), time.time()))
                conn.commit()
                self._prune_roles()
            return True
        except Exception as exc:  # تدهور آمن: فشل صامت
            logger.warning("RoleRewards award failed: %s", exc)
            return False

    # ── الاستحضار ──────────────────────────────────────────────────────────

    def top_roles_for_skill(self, skill: str, k: int = 5) -> List[Dict[str, Any]]:
        """أفضل الأدوار لمهارة معطاة (بدرجة المهارة × نقاط الخبرة)."""
        if not skill:
            return []
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT s.role_id, s.score, r.xp, r.successes_n, r.tasks_n
                    FROM rr_skills s
                    JOIN rr_roles r ON r.role_id = s.role_id
                    WHERE s.skill LIKE ? AND s.score > 0
                    ORDER BY s.score * (r.xp + 10) DESC
                    LIMIT ?
                """, ("%" + skill.strip()[:_MAX_SKILL_CHARS] + "%", k))
                return [{"role": r[0], "skill_score": r[1], "xp": r[2],
                         "successes": r[3], "tasks": r[4]} for r in rows]
        except Exception as exc:
            logger.warning("RoleRewards top_roles_for_skill failed: %s", exc)
            return []

    def skills_for_role(self, role_id: str, k: int = 10) -> List[Dict[str, Any]]:
        """أعلى مهارات دور معطى."""
        if not role_id:
            return []
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT skill, score, hits FROM rr_skills
                    WHERE role_id = ? AND score > 0
                    ORDER BY score DESC, hits DESC
                    LIMIT ?
                """, (role_id.strip()[:_MAX_ROLE_CHARS], k))
                return [{"skill": r[0], "score": r[1], "hits": r[2]}
                        for r in rows]
        except Exception as exc:
            logger.warning("RoleRewards skills_for_role failed: %s", exc)
            return []

    def role_summary(self, role_id: str) -> Optional[Dict[str, Any]]:
        """ملخص دور: نقاطه وإحصاءاته وأفضل مهاراته."""
        try:
            with self._lock, _connect(self._db) as conn:
                row = conn.execute("""
                    SELECT role_id, role_type, xp, tasks_n, successes_n,
                           failures_n, avg_score, last_seen
                    FROM rr_roles WHERE role_id = ?
                """, (role_id.strip()[:_MAX_ROLE_CHARS],)).fetchone()
                if not row:
                    return None
                return {"role_id": row[0], "role_type": row[1], "xp": row[2],
                        "tasks": row[3], "successes": row[4],
                        "failures": row[5], "avg_score": row[6],
                        "last_seen": row[7],
                        "top_skills": self.skills_for_role(row[0], 5)}
        except Exception as exc:
            logger.warning("RoleRewards role_summary failed: %s", exc)
            return None

    def ranking(self, k: int = 10) -> List[Dict[str, Any]]:
        """ترتيب الأدوار الأعلى نقاط خبرة (متأخر النشاط)."""
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT role_id, role_type, xp, successes_n, tasks_n,
                           last_seen
                    FROM rr_roles
                    ORDER BY xp DESC, last_seen DESC
                    LIMIT ?
                """, (k,))
                return [{"role": r[0], "role_type": r[1], "xp": r[2],
                         "successes": r[3], "tasks": r[4]} for r in rows]
        except Exception as exc:
            logger.warning("RoleRewards ranking failed: %s", exc)
            return []

    def stats(self) -> Dict[str, Any]:
        try:
            with self._lock, _connect(self._db) as conn:
                r = conn.execute(
                    "SELECT COUNT(*) FROM rr_roles").fetchone()[0]
                s = conn.execute(
                    "SELECT COUNT(*) FROM rr_skills").fetchone()[0]
                top = conn.execute(
                    "SELECT COALESCE(SUM(xp),0) FROM rr_roles").fetchone()[0]
                return {"roles": r, "skill_entries": s, "total_xp": round(top, 1)}
        except Exception as exc:
            logger.warning("RoleRewards stats failed: %s", exc)
            return {"roles": 0, "skill_entries": 0, "total_xp": 0.0}

    def latest(self, k: int = 10) -> List[Dict[str, Any]]:
        """آخر الأدوار نشاطًا."""
        try:
            with self._lock, _connect(self._db) as conn:
                rows = conn.execute("""
                    SELECT role_id, role_type, xp, tasks_n, successes_n,
                           last_seen
                    FROM rr_roles ORDER BY last_seen DESC LIMIT ?
                """, (k,))
                return [{"role": r[0], "role_type": r[1], "xp": r[2],
                         "tasks": r[3], "successes": r[4]} for r in rows]
        except Exception as exc:
            logger.warning("RoleRewards latest failed: %s", exc)
            return []

    # ── الصيانة ─────────────────────────────────────────────────────────────

    def _prune_roles(self) -> None:
        """حذف الأدوار القديمة الزائدة (LRU) — يُستدعى داخل القفل."""
        try:
            cur = self._lock  # فقط للتأكد أننا داخل قفل caller
            with _connect(self._db) as conn:
                excess = conn.execute(
                    "SELECT COUNT(*) - ? FROM rr_roles", (_MAX_ROLES,)).fetchone()[0]
                if excess and excess > 0:
                    conn.execute("""
                        DELETE FROM rr_roles WHERE role_id IN (
                            SELECT role_id FROM rr_roles
                            ORDER BY last_seen ASC LIMIT ?
                        )
                    """, (excess,))
                    conn.execute("""
                        DELETE FROM rr_skills WHERE role_id NOT IN (
                            SELECT role_id FROM rr_roles
                        )
                    """)
                conn.execute(
                    "DELETE FROM rr_skills WHERE score <= ?", (_MIN_SKILL_SCORE,))
                conn.commit()
        except Exception as exc:
            logger.warning("RoleRewards prune failed: %s", exc)

    def reset(self) -> None:
        """حذف كل السجلات (للاختبار فقط)."""
        try:
            with self._lock, _connect(self._db) as conn:
                conn.execute("DELETE FROM rr_skills")
                conn.execute("DELETE FROM rr_roles")
                conn.commit()
        except Exception as exc:
            logger.warning("RoleRewards reset failed: %s", exc)


# ── singleton + helpers ────────────────────────────────────────────────────

_rr_instance = None


def get_role_rewards(db_path: Optional[str] = None) -> RoleRewards:
    global _rr_instance
    if _rr_instance is None:
        _rr_instance = RoleRewards(db_path)
    return _rr_instance


def reset_role_rewards() -> None:
    global _rr_instance
    _rr_instance = None


# ── واجهة عالية المستوى ─────────────────────────────────────────────────────

def reward_role(role_id: str, outcome: Optional[str] = None,
                role_type: Optional[str] = None, task_id: Optional[str] = None,
                skills: Optional[List[str]] = None) -> bool:
    """إعطاب/خصم نقاط خبرة لدور بعد مهمة."""
    try:
        return get_role_rewards().award(role_id, outcome, role_type,
                                        task_id, skills)
    except Exception:
        return False


def best_roles_for(skill: str, k: int = 5) -> List[Dict[str, Any]]:
    return get_role_rewards().top_roles_for_skill(skill, k)


def role_skills(role_id: str, k: int = 10) -> List[Dict[str, Any]]:
    return get_role_rewards().skills_for_role(role_id, k)


def rr_stats() -> Dict[str, Any]:
    return get_role_rewards().stats()


def rr_latest(k: int = 10) -> List[Dict[str, Any]]:
    return get_role_rewards().latest(k)
