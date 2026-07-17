"""
ai/task_manager.py
====================
نظام المهام المتعددة (Multi-Task Manager).

يعالج فجوتين في ai/nsm_planner.py القديم:
  1. حقل `depends_on` في PlanTask كان يُقرأ ويُعرض فقط، ولم يكن يُستخدم فعلياً
     في ترتيب التنفيذ — المهام كانت تُنفَّذ بترتيب القائمة فقط بغض النظر عن
     تبعياتها. هذا الملف يوفر ترتيب طوبولوجي حقيقي (Kahn's algorithm).
  2. كل خطة (AppPlan) كانت تعيش فقط في الذاكرة أثناء التنفيذ ثم تُفقد —
     لا وجود لتتبّع خطط متعددة متزامنة أو حالتها عبر الجلسات. هذا الملف
     يحفظ كل خطة ومهامها في SQLite (memory/task_manager.db) بحيث يمكن أن
     تتعايش عدة خطط (multi-task) ويُستعلَم عن حالتها لاحقاً.

الاستخدام النموذجي (من ai/nsm_planner.py):
    from ai.task_manager import (
        create_plan, update_task_status, mark_plan_status,
        topological_order, format_status_report,
    )

    plan_id = create_plan(plan)                       # عند بناء الخطة
    ordered_tasks = topological_order(plan.tasks)      # ترتيب تنفيذ صحيح
    ...
    update_task_status(plan_id, task.id, "done", result)
    ...
    mark_plan_status(plan_id, "done")
"""
from __future__ import annotations

import json
import sqlite3
import threading
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Sequence

DB_PATH = Path("memory/task_manager.db")
_LOCK = threading.Lock()

_SCHEMA = """
CREATE TABLE IF NOT EXISTS plans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    idea        TEXT,
    app_name    TEXT,
    app_type    TEXT,
    description TEXT,
    tech_stack  TEXT,
    status      TEXT DEFAULT 'running',   -- running | done | failed
    created_at  TEXT DEFAULT CURRENT_TIMESTAMP,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS plan_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id     INTEGER NOT NULL,
    task_id     INTEGER NOT NULL,
    title       TEXT,
    description TEXT,
    task_type   TEXT,
    files       TEXT,
    depends_on  TEXT,
    status      TEXT DEFAULT 'pending',   -- pending | running | done | failed
    result      TEXT,
    updated_at  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_plan_tasks_plan ON plan_tasks(plan_id);
"""

MAX_PLANS = 500  # سقف الاحتفاظ لمنع نمو الملف بلا حدود


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.executescript(_SCHEMA)
    return conn


# ══════════════════════════════════════════════════════════════════
# ترتيب طوبولوجي حقيقي يحترم depends_on
# ══════════════════════════════════════════════════════════════════

def topological_order(tasks: Sequence[Any]) -> List[Any]:
    """
    يُعيد المهام مرتّبة بحيث لا تُنفَّذ أي مهمة قبل كل ما تعتمد عليه
    (Kahn's algorithm). كل عنصر في tasks يجب أن يملك خاصيتي `id`
    و`depends_on` (قائمة أرقام معرّفات).

    - يتجاهل تبعيات لمعرّفات غير موجودة في القائمة (لا يعلّق التنفيذ).
    - عند وجود حلقة تبعية دائرية (خطأ من LLM): يُعيد الترتيب الأصلي
      بدل التعليق — أفضل من توقف كامل.
    - يحافظ على ترتيب القائمة الأصلي بين المهام المتساوية بلا تبعية
      (ترتيب ثابت ومتوقع).
    """
    if not tasks:
        return []

    id_to_task = {t.id: t for t in tasks}
    valid_ids = set(id_to_task.keys())

    # عدد التبعيات "الحقيقية" لكل مهمة (تجاهل التبعيات لمعرّفات غير موجودة)
    indegree: Dict[int, int] = {}
    dependents: Dict[int, List[int]] = {tid: [] for tid in valid_ids}
    for t in tasks:
        deps = [d for d in (t.depends_on or []) if d in valid_ids and d != t.id]
        indegree[t.id] = len(deps)
        for d in deps:
            dependents[d].append(t.id)

    # ابدأ بالمهام بلا تبعيات، بنفس ترتيب القائمة الأصلية
    queue = deque([t.id for t in tasks if indegree[t.id] == 0])
    visited = set()
    ordered_ids: List[int] = []

    while queue:
        tid = queue.popleft()
        if tid in visited:
            continue
        visited.add(tid)
        ordered_ids.append(tid)
        for nxt in dependents.get(tid, []):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(ordered_ids) != len(tasks):
        # حلقة دائرية اكتُشفت — فشل الترتيب الطوبولوجي، ارجع للترتيب الأصلي
        return list(tasks)

    return [id_to_task[tid] for tid in ordered_ids]


# ══════════════════════════════════════════════════════════════════
# الحفظ الدائم (multi-task persistence)
# ══════════════════════════════════════════════════════════════════

def create_plan(plan: Any) -> int:
    """يحفظ خطة جديدة وكل مهامها، ويعيد plan_id. لا يرمي استثناءً أبداً
    (يعيد -1 عند الفشل) حتى لا يُعطّل تدفق التخطيط الرئيسي."""
    with _LOCK:
        try:
            conn = _connect()
            with conn:
                cur = conn.execute(
                    """INSERT INTO plans (idea, app_name, app_type, description, tech_stack)
                       VALUES (?, ?, ?, ?, ?)""",
                    (plan.idea, plan.app_name, plan.app_type, plan.description,
                     json.dumps(plan.tech_stack, ensure_ascii=False)),
                )
                plan_id = cur.lastrowid
                for t in plan.tasks:
                    conn.execute(
                        """INSERT INTO plan_tasks
                           (plan_id, task_id, title, description, task_type, files, depends_on, status)
                           VALUES (?, ?, ?, ?, ?, ?, ?, 'pending')""",
                        (plan_id, t.id, t.title, t.description, t.task_type,
                         json.dumps(t.files, ensure_ascii=False),
                         json.dumps(t.depends_on, ensure_ascii=False)),
                    )
                # تقليم الخطط القديمة جداً
                conn.execute(
                    """DELETE FROM plans WHERE id NOT IN (
                           SELECT id FROM plans ORDER BY id DESC LIMIT ?
                       )""",
                    (MAX_PLANS,),
                )
            conn.close()
            return int(plan_id)
        except Exception:
            return -1


def update_task_status(plan_id: int, task_id: int, status: str, result: str = "") -> None:
    if plan_id is None or plan_id < 0:
        return
    with _LOCK:
        try:
            conn = _connect()
            with conn:
                conn.execute(
                    """UPDATE plan_tasks SET status = ?, result = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE plan_id = ? AND task_id = ?""",
                    (status, (result or "")[:2000], plan_id, task_id),
                )
            conn.close()
        except Exception:
            pass


def mark_plan_status(plan_id: int, status: str) -> None:
    if plan_id is None or plan_id < 0:
        return
    with _LOCK:
        try:
            conn = _connect()
            with conn:
                conn.execute(
                    "UPDATE plans SET status = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                    (status, plan_id),
                )
            conn.close()
        except Exception:
            pass


def get_active_plans(limit: int = 20) -> List[Dict[str, Any]]:
    """يُعيد ملخصاً لآخر N خطة (نشطة أو منتهية) — أساس تتبّع المهام المتعددة."""
    with _LOCK:
        try:
            conn = _connect()
            rows = conn.execute(
                """SELECT id, idea, app_name, app_type, status, created_at
                   FROM plans ORDER BY id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
            result = []
            for r in rows:
                plan_id = r[0]
                counts = conn.execute(
                    """SELECT
                           SUM(CASE WHEN status='done' THEN 1 ELSE 0 END),
                           SUM(CASE WHEN status='failed' THEN 1 ELSE 0 END),
                           COUNT(*)
                       FROM plan_tasks WHERE plan_id = ?""",
                    (plan_id,),
                ).fetchone()
                done, failed, total = (counts[0] or 0), (counts[1] or 0), (counts[2] or 0)
                result.append({
                    "id": plan_id, "idea": r[1], "app_name": r[2], "app_type": r[3],
                    "status": r[4], "created_at": r[5],
                    "done": done, "failed": failed, "total": total,
                })
            conn.close()
            return result
        except Exception:
            return []


def format_status_report(limit: int = 10) -> str:
    """تقرير عربي مختصر عن حالة آخر الخطط — لعرضه مباشرة في المحادثة."""
    plans = get_active_plans(limit=limit)
    if not plans:
        return "📋 لا توجد أي خطط مسجَّلة حتى الآن."

    lines = [f"## 📋 حالة المهام (آخر {len(plans)} خطة)\n"]
    for p in plans:
        icon = {"running": "🔄", "done": "✅", "failed": "⚠️"}.get(p["status"], "•")
        lines.append(
            f"{icon} **{p['app_name']}** — {p['done']}/{p['total']} مهمة منجزة"
            + (f" ({p['failed']} فشلت)" if p["failed"] else "")
            + f"\n   {p['idea'][:70]}"
        )
    return "\n".join(lines) + "\n"
