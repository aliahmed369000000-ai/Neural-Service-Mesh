"""
ai/agent_background_tasks.py
=============================
🆕 مهام خلفية حقيقية للوكلاء (Long-Running Background Tasks).

تحوّل AutonomousWill من "باحث-مقترِح" إلى "منفّذ" يعمل في الخلفية:

  • Queue مهام دائم (SQLite في artifacts/agent_background/)
  • منفّذ خيط خلفي (daemon) يعمل على حلقة agent_loop الكاملة
  • poll من الواجهة كل ن ثوانٍ (get_task / list_tasks)
  • سجل مراجعة كامل (audit) في artifacts/agent_loop/audit/
  • قابلية الإيقاف/الاستئناف/الحذف

استخدام من الدردشة (مثال):
  "نفّذ في الخلفية: افحص ai/goal_planner.py وأصلح أي خطأ"
  → enqueue_background_task(prompt) → loop_id
  → الواجهة تستعلم عبر list_background_tasks / get_background_task(loop_id)
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("NeuralServiceMesh.BackgroundTasks")

ROOT = Path(__file__).resolve().parent.parent
_DB_DIR = ROOT / "artifacts" / "agent_background"
_DB_DIR.mkdir(parents=True, exist_ok=True)
_DB_PATH = _DB_DIR / "tasks.db"

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=10)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bg_tasks (
            loop_id TEXT PRIMARY KEY,
            prompt TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            result TEXT,
            rounds INTEGER DEFAULT 0,
            tools_used INTEGER DEFAULT 0
        )
    """)
    conn.commit()
    return conn


# ── واجهة عامة ───────────────────────────────────────────────────────
def enqueue_background_task(prompt: str) -> Dict[str, Any]:
    """إضافة مهمة إلى قائمة الانتظار — تعيد loop_id فورًا."""
    loop_id = f"bg_{int(time.time() * 1000) % 10**8:08d}_{hex(hash(prompt))[-6:]}"
    with _lock:
        conn = _connect()
        conn.execute(
            "INSERT OR REPLACE INTO bg_tasks "
            "(loop_id, prompt, status, created_at) VALUES (?,?,?,?)",
            (loop_id, (prompt or "").strip()[:2000], "queued", _now()))
        conn.commit()
    _ensure_runner()
    logger.info("[BackgroundTasks] enqueued %s", loop_id)
    return {"ok": True, "loop_id": loop_id, "status": "queued"}


def list_background_tasks(limit: int = 30) -> List[Dict[str, Any]]:
    with _lock:
        conn = _connect()
        rows = conn.execute(
            "SELECT loop_id, prompt, status, created_at, started_at, "
            "finished_at, rounds, tools_used FROM bg_tasks "
            "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
    return [{"loop_id": r[0], "prompt": r[1][:200], "status": r[2],
             "created_at": r[3], "started_at": r[4], "finished_at": r[5],
             "rounds": r[6], "tools_used": r[7]} for r in rows]


def get_background_task(loop_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _connect()
        row = conn.execute(
            "SELECT * FROM bg_tasks WHERE loop_id=?", (loop_id,)).fetchone()
    if not row:
        return None
    cols = ["loop_id", "prompt", "status", "created_at", "started_at",
            "finished_at", "result", "rounds", "tools_used"]
    return dict(zip(cols, row))


def cancel_background_task(loop_id: str) -> Dict[str, Any]:
    with _lock:
        conn = _connect()
        cur = conn.execute(
            "UPDATE bg_tasks SET status='cancelled', finished_at=? "
            "WHERE loop_id=? AND status IN ('queued','running')",
            (_now(), loop_id))
        conn.commit()
    if cur.rowcount:
        cancelled_ids.append(loop_id)
        return {"ok": True, "loop_id": loop_id, "status": "cancelled"}
    return {"ok": False, "error": "مهمة غير موجودة أو منتهية"}


def delete_background_task(loop_id: str) -> Dict[str, Any]:
    with _lock:
        conn = _connect()
        cur = conn.execute("DELETE FROM bg_tasks WHERE loop_id=?", (loop_id,))
        conn.commit()
    return {"ok": bool(cur.rowcount), "deleted": loop_id}


cancelled_ids: List[str] = []


# ── المنفّذ الخلفي ────────────────────────────────────────────────────
_runner: Optional[threading.Thread] = None
_runner_started = False


def _ensure_runner() -> None:
    global _runner, _runner_started
    if _runner_started and _runner and _runner.is_alive():
        return
    _runner_started = True
    _runner = threading.Thread(target=_run_loop, name="AgentBackgroundRunner",
                               daemon=True)
    _runner.start()


def _mark(loop_id: str, **fields: Any) -> None:
    sets, vals = [], []
    for k, v in fields.items():
        sets.append(f"{k}=?")
        vals.append(v)
    if sets:
        with _lock:
            conn = _connect()
            conn.execute(f"UPDATE bg_tasks SET {','.join(sets)} "
                         f"WHERE loop_id=?", vals + [loop_id])
            conn.commit()


def _run_loop() -> None:
    """حلقة منفّذ: تسحب مهمة queued وتنفّذها عبر run_agent_loop."""
    logger.info("[BackgroundTasks] runner started")
    while True:
        try:
            task = None
            with _lock:
                conn = _connect()
                row = conn.execute(
                    "SELECT loop_id, prompt FROM bg_tasks "
                    "WHERE status='queued' ORDER BY created_at ASC LIMIT 1"
                ).fetchone()
                if row:
                    task = {"loop_id": row[0], "prompt": row[1]}
                    conn.execute(
                        "UPDATE bg_tasks SET status='running', started_at=? "
                        "WHERE loop_id=?", (_now(), row[0]))
                    conn.commit()
            if task is None:
                time.sleep(5)
                continue
            loop_id = task["loop_id"]
            if loop_id in cancelled_ids:
                cancelled_ids.remove(loop_id)
                continue
            try:
                from ai.agent_loop import run_agent_loop

                events: List[Dict[str, Any]] = []
                for ev in run_agent_loop(task["prompt"]):
                    events.append(ev)
                    if loop_id in cancelled_ids:
                        cancelled_ids.remove(loop_id)
                        _mark(loop_id, status="cancelled", finished_at=_now())
                        break
                else:
                    n_ans = sum(1 for e in events if e.get("type") == "answer")
                    # تصنيف الحالة الفعلي: إن كانت آخر رسالة نص تحذير فشل
                    # الحلقة (تعذّر/لا أستطع) فالنتيجة failed وليست completed
                    final_status = "completed"
                    last_answer = next((e for e in reversed(events)
                                        if e.get("type") == "answer"), None)
                    if last_answer:
                        txt = str(last_answer.get("text", ""))
                        if txt.startswith("⚠️") and (
                                "تعذّر" in txt or "استطع" in txt):
                            final_status = "failed"
                    _mark(loop_id, status=final_status, finished_at=_now(),
                          result=json.dumps({"events": len(events),
                                             "answers": n_ans})[:3000],
                          rounds=max((e.get("round", 0) for e in events
                                      if e.get("type") == "status"), default=0),
                          tools_used=max((e.get("total_tools", 0) for e in events
                                          if e.get("type") == "status"), default=0))
            except Exception as e:
                logger.warning("[BackgroundTasks] %s failed: %s", loop_id, e)
                _mark(loop_id, status="failed", finished_at=_now(),
                      result=str(e)[:2000])
        except Exception as exc:
            logger.warning("[BackgroundTasks] runner tick error: %s", exc)
            time.sleep(10)


# ── تكامل التنفيذ الذاتي المحكوم (AutonomousWill — المرحلة 5) ────────
def _record_autonomous_execution(topic: str, motive: str) -> dict:
    """تسجيل إنجاز ذاتي موثق في قائمة المهام (للتنسيق مع AutonomousWill)."""
    res = enqueue_background_task(f"[إرادة ذاتية — {motive}] {topic}")
    return {"ok": True, "loop_id": res["loop_id"], "topic": topic,
            "motive": motive, "ts": _now()}
