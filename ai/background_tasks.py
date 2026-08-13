# -*- coding: utf-8 -*-
"""
ai/background_tasks.py — مهام الخلفية بين الوكلاء
==================================================
تنفيذ المهام الثقيلة دون حجز واجهة Streamlit، مع سجل دائم (SQLite)
وأحداث فورية على ناقل الأحداث (bg_task_started/running/done/failed/cancelled).

التصميم:
- BackgroundTaskManager (singleton): طابور محدود MAX_PENDING=8، منع تكرار نفس
  المهمة قيد التنفيذ، منع تجاوز MAX_CONCURRENT=3 مهام متزامنة.
- كل مهمة تُنفَّذ في خيط daemon عبر delegate_to_unified_chat (المسار الموحد
  للوكلاء) — بدون أي اعتماد على streamlit داخل الوحدة.
- كل مسار (submit/join/list) قابل للاختبار بدون مفاتيح API حقيقية.
"""
from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
import uuid
from collections import Counter
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ai.background_tasks")

MAX_PENDING = 8          # الحد الأقصى للمهام المعلقة
MAX_CONCURRENT = 3       # الحد الأقصى للمهام المتزامنة
MAX_HISTORY = 500        # ضغط السجل عند التجاوز (LRU)
DB_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "background_tasks.db")

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

BG_EVENTS = {
    "bg_task_started",
    "bg_task_running",
    "bg_task_done",
    "bg_task_failed",
    "bg_task_cancelled",
}


def _now_iso() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


class TaskRecord:
    """سجل مهمة خلفية."""

    def __init__(self, prompt: str, task_id: Optional[str] = None, **extra: Any):
        self.task_id = task_id or uuid.uuid4().hex[:12]
        self.prompt = (prompt or "").strip()
        self.status = STATUS_PENDING
        self.title = self.prompt[:60] or "مهمة خلفية"
        self.route = ""
        self.response = ""
        self.error = ""
        self.created_at = _now_iso()
        self.started_at = ""
        self.finished_at = ""
        self.duration_ms = 0.0
        for k, v in extra.items():
            setattr(self, k, v)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "title": self.title,
            "prompt": self.prompt,
            "status": self.status,
            "route": self.route,
            "response": self.response,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_ms": round(self.duration_ms, 1),
        }


class BackgroundTaskManager:
    """منسّق مهام الخلفية — واحد لكل عملية."""

    def __init__(self, db_path: str = DB_DEFAULT, execute_fn: Optional[Callable] = None):
        self.db_path = db_path
        self._execute_fn = execute_fn or self._default_execute
        self._lock = threading.Lock()
        self._tasks: Dict[str, TaskRecord] = {}
        self._running_keys: Dict[str, str] = {}  # normalized prompt -> task_id قيد التنفيذ
        self._queue: List[str] = []              # task_ids
        self._ensure_db()
        self._load_history()

    # ── قاعدة البيانات ─────────────────────────────────────────────
    def _ensure_db(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS background_tasks (
                        task_id TEXT PRIMARY KEY,
                        title TEXT, prompt TEXT, status TEXT, route TEXT,
                        response TEXT, error TEXT,
                        created_at TEXT, started_at TEXT, finished_at TEXT,
                        duration_ms REAL
                    )
                """)
                conn.commit()
        except Exception as exc:  # لا تعطل الوحدة إذا تعذر SQLite (اختبار في tmp readonly)
            logger.warning("background_tasks: تعذّر تهيئة قاعدة البيانات %s: %s", self.db_path, exc)

    def _persist(self, task: TaskRecord) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO background_tasks
                    (task_id, title, prompt, status, route, response, error,
                     created_at, started_at, finished_at, duration_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (task.task_id, task.title, task.prompt, task.status, task.route,
                      task.response, task.error, task.created_at, task.started_at,
                      task.finished_at, task.duration_ms))
                conn.commit()
        except Exception as exc:
            logger.warning("background_tasks: تعذّر حفظ المهمة %s: %s", task.task_id, exc)

    def _load_history(self) -> None:
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("""
                    SELECT task_id, title, prompt, status, route, response, error,
                           created_at, started_at, finished_at, duration_ms
                    FROM background_tasks
                    ORDER BY created_at DESC LIMIT ?
                """, (MAX_HISTORY,))
                cols = [d[0] for d in cur.description]
                for row in cur:
                    rec = dict(zip(cols, row))
                    task = TaskRecord(rec.get("prompt", ""))
                    for k, v in rec.items():
                        setattr(task, k, v)
                    if task.status not in (STATUS_DONE, STATUS_FAILED):
                        continue  # لا نحمّل مهامًا غير مكتملة من دورات سابقة
                    self._tasks[task.task_id] = task
        except Exception as exc:
            logger.warning("background_tasks: تعذّر تحميل السجل: %s", exc)

    def _prune(self) -> None:
        """ضغط السجل إلى MAX_HISTORY بالأقدمية (تجاهل قيد التنفيذ)."""
        completed = [t for t in self._tasks.values() if t.status in (STATUS_DONE, STATUS_FAILED, STATUS_CANCELLED)]
        if len(completed) <= MAX_HISTORY:
            return
        completed.sort(key=lambda t: t.created_at)
        overflow = completed[: len(completed) - MAX_HISTORY]
        for task in overflow:
            self._tasks.pop(task.task_id, None)
            try:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("DELETE FROM background_tasks WHERE task_id = ?", (task.task_id,))
                    conn.commit()
            except Exception:
                pass

    # ── التنفيذ ────────────────────────────────────────────────────
    @staticmethod
    def _default_execute(prompt: str) -> tuple:
        """المسار الافتراضي: وكيل عبر delegate_to_unified_chat → (response, meta)."""
        try:
            from ai.agent_orchestrator import delegate_to_unified_chat
            return delegate_to_unified_chat(prompt)
        except Exception as exc:
            return "", {"route_method": "", "category_title": "", "error": str(exc)}

    def _run_task(self, task: TaskRecord) -> None:
        t0 = time.time()
        task.status = STATUS_RUNNING
        task.started_at = _now_iso()
        self._persist(task)
        self._emit("bg_task_running", task, detail="بدأ تنفيذ المهمة في الخلفية")
        try:
            response, meta = self._execute_fn(task.prompt)
            route = meta.get("route_method") or meta.get("category_key") or ""
            task.route = meta.get("category_title", "") or route
            task.response = response or ""
            task.status = STATUS_DONE
            self._emit("bg_task_done", task,
                       detail=f"اكتملت المهمة: {task.title}")
        except Exception as exc:  # لا تسمح بانكسار الخيط الصامت
            logger.error("background_tasks: فشل %s: %s", task.task_id, exc)
            task.status = STATUS_FAILED
            task.error = str(exc)[:200]
            self._emit("bg_task_failed", task, detail=f"فشلت المهمة: {task.error}")
        finally:
            task.finished_at = _now_iso()
            task.duration_ms = (time.time() - t0) * 1000
            with self._lock:
                self._running_keys.pop(self._norm(task.prompt), None)
            self._persist(task)
            self._prune()

    @staticmethod
    def _norm(prompt: str) -> str:
        return " ".join((prompt or "").strip().split()).lower()

    def _emit(self, event_type: str, task: TaskRecord, detail: str = "") -> None:
        try:
            from ai.agent_event_bus import emit_event
            emit_event(
                event_type, agent_id="background", title=task.title,
                status=task.status, detail=detail,
                metadata={
                    "task_id": task.task_id,
                    "status": task.status,
                    "route": task.route,
                    "duration_ms": task.duration_ms,
                },
            )
        except Exception as exc:
            logger.debug("background_tasks: تعذّر إطلاق حدث %s: %s", event_type, exc)

    # ── API العام ──────────────────────────────────────────────────
    def submit(self, prompt: str, title: Optional[str] = None) -> Optional[TaskRecord]:
        """تقديم مهمة خلفية. يُرجع None عند تجاوز الطابور أو تكرار قيد التنفيذ."""
        prompt = (prompt or "").strip()
        if not prompt:
            return None
        with self._lock:
            norm = self._norm(prompt)
            if norm in self._running_keys:
                return self._tasks.get(self._running_keys[norm])  # إعادة المهمة الجارية
            running_count = sum(1 for t in self._tasks.values() if t.status == STATUS_RUNNING)
            if len(self._queue) >= MAX_PENDING or running_count >= MAX_CONCURRENT and self._queue:
                return None
            task = TaskRecord(prompt)
            if title:
                task.title = title
            self._tasks[task.task_id] = task
            self._queue.append(task.task_id)
            self._running_keys[norm] = task.task_id  # حجز المفتاح منذ الجدولة لمنع التكرار
        self._persist(task)
        self._emit("bg_task_started", task,
                   detail=f"مهمة جديدة في الخلفية: {task.title}")
        threading.Thread(
            target=self._run_task, args=(task,), name=f"NSM-bg-{task.task_id[:6]}",
            daemon=True,
        ).start()
        return task

    def cancel(self, task_id: str) -> bool:
        """إلغاء مهمة معلقة فقط."""
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None or task.status != STATUS_PENDING:
                return False
            task.status = STATUS_CANCELLED
            task.finished_at = _now_iso()
            self._queue = [t for t in self._queue if t != task_id]
        self._persist(task)
        self._emit("bg_task_cancelled", task, detail="ألغى المستخدم المهمة")
        return True

    def get(self, task_id: str) -> Optional[TaskRecord]:
        return self._tasks.get(task_id)

    def list_tasks(self, limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
        tasks = [t for t in self._tasks.values() if status is None or t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return [t.to_dict() for t in tasks[:limit]]

    def latest_done(self) -> Optional[Dict[str, Any]]:
        """أحدث مهمة مكتملة (للاشتراك اللحظي من الواجهة)."""
        done = [t for t in self._tasks.values() if t.status == STATUS_DONE]
        done.sort(key=lambda t: t.finished_at, reverse=True)
        return done[0].to_dict() if done else None

    def status(self) -> Dict[str, Any]:
        counter = Counter(t.status for t in self._tasks.values())
        return {
            "total": len(self._tasks),
            "pending": counter.get(STATUS_PENDING, 0),
            "running": counter.get(STATUS_RUNNING, 0),
            "done": counter.get(STATUS_DONE, 0),
            "failed": counter.get(STATUS_FAILED, 0),
            "cancelled": counter.get(STATUS_CANCELLED, 0),
            "max_pending": MAX_PENDING,
            "max_concurrent": MAX_CONCURRENT,
        }


_MANAGER: Optional[BackgroundTaskManager] = None
_MANAGER_LOCK = threading.Lock()


def get_background_task_manager(execute_fn: Optional[Callable] = None) -> BackgroundTaskManager:
    """singleton على مستوى العملية."""
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = BackgroundTaskManager(execute_fn=execute_fn)
    if execute_fn is not None:
        _MANAGER._execute_fn = execute_fn
    return _MANAGER


__all__ = [
    "BackgroundTaskManager",
    "TaskRecord",
    "get_background_task_manager",
    "MAX_PENDING",
    "MAX_CONCURRENT",
    "MAX_HISTORY",
    "STATUS_PENDING", "STATUS_RUNNING", "STATUS_DONE", "STATUS_FAILED", "STATUS_CANCELLED",
    "BG_EVENTS",
]
