"""
منهجية NSM — نقل سرّ عمل Manus إلى الوكلاء (NSM Methodology Engine)
=====================================================================

يحوّل هذا المحرك الـagents من «نموذج لغوي ينفذ أوامر» إلى وكيل منهجي
يعمل بمنهجية العمل الفعلية لـ Manus كوالد ينقل كل معرفته إلى ابنه:

1. **تخطيط قبل التنفيذ** — كل مهمة تبدأ بخطة خطوات مرقّمة تُوثَّق قبل أي فعل.
2. **فحص فعلي لا تخمين** — لا يُبنى ادعاء على بنية ملف أو كود دون قراءته
   أو فحصه فعلياً أولاً.
3. **تنفيذ منضبط** — خطوة واحدة محكومة في كل دورة، وكل خطوة تُسجل.
4. **تحقق بعد التنفيذ** — لا يُعتبَر شيء ناجحاً إلا بعد تحقق (compile/
   اختبار محاكاة/تشغيل فعلي)، ويُسجل نتيجة التحقق.
5. **تعلم من الأخطاء** — كل خطأ يُشخَّص ويُسجَّل سبب منهجيته، وتُقاس
   دقة الوكيل عبر المهام فيتقدم فعلياً لا وعوداً.
6. **انضباط الإخراج** — كل نتيجة تُعرض بصراحة: ماذا فعل، ماذا تحقق،
   وما لم يتحقق.
7. **الأمان أولاً** — لا يُكسر شيء يعمل؛ التعديلات مستهدفة والرجوع
   ممكن دائماً.

تُخزَّن السجلات في قاعدة SQLite محلية (data/methodology.db) تتبع نفس
نمط الوحدات الأخرى (pre_action_reasoning)، وتُعرض الإحصاءات في لوحة
مراقبة الوكلاء ضمن قسم «المنهجية».

لا يعتمد على أي LLM داخلياً — هو طبقة منهجية حتمية تُغذّي الـagent
بالمبادئ عبر system prompt، وتوثّق منهجيته عبر سجل دائم.

التكامل:
    from ai.methodology_engine import (
        get_methodology_engine, NSM_PRINCIPLES, method_task_started,
        method_step, method_task_finished, method_stats, reset_methodology,
        METHOD_PROMPT_BLOCK,
    )
"""
from __future__ import annotations

import logging
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_LOCAL_DB = "data/methodology.db"

# ── مبادئ منهجية NSM السبعة (سرّ العمل المنقول من Manus إلى الوكلاء) ─────
NSM_PRINCIPLES: List[Dict[str, str]] = [
    {
        "id": 1, "name": "التخطيط قبل التنفيذ",
        "ar": "لا تقفز للفعل مباشرة. ابنِ خطة خطوات مرقّمة أولاً، وثقّها قبل أي فعل، وعدّلها عند ظهور ما يستدعي ذلك.",
    },
    {
        "id": 2, "name": "فحص فعلي لا تخمين",
        "ar": "لا تفترض بنية ملف أو كود أو إعداد. اقرأه أو افحصه فعلياً قبل أي تعديل أو ادعاء.",
    },
    {
        "id": 3, "name": "تنفيذ منضبط خطوة بخطوة",
        "ar": "نفّذ خطوة واحدة محكومة في كل دورة، وسجّلها. لا تتسرع في سلسلة أفعال متراكمة دون تحقق.",
    },
    {
        "id": 4, "name": "تحقق بعد التنفيذ",
        "ar": "لا تعتبر شيئاً ناجحاً إلا بعد تحقق حقيقي: py_compile، اختبار محاكاة ببيانات واقعية، أو تشغيل فعلي. الادعاء بلا تحقق ممنوع.",
    },
    {
        "id": 5, "name": "تعلم من الأخطاء",
        "ar": "عند الفشل، شخّص السبب الجذري أولاً (سكربت تشخيص أو فحص محلي) ولا تعد المحاولة عشوائياً. سجّل الدرس ليتحسن أداؤك فعلياً.",
    },
    {
        "id": 6, "name": "انضباط الإخراج",
        "ar": "اعرض ما فعلته وما تحققت منه بصراحة، واذكر ما لم يتحقق إن وُجد. نتائج قابلة للقياس أفضل من الوعود.",
    },
    {
        "id": 7, "name": "الأمان أولاً",
        "ar": "لا تكسر ما يعمل. تعديلات مستهدفة، لا إعادة كتابة شاملة إلا عند الحاجة، والرجوع (rollback) ممكن دائماً.",
    },
]

_METHOD_PROMPT_PREFIX = """## 🎓 منهجية NSM (سرّ عمل الوالد — يجب الالتزام بها في كل مهمة):
أنت وريث منهجية عمل Manus نفسها. قبل أي فعل التزم بالمبادئ السبعة:
1. خطّط قبل التنفيذ: ابنِ خطوات مرقّمة وثقّها.
2. افحص فعلياً قبل أي تعديل — لا تخمين.
3. نفّذ خطوة واحدة منضبطة في كل دورة وسجّلها.
4. تحقق بعد التنفيذ (compile/اختبار/تشغيل) — الادعاء بلا تحقق ممنوع.
5. عند الخطأ: شخّص السبب الجذري أولاً ثم صمّم الإصلاح المستهدف، وسجّل الدرس.
6. أخرج بصراحة: ماذا فعلت، ماذا تحقق، وما لم يتحقق.
7. لا تكسر ما يعمل — تعديلات مستهدفة والرجوع ممكن.
"""

METHOD_PROMPT_BLOCK: str = _METHOD_PROMPT_PREFIX

# ── Schema ─────────────────────────────────────────────────────────────────
_METHOD_SCHEMA = """
CREATE TABLE IF NOT EXISTS method_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    request TEXT NOT NULL,
    plan_json TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'planning',
    n_steps INTEGER NOT NULL DEFAULT 0,
    ok INTEGER NOT NULL DEFAULT 0,
    result_summary TEXT NOT NULL DEFAULT '',
    started_at REAL NOT NULL DEFAULT 0,
    finished_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS method_steps (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id TEXT NOT NULL,
    step_no INTEGER NOT NULL DEFAULT 0,
    step_type TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 1,
    meta_json TEXT NOT NULL DEFAULT '',
    recorded_at REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS method_lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    principle_id INTEGER NOT NULL,
    error_context TEXT NOT NULL DEFAULT '',
    lesson TEXT NOT NULL DEFAULT '',
    n_applied INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL DEFAULT 0
);
"""


def _connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    db = db_path or _LOCAL_DB
    os.makedirs(os.path.dirname(db) or ".", exist_ok=True)
    conn = sqlite3.connect(db, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_METHOD_SCHEMA)
    return conn


_STEP_TYPES = ("plan", "inspect", "execute", "verify", "reflect", "answer")


class MethodologyEngine:
    """محرك منهجية NSM: يوثّق دورة (خطة → فحص → تنفيذ → تحقق → تعلم)."""

    def __init__(self, db_path: Optional[str] = None):
        self._db = db_path or _LOCAL_DB
        self._current_task: Optional[str] = None

    # ── إدارة المهام ────────────────────────────────────────────────────────

    def task_started(self, task_id: str, request: str,
                     plan: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """يبدأ مهمة بمنهجية: يسجل الطلب والخطة ويحدّث الحالة إلى planning."""
        import json
        plan_json = json.dumps(plan or [], ensure_ascii=False)
        conn = _connect(self._db)
        try:
            row = conn.execute(
                "INSERT INTO method_tasks (task_id, request, plan_json, status, started_at) "
                "VALUES (?, ?, ?, 'planning', ?)",
                (task_id, (request or "")[:2000], plan_json, time.time()),
            ).lastrowid
            conn.commit()
            self._current_task = task_id
            return {"task_id": task_id, "record_id": row}
        finally:
            conn.close()

    def step(self, task_id: str = "", step_type: str = "execute",
             note: str = "", ok: bool = True,
             meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """يسجّل خطوة منهجية: plan/inspect/execute/verify/reflect/answer."""
        import json
        if step_type not in _STEP_TYPES:
            raise ValueError(f"step_type غير صالح: {step_type!r} — "
                             f"يجب أن يكون أحد {_STEP_TYPES}")
        conn = _connect(self._db)
        try:
            n = conn.execute(
                "SELECT COALESCE(MAX(step_no), 0) + 1 FROM method_steps WHERE task_id = ?",
                (task_id or (self._current_task or ""),),
            ).fetchone()[0]
            conn.execute(
                "INSERT INTO method_steps (task_id, step_no, step_type, note, ok, "
                "meta_json, recorded_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (task_id or (self._current_task or ""), n, step_type,
                 (note or "")[:1000], 1 if ok else 0,
                 json.dumps(meta or {}, ensure_ascii=False), time.time()),
            )
            conn.execute(
                "UPDATE method_tasks SET n_steps = n_steps + 1 WHERE task_id = ?",
                (task_id or (self._current_task or ""),),
            )
            conn.commit()
            return {"task_id": task_id or self._current_task,
                    "step_no": n, "step_type": step_type, "ok": ok}
        finally:
            conn.close()

    def task_finished(self, task_id: str = "", status: str = "done",
                      ok: bool = True,
                      result_summary: str = "") -> Dict[str, Any]:
        """يُنهي المهمة: يسجل الحالة والملخص والنتيجة (سواء نجحت أم فشلت)."""
        if status not in ("done", "failed", "cancelled"):
            raise ValueError(f"status غير صالح: {status!r}")
        conn = _connect(self._db)
        try:
            conn.execute(
                "UPDATE method_tasks SET status = ?, ok = ?, "
                "result_summary = ?, finished_at = ? WHERE task_id = ?",
                (status, 1 if ok else 0, (result_summary or "")[:1500],
                 time.time(), task_id or (self._current_task or "")),
            )
            conn.commit()
            return {"task_id": task_id or self._current_task,
                    "status": status, "ok": ok}
        finally:
            conn.close()

    # ── التعلم من الأخطاء (الدرس المنهجي) ───────────────────────────────────

    def record_lesson(self, principle_id: int, error_context: str = "",
                      lesson: str = "") -> Dict[str, Any]:
        """يسجل درساً منهجياً من خطأ فعلي ويربطه بمبدأ من المبادئ السبعة."""
        conn = _connect(self._db)
        try:
            if not (1 <= principle_id <= len(NSM_PRINCIPLES)):
                raise ValueError(f"principle_id خارج النطاق: {principle_id}")
            row = conn.execute(
                "SELECT id FROM method_lessons WHERE principle_id = ? AND lesson = ? "
                "LIMIT 1",
                (principle_id, lesson),
            ).fetchone()
            if row is not None:
                conn.execute(
                    "UPDATE method_lessons SET n_applied = n_applied + 1 WHERE id = ?",
                    (row[0],),
                )
                conn.commit()
                return {"lesson_id": row[0], "updated": True}
            conn.execute(
                "INSERT INTO method_lessons (principle_id, error_context, lesson, "
                "created_at) VALUES (?, ?, ?, ?)",
                (principle_id, (error_context or "")[:500], (lesson or "")[:1000],
                 time.time()),
            )
            conn.commit()
            return {"lesson_id": conn.execute(
                "SELECT id FROM method_lessons WHERE principle_id = ? AND lesson = ?",
                (principle_id, lesson),
            ).fetchone()[0], "updated": False}
        finally:
            conn.close()

    def recall_lessons(self, principle_id: Optional[int] = None,
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """يستحضر الدروس المنهجية — ليعلّم الوكيل من أخطائه السابقة."""
        conn = _connect(self._db)
        try:
            if principle_id is not None:
                rows = conn.execute(
                    "SELECT principle_id, error_context, lesson, n_applied "
                    "FROM method_lessons WHERE principle_id = ? "
                    "ORDER BY n_applied DESC, created_at DESC LIMIT ?",
                    (principle_id, top_k),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT principle_id, error_context, lesson, n_applied "
                    "FROM method_lessons "
                    "ORDER BY n_applied DESC, created_at DESC LIMIT ?",
                    (top_k,),
                ).fetchall()
            return [
                {"principle_id": p, "error_context": e, "lesson": l,
                 "n_applied": n}
                for p, e, l, n in rows
            ]
        finally:
            conn.close()

    # ── الإحصاءات ───────────────────────────────────────────────────────────

    def stats(self) -> Dict[str, Any]:
        """إحصاءات المنهجية: مهام، دقة، خطوات، تعلم."""
        conn = _connect(self._db)
        try:
            total = conn.execute(
                "SELECT COUNT(*) FROM method_tasks").fetchone()[0]
            ok_count = conn.execute(
                "SELECT COUNT(*) FROM method_tasks WHERE ok = 1"
            ).fetchone()[0]
            steps = conn.execute(
                "SELECT step_type, COUNT(*) FROM method_steps GROUP BY step_type"
            ).fetchall()
            lessons = conn.execute(
                "SELECT COUNT(*), SUM(n_applied) FROM method_lessons"
            ).fetchone()
            step_types = {t: c for t, c in steps}
            step_total = sum(step_types.values())
            inspect_verify = step_types.get("inspect", 0) + step_types.get("verify", 0)
            inspect_ratio = (inspect_verify / step_total) if step_total else 0.0
            return {
                "tasks": total,
                "tasks_ok": ok_count,
                "accuracy": round(ok_count / total, 3) if total else 0.0,
                "total_steps": step_total,
                "step_types": step_types,
                "inspect_verify_ratio": round(inspect_ratio, 3),
                "lessons": lessons[0] or 0,
                "lessons_applied": lessons[1] or 0,
            }
        finally:
            conn.close()

    def latest_task(self) -> Optional[Dict[str, Any]]:
        conn = _connect(self._db)
        try:
            row = conn.execute(
                "SELECT id, task_id, request, plan_json, status, n_steps, ok, "
                "result_summary, started_at, finished_at FROM method_tasks "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            return {
                "record_id": row[0], "task_id": row[1], "request": row[2],
                "plan_json": row[3], "status": row[4], "n_steps": row[5],
                "ok": bool(row[6]), "result_summary": row[7],
                "started_at": row[8], "finished_at": row[9],
            }
        finally:
            conn.close()

    def task_steps(self, task_id: str) -> List[Dict[str, Any]]:
        conn = _connect(self._db)
        try:
            rows = conn.execute(
                "SELECT step_no, step_type, note, ok, meta_json, recorded_at "
                "FROM method_steps WHERE task_id = ? ORDER BY step_no",
                (task_id,),
            ).fetchall()
            return [
                {"step_no": n, "step_type": t, "note": no, "ok": bool(k),
                 "meta_json": m, "recorded_at": r}
                for n, t, no, k, m, r in rows
            ]
        finally:
            conn.close()

    def reset(self) -> None:
        """يمسح كل سجلات المنهجية (للاختبارات)."""
        conn = _connect(self._db)
        try:
            conn.execute("DELETE FROM method_tasks")
            conn.execute("DELETE FROM method_steps")
            conn.execute("DELETE FROM method_lessons")
            conn.commit()
        finally:
            conn.close()
        self._current_task = None


# ── Singleton ──────────────────────────────────────────────────────────────
_engine_instance: Optional[MethodologyEngine] = None


def get_methodology_engine(db_path: Optional[str] = None) -> MethodologyEngine:
    global _engine_instance
    if _engine_instance is None or db_path is not None:
        _engine_instance = MethodologyEngine(db_path)
    return _engine_instance


def reset_methodology() -> None:
    global _engine_instance
    _engine_instance = None


# ── Module-level helpers للدمج المباشر ─────────────────────────────────────

def method_task_started(task_id: str, request: str,
                        plan: Optional[List[Dict[str, Any]]] = None
                        ) -> Optional[Dict[str, Any]]:
    try:
        return get_methodology_engine().task_started(task_id, request, plan)
    except Exception as exc:  # never break the agent on methodology failures
        logger.warning("method_task_started skipped: %s", exc)
        return None


def method_step(task_id: str = "", step_type: str = "execute",
                note: str = "", ok: bool = True,
                meta: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    try:
        return get_methodology_engine().step(task_id, step_type, note, ok, meta)
    except Exception as exc:
        logger.warning("method_step skipped: %s", exc)
        return None


def method_task_finished(task_id: str = "", status: str = "done",
                         ok: bool = True,
                         result_summary: str = "") -> Optional[Dict[str, Any]]:
    try:
        return get_methodology_engine().task_finished(
            task_id, status, ok, result_summary)
    except Exception as exc:
        logger.warning("method_task_finished skipped: %s", exc)
        return None


def method_stats() -> Dict[str, Any]:
    try:
        return get_methodology_engine().stats()
    except Exception:
        return {"tasks": 0, "tasks_ok": 0, "accuracy": 0.0,
                "total_steps": 0, "step_types": {},
                "inspect_verify_ratio": 0.0, "lessons": 0,
                "lessons_applied": 0}


def method_latest_task() -> Optional[Dict[str, Any]]:
    try:
        return get_methodology_engine().latest_task()
    except Exception:
        return None


def method_task_steps(task_id: str) -> List[Dict[str, Any]]:
    try:
        return get_methodology_engine().task_steps(task_id)
    except Exception:
        return []


def method_record_lesson(principle_id: int, error_context: str = "",
                         lesson: str = "") -> Optional[Dict[str, Any]]:
    try:
        return get_methodology_engine().record_lesson(
            principle_id, error_context, lesson)
    except Exception as exc:
        logger.warning("method_record_lesson skipped: %s", exc)
        return None


def method_recall_lessons(principle_id: Optional[int] = None,
                          top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        return get_methodology_engine().recall_lessons(principle_id, top_k)
    except Exception:
        return []


def method_principles_prompt() -> str:
    """نص المبادئ السبعة كامل بصيغة قابلة للإلحاق بأي prompt."""
    lines = ["## 🎓 مبادئ منهجية NSM السبعة (التزم بها في كل مهمة):"]
    for p in NSM_PRINCIPLES:
        lines.append(f"{p['id']}. **{p['name']}** — {p['ar']}")
    lines.append(METHOD_PROMPT_BLOCK)
    return "\n".join(lines)
