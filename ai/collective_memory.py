"""
ai/collective_memory.py — الذاكرة الجماعية طويلة الأمد بين الوكلاء
===================================================================
تخزّن **الدروس المستفادة** (Lessons Learned) من كل مهمة ناجحة أو فاشلة
يقوم بها فريق الوكلاء — مهما اختلف الوكيل الذي نفذها — في قاعدة SQLite
دائمة (memory/collective_memory.db)، ثم تسترجعها عند التخطيط لمهام
جديدة فتُحقن ضمن برومبت المدير الموحّد.

لماذاSQLite منفصل عن data/mesh.db؟
- memory_engine يركز على إحصائيات المسارات والعقد (routing/latency)،
  بينما هذه الوحدة تركز على معرفة نوعية نصية (ماذا فعلنا/ما الذي نجح).
- الفصل يسهّل الاختبار والصيانة وعدم تأثر كل طرف بالآخر.

الدورة:
  1) record_task_result(...) بعد كل مهمة — أو digest_from_run(result_dict)
     الذي يستخرج درسًا نصيًا تلقائيًا عبر llm_generate (مع fallback آمن).
  2) events: lesson_learned / lesson_recalled على ناقل الأحداث.
  3) recall(task, domain) يعيد أفضل الدروس ذات الصلة للمهمة الجديدة.

لا يفشل أي استدعاء من API الرئيسي في هذه الوحدة ولا يرفع استثناءً
خارجيًا — كل شيء محمي بـ try/except مع تسجيل تحذير.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── الثوابت ────────────────────────────────────────────────────────────────

DB_PATH = Path("memory/collective_memory.db")
MAX_RECALL = 5                 # أقصى عدد دروس تُحقن في برومبت واحد
MAX_LESSON_CHARS = 2000        # سقف طول الدرس الواحد (حماية من الإطالة)
MAX_LESSONS = 500              # سقف إجمالي الدروس المحفوظة (LRU بالأقدم)
MAX_TASK_HITS = 3              # عدد مرات ظهور الدرس في مهام ناجحة لتعزيزه
QUALITY_THRESHOLD = 0.4        # درس بأدنى جودة يُحذف عند الضغط

MIN_TASK_HITS = 2


# ── الدوال المساعدة ────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_emit(event_type: str, agent_id: str, title: str, status: str,
               detail: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    """إطلاق حدث على ناقل الأحداث دون أي اعتماديات قاسية."""
    try:
        from ai.agent_event_bus import emit_event
        emit_event(event_type, agent_id=agent_id, title=title,
                   status=status, detail=detail, metadata=metadata)
    except Exception as exc:  # pragma: no cover - حماية فقط
        logger.warning("collective_memory: emit_event failed: %s", exc)


def _extract_domain(task: str) -> str:
    """يشتق مجالًا عامًا من نص المهمة لاستخدامه كمفتاح فهرسة."""
    keywords = {
        "برمجة": ["كود", "برمجة", "python", "api", "تطبيق", "دالة", "خطأ برمجي", "برامج"],
        "محتوى": ["محتوى", "مقال", "تقرير", "نص", "كتابة", "ملخص", "ترجمة"],
        "بحث": ["بحث", "معلومات", "تحليل", "دراسة", "اكتشاف", "استكشاف"],
        "تصميم": ["تصميم", "واجهة", "ui", "css", "أشكال", "ألوان", "شعار"],
        "بيانات": ["بيانات", "تحليل", "إحصاء", "جدول", "exce", "بيانات"],
        "مساعدة": ["مساعدة", "سؤال", "استشارة", "اقتراح", "نصيحة"],
    }
    task_lower = task.lower()
    for domain, markers in keywords.items():
        if any(m.lower() in task_lower for m in markers):
            return domain
    return "عام"


# ── الوحدة الرئيسية ────────────────────────────────────────────────────────

class CollectiveMemory:
    """ذاكرة جماعية طويلة الأمد للدروس المستفادة بين الوكلاء."""

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = Path(db_path).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()
        logger.info("CollectiveMemory ready at %s", self.db_path)

    # ── المخطط ────────────────────────────────────────────────────────────

    def _init_db(self) -> None:
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS collective_lessons (
                    lesson_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain        TEXT    NOT NULL DEFAULT 'عام',
                    question_hint TEXT    NOT NULL,
                    lesson        TEXT    NOT NULL,
                    evidence      TEXT    NOT NULL DEFAULT '',
                    source_agent  TEXT    NOT NULL DEFAULT '',
                    source_run_id TEXT    NOT NULL DEFAULT '',
                    quality       REAL    NOT NULL DEFAULT 0.0,
                    task_hits     INTEGER NOT NULL DEFAULT 1,
                    task_fails    INTEGER NOT NULL DEFAULT 0,
                    created_at    TEXT    NOT NULL,
                    seen_at       TEXT    NOT NULL
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cm_domain ON collective_lessons(domain)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cm_quality ON collective_lessons(quality DESC)"
            )
            conn.commit()

    # ── الكتابة ────────────────────────────────────────────────────────────

    def record_lesson(self, domain: str, question_hint: str, lesson: str,
                      evidence: str = "", source_agent: str = "",
                      source_run_id: str = "", quality: float = 0.0) -> Optional[int]:
        """تسجيل درس جديد يدويًا أو تلقائيًا."""
        if not lesson or not lesson.strip():
            return None
        lesson = lesson.strip()[:MAX_LESSON_CHARS]
        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                cursor = conn.execute("""
                    INSERT INTO collective_lessons
                        (domain, question_hint, lesson, evidence, source_agent,
                         source_run_id, quality, task_hits, task_fails,
                         created_at, seen_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 1, 0, ?, ?)
                """, (domain or "عام", question_hint or "", lesson,
                      evidence[:1000], source_agent or "", source_run_id or "",
                      float(min(1.0, max(-1.0, quality))), _now(), _now()))
                conn.commit()
                lesson_id = cursor.lastrowid
            _safe_emit(
                "lesson_learned",
                agent_id=source_agent or "collective_memory",
                title="الذاكرة الجماعية",
                status="done",
                detail=f"درس جديد: {lesson[:120]}",
                metadata={"domain": domain, "quality": quality,
                          "source_agent": source_agent, "lesson_id": lesson_id},
            )
            logger.info("CollectiveMemory: درس جديد #%s (%s)", lesson_id, domain)
            return lesson_id
        except Exception as exc:  # never break the caller
            logger.warning("collective_memory.record_lesson: %s", exc)
            return None

    def record_task_result(self, task: str, success: bool,
                           duration_ms: float, agent_id: str = "",
                           run_id: str = "", output_hint: str = "") -> None:
        """
        تسجيل بسيط من نقطة التنفيذ: درس إحصائي (نجاح/فشل حسب المجال).
        يرفع quality تدريجيًا عند النجاح المتكرر، ويخفضها عند الفشل.
        """
        try:
            domain = _extract_domain(task)
            hint = output_hint or task[:200]
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT lesson_id, quality, task_hits, task_fails, seen_at "
                    "FROM collective_lessons WHERE question_hint = ? LIMIT 1",
                    (hint,),
                ).fetchone()
                now = _now()
                if row:
                    lid, q, hits, fails, seen = row
                    if success:
                        new_hits = hits + 1
                        new_q = min(1.0, q + 0.15)
                    else:
                        new_hits = hits
                        new_fails = fails + 1
                        new_q = max(-1.0, q - 0.2)
                    conn.execute("""
                        UPDATE collective_lessons
                        SET quality=?, task_hits=?, task_fails=?, seen_at=?
                        WHERE lesson_id=?
                    """, (new_q, new_hits, new_fails, now, lid))
                else:
                    lesson = (
                        "مسار ناجح سابق: «%s» (%.1fms)" % (task[:100], duration_ms)
                        if success else
                        "مسار فاشل سابق: «%s» — تجنبه أو جرّب بديلاً" % task[:100]
                    )
                    q = 0.5 if success else -0.5
                    conn.execute("""
                        INSERT INTO collective_lessons
                            (domain, question_hint, lesson, evidence, source_agent,
                             source_run_id, quality, task_hits, task_fails,
                             created_at, seen_at)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """, (domain, hint, lesson, "", agent_id or "", run_id or "",
                          q, 1 if success else 0, 0 if success else 1, now, now))
                conn.commit()
            status = "done" if success else "fail"
            _safe_emit(
                "lesson_learned",
                agent_id=agent_id or "collective_memory",
                title="الذاكرة الجماعية",
                status=status,
                detail=("درس نجاح" if success else "درس فشل") + f" في مجال {domain}",
                metadata={"domain": domain, "success": success,
                          "duration_ms": duration_ms},
            )
        except Exception as exc:
            logger.warning("collective_memory.record_task_result: %s", exc)

    # ── الاسترجاع ──────────────────────────────────────────────────────────

    def recall(self, task: str, domain: Optional[str] = None,
               top_k: int = MAX_RECALL) -> List[Dict[str, Any]]:
        """استرجاع أفضل الدروس ذات الصلة بالمهمة الجديدة."""
        domain = domain or _extract_domain(task)
        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                rows = conn.execute("""
                    SELECT lesson_id, domain, question_hint, lesson, evidence,
                           source_agent, quality, task_hits, task_fails, seen_at
                    FROM collective_lessons
                    WHERE domain = ? OR domain = 'عام'
                    ORDER BY quality DESC, task_hits DESC, seen_at DESC
                    LIMIT ?
                """, (domain, top_k * 2)).fetchall()
        except Exception as exc:
            logger.warning("collective_memory.recall: %s", exc)
            return []

        result = []
        seen_ids = set()
        for row in rows:
            if len(result) >= top_k:
                break
            lid, d, hint, lesson, ev, agent, q, hits, fails, seen = row
            if lid in seen_ids:
                continue
            # فلتر الجودة: نعرض فقط الدروس ذات الصلة (نقاط إيجابية)
            if (q + 0.05 * hits - 0.1 * fails) < 0.0:
                continue
            seen_ids.add(lid)
            result.append({
                "lesson_id": lid, "domain": d, "question_hint": hint,
                "lesson": lesson, "evidence": ev, "source_agent": agent,
                "quality": q, "task_hits": hits, "task_fails": fails,
                "seen_at": seen,
            })
        return result[:top_k]

    def vote(self, lesson_id: int, delta: int) -> bool:
        """تصويت من المستخدم/المنسق على فائدة درس (+1/-1)."""
        if delta not in (-1, 1):
            return False
        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                row = conn.execute(
                    "SELECT quality, task_hits, task_fails FROM collective_lessons "
                    "WHERE lesson_id=?", (lesson_id,)).fetchone()
                if not row:
                    return False
                q, hits, fails = row
                new_q = min(1.0, max(-1.0, q + 0.2 * delta))
                conn.execute(
                    "UPDATE collective_lessons SET quality=? WHERE lesson_id=?",
                    (new_q, lesson_id))
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("collective_memory.vote: %s", exc)
            return False

    # ── الصيانة ────────────────────────────────────────────────────────────

    def _prune(self) -> None:
        """حذف أقدم الدروس ذات الجودة المنخفضة عند تجاوز السقف."""
        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM collective_lessons").fetchone()[0]
                if total > MAX_LESSONS:
                    conn.execute(
                        "DELETE FROM collective_lessons "
                        "WHERE quality < ? "
                        "ORDER BY created_at ASC "
                        "LIMIT ?", (QUALITY_THRESHOLD, total - MAX_LESSONS))
                    conn.commit()
        except Exception as exc:
            logger.warning("collective_memory._prune: %s", exc)

    # ── الاستخراج التلقائي من نتائج المهام ────────────────────────────────

    def digest_from_run(self, run_dict: Dict[str, Any],
                        llm_generate: Optional[Callable[..., Any]] = None) -> int:
        """
        يستخرج درسًا نصيًا من نتيجة مهمة واحدة (ناجحة أو فاشلة) عبر
        llm_generate. عند عدم توفره أو فشله يستخدم fallback إحصائيًا آمنًا.
        يُرجع lesson_id أو -1 عند التعذر.
        """
        goal = (run_dict.get("goal") or run_dict.get("task") or "")[:300]
        status = run_dict.get("status") or ""
        success = status in ("done", "success")
        domain = _extract_domain(goal or status or "عام")
        result_text = ""
        try:
            tasks = run_dict.get("tasks") or []
            for t in tasks:
                if t.get("status") == "done":
                    res = t.get("result") or {}
                    result_text = (res.get("result_text") or "")[:800]
                    if result_text:
                        break
        except Exception:
            result_text = ""

        lesson = ""
        evidence = ""
        quality = 0.5 if success else -0.5
        if llm_generate is not None:
            try:
                if success:
                    prompt = (
                        "استخرج من نتيجة المهمة التالية درسًا واحدًا عمليًا "
                        "قصيرًا (جملة أو جملتان بالعربية) يمكن لوكيل آخر "
                        "الاستفادة منه في مهام مماثلة مستقبلًا. لا تختلق "
                        "معلومات غير موجودة في النص.\n\n"
                        f"المهمة: {goal}\n\nالنتيجة:\n{result_text or '(لا يوجد نص نتيجة)'}"
                    )
                else:
                    prompt = (
                        "استخرج من فشل المهمة التالية درسًا تحذيريًا واحدًا "
                        "قصيرًا (جملة أو جملتان بالعربية) لما يجب تجنبه "
                        "أو تجريبه بديلاً.\n\n"
                        f"المهمة: {goal}\n\nالنتيجة:\n{result_text or '(لا يوجد نص نتيجة)'}"
                    )
                resp = llm_generate(prompt)
                text = resp.text if hasattr(resp, "text") else str(resp or "")
                lesson = (text or "").strip()[:MAX_LESSON_CHARS]
                evidence = f"run:{(run_dict.get('run_id') or '')[:40]} goal:{goal[:120]}"[:500]
            except Exception as exc:
                logger.warning("collective_memory.digest_from_run llm failed: %s", exc)
                lesson = ""

        if not lesson:
            # fallback إحصائي: لا نختلق نصًا — درس عام بسيط
            lesson = (
                "مسار ناجح موثق: «%s» — قابل لإعادة الاستخدام." % goal[:150]
                if success else
                "مسار فاشل موثق: «%s» — راجع السجل قبل إعادة المحاولة." % goal[:150]
            )
            evidence = f"auto-fallback:{(run_dict.get('run_id') or '')[:40]}".strip(":")

        source_agent = ""
        source_run_id = ""
        try:
            tasks = run_dict.get("tasks") or []
            for t in tasks:
                if t.get("status") == "done":
                    source_agent = t.get("assigned_agent_id") or t.get("agent_id") or ""
                    source_run_id = t.get("task_id") or ""
                    break
        except Exception:
            pass

        lid = self.record_lesson(
            domain=domain, question_hint=goal, lesson=lesson,
            evidence=evidence, source_agent=source_agent,
            source_run_id=source_run_id, quality=quality,
        )
        self._prune()
        return lid or -1

    # ── التقرير ────────────────────────────────────────────────────────────

    def lessons_list(self, limit: int = 50) -> List[Dict[str, Any]]:
        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                cur = conn.execute("""
                    SELECT lesson_id, domain, question_hint, lesson,
                           source_agent, quality, task_hits, task_fails,
                           created_at, seen_at
                    FROM collective_lessons
                    ORDER BY quality DESC, task_hits DESC
                    LIMIT ?
                """, (limit,))
                cols = [d[0] for d in cur.description]
                rows = [row for row in cur]
        except Exception as exc:
            logger.warning("collective_memory.lessons_list: %s", exc)
            return []
        return [dict(zip(cols, row)) for row in rows]

    def summary(self) -> Dict[str, Any]:
        try:
            with self._lock, sqlite3.connect(str(self.db_path)) as conn:
                total = conn.execute(
                    "SELECT COUNT(*) FROM collective_lessons").fetchone()[0]
                domains = conn.execute(
                    "SELECT domain, COUNT(*) FROM collective_lessons "
                    "GROUP BY domain ORDER BY COUNT(*) DESC").fetchall()
                top = conn.execute("""
                    SELECT domain, lesson FROM collective_lessons
                    ORDER BY quality DESC, task_hits DESC LIMIT 3
                """).fetchall()
        except Exception as exc:
            logger.warning("collective_memory.summary: %s", exc)
            return {"total_lessons": 0, "domains": {}, "top_lessons": [],
                    "db_path": str(self.db_path)}
        return {
            "total_lessons": total,
            "domains": {d: c for d, c in domains},
            "top_lessons": [{"domain": d, "lesson": l[:120]} for d, l in top],
            "db_path": str(self.db_path),
        }

    def to_dict(self) -> Dict[str, Any]:
        return self.summary()


# ── Singleton ──────────────────────────────────────────────────────────────

_default: Optional[CollectiveMemory] = None


def get_collective_memory(db_path: Path = DB_PATH) -> CollectiveMemory:
    global _default
    if _default is None:
        _default = CollectiveMemory(db_path)
    return _default
