"""
ai/content_job_manager.py
==========================
مدير مهام خلفية لخط أنابيب صناعة المحتوى (ai/content_agent.py).

المشكلة: run_content_pipeline() يستدعي LLM (كتابة مقال) + بحث ويب
(اكتشاف ترند) + نشر/جدولة اختياري — قد يأخذ عشرات الثواني، وكان
يُستدعى بشكل متزامن (synchronous) داخل CategoryAgentChat.chat()
(ai/agent_categories.py)، ما يُجمّد واجهة Streamlit بالكامل حتى
انتهاء الخط بالكامل.

الحل: نفس نمط SocialAgentManager (ai/social_agent.py) — خيط خلفية
(threading.Thread, daemon=True) مستقل لكل مهمة، مع قاموس حالة محمي
بقفل (threading.Lock) بدل حظر الطلب الرئيسي. المستخدم يستمر باستخدام
الواجهة فوراً، ويستعلم عن النتيجة لاحقاً بمعرّف المهمة.

🆕 تسريب ذاكرة طويل المدى: عملية Streamlit Cloud تبقى حيّة لأيام/أسابيع
(reboot نادر)، و`_jobs` كان يكبر بلا حد أعلى — كل مقال يُطلب (حتى لو
فشل أو انتهى منذ أشهر) يبقى في الذاكرة للأبد. الآن يُقلَّم القاموس بعد
كل مهمة جديدة إلى MAX_JOBS، مع حذف المهام المنتهية (done/failed) الأقدم
أولاً، وعدم لمس أي مهمة لا تزال running (نفس منطق task_manager.py).

🆕 استمرارية عبر توقف مفاجئ: `_jobs` كانت في الذاكرة فقط بالكامل — أي
توقف مفاجئ للعملية (crash/redeploy/OOM، شائع على Streamlit Cloud حسب
الملاحظات السابقة في هذا المشروع) أثناء تنفيذ خط الأنابيب في خيط خلفية
كان يفقد المهمة كلياً: لا أثر لها حتى في list_jobs()، ولا أي طريقة
لمعرفة أنها كانت قيد التنفيذ أصلاً. الآن تُحفظ كل مهمة في
memory/content_jobs.db (SQLite) فور بدئها (قبل أي عمل فعلي) وبعد
انتهائها، ويُستأنف تلقائياً أي عمل بقي 'running' من عملية سابقة عبر
resume_interrupted() — تُستدعى من core/mesh_bundle.py عند إقلاع
الحزمة، بنفس kwargs المحفوظة ونفس job_id (استئناف حقيقي، وليس فقط
تسجيل الفشل). هذا ممكن هنا تحديداً (بخلاف ai/video_job_manager.py
المشابه) لأن run_content_pipeline() دالة واحدة معروفة مسبقاً بمُدخلات
JSON بسيطة (topic/platforms)، وليست fn عشوائية يصعب إعادة بنائها بأمان
بعد إعادة التشغيل.
"""
from __future__ import annotations

import itertools
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ContentJobManager")

MAX_JOBS = 300  # سقف الاحتفاظ لمنع نمو الذاكرة/القرص بلا حدود على عملية طويلة العمر
MAX_RESUME_ATTEMPTS = 3  # مهمة تقتل العملية في كل استئناف تُخلّى عنها بدل حلقة انهيار لا تنتهي
_DB_PATH = Path("memory/content_jobs.db")


def _safe_json(value: Any) -> Optional[str]:
    """json.dumps آمن: نتيجة run_content_pipeline() كائن عادي (article/
    ContentResult...) وليس قاموساً بسيطاً — json.dumps() المباشر عليه
    يرفع TypeError. لو فشل الترميز المباشر، يُخزَّن تمثيله النصي (str)
    بدل ذلك، بدل أن يفشل _persist_job_locked بالكامل ويفوّت تحديث status
    نفسها (وهذا بالضبط ما كان يحدث قبل هذا الإصلاح: مهمة اكتملت فعلياً
    تبقى 'running' على القرص للأبد لأن حفظ نتيجتها فشل صامتاً، فيعيد
    resume_interrupted() تشغيلها من جديد بلا داعٍ بعد كل إعادة تشغيل)."""
    if value is None:
        return None
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return json.dumps(str(value), ensure_ascii=False)
        except Exception:
            return None


@dataclass
class ContentJob:
    job_id: int
    status: str = "running"          # running | done | failed
    result: Any = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # عدد مرات الاستئناف بعد توقف مفاجئ — يُحفظ مع المهمة (انظر
    # MAX_RESUME_ATTEMPTS في resume_interrupted).
    resume_attempts: int = 0
    kwargs: Dict[str, Any] = field(default_factory=dict)


class ContentJobManager:
    """Singleton على مستوى العملية — يشغّل run_content_pipeline في خيط
    خلفية منفصل لكل استدعاء، ويحتفظ بحالة كل مهمة قابلة للاستعلام لاحقاً
    (running/done/failed + النتيجة أو الخطأ)."""

    _instance: Optional["ContentJobManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Path = _DB_PATH):
        self._lock = threading.Lock()
        self._jobs: Dict[int, ContentJob] = {}
        self._db_path = Path(db_path).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # عدّاد job_id يبدأ بعد آخر معرّف محفوظ على القرص من عمليات سابقة،
        # وليس من 1 دائماً — وإلا تصطدم مهمة جديدة بمعرّف مهمة قديمة
        # محفوظة في قاعدة البيانات (تصادم job_id عبر إعادة تشغيل الحاوية).
        self._job_id_counter = itertools.count(self._max_persisted_job_id() + 1)

    # ── تخزين دائم (SQLite) ──────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS content_jobs (
                        job_id       INTEGER PRIMARY KEY,
                        status       TEXT NOT NULL,
                        kwargs       TEXT NOT NULL,
                        result       TEXT,
                        error        TEXT,
                        started_at   REAL NOT NULL,
                        finished_at  REAL,
                        resume_attempts INTEGER NOT NULL DEFAULT 0
                    )
                """)
                # ALTER TABLE لقاعدة موجودة من قبل هذا العمود (CREATE TABLE
                # IF NOT EXISTS لا يضيف عموداً لجدول موجود أصلاً).
                try:
                    conn.execute(
                        "ALTER TABLE content_jobs ADD COLUMN resume_attempts "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                except sqlite3.OperationalError:
                    pass  # العمود موجود أصلاً
                conn.commit()
        except Exception as e:
            logger.warning(f"ContentJobManager: تعذّر تهيئة قاعدة بيانات المهام: {e}")

    def _max_persisted_job_id(self) -> int:
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute("SELECT MAX(job_id) FROM content_jobs").fetchone()
                return row[0] or 0
        except Exception:
            return 0

    def _persist_job_locked(self, job: ContentJob) -> None:
        """يُستدعى تحت self._lock فقط. يُدرج أو يحدّث نفس السجل (بنفس
        job_id). لا يرفع استثناء أبداً — التخزين الدائم لا يجب أن يُعطّل
        تنفيذ المهمة."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    INSERT INTO content_jobs
                        (job_id, status, kwargs, result, error, started_at, finished_at, resume_attempts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status, result=excluded.result,
                        error=excluded.error, finished_at=excluded.finished_at,
                        resume_attempts=excluded.resume_attempts
                """, (
                    job.job_id, job.status,
                    _safe_json(job.kwargs) or "{}",
                    _safe_json(job.result),
                    job.error, job.started_at, job.finished_at, job.resume_attempts,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"ContentJobManager: تعذّر حفظ المهمة #{job.job_id}: {e}")

    def _prune_db_locked(self) -> None:
        """نفس منطق _prune_locked لكن على القرص — يحذف أقدم المهام
        المنتهية فقط، ولا يلمس أي مهمة running."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                total = conn.execute("SELECT COUNT(*) FROM content_jobs").fetchone()[0]
                overflow = total - MAX_JOBS
                if overflow <= 0:
                    return
                ids = conn.execute("""
                    SELECT job_id FROM content_jobs WHERE status != 'running'
                    ORDER BY job_id ASC LIMIT ?
                """, (overflow,)).fetchall()
                conn.executemany("DELETE FROM content_jobs WHERE job_id = ?", ids)
                conn.commit()
        except Exception as e:
            logger.warning(f"ContentJobManager: تعذّر تقليم قاعدة بيانات المهام: {e}")

    @classmethod
    def instance(cls) -> "ContentJobManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self, **pipeline_kwargs: Any) -> int:
        """يبدأ خط أنابيب صناعة المحتوى في خيط خلفية ويعيد job_id فوراً
        دون انتظار الانتهاء."""
        job_id = next(self._job_id_counter)
        job = ContentJob(job_id=job_id, kwargs=pipeline_kwargs)
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()
            # 🆕 لقطة أولية على القرص قبل أي عمل فعلي — أساس الاستئناف
            # لو تعطّلت العملية أثناء تنفيذ run_content_pipeline نفسها.
            self._persist_job_locked(job)
            self._prune_db_locked()
        self._launch(job, pipeline_kwargs)
        return job_id

    def _launch(self, job: ContentJob, pipeline_kwargs: Dict[str, Any]) -> None:
        def _run() -> None:
            from ai.content_agent import run_content_pipeline
            try:
                result = run_content_pipeline(**pipeline_kwargs)
                with self._lock:
                    job.result = result
                    job.status = "done"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)
            except Exception as e:  # noqa: BLE001
                logger.exception("فشلت مهمة صناعة المحتوى #%s", job.job_id)
                with self._lock:
                    job.error = str(e)
                    job.status = "failed"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)

        threading.Thread(target=_run, daemon=True,
                          name=f"content-job-{job.job_id}").start()

    def resume_interrupted(self) -> List[int]:
        """يفحص قاعدة البيانات بحثاً عن مهام بقيت 'running' من عملية
        سابقة. بما أن هذا الاستدعاء يحدث دائماً في بداية عمر singleton
        جديد تماماً (عند إقلاع MeshBundle)، أي مهمة 'running' هنا تعني
        حتماً أن العملية التي كانت تنفّذها توقفت فجأة قبل أن تُغلق حالتها
        (done/failed) — لا شيء في هذه العملية الجديدة ينفّذها فعلاً.
        يعيد تشغيل كل مهمة كهذه من جديد (run_content_pipeline(**kwargs)
        بنفس kwargs المحفوظة) في خيط خلفية جديد، بنفس job_id حتى تبقى أي
        إشارة سابقة إليه صالحة. يعيد قائمة معرّفات المهام المُستأنفة."""
        resumed: List[int] = []
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM content_jobs WHERE status = 'running'"
                ).fetchall()
        except Exception as e:
            logger.warning(f"ContentJobManager: تعذّر فحص مهام المحتوى المتوقفة: {e}")
            return resumed

        for row in rows:
            try:
                kwargs = json.loads(row["kwargs"]) if row["kwargs"] else {}
            except Exception:
                kwargs = {}
            prev_attempts = row["resume_attempts"] if "resume_attempts" in row.keys() else 0
            job = ContentJob(
                job_id=row["job_id"], status="running",
                started_at=row["started_at"] or time.time(), kwargs=kwargs,
                resume_attempts=(prev_attempts or 0) + 1,
            )
            with self._lock:
                self._jobs[job.job_id] = job

            if job.resume_attempts > MAX_RESUME_ATTEMPTS:
                # مهمة تقتل العملية في كل مرة تُستأنف — تُخلّى عنها بدل
                # حلقة انهيار لا تنتهي عند كل إقلاع. تبقى في القرص بحالة
                # 'abandoned' للفحص اليدوي، ولا تُدرَج ضمن resumed.
                with self._lock:
                    job.status = "abandoned"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)
                logger.error(
                    "ContentJobManager: المهمة #%s تُخلّي عنها بعد %s محاولات "
                    "استئناف فاشلة", job.job_id, prev_attempts,
                )
                continue

            # يُحفظ العدّاد المرفوع فوراً قبل أي عمل فعلي — يبقى حتى لو
            # ماتت العملية أثناء هذا الاستئناف بالذات.
            with self._lock:
                self._persist_job_locked(job)
            self._launch(job, kwargs)
            resumed.append(job.job_id)
            logger.info("ContentJobManager: استُؤنفت مهمة محتوى متوقفة #%s (محاولة %s)",
                        job.job_id, job.resume_attempts)
        return resumed

    def _prune_locked(self) -> None:
        """يُستدعى تحت self._lock فقط. يحذف أقدم المهام المنتهية
        (done/failed) حتى يعود العدد إلى MAX_JOBS، دون المساس بأي مهمة
        لا تزال running (حتى لو تجاوز العدد الإجمالي السقف مؤقتاً)."""
        overflow = len(self._jobs) - MAX_JOBS
        if overflow <= 0:
            return
        finished_ids = sorted(
            (jid for jid, j in self._jobs.items() if j.status != "running"),
        )
        for jid in finished_ids[:overflow]:
            del self._jobs[jid]

    def get(self, job_id: int) -> Optional[ContentJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[ContentJob]:
        """أحدث مهمة أولاً."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.job_id, reverse=True)


def get_content_job_manager() -> ContentJobManager:
    return ContentJobManager.instance()
