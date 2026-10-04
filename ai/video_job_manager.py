"""
ai/video_job_manager.py
========================
مدير مهام خلفية لعمليات ai/video_editor.py (قص/دمج/رفع دقة/تحسين ذكي...).

المشكلة: كل عمليات محرر الفيديو (ui_pages/video_editor_ui.py) كانت تُستدعى
بشكل متزامن داخل st.spinner — عمليات مثل upscale/ai_enhance/quality_boost
تستدعي ffmpeg (وأحياناً نماذج AI عبر الشبكة) وقد تأخذ من عشرات الثواني حتى
عدة دقائق لفيديو طويل، فتُجمّد واجهة Streamlit بالكامل طوال تلك المدة —
تماماً نفس مشكلة run_content_pipeline التي عولجت سابقاً في
ai/content_job_manager.py.

الحل: نفس نمط SocialAgentManager/ContentJobManager — خيط خلفية
(threading.Thread, daemon=True) مستقل لكل استدعاء، مع قاموس حالة محمي
بقفل. عام (generic) هنا بدل مخصص لدالة واحدة لأن محرر الفيديو له أكثر
من 10 عمليات مختلفة (trim/concat/mute/upscale/...).

🆕 نفس تسريب الذاكرة المُصلَح في ai/content_job_manager.py: `_jobs` كان
يكبر بلا حد أعلى على عملية Streamlit Cloud طويلة العمر. يُقلَّم الآن
إلى MAX_JOBS بعد كل مهمة جديدة، بحذف المهام المنتهية (done/failed)
الأقدم أولاً فقط — لا تُحذف أي مهمة running.

🆕 استمرارية عبر توقف مفاجئ (الفجوة الموثّقة في كوميت 74826ee لدى
ai/content_job_manager.py، متروكة هنا عمداً وقتها): كانت `_jobs` بالكامل
في الذاكرة، ولا يمكن ببساطة تسلسل (serialize) `fn` عشوائية إلى القرص
لإعادة بنائها بعد إعادة تشغيل العملية — هذا بالضبط ما منع تطبيق نفس حل
ContentJobManager مباشرة. الحل هنا: بدل تخزين fn نفسها، تُخزَّن نقطة
تفتيش (job_id, op_name, kwargs) في memory/video_jobs.db فور start()،
حيث op_name اسم قابل لإعادة الاستيراد (getattr) فقط إذا كانت fn دالة
معروفة من ai/video_editor.py (الوحدة الموثوقة الوحيدة هنا — انظر
_resolve_op_name). resume_interrupted() تستأنف فقط المهام ذات op_name
معروف بإعادة استيراد نفس الدالة من ai.video_editor وتنفيذها بنفس
kwargs المحفوظة؛ أي مهمة أخرى (كدوال الاختبار المحلية closures، أو fn
من وحدة غير موثوقة) تبقى غير قابلة للاستئناف فعلياً — تُغلق كفاشلة صراحة
بدل أن تبقى 'running' على القرص للأبد بلا أي شيء ينفّذها (نفس المنطق
الذي صحّح خللاً مشابهاً في content_job_manager: مهمة 'running' منسية
تُضلِّل أي عرض لاحق للحالة).
"""
from __future__ import annotations

import importlib
import itertools
import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("VideoJobManager")

MAX_JOBS = 300  # سقف الاحتفاظ لمنع نمو الذاكرة بلا حدود على عملية طويلة العمر
MAX_RESUME_ATTEMPTS = 3  # مهمة تقتل العملية في كل استئناف تُخلّى عنها بدل حلقة انهيار لا تنتهي
_DB_PATH = Path("memory/video_jobs.db")
_RESUMABLE_MODULE = "ai.video_editor"  # الوحدة الوحيدة التي يُثَق بإعادة استيراد دوالها بالاسم


def _safe_json(value: Any) -> Optional[str]:
    """json.dumps آمن: kwargs عادةً بسيطة (مسارات/أرقام/نصوص) لكن قد
    تحوي Path (غير قابلة للتسلسل افتراضياً) أو نتيجة غير متوقعة. default=str
    يحوّل Path لنص تلقائياً؛ لو فشل حتى هذا، يُخزَّن str(value) كبديل آمن
    بدل تعطيل الحفظ بالكامل (نفس _safe_json في content_job_manager.py)."""
    if value is None:
        return None
    try:
        return json.dumps(value, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        try:
            return json.dumps(str(value), ensure_ascii=False)
        except Exception:
            return None


def _resolve_op_name(fn: Callable[..., Any]) -> Optional[str]:
    """اسم قابل لإعادة البناء بأمان لـfn بعد إعادة تشغيل العملية، أو None
    إن لم يكن ذلك ممكناً. مقصورة عمداً على دوال top-level من ai/video_editor.py
    (الوحدة الوحيدة الموثوقة هنا): دالة اختبار محلية (closure) أو fn من
    وحدة أخرى غير معروفة السلامة لا تُخزَّن كاسم — لا طريقة آمنة لإعادة
    استيرادها لاحقاً بمجرد اسمها."""
    if getattr(fn, "__module__", None) != _RESUMABLE_MODULE:
        return None
    name = getattr(fn, "__qualname__", None)
    if not name or "." in name or name.startswith("_"):
        return None
    return name


def _resolve_op(op_name: str) -> Callable[..., Any]:
    """يعكس _resolve_op_name: يستورد _RESUMABLE_MODULE (ديناميكياً عبر
    importlib بدل استيراد ثابت) ويرجع الدالة المطابقة بالاسم. الاستيراد
    الديناميكي هنا (وليس `import ai.video_editor as ve` مباشرة) يسمح
    للاختبارات بمحاكاة الآلية كاملة عبر تبديل _RESUMABLE_MODULE لوحدة
    اختبار بدل ai/video_editor.py الحقيقية (بدون شبكة/ffmpeg فعلي). يرفع
    AttributeError/ImportError إن لم تعد الدالة أو الوحدة موجودة (نادر:
    إعادة تسمية بين إصدارين) — يُلتقط في resume_interrupted."""
    module = importlib.import_module(_RESUMABLE_MODULE)
    return getattr(module, op_name)


@dataclass
class VideoJob:
    job_id: int
    label: str                        # اسم العملية للعرض، مثل "قص (trim)"
    status: str = "running"           # running | done | failed
    result: Any = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    # عدد مرات الاستئناف بعد توقف مفاجئ — يُحفظ مع المهمة (انظر
    # MAX_RESUME_ATTEMPTS في resume_interrupted).
    resume_attempts: int = 0
    op_name: Optional[str] = None     # اسم الدالة القابل لإعادة البناء، أو None إن لم تكن fn معروفة
    kwargs: Dict[str, Any] = field(default_factory=dict)


class VideoJobManager:
    """Singleton على مستوى العملية — يشغّل أي دالة من ai/video_editor.py
    في خيط خلفية منفصل لكل استدعاء، ويحتفظ بحالة كل مهمة قابلة للاستعلام
    لاحقاً (running/done/failed + الناتج أو الخطأ) بدل حظر طلب الواجهة."""

    _instance: Optional["VideoJobManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self, db_path: Path = _DB_PATH):
        self._lock = threading.Lock()
        self._jobs: Dict[int, VideoJob] = {}
        self._db_path = Path(db_path).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        # عدّاد job_id يبدأ بعد آخر معرّف محفوظ على القرص من عمليات سابقة،
        # وليس من 1 دائماً — وإلا تصطدم مهمة جديدة بمعرّف مهمة قديمة محفوظة
        # (نفس منطق ContentJobManager).
        self._job_id_counter = itertools.count(self._max_persisted_job_id() + 1)

    @classmethod
    def instance(cls) -> "VideoJobManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── تخزين دائم (SQLite) ──────────────────────────────────────────────

    def _init_db(self) -> None:
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS video_jobs (
                        job_id       INTEGER PRIMARY KEY,
                        label        TEXT NOT NULL,
                        status       TEXT NOT NULL,
                        op_name      TEXT,
                        kwargs       TEXT NOT NULL,
                        result       TEXT,
                        error        TEXT,
                        started_at   REAL NOT NULL,
                        finished_at  REAL,
                        resume_attempts INTEGER NOT NULL DEFAULT 0
                    )
                """)
                try:
                    conn.execute(
                        "ALTER TABLE video_jobs ADD COLUMN resume_attempts "
                        "INTEGER NOT NULL DEFAULT 0"
                    )
                except sqlite3.OperationalError:
                    pass  # العمود موجود أصلاً
                conn.commit()
        except Exception as e:
            logger.warning(f"VideoJobManager: تعذّر تهيئة قاعدة بيانات المهام: {e}")

    def _max_persisted_job_id(self) -> int:
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                row = conn.execute("SELECT MAX(job_id) FROM video_jobs").fetchone()
                return row[0] or 0
        except Exception:
            return 0

    def _persist_job_locked(self, job: VideoJob) -> None:
        """يُستدعى تحت self._lock فقط. يُدرج أو يحدّث نفس السجل (بنفس
        job_id). لا يرفع استثناء أبداً — التخزين الدائم لا يجب أن يُعطّل
        تنفيذ المهمة الفعلي."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.execute("""
                    INSERT INTO video_jobs
                        (job_id, label, status, op_name, kwargs, result, error, started_at, finished_at, resume_attempts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(job_id) DO UPDATE SET
                        status=excluded.status, result=excluded.result,
                        error=excluded.error, finished_at=excluded.finished_at,
                        resume_attempts=excluded.resume_attempts
                """, (
                    job.job_id, job.label, job.status, job.op_name,
                    _safe_json(job.kwargs) or "{}",
                    _safe_json(job.result),
                    job.error, job.started_at, job.finished_at, job.resume_attempts,
                ))
                conn.commit()
        except Exception as e:
            logger.warning(f"VideoJobManager: تعذّر حفظ المهمة #{job.job_id}: {e}")

    def _prune_db_locked(self) -> None:
        """نفس منطق _prune_locked لكن على القرص — يحذف أقدم المهام
        المنتهية فقط، ولا يلمس أي مهمة running."""
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                total = conn.execute("SELECT COUNT(*) FROM video_jobs").fetchone()[0]
                overflow = total - MAX_JOBS
                if overflow <= 0:
                    return
                ids = conn.execute("""
                    SELECT job_id FROM video_jobs WHERE status != 'running'
                    ORDER BY job_id ASC LIMIT ?
                """, (overflow,)).fetchall()
                conn.executemany("DELETE FROM video_jobs WHERE job_id = ?", ids)
                conn.commit()
        except Exception as e:
            logger.warning(f"VideoJobManager: تعذّر تقليم قاعدة بيانات المهام: {e}")

    def start(self, fn: Callable[..., Any], label: str, **kwargs: Any) -> int:
        """يبدأ تنفيذ fn(**kwargs) في خيط خلفية ويعيد job_id فوراً دون
        انتظار الانتهاء. لو كانت fn دالة معروفة من ai/video_editor.py،
        تُحفظ لقطة أولية (job_id/op_name/kwargs) على القرص فور البدء —
        أساس resume_interrupted() لو انهارت العملية أثناء التنفيذ. دوال
        أخرى (اختبارات، إغلاقات محلية) تُسجَّل بلا op_name: تظهر بحالتها
        الحقيقية لكن لا تُستأنف تلقائياً."""
        job_id = next(self._job_id_counter)
        op_name = _resolve_op_name(fn)
        job = VideoJob(job_id=job_id, label=label, op_name=op_name, kwargs=kwargs)
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()
            self._persist_job_locked(job)
            self._prune_db_locked()
        self._launch(job, fn, kwargs)
        return job_id

    def _launch(self, job: VideoJob, fn: Callable[..., Any], kwargs: Dict[str, Any]) -> None:
        def _run() -> None:
            try:
                result = fn(**kwargs)
                with self._lock:
                    job.result = result
                    job.status = "done"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)
            except Exception as e:  # noqa: BLE001
                logger.exception("فشلت مهمة الفيديو #%s (%s)", job.job_id, job.label)
                with self._lock:
                    job.error = str(e)
                    job.status = "failed"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)

        threading.Thread(target=_run, daemon=True,
                          name=f"video-job-{job.job_id}").start()

    def resume_interrupted(self) -> List[int]:
        """يفحص قاعدة البيانات بحثاً عن مهام فيديو بقيت 'running' من عملية
        سابقة توقفت فجأة. تُستأنف فقط المهام ذات op_name معروف (دالة من
        ai/video_editor.py أُعيد استيرادها بالاسم بنجاح) بنفس kwargs
        المحفوظة ونفس job_id. أي مهمة أخرى — op_name غائب (fn لم تكن من
        الوحدة الموثوقة أصلاً) أو الدالة لم تعد موجودة (إعادة تسمية) —
        تُغلق صراحةً كفاشلة بدل أن تبقى 'running' على القرص للأبد بلا أي
        شيء ينفّذها فعلياً بعد كل إعادة تشغيل. يرجع قائمة معرّفات المهام
        المُستأنفة فعلياً (المُغلَقة كفاشلة غير مُدرَجة)."""
        resumed: List[int] = []
        try:
            with sqlite3.connect(str(self._db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM video_jobs WHERE status = 'running'"
                ).fetchall()
        except Exception as e:
            logger.warning(f"VideoJobManager: تعذّر فحص مهام الفيديو المتوقفة: {e}")
            return resumed

        for row in rows:
            try:
                kwargs = json.loads(row["kwargs"]) if row["kwargs"] else {}
            except Exception:
                kwargs = {}
            op_name = row["op_name"]
            prev_attempts = row["resume_attempts"] if "resume_attempts" in row.keys() else 0
            job = VideoJob(
                job_id=row["job_id"], label=row["label"] or "مهمة فيديو",
                status="running", started_at=row["started_at"] or time.time(),
                op_name=op_name, kwargs=kwargs,
                resume_attempts=(prev_attempts or 0) + 1,
            )
            with self._lock:
                self._jobs[job.job_id] = job

            if job.resume_attempts > MAX_RESUME_ATTEMPTS:
                # مهمة تقتل العملية في كل مرة تُستأنف (غالباً ffmpeg على
                # مُدخل تالف) — تُخلّى عنها بدل حلقة انهيار لا تنتهي.
                with self._lock:
                    job.status = "abandoned"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)
                logger.error(
                    "VideoJobManager: المهمة #%s تُخلّي عنها بعد %s محاولات "
                    "استئناف فاشلة", job.job_id, prev_attempts,
                )
                continue

            if not op_name:
                with self._lock:
                    job.status = "failed"
                    job.error = "تعذّر الاستئناف: العملية الأصلية لم تُسجَّل باسم قابل لإعادة البناء"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)
                logger.warning(
                    "VideoJobManager: مهمة #%s غير قابلة للاستئناف (fn غير معروفة) — أُغلقت كفاشلة",
                    job.job_id,
                )
                continue

            try:
                fn = _resolve_op(op_name)
            except Exception as e:
                with self._lock:
                    job.status = "failed"
                    job.error = f"تعذّر إيجاد العملية '{op_name}' بعد إعادة التشغيل: {e}"
                    job.finished_at = time.time()
                    self._persist_job_locked(job)
                logger.warning(
                    "VideoJobManager: مهمة #%s: العملية '%s' لم تعد موجودة — أُغلقت كفاشلة",
                    job.job_id, op_name,
                )
                continue

            with self._lock:
                self._persist_job_locked(job)  # عدّاد المحاولات يُحفظ قبل أي عمل فعلي
            self._launch(job, fn, kwargs)
            resumed.append(job.job_id)
            logger.info(
                "VideoJobManager: استُؤنفت مهمة فيديو متوقفة #%s (%s، محاولة %s)",
                job.job_id, op_name, job.resume_attempts,
            )
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

    def get(self, job_id: int) -> Optional[VideoJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self, job_ids: Optional[List[int]] = None) -> List[VideoJob]:
        """أحدث مهمة أولاً. إن مُرِّر job_ids تُقيَّد النتيجة بها (لعرض
        مهام الجلسة الحالية فقط بدل كل مهام العملية)."""
        with self._lock:
            jobs = list(self._jobs.values())
        if job_ids is not None:
            wanted = set(job_ids)
            jobs = [j for j in jobs if j.job_id in wanted]
        return sorted(jobs, key=lambda j: j.job_id, reverse=True)


def get_video_job_manager() -> VideoJobManager:
    return VideoJobManager.instance()
