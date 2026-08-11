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
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("VideoJobManager")

_JOB_ID_COUNTER = itertools.count(1)

MAX_JOBS = 300  # سقف الاحتفاظ لمنع نمو الذاكرة بلا حدود على عملية طويلة العمر


@dataclass
class VideoJob:
    job_id: int
    label: str                        # اسم العملية للعرض، مثل "قص (trim)"
    status: str = "running"           # running | done | failed
    result: Any = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None


class VideoJobManager:
    """Singleton على مستوى العملية — يشغّل أي دالة من ai/video_editor.py
    في خيط خلفية منفصل لكل استدعاء، ويحتفظ بحالة كل مهمة قابلة للاستعلام
    لاحقاً (running/done/failed + الناتج أو الخطأ) بدل حظر طلب الواجهة."""

    _instance: Optional["VideoJobManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[int, VideoJob] = {}

    @classmethod
    def instance(cls) -> "VideoJobManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self, fn: Callable[..., Any], label: str, **kwargs: Any) -> int:
        """يبدأ تنفيذ fn(**kwargs) في خيط خلفية ويعيد job_id فوراً دون
        انتظار الانتهاء."""
        job_id = next(_JOB_ID_COUNTER)
        job = VideoJob(job_id=job_id, label=label)
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()

        def _run() -> None:
            try:
                result = fn(**kwargs)
                with self._lock:
                    job.result = result
                    job.status = "done"
                    job.finished_at = time.time()
            except Exception as e:  # noqa: BLE001
                logger.exception("فشلت مهمة الفيديو #%s (%s)", job_id, label)
                with self._lock:
                    job.error = str(e)
                    job.status = "failed"
                    job.finished_at = time.time()

        threading.Thread(target=_run, daemon=True,
                          name=f"video-job-{job_id}").start()
        return job_id

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
