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
"""
from __future__ import annotations

import itertools
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ContentJobManager")

_JOB_ID_COUNTER = itertools.count(1)

MAX_JOBS = 300  # سقف الاحتفاظ لمنع نمو الذاكرة بلا حدود على عملية طويلة العمر


@dataclass
class ContentJob:
    job_id: int
    status: str = "running"          # running | done | failed
    result: Any = None
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)


class ContentJobManager:
    """Singleton على مستوى العملية — يشغّل run_content_pipeline في خيط
    خلفية منفصل لكل استدعاء، ويحتفظ بحالة كل مهمة قابلة للاستعلام لاحقاً
    (running/done/failed + النتيجة أو الخطأ)."""

    _instance: Optional["ContentJobManager"] = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._jobs: Dict[int, ContentJob] = {}

    @classmethod
    def instance(cls) -> "ContentJobManager":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def start(self, **pipeline_kwargs: Any) -> int:
        """يبدأ خط أنابيب صناعة المحتوى في خيط خلفية ويعيد job_id فوراً
        دون انتظار الانتهاء."""
        job_id = next(_JOB_ID_COUNTER)
        job = ContentJob(job_id=job_id, kwargs=pipeline_kwargs)
        with self._lock:
            self._jobs[job_id] = job
            self._prune_locked()

        def _run() -> None:
            from ai.content_agent import run_content_pipeline
            try:
                result = run_content_pipeline(**pipeline_kwargs)
                with self._lock:
                    job.result = result
                    job.status = "done"
                    job.finished_at = time.time()
            except Exception as e:  # noqa: BLE001
                logger.exception("فشلت مهمة صناعة المحتوى #%s", job_id)
                with self._lock:
                    job.error = str(e)
                    job.status = "failed"
                    job.finished_at = time.time()

        threading.Thread(target=_run, daemon=True,
                          name=f"content-job-{job_id}").start()
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

    def get(self, job_id: int) -> Optional[ContentJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> List[ContentJob]:
        """أحدث مهمة أولاً."""
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.job_id, reverse=True)


def get_content_job_manager() -> ContentJobManager:
    return ContentJobManager.instance()
