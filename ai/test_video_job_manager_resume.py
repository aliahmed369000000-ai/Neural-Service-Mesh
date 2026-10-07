"""اختبارات استمرارية VideoJobManager عبر توقف مفاجئ للعملية.

قبل هذا: `_jobs` كانت في الذاكرة فقط بالكامل، وfn عشوائية (أي دالة من
ai/video_editor.py) لا يمكن تسلسلها لإعادة بنائها بعد إعادة التشغيل —
هذا بالضبط ما تركها كفجوة موثّقة صراحة في كوميت 74826ee. الحل هنا:
تخزين (job_id, op_name, kwargs) بدل fn نفسها، حيث op_name اسم قابل
لإعادة الاستيراد فقط لدوال من الوحدة الموثوقة (_RESUMABLE_MODULE).

الاختبارات تستخدم دوال top-level حقيقية من *هذا الملف نفسه* بدل
ai/video_editor.py (لا ffmpeg ولا شبكة فعلية)، عبر تبديل
_RESUMABLE_MODULE مؤقتاً لوحدة الاختبار — يختبر الآلية كاملة (تحليل
اسم، استيراد ديناميكي، استئناف) دون الاعتماد على أدوات خارجية.

كل اختبار يستخدم قاعدة بيانات معزولة في tmp_path (وليس الملف الحقيقي
memory/video_jobs.db).
"""
from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai.video_job_manager import VideoJobManager

_THIS_MODULE = "ai.test_video_job_manager_resume"


# دوال top-level حقيقية (وليست closures محلية) — شرط _resolve_op_name
# لجعلها "قابلة لإعادة البناء بالاسم" بعد إعادة الاستيراد.
def fake_trim(path: str, start: float = 0.0, end: float = None) -> str:
    return f"/tmp/trimmed_{path}"


def fake_slow_op(path: str) -> str:
    time.sleep(0.3)
    return f"/tmp/out_{path}"


def _wait_until_not_running(mgr: VideoJobManager, job_id: int, timeout: float = 5.0) -> None:
    """يجب استدعاؤها قبل انتهاء كل اختبار يبدأ مهمة فعلية عبر start() —
    وإلا يبقى الخيط الخلفي يعمل (ويكتب إلى tmp_path) بعد أن يحذفه tearDown،
    فيفشل الحذف بـ'Directory not empty' (سباق لوحظ فعلياً تحت حِمل تشغيل
    المستودع كاملاً، وليس افتراضياً فقط)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = mgr.get(job_id)
        if job is not None and job.status != "running":
            return
        time.sleep(0.02)
    raise AssertionError(f"job #{job_id} ظلّت 'running' بعد {timeout}s — الخيط لم ينتهِ")


class TestVideoJobPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "video_jobs.db"
        self._patch = patch("ai.video_job_manager._RESUMABLE_MODULE", _THIS_MODULE)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def test_job_persisted_to_disk_immediately_with_op_name(self):
        """لقطة أولية تُكتب فور start() — بما فيها op_name القابل لإعادة
        البناء. تتعمّد استخدام fake_slow_op (تنام 0.3 ثانية) لا fake_trim:
        fake_trim شبه فورية، فقد يكتمل الخيط الخلفي (ويكتب status='done')
        قبل أن تصل القراءة المتزامنة أدناه أصلاً — سباق حقيقي رُصد فعلياً
        (الاختبار يفشل أحياناً بـ'done' != 'running' حتى معزولاً عن بقية
        الملف، لا علاقة له بحِمل تشغيل المستودع كاملاً كما تبيّن لاحقاً)."""
        mgr = VideoJobManager(db_path=self.db_path)
        job_id = mgr.start(fake_slow_op, "عملية بطيئة", path="in.mp4")

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status, op_name, kwargs FROM video_jobs WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        self.assertIsNotNone(row, "يجب أن تُحفظ لقطة أولية فور بدء المهمة")
        self.assertEqual(row[0], "running")
        self.assertEqual(row[1], "fake_slow_op")
        self.assertIn("in.mp4", row[2])
        # 🆕 لا بد من انتظار اكتمال الخيط الفعلي قبل tearDown: تحت حِمل
        # تشغيل كامل (كل اختبارات المستودع معاً) قد يتأخر جدولة الخيط بما
        # يكفي ليبقى يكتب إلى قاعدة البيانات داخل tmp_path بعد أن يبدأ
        # tearDown حذف المجلد — فيفشل الحذف بخطأ 'Directory not empty'
        # (سباق كلاسيكي بين خيط خلفي وحذف tmp_path).
        _wait_until_not_running(mgr, job_id)

    def test_unknown_fn_persisted_without_op_name(self):
        """دالة محلية (closure) لا تحمل __module__ الموثوقة → تُحفظ بلا
        op_name (لا تُستأنف تلقائياً لاحقاً، لكن تظل مرئية بحالتها)."""
        mgr = VideoJobManager(db_path=self.db_path)

        def _local_closure(path):
            return path

        job_id = mgr.start(_local_closure, "عملية محلية", path="x.mp4")

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT op_name FROM video_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        self.assertIsNone(row[0])
        _wait_until_not_running(mgr, job_id)  # انظر ملاحظة _wait_until_not_running

    def test_job_persisted_as_done_after_completion(self):
        mgr = VideoJobManager(db_path=self.db_path)
        job_id = mgr.start(fake_trim, "قص", path="in.mp4")
        _wait_until_not_running(mgr, job_id)

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status, result FROM video_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row[0], "done")
        self.assertIn("trimmed_in.mp4", row[1])

    def test_job_id_counter_survives_process_restart(self):
        mgr1 = VideoJobManager(db_path=self.db_path)
        first_id = mgr1.start(fake_trim, "قص", path="a.mp4")
        _wait_until_not_running(mgr1, first_id)

        # "عملية جديدة" مبنية على نفس ملف القاعدة — تحاكي إعادة تشغيل الحاوية.
        mgr2 = VideoJobManager(db_path=self.db_path)
        second_id = mgr2.start(fake_trim, "قص", path="b.mp4")
        _wait_until_not_running(mgr2, second_id)  # نفس سبب الانتظار أعلاه
        self.assertGreater(second_id, first_id)


class TestVideoJobResumeAfterCrash(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "video_jobs.db"
        self._patch = patch("ai.video_job_manager._RESUMABLE_MODULE", _THIS_MODULE)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmp.cleanup()

    def _simulate_crash_leaving_running_job(self, job_id: int, op_name, kwargs: dict) -> None:
        """يحاكي مهمة توقفت فجأة أثناء التنفيذ: تُكتب لقطة 'running' على
        القرص مباشرة (كما يفعل start() قبل انتهاء أي عمل)، دون تشغيل أي
        خيط فعلي — تماماً كأن العملية ماتت قبل أن يُغلق الخيط الحالة."""
        import json
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS video_jobs (
                    job_id INTEGER PRIMARY KEY, label TEXT NOT NULL,
                    status TEXT NOT NULL, op_name TEXT, kwargs TEXT NOT NULL,
                    result TEXT, error TEXT,
                    started_at REAL NOT NULL, finished_at REAL
                )
            """)
            conn.execute(
                "INSERT INTO video_jobs (job_id, label, status, op_name, kwargs, started_at) "
                "VALUES (?, 'مهمة معلَّقة', 'running', ?, ?, ?)",
                (job_id, op_name, json.dumps(kwargs), time.time()),
            )
            conn.commit()

    def test_resume_relaunches_job_with_known_op_name(self):
        self._simulate_crash_leaving_running_job(7, "fake_trim", {"path": "stuck.mp4"})
        mgr = VideoJobManager(db_path=self.db_path)  # "عملية جديدة" بعد التعطّل

        resumed = mgr.resume_interrupted()
        self.assertEqual(resumed, [7])
        _wait_until_not_running(mgr, 7)

        job = mgr.get(7)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.result, "/tmp/trimmed_stuck.mp4")

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM video_jobs WHERE job_id = ?", (7,)
            ).fetchone()
        self.assertEqual(row[0], "done", "الحالة يجب أن تُغلق على القرص بعد الاستئناف")

    def test_resume_closes_unknown_op_name_as_failed_instead_of_stuck_forever(self):
        """مهمة بلا op_name (fn كانت closure محلية أو من وحدة غير موثوقة)
        لا يمكن استئنافها فعلياً — يجب أن تُغلق كفاشلة بدل أن تبقى
        'running' على القرص للأبد."""
        self._simulate_crash_leaving_running_job(9, None, {"path": "x.mp4"})
        mgr = VideoJobManager(db_path=self.db_path)

        resumed = mgr.resume_interrupted()
        self.assertEqual(resumed, [])  # لم تُستأنف فعلياً

        job = mgr.get(9)
        self.assertEqual(job.status, "failed")
        self.assertIn("تعذّر الاستئناف", job.error)

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM video_jobs WHERE job_id = ?", (9,)
            ).fetchone()
        self.assertEqual(row[0], "failed")

    def test_resume_closes_missing_operation_as_failed(self):
        """op_name محفوظ لكن الدالة لم تعد موجودة في الوحدة (إعادة تسمية
        بين إصدارين) — يُغلق كفاشلة بخطأ واضح بدل رفع استثناء يكسر الإقلاع."""
        self._simulate_crash_leaving_running_job(11, "no_such_function_anymore", {})
        mgr = VideoJobManager(db_path=self.db_path)

        resumed = mgr.resume_interrupted()
        self.assertEqual(resumed, [])

        job = mgr.get(11)
        self.assertEqual(job.status, "failed")
        self.assertIn("no_such_function_anymore", job.error)

    def test_resume_is_noop_when_nothing_stuck(self):
        mgr = VideoJobManager(db_path=self.db_path)
        self.assertEqual(mgr.resume_interrupted(), [])

    def test_resume_does_not_touch_already_finished_jobs(self):
        mgr = VideoJobManager(db_path=self.db_path)
        done_id = mgr.start(fake_trim, "قص", path="done.mp4")
        _wait_until_not_running(mgr, done_id)

        resumed = mgr.resume_interrupted()
        self.assertEqual(resumed, [])


if __name__ == "__main__":
    unittest.main()
