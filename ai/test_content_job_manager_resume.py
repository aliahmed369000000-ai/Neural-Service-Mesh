"""اختبارات استمرارية ContentJobManager عبر توقف مفاجئ للعملية.

قبل هذا: _jobs كانت في الذاكرة فقط بالكامل — أي إعادة تشغيل فجائية
للحاوية أثناء تنفيذ run_content_pipeline() تفقد المهمة كلياً بلا أي أثر.
الآن تُحفظ كل مهمة في SQLite قبل التنفيذ وبعده، وresume_interrupted()
يستأنف أي مهمة بقيت 'running' من عملية سابقة.

كل اختبار يستخدم قاعدة بيانات معزولة في tmp_path (وليس الملف الحقيقي
memory/content_jobs.db) — لا يحتاج مفاتيح API حقيقية.
"""
from __future__ import annotations

import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai.content_job_manager import ContentJobManager


class _FakeResult:
    def __init__(self, tag="ok"):
        self.tag = tag


def _wait_until_not_running(mgr: ContentJobManager, job_id: int, timeout: float = 2.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = mgr.get(job_id)
        if job is not None and job.status != "running":
            return
        time.sleep(0.02)


class TestContentJobPersistence(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "content_jobs.db"

    def tearDown(self):
        self._tmp.cleanup()

    def test_job_persisted_to_disk_immediately_on_start(self):
        """اللقطة الأولية يجب أن تُكتب على القرص فور start()، قبل انتهاء
        خط الأنابيب — هذا هو أساس الاستئناف لاحقاً."""
        mgr = ContentJobManager(db_path=self.db_path)

        def _slow(**kwargs):
            time.sleep(0.3)
            return _FakeResult()

        with patch("ai.content_agent.run_content_pipeline", side_effect=_slow):
            job_id = mgr.start(topic="اختبار الاستمرارية")

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status, kwargs FROM content_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        self.assertIsNotNone(row, "يجب أن تُحفظ لقطة أولية فور بدء المهمة")
        self.assertEqual(row[0], "running")
        self.assertIn("اختبار الاستمرارية", row[1])

    def test_job_persisted_as_done_after_completion(self):
        mgr = ContentJobManager(db_path=self.db_path)
        with patch("ai.content_agent.run_content_pipeline", return_value=_FakeResult("x")):
            job_id = mgr.start(topic="مهمة سريعة")
        _wait_until_not_running(mgr, job_id)

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM content_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row[0], "done")

    def test_job_id_counter_survives_process_restart(self):
        """job_id لا يجوز أن يصطدم بمعرّف قديم محفوظ من 'عملية سابقة' —
        هنا نحاكي ذلك بإنشاء ContentJobManager ثانية على نفس ملف القاعدة."""
        mgr1 = ContentJobManager(db_path=self.db_path)
        with patch("ai.content_agent.run_content_pipeline", return_value=_FakeResult()):
            first_id = mgr1.start(topic="أولى")
        _wait_until_not_running(mgr1, first_id)

        # 🆕 "عملية جديدة" مبنية على نفس ملف القاعدة (بدل الذاكرة المشتركة
        # لـmgr1) — تحاكي إعادة تشغيل الحاوية فعلياً.
        mgr2 = ContentJobManager(db_path=self.db_path)
        with patch("ai.content_agent.run_content_pipeline", return_value=_FakeResult()):
            second_id = mgr2.start(topic="ثانية بعد إعادة التشغيل")
        self.assertGreater(second_id, first_id)


class TestContentJobResumeAfterCrash(unittest.TestCase):
    def setUp(self):
        self._tmp = TemporaryDirectory()
        self.db_path = Path(self._tmp.name) / "content_jobs.db"

    def tearDown(self):
        self._tmp.cleanup()

    def _simulate_crash_leaving_running_job(self, topic: str) -> int:
        """يحاكي مهمة توقفت فجأة أثناء التنفيذ: تُكتب لقطة 'running' على
        القرص مباشرة (كما يفعل start() قبل انتهاء أي عمل)، دون تشغيل أي
        خيط فعلي — تماماً كأن العملية ماتت قبل أن يُغلق الخيط الحالة."""
        import json
        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS content_jobs (
                    job_id INTEGER PRIMARY KEY, status TEXT NOT NULL,
                    kwargs TEXT NOT NULL, result TEXT, error TEXT,
                    started_at REAL NOT NULL, finished_at REAL
                )
            """)
            conn.execute(
                "INSERT INTO content_jobs (job_id, status, kwargs, started_at) "
                "VALUES (?, 'running', ?, ?)",
                (7, json.dumps({"topic": topic}), time.time()),
            )
            conn.commit()
        return 7

    def test_resume_interrupted_relaunches_stuck_job_with_same_id_and_kwargs(self):
        job_id = self._simulate_crash_leaving_running_job("موضوع تعطّل قبل اكتماله")
        mgr = ContentJobManager(db_path=self.db_path)  # "عملية جديدة" بعد التعطّل

        captured = {}

        def _capture(**kwargs):
            captured.update(kwargs)
            return _FakeResult("resumed")

        with patch("ai.content_agent.run_content_pipeline", side_effect=_capture):
            resumed_ids = mgr.resume_interrupted()
            self.assertEqual(resumed_ids, [job_id])
            _wait_until_not_running(mgr, job_id)

        self.assertEqual(captured.get("topic"), "موضوع تعطّل قبل اكتماله")
        job = mgr.get(job_id)
        self.assertEqual(job.status, "done")

        import sqlite3
        with sqlite3.connect(str(self.db_path)) as conn:
            row = conn.execute(
                "SELECT status FROM content_jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
        self.assertEqual(row[0], "done", "الحالة يجب أن تُغلق على القرص بعد الاستئناف")

    def test_resume_interrupted_is_noop_when_nothing_stuck(self):
        mgr = ContentJobManager(db_path=self.db_path)
        self.assertEqual(mgr.resume_interrupted(), [])

    def test_resume_interrupted_does_not_touch_already_finished_jobs(self):
        mgr = ContentJobManager(db_path=self.db_path)
        with patch("ai.content_agent.run_content_pipeline", return_value=_FakeResult()):
            done_id = mgr.start(topic="مهمة اكتملت طبيعياً")
        _wait_until_not_running(mgr, done_id)

        with patch("ai.content_agent.run_content_pipeline") as mocked:
            resumed = mgr.resume_interrupted()

        self.assertEqual(resumed, [])
        mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
