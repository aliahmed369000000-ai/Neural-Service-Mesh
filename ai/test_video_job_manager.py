"""اختبارات: عمليات محرر الفيديو تشتغل بالخلفية ولا تجمّد الاستدعاء
المباشر (start يرجع فوراً حتى لو ffmpeg بطيء).
يُشغَّل عبر: python -m unittest ai.test_video_job_manager -v
"""
from __future__ import annotations

import time
import unittest

from ai.video_job_manager import VideoJobManager


class TestVideoJobManagerBackground(unittest.TestCase):
    def test_start_returns_immediately_without_waiting(self):
        mgr = VideoJobManager()  # نسخة معزولة، لا Singleton العملية

        def _slow_ffmpeg(path, crf=16):
            time.sleep(0.3)
            return f"/tmp/out_{path}_{crf}.mp4"

        t0 = time.time()
        job_id = mgr.start(_slow_ffmpeg, "رفع الدقة", path="in.mp4", crf=16)
        elapsed = time.time() - t0

        self.assertLess(elapsed, 0.1, "start() لازم يرجع فوراً بدون انتظار العملية")
        job = mgr.get(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, "running")
        self.assertEqual(job.label, "رفع الدقة")

        for _ in range(50):
            if mgr.get(job_id).status != "running":
                break
            time.sleep(0.02)
        self.assertEqual(mgr.get(job_id).status, "done")
        self.assertEqual(mgr.get(job_id).result, "/tmp/out_in.mp4_16.mp4")

    def test_failed_operation_recorded_as_failed_not_raised(self):
        mgr = VideoJobManager()

        def _boom(path):
            raise RuntimeError("فشل ffmpeg مقصود للاختبار")

        job_id = mgr.start(_boom, "قص", path="in.mp4")
        for _ in range(50):
            if mgr.get(job_id).status != "running":
                break
            time.sleep(0.02)

        job = mgr.get(job_id)
        self.assertEqual(job.status, "failed")
        self.assertIn("فشل ffmpeg مقصود", job.error)

    def test_get_unknown_job_returns_none(self):
        mgr = VideoJobManager()
        self.assertIsNone(mgr.get(999999))

    def test_list_jobs_filtered_by_ids_newest_first(self):
        mgr = VideoJobManager()

        def _fast(path):
            return path

        j1 = mgr.start(_fast, "قص", path="a.mp4")
        j2 = mgr.start(_fast, "ضغط", path="b.mp4")
        j3 = mgr.start(_fast, "كتم", path="c.mp4")

        jobs = mgr.list_jobs(job_ids=[j1, j3])
        ids = [j.job_id for j in jobs]
        self.assertEqual(ids, [j3, j1])  # الأحدث أولاً، j2 مستبعد


if __name__ == "__main__":
    unittest.main()
