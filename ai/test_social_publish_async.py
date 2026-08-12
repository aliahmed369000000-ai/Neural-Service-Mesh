"""اختبارات: publish_async في ai/social_agent.py يبدأ النشر بالخلفية
ويرجع فوراً بدل انتظار publish_to حتى انتهاء كل المنصات (كان زر «نشر
الآن» في ui_pages/social_agent.py يجمّد واجهة Streamlit بسبب هذا الانتظار
المتزامن). لا يحتاج مفاتيح API حقيقية — publish_to مُحاكى بالكامل.

يُشغَّل عبر: python -m unittest ai.test_social_publish_async -v
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from ai.social_agent import SocialAgentManager


class TestPublishAsync(unittest.TestCase):
    def setUp(self):
        # نسخة معزولة عن الـSingleton العملية حتى لا تتشارك حالة بين الاختبارات
        self.mgr = SocialAgentManager()

    def test_publish_async_returns_immediately(self):
        def _slow_publish_to(platforms, text, resume_key=None, per_platform_text=None):
            time.sleep(0.3)
            return {p: f"post_{p}_123" for p in platforms}

        with patch.object(self.mgr, "publish_to", side_effect=_slow_publish_to):
            t0 = time.time()
            job_id = self.mgr.publish_async(["telegram", "discord"], "نص تجريبي")
            elapsed = time.time() - t0

        self.assertLess(elapsed, 0.1, "publish_async لازم يرجع فوراً بدون انتظار publish_to")
        job = self.mgr.get_publish_job(job_id)
        self.assertEqual(job["status"], "running")

        for _ in range(50):
            if self.mgr.get_publish_job(job_id)["status"] != "running":
                break
            time.sleep(0.02)

        job = self.mgr.get_publish_job(job_id)
        self.assertEqual(job["status"], "done")
        self.assertEqual(job["results"]["telegram"], "post_telegram_123")
        self.assertEqual(job["results"]["discord"], "post_discord_123")

    def test_publish_async_records_failure_without_raising(self):
        with patch.object(self.mgr, "publish_to", side_effect=RuntimeError("خطأ مقصود")):
            job_id = self.mgr.publish_async(["telegram"], "نص")
            for _ in range(50):
                if self.mgr.get_publish_job(job_id)["status"] != "running":
                    break
                time.sleep(0.02)

        job = self.mgr.get_publish_job(job_id)
        self.assertEqual(job["status"], "failed")
        self.assertIn("خطأ مقصود", job["error"])

    def test_get_publish_job_unknown_returns_none(self):
        self.assertIsNone(self.mgr.get_publish_job(999999))

    def test_multiple_publish_jobs_have_distinct_ids(self):
        with patch.object(self.mgr, "publish_to", return_value={"telegram": "ok"}):
            id1 = self.mgr.publish_async(["telegram"], "نص1")
            id2 = self.mgr.publish_async(["telegram"], "نص2")
        self.assertNotEqual(id1, id2)


if __name__ == "__main__":
    unittest.main()
