"""اختبارات: خط أنابيب صناعة المحتوى يشتغل بالخلفية ولا يجمّد الاستدعاء
المباشر، واستعلامات الحالة (جاهز؟) تعكس التقدّم الفعلي.
لا يحتاج مفاتيح API حقيقية — run_content_pipeline مُحاكى بالكامل.
يُشغَّل عبر: python -m unittest ai.test_content_job_manager -v
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from ai.content_job_manager import ContentJobManager


class _FakeArticle:
    title = "عنوان تجريبي"
    meta_description = "وصف تجريبي"
    keywords = ["كلمة1", "كلمة2"]
    word_count = 700
    seo_score = 90
    structured = True
    seo_issues: list = []

    def to_markdown(self) -> str:
        return "# محتوى تجريبي"


class _FakeResult:
    def __init__(self):
        self.topic = "موضوع تجريبي"
        self.geo = "SA"
        self.article = _FakeArticle()
        self.teaser = "تشويقة تجريبية"
        self.per_platform_text = {}
        self.platforms = []
        self.publish_mode = "skipped"
        self.publish_result = {}
        self.schedule_id = None
        self.errors = []


class TestContentJobManagerBackground(unittest.TestCase):
    def test_start_returns_immediately_without_waiting_for_pipeline(self):
        """start() لازم يرجع فوراً حتى لو خط الأنابيب بطيء (محاكاة LLM بطيء)،
        وهذا هو جوهر تشغيله بالخلفية بدل تجميد الواجهة."""
        mgr = ContentJobManager()  # نسخة معزولة، لا Singleton العملية

        def _slow_pipeline(**kwargs):
            time.sleep(0.3)
            return _FakeResult()

        with patch("ai.content_agent.run_content_pipeline", side_effect=_slow_pipeline):
            t0 = time.time()
            job_id = mgr.start(topic="اختبار")
            elapsed = time.time() - t0

        self.assertLess(elapsed, 0.1, "start() لازم يرجع فوراً بدون انتظار الخط")
        job = mgr.get(job_id)
        self.assertIsNotNone(job)
        self.assertEqual(job.status, "running")

        # ننتظر انتهاء الخيط الخلفي فعلياً قبل التحقق من النتيجة النهائية
        for _ in range(50):
            if mgr.get(job_id).status != "running":
                break
            time.sleep(0.02)
        self.assertEqual(mgr.get(job_id).status, "done")
        self.assertIsNotNone(mgr.get(job_id).result)

    def test_failed_pipeline_recorded_as_failed_not_raised(self):
        mgr = ContentJobManager()

        def _boom(**kwargs):
            raise RuntimeError("فشل مقصود للاختبار")

        with patch("ai.content_agent.run_content_pipeline", side_effect=_boom):
            job_id = mgr.start(topic="اختبار فشل")
            for _ in range(50):
                if mgr.get(job_id).status != "running":
                    break
                time.sleep(0.02)

        job = mgr.get(job_id)
        self.assertEqual(job.status, "failed")
        self.assertIn("فشل مقصود", job.error)

    def test_get_unknown_job_returns_none(self):
        mgr = ContentJobManager()
        self.assertIsNone(mgr.get(999999))


class TestAgentCategoriesContentDispatch(unittest.TestCase):
    """يتحقق أن _handle_content_command يبدأ مهمة خلفية فوراً (رد فوري
    يحوي معرّف المهمة) بدل انتظار run_content_pipeline، وأن استعلام
    الحالة يعمل صح."""

    def test_content_request_starts_background_job_and_replies_immediately(self):
        from ai import agent_categories

        def _slow_pipeline(**kwargs):
            time.sleep(0.3)
            return _FakeResult()

        with patch("ai.content_agent.run_content_pipeline", side_effect=_slow_pipeline):
            t0 = time.time()
            reply = agent_categories._handle_content_command("اكتب مقال عن الذكاء الاصطناعي")
            elapsed = time.time() - t0

        self.assertLess(elapsed, 0.1)
        self.assertIsNotNone(reply)
        self.assertIn("بالخلفية", reply)
        self.assertIn("#", reply)

    def test_status_query_before_job_exists(self):
        from ai import agent_categories
        from ai.content_job_manager import get_content_job_manager

        # منظّف: نتأكد من قراءة حالة معرّف مهمة غير موجود قطعاً
        reply = agent_categories._handle_content_command("حالة المهمة #987654321")
        self.assertIsNotNone(reply)
        self.assertIn("لا توجد", reply)


if __name__ == "__main__":
    unittest.main()
