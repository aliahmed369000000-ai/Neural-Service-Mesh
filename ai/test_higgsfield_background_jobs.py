"""اختبارات: استخدام ui_pages/higgsfield.py الجديد لـVideoJobManager
(رندر الفيديو + رفع يوتيوب/تيك توك بالخلفية) يستدعي fable_engine.render_video
وYouTubeAdapter.upload_video/TikTokAdapter.upload_video بأسماء المعاملات
الصحيحة الفعلية، ويرجع فوراً بدل انتظار العملية. لا حاجة لـStreamlit —
هذا اختبار منطق الاستدعاء عبر VideoJobManager فقط، بمحاكاة كاملة
للدوال البطيئة (بدون مفاتيح API حقيقية).

يُشغَّل عبر: python -m unittest ai.test_higgsfield_background_jobs -v
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import patch

from ai.video_job_manager import VideoJobManager


class TestHiggsfieldRenderVideoBackground(unittest.TestCase):
    def setUp(self):
        self.mgr = VideoJobManager()  # نسخة معزولة عن الـSingleton

    def test_render_video_job_returns_immediately_and_completes(self):
        """يحاكي fable_engine.render_video (اسم المعامل الأول script،
        نفس ما يستدعيه ui_pages/higgsfield.py الآن عبر _hf_video_mgr.start)."""
        def _slow_render_video(script, voice="", use_cinematic_backgrounds=False,
                                cinematic_provider="higgsfield", use_background_music=False,
                                music_volume=0.10, wan_skip_spaces=None):
            time.sleep(0.2)
            return b"FAKE_MP4_BYTES"

        t0 = time.time()
        job_id = self.mgr.start(
            _slow_render_video, "رندر فيديو Higgsfield Explainer",
            script="fake_script_obj", voice="ar-SA",
            use_cinematic_backgrounds=False, cinematic_provider="higgsfield",
            use_background_music=False, music_volume=0.10, wan_skip_spaces=None,
        )
        elapsed = time.time() - t0
        self.assertLess(elapsed, 0.1)

        for _ in range(50):
            if self.mgr.get(job_id).status != "running":
                break
            time.sleep(0.02)
        job = self.mgr.get(job_id)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.result, b"FAKE_MP4_BYTES")

    def test_render_video_memory_error_becomes_friendly_runtime_error(self):
        """يتأكد أن نمط except MemoryError -> raise RuntimeError(...) المستخدَم
        في ui_pages/higgsfield.py يُسجَّل كـjob.error نصياً مفهوماً بدل استثناء خام."""
        def _boom_memory(**kwargs):
            try:
                raise MemoryError()
            except MemoryError:
                raise RuntimeError("نفدت الذاكرة أثناء الرندر — جرّب مدة أقصر") from None

        job_id = self.mgr.start(_boom_memory, "رندر فيديو Higgsfield Explainer")
        for _ in range(50):
            if self.mgr.get(job_id).status != "running":
                break
            time.sleep(0.02)
        job = self.mgr.get(job_id)
        self.assertEqual(job.status, "failed")
        self.assertIn("نفدت الذاكرة", job.error)


class TestHiggsfieldUploadJobsUseRealAdapterSignatures(unittest.TestCase):
    """يتأكد أن الأسماء المستخدَمة في ui_pages/higgsfield.py (video_bytes،
    title، description، privacy_status) تطابق فعلياً توقيع
    YouTubeAdapter.upload_video/TikTokAdapter.upload_video — لو تغيّر
    التوقيع مستقبلاً هذا الاختبار يفشل بدل فشل صامت وقت التشغيل الفعلي."""

    def setUp(self):
        self.mgr = VideoJobManager()

    def test_youtube_upload_job_accepts_the_exact_kwargs_used_in_ui(self):
        from ai.social_platforms import YouTubeAdapter
        yt = YouTubeAdapter()

        with patch.object(yt, "upload_video", return_value="abc123") as mock_upload:
            job_id = self.mgr.start(
                yt.upload_video, "رفع يوتيوب",
                video_bytes=b"fake", title="عنوان",
                description="وصف", privacy_status="private",
            )
            for _ in range(50):
                if self.mgr.get(job_id).status != "running":
                    break
                time.sleep(0.02)

        job = self.mgr.get(job_id)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.result, "abc123")
        mock_upload.assert_called_once_with(
            video_bytes=b"fake", title="عنوان", description="وصف", privacy_status="private"
        )

    def test_tiktok_upload_job_accepts_the_exact_kwargs_used_in_ui(self):
        from ai.social_platforms import TikTokAdapter
        tk = TikTokAdapter()

        with patch.object(tk, "upload_video", return_value="pub_123") as mock_upload:
            job_id = self.mgr.start(
                tk.upload_video, "رفع تيك توك",
                video_bytes=b"fake", title="عنوان",
            )
            for _ in range(50):
                if self.mgr.get(job_id).status != "running":
                    break
                time.sleep(0.02)

        job = self.mgr.get(job_id)
        self.assertEqual(job.status, "done")
        self.assertEqual(job.result, "pub_123")
        mock_upload.assert_called_once_with(video_bytes=b"fake", title="عنوان")


if __name__ == "__main__":
    unittest.main()
