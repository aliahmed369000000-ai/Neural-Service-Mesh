"""اختبار تصدير TikTok (export_tiktok) — كامل بدون مفاتيح API حقيقية.

يُحوّل فيديو تجريبيًا فعليًا بـffmpeg ويحقق المواصفات النهائية بـffprobe.
يتحقق من: التحويل الكامل عند عدم المطابقة · عدم إعادة الترميز عند المطابقة
· ضغط الحجم المتدرج · معالجة المدخلات الفارغة · fallback عند غياب ffmpeg.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _make_test_video(width: int, height: int, fps: int,
                     profile: str = "main", audio: bool = True,
                     duration: float = 3.0) -> bytes:
    """ينشئ فيديو mp4 تجريبي بالمواصفات المطلوبة (بدون أي خدمة خارجية)."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg غير متوفر في البيئة")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name
    cmd = [ffmpeg, "-y", "-v", "error",
           "-f", "lavfi", "-i", "testsrc=size=%dx%d:rate=%d" % (width, height, fps),
           "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000"]
    vf = "format=yuv420p"
    if audio:
        cmd += ["-t", str(duration), "-vf", vf,
                "-c:v", "libx264", "-profile:v", profile, "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-crf", "28",
                "-c:a", "aac", "-b:a", "128k", "-ac", "2", "-ar", "48000",
                "-movflags", "+faststart", out_path]
    else:
        cmd += ["-an", "-t", str(duration), "-vf", vf,
                "-c:v", "libx264", "-profile:v", profile, "-preset", "fast",
                "-pix_fmt", "yuv420p",
                "-crf", "28", out_path]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, f"فشل إنشاء فيديو تجريبي: {proc.stderr}"
    with open(out_path, "rb") as _f:
        data = _f.read()
    os.remove(out_path)
    return data


def _probe(bytes_data: bytes):
    ffprobe = shutil.which("ffprobe") or shutil.which("ffmpeg")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(bytes_data)
        path = f.name
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-print_format", "json",
             "-show_streams", "-i", path],
            capture_output=True, text=True, timeout=30).stdout
        return json.loads(out or "{}")
    finally:
        os.remove(path)


class TestTikTokExport(unittest.TestCase):
    maxDiff = None

    def test_01_non_matching_video_gets_converted(self):
        """فيديو 720x1280 بـ24fps وprofile=main يجب أن يتحول لمواصفات TikTok."""
        from ai.video_engine import export_tiktok
        raw = _make_test_video(720, 1280, 24, profile="main")
        res = export_tiktok(raw)
        self.assertTrue(res["reencoded"], "يجب أن تتم إعادة الترميز عند عدم المطابقة")
        self.assertTrue(res["fits_tiktok"])
        self.assertGreater(len(res["bytes"]), 0)
        info = _probe(res["bytes"])
        streams = info.get("streams") or []
        vid = next(s for s in streams if s["codec_type"] == "video")
        aud = next(s for s in streams if s["codec_type"] == "audio")
        n, d = str(vid["r_frame_rate"]).split("/")
        self.assertEqual((vid["width"], vid["height"]), (1080, 1920),
                         "الأبعاد يجب أن تصبح 1080×1920")
        self.assertEqual(int(n) // int(d), 30, "fps يجب أن يصبح 30")
        self.assertEqual(vid["codec_name"], "h264")
        self.assertTrue(vid.get("profile", "").lower().startswith("high"),
                        "يجب أن يكون profile=high")
        self.assertEqual(aud["codec_name"], "aac")
        self.assertGreaterEqual(int(aud["channels"]), 2, "صوت stereo مطلوب")
        self.assertEqual(aud["sample_rate"], "48000")

    def test_02_matching_video_passes_through(self):
        """فيديو مطابق تمامًا يجب أن يرجع دون إعادة ترميز."""
        from ai.video_engine import export_tiktok
        raw = _make_test_video(1080, 1920, 30, profile="high")
        res = export_tiktok(raw)
        self.assertFalse(res["reencoded"], "لا حاجة لإعادة ترميز — المواصفات مطابقة")
        self.assertEqual(res["bytes"], raw)
        self.assertIn("مطابقة", res["reason"])

    def test_03_size_limit_triggers_recompression(self):
        """حد حجم صغير جدًا يجب أن يفعّل الضغط المتدرج."""
        from ai.video_engine import export_tiktok
        raw = _make_test_video(720, 1280, 24, profile="main", duration=4.0)
        res = export_tiktok(raw, max_size_bytes=200 * 1024)
        self.assertTrue(res["reencoded"])
        # upscale من 720×1280 إلى 1080×1920 يزيد الحجم حتى مع ضغط أعلى؛
        # المهم أن المواصفات النهائية مطابقة لمتطلبات TikTok وأن الحالة
        # (fits_tiktok) تُعكس بدقة — لا نعد بحجم أصغر من الأصل أبدًا.
        info = _probe(res["bytes"])
        vid = next(s for s in (info.get("streams") or [])
                   if s["codec_type"] == "video")
        aud = next((s for s in (info.get("streams") or [])
                    if s["codec_type"] == "audio"), None)
        self.assertEqual((vid["width"], vid["height"]), (1080, 1920))
        self.assertEqual(vid["codec_name"], "h264")
        # حد 200KB مع upscale من 720×1280 إلى 1080×1920:
        # الفيديو يصبح أكبر من أصله حتى مع ضغط أعلى، والمهم أن الحجم
        # انخفض تدريجيًا عبر CRF المتدرج وانتهى تحت الحد المطلوب (200KB).
        self.assertLessEqual(res["exported_size"], 200 * 1024)
        self.assertTrue(res["fits_tiktok"])
        if aud is None:
            self.assertIn("حجم", res["reason"])

    def test_04_no_audio_stream_gets_fixed(self):
        """فيديو بلا مصدر صوت لا يمكن أن يُخلق منه صوت — يجب أن يعيد الترميز
        مع بقاء مواصفات الفيديو مطابقة لمتطلبات TikTok ودون أي استثناء."""
        from ai.video_engine import export_tiktok
        raw = _make_test_video(1080, 1920, 30, profile="high", audio=False)
        res = export_tiktok(raw)
        self.assertTrue(res["reencoded"])
        info = _probe(res["bytes"])
        streams = info.get("streams") or []
        vid = next((s for s in streams if s["codec_type"] == "video"), None)
        self.assertIsNotNone(vid, "يجب بقاء تيار الفيديو h264")
        self.assertEqual(vid["codec_name"], "h264")
        self.assertEqual((vid["width"], vid["height"]), (1080, 1920))

    def test_05_empty_input_safe(self):
        """مدخلات فارغة يجب أن ترجع بأمان دون استثناء."""
        from ai.video_engine import export_tiktok
        res = export_tiktok(b"")
        self.assertFalse(res["reencoded"])
        self.assertEqual(res["bytes"], b"")
        self.assertIn("غير صالحة", res["reason"])

    def test_06_missing_ffmpeg_safe(self):
        """عند غياب ffmpeg يجب الرجوع للأصل بأمان مع توثيق السبب."""
        from ai import video_engine as ve
        raw = _make_test_video(720, 1280, 24)
        real_get = ve._get_ffmpeg_binary
        try:
            ve._get_ffmpeg_binary = lambda: None  # محاكاة غياب ffmpeg
            res = ve.export_tiktok(raw)
            self.assertFalse(res["reencoded"])
            self.assertEqual(res["bytes"], raw)
            self.assertIn("ffmpeg", res["reason"])
        finally:
            ve._get_ffmpeg_binary = real_get

    def test_07_fable_engine_integration(self):
        """FableEngine.generate_tiktok_export يجب أن تعمل عبر lazy import."""
        from ai.fable_engine import FableEngine
        _dummy_llm = lambda prompt, **kw: {"success": True, "text": "ok"}
        engine = FableEngine(llm_fallback=_dummy_llm)
        raw = _make_test_video(720, 1280, 24)
        res = engine.generate_tiktok_export(raw,
                                            max_size_bytes=287 * 1024 * 1024)
        self.assertIsInstance(res, dict)
        self.assertIn("bytes", res)
        self.assertIn("reason", res)
        # يجب أن تطابق المواصفات النهائية مهما كان مسار التحويل
        info = _probe(res["bytes"])
        streams = info.get("streams") or []
        vid = next(s for s in streams if s["codec_type"] == "video")
        aud = next(s for s in streams if s["codec_type"] == "audio")
        self.assertEqual((vid["width"], vid["height"]), (1080, 1920))
        self.assertEqual(vid["codec_name"], "h264")
        self.assertEqual(aud["codec_name"], "aac")


if __name__ == "__main__":
    unittest.main(verbosity=2)
