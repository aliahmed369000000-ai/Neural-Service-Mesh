"""اختبار ميزة الترجمة AI بمزامنة الكلمات (بدون مفاتيح مدفوعة).

Edge TTS مجاني تماماً (لا يحتاج مفتاح) -- نستخدمه فعلياً لاستخراج
توقيتات الكلمات الحقيقية (WordBoundary بـ boundary="WordBoundary" صريح،
والجمع مع حفظ الصوت في coroutine واحدة لأن stream() يُستهلك مرة واحدة
فقط)، ثم نبني SRT + VTT بمزامنة دقيقة، ونحرق الترجمة على فيديو تجريبي
بـffmpeg ونتحقق من النتيجة بصرياً بـffprobe + PIL.

ملاحظات مكتشفة أثناء البناء:
  - الافتراضي في edge-tts هو "SentenceBoundary" فلا توقيت كلمة أبداً؛
    يجب Communicate(..., boundary="WordBoundary").
  - offset/duration بوحدات 100ns ticks (1e-7s) -- تحقّق عملي (v7.2.8):
    خام 'الذكاء' = 66,625,000 tick = 6.6625s، و'بسم' تبدأ عند 1,000,000 =
    0.1s. أي قسمة على 1e7 (وليس 1000).
  - stream() يستهلك المصدر؛ لا يمكن إعادة استدعائه بعد save() أو مرة
    ثانية على Communicate جديد لنفس النص.
  - short في المشروع هو ExplainerScript (segments = ExplainerSegment)
    ولا يوجد ShortSegment إطلاقاً -- يجب audio_bytes حقيقية لكل مقطع
    لأن build_srt يقيس المدة بـmoviepy.AudioFileClip.
"""
import asyncio
import os
import subprocess
import sys
import tempfile
import unittest

import edge_tts

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

ARABIC_SENTENCE = ("بسم الله الرحمن الرحيم وفي هذه الحلقة سنتعلم "
                   "أشياء جديدة ومفيدة جدا في عالم الذكاء الاصطناعي")


def _run(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _edge_tts_sync(text, voice="ar-SA-HamedNeural"):
    """يستدعي Edge TTS مع WordBoundary صريح ويرجع (الصوت, التوقيتات).

    دمج الحفظ + جمع التوقيتات في coroutine واحدة لأن comm.stream()
    يُستهلك مرة واحدة فقط (لا يمكن إعادة تشغيله على Communicate جديد
    لنفس النص، وsave() يستهلكه أيضاً).
    """
    comm = edge_tts.Communicate(text, voice=voice,
                                boundary="WordBoundary")
    audio_chunks = []
    timings = []

    async def _inner():
        async for chunk in comm.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                # edge-tts WordBoundary بوحدات 100ns ticks (1 tick = 1e-7s)
                timings.append({
                    "word": chunk["text"],
                    "start": chunk["offset"] / 1e7,
                    "duration": chunk["duration"] / 1e7,
                })

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_inner())
    finally:
        loop.close()
    return b"".join(audio_chunks), timings


def _make_explainer_with_timings(timings, audio_bytes):
    """يبني ExplainerSegment/Script بها word_timings حقيقية
    (بالصيغة التي يضعها render_audio من TTSEngine: (نص, ثانية, ثانية))."""
    from ai.fable_engine import ExplainerScript, ExplainerSegment

    timing_tuples = [(t["word"], t["start"], t["duration"])
                     for t in timings]
    total = sum(t["duration"] for t in timings)
    seg = ExplainerSegment(
        index=1,
        narration=ARABIC_SENTENCE,
        visual_notes="مشهد تعليمي",
        est_seconds=max(5, int(round(total))),
        audio_bytes=audio_bytes,
        audio_format="mp3",
        audio_provider="edge_tts",
        word_timings=timing_tuples,
    )

    class _FakeShort:
        title = "اختبار ترجمة AI"
        style = "classic"
        narrator = "راوي"
        segments = [seg]
        script_text = ARABIC_SENTENCE
        provider = "edge_tts"

        @property
        def has_audio(self):
            return bool(self.segments) and all(
                s.audio_bytes for s in self.segments)

    return _FakeShort()


class TestAiSubtitles(unittest.TestCase):
    """اختبارات حقيقية (Edge TTS مجاني + ffmpeg فعلي) ومحاكاة fallback."""

    @classmethod
    def setUpClass(cls):
        cls.audio_bytes, cls.timings = _edge_tts_sync(ARABIC_SENTENCE)
        cls.short = _make_explainer_with_timings(cls.timings,
                                                 cls.audio_bytes)
        assert cls.timings, "Edge TTS يجب أن يُخرج WordBoundary على الأقل"
        assert cls.audio_bytes, "الصوت يجب ألا يكون فارغاً"

    # ── 1. مزامنة الكلمات الحقيقية من TTS ───────────────────────────────
    def test_01_edge_tts_word_timings(self):
        t = self.timings
        words = [x["word"] for x in t]
        joined = "".join(words)
        self.assertIn("بسم", joined)
        self.assertGreater(len(t), 8, "يجب أن يُستخرج أكثر من 8 كلمات")
        # التوقيتات مرتبة تصاعدياً (شرط أساسي للمزامنة)
        starts = [x["start"] for x in t]
        self.assertEqual(starts, sorted(starts), "التوقيتات غير مرتبة!")
        # المدة الإجمالية منطقية (جملة ~8 ثوانٍ) وليست جزء من ألف ثانية
        total = sum(x["duration"] for x in t)
        self.assertGreater(total, 3.0,
                           "المدة أقل من 3 ثوانٍ — على الأغلب خطأ وحدة "
                           "قياس (ms vs ns ticks)")
        # الكلمة الأولى تبدأ مبكراً جداً (قرب 0) والأخيرة قرب نهاية الصوت
        first_word_end = t[0]["start"] + t[0]["duration"]
        self.assertLess(first_word_end, 2.0)

    # ── 2. بناء SRT/VTT بمزامنة دقيقة ───────────────────────────────────
    def test_02_word_synced_srt_vtt(self):
        from ai.video_engine import generate_word_synced_subtitles

        srt = generate_word_synced_subtitles(self.short, max_words=1)
        self.assertIn("1\n", srt)
        self.assertIn("-->", srt)
        # نمط الكلمة الواحدة: كل سطر نصي لا يتجاوز 3 كلمات
        lines = [ln for ln in srt.split("\n") if ln and not ln.isdigit()
                 and "-->" not in ln]
        for ln in lines:
            self.assertLessEqual(len(ln.strip().split()), 3,
                                 f"سطر نصي طويل بنمط max_words=1: {ln}")

        vtt = generate_word_synced_subtitles(self.short, max_words=1,
                                             subtitle_format="vtt")
        self.assertTrue(vtt.startswith("WEBVTT"))
        vtt_body = vtt.split("WEBVTT", 1)[1][:400]
        self.assertIn(".", vtt_body)
        self.assertNotIn(",", vtt_body)

        # صيغة غير معروفة تتراجع لـSRT
        bad = generate_word_synced_subtitles(self.short,
                                             subtitle_format="xyz")
        self.assertFalse(bad.startswith("WEBVTT"))

    # ── 3. fallback: مقطع بلا word_timings (لا استثناء أبداً) ──────────
    def test_03_fallback_no_timings(self):
        from ai.video_engine import build_srt
        from ai.fable_engine import ExplainerScript, ExplainerSegment

        # نفس المدة الصوتية لكن بلا توقيتات وبنفس audio_bytes الحقيقية
        seg = ExplainerSegment(
            index=1,
            narration=ARABIC_SENTENCE,
            visual_notes="",
            est_seconds=8,
            audio_bytes=self.audio_bytes,
            audio_format="mp3",
            audio_provider="other_tts",
            word_timings=[],   # لا توقيتات -- يتراجع للتقدير التناسبي
        )

        class _Short:
            title = "بدون توقيتات"
            style = "classic"
            narrator = "راوي"
            segments = [seg]
            script_text = ARABIC_SENTENCE
            provider = "other_tts"

            @property
            def has_audio(self):
                return bool(self.segments) and all(
                    s.audio_bytes for s in self.segments)

        # لا استثناء -- التقدير التناسبي القديم يعمل
        srt = build_srt(_Short())
        self.assertIn("-->", srt)
        self.assertGreater(len(srt), 50)

    # ── 3ب. الوحدة القياسية للتوقيتات (100ns ticks وليس ms) ──────────
    def test_03b_timing_unit_is_100ns_ticks(self):
        # تحقّق دفاعي: 'بسم' تبدأ عند 0.1s وليس 1000s ولا 0.0001s
        first = self.timings[0]
        self.assertEqual(first["word"], "بسم")
        self.assertGreater(first["start"], 0.0)
        self.assertLess(first["start"], 1.0)
        # 'الذكاء' وسط الجملة: بين 5 و 7.5 ثانية (صوت إجمالي ~8.9s)
        ai_idx = next(i for i, t in enumerate(self.timings)
                      if t["word"] == "الذكاء")
        self.assertGreater(self.timings[ai_idx]["start"], 5.0)
        self.assertLess(self.timings[ai_idx]["start"], 7.5)

    # ── 4. حرق الترجمة على فيديو فعلي ───────────────────────────────────
    def test_04_burn_subtitles_real_ffmpeg(self):
        from ai.video_engine import (burn_subtitles,
                                     generate_word_synced_subtitles)

        srt = generate_word_synced_subtitles(self.short, max_words=2)

        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src.mp4")
            _run([
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i", "testsrc=duration=5:size=1080x1920:rate=30",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
                "-c:v", "libx264", "-c:a", "aac", "-shortest", src,
            ])
            if not os.path.isfile(src):
                self.skipTest("تعذر إنشاء الفيديو التجريبي بـffmpeg")
            video = open(src, "rb").read()

            out = burn_subtitles(video, srt)
            self.assertTrue(out["reencoded"],
                            f"يجب إعادة الترميز: {out['reason']}")
            self.assertGreater(len(out["bytes"]), 10000)
            self.assertLessEqual(len(out["bytes"]), len(video) * 3)

            # التحقق البصري: النص محروق فعلياً على الإطارات
            probe_frame = os.path.join(tmp, "frame.png")
            _run(["ffmpeg", "-y", "-v", "error", "-ss", "1",
                  "-i", src, "-vframes", "1", probe_frame])
            from PIL import Image
            before = Image.open(probe_frame).convert("L")

            burned_path = os.path.join(tmp, "burned.mp4")
            with open(burned_path, "wb") as f:
                f.write(out["bytes"])
            probe_burned = os.path.join(tmp, "frame_burned.png")
            _run(["ffmpeg", "-y", "-v", "error", "-ss", "1",
                  "-i", burned_path, "-vframes", "1", probe_burned])
            after = Image.open(probe_burned).convert("L")

            # الفرق بين الإطارين يجب أن يكون واضحاً (نص محروق)
            import numpy as np
            diff = np.abs(np.array(before).astype(int)
                          - np.array(after).astype(int))
            changed = (diff > 40).sum()
            self.assertGreater(changed, 500,
                               f"النص لم يُحرق بشكل واضح ({changed} بكسل)")

    # ── 5. حالات حماية: مدخلات فارغة / ffmpeg غائب ─────────────────────
    def test_05_empty_inputs_safe(self):
        from ai.video_engine import burn_subtitles

        out = burn_subtitles(b"", "1\n00:00:01,000 --> 00:00:02,000\ntext")
        self.assertFalse(out["reencoded"])
        self.assertEqual(len(out["bytes"]), 0)

        out = burn_subtitles(b"\x00" * 100, "")
        self.assertFalse(out["reencoded"])

    def test_06_ffmpeg_missing(self):
        from ai import video_engine as ve

        orig_which = ve.shutil.which
        try:
            ve.shutil.which = lambda *a: None
            out = ve.burn_subtitles(b"\x00" * 100, "1\n0:0:1,000 --> 0:0:2,000\nt")
            self.assertFalse(out["reencoded"])
            self.assertIn("ffmpeg", out["reason"])
        finally:
            ve.shutil.which = orig_which


if __name__ == "__main__":
    unittest.main(verbosity=2)
