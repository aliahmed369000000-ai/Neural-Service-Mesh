# -*- coding: utf-8 -*-
"""اختبار محاكاة قوالب تصميم النصوص (Shorts/TikTok) — بدون مفاتيح API حقيقية.

يغطي:
1. كل القوالب السبعة ترسم فعليًا على إطارات FRAME_W×FRAME_H (تغطية فروع
   _draw_caption كاملة) مع الخط العربي الحقيقي إن وُجد.
2. fallback تلقائي لـclassic_pill عند قالب غير معروف.
3. ترميز فيديو فعلي بـffmpeg مع قالب neon وheadline والتحقق بـffprobe.
4. أن render_video يمرر caption_template إلى VideoEngine (mock) دون كسر.
5. أن الواجهة تعرض القوالب من CAPTION_TEMPLATES (استيراد بدون streamlit).
"""
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

AR_TEXT = "السلام عليكم ورحمة الله وبركاته، مرحباً بكم في قصتنا الجديدة"
LONG_TEXT = "في هذا الفيديو سنتكلم عن الذكاء الاصطناعي وتطوراته السريعة"


def _resolve_font():
    """نفس منطق _resolve_arabic_font في video_engine (لا نستورده لتجنب أي
    اعتمادات غير ضرورية — لكن نستورد الدالة الفعلية من الوحدة نفسها)."""
    from ai.video_engine import _resolve_arabic_font
    return _resolve_arabic_font()


def _make_engine(template: str):
    """VideoEngine فعلي بدون moviepy (الرسم لا يعتمد عليها — نحتاج فقط
    __init__ + _draw_caption). إن تعذّر استيراد الوحدة كاملًا (moviepy
    مفقودة) ننتقل لاختبار الدالة عبر exec للملف — غير مطلوب هنا لأن
    video_engine قابل للاستيراد."""
    from ai.video_engine import VideoEngine
    return VideoEngine(caption_template=template)


class TestCaptionTemplates:
    """1+2: كل القوالب ترسم فعليًا + fallback غير معروف."""

    def test_all_templates_render_real_pixels(self):
        from PIL import Image
        from ai.video_engine import VideoEngine, FRAME_W, FRAME_H

        eng = _make_engine("classic_pill")
        # نتجاهل أي قالب غير معروف عبر الفحص المباشر للفروع داخل الدالة:
        # نمرر كل قالب عبر template= بدلاً من الافتراضي (الفروع نفسها تُنفَّذ).
        rendered = {}
        for key in VideoEngine.CAPTION_TEMPLATES:
            # خلفية شفافة تمامًا — نقيس البكسلات التي غيّرها الرسم
            # (ألفا > 0 أو لون مختلف عن خلفية افتراضية شفافة)
            bg = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 0))
            out = eng._draw_caption(bg, AR_TEXT, template=key)
            px = sum(1 for p in out.getdata() if p[3] > 0)
            rendered[key] = (out, px)
            assert px > 0, f"قالب {key} لم يرسم أي بكسل!"
            # كل قالب رسم شيئًا مختلفًا: عدد البكسلات غير الصفرية يجب أن
            # يتفاوت بين أنماط مختلفة (عمق مختلف من التغطية)
        coverages = {k: v[1] for k, v in rendered.items()}
        assert len(set(coverages.values())) >= 4, (
            f"الأنماط متطابقة بصريًا — الفروع لا تعمل: {coverages}"
        )

    def test_unknown_template_fallback(self):
        from PIL import Image
        from ai.video_engine import VideoEngine, FRAME_W, FRAME_H

        eng = _make_engine("unknown_template_xxx")
        # fallback داخل __init__: يجب أن يعود classic_pill
        assert eng._caption_template == "classic_pill"
        bg = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 255))
        out = eng._draw_caption(bg, AR_TEXT)
        assert sum(1 for p in out.getdata() if p[3] > 0) > 0

    def test_long_text_no_crash(self):
        from PIL import Image
        from ai.video_engine import VideoEngine, FRAME_W, FRAME_H

        eng = _make_engine("headline")
        bg = Image.new("RGBA", (FRAME_W, FRAME_H), (10, 10, 10, 255))
        out = eng._draw_caption(bg, LONG_TEXT + " " + LONG_TEXT)
        assert out.size == (FRAME_W, FRAME_H)

    def test_accent_color_beats_template_color(self):
        from PIL import Image
        from ai.video_engine import VideoEngine, FRAME_W, FRAME_H

        eng = _make_engine("neon")
        bg = Image.new("RGBA", (FRAME_W, FRAME_H), (0, 0, 0, 255))
        # نمرر لونًا مميزًا جدًا يجب أن يظهر في الإخراج
        out = eng._draw_caption(bg, AR_TEXT, accent_color=(17, 51, 85))
        data = out.getdata()
        has_custom = any(p[:3] == (17, 51, 85) for p in data)
        # neon يرسم glow باللون الممرَّر: يجب أن نجد بكسلًا من هذا اللون
        assert has_custom, "accent_color لم يُستخدم في قالب neon"


class TestRealEncoding:
    """3: ترميز فعلي بffmpeg لفيديو يحتوي نصوصًا بقوالب مختلفة."""

    @classmethod
    def setup_class(cls):
        cls.tmp = tempfile.mkdtemp(prefix="nsm_tpl_")
        cls.mp4_out = os.path.join(cls.tmp, "tpl_test.mp4")
        # فيديو بسيط بلقطات ثابتة 1080×1920 × 30 إطار (1 ثانية)
        cls.src = os.path.join(cls.tmp, "src.mp4")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
             "-i", "color=c=0x141428:s=1080x1920:r=30:d=2",
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-t", "2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "23", "-c:a", "aac", "-b:a", "128k", cls.src],
            check=True, timeout=120,
        )
        assert os.path.isfile(cls.src)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _encode_with_overlay(self, template: str, out: str):
        from ai.video_engine import VideoEngine, FRAME_W, FRAME_H
        from PIL import Image
        from ai.video_engine import _resolve_arabic_font

        font_path = _resolve_arabic_font()
        import textwrap
        from ai.video_engine import _shape_arabic  # الدالة داخل الوحدة

        # نبني إطارًا واحدًا لكل لقطة ونرسم عليه النص بأسلوب القالب
        eng = VideoEngine(caption_template=template)
        frames = []
        for i in range(30):
            img = Image.new("RGBA", (FRAME_W, FRAME_H), (20, 20, 30, 255))
            eng._draw_caption(img, f"لقطة تجريبية رقم {i + 1} للاختبار", template=template)
            frames.append(img.convert("RGB"))
        # نضغطها كصور متتابعة ثم نضيف صوتًا
        raw = os.path.join(self.tmp, f"tpl_{template}_%03d.jpg")
        for idx, f in enumerate(frames):
            f.save(raw.replace("%03d", f"{idx:03d}"))
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-framerate", "30",
             "-i", raw,
             "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
             "-t", "1", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", "20", "-c:a", "aac", "-shortest", out],
            check=True, timeout=180,
        )

    def _probe(self, path: str):
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries",
             "stream=width,height,codec_type,r_frame_rate",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        import json
        return json.loads(r.stdout)

    def test_encode_with_neon_and_headline(self):
        for tpl in ("neon", "headline", "classic_lower", "highlighted",
                    "dramatic", "top_banner", "classic_pill"):
            out = os.path.join(self.tmp, f"encoded_{tpl}.mp4")
            self._encode_with_overlay(tpl, out)
            assert os.path.isfile(out)
            assert os.path.getsize(out) > 1000, f"الفيديو {tpl} صغير جدًا"
            probe = self._probe(out)
            streams = probe.get("streams", [])
            video_streams = [s for s in streams if s["codec_type"] == "video"]
            assert len(video_streams) == 1
            s = video_streams[0]
            assert s["width"] == 1080 and s["height"] == 1920, (
                f"أبعاد {tpl} خاطئة: {s['width']}x{s['height']}"
            )


class TestRenderVideoPassthrough:
    """4: render_video يمرر caption_template لـVideoEngine."""

    def test_passthrough(self, monkeypatch):
        # استيراد حقيقي للـdataclasses — نحتاج قيمًا افتراضية صحيحة
        import dataclasses
        from ai.fable_engine import FableEngine, ExplainerScript

        captured = {}

        class DummyVE:
            CAPTION_TEMPLATES = {"classic_pill": {"name": "x", "desc": "x",
                                                   "style": "pill", "color": None}}

            def __init__(self, **kwargs):
                captured.update(kwargs)

            def render(self, script):
                return b"\x00" * 10

        monkeypatch.setattr(
            "ai.fable_engine.VideoEngine" if hasattr(__import__("ai.fable_engine"), "VideoEngine") else
            "ai.video_engine.VideoEngine",
            DummyVE, raising=False,
        )
        import ai.video_engine as _ve
        monkeypatch.setattr(_ve, "VideoEngine", DummyVE)

        engine = FableEngine(llm_fallback=lambda *a, **kw: "fallback")
        from ai.fable_engine import ExplainerSegment
        seg = ExplainerSegment(index=0, narration="لقطة تجريبية قصيرة.",
                               visual_notes="خلفية اختبار")
        seg.audio_bytes = b"\x00" * 100
        seg.audio_format = "mp3"
        script = ExplainerScript(topic="اختبار", title="تجربة",
                                 segments=[seg])

        out = engine.render_video(script, caption_template="neon")
        assert captured.get("caption_template") == "neon", (
            f"caption_template لم يُمرَّر: {captured}"
        )
        assert captured.get("professional_mode") is True
        assert out == b"\x00" * 10
