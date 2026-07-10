"""
Video Engine — محرك رندر الفيديو الفعلي — NSM
=================================================
يركّب سيناريو ExplainerScript (من FableEngine.generate_short/generate_explainer)
+ الصوت المولَّد عبر TTSEngine → فيديو mp4 عمودي فعلي بأسلوب الترجمات
المتحركة كلمة-بكلمة (Kinetic Captions بنمط CapCut/Submagic/Opus Clip) —
كل عبارة قصيرة تظهر بحاجز (pill) ملوّن ونص عريض بحدّ أبيض وتأثير "نبضة"
عند الظهور + زووم Ken-Burns مستمر عبر المشهد بالكامل، بدون أي اعتماد على
ImageMagick (كل النص يُرسم عبر Pillow مباشرة).

المتطلبات (requirements.txt):
    moviepy>=2.0
    imageio-ffmpeg>=0.4.9   # يحمل ثنائي ffmpeg تلقائياً، بدون حاجة لـ apt
    pillow                  # موجود أصلاً بالمشروع (يلزم Pillow>=8.0 لدعم stroke_width في draw.text)
    arabic-reshaper         # موجود أصلاً بالمشروع
    python-bidi             # موجود أصلاً بالمشروع

اختياري (Streamlit Cloud) — packages.txt:
    ffmpeg
    fonts-noto-core

الاستخدام:
    from ai.fable_engine import FableEngine
    engine = FableEngine(llm_fallback=my_llm_fallback)
    script = engine.generate_short(source_text, target_seconds=60)
    engine.render_audio(script)          # يملأ الصوت الفعلي لكل مشهد
    mp4_bytes = engine.render_video(script)   # فيديو mp4 فعلي جاهز
    with open("short.mp4", "wb") as f:
        f.write(mp4_bytes)
"""

from __future__ import annotations

import logging
import os
import tempfile
import textwrap
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger("VideoEngine")

# أبعاد فيديو رأسي قياسي (9:16) — نفس نسبة NotebookLM Shorts
FRAME_W, FRAME_H = 1080, 1920
FPS = 30

# لوحة ألوان تدرّجية تتناوب بين المشاهد (RGB)
_GRADIENT_PAIRS = [
    ((20, 24, 38), (58, 33, 92)),
    ((15, 32, 39), (32, 58, 67)),
    ((44, 20, 60), (90, 30, 60)),
    ((10, 30, 50), (40, 70, 110)),
    ((30, 15, 45), (75, 40, 90)),
]

# مسارات خطوط عربية شائعة في بيئات Linux (Replit / Streamlit Cloud / Debian)
# ترتيب الأولوية: خطوط عريضة (Bold) أولاً — أوضح وأقوى بصرياً لأسلوب
# الترجمات المتحركة (Kinetic Captions) المستخدم بمنصات مثل CapCut/Submagic —
# ثم الأوزان العادية، وDejaVuSans (لا يدعم العربي) آخر خيار مطلق فقط عشان
# لا يفشل الرسم بالكامل.
_SYSTEM_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoKufiArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoKufiArabic-Bold.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabicUI-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/kacst/KacstOne.ttf",
]
_LAST_RESORT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"  # لا يدعم العربي

# مصدر تنزيل احتياطي (GitHub raw، مُتحقَّق منه) إن لم يوجد الخط بالنظام —
# نفضّل Noto Kufi Arabic Bold (خط عرض هندسي عريض، مثالي للعناوين/الترجمات
# المتحركة)، ويُخزَّن محلياً بعد أول تنزيل فلا يُعاد الطلب كل مرة.
_FONT_FALLBACK_URL = (
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/"
    "fonts/NotoKufiArabic/hinted/ttf/NotoKufiArabic-Bold.ttf"
)
_FONT_CACHE_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_CACHE_PATH = _FONT_CACHE_DIR / "NotoKufiArabic-Bold.ttf"


class VideoEngineError(RuntimeError):
    pass


def _resolve_arabic_font() -> Optional[str]:
    """يبحث عن خط عربي صالح بترتيب أولوية صارم:
    1) خطوط عربية بالنظام  2) نسخة مخبأة محلياً من تنزيل سابق
    3) محاولة تنزيل من GitHub (مرة واحدة، تُخزَّن للمرات القادمة)
    4) DejaVuSans كخيار أخير مطلق (لن يعرض العربية بشكل صحيح، لكن أفضل من فشل الرسم)."""
    for path in _SYSTEM_FONT_CANDIDATES:
        if os.path.isfile(path):
            return path

    if _FONT_CACHE_PATH.is_file():
        return str(_FONT_CACHE_PATH)

    try:
        import urllib.request

        _FONT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_FONT_FALLBACK_URL, _FONT_CACHE_PATH)
        logger.info("تم تنزيل خط عربي احتياطي إلى %s", _FONT_CACHE_PATH)
        return str(_FONT_CACHE_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "تعذّر إيجاد/تنزيل خط عربي (%s) — سيُستخدم خط بديل لا يدعم "
            "العربية بشكل صحيح. أضِف 'fonts-noto-core' لـ packages.txt "
            "(Streamlit Cloud) لحل هذا بشكل دائم.", exc,
        )
        return _LAST_RESORT_FONT if os.path.isfile(_LAST_RESORT_FONT) else None


def _shape_arabic(text: str) -> str:
    """يهيّئ النص العربي للعرض الصحيح (اتصال الحروف + اتجاه RTL)."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display

        reshaped = arabic_reshaper.reshape(text)
        return get_display(reshaped)
    except Exception:  # noqa: BLE001
        return text  # نص لاتيني أو فشل التشكيل — يُعرض كما هو


class VideoEngine:
    """يحوّل ExplainerScript (مع صوت مُولَّد مسبقاً) إلى فيديو mp4 فعلي."""

    def __init__(self) -> None:
        self._font_path = _resolve_arabic_font()

    # ── بناء صورة خلفية متدرّجة للمشهد رقم N ─────────────────────────
    def _build_background(self, seg_index: int) -> "Image.Image":
        from PIL import Image

        top, bottom = _GRADIENT_PAIRS[seg_index % len(_GRADIENT_PAIRS)]
        img = Image.new("RGB", (FRAME_W, FRAME_H))
        pixels = img.load()
        for y in range(FRAME_H):
            t = y / FRAME_H
            r = int(top[0] + (bottom[0] - top[0]) * t)
            g = int(top[1] + (bottom[1] - top[1]) * t)
            b = int(top[2] + (bottom[2] - top[2]) * t)
            for x in range(0, FRAME_W, 4):  # خطوة 4px لتسريع الرسم (فرق غير محسوس)
                pixels[x, y] = (r, g, b)
                if x + 1 < FRAME_W:
                    pixels[x + 1, y] = (r, g, b)
                if x + 2 < FRAME_W:
                    pixels[x + 2, y] = (r, g, b)
                if x + 3 < FRAME_W:
                    pixels[x + 3, y] = (r, g, b)
        return img

    # ── تقسيم النص لعبارات قصيرة (كلمة-بكلمة/عبارة-بعبارة) ──────────
    @staticmethod
    def _split_into_chunks(text: str, max_words: int = 3) -> List[str]:
        words = text.split()
        if not words:
            return [text] if text else [""]
        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    # ── رسم عبارة نصية واحدة بأسلوب الترجمات المتحركة (Kinetic Caption):
    #    حاجز (pill) ملوّن خلف نص عريض بحدّ أبيض، أعلى قابلية للقراءة
    #    وأقرب لأسلوب CapCut/Submagic الاحترافي ──────────────────────
    def _draw_caption(
        self,
        img: "Image.Image",
        text: str,
        font_size: int = 84,
        accent_color: Optional[Tuple[int, int, int]] = None,
    ) -> "Image.Image":
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img, "RGBA")
        try:
            font = ImageFont.truetype(self._font_path, font_size) if self._font_path else ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()

        # ⚠️ مهم جداً — ترتيب العمليات هنا يمنع مشكلة النص المشوّه/المبعثر:
        # يجب لفّ السطور بالترتيب المنطقي الأصلي (حسب الكلمات) *قبل* تطبيق
        # التشكيل (reshape) وBiDi. تطبيق get_display (الذي يعكس النص لترتيب
        # العرض البصري) ثم تمرير الناتج إلى textwrap.wrap لاحقاً يقسّم سطراً
        # مُعاد ترتيبه بصرياً بالفعل حسب عدّ الأحرف، فتُقطَّع الكلمات في
        # منتصف تسلسلها البصري وتظهر متكسّرة/معكوسة — بالضبط الخلل السابق.
        stroke_w = max(4, font_size // 14)
        logical_lines = textwrap.wrap(text, width=16) or [text]
        wrapped_lines = [_shape_arabic(line) for line in logical_lines]

        line_heights: List[int] = []
        line_widths: List[int] = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font, stroke_width=stroke_w)
            line_heights.append(bbox[3] - bbox[1])
            line_widths.append(bbox[2] - bbox[0])
        total_h = sum(line_heights) + max(0, len(wrapped_lines) - 1) * 22

        y = (FRAME_H - total_h) // 2

        if accent_color is not None:
            pad_x, pad_y = 44, 26
            block_w = max(line_widths) + pad_x * 2
            block_h = total_h + pad_y * 2
            bx = (FRAME_W - block_w) // 2
            by = y - pad_y
            draw.rounded_rectangle(
                [bx, by, bx + block_w, by + block_h],
                radius=32, fill=(*accent_color, 235),
            )
            text_fill = (18, 14, 10)
            stroke_fill = (255, 255, 255)
        else:
            text_fill = (255, 255, 255)
            stroke_fill = (0, 0, 0)

        for line, lh, lw in zip(wrapped_lines, line_heights, line_widths):
            x = (FRAME_W - lw) // 2
            draw.text(
                (x, y), line, font=font,
                fill=text_fill, stroke_width=stroke_w, stroke_fill=stroke_fill,
            )
            y += lh + 22
        return img

    # ── بناء مقطع فيديو واحد (مشهد) بالصوت المرافق — بأسلوب الترجمات
    #    المتحركة كلمة/عبارة-بعبارة (Kinetic Captions) ─────────────────
    _ACCENT_COLORS = [
        (255, 199, 0),    # أصفر ذهبي
        (255, 92, 92),    # أحمر مرجاني
        (94, 211, 255),   # سماوي
        (178, 130, 255),  # بنفسجي
        (110, 231, 172),  # أخضر نعناعي
    ]

    def _build_segment_clip(self, segment, index: int, tmp_dir: str):
        from moviepy import AudioFileClip, ImageClip, concatenate_videoclips
        import numpy as np

        if not segment.audio_bytes:
            raise VideoEngineError(
                f"المشهد {segment.index} بدون صوت — استدعِ render_audio() أولاً."
            )

        audio_path = os.path.join(tmp_dir, f"seg_{index}.{segment.audio_format or 'mp3'}")
        with open(audio_path, "wb") as f:
            f.write(segment.audio_bytes)

        audio_clip = AudioFileClip(audio_path)
        duration = max(1.2, audio_clip.duration)

        bg_base = self._build_background(index)

        # نقسّم سرد المشهد إلى عبارات قصيرة (2-3 كلمات) تظهر تباعاً —
        # نفس منطق CapCut/Submagic لترجمات كلمة-بكلمة أكثر جاذبية من فقرة
        # ثابتة كاملة طوال المشهد.
        chunks = self._split_into_chunks(segment.narration, max_words=3)
        total_chars = sum(len(c) for c in chunks) or 1
        min_chunk_dur = 0.42
        raw_durations = [max(min_chunk_dur, duration * (len(c) / total_chars)) for c in chunks]
        scale = duration / sum(raw_durations)
        chunk_durations = [d * scale for d in raw_durations]

        sub_clips = []
        elapsed = 0.0
        for i, (chunk_text, chunk_dur) in enumerate(zip(chunks, chunk_durations)):
            accent = self._ACCENT_COLORS[i % len(self._ACCENT_COLORS)]
            frame_img = bg_base.copy()
            frame_img = self._draw_caption(frame_img, chunk_text, accent_color=accent)
            frame_array = np.array(frame_img)

            seg_progress_start = elapsed / duration
            seg_progress_end = min(1.0, (elapsed + chunk_dur) / duration)
            pop_dur = min(0.14, chunk_dur * 0.4)

            def _combined_scale(t, s=seg_progress_start, e=seg_progress_end, cd=chunk_dur, pd=pop_dur):
                # زووم Ken-Burns مستمر ومتصاعد عبر كامل المشهد (وليس مُعاد
                # الانطلاق مع كل عبارة) — إحساس حركي سينمائي متسق.
                local = s + (t / cd) * (e - s) if cd > 0 else s
                base_zoom = 1.0 + 0.14 * local
                # "نبضة" ظهور خفيفة (scale-in) في أول لحظات كل عبارة، بنفس
                # روح أنيميشن "Pop Up" بمنصات الترجمات الاحترافية.
                pop = 0.88 + 0.12 * min(1.0, t / pd) if pd > 0 else 1.0
                return base_zoom * pop

            chunk_clip = (
                ImageClip(frame_array)
                .with_duration(chunk_dur)
                .resized(_combined_scale)
                .with_position("center")
            )
            sub_clips.append(chunk_clip)
            elapsed += chunk_dur

        captioned = concatenate_videoclips(sub_clips, method="compose")
        captioned = captioned.with_audio(audio_clip.with_duration(duration))

        from moviepy import vfx
        return captioned.with_effects([vfx.CrossFadeIn(0.2)])

    # ── الواجهة العامة: رندر الفيديو الكامل ──────────────────────────
    def render(self, script) -> bytes:
        """يبني mp4 فعلي من ExplainerScript (segments لازم تحتوي audio_bytes
        مسبقاً عبر FableEngine.render_audio). يُرجِع bytes الفيديو النهائي."""
        if not script.segments:
            raise VideoEngineError("السيناريو لا يحتوي أي مشاهد.")
        if not script.has_audio:
            raise VideoEngineError(
                "السيناريو بدون صوت مُولَّد — نفّذ render_audio(script) قبل render_video()."
            )

        from moviepy import CompositeVideoClip, concatenate_videoclips

        with tempfile.TemporaryDirectory(prefix="nsm_video_") as tmp_dir:
            clips = [
                self._build_segment_clip(seg, i, tmp_dir)
                for i, seg in enumerate(script.segments)
            ]
            final = concatenate_videoclips(clips, method="compose", padding=-0.15)
            final = final.with_fps(FPS)

            out_path = os.path.join(tmp_dir, "output.mp4")
            final.write_videofile(
                out_path,
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                preset="veryfast",
                threads=2,
                logger=None,
            )

            for c in clips:
                c.close()
            final.close()

            with open(out_path, "rb") as f:
                return f.read()
