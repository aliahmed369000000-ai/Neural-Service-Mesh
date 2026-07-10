"""
Video Engine — محرك رندر الفيديو الفعلي — NSM
=================================================
يركّب سيناريو ExplainerScript (من FableEngine.generate_short/generate_explainer)
+ الصوت المولَّد عبر TTSEngine → فيديو mp4 عمودي فعلي (Kinetic Typography)،
بدون أي اعتماد على ImageMagick (كل النص يُرسم عبر Pillow مباشرة).

المتطلبات (requirements.txt):
    moviepy==1.0.3
    imageio-ffmpeg>=0.4.9   # يحمل ثنائي ffmpeg تلقائياً، بدون حاجة لـ apt
    pillow                  # موجود أصلاً بالمشروع
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
# ترتيب الأولوية مهم: خطوط تدعم العربية أولاً، وDejaVuSans (لا يدعم العربي)
# آخر خيار مطلق فقط عشان لا يفشل الرسم بالكامل.
_SYSTEM_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabicUI-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/kacst/KacstOne.ttf",
]
_LAST_RESORT_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"  # لا يدعم العربي

# مصدر تنزيل احتياطي (GitHub raw، مُتحقَّق منه) إن لم يوجد الخط بالنظام —
# يُخزَّن محلياً بعد أول تنزيل فلا يُعاد الطلب كل مرة
_FONT_FALLBACK_URL = (
    "https://raw.githubusercontent.com/notofonts/notofonts.github.io/main/"
    "fonts/NotoNaskhArabic/hinted/ttf/NotoNaskhArabic-Regular.ttf"
)
_FONT_CACHE_DIR = Path(__file__).resolve().parent.parent / "assets" / "fonts"
_FONT_CACHE_PATH = _FONT_CACHE_DIR / "NotoNaskhArabic-Regular.ttf"


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

    # ── رسم نص عربي متوسّط الشاشة مع التفاف تلقائي ──────────────────
    def _draw_caption(self, img: "Image.Image", text: str, font_size: int = 68) -> "Image.Image":
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype(self._font_path, font_size) if self._font_path else ImageFont.load_default()
        except Exception:  # noqa: BLE001
            font = ImageFont.load_default()

        shaped = _shape_arabic(text)
        # التفاف تقريبي بعدد الأحرف؛ نضبط العرض الفعلي عبر textbbox
        max_width_px = int(FRAME_W * 0.82)
        wrapped_lines: List[str] = []
        for raw_line in textwrap.wrap(shaped, width=26):
            wrapped_lines.append(raw_line)

        line_heights = []
        for line in wrapped_lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_heights.append(bbox[3] - bbox[1])
        total_h = sum(line_heights) + (len(wrapped_lines) - 1) * 18

        y = (FRAME_H - total_h) // 2
        for line, lh in zip(wrapped_lines, line_heights):
            bbox = draw.textbbox((0, 0), line, font=font)
            line_w = bbox[2] - bbox[0]
            x = (FRAME_W - line_w) // 2
            # ظل خفيف لتحسين القراءة فوق أي خلفية
            draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0, 140))
            draw.text((x, y), line, font=font, fill=(255, 255, 255))
            y += lh + 18
        return img

    # ── بناء مقطع فيديو واحد (مشهد) بالصوت المرافق ──────────────────
    def _build_segment_clip(self, segment, index: int, tmp_dir: str):
        from moviepy.editor import AudioFileClip, ImageClip
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

        bg = self._build_background(index)
        bg = self._draw_caption(bg, segment.narration)
        frame_array = np.array(bg)

        # تأثير Ken Burns بسيط (زووم تدريجي خفيف) لإحساس حركي بدون رسوم AI
        clip = (
            ImageClip(frame_array)
            .set_duration(duration)
            .resize(lambda t: 1.0 + 0.045 * (t / duration))
            .set_position("center")
        )
        clip = clip.set_audio(audio_clip.set_duration(duration))
        return clip.crossfadein(0.25)

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

        from moviepy.editor import CompositeVideoClip, concatenate_videoclips

        with tempfile.TemporaryDirectory(prefix="nsm_video_") as tmp_dir:
            clips = [
                self._build_segment_clip(seg, i, tmp_dir)
                for i, seg in enumerate(script.segments)
            ]
            final = concatenate_videoclips(clips, method="compose", padding=-0.15)
            final = final.set_fps(FPS)

            out_path = os.path.join(tmp_dir, "output.mp4")
            final.write_videofile(
                out_path,
                fps=FPS,
                codec="libx264",
                audio_codec="aac",
                preset="veryfast",
                threads=2,
                verbose=False,
                logger=None,
            )

            for c in clips:
                c.close()
            final.close()

            with open(out_path, "rb") as f:
                return f.read()
