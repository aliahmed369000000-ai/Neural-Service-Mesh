"""
PDF Export — تصدير القصص والقصائد من تبويب 🎭 إبداع كملفات PDF عربية منسَّقة
===========================================================================
يستخدم نفس أسلوب تشكيل النص العربي المستخدَم مسبقاً في ai/video_engine.py
(arabic_reshaper + bidi.algorithm.get_display) لأن reportlab لا يشكّل
الحروف العربية أو يرتّب اتجاهها تلقائياً — الأحرف بدون هذا التشكيل تظهر
منفصلة وبترتيب معكوس.

يعتمد على خط assets/fonts/NotoNaskhArabic-Regular.ttf المرفق أصلاً مع
المشروع (مع مسارات نظام احتياطية إن وُجدت)، فلا حاجة لملف خط جديد.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import List, Optional

from reportlab.lib.pagesizes import A5
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_RIGHT, TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

logger = logging.getLogger(__name__)

_FONT_NAME = "NotoNaskhArabic"
_FONT_PATH_CANDIDATES = [
    "assets/fonts/NotoNaskhArabic-Regular.ttf",
    str(Path(__file__).resolve().parent.parent / "assets/fonts/NotoNaskhArabic-Regular.ttf"),
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabicUI-Regular.ttf",
]

_font_registered_name: Optional[str] = None


def _ensure_font() -> str:
    """يسجّل خط NotoNaskhArabic في reportlab مرة واحدة فقط (لكل عملية).
    عند فشل كل المسارات يعود إلى Helvetica (لن يعرض عربياً بشكل صحيح،
    لكنه يمنع انهيار التصدير بالكامل)."""
    global _font_registered_name
    if _font_registered_name:
        return _font_registered_name
    for path in _FONT_PATH_CANDIDATES:
        try:
            if not Path(path).is_file():
                continue
            pdfmetrics.registerFont(TTFont(_FONT_NAME, path))
            _font_registered_name = _FONT_NAME
            return _FONT_NAME
        except Exception as e:  # noqa: BLE001
            logger.debug(f"pdf_export: فشل تحميل الخط {path}: {e}")
    logger.warning("pdf_export: لم يُعثر على خط عربي — النص العربي لن يظهر بشكل صحيح")
    _font_registered_name = "Helvetica"
    return _font_registered_name


def _shape(text: str) -> str:
    """يشكّل النص العربي (reshape + bidi) ليُعرض متصلاً وبالاتجاه الصحيح."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(text))
    except Exception as e:  # noqa: BLE001
        logger.debug(f"pdf_export: فشل تشكيل النص: {e}")
        return text


def _base_doc(buf: io.BytesIO) -> SimpleDocTemplate:
    return SimpleDocTemplate(
        buf, pagesize=A5,
        rightMargin=1.8 * cm, leftMargin=1.8 * cm,
        topMargin=1.8 * cm, bottomMargin=1.8 * cm,
    )


def story_to_pdf(title: str, mode: str, character: str, full_text: str) -> bytes:
    """يحوّل نص قصة كاملة (فصول مفصولة بسطرين فارغين عادةً) إلى PDF
    بمحاذاة يمين مناسبة للنثر."""
    font = _ensure_font()
    buf = io.BytesIO()
    doc = _base_doc(buf)

    title_style = ParagraphStyle(
        "ArabicTitle", fontName=font, fontSize=18, leading=26,
        alignment=TA_CENTER, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ArabicSubtitle", fontName=font, fontSize=11, leading=16,
        alignment=TA_CENTER, textColor=HexColor("#666666"), spaceAfter=18,
    )
    body_style = ParagraphStyle(
        "ArabicBody", fontName=font, fontSize=13, leading=24,
        alignment=TA_RIGHT, spaceAfter=12,
    )

    story = [Paragraph(_shape(title or "قصتي"), title_style)]
    subtitle = " · ".join(p for p in [mode, character] if p)
    if subtitle:
        story.append(Paragraph(_shape(subtitle), subtitle_style))
    story.append(Spacer(1, 6))

    paragraphs = [p.strip() for p in (full_text or "").split("\n\n") if p.strip()]
    if not paragraphs:
        paragraphs = [(full_text or "").strip() or "(لا يوجد نص)"]
    for para in paragraphs:
        story.append(Paragraph(_shape(para), body_style))

    doc.build(story)
    return buf.getvalue()


def poem_to_pdf(title: str, topic: str, meter: str, poem_text: str) -> bytes:
    """يحوّل نص قصيدة إلى PDF بمحاذاة توسيطية مناسبة للأبيات الشعرية."""
    font = _ensure_font()
    buf = io.BytesIO()
    doc = _base_doc(buf)

    title_style = ParagraphStyle(
        "ArabicTitle", fontName=font, fontSize=18, leading=26,
        alignment=TA_CENTER, spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ArabicSubtitle", fontName=font, fontSize=11, leading=16,
        alignment=TA_CENTER, textColor=HexColor("#666666"), spaceAfter=18,
    )
    verse_style = ParagraphStyle(
        "ArabicVerse", fontName=font, fontSize=14, leading=26,
        alignment=TA_CENTER, spaceAfter=8,
    )

    story = [Paragraph(_shape(title or "قصيدتي"), title_style)]
    subtitle = " — ".join(p for p in [topic, meter] if p)
    if subtitle:
        story.append(Paragraph(_shape(subtitle), subtitle_style))
    story.append(Spacer(1, 10))

    lines = [l.strip() for l in (poem_text or "").split("\n") if l.strip()]
    if not lines:
        lines = ["(لا يوجد نص)"]
    for line in lines:
        story.append(Paragraph(_shape(line), verse_style))

    doc.build(story)
    return buf.getvalue()
