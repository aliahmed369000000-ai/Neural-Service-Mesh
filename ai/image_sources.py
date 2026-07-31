"""
ai/image_sources.py
=====================
طبقة موحّدة لجلب/توليد الصور المستخدمة في محرك الموشن جرافيك — كل المصادر
هنا قانونية للاستخدام التجاري، بلا أي استخراج (scraping) من Pinterest أو
أي منصة تمنع ذلك في شروط استخدامها.

ثلاثة مصادر مُفعَّلة الآن (المصدر الرابع — ربط حساب Pinterest الخاص
بالعميل عبر OAuth — يُضاف لاحقاً بعد تسجيل تطبيق مطوّر على
developers.pinterest.com):

  1. ImageSource.UPLOADED   — صورة يرفعها العميل مباشرة (bytes) — لا API خارجي.
  2. ImageSource.STOCK      — بحث في مكتبات مرخّصة تجارياً (Pixabay أولاً،
                              Openverse احتياطي إن فشل Pixabay أو لم يوجد مفتاح).
  3. ImageSource.STYLE_INSPIRED — يأخذ رابط صورة مرجعية (مثلاً من Pinterest)
                              **كمرجع أسلوب فقط**، يحلّله عبر Gemini Vision
                              (لون/تكوين/إحساس عام)، ثم يعيد وصفاً نصياً
                              يُستخدم لاحقاً لتوليد صورة أصلية جديدة — لا يُنسخ
                              أي بكسل من الصورة المرجعية ولا يُعاد استخدامها
                              مباشرة في الناتج النهائي.

متغيرات البيئة المطلوبة (اختيارية — كل مصدر يعمل بدون البقية):
    PIXABAY_API_KEY   — https://pixabay.com/api/docs/ (تسجيل مجاني بدون بطاقة)
    GOOGLE_API_KEY    — نفس المفتاح المستخدم في ai/higgsfield_engine.py لـ Gemini
"""

from __future__ import annotations

import base64
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from ai.offline_mode import is_offline, offline_message

logger = logging.getLogger("ImageSources")

_TIMEOUT = 15
_GEMINI_VISION_MODEL = "gemini-2.0-flash"


class ImageSource(str, Enum):
    UPLOADED = "uploaded"
    STOCK = "stock"
    STYLE_INSPIRED = "style_inspired"


@dataclass
class ImageResult:
    source: ImageSource
    # لمصدر UPLOADED وSTOCK: رابط/بيانات صورة فعلية جاهزة للاستخدام
    url: Optional[str] = None
    image_bytes: Optional[bytes] = None
    # لمصدر STOCK: نسب الفضل الواجب (بعض التراخيص المجانية تتطلبه أخلاقياً)
    attribution: Optional[str] = None
    license_name: Optional[str] = None
    # لمصدر STYLE_INSPIRED: لا صورة فعلية، بل وصف نصي يُستخدم للتوليد لاحقاً
    style_description: Optional[str] = None
    error: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════
# 1) صور مرفوعة من العميل — لا يحتاج أي API خارجي
# ══════════════════════════════════════════════════════════════════════════

def from_upload(image_bytes: bytes, filename: str = "") -> ImageResult:
    """يغلّف صورة رفعها العميل مباشرة كـ ImageResult موحّد."""
    if not image_bytes:
        return ImageResult(source=ImageSource.UPLOADED, error="لا توجد بيانات صورة")
    return ImageResult(
        source=ImageSource.UPLOADED,
        image_bytes=image_bytes,
        attribution=filename or None,
    )


# ══════════════════════════════════════════════════════════════════════════
# 2) مكتبات صور مرخّصة تجارياً — Pixabay أولاً، Openverse احتياطي
# ══════════════════════════════════════════════════════════════════════════

def _search_pixabay(query: str, per_page: int = 5) -> List[ImageResult]:
    key = os.getenv("PIXABAY_API_KEY", "").strip()
    if not key:
        return []
    params = urllib.parse.urlencode({
        "key": key,
        "q": query,
        "image_type": "photo",
        "safesearch": "true",
        "per_page": max(3, min(per_page, 20)),
    })
    url = f"https://pixabay.com/api/?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NSM-ImageSources/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("فشل بحث Pixabay: %s", exc)
        return []

    results: List[ImageResult] = []
    for hit in data.get("hits", []):
        results.append(ImageResult(
            source=ImageSource.STOCK,
            url=hit.get("largeImageURL") or hit.get("webformatURL"),
            attribution=f"صورة من {hit.get('user', 'Pixabay')} عبر Pixabay",
            license_name="Pixabay License (استخدام تجاري حر)",
        ))
    return results


def _search_openverse(query: str, per_page: int = 5) -> List[ImageResult]:
    """احتياطي بلا مفتاح API — Openverse (مشروع Creative Commons)."""
    params = urllib.parse.urlencode({
        "q": query,
        "page_size": max(3, min(per_page, 20)),
        # نستبعد صراحة أي ترخيص يمنع الاستخدام التجاري أو الأعمال المشتقة
        "license_type": "commercial,modification",
    })
    url = f"https://api.openverse.org/v1/images/?{params}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "NSM-ImageSources/1.0"})
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        logger.warning("فشل بحث Openverse: %s", exc)
        return []

    results: List[ImageResult] = []
    for item in data.get("results", []):
        results.append(ImageResult(
            source=ImageSource.STOCK,
            url=item.get("url"),
            attribution=f"{item.get('title', 'صورة')} — {item.get('creator', 'مجهول')} (Openverse)",
            license_name=item.get("license", "").upper() or "Creative Commons",
        ))
    return results


def search_stock_images(query: str, per_page: int = 5) -> List[ImageResult]:
    """
    يبحث عن صور مرخّصة تجارياً بكلمات مفتاحية. يجرّب Pixabay أولاً (جودة
    أعلى عادة)، وإن لم يوجد مفتاح أو فشل البحث ينتقل تلقائياً لـ Openverse
    (لا يحتاج مفتاح API إطلاقاً).
    """
    if is_offline():
        logger.info(offline_message("بحث الصور (Pixabay/Openverse)"))
        return []
    results = _search_pixabay(query, per_page)
    if results:
        return results
    return _search_openverse(query, per_page)


# ══════════════════════════════════════════════════════════════════════════
# 3) إلهام الأسلوب — تحليل صورة مرجعية (مثلاً من Pinterest) عبر Gemini
#    Vision، بدون نسخ الصورة نفسها — الناتج وصف نصي فقط يُستخدم للتوليد
# ══════════════════════════════════════════════════════════════════════════

_STYLE_ANALYSIS_PROMPT = (
    "حلّل الأسلوب البصري لهذه الصورة فقط (بدون وصف محتواها الحرفي بتفصيل "
    "قد يُنسخ): الألوان السائدة، نوع الإضاءة، التكوين (composition)، "
    "المزاج العام (vibe)، ونمط التصميم (مثلاً: مينيمال، عصري، كلاسيكي). "
    "أعد وصفاً نصياً موجزاً بالعربية (3-4 جمل) يُستخدم كموجّه أسلوب "
    "(style guide) لتوليد تصميم موشن جرافيك أصلي جديد بنفس الروح — "
    "لا تصف عناصر قابلة للتماثل الحرفي (مثل نص محدد أو شعار محدد ظاهر في "
    "الصورة)."
)


def analyze_style_from_url(image_url: str) -> ImageResult:
    """
    يأخذ رابط صورة مرجعية (مثلاً pin من Pinterest يعطيه العميل كمرجع
    أسلوب) ويعيد وصفاً نصياً للأسلوب البصري عبر Gemini Vision — لا يُنزّل
    أو يُخزّن أو يعيد استخدام بكسلات الصورة نفسها في أي ناتج نهائي.
    """
    if is_offline():
        return ImageResult(
            source=ImageSource.STYLE_INSPIRED,
            error=offline_message("تحليل الأسلوب البصري (Gemini Vision)"),
        )

    key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return ImageResult(
            source=ImageSource.STYLE_INSPIRED,
            error="GOOGLE_API_KEY غير مضبوط — لا يمكن تحليل الأسلوب",
        )

    try:
        img_req = urllib.request.Request(image_url, headers={"User-Agent": "NSM-ImageSources/1.0"})
        with urllib.request.urlopen(img_req, timeout=_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "image/jpeg").split(";")[0]
            image_bytes = resp.read()
    except (urllib.error.URLError, TimeoutError) as exc:
        return ImageResult(
            source=ImageSource.STYLE_INSPIRED,
            error=f"تعذّر جلب الصورة المرجعية: {exc}",
        )

    b64_image = base64.b64encode(image_bytes).decode("ascii")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{_GEMINI_VISION_MODEL}:generateContent?key={key}"
    )
    payload = {
        "contents": [{
            "parts": [
                {"text": _STYLE_ANALYSIS_PROMPT},
                {"inline_data": {"mime_type": content_type, "data": b64_image}},
            ]
        }]
    }
    try:
        req = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        description = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
        if not description:
            return ImageResult(
                source=ImageSource.STYLE_INSPIRED,
                error="رد Gemini لم يحتوِ وصفاً قابلاً للاستخدام",
            )
        return ImageResult(source=ImageSource.STYLE_INSPIRED, style_description=description)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, IndexError, KeyError) as exc:
        return ImageResult(
            source=ImageSource.STYLE_INSPIRED,
            error=f"فشل تحليل الأسلوب عبر Gemini Vision: {exc}",
        )


# ══════════════════════════════════════════════════════════════════════════
# واجهة موحّدة اختيارية — يستخدمها محرك الموشن جرافيك لاحقاً
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class ImageSourcesConfig:
    """يلخّص أي مصادر جاهزة للعمل فعلياً بحسب متغيرات البيئة المتوفرة."""
    pixabay_available: bool = field(default_factory=lambda: bool(os.getenv("PIXABAY_API_KEY", "").strip()))
    openverse_available: bool = True  # لا يحتاج مفتاح
    style_analysis_available: bool = field(default_factory=lambda: bool(os.getenv("GOOGLE_API_KEY", "").strip()))


def get_config() -> ImageSourcesConfig:
    return ImageSourcesConfig()
