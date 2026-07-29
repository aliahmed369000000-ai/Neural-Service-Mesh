"""
NSM Platform Profiles — ai/platform_profiles.py
================================================
المرحلة 5 من تطوير الوكيل الاجتماعي: تكييف المحتوى لكل منصة بدل نشر نفس
النص بنفس الشكل في كل مكان (وهو ما كان يفعله content_agent.py سابقاً عبر
تشويقة واحدة تُنشر حرفياً على كل المنصات).

كل منصة لها حد أقصى للأحرف وعُرف مختلف لعدد الهاشتاقات المناسب. هذه
الحدود تقريبية ومبنية على الممارسات المعروفة لكل منصة، وقابلة للتعديل هنا
فقط دون المساس بمنطق النشر في social_agent.py.

⚠️ لا اختلاق: لو طُلبت منصة غير معرّفة هنا، تُستخدم قيم افتراضية محافظة
(280 حرفاً، هاشتاقين) بدل رفع استثناء يُعطّل خط الأنابيب كاملاً.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class PlatformProfile:
    max_chars: int
    hashtag_count: int  # عدد الهاشتاقات المُضافة تلقائياً في نهاية النص


_DEFAULT_PROFILE = PlatformProfile(max_chars=280, hashtag_count=2)

PLATFORM_PROFILES: Dict[str, PlatformProfile] = {
    "twitter":    PlatformProfile(max_chars=280,   hashtag_count=2),
    "threads":    PlatformProfile(max_chars=500,   hashtag_count=2),
    "facebook":   PlatformProfile(max_chars=600,   hashtag_count=3),
    "instagram":  PlatformProfile(max_chars=2200,  hashtag_count=6),
    "linkedin":   PlatformProfile(max_chars=700,   hashtag_count=4),
    "tiktok":     PlatformProfile(max_chars=2200,  hashtag_count=5),
    "youtube":    PlatformProfile(max_chars=1000,  hashtag_count=3),
    "reddit":     PlatformProfile(max_chars=1000,  hashtag_count=0),
    "discord":    PlatformProfile(max_chars=2000,  hashtag_count=0),
    "telegram":   PlatformProfile(max_chars=1000,  hashtag_count=2),
    "whatsapp":   PlatformProfile(max_chars=1000,  hashtag_count=0),
    "pinterest":  PlatformProfile(max_chars=500,   hashtag_count=4),
}


def get_profile(platform: str) -> PlatformProfile:
    return PLATFORM_PROFILES.get(platform, _DEFAULT_PROFILE)


def _clean_hashtag(tag: str) -> str:
    return "#" + tag.strip().lstrip("#").replace(" ", "_")


def adapt_text_for_platform(base_text: str, platform: str,
                             hashtags: Optional[List[str]] = None) -> str:
    """يبني نصاً مناسباً لمنصة واحدة: يقتطع النص الأساسي عند الحاجة ثم يضيف
    عدد الهاشتاقات المناسب لهذه المنصة (وليس كل الهاشتاقات دائماً)، مع
    ضمان عدم تجاوز الحد الأقصى للأحرف إطلاقاً."""
    profile = get_profile(platform)
    hashtags = hashtags or []
    chosen_tags = [_clean_hashtag(t) for t in hashtags[:profile.hashtag_count] if t.strip()]
    tag_block = ("\n\n" + " ".join(chosen_tags)) if chosen_tags else ""

    available = profile.max_chars - len(tag_block)
    text = base_text
    if available < 1:
        # حتى الهاشتاقات وحدها لا تتسع — تخلَّ عنها بدل نص فارغ أو مكسور
        tag_block = ""
        available = profile.max_chars

    if len(text) > available:
        text = text[:max(available - 1, 0)].rstrip() + "…"

    return (text + tag_block).strip()


def adapt_text_for_platforms(base_text: str, platforms: List[str],
                              hashtags: Optional[List[str]] = None) -> Dict[str, str]:
    """يبني قاموس {platform: نص مُكيَّف} لكل منصة في القائمة، بدل نص واحد
    مطابق يُقصّ بنفس الطريقة للجميع."""
    return {p: adapt_text_for_platform(base_text, p, hashtags) for p in platforms}
