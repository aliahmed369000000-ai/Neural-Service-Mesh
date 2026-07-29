"""
NSM Content Agent — ai/content_agent.py
===========================================
اللبنة الثالثة والأخيرة من "وكيل صناعة المحتوى". يربط:
  1) ai/web_search_tool.py  → get_trending_topics()      (اكتشاف الترند)
  2) ai/content_writer.py   → generate_seo_article()      (كتابة SEO)
  3) ai/social_agent.py     → schedule_post() / publish_to() (النشر)
في خط أنابيب واحد قابل للاستدعاء البرمجي المباشر، أو من فئة الوكلاء
"content" في ai/agent_categories.py.

⚠️ ملاحظة صادقة عن حدود المنصات: منشورات السوشيال ميديا (Twitter/X خصوصاً)
لها حد أقصى لطول النص. هذا الملف لا ينشر المقال الكامل كمنشور واحد أبداً —
ينشر "تشويقة" (teaser) قصيرة (العنوان + الوصف التعريفي + هاشتاقين)، بينما
المقال الكامل يُرجَع كـ Markdown في النتيجة لمراجعته أو نشره يدوياً على
مدونة (المشروع لا يملك منصة مدونة مدمجة حالياً).

⚠️ لا نشر صامت: لو platforms فارغة، خط الأنابيب يتوقف بعد كتابة المقال
(publish_mode="skipped") — النشر التلقائي الفعلي يتطلب طلباً صريحاً بقائمة
منصات محدَّدة.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

from ai.web_search_tool import get_trending_topics
from ai.content_writer import SEOArticle, generate_seo_article
from ai.platform_profiles import adapt_text_for_platforms

logger = logging.getLogger("ContentAgent")

try:
    from ai.social_agent import schedule_post, get_manager
    _SOCIAL_OK = True
except Exception:
    schedule_post = None
    get_manager = None
    _SOCIAL_OK = False

try:
    from ai.agent_audit import get_default_audit_log
    _AUDIT_OK = True
except Exception:
    get_default_audit_log = None
    _AUDIT_OK = False


_MAX_TEASER_CHARS = 260  # هامش أمان تحت حد Twitter/X (280 حرفاً)


@dataclass
class ContentPipelineResult:
    topic:          str
    geo:            str
    article:        Optional[SEOArticle]
    teaser:         str
    per_platform_text: Dict[str, str] = field(default_factory=dict)
    platforms:      List[str] = field(default_factory=list)
    publish_mode:   str = "skipped"          # "scheduled" | "published" | "skipped"
    publish_result: Dict[str, str] = field(default_factory=dict)
    schedule_id:    Optional[int] = None
    errors:         List[str] = field(default_factory=list)


def _build_teaser(article: SEOArticle) -> str:
    """يبني منشوراً قصيراً (تشويقة) من المقال — وليس المقال كاملاً، لأن
    منصات مثل Twitter/X لها حد أقصى لطول النص."""
    hashtags = " ".join(f"#{kw.strip().replace(' ', '_')}" for kw in article.keywords[:2] if kw.strip())
    text = article.title
    if article.meta_description:
        text += f"\n\n{article.meta_description}"
    if hashtags:
        text += f"\n\n{hashtags}"
    if len(text) > _MAX_TEASER_CHARS:
        text = text[:_MAX_TEASER_CHARS - 1].rstrip() + "…"
    return text


def _pick_topic(geo: str) -> Tuple[str, Optional[str]]:
    """يختار أول موضوع رائج حقيقي فعلياً (بدون اختلاق)، ويعيد (topic, context)."""
    trends = get_trending_topics(geo=geo, max_results=5)
    if not trends:
        raise RuntimeError(f"لا توجد مواضيع رائجة متاحة حالياً للمنطقة '{geo}'")
    top = trends[0]
    context_parts: List[str] = []
    if top.get("news_title"):
        context_parts.append(top["news_title"])
    if top.get("traffic"):
        context_parts.append(f"(حجم بحث تقريبي: {top['traffic']})")
    context = " ".join(context_parts) if context_parts else None
    return top["title"], context


def run_content_pipeline(
    geo: str = "SA",
    topic: Optional[str] = None,
    platforms: Optional[List[str]] = None,
    auto_publish: bool = False,
    scheduled_at: Optional[str] = None,
    target_words: int = 700,
) -> ContentPipelineResult:
    """
    خط الأنابيب الكامل: اكتشاف ترند (أو موضوع مُعطى مباشرة) → كتابة مقال
    SEO → بناء تشويقة → نشرها فوراً أو جدولتها.

    topic:        لو مُعطى، يتخطى اكتشاف الترند ويكتب مباشرة عن هذا الموضوع.
    platforms:    قائمة معرّفات منصات (twitter, telegram, ...). لو فارغة/None
                  لا يُنشر ولا يُجدول أي شيء — يُرجع المقال والتشويقة فقط
                  (وضع "مراجعة قبل النشر").
    auto_publish: True = نشر فوري عبر publish_to؛ False (افتراضي) = جدولة
                  عبر schedule_post بدل النشر الفوري المباشر.
    scheduled_at: ISO 8601؛ لو لم يُحدَّد يُستخدَم بعد ساعة من الآن (UTC).
    """
    used_topic = topic
    context: Optional[str] = None

    if not used_topic:
        used_topic, context = _pick_topic(geo)

    article = generate_seo_article(used_topic, context=context, target_words=target_words)
    teaser = _build_teaser(article)

    platforms = platforms or []
    publish_mode = "skipped"
    publish_result: Dict[str, str] = {}
    schedule_id: Optional[int] = None
    errors: List[str] = []

    # تكييف المحتوى لكل منصة (المرحلة 5): بدل نشر التشويقة الموحّدة حرفياً
    # في كل مكان، نبني نصاً مقتطعاً بحد أحرف وعدد هاشتاقات مناسب لكل
    # منصة على حدة (twitter قصير جداً، instagram/tiktok أطول وبهاشتاقات
    # أكثر، ...). التشويقة الموحّدة (teaser) تبقى القيمة المُرجعة
    # للمراجعة، لكن ما يُنشر فعلياً هو النسخة المكيّفة لكل منصة.
    per_platform_text: Dict[str, str] = (
        adapt_text_for_platforms(f"{article.title}\n\n{article.meta_description}".strip(),
                                  platforms, article.keywords)
        if platforms else {}
    )

    if platforms:
        if not _SOCIAL_OK:
            errors.append("وحدة social_agent غير متاحة — تعذّر النشر/الجدولة")
        elif auto_publish:
            try:
                publish_result = get_manager().publish_to(
                    platforms, teaser, per_platform_text=per_platform_text)
                publish_mode = "published"
            except Exception as e:
                errors.append(f"فشل النشر الفوري: {e}")
        else:
            when = scheduled_at or (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat()
            try:
                schedule_id = schedule_post(platforms, teaser, when)
                publish_mode = "scheduled"
            except Exception as e:
                errors.append(f"فشل الجدولة: {e}")

    result = ContentPipelineResult(
        topic=used_topic,
        geo=geo,
        article=article,
        teaser=teaser,
        per_platform_text=per_platform_text,
        platforms=platforms,
        publish_mode=publish_mode,
        publish_result=publish_result,
        schedule_id=schedule_id,
        errors=errors,
    )
    _log_pipeline(result)
    return result


def _log_pipeline(result: ContentPipelineResult) -> None:
    """يسجّل نتيجة خط الأنابيب في سجل التدقيق — لا يرفع استثناء أبداً
    (فشل التدقيق لا يجب أن يُعطّل خط الأنابيب)."""
    if not _AUDIT_OK or get_default_audit_log is None:
        return
    try:
        extra: Dict[str, Any] = {
            "geo": result.geo,
            "publish_mode": result.publish_mode,
            "platforms": result.platforms,
            "schedule_id": result.schedule_id,
            "errors": result.errors,
        }
        if result.article:
            extra["seo_score"] = result.article.seo_score
            extra["structured"] = result.article.structured
            extra["word_count"] = result.article.word_count

        get_default_audit_log().log_event(
            category_key="content",
            category_title="صناعة المحتوى",
            source="pipeline",
            question=f"موضوع: {result.topic}",
            response=result.teaser,
            provider=(result.article.provider if result.article else None),
            web_used=True,
            extra=extra,
        )
    except Exception:
        pass
