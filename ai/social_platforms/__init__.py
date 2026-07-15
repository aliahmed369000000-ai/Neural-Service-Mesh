"""
Social Platforms — محولات المنصات لوكيل SOCIAL AGENT الموحد
================================================================
كل محول (adapter) يطبّق واجهة PlatformAdapter الموجودة في base.py:
publish() / fetch_new_items() / reply() / is_configured().

المحولات لا تصنع بيانات مزيّفة أبداً — إن لم تتوفر بيانات الاعتماد
اللازمة (رمز API/رمز بوت)، ترفع NotConfiguredError بوضوح بدل تلفيق نتائج.

قيود حقيقية موثَّقة صراحة بدل التحايل عليها (راجع WEBHOOKS.md للتفصيل):
- `supports_webhook`: تيليجرام وواتساب فقط (setWebhook/Meta webhooks
  حقيقيان وموثّقان). واتساب تحديداً webhook إلزامي (لا بديل REST من Meta).
- `supports_monitoring = False`: Pinterest — النشر (publish) يعمل، لكن
  API v5 لا يوفّر أي endpoint لقراءة/الرد على التعليقات إطلاقاً.
- Snapchat: لا يوجد محول أصلاً — لا مسار API علني ينشر محتوى عضوياً أو
  يراقب/يرد نيابة عن حساب (راجع قسم Snapchat بأسفل WEBHOOKS.md).
"""

from .base import PlatformAdapter, SocialItem, NotConfiguredError, PlatformCapabilityError
from .discord_adapter import DiscordAdapter
from .telegram_adapter import TelegramAdapter
from .instagram_adapter import InstagramAdapter
from .facebook_adapter import FacebookAdapter
from .youtube_adapter import YouTubeAdapter
from .tiktok_adapter import TikTokAdapter
from .reddit_adapter import RedditAdapter
from .threads_adapter import ThreadsAdapter
from .whatsapp_adapter import WhatsAppAdapter
from .pinterest_adapter import PinterestAdapter

# ملاحظة: تمّ تعطيل Twitter/X وLinkedIn من القائمة النشطة (2026-07-15)
# بسبب تكلفة API غير مجانية. ملفا twitter_adapter.py وlinkedin_adapter.py
# ما زالا موجودين في المجلد (غير محذوفين) لإعادة التفعيل لاحقاً لو لزم.

ALL_ADAPTERS = {
    "discord": DiscordAdapter,
    "telegram": TelegramAdapter,
    "instagram": InstagramAdapter,
    "facebook": FacebookAdapter,
    "youtube": YouTubeAdapter,
    "tiktok": TikTokAdapter,
    "reddit": RedditAdapter,
    "threads": ThreadsAdapter,
    "whatsapp": WhatsAppAdapter,
    "pinterest": PinterestAdapter,
}

PLATFORM_LABELS_AR = {
    "discord": "🎮 Discord",
    "telegram": "✈️ Telegram",
    "instagram": "📷 Instagram",
    "facebook": "📘 Facebook",
    "youtube": "▶️ YouTube",
    "tiktok": "🎵 TikTok",
    "reddit": "👽 Reddit",
    "threads": "🧵 Threads",
    "whatsapp": "💬 WhatsApp",
    "pinterest": "📌 Pinterest",
}

#: أقصى طول نص معروف موثَّق لكل منصة (لتحذير المستخدم بالواجهة *قبل*
#: النشر بدل اكتشاف فشل النشر بعد إرساله). None = لا حد عملي معروف/غير
#: قابل للتطبيق. أرقام تقريبية موثّقة من كل منصة وقد تتغيّر — تحذير فقط،
#: ليست تحققاً صارماً (المنصة نفسها هي الحكم النهائي عند النشر الفعلي).
PLATFORM_CHAR_LIMITS = {
    "threads": 500,
    "discord": 2000,
    "telegram": 4096,
    "whatsapp": 4096,
    "facebook": None,
    "instagram": 2200,
    "youtube": None,
    "tiktok": 2200,
    "reddit": None,
    "pinterest": 100,  # العنوان (title) فقط — الوصف الكامل يُستخدم كـdescription بلا حد صارم
}

__all__ = [
    "PlatformAdapter", "SocialItem", "NotConfiguredError", "PlatformCapabilityError",
    "DiscordAdapter", "TelegramAdapter",
    "InstagramAdapter", "FacebookAdapter", "YouTubeAdapter", "TikTokAdapter",
    "RedditAdapter", "ThreadsAdapter",
    "WhatsAppAdapter", "PinterestAdapter",
    "ALL_ADAPTERS", "PLATFORM_LABELS_AR", "PLATFORM_CHAR_LIMITS",
]
