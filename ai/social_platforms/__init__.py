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
from .twitter_adapter import TwitterAdapter
from .instagram_adapter import InstagramAdapter
from .facebook_adapter import FacebookAdapter
from .youtube_adapter import YouTubeAdapter
from .tiktok_adapter import TikTokAdapter
from .reddit_adapter import RedditAdapter
from .linkedin_adapter import LinkedInAdapter
from .threads_adapter import ThreadsAdapter
from .whatsapp_adapter import WhatsAppAdapter
from .pinterest_adapter import PinterestAdapter

ALL_ADAPTERS = {
    "discord": DiscordAdapter,
    "telegram": TelegramAdapter,
    "twitter": TwitterAdapter,
    "instagram": InstagramAdapter,
    "facebook": FacebookAdapter,
    "youtube": YouTubeAdapter,
    "tiktok": TikTokAdapter,
    "reddit": RedditAdapter,
    "linkedin": LinkedInAdapter,
    "threads": ThreadsAdapter,
    "whatsapp": WhatsAppAdapter,
    "pinterest": PinterestAdapter,
}

PLATFORM_LABELS_AR = {
    "discord": "🎮 Discord",
    "telegram": "✈️ Telegram",
    "twitter": "𝕏 Twitter/X",
    "instagram": "📷 Instagram",
    "facebook": "📘 Facebook",
    "youtube": "▶️ YouTube",
    "tiktok": "🎵 TikTok",
    "reddit": "👽 Reddit",
    "linkedin": "💼 LinkedIn",
    "threads": "🧵 Threads",
    "whatsapp": "💬 WhatsApp",
    "pinterest": "📌 Pinterest",
}

__all__ = [
    "PlatformAdapter", "SocialItem", "NotConfiguredError", "PlatformCapabilityError",
    "DiscordAdapter", "TelegramAdapter", "TwitterAdapter",
    "InstagramAdapter", "FacebookAdapter", "YouTubeAdapter", "TikTokAdapter",
    "RedditAdapter", "LinkedInAdapter", "ThreadsAdapter",
    "WhatsAppAdapter", "PinterestAdapter",
    "ALL_ADAPTERS", "PLATFORM_LABELS_AR",
]
