"""
Social Platforms — محولات المنصات لوكيل SOCIAL AGENT الموحد
================================================================
كل محول (adapter) يطبّق واجهة PlatformAdapter الموجودة في base.py:
publish() / fetch_new_items() / reply() / is_configured().

المحولات لا تصنع بيانات مزيّفة أبداً — إن لم تتوفر بيانات الاعتماد
اللازمة (رمز API/رمز بوت)، ترفع NotConfiguredError بوضوح بدل تلفيق نتائج.
"""

from .base import PlatformAdapter, SocialItem, NotConfiguredError
from .discord_adapter import DiscordAdapter
from .telegram_adapter import TelegramAdapter
from .twitter_adapter import TwitterAdapter
from .instagram_adapter import InstagramAdapter
from .facebook_adapter import FacebookAdapter
from .youtube_adapter import YouTubeAdapter

ALL_ADAPTERS = {
    "discord": DiscordAdapter,
    "telegram": TelegramAdapter,
    "twitter": TwitterAdapter,
    "instagram": InstagramAdapter,
    "facebook": FacebookAdapter,
    "youtube": YouTubeAdapter,
}

PLATFORM_LABELS_AR = {
    "discord": "🎮 Discord",
    "telegram": "✈️ Telegram",
    "twitter": "𝕏 Twitter/X",
    "instagram": "📷 Instagram",
    "facebook": "📘 Facebook",
    "youtube": "▶️ YouTube",
}

__all__ = [
    "PlatformAdapter", "SocialItem", "NotConfiguredError",
    "DiscordAdapter", "TelegramAdapter", "TwitterAdapter",
    "InstagramAdapter", "FacebookAdapter", "YouTubeAdapter",
    "ALL_ADAPTERS", "PLATFORM_LABELS_AR",
]
