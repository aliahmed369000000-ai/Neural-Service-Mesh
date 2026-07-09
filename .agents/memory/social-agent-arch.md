---
name: Social Agent Architecture
description: Process-wide background social agent for 6 platforms — key design decisions and credentials map
---

# Social Agent — دروس معمارية

## بنية الملفات
- `ai/social_agent.py` — singleton manager, background thread, DB, reply via GODMODE/OpenRouter
- `ai/social_platforms/` — محول per-platform يطبّق `PlatformAdapter` (publish/fetch_new_items/reply/is_configured)

## قرارات مهمة
**Why process-global singleton:** Streamlit يُعيد التشغيل per-session؛ الخيط الخلفي يجب أن يعيش على مستوى العملية لا الجلسة.

**How to apply:** `get_manager()` يستدعي `SocialAgentManager.instance()` — دائماً singleton واحد. لا تُنشئ instance جديداً أبداً.

**DB:** `memory/social_agent.db` — WAL mode + busy_timeout=5000ms — منفصل عن conversations.db.

## بيانات الاعتماد المطلوبة (secrets)
| المنصة | Secrets المطلوبة | ملاحظة |
|--------|-----------------|--------|
| Discord | `DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID` | اختياري: `DISCORD_BOT_USER_ID` |
| Telegram | `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | polling فقط |
| X/Twitter | `TWITTER_API_KEY`, `TWITTER_API_SECRET`, `TWITTER_ACCESS_TOKEN`, `TWITTER_ACCESS_TOKEN_SECRET` | يتطلب خطة X API مدفوعة للنشر |
| Instagram | `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_BUSINESS_ACCOUNT_ID` | النشر نصي غير مدعوم — صورة/فيديو فقط |
| Facebook | `FACEBOOK_PAGE_ACCESS_TOKEN`, `FACEBOOK_PAGE_ID` | Page Access Token طويل الأمد |
| YouTube | `YOUTUBE_API_KEY`, `YOUTUBE_CHANNEL_ID` (قراءة) + `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_OAUTH_REFRESH_TOKEN` (كتابة) | |

## نقاط لا للمساومة
- **لا بيانات مزيّفة** — أي منصة بلا credentials ترفع `NotConfiguredError` صريحة.
- الرد التلقائي يمر عبر نفس `OPENROUTER_API_KEY` + `GODMODE_SYSTEM_PROMPT` — ليس نظاماً منفصلاً.
- `requests_oauthlib` مطلوبة لـ Twitter OAuth1 — موجودة في requirements.txt.
