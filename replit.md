# Neural Service Mesh (NSM) — نظام معرفي عربي

## نظرة عامة
NSM هو نظام ذكاء اصطناعي عربي متكامل مبني على Streamlit، يجمع بين:
- **NSM Agent** (وكيل ذكي للمهام البرمجية)
- **LLM Fallback** (دعم متعدد المزوّدين: Anthropic, Cloudflare, Gemini, Groq, OpenAI, ...)
- **Cognitive Knowledge Graph** (رسم معرفي مدرَّب)
- **ConversationMemory** (ذاكرة محادثة متعددة الأدوار)

## تشغيل المشروع
```bash
streamlit run streamlit_app.py
```
المنفذ: 5000

## الملفات الرئيسية
| الملف | الوظيفة |
|-------|---------|
| `streamlit_app.py` | واجهة Streamlit الرئيسية |
| `ai/llm_fallback.py` | محرك LLM متعدد المزوّدين |
| `ai/anthropic_advanced.py` | واجهة Anthropic API المتقدمة |
| `nsm_chat_plus.py` | طبقة المحادثة الذكية |
| `nsm_chat.py` | NSMChat الأصلي |
| `nsm_memory.py` | ذاكرة المحادثة |
| `That.md` | Claude.ai System Prompt (مرجع API) |

## متغيرات البيئة (Secrets)
| المتغير | الوظيفة |
|---------|---------|
| `ANTHROPIC_API_KEY` | Anthropic Claude — الأولوية الأولى |
| `CF_API_TOKEN` | Cloudflare Workers AI (مجاني 10k/يوم) |
| `CF_ACCOUNT_ID` | معرّف حساب Cloudflare |
| `GOOGLE_API_KEY` | Google Gemini |
| `OPENROUTER_API_KEY` | OpenRouter |
| `GROQ_API_KEY` | Groq |
| `OPENAI_API_KEY` | OpenAI |
| `TOGETHER_API_KEY` | Together.xyz |

## نماذج Anthropic المتاحة (2026)
| الاسم | الكود | الاستخدام |
|-------|-------|-----------|
| Sonnet 4 | `claude-sonnet-4-6` | الافتراضي — توازن مثالي |
| Opus 4 | `claude-opus-4-8` | المهام المعقدة |
| Haiku 4 | `claude-haiku-4-5-20251001` | الردود السريعة |
| Sonnet 4 Stable | `claude-sonnet-4-20250514` | الإنتاج المستقر |

المصدر: Claude.ai System Prompt export (That.md)

## تفضيلات المستخدم
- اللغة العربية الفصحى في كل الردود
- الحفاظ على بنية الملفات الموجودة
