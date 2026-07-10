---
name: Nova System Architecture
description: بنية نظام Nova المُوزَّعة على وحدات Python متعددة، ودمجها في app.py
---

# بنية نظام Nova

## الوحدات

| الوحدة | المحتوى | الجزء |
|--------|---------|-------|
| `ai/nova_system.py` | NOVA_SYSTEM_PROMPT، فحوصات السلامة، نظام الذاكرة، رفاهية المستخدم، معلومات المنتج | part1 |
| `ai/nova_memory_prefs.py` | تطبيق الذاكرة، Storage API، MCP Apps، تفضيلات المستخدم، memory_user_edits | part2 |
| `ai/nova_search_copyright.py` | قواعد البحث، حقوق النشر (15 كلمة، 1 اقتباس/مصدر) | part3 |
| `ai/nova_tools_registry.py` | مخططات الأدوات (web_search, image_search, places_*, recipe_*, etc.) | parts 4-6 |

## نقطة التجميع

`build_full_nova_prompt()` في `nova_system.py` تجمع System Prompt من جميع الأجزاء 1-6 بشكل آمن مع try/except.

## الدمج في app.py

- استيراد Nova في قسم imports (مع graceful fallback إذا فشل)
- شخصية "nova" أُضيفت أول PERSONAS dict بـ emoji ✨
- فحص السلامة يعمل **قبل** إرسال الرسالة للـ API
- ترتيب الفحوصات: كلمات أغاني → سلامة عامة → طلب رعاية نفسية → موضوع سياسي
- ذيل الرفاهية يُضاف بعد رد المساعد عند الحاجة

**Why:** الاستيراد الفاشل لـ nova_system.py يمنع بناء PERSONAS بالكامل — لذا كل استيراد محاط بـ try/except.
**How to apply:** عند تعديل nova_system.py، تحقق أن `NOVA_SYSTEM_PROMPT` يُعرَّف قبل أي استيرادات من الوحدات الفرعية.
