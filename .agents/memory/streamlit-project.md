---
name: Streamlit Project Structure
description: هذا المشروع Python/Streamlit وليس pnpm — الكود الحقيقي في جذر workspace
---

# بنية المشروع

## الحقيقة

المشروع هو **Python/Streamlit** وليس Node.js/pnpm.
Replit importer أنشأ scaffold pnpm في البداية لكنه غير مستخدم.

## الملفات الرئيسية

- `app.py` — Streamlit entry point (1400+ سطر)
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — إعدادات Streamlit
- `ai/` — وحدات Python
- `.migration-backup/` — الملفات الأصلية (للقراءة فقط، لا تحذف)

## Workflow

- الاسم: "Start application"
- الأمر: `pip install -r requirements.txt -q && streamlit run app.py --server.port 5000 --server.headless true --server.address 0.0.0.0`
- المنفذ: 5000

## Python path

Python متاح عبر Streamlit's bundled env لكن `python`/`python3` غير متاح مباشرة من bash.

**Why:** المشروع مُهاجَر من بيئة Python أصلية (nova.ai) إلى Replit.
**How to apply:** استخدم `pip install` دائماً قبل تشغيل التطبيق إذا أضفت مكتبات جديدة.
