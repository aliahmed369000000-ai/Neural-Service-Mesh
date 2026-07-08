---
name: G0DM0D3 Streamlit port
description: تفاصيل تحويل G0DM0D3 من Next.js إلى Streamlit
---

## القرار
المستخدم يريد Streamlit فقط — لا React/Vite.

## البنية
- `app.py` — التطبيق الكامل (models, personas, themes CSS, OpenRouter streaming)
- `requirements.txt` — streamlit, requests فقط
- `.streamlit/config.toml` — منسوخ من migration-backup، port=5000
- Workflow: `G0DM0DE` → `streamlit run app.py --server.port 5000`

**Why:** المستخدم صريح أنه يعمل فقط في Streamlit.

**How to apply:** أي تعديل مستقبلي يكون في `app.py` فقط، لا تُنشئ artifacts من نوع react-vite.

## مشاكل تم حلها
- `streamlit: command not found` → تثبيت `python-3.11` عبر installProgrammingLanguage ثم installLanguagePackages
- non-200 JSON parse error في stream_response → hardened to try/except fallback to resp.text
