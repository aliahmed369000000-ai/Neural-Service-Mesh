# G0DM0D3 — Neural Service Mesh

تطبيق دردشة ذكاء اصطناعي مبني بـ Python/Streamlit يدعم شخصيات متعددة أبرزها Nova (مساعد Aurora Labs) مع نظام سلامة شامل.

## Run & Operate

- `streamlit run app.py --server.port 5000 --server.headless true --server.address 0.0.0.0` — تشغيل التطبيق
- `pip install -r requirements.txt` — تثبيت المتطلبات
- `bash push_to_github.sh "رسالة الـ commit"` — دفع التغييرات لـ GitHub

## Stack

- Python 3.12 + Streamlit
- OpenRouter API (بث الاستجابات)
- SQLite (تاريخ المحادثات)
- GitHub: aliahmed369000000-ai/Neural-Service-Mesh

## Where things live

- `app.py` — نقطة الدخول الرئيسية (Streamlit)
- `ai/nova_system.py` — نظام Nova: part1 (معلومات المنتج، السلامة، النبرة، الرفاهية، الذاكرة)
- `ai/nova_memory_prefs.py` — part2 (ذاكرة، تفضيلات، MCP، Storage API)
- `ai/nova_search_copyright.py` — part3 (البحث، حقوق النشر)
- `ai/nova_tools_registry.py` — parts 4-6 (مخططات الأدوات)
- `ai/godmode.py` — شخصية GODMODE والـ AutoTune
- `ai/harm_classifier.py` — مصنّف الضرر
- `.migration-backup/` — الملفات المرجعية (للقراءة فقط)
- `push_to_github.sh` — سكريبت الدفع لـ GitHub

## Architecture decisions

- Python/Streamlit (وليس Next.js) — المشروع الحقيقي مُهاجَر من .migration-backup/
- شخصية Nova تُضاف أول القائمة في PERSONAS (مع فحص سلامة مُدمج)
- فحص السلامة يعمل قبل إرسال الرسالة للـ API — يوقف التنفيذ عند اكتشاف محتوى محظور
- حقوق النشر: 15 كلمة حد أقصى لكل اقتباس، اقتباس واحد لكل مصدر
- build_full_nova_prompt() يجمع System Prompt من جميع الأجزاء 1-6

## Product

- دردشة AI متعددة الشخصيات: Nova ✨، GODMODE 🜏، CIPHER 🔐، ORACLE 🔮، SAGE 📡، REBEL ⚡، GLITCH 👾
- بث الاستجابات عبر OpenRouter
- حفظ سجل المحادثات في SQLite
- دعم الملفات والصور
- نظام AutoTune لضبط معاملات النموذج تلقائياً
- Nova: مساعد آمن من Aurora Labs مع قواعد سلامة شاملة

## User preferences

- `.migration-backup/` للقراءة فقط — لا تحذفه
- GitHub remote: aliahmed369000000-ai/Neural-Service-Mesh
- الدفع لـ GitHub بعد إكمال كل جزء (part)
- الـ workflow يعمل على المنفذ 5000

## Gotchas

- git add/commit محظوران في المُوكِّل الرئيسي — استخدم `push_to_github.sh` أو project_tasks
- python/python3 غير متاح مباشرةً من bash — Streamlit يُشغَّل عبر workflow
- NOVA_SYSTEM_PROMPT يُستورد قبل بناء PERSONAS — أي خطأ في nova_system.py يمنع الاستيراد

## Pointers

- ملفات part1-6 في `.migration-backup/` — المصدر الحقيقي لتعليمات Nova
- `build_full_nova_prompt()` في nova_system.py يجمع كل الأجزاء
