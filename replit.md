# Neural Service Mesh (NSM) — النظام المعرفي العربي

منصة ذكاء اصطناعي عربية (Streamlit) تجمع محركاً معرفياً ذاتي التعلّم (CKG)
مع نماذج لغوية كبيرة، بحث ويب وصور حقيقي، ذاكرة محادثة دائمة، ومحرر واجهات
تفاعلية (Artifacts)، مع تخصص بالمعرفة الإسلامية والقرآن الكريم.

## Run & Operate

- التشغيل: workflow `NSM Streamlit App` يشغّل `streamlit run streamlit_app.py --server.port 5000`
- `api_server.py` (FastAPI) خادم مستقل اختياري لمحرك `core.engine` — غير مربوط بواجهة Streamlit حالياً
- بيانات دائمة: SQLite في `memory/nsm_context.db` (المحادثات)، `data/artifacts.db` (الواجهات التفاعلية + الإعدادات)، `data/mesh.db` (Node/Engine)

## Stack

- Python 3.12، Streamlit، SQLite
- بحث ويب: DuckDuckGo (بدون مفتاح) عبر `ai/web_search_tool.py`
- بحث صور: Unsplash API عبر `ai/image_search_tool.py` (يتطلب `UNSPLASH_ACCESS_KEY`)
- نماذج لغوية: OpenRouter (اختياري)، Anthropic (اختياري)، NSM/LLMFallback (احتياطي محلي)

## Where things live

- `streamlit_app.py` — الواجهة الرئيسية (كل التبويبات: بحث، محادثة، ذاكرة، واجهات تفاعلية، لوحة مطوّر...)
- `nsm_memory.py` — ذاكرة المحادثة (نافذة قصيرة + SQLite طويلة الأمد + بحث دلالي)
- `ai/` — أدوات ووكلاء (بحث ويب/صور، وكيل الكود، محرك NSM، إلخ)
- `core/` — محرك Node/Engine + `artifacts_store.py` (تخزين الواجهات التفاعلية والإعدادات الدائمة)
- `storage/` — طبقة SQLite أقدم لمحرك Node/Engine (`db.py`)

## Architecture decisions

- بحث الويب والصور بدون طبقة وسيطة إضافية — يُستدعيان مباشرة من `streamlit_app.py` عبر دوال في `ai/`، ويرفعان استثناءات صريحة بدل نتائج وهمية عند الفشل.
- لوحة المطوّر (تنفيذ Bash/Python) محمية بمفتاح `NSM_ADMIN_KEY` — مقفلة افتراضياً حتى إدخال المفتاح الصحيح في الجلسة.
- `ai/github_sync.py` يقبل إما `GITHUB_TOKEN` أو `GITHUB_PERSONAL_ACCESS_TOKEN` كمصدر للتوكن (كلاهما مدعوم).

## Product

- بحث معرفي + بحث ويب حقيقي + بحث صور (Unsplash)
- محادثة ذكية بذاكرة تُستعرض وتُبحث من تبويب "🧠 الذاكرة"
- محرر واجهات تفاعلية (HTML/SVG) قابلة للحفظ، مع أداة استدعاء API عام
- لوحة مطوّر محمية لتنفيذ أوامر Bash/Python
- تبويب "ℹ️ عن NSM" يشرح المنتج للمستخدم الجديد

## User preferences

- المستخدم يتواصل بالعربية ويفضّل تنفيذ الميزات بالترتيب المُعطى مع commit + push إلى GitHub بعد كل ميزة.
- بحث الويب: يُبقى مجانياً عبر DuckDuckGo (لا مزوّد مدفوع حالياً).
- ميزات MCP (Google Drive / Slack) مؤجلة بطلب المستخدم — لم تُبنَ بعد.

## Gotchas

- Streamlit يحتاج `enableCORS = false` و`enableXsrfProtection = false` في `.streamlit/config.toml` ليعمل صحيحاً خلف iframe المعاينة في Replit.
- `NSM_ADMIN_KEY` و`UNSPLASH_ACCESS_KEY` أسرار مطلوبة — بدونها تتعطّل ميزتا لوحة المطوّر وبحث الصور بأمان (رسالة خطأ صريحة، لا فشل صامت).
- الدفع لـ GitHub يتطلب `GITHUB_PERSONAL_ACCESS_TOKEN` (أو `GITHUB_TOKEN`) + `GITHUB_USER` + `GITHUB_REMOTE` (الأخيران env vars عاديان، مضبوطان مسبقاً).

## Pointers

- `ai/web_search_tool.py`, `ai/image_search_tool.py` — أدوات البحث
- `core/artifacts_store.py` — تخزين الواجهات التفاعلية + إعدادات المستخدم الدائمة
- `nsm_memory.py` — منطق الذاكرة الكامل


## Living Mesh — تشغيل عقد دائمة ولوحة التحكم

### تشغيل البذرة (seed)
```bash
python ai/node_launcher.py --id mesh_seed --host 0.0.0.0 --port $PORT
```
أو: `bash scripts/run_mesh_seed.sh` (يستخدم `PORT` و`NODE_ID` و`NSM_NODE_DATA_DIR`).

### تشغيل عامل (worker)
```bash
SEED_NODE_URL=127.0.0.1:7860 NODE_ID=worker_1 PORT=7861 bash scripts/run_mesh_worker.sh
```

### مجموعة محلية محافظة
`NSM_NODE_COUNT=1` (افتراضي = بذرة فقط) وحتى 10 كحد أقصى:
```bash
NSM_NODE_COUNT=3 python scripts/run_local_mesh.py
```

### عزل الحالة
كل عقدة لها مجلد مستقل: `artifacts/living_mesh/nodes/<NODE_ID>/` (مفاتيح + `network_state.json`).  
لا تُكتب عقد وهمية يدوياً — تظهر بعد `join_network` وتشغيل العملية فعلياً.

### لوحة التحكم
بعد تشغيل البذرة افتح:
- **`/dashboard`** — مقاييس الشبكة، جداول المهام/القرارات، أزرار تجربة سريعة (بحث / تلخيص / بحث+تلخيص) بنصوص محلية.
- التحديث: الجداول تُحدَّث كل ~5ث من `/v2/jobs`؛ الصفحة تُعاد كل 15ث.

### واجهات HTTP الأساسية
| المسار | الوظيفة |
|--------|---------|
| `GET /status` | حالة العقدة + الأقران online |
| `GET /health` | صحة الشبكة |
| `GET /v2/status` | لقطة موسّعة + سمعة |
| `GET /dashboard` | لوحة HTML حية |
| `GET /v2/jobs` | `{ jobs, decisions }` |
| `POST /v2/job` | مهمة متعددة العمال (`kind`, `payload`, `n_workers`, `strategy`, `quorum`) |
| `POST /v2/collective-search` | بحث جماعي — **corpus مُمرَّر فقط** (بلا SSRF) |
| `POST /v2/collective-summary` | تلخيص جماعي مع `source_hash` و provenance |
| `POST /v2/web-task` | مهمة ويب آمنة (HTTPS فقط + حماية SSRF) |
| `GET /ws` | WebSocket P2P موقّع |

### مثال بحث جماعي
```bash
curl -s -X POST "http://127.0.0.1:$PORT/v2/collective-search" \
  -H "Content-Type: application/json" \
  -d '{"query":"نصاب","corpus":[{"source_id":"a","text":"النصاب والشبكات الموزعة"}],"n_workers":1,"quorum":1}'
```

### ملاحظات البيئة
- على Replit / Streamlit Cloud غالباً منفذ خارجي واحد: البذرة تُعرَّض للخارج والعمال محليون عبر `SEED_NODE_URL`.
- سكربتات مساعدة: `scripts/run_mesh_seed.sh`, `scripts/run_mesh_worker.sh`, `scripts/run_local_mesh.py`, `scripts/bench_mesh_rpc.py`.
