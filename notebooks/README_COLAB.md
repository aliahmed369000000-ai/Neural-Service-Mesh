# تشغيل وكيل NSM على Google Colab

## الطريقة المدعومة: الوكيل **داخل** Colab

1. افتح: [Google Colab](https://colab.research.google.com/)
2. File → Upload notebook → `NSM_Colab_Training_Agent.ipynb`
   أو من GitHub مباشرة بعد رفع المستودع.
3. Runtime → Change runtime type → **GPU (T4)**
4. نفّذ الخلايا بالترتيب.

الملف: `notebooks/NSM_Colab_Training_Agent.ipynb`  
التهيئة السريعة: `scripts/colab_bootstrap.py`

## متغيرات مهمة
| متغير | المعنى |
|--------|--------|
| `NSM_ALLOW_GPU=1` | فرض استخدام CUDA إن وُجد |
| `MOUNT_DRIVE=True` | ربط Google Drive وحفظ `.pt` |

## غير مدعوم في المستودع
التحكم في واجهة Colab من متصفح خارجي (Playwright/Selenium) — غير مستقر وغالباً مخالف لشروط الخدمة.


## ربط النتائج بخادمك (بدون أتمتة متصفح)

```bash
# على السيرفر
export NSM_REMOTE_WEBHOOK_SECRET=mysecret
uvicorn api_server:app --host 0.0.0.0 --port 5000

# في Colab
export NSM_REMOTE_WEBHOOK_URL=https://YOUR_HOST:5000/training/remote-results
export NSM_REMOTE_WEBHOOK_SECRET=mysecret
python scripts/colab_result_push.py --csv data/samples/classification_demo.csv
```

الواجهة: `POST /training/remote-results` · `GET /training/remote-status`  
الكود: `ai/remote_gpu_provider.py`

## المنسّق الموحّد + تدريب فعّال

من داخل NSM أو بعد bootstrap:

```
مهمة colab
حالة المنصات
خطة كفاءة
```

أو في خلية Colab:

```python
from ai.remote_training_orchestrator import efficient_nn_training_source
exec(compile(efficient_nn_training_source("colab_local"), "t.py", "exec"))
```

لا يوجد تحكم بمتصفح Colab من الخارج (سياسة المشروع + شروط Google).

## SurahChain Pre-training (دفتر مخصص)

الملف: `notebooks/SurahChain_Pretrain_Colab.ipynb`

1. ارفع الدفتر إلى Colab أو افتحه من GitHub
2. Runtime → GPU
3. ضع `GITHUB_TOKEN` في خلية الإعدادات
4. نفّذ الخلايا بالترتيب (تدريب ثم رفع تلقائي)

بديل سكربت واحد بعد clone:

```bash
export GITHUB_TOKEN=ghp_xxx
export SCN_N=30000 SCN_EPOCHS=10 SCN_D_MODEL=128 SCN_BATCH=32
python experiments/surah_chain_network/colab_run_pretrain.py
```
