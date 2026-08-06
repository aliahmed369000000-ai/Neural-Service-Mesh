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
