# تشغيل وكيل NSM على Kaggle

Kaggle يوفّر **Dual T4 GPUs** مجاناً و~30 ساعة أسبوعياً — غالباً أكثر استقراراً من Colab للجلسات الطويلة.

## الطريقة 1 — عبر Kaggle API (من جهازك / السيرفر)

الأقوى: الوكيل يجهّز Kernel ويدفعه للتشغيل على GPU دون فتح متصفح.

### 1) إعداد الاعتمادات
1. من [Kaggle Account](https://www.kaggle.com/settings) → **Create New Token** → تحميل `kaggle.json`
2. ضعه في `~/.kaggle/kaggle.json` (صلاحيات 600)  
   **أو** صدّر:
   ```bash
   export KAGGLE_USERNAME=your_user
   export KAGGLE_KEY=your_key
   ```
3. `pip install kaggle`

### 2) أوامر الوكيل (عربية)
| أمر | الوظيفة |
|-----|---------|
| `حالة kaggle` | تقرير الاعتمادات + GPU + المهام |
| `جهّز kaggle` | إنشاء مهمة + سكربت + metadata |
| `درّب kaggle csv data/samples/classification_demo.csv` | تجهيز مع CSV |
| `ادفع kaggle <job_id>` | `kaggle kernels push` وتشغيل GPU |
| `حالة kaggle <job_id>` | مراقبة الحالة |
| `حمّل kaggle <job_id>` | تنزيل الأوزان والتقرير |
| `قائمة kaggle` | المهام المحلية |

الملفات تُحفظ تحت: `artifacts/model_training/kaggle_jobs/<job_id>/`

### 3) يدوياً (بدون وكيل)
```bash
cd artifacts/model_training/kaggle_jobs/<job_id>
kaggle kernels push -p .
kaggle kernels status <username>/<slug>
kaggle kernels output <username>/<slug> -p ./output
```

## الطريقة 2 — داخل دفتر Kaggle

1. New Notebook على Kaggle
2. **Settings → Accelerator → GPU T4 x2** (Dual T4)
3. Internet ON إن احتجت استنساخ المستودع
4. خلايا:

```python
!git clone https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git
%cd Neural-Service-Mesh
%run scripts/kaggle_bootstrap.py
```

```python
from ai.kaggle_provider import handle_kaggle_command, detect_kaggle_gpus, wrap_model_for_multi_gpu
print(handle_kaggle_command("حالة kaggle"))
print(detect_kaggle_gpus())
```

### Multi-GPU (DataParallel)
الوكيل يوفّر غلافاً جاهزاً:

```python
from ai.kaggle_provider import wrap_model_for_multi_gpu, multi_gpu_training_snippet
# model, note = wrap_model_for_multi_gpu(my_model)
print(multi_gpu_training_snippet())
```

عند الحفظ من `DataParallel` استخدم `model.module.state_dict()`.

## مقارنة سريعة مع Colab

| | Kaggle | Colab |
|--|--------|-------|
| GPU مجاني | Dual T4 | T4 (غالباً) |
| ساعات/أسبوع | ~30 | متغيرة + انقطاع |
| Datasets | ملايين جاهزة بنفس الخوادم | Drive / رفع يدوي |
| API للتشغيل البعيد | Kernels API رسمي | Webhook / يدوي |
| أتمتة متصفح | غير مطلوبة | غير مدعومة في NSM |

## الكود
- `ai/kaggle_provider.py` — المزود + الأوامر
- `scripts/kaggle_bootstrap.py` — تهيئة داخل الدفتر
- تكامل مع `ai/model_training_agent.py` و `ai/remote_gpu_provider.py`

## أمان
لا ترفع `kaggle.json` إلى Git. أضفه لـ `.gitignore` (موجود مسبقاً لأنماط الأسرار الشائعة). استخدم متغيرات بيئة على السيرفر.


## تفعيل تسريع GPU (مهم)

الوكيل يضبط تلقائياً:
- `"enable_gpu": true`
- `"machine_shape": "NvidiaTeslaT4"`
- `kaggle kernels push --accelerator NvidiaTeslaT4`
- نوع Kernel = **notebook** (أفضل من script لتفعيل GPU)

### إذا بقي التشغيل على CPU
أسباب شائعة:
1. **نفاد حصة GPU الأسبوعية** (~30 ساعة) — راجع [Kaggle Settings](https://www.kaggle.com/settings)
2. **أول تفعيل لنواة جديدة** — Kaggle أحياناً يتجاهل الـaccelerator من API:
   - افتح الرابط الذي يطبعه الوكيل بعد الدفع
   - Settings → **Accelerator → GPU T4 ×2**
   - Save Version → Run All
   - بعد ذلك، الدفعات التالية عبر API تحافظ على الإعداد غالباً
3. تأكد أن Internet ON إن احتجت تحميل حزم

### أوامر الوكيل المتعلقة بالـGPU
```
جهّز kaggle          # T4 افتراضي
درّب kaggle dual t4  # يطلب T4 (Dual×2 من الواجهة إن لزم)
ادفع kaggle <id>
حالة kaggle <id>
حمّل kaggle <id>
```

## المنسّق الموحّد (Orchestrator)

```
حالة المنصات
درّب بعيد kaggle
درّب بعيد kaggle وادفع epochs 30
خطة كفاءة
```

الكود: `ai/remote_training_orchestrator.py` — تدريب فعّال (AMP + DataParallel + AdamW + early stop).
