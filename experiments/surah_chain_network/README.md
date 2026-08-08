# Surah-Chain Network — نموذج لغوي بوسط سوري

شبكة تدمج **طرفَي نموذج لغوي (LLM-style)** مع **114 طبقة وسطى** مبنية من
`شبكهه_114-1.xlsx` (أبعاد كل طبقة = عدد آيات سورة → السورة التالية).

## المعمارية الحالية

```
النص
  → HashTokenizer
  → [Input LLM]  Embedding(d_model) + موضع جيبي + LayerNorm
  → Adapter W_in:  d_model → 7
  → [SurahChain ×114]  من ملف السور (Residual + LayerNorm)
  → Adapter W_out: 7 → d_model
  → [Output LLM]  LM Head مربوط (logits = h @ Eᵀ)
  → توزيع على المفردات
```

| الجزء | المصدر | الوصف |
|--------|--------|--------|
| طبقة الإدخال | LLM-style | تضمين رموز + موضع + LN + إسقاط إلى عرض السلسلة (7) |
| 114 طبقة وسطى | `شبكهه_114-1.xlsx` | سلسلة FC بأبعاد آيات السور المتتالية |
| طبقة الإخراج | LLM-style | إسقاط من 7 + رأس لغة مربوط بالأوزان (weight tying) |

`d_model` الافتراضي = **512** (يمكن تمرير `d_model=` أو `project_d_model=` للتوافق).

## الفرق عن النسخة السابقة

| سابقاً | الآن |
|--------|------|
| `W_down` / `W_up` عشوائيان صغيران | طرفا LLM: Embedding + Tied LM Head |
| `TokenEmbedding` + `OutputHead` منفصلان | رأس مربوط بنفس مصفوفة `E` |
| بدون موضع | ترميز موضع جيبي |
| `d_model` تجريبي 256 | افتراضي 512 بأسلوب LLM صغير |

الوسط (SurahChain) **لم يتغيّر**: ما زال 114 طبقة من ملف السور.

## التدريب

```bash
# من جذر المستودع
python experiments/surah_chain_network/train_no_decay.py
# أو فحص سريع
python experiments/surah_chain_network/hybrid_experiment.py
```

- `hybrid_experiment.py` — التعريف: `SurahChainLayer` / `SurahChainNetwork` / `HybridExperimentModel`
- `hybrid_data.py` — جمل تدريب قصيرة
- `train_no_decay.py` — حلقات تدريب مع حفظ أفضل checkpoint
- `surah_layer_dims.json` — أبعاد الطبقات الـ114

## الحالة

⚠️ **تجريبية** — غير موصولة بعد بمسار `qa_engine` / Streamlit.
إثبات أن طرفَي LLM + وسط سوري يتعلّمان معاً على نص عربي/CKG.


## قدرات النموذج اللغوي (منفّذة)

| القدرة | التفاصيل |
|--------|----------|
| **Tokenizer** | `WordTokenizer` (decode حقيقي) + بناء قاموس من نصوص التدريب |
| **generate()** | توليد سببي مع `temperature` و `top_k` حتى EOS |
| **train_batch** | دفعات نصوص + متوسط خسارة |
| **LR schedule** | `cosine_lr`: warmup خطي ثم cosine decay |

```bash
# تدريب LM سريع (جمل hybrid_data)
python experiments/surah_chain_network/train_lm.py

# فحص generate بعد بناء قاموس
python experiments/surah_chain_network/hybrid_experiment.py
```

متغيرات اختيارية: `SCN_EPOCHS`, `SCN_BATCH`, `SCN_D_MODEL`, `SCN_LR`, `SCN_MAX_LEN`.


## المسار نحو نموذج لغوي كامل (منفّذ تدريجياً)

| الخطوة | الحالة |
|--------|--------|
| طرفا LLM + SurahChain وسط | ✅ |
| WordTokenizer + generate + batch + cosine LR | ✅ |
| **مسار متبقٍّ حول اختناق البعد 7** (`W_skip`) | ✅ |
| سياق أطول (max_len 64، generate ctx 96) | ✅ |
| **تدريب على جمل CKG** | ✅ `train_ckg_lm.py` |
| انتباه / دمج NSM الحي | لاحقاً |

```bash
# تدريب على CKG (عيّنة قابلة للضبط)
SCN_N=1500 SCN_EPOCHS=8 python experiments/surah_chain_network/train_ckg_lm.py
```


## انتباه قوي + تدريب طويل

المعمارية الحالية:

```
Embedding+Pos+LN
  → TransformerBlock ×2  (Multi-Head Causal Attention 8 رؤوس + FFN)
  → Adapter → SurahChain×114 → Adapter + W_skip
  → TransformerBlock ×2
  → Tied LM Head
```

```bash
# تدريب طويل على CKG (افتراضي: 5000 جملة × 20 عصر × 8 رؤوس)
python experiments/surah_chain_network/train_ckg_lm.py

SCN_N=8000 SCN_EPOCHS=30 SCN_D_MODEL=256 SCN_N_HEADS=8 \
  SCN_N_PRE=2 SCN_N_POST=2 python experiments/surah_chain_network/train_ckg_lm.py
```


## Tokenizer / Activations / Backprop / Generate (محدّث)

| المكوّن | التنفيذ |
|---------|---------|
| **StrongTokenizer** | كلمات + حروف + دمج BPE-lite، encode/decode ثنائي |
| **Activations** | **GELU** في SurahChain وFFN (مع مشتقات backward) |
| **LayerNorm** | backprop كامل على x وγ وβ |
| **Attention** | Multi-Head Causal + سياق حتى **256** |
| **generate** | top-k + top-p + repetition penalty + min_new_tokens |

```python
from experiments.surah_chain_network.hybrid_experiment import HybridExperimentModel
m = HybridExperimentModel(tokenizer="strong", d_model=256, n_heads=8)
m.build_tokenizer_from_texts(texts)
print(m.generate("الصبر", max_new_tokens=40, top_p=0.9, repetition_penalty=1.2))
```
