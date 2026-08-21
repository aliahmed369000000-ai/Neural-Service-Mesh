# بيانات Pretrain لـ SurahChain

| ملف | الوظيفة |
|-----|---------|
| `pretrain_sentences.pkl` | الجمل الموحدة (كبير — عبر Git LFS أو من Kaggle) |
| `data_registry.json` | سجل كل دفعات الدمج وعدد الجمل الفريدة |
| `training_cursor.json` | أين وصل التدريب — يمنع إعادة استخدام نفس الجمل |

## النسخة المؤمّنة حالياً
- **Kaggle Dataset:** [aliahmedmo/nsm-ar-shards-mo-wiki](https://www.kaggle.com/datasets/aliahmedmo/nsm-ar-shards-mo-wiki)
- **حجم تقريبي:** ~350,000 جملة فريدة (~150 MB)
- الملف المحلي `pretrain_sentences.pkl` يُنشأ بالدمج ولا يُرفع لـ GitHub بدون Git LFS (حد 100MB)

## الدمج والتتبّع
```bash
# دمج شاردات جديدة من مخرجات Kaggle
python experiments/surah_chain_network/merge_and_track_data.py --input-dir /path/to/pkls

# حالة
python experiments/surah_chain_network/merge_and_track_data.py --status

# بعد جولة تدريب
python experiments/surah_chain_network/merge_and_track_data.py --mark-used 50000 --note "xlarge run"
```

## التجميع
كل حساب Kaggle = SHARD_ID مختلف (0–6). حد الجلسة 12 ساعة → توقف آمن عند 10.5س.
