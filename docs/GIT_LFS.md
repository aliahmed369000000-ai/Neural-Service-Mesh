# Git LFS — للجميع

ملفات المعرفة والأوزان الكبيرة عبر **Git LFS**. بدون سحبها يبقى CKG فارغاً.

## إعداد

```bash
bash scripts/setup_git_lfs.sh
```

أو يدوياً: تثبيت [git-lfs](https://git-lfs.com) ثم:

```bash
git lfs install
git lfs pull
```

## بعد كل clone

```bash
git clone <url>
cd Neural-Service-Mesh
bash scripts/setup_git_lfs.sh
```

## تحقق

```bash
head -c 40 knowledge/cognitive_graph.json   # يجب أن يبدأ بـ {
wc -c knowledge/cognitive_graph.json       # عشرات MB وليس ~130 بايت
python3 scripts/ckg_quality_report.py --ckg-only
```

## للمساهمين

- الأنماط في `.gitattributes` تتبّع الملفات الكبيرة تلقائياً.
- إن ظهر مؤشر بدل المحتوى: `git lfs pull`.
