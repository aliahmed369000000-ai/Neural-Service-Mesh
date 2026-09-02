# حماية الفرع main

## لماذا
بعد `v0.1.0-federation-core` و workflow **Federation Release Gate**، يُفضَّل منع الدمج دون PASS.

## تفعيل يدوي (GitHub UI)
1. Settings → Branches → Add rule / Edit for `main`
2. ✅ Require a pull request before merging (اختياري لكن مُستحسن)
3. ✅ Require status checks to pass before merging
4. ابحث عن واختر: **Federation Release Gate** / `release_gate.py → PASS`
5. ✅ Require branches to be up to date (اختياري)
6. لا تُعطِّل القوة الإدارية إلا إذا كان الفريق جاهزاً

## ملاحظة
تفعيل الحماية عبر API يحتاج صلاحية `Administration` على المستودع؛ التوكن المستخدم للتطوير قد لا يكفي — الواجهة الرسومية هي المسار الموثوق.

## تشغيل ميداني طويل الأمد (بديل/مكمّل)
```bash
# بذرة
python -m ai.node_launcher --id seed --host 0.0.0.0 --port 7860
# عامل
python -m ai.node_launcher --id worker1 --host 0.0.0.0 --port 7861
# مراقبة
watch -n 30 'curl -s http://SEED:7860/health; curl -s http://WORKER:7861/health'
```
لا تُضف ميزات جديدة أثناء المراقبة الأولى؛ راقب `/health` و`/v2/tasks` فقط.
