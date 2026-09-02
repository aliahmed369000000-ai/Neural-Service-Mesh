# NSM Release Notes — إصدار إنتاجي مستقر (نواة الاتحاد)

**الإصدار:** `v0.1.0-federation-core`  
**الوسم:** git tag `v0.1.0-federation-core`  
**حالة البوابة:** شغّل `python3 scripts/release_gate.py` → يجب `GATE: PASS`  
**CI:** `.github/workflows/federation-release-gate.yml` على PR/push لمسارات الاتحاد

**بروتوكول الانضمام:** `nsm-join-v1`

---
## ما هو مدعوم في هذا الإصدار

### API عقدة (`ai.node_launcher`)
| المسار | الوصف |
|--------|--------|
| `GET /health` | صحة العقدة |
| `GET /v2/status` | حالة موسّعة |
| `GET /v2/join-info` | اكتشاف البذرة |
| `POST /v2/join` | تسجيل عقدة خارجية |
| `POST /v2/accept-peer-key` | تبادل مفاتيح ثنائي |
| `POST /v2/first-task` | أول مهمة موثّقة (محلية على العقدة) |
| `POST /v2/task` | مهمة قابلة للتحقق |
| `GET /v2/tasks` | سجل مهام حديثة |
| `GET /v2/routes` | مسارات الأقران |
| `GET /ws` | WebSocket P2P موقّع |

### قدرات الاتحاد (مكتبات)
- هوية دائمة + انتخاب قائد مؤقت + استكمال جولة بعد الفشل
- VCEN: توقيع + Hash + مدقق مستقل + Quorum
- CCL: ذاكرة / نموذج / قرارات جماعية قابلة للتدقيق
- Byzantine Guard: roster + majority اتحادي + منع split-brain
- Private FL: بلا عيّنات خام، clip/ضوضاء، تجميع بأقنعة
- مسار انضمام خارجي حتى مهمة موثّقة على العامل

### سكربتات إثبات
```bash
python3 scripts/release_gate.py
python3 scripts/prove_federation.py
python3 scripts/run_live_join_demo.py   # يتطلب aiohttp
python3 scripts/join_external_path.py --seed http://HOST:PORT
```

### توثيق
- `FEDERATION.md`
- `docs/JOIN_FEDERATION.md`
- `docs/API_JOIN_PATH.md`

---

## خارج نطاق هذا الإصدار (تجريبي / لاحقاً)

- سوق قدرات ذكاء موزّع (تجاري)
- بحث ويب آمن متعدد المصادر (#11–#13) كمنتج مستخدم نهائي
- وكلاء ومحادثات وSurah/HF كاعتماد إنتاجي للاتحاد
- ضمانات خصوصية تفاضلية كاملة (DP-SGD) أو SMPC إنتاجي
- اتفاق عالمي عبر القارات بدون تشغيل ميداني إضافي

---

## سياسة التغيير

1. أي كسر لـ `nsm-join-v1` أو مسارات `/v2/join*` يتطلب رفع إصدار البروتوكول.
2. لا تُدمج ميزات جديدة في الفرع المستقر دون `release_gate.py → PASS`.
3. الطبقات التجريبية تبقى خلف أعلام/صفحات منفصلة ولا تُعتبر عقداً للاتحاد.

---

## تشغيل سريع مستقر

```bash
# بذرة
python -m ai.node_launcher --id seed --host 0.0.0.0 --port 7860

# عامل (جهاز/منفذ آخر) ثم:
python3 scripts/join_external_path.py --seed http://SEED:7860 --node-id worker1
# أو العرض الحي المزدوج:
python3 scripts/run_live_join_demo.py
```
