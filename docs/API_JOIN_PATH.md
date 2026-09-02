# مسار انضمام عقدة خارجية عبر API

من الصفر حتى **أول مهمة موثّقة** — بدون نموذج أو وكيل جديد.

## التسلسل

```
1) GET  /v2/join-info     ← اكتشف البذرة والبروتوكول
2) أنشئ هوية محلية        ← LivingMeshNode (RSA دائم)
3) POST /v2/join          ← سجّل node_id + public_key + host/port
4) POST /v2/first-task    ← أول مهمة map_reduce موثّقة (إيصال + تحقق)
5) GET  /health           ← تأكيد صحة البذرة
```

## تشغيل البذرة

```bash
python -m ai.node_launcher --id seed --host 0.0.0.0 --port 7860
```

## انضمام سريع (سكربت)

```bash
python3 scripts/join_external_path.py --seed http://SEED_IP:7860 --node-id my_external
# أو عرض محلي كامل:
python3 scripts/join_external_path.py --local-demo
```

## أمثلة curl

```bash
# 1
curl -s http://SEED:7860/v2/join-info | jq .

# 3
curl -s -X POST http://SEED:7860/v2/join \
  -H 'Content-Type: application/json' \
  -d '{"node_id":"ext1","host":"1.2.3.4","port":7861,"public_key":"-----BEGIN PUBLIC KEY-----\n..."}'

# 4 أول مهمة موثّقة
curl -s -X POST http://SEED:7860/v2/first-task \
  -H 'Content-Type: application/json' \
  -d '{"lines":["hello from external","verified task"]}'
```

## الاستجابة المتوقعة لـ `/v2/first-task`

- `ok: true`
- `receipt` موقّع من عقدة البذرة
- `verification.signature_valid` و `digest_valid`
- `result` (مثلاً wordcount)

## ملاحظات

- `/v2/join` يحفظ مفتاح العقدة في `keys/{node_id}.pub` ويسجّلها في `network_state`
- المهمة الأولى الافتراضية: `map_reduce_map` محلية على البذرة مع إيصال قابل للتحقق
- للتنفيذ على عقدتك أنت: شغّل `node_launcher` محلياً ثم `POST /v2/first-task` على **منفذ عقدتك**


## مسار المهمة على العامل (إيصال من العقدة الخارجية)

```
1) GET  seed/v2/join-info
2) POST seed/v2/join              ← البذرة تحفظ مفتاح العامل
3) POST worker/v2/accept-peer-key ← العامل يحفظ مفتاح البذرة
4) POST seed/v2/accept-peer-key   ← تأكيد ثنائي (اختياري إن تم في 2)
5) POST worker/v2/first-task      ← المهمة تُنفَّذ على العامل؛ الإيصال موقّع منه
6) البذرة تتحقق من الإيصال بمفتاح العامل المخزّن
```

```bash
python3 scripts/run_live_join_demo.py
# worker:19901 · seed:19876
```


## حلقة التكليف من البذرة (dispatch)

```
POST seed/v2/dispatch-task
{
  "target_url": "http://worker:19901",
  "path": "/v2/first-task",
  "payload": {"lines": ["..."]}
}
→ العامل ينفّذ ويوقّع الإيصال
→ البذرة تتحقق verification_on_seed
```

بروتوكول: `nsm-dispatch-v1`
