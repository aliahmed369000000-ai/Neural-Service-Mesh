# انضمام عقدة لاتحاد NSM خلال دقيقة

## المتطلبات
- Python 3.10+
- مستودع Neural Service Mesh

## الخطوات

```bash
git clone https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git
cd Neural-Service-Mesh
pip install -r requirements.txt
```

### تشغيل عقدة
```bash
python -m ai.node_launcher \
  --id my_node \
  --host 0.0.0.0 \
  --port 7860 \
  --seed-host <IP_أو_مضيف_البذرة> \
  --seed-port 7860
```

### مسارات مفيدة على العقدة
- `GET /health` — صحة
- `GET /v2/status` — حالة موسّعة
- `GET /v2/routes` — مسارات الأقران
- `GET /dashboard` — لوحة مختصرة
- `WS /ws` — شبكة P2P

### واجهة الاتحاد
من تطبيق Streamlit: **النظام → 🏛️ الاتحاد**

### إثبات محلي للاتحاد
```bash
python3 scripts/prove_federation.py
```

## مبادئ الاتحاد
1. لا مركز دائم — قائد مؤقت منتخب
2. لا نتيجة بدون تحقق (VCEN)
3. لا عيّنات خام في التعلم الجماعي
4. القرارات بنصاب اتحادي ومقاومة للانقسام

التفاصيل: [FEDERATION.md](../FEDERATION.md)
