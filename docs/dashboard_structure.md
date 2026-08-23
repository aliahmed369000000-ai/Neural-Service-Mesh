# هيكلية واجهة لوحة السرب الموحدة (Unified Swarm Dashboard)

## المكونات الحالية:
1. **render_section_header**: ترويسات الأقسام.
2. **render_kpi_cards**: بطاقات الأداء الرئيسية (وكلاء نشطون، اكتمال الاندماج، إلخ).
3. **performance_timeline**: رسوم بيانية تفاعلية (Plotly) لاستخدام الذاكرة ووقت الاستجابة.
4. **render_alert_cards**: عرض التنبيهات النشطة.
5. **render_agent_cards**: عرض بطاقات الوكلاء الفردية.

## البيانات المستوردة:
- `unified_dashboard_snapshot` من `ai.unified_swarm_dashboard`.
- `get_network_snapshot` من `ai.living_mesh`.

## خطة التطوير:
- إضافة قسم "مراقبة حيّة" (Live Monitoring) يعتمد على WebSockets.
- دمج "خريطة الثقة الرقمية" لعرض مفاتيح RSA العامة.
- تحويل "اكتمال الاندماج" إلى عداد حي يتفاعل مع النبضات.
