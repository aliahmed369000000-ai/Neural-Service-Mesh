# تقييم دعم Webhook الحقيقي لكل منصة

هذا الملف يوثّق، لكل منصة، هل تحويلها من polling إلى webhook (دفع فوري
للأحداث الواردة عبر HTTP) ممكن فعلياً وفق الوثائق الرسمية الحالية —
وليس افتراضاً. `supports_webhook = True` في الكود مقصور على ما هو محقَّق
هنا فعلاً بأدلة تقنية، لا تخمين.

## ✅ مُطبَّق فعلياً في هذا الإصدار

### Telegram — `supports_webhook = True`
`setWebhook` رسمي وموثَّق بالكامل: تيليجرام يرسل كل تحديث كطلب POST إلى
عنوان HTTPS نحدده، بنفس بنية JSON التي يعيدها `getUpdates`. هذا أوضح حالة
تقنياً ولذلك بدأنا به. مُطبَّق في `telegram_adapter.py`
(`set_webhook`/`delete_webhook`/`webhook_info`) و`api_server.py`
(`POST /webhook/telegram/{secret}`)، مع التحقق من `secret_token` قبل أي
معالجة، وإبقاء `getUpdates` كخيار احتياطي كامل لمن لا يملك endpoint عام.

## ⚠️ ممكن نظرياً لكن غير مُطبَّق بعد (يتطلب عملاً إضافياً منفصلاً)

### Facebook / Instagram — Meta Graph API
Meta توفّر فعلياً نظام Webhooks Subscriptions لصفحات Facebook وحسابات
Instagram Business (تعليقات، منشنز، رسائل خاصة) — نفس الآلية المستخدمة
أصلاً في `whatsapp_gateway/api/webhook.py` بهذا المشروع (تحقّق GET عبر
`hub.challenge` + استقبال POST). تقنياً هذا أقرب مرشّح تالٍ لتيليجرام،
لكنه يتطلب: تسجيل تطبيق Meta Developer منفصل، مراجعة أذونات
(`pages_messaging`, `instagram_manage_comments`, ...)، واشتراك Webhook
لكل صفحة/حساب على حدة من لوحة تحكم Meta — عمل إعداد خارج نطاق تعديل
الكود وحده. تُرك `supports_webhook = False` حتى إتمام هذا الإعداد فعلياً.

## ❌ غير عملي بوضع "مراقبة/رد تلقائي عام" الحالي

### Discord
Discord توفّر نوعين مختلفين تماماً عمّا نحتاجه:
- **Outgoing Webhooks**: لدفع رسائل *صادرة* لقناة (نشر فقط) — لا تصلح
  لاستقبال شيء، والنشر الحالي أصلاً عبر Bot API لا يحتاجها.
- **Interactions Endpoint**: لاستقبال ضغطات أزرار/أوامر Slash فقط
  (يتطلب توقيع Ed25519 لكل طلب) — ليس لمراقبة كل الرسائل/المنشنز في قناة.
المراقبة العامة للرسائل تتطلب **Gateway** (اتصال WebSocket دائم)، وهو
شيء مختلف جذرياً عن HTTP webhook (اتصال دائم لا طلب/استجابة)، ويحتاج
عملية خلفية مستقلة تُبقي الاتصال مفتوحاً — خارج نطاق "تحويل polling إلى
webhook" المطلوب هنا. يبقى `supports_webhook = False`.

### Twitter/X
Account Activity API (الذي يوفّر webhooks حقيقية للمنشنز/الردود) أصبح
حصرياً لمستوى Enterprise المدفوع بشدة، وغير متاح على مستويات API
الأساسية/Pro المستخدمة عادة بمشاريع كهذا. `supports_webhook = False`.

### Reddit
لا يوجد push webhook رسمي للتعليقات/المنشنز في PRAW/Reddit API العامة —
polling عبر `/new` أو streams هو الأسلوب الموثَّق الوحيد. `supports_webhook = False`.

### LinkedIn
لا يوفّر LinkedIn API عاماً webhooks للتعليقات/المنشنز على منشورات
الأفراد أو الصفحات بالطريقة التي نحتاجها هنا. `supports_webhook = False`.

### YouTube
يوجد PubSubHubbub/WebSub لكن فقط لإشعارات "فيديو جديد على قناة" — ليس
للتعليقات، وهي أصلاً منصة نشر فيديوهات لا مراقبة منشنز نصية بنفس معنى
بقية المحولات هنا. `supports_webhook = False`.

### TikTok / Threads
لا تتوفّر لأي منهما webhooks عامة موثَّقة لاستقبال تفاعلات/منشنز ضمن
نطاق صلاحيات التطبيقات العادية (غير الشراكات الخاصة). `supports_webhook = False`.
