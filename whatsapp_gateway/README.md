# بوابة واتساب لـ NSM (Vercel)

مشروع Vercel **منفصل تماماً** عن نشرة Streamlit الرئيسية لنفس المستودع.
السبب: `requirements.txt` الرئيسي بجذر المستودع ثقيل (librosa, moviepy,
numba, scikit-learn...) وغير مناسب لدالة serverless خفيفة، و`memory/
accounts.db` (SQLite) على قرص Streamlit غير متاح لدالة Vercel أصلاً
(نظاما ملفات منفصلان). راجع رسائل الكوميت لهذا المجلد للتفاصيل الكاملة.

## خطوات النشر

1. بلوحة Vercel: **New Project** → اختر نفس مستودع GitHub
   `Neural-Service-Mesh`.
2. بإعدادات المشروع (**Root Directory**): اضبطها على `whatsapp_gateway`
   — هذا يعزل النشرة عن باقي المستودع بالكامل (اعتماديات، حجم، إلخ).
3. متغيرات البيئة المطلوبة (Project Settings → Environment Variables):

   | المتغير | المصدر |
   |---|---|
   | `WHATSAPP_VERIFY_TOKEN` | تختاره أنت (نص عشوائي)، يُدخَل بلوحة Meta أيضاً |
   | `WHATSAPP_ACCESS_TOKEN` | من Meta Business/App |
   | `WHATSAPP_PHONE_NUMBER_ID` | من لوحة Meta Cloud API |
   | `UPSTASH_REDIS_REST_URL` | من لوحة Upstash |
   | `UPSTASH_REDIS_REST_TOKEN` | من لوحة Upstash |

   **⚠️ نفس متغيرات Upstash يجب ضبطها أيضاً بإعدادات Streamlit** (Secrets)
   لتفعيل مزامنة `ai/accounts.py` — بدونها، فرع "حالة حسابي" بواتساب
   يبقى يرد "غير مرتبط" لكل الأرقام حتى لو كانت مسجّلة فعلاً.

4. الرابط الناتج من Vercel (مثال: `https://xxx.vercel.app/api/webhook`)
   يُدخَل بلوحة Meta Developer كـ Webhook URL، مع نفس قيمة
   `WHATSAPP_VERIFY_TOKEN`.

## نطاق متعمَّد (لا تفاجأ بالمحدودية)

- بحث نص آية بالرقم فقط (`سورة:آية`) — لا تفسير، لا بحث دلالي مفتوح.
- حالة حساب مباشرة (اسم المستخدم + تاريخ التسجيل) — لا تفاصيل حساب أخرى.
- أي رسالة خارج هذين المسارين تُعاد توجيهها للقائمة، لا تُفسَّر بحرية.

هذا مطابق عمداً لسياسة Meta (يناير 2026): بوتات WhatsApp Business API
يجب أن تكون محددة المهام، لا محادثة عامة مفتوحة.

## لماذا الملفات مكرَّرة عن `ai/whatsapp/` بجذر المستودع؟

لأن Vercel بجذر `whatsapp_gateway/` لا يرى أي شي خارج هذا المجلد.
أي تعديل منطقي على `router.py`/`quran_lookup.py`/`state_store.py`/
`whatsapp_client.py` هنا يجب تكراره يدوياً بالنسخة المقابلة تحت
`ai/whatsapp/` بالمستودع الرئيسي (وأصلاً تُستخدم من هناك فقط للاختبار
المحلي السريع بدون الحاجة لنشرة Vercel كاملة في كل تعديل).
