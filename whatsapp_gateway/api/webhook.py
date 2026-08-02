"""
api/webhook.py — نقطة الدخول الفعلية على Vercel لبوابة واتساب.

GET  → التحقق الأولي من Meta عند ربط الرقم بلوحة التحكم.
POST → استقبال رسالة واردة، معالجتها عبر lib/router.py، وإرسال الرد.

مسؤولية هذا الملف: "الغراء" فقط (تحليل HTTP، قراءة متغيرات البيئة عبر
lib/*، إرجاع استجابة صحيحة لـMeta) — كل المنطق الفعلي بـlib/.

بنية المجلد (Vercel "Root Directory" = whatsapp_gateway/ بإعدادات
المشروع على لوحة Vercel، منفصل تماماً عن مشروع Streamlit الرئيسي):
    whatsapp_gateway/
        api/webhook.py      ← هذا الملف
        lib/                ← المنطق (router, quran_lookup, state_store, ...)
        knowledge/           ← بيانات قرآن خفيفة فقط (6.4 م.ب)
        requirements.txt     ← "requests" فقط، معزول عن requirements.txt الرئيسي
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# أمان إضافي: نضمن وجود جذر whatsapp_gateway/ بمسار الاستيراد بغض النظر
# عن دليل العمل الفعلي وقت تشغيل الدالة على Vercel (بدل الاعتماد ضمنياً
# على سلوك افتراضي غير موثّق بدقة كافية).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.router import handle_incoming_message
from lib.state_store import get_state_store
from lib.whatsapp_client import (
    verify_webhook_challenge,
    verify_signature,
    send_text_message,
    extract_incoming_message,
    WhatsAppSendError,
)


class handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        query = parse_qs(urlparse(self.path).query)
        mode = query.get("hub.mode", [None])[0]
        token = query.get("hub.verify_token", [None])[0]
        challenge = query.get("hub.challenge", [None])[0]

        result = verify_webhook_challenge(mode, token, challenge)
        if result is not None:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(result.encode("utf-8"))
        else:
            self.send_response(403)
            self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b"{}"

        signature = self.headers.get("X-Hub-Signature-256")
        if not verify_signature(raw_body, signature):
            self.send_response(403)
            self.end_headers()
            return

        # نرد 200 لـMeta دائماً بأسرع وقت حتى لو تعذّرت المعالجة الداخلية
        # (Meta تُعيد المحاولة/تعطّل الـwebhook لو رأت أخطاء متكررة) —
        # أي خطأ داخلي يُسجَّل فقط، لا يُرجَع كـHTTP error لـMeta.
        # (هذا الرد يأتي بعد التحقق من التوقيع أعلاه — طلب غير موقَّع
        # بشكل صحيح يُرفض بـ403 ولا يصل لهذه النقطة إطلاقاً.)
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return

        extracted = extract_incoming_message(payload)
        if extracted is None:
            return  # حدث ليس رسالة نصية واردة (تسليم/وسائط) — تجاهل صامت متعمَّد

        phone, text = extracted
        store = get_state_store()
        current_state = store.get(phone)
        reply_text, new_state = handle_incoming_message(phone, text, current_state)
        store.set(phone, new_state)

        try:
            send_text_message(phone, reply_text)
        except WhatsAppSendError:
            # فشل الإرسال يعني المستخدم لن يستلم رداً — لا يوجد شي إضافي
            # نقدر نسويه هنا (لا قناة إعادة محاولة حالياً)، يُسجَّل ضمنياً
            # عبر سجلات Vercel Function logs.
            pass
