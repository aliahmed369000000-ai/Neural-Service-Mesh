"""
Neural Service Mesh — FastAPI Backend
خادم API لمشروع النظام المعرفي العربي
يعمل على المنفذ 5000 عبر Uvicorn
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# إضافة مسار المشروع
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# محاولة استيراد المكونات الداخلية
try:
    from core.engine import Engine
    from core.registry import Registry
    _CORE_OK = True
except Exception as _e:
    _CORE_OK = False
    _CORE_ERR = str(_e)

app = FastAPI(
    title="Neural Service Mesh API",
    description="واجهة برمجية للنظام المعرفي العربي",
    version="1.0.0",
)

# السماح بجميع الاتصالات (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "Neural Service Mesh API", "version": "1.0.0"}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "core_available": _CORE_OK,
        "core_error": None if _CORE_OK else _CORE_ERR,
    }


@app.post("/process")
async def process(payload: dict):
    """معالجة النص عبر شبكة الخدمات"""
    if not _CORE_OK:
        return JSONResponse(
            status_code=503,
            content={"error": "Core engine not available", "detail": _CORE_ERR},
        )
    try:
        engine = Engine()
        result = engine.process(payload)
        return {"status": "ok", "result": result}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/webhook/telegram/{secret}")
async def telegram_webhook(secret: str, request: Request):
    """نقطة استقبال webhook تيليجرام (بديل عن getUpdates عند تفعيله عبر
    SocialAgentManager.enable_webhook). التحقق مزدوج قبل أي معالجة:
    1) الجزء السري في المسار نفسه يجب أن يطابق TELEGRAM_WEBHOOK_SECRET.
    2) رأس X-Telegram-Bot-Api-Secret-Token (يرسله تيليجرام تلقائياً مع
       كل طلب بعد تفعيل webhook بـsecret_token) يجب أن يطابقه أيضاً.
    أي طلب لا يطابق الاثنين يُرفض بـ403 دون قراءة الجسم أو لمس قاعدة
    البيانات — نفس مبدأ 'لا بيانات مزيّفة، لا معالجة غير موثّقة' المتّبع
    بكل محولات المنصة."""
    configured_secret = os.environ.get("TELEGRAM_WEBHOOK_SECRET")
    header_secret = request.headers.get("x-telegram-bot-api-secret-token")
    if not configured_secret or secret != configured_secret or header_secret != configured_secret:
        return JSONResponse(status_code=403, content={"error": "secret غير مطابق"})

    try:
        payload = await request.json()
    except Exception:
        return JSONResponse(status_code=400, content={"error": "جسم JSON غير صالح"})

    try:
        from ai.social_agent import get_manager
        from ai.social_platforms.telegram_adapter import TelegramAdapter

        item = TelegramAdapter._parse_update(payload)
        if item is not None:
            get_manager().ingest_webhook_item("telegram", item)
    except Exception as e:
        # تيليجرام يعيد المحاولة/يعطّل الـwebhook عند رؤية أخطاء متكررة —
        # نرجّع 200 دائماً ونسجّل الخطأ فقط، بنفس نمط whatsapp_gateway.
        return JSONResponse(status_code=200, content={"status": "ok", "warning": str(e)})

    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=5000, reload=True)
