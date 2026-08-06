"""
Neural Service Mesh — FastAPI Backend
خادم API لمشروع النظام المعرفي العربي
يعمل على المنفذ 5000 عبر Uvicorn
"""

from __future__ import annotations

import hmac
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
    from core.engine import ExecutionEngine as Engine
    from core.registry import NodeRegistry as Registry
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
async def process(payload: dict, request: Request):
    """معالجة النص عبر شبكة الخدمات.
    محمي بمفتاح NSM_API_KEY (هيدر X-API-Key) — fail-closed: لو المفتاح
    غير مضبوط بالبيئة، الـendpoint يبقى معطّلاً بالكامل (403) بدل ما
    يُترك مفتوحاً بلا مصادقة افتراضياً. هذا endpoint حالياً غير فعّال
    عملياً (Engine() تحتاج registry/graph/storage غير مُمرَّرة هنا)،
    لكن الحماية أُضيفت استباقياً حتى لا يصير باباً مفتوحاً بلا مصادقة
    بمجرد ما يُكمَّل ربطه مستقبلاً."""
    configured_key = os.environ.get("NSM_API_KEY", "").strip()
    provided_key = request.headers.get("x-api-key", "")
    if not configured_key or not hmac.compare_digest(provided_key, configured_key):
        return JSONResponse(status_code=403, content={"error": "X-API-Key غير مطابق أو NSM_API_KEY غير مضبوط"})

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
    header_secret = request.headers.get("x-telegram-bot-api-secret-token") or ""
    path_ok = bool(configured_secret) and hmac.compare_digest(secret, configured_secret)
    header_ok = bool(configured_secret) and hmac.compare_digest(header_secret, configured_secret)
    if not (path_ok and header_ok):
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


@app.get("/webhook/whatsapp")
async def whatsapp_webhook_verify(request: Request):
    """تحقق Meta الأولي عند ربط الـwebhook (hub.mode/hub.verify_token/
    hub.challenge) — يجب إرجاع hub.challenge كنص خام إن تطابق الرمز."""
    from ai.social_platforms.whatsapp_adapter import WhatsAppAdapter

    q = request.query_params
    result = WhatsAppAdapter.verify_webhook_challenge(
        q.get("hub.mode"), q.get("hub.verify_token"), q.get("hub.challenge"),
    )
    if result is None:
        return JSONResponse(status_code=403, content={"error": "verify_token غير مطابق"})
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(result)


@app.post("/webhook/whatsapp")
async def whatsapp_webhook_receive(request: Request):
    """استقبال رسائل واتساب الواردة. تحقق التوقيع (X-Hub-Signature-256)
    إلزامي قبل أي معالجة أو حتى قبل قراءة JSON — نفس مبدأ الرفض الآمن
    المتّبع بمحول تيليجرام. نرد 200 لـMeta دائماً بعد التحقق الناجح حتى
    لو فشلت المعالجة الداخلية، تفادياً لتعطيل الـwebhook من طرف Meta."""
    from ai.social_platforms.whatsapp_adapter import WhatsAppAdapter

    raw_body = await request.body()
    signature = request.headers.get("x-hub-signature-256")
    if not WhatsAppAdapter.verify_signature(raw_body, signature):
        return JSONResponse(status_code=403, content={"error": "توقيع غير صالح"})

    try:
        import json as _json
        payload = _json.loads(raw_body)
    except Exception:
        return JSONResponse(status_code=400, content={"error": "جسم JSON غير صالح"})

    try:
        from ai.social_agent import get_manager

        items = WhatsAppAdapter.parse_webhook_payload(payload)
        for item in items:
            WhatsAppAdapter.enqueue_incoming(item)
            get_manager().ingest_webhook_item("whatsapp", item)
    except Exception as e:
        return JSONResponse(status_code=200, content={"status": "ok", "warning": str(e)})

    return {"status": "ok"}




# ── Model Training Agent: registry + experimental inference ───────────────
@app.get("/training/registry")
def training_registry():
    try:
        from ai.training_feedback_loop import load_registry, registry_report
        return {"ok": True, "registry": load_registry(), "report": registry_report()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/training/champion")
def training_champion():
    try:
        from ai.training_feedback_loop import get_champion
        ch = get_champion()
        if not ch:
            return {"ok": False, "error": "no champion"}
        return {"ok": True, "champion": ch}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/training/predict")
async def training_predict(request: Request):
    """استدلال تجريبي على بطل السجل — body: {"features": [..]}"""
    try:
        body = await request.json()
    except Exception:
        body = {}
    features = body.get("features") if isinstance(body, dict) else None
    try:
        from ai.training_feedback_loop import predict_with_champion_demo
        text = predict_with_champion_demo(features=features)
        return {"ok": True, "result": text}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/training/drift")
def training_drift_status():
    try:
        from pathlib import Path as _P
        import json as _json
        base = _P("artifacts/model_training/drift/baseline.json")
        last = _P("artifacts/model_training/drift/last_check.json")
        return {
            "ok": True,
            "baseline": _json.loads(base.read_text()) if base.is_file() else None,
            "last_check": _json.loads(last.read_text()) if last.is_file() else None,
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)



# ── AIaaS: multi-tenant training as a service ─────────────────────────────
@app.get("/aiaas/status")
def aiaas_status():
    try:
        from ai.aiaas_platform import platform_status, load_tenants_index, PLANS, DOMAINS
        return {
            "ok": True,
            "tenants": len((load_tenants_index().get("tenants") or {})),
            "plans": PLANS,
            "domains": {k: v.get("status") for k, v in DOMAINS.items()},
            "report": platform_status(),
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/aiaas/tenants")
async def aiaas_create_tenant(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from ai.aiaas_platform import create_tenant
        rec = create_tenant(
            name=str(body.get("name") or "api-tenant"),
            plan=str(body.get("plan") or "free"),
            email=str(body.get("email") or ""),
        )
        return {"ok": True, "tenant": rec}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/aiaas/jobs")
async def aiaas_run_job(request: Request):
    """Header: X-API-Key: nsm_...  Body: {domain, epochs?, goal?}"""
    api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key") or ""
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from ai.aiaas_platform import authenticate_api_key, run_tenant_job
        ten = authenticate_api_key(api_key)
        if not ten:
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        job = run_tenant_job(
            ten["id"],
            domain=str(body.get("domain") or "tabular_classification"),
            epochs=body.get("epochs"),
            goal=body.get("goal"),
        )
        return job
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/aiaas/invoice/{tenant_id}")
def aiaas_invoice(tenant_id: str):
    try:
        from ai.aiaas_platform import estimate_invoice
        return {"ok": True, "invoice": estimate_invoice(tenant_id)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=5000, reload=True)
