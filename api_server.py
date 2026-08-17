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


@app.post("/training/remote-results")
async def training_remote_results(request: Request):
    """استقبال نتائج تدريب من Colab/عقدة بعيدة (ميتا JSON)."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "invalid json"}, status_code=400)
    try:
        import os
        from ai.remote_gpu_provider import ingest_remote_package
        secret = os.environ.get("NSM_REMOTE_WEBHOOK_SECRET") or ""
        result = ingest_remote_package(body if isinstance(body, dict) else {}, expected_secret=secret)
        code = 200 if result.get("ok") else 401
        return JSONResponse(result, status_code=code)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/training/remote-status")
def training_remote_status():
    try:
        from ai.remote_gpu_provider import remote_status_report, get_provider
        return {
            "ok": True,
            "report": remote_status_report(),
            "active_provider": get_provider().status(),
        }
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

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




@app.post("/aiaas/upload")
async def aiaas_upload(request: Request):
    """رفع CSV لمساحة المستأجر. Header X-API-Key. multipart: file"""
    api_key = request.headers.get("X-API-Key") or request.headers.get("x-api-key") or ""
    try:
        from ai.aiaas_platform import authenticate_api_key, save_tenant_upload
        ten = authenticate_api_key(api_key)
        if not ten:
            return JSONResponse({"ok": False, "error": "unauthorized"}, status_code=401)
        form = await request.form()
        f = form.get("file")
        if f is None:
            return JSONResponse({"ok": False, "error": "file required"}, status_code=400)
        content = await f.read()
        name = getattr(f, "filename", None) or "upload.csv"
        rel = save_tenant_upload(ten["id"], str(name), content)
        return {"ok": True, "path": rel, "tenant_id": ten["id"]}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/aiaas/invoice/{tenant_id}")
def aiaas_invoice(tenant_id: str):
    try:
        from ai.aiaas_platform import estimate_invoice
        return {"ok": True, "invoice": estimate_invoice(tenant_id)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ── نقاط لوحة السرب والوكلاء (NSM Agent / Swarm API) ─────────────
def _nsm_key_check(request: Request):
    """فحص fail-closed لمفتاح NSM_API_KEY عبر X-API-Key — نفس نمط /process:
    لو المفتاح غير مضبوط بالبيئة تُرفض كل النقاط بـ403 بدل أن تبقى
    مفتوحة بلا مصادقة."""
    configured_key = os.environ.get("NSM_API_KEY", "").strip()
    provided_key = request.headers.get("x-api-key", "")
    if not configured_key or not hmac.compare_digest(provided_key,
                                                      configured_key):
        return JSONResponse(
            status_code=403,
            content={"error": "X-API-Key غير مطابق أو NSM_API_KEY غير مضبوط"},
        )
    return None


@app.get("/agents/states")
async def agents_states(request: Request):
    """أحدث حالة لكل وكيل + ملخص زمن الاستجابة من ناقل الأحداث."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.agent_event_bus import (
            current_agent_states, get_events, performance_summary)
        events = get_events(200)
        return {"ok": True, "agents": current_agent_states(events),
                "performance": performance_summary(events)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/agents/events")
async def agents_events(request: Request):
    """آخر الأحداث في ناقل الأحداث (limit قابل للتعديل بين 1 و500)."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        limit = min(max(int(request.query_params.get("limit", "80")), 1), 500)
        from ai.agent_event_bus import get_events
        return {"ok": True, "events": get_events(limit)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/swarm/dashboard")
async def swarm_dashboard(request: Request):
    """لقطة لوحة السرب الموحدة: وكلاء + سرب + مهام طويلة + أداء + تنبيهات."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.unified_swarm_dashboard import unified_dashboard_snapshot
        return {"ok": True, "dashboard": unified_dashboard_snapshot()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/swarm/alerts")
async def swarm_alerts(request: Request):
    """تقييم التنبيهات وفق القواعد المخصصة الحالية."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.unified_swarm_dashboard import evaluate_alerts
        return {"ok": True, "alerts": evaluate_alerts()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.post("/swarm/apply-actions")
async def swarm_apply_actions(request: Request):
    """تطبيق الإجراءات التلقائية المفعّلة على التنبيهات الحالية."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.unified_swarm_dashboard import (
            apply_auto_actions, evaluate_alerts)
        applied = apply_auto_actions(evaluate_alerts())
        return {"ok": True, "applied": applied, "count": len(applied)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/swarm/long-horizon")
async def swarm_long_horizon(request: Request):
    """المهام طويلة الأمد قيد الإدارة."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.unified_swarm_dashboard import long_horizon_status
        return {"ok": True, "long_horizon": long_horizon_status()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/mesh/history")
async def mesh_history(request: Request):
    """ملخص نظام السرب من MeshBundle (تقييم/ذاكرة/سمعة/سجل تنفيذ مدمج)."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from core.mesh_bundle import get_mesh_bundle
        bundle = get_mesh_bundle()
        return {"ok": True, "summary": bundle.summary(),
                "limit_note": ("MeshBundle يقدّم ملخصًا مجمعًا للتقييم/"
                               "الذاكرة/السمعة عبر summary() — لا يسجّل "
                               "قائمة سجلات تنفيذ فردية للاستعلام المباشر")}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/performance/system")
async def performance_system(request: Request):
    """مؤشرات أداء النظام: استخدام الذاكرة / ذروة RSS / حِمْل النظام."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.unified_swarm_dashboard import system_performance
        return {"ok": True, "system": system_performance()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


# ---------------------------------------------------------------- Backend
@app.get("/backend/counts")
async def backend_counts(request: Request):
    """عدّادات جداول مركز البيانات (KV/وكلاء/مهام/ذاكرة/رسائل)."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import backend_counts as _counts
        return {"ok": True, "counts": _counts()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/kv")
async def backend_kv_get(request: Request):
    """قراءة مفتاح من مخزن KV: ?key=&domain=general&default="""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import kv_get, kv_list
        key = request.query_params.get("key", "")
        domain = request.query_params.get("domain", "general")
        if not key:
            return {"ok": True, "list": kv_list(
                domain=domain if domain != "general" else None,
                limit=int(request.query_params.get("limit", "100")))}
        return {"ok": True, "key": key, "domain": domain,
                "value": kv_get(key, domain)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/backend/kv")
async def backend_kv_set(request: Request):
    """حفظ مفتاح/قيمة: {"key", "value", "domain"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        key = str(body.get("key", ""))
        if not key:
            return JSONResponse({"ok": False, "error": "key مطلوب"},
                                status_code=400)
        from ai.backend_layer import kv_set
        return {"ok": True, **kv_set(key, body.get("value", ""),
                                     str(body.get("domain", "general")))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.delete("/backend/kv")
async def backend_kv_delete(request: Request):
    """حذف مفتاح: ?key=&domain=general"""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import kv_delete
        key = request.query_params.get("key", "")
        if not key:
            return JSONResponse({"ok": False, "error": "key مطلوب"},
                                status_code=400)
        return {"ok": True, **kv_delete(key,
                                        request.query_params.get("domain",
                                                                 "general"))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/agents")
async def backend_agents_list(request: Request):
    """قائمة الوكلاء المسجلين في مركز البيانات."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import agent_list
        return {"ok": True, "agents": agent_list(
            limit=int(request.query_params.get("limit", "100")))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/backend/agents")
async def backend_agents_register(request: Request):
    """تسجيل وكيل: {"id", "role", "config"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        agent_id = str(body.get("id", ""))
        if not agent_id:
            return JSONResponse({"ok": False, "error": "id مطلوب"},
                                status_code=400)
        from ai.backend_layer import agent_register
        return {"ok": True, **agent_register(agent_id,
                                             str(body.get("role", "")),
                                             body.get("config"))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/agents/{agent_id}")
async def backend_agent_get(request: Request, agent_id: str):
    """تفاصيل وكيل واحد."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import agent_get
        agent = agent_get(agent_id)
        if agent is None:
            return JSONResponse({"ok": False, "error": "وكيل غير موجود"},
                                status_code=404)
        return {"ok": True, "agent": agent}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.put("/backend/agents/{agent_id}")
async def backend_agent_update(request: Request, agent_id: str):
    """تحديث وكيل: {"role", "status", "config"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        from ai.backend_layer import agent_update
        result = agent_update(agent_id, body)
        if not result.get("updated"):
            return JSONResponse({"ok": False,
                                 "error": "فشل التحديث (وكيل غير موجود?)"},
                                status_code=404)
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.delete("/backend/agents/{agent_id}")
async def backend_agent_unregister(request: Request, agent_id: str):
    """إزالة تسجيل وكيل."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import agent_unregister
        return {"ok": True, **agent_unregister(agent_id)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/tasks")
async def backend_tasks_list(request: Request):
    """قائمة المهام (تصفية اختيارية ?status=)."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import task_list
        status = request.query_params.get("status") or None
        return {"ok": True, "tasks": task_list(
            status=status, limit=int(request.query_params.get("limit", "100")))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/backend/tasks")
async def backend_tasks_create(request: Request):
    """إنشاء مهمة: {"title", "type", "payload"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        from ai.backend_layer import task_create
        return {"ok": True, **task_create(str(body.get("title", "")),
                                          str(body.get("type", "general")),
                                          body.get("payload"))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/tasks/{task_id}")
async def backend_task_get(request: Request, task_id: str):
    """تفاصيل مهمة واحدة."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import task_get
        task = task_get(task_id)
        if task is None:
            return JSONResponse({"ok": False, "error": "مهمة غير موجودة"},
                                status_code=404)
        return {"ok": True, "task": task}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.put("/backend/tasks/{task_id}")
async def backend_task_update(request: Request, task_id: str):
    """تحديث مهمة: {"status", "result", "payload", "title"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        from ai.backend_layer import task_update
        result = task_update(task_id, body)
        if not result.get("updated"):
            return JSONResponse({"ok": False,
                                 "error": "فشل التحديث (مهمة غير موجودة?)"},
                                status_code=404)
        return {"ok": True, **result}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/memories")
async def backend_memories(request: Request):
    """الذاكرة: ?q= بحث نصي أو قائمة ?limit=100."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import memory_list, memory_search
        q = request.query_params.get("q")
        if q:
            return {"ok": True, "memories": memory_search(q, limit=25)}
        return {"ok": True, "memories": memory_list(
            limit=int(request.query_params.get("limit", "100")))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/backend/memories")
async def backend_memory_add(request: Request):
    """إضافة ذاكرة: {"subject", "content", "tags", "importance"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        if not body.get("subject"):
            return JSONResponse({"ok": False, "error": "subject مطلوب"},
                                status_code=400)
        from ai.backend_layer import memory_add
        return {"ok": True, **memory_add(
            str(body.get("subject")), str(body.get("content", "")),
            body.get("tags"), float(body.get("importance", 0.5)))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.get("/backend/messages")
async def backend_messages(request: Request):
    """صندوق الوارد: ?receiver=&unread_only=&limit=50"""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import message_inbox
        receiver = request.query_params.get("receiver", "")
        if not receiver:
            return JSONResponse({"ok": False, "error": "receiver مطلوب"},
                                status_code=400)
        return {"ok": True, "messages": message_inbox(
            receiver,
            limit=int(request.query_params.get("limit", "50")),
            unread_only=request.query_params.get("unread_only") == "1")}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.post("/backend/messages")
async def backend_message_send(request: Request):
    """إرسال رسالة: {"sender", "receiver", "subject", "body", "headers"}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
        if not body.get("receiver"):
            return JSONResponse({"ok": False, "error": "receiver مطلوب"},
                                status_code=400)
        from ai.backend_layer import message_send
        return {"ok": True, **message_send(
            str(body.get("sender", "")), str(body["receiver"]),
            str(body.get("subject", "")), str(body.get("body", "")),
            body.get("headers"))}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

@app.put("/backend/messages/{msg_id}/read")
async def backend_message_mark_read(request: Request, msg_id: str):
    """وضع علامة مقروء على رسالة."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.backend_layer import message_mark_read
        return {"ok": True, **message_mark_read(msg_id)}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)

# ── Microservices & External Connectors ──────────────────────────
@app.get("/services/list")
async def services_list(request: Request):
    """قائمة الخدمات المصغرة المسجّلة (Microservices Layer)."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.microservices import list_services
        return {"ok": True, "services": list_services(),
                "schema_version": "nsm-ms/1.0"}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=500)

@app.get("/services/describe")
async def services_describe(request: Request):
    """وصف خدمة: ?service=meta أو backend أو connectors..."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from ai.microservices import call_service
        return call_service("meta", "describe_service",
                            {"service": request.query_params.get(
                                "service", "")})
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=500)

@app.post("/services/call")
async def services_call(request: Request):
    """استدعاء خدمة داخلية بنمط الطلب/الاستجابة الثابت:
    {"service": "backend", "action": "counts", "payload": {...}}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from ai.microservices import call_service
        return call_service(str(body.get("service", "")),
                            str(body.get("action", "")),
                            body.get("payload"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=500)

@app.get("/connectors/list")
async def connectors_list(request: Request):
    """الموصلات الخارجية المسجّلة وقدراتها (دفع/خرائط/رسائل)."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from connectors.external_services import list_connectors
        return {"ok": True, "connectors": list_connectors()}
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=500)

@app.get("/connectors/describe")
async def connectors_describe(request: Request):
    """وصف موصل وإجراءاته: ?service=payment أو maps أو sms."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        from connectors.external_services import describe_connector
        return describe_connector(
            str(request.query_params.get("service", "")))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=500)

@app.post("/connectors/call")
async def connectors_call(request: Request):
    """استدعاء إجراء على موصل خارجي (محاكاة الآن):
    {"service": "sms", "action": "send_otp", "payload": {"to": "+966..."}}."""
    auth = _nsm_key_check(request)
    if auth is not None:
        return auth
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from connectors.external_services import call_connector
        return call_connector(str(body.get("service", "")),
                              str(body.get("action", "")),
                              body.get("payload"))
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)},
                            status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api_server:app", host="0.0.0.0", port=5000, reload=True)


@app.post("/billing/checkout")
async def billing_checkout(request: Request):
    try:
        body = await request.json()
    except Exception:
        body = {}
    try:
        from ai.stripe_billing import create_checkout_session
        return create_checkout_session(
            plan=str(body.get("plan") or "pro"),
            success_url=str(body.get("success_url") or "https://example.com/success"),
            cancel_url=str(body.get("cancel_url") or "https://example.com/cancel"),
            customer_email=body.get("email"),
        )
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=500)


@app.get("/gtm/status")
def gtm_status():
    """حالة جاهزية Go-to-Market."""
    from pathlib import Path as _P
    return {
        "ok": True,
        "docker_compose_prod": _P("docker-compose.prod.yml").is_file(),
        "stripe_configured": bool(os.environ.get("STRIPE_SECRET_KEY")),
        "mcp_server": _P("mcp_server/server.py").is_file(),
        "social_daemon": _P("scripts/social_swarm_daemon.py").is_file(),
    }
