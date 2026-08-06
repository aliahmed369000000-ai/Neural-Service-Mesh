"""
AIaaS Platform — الذكاء الاصطناعي كخدمة فوق وكيل التدريب
========================================================
  • Multi-tenancy: عزل بيانات/نماذج كل عميل تحت tenants/<id>/
  • Monetization: خطط اشتراك + قياس استهلاك (ساعات تقريبية، مهام، نماذج)
  • Domains: سجل مجالات مدعومة (NLP جدولي، نص، رؤية تجريبية، …)
  • Self-Evolution: مراقبة أداء المصنع + اقتراحات ترقية (موافقة بشرية)

لا يفرض بوابة دفع حقيقية — يوفر حصصاً وفواتير استخدام قابلة للربط لاحقاً.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("AIaaS")

ROOT = Path(__file__).resolve().parent.parent
AIAAS = ROOT / "artifacts" / "model_training" / "aiaas"
TENANTS = AIAAS / "tenants"
USAGE = AIAAS / "usage"
EVOLVE = AIAAS / "evolution"
for d in (AIAAS, TENANTS, USAGE, EVOLVE):
    d.mkdir(parents=True, exist_ok=True)

# خطط اشتراك افتراضية (وحدات منطقية — ليست فوترة حقيقية)
PLANS: Dict[str, Dict[str, Any]] = {
    "free": {
        "name": "Free",
        "price_usd_month": 0,
        "max_jobs_per_day": 5,
        "max_models": 3,
        "max_upload_mb": 5,
        "max_epochs": 20,
        "concurrent_jobs": 1,
    },
    "pro": {
        "name": "Pro",
        "price_usd_month": 49,
        "max_jobs_per_day": 50,
        "max_models": 30,
        "max_upload_mb": 50,
        "max_epochs": 50,
        "concurrent_jobs": 3,
    },
    "enterprise": {
        "name": "Enterprise",
        "price_usd_month": 299,
        "max_jobs_per_day": 500,
        "max_models": 500,
        "max_upload_mb": 500,
        "max_epochs": 100,
        "concurrent_jobs": 20,
    },
}

DOMAINS: Dict[str, Dict[str, Any]] = {
    "tabular_classification": {
        "title": "تصنيف جدولي",
        "status": "ga",
        "prefer": "torch",
        "sample": "data/samples/classification_demo.csv",
    },
    "tabular_regression": {
        "title": "انحدار جدولي",
        "status": "ga",
        "prefer": "torch",
        "sample": "data/samples/regression_demo.csv",
    },
    "nlp_text_classification": {
        "title": "تصنيف نصوص / مشاعر",
        "status": "ga",
        "prefer": "text",
        "sample": "data/samples/text_sentiment_demo.csv",
    },
    "arabic_dialects": {
        "title": "اللهجات العربية (تجريبي)",
        "status": "beta",
        "prefer": "text",
        "sample": "data/samples/text_sentiment_demo.csv",
    },
    "computer_vision": {
        "title": "رؤية حاسوبية",
        "status": "planned",
        "prefer": "cnn",
        "sample": None,
    },
    "finance_forecast": {
        "title": "تنبؤ مالي",
        "status": "planned",
        "prefer": "torch",
        "sample": None,
    },
    "image_generation": {
        "title": "توليد صور",
        "status": "planned",
        "prefer": None,
        "sample": None,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tenant_dir(tenant_id: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", tenant_id)[:64]
    d = TENANTS / safe
    for sub in ("data", "models", "jobs", "logs"):
        (d / sub).mkdir(parents=True, exist_ok=True)
    return d


def _tenants_index() -> Path:
    return AIAAS / "tenants_index.json"


def load_tenants_index() -> Dict[str, Any]:
    p = _tenants_index()
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"tenants": {}}


def save_tenants_index(idx: Dict[str, Any]) -> None:
    _tenants_index().write_text(json.dumps(idx, ensure_ascii=False, indent=2), encoding="utf-8")


def create_tenant(name: str, plan: str = "free", email: str = "") -> Dict[str, Any]:
    plan = plan if plan in PLANS else "free"
    tid = f"ten_{uuid.uuid4().hex[:10]}"
    api_key = f"nsm_{secrets.token_urlsafe(24)}"
    key_hash = hashlib.sha256(api_key.encode()).hexdigest()
    rec = {
        "id": tid,
        "name": name or tid,
        "email": email,
        "plan": plan,
        "api_key_hash": key_hash,
        "api_key_prefix": api_key[:12] + "…",
        "created_at": _now(),
        "usage": {"jobs_total": 0, "models_total": 0, "train_seconds": 0.0, "jobs_today": 0, "day": ""},
    }
    _tenant_dir(tid)
    idx = load_tenants_index()
    idx.setdefault("tenants", {})[tid] = {k: v for k, v in rec.items() if k != "api_key_hash"}
    idx["tenants"][tid]["api_key_hash"] = key_hash
    save_tenants_index(idx)
    # أعد المفتاح مرة واحدة فقط للمستدعي
    out = dict(rec)
    out["api_key"] = api_key
    return out


def get_tenant(tenant_id: str) -> Optional[Dict[str, Any]]:
    return (load_tenants_index().get("tenants") or {}).get(tenant_id)


def authenticate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    if not api_key:
        return None
    h = hashlib.sha256(api_key.encode()).hexdigest()
    for tid, rec in (load_tenants_index().get("tenants") or {}).items():
        if rec.get("api_key_hash") == h:
            return {**rec, "id": tid}
    return None


def _reset_daily_if_needed(rec: Dict[str, Any]) -> None:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    usage = rec.setdefault("usage", {})
    if usage.get("day") != day:
        usage["day"] = day
        usage["jobs_today"] = 0


def check_quota(tenant_id: str) -> Tuple[bool, str]:
    rec = get_tenant(tenant_id)
    if not rec:
        return False, "مستأجر غير موجود"
    _reset_daily_if_needed(rec)
    plan = PLANS.get(rec.get("plan") or "free", PLANS["free"])
    usage = rec.get("usage") or {}
    if int(usage.get("jobs_today") or 0) >= int(plan["max_jobs_per_day"]):
        return False, f"تجاوز حد المهام اليومية ({plan['max_jobs_per_day']}) لخطة {plan['name']}"
    if int(usage.get("models_total") or 0) >= int(plan["max_models"]):
        return False, f"تجاوز حد النماذج ({plan['max_models']})"
    return True, "ok"


def record_usage(tenant_id: str, train_seconds: float = 0.0, models_delta: int = 0, job: bool = True) -> None:
    idx = load_tenants_index()
    rec = (idx.get("tenants") or {}).get(tenant_id)
    if not rec:
        return
    _reset_daily_if_needed(rec)
    usage = rec.setdefault("usage", {})
    if job:
        usage["jobs_total"] = int(usage.get("jobs_total") or 0) + 1
        usage["jobs_today"] = int(usage.get("jobs_today") or 0) + 1
    usage["train_seconds"] = float(usage.get("train_seconds") or 0) + float(train_seconds)
    usage["models_total"] = int(usage.get("models_total") or 0) + int(models_delta)
    # approximate cost units: second of training
    usage["billable_units"] = round(float(usage.get("train_seconds") or 0) / 60.0, 3)  # دقائق
    save_tenants_index(idx)
    # usage log
    logp = USAGE / f"{tenant_id}_{int(time.time())}.json"
    logp.write_text(
        json.dumps(
            {
                "tenant_id": tenant_id,
                "ts": _now(),
                "train_seconds": train_seconds,
                "models_delta": models_delta,
                "job": job,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )



def save_tenant_upload(tenant_id: str, filename: str, content: bytes) -> str:
    """حفظ ملف مرفوع داخل مساحة المستأجر فقط."""
    if not get_tenant(tenant_id):
        raise ValueError("مستأجر غير موجود")
    plan = PLANS.get((get_tenant(tenant_id) or {}).get("plan") or "free", PLANS["free"])
    max_mb = float(plan.get("max_upload_mb") or 5)
    if len(content) > max_mb * 1024 * 1024:
        raise ValueError(f"الملف أكبر من حد الخطة ({max_mb} MB)")
    safe = re.sub(r"[^a-zA-Z0-9._\-]", "_", filename)[:120]
    if not safe.lower().endswith(".csv"):
        safe += ".csv"
    dest = _tenant_dir(tenant_id) / "data" / safe
    dest.write_bytes(content)
    return str(dest.relative_to(ROOT))


def estimate_invoice(tenant_id: str) -> Dict[str, Any]:
    rec = get_tenant(tenant_id) or {}
    plan_key = rec.get("plan") or "free"
    plan = PLANS[plan_key]
    usage = rec.get("usage") or {}
    minutes = float(usage.get("billable_units") or 0)
    # تسعير بسيط: اشتراك + 0.05$ لكل دقيقة تدريب فوق الخطة المجانية
    overage = max(0.0, minutes - (30 if plan_key == "free" else 300)) * 0.05
    return {
        "tenant_id": tenant_id,
        "plan": plan_key,
        "subscription_usd": plan["price_usd_month"],
        "train_minutes": minutes,
        "overage_usd": round(overage, 2),
        "estimated_total_usd": round(plan["price_usd_month"] + overage, 2),
        "note": "تقدير داخلي — ليس تحصيل دفع فعلي",
    }


def run_tenant_job(
    tenant_id: str,
    domain: str = "tabular_classification",
    dataset_rel: Optional[str] = None,
    epochs: Optional[int] = None,
    goal: Optional[str] = None,
) -> Dict[str, Any]:
    """تشغيل مهمة تدريب معزولة للمستأجر."""
    ok, msg = check_quota(tenant_id)
    if not ok:
        return {"ok": False, "error": msg}

    rec = get_tenant(tenant_id)
    plan = PLANS.get((rec or {}).get("plan") or "free", PLANS["free"])
    domain_info = DOMAINS.get(domain) or DOMAINS["tabular_classification"]
    if domain_info.get("status") == "planned":
        return {"ok": False, "error": f"المجال '{domain}' غير متاح بعد (planned)"}

    tdir = _tenant_dir(tenant_id)
    ds = dataset_rel or domain_info.get("sample")
    if not ds:
        return {"ok": False, "error": "لا بيانات"}
    # انسخ العيّنة إلى مساحة المستأجر إن كانت من samples
    src = ROOT / ds
    if not src.is_file():
        return {"ok": False, "error": f"ملف غير موجود: {ds}"}
    dest = tdir / "data" / src.name
    if not dest.is_file():
        dest.write_bytes(src.read_bytes())

    max_ep = int(plan["max_epochs"])
    ep = min(int(epochs or 15), max_ep)
    prefer = domain_info.get("prefer") or "torch"
    t0 = time.time()
    job_id = f"job_{uuid.uuid4().hex[:10]}"
    result_text = ""
    try:
        if goal:
            from ai.training_factory import run_factory

            result_text = run_factory(goal)
        else:
            from ai.model_training_agent import train_from_csv

            # درّب من نسخة المستأجر — المسار النسبي من ROOT
            rel = str(dest.relative_to(ROOT))
            result_text = train_from_csv(rel, epochs=ep, prefer=prefer)
        elapsed = time.time() - t0
        # انسخ آخر نموذج إلى tenant models إن وُجد
        art = ROOT / "artifacts" / "model_training"
        pts = sorted(art.glob("torch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
        saved = None
        if pts:
            saved = tdir / "models" / f"{job_id}_{pts[0].name}"
            saved.write_bytes(pts[0].read_bytes())
        record_usage(tenant_id, train_seconds=elapsed, models_delta=1 if saved else 0, job=True)
        job_rec = {
            "job_id": job_id,
            "tenant_id": tenant_id,
            "domain": domain,
            "dataset": str(dest.relative_to(ROOT)),
            "elapsed_s": round(elapsed, 2),
            "model_path": str(saved.relative_to(ROOT)) if saved else None,
            "result_preview": (result_text or "")[:2000],
            "finished_at": _now(),
            "ok": True,
        }
        (tdir / "jobs" / f"{job_id}.json").write_text(
            json.dumps(job_rec, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return job_rec
    except Exception as e:
        record_usage(tenant_id, train_seconds=time.time() - t0, models_delta=0, job=True)
        return {"ok": False, "error": str(e), "job_id": job_id}


def list_domains() -> str:
    lines = ["## 🧩 المجالات المدعومة (AIaaS)", ""]
    for k, v in DOMAINS.items():
        lines.append(f"- `{k}` — **{v['title']}** — status=`{v['status']}` — prefer={v.get('prefer')}")
    return "\n".join(lines)


def platform_status() -> str:
    idx = load_tenants_index()
    tenants = idx.get("tenants") or {}
    lines = [
        "## ☁️ منصة AIaaS — الحالة",
        f"- عدد المستأجرين: **{len(tenants)}**",
        f"- خطط: {', '.join(PLANS.keys())}",
        f"- مجالات: {len(DOMAINS)}",
        "",
        "### المستأجرون",
    ]
    for tid, rec in list(tenants.items())[:20]:
        u = rec.get("usage") or {}
        lines.append(
            f"- `{tid}` — {rec.get('name')} — plan={rec.get('plan')} — "
            f"jobs_today={u.get('jobs_today', 0)} models={u.get('models_total', 0)}"
        )
    if not tenants:
        lines.append("لا مستأجرين بعد. أنشئ بـ: `أنشئ مستأجر اسم=demo خطة=free`")
    lines.append("")
    lines.append(list_domains())
    return "\n".join(lines)


# ── Self-evolution (مقترحات فقط + موافقة) ─────────────────────────────────

def propose_self_evolution() -> str:
    """
    يراقب أداء المصنع/السجل ويقترح ترقيات (لا يعدّل كود الإنتاج تلقائياً).
    يمكنه البحث عبر الويب المحكوم عن اتجاهات حديثة.
    """
    suggestions: List[str] = []
    try:
        from ai.training_feedback_loop import load_registry

        reg = load_registry()
        n = len(reg.get("models") or [])
        suggestions.append(f"Registry يحوي {n} نموذجاً — البطل={reg.get('champion_id')}")
    except Exception as e:
        suggestions.append(f"registry: {e}")

    try:
        from ai.training_web_access import search_arxiv, _offline

        if not _offline():
            arx = search_arxiv("autonomous machine learning agent pipeline", max_results=2)
            suggestions.append("بحث arXiv عن وكلاء تعلّم آلي ذاتيين (مقتطف):")
            suggestions.append(arx[:600])
        else:
            suggestions.append("offline — تخطي بحث الترقية")
    except Exception as e:
        suggestions.append(f"web: {e}")

    proposal = {
        "id": f"evo_{uuid.uuid4().hex[:8]}",
        "created_at": _now(),
        "status": "pending_approval",
        "suggestions": [
            "تحسين Early Stopping التكيّفي حسب حجم البيانات",
            "إضافة طابور مهام متعدد المستأجرين (worker pool)",
            "دعم CV عبر CNN على صور عند توفر بيانات",
            "ربط بوابة دفع خارجية (Stripe) للحصص Pro/Enterprise",
        ],
        "research_notes": suggestions,
    }
    path = EVOLVE / f"{proposal['id']}.json"
    path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "## 🧬 اقتراح تطوير ذاتي للنظام (يتطلب موافقة)",
        f"- id: `{proposal['id']}`",
        f"- الحالة: **pending_approval** — لن يُعدَّل كود الإنتاج تلقائياً",
        "",
        "### اقتراحات",
    ]
    for s in proposal["suggestions"]:
        lines.append(f"- {s}")
    lines.append("")
    lines.append("### ملاحظات بحث")
    lines.extend(suggestions[:8])
    lines.append("")
    lines.append(f"للموافقة الرمزية: `وافق ترقية {proposal['id']}` (يسجّل الموافقة فقط)")
    lines.append(f"الملف: `{path.relative_to(ROOT)}`")
    return "\n".join(lines)


def approve_evolution(evo_id: str) -> str:
    path = EVOLVE / f"{evo_id}.json"
    if not path.is_file():
        matches = list(EVOLVE.glob(f"{evo_id}*.json"))
        if not matches:
            return f"❌ اقتراح غير موجود: {evo_id}"
        path = matches[0]
    data = json.loads(path.read_text(encoding="utf-8"))
    data["status"] = "approved_logged"
    data["approved_at"] = _now()
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return (
        f"✅ سُجّلت الموافقة على `{data.get('id')}`.\n"
        "ملاحظة أمان: لا يُطبَّق تعديل تلقائي على كود المستودع — "
        "التنفيذ الفعلي يبقى يدوياً أو عبر PR بشري."
    )


def handle_aiaas_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(r"(حالة|status).{0,10}(aiaas|المنصة|saas)", text, re.I) or text in (
        "حالة المنصة",
        "aiaas",
    ):
        return platform_status()

    if re.search(r"(قائمة|list).{0,10}(مجالات|domains)", text, re.I):
        return list_domains()

    m = re.search(
        r"أنشئ\s*مستأجر(?:\s+اسم\s*=\s*([\w\-]+))?(?:\s+خطة\s*=\s*(free|pro|enterprise))?",
        text,
        re.I,
    )
    if m or re.search(r"create\s*tenant", text, re.I):
        name = (m.group(1) if m and m.group(1) else "demo")
        plan = (m.group(2) if m and m.group(2) else "free")
        if not m:
            plan = "free"
            name = "demo"
        rec = create_tenant(name=name, plan=plan.lower())
        return (
            f"## ✅ مستأجر جديد\n"
            f"- id: `{rec['id']}`\n"
            f"- name: {rec['name']}\n"
            f"- plan: **{rec['plan']}**\n"
            f"- api_key (يظهر مرة واحدة): `{rec['api_key']}`\n"
            f"- احفظ المفتاح بأمان — يُخزَّن الهاش فقط."
        )

    m = re.search(r"فاتورة\s+(ten_[\w]+)", text, re.I)
    if m:
        inv = estimate_invoice(m.group(1))
        return "## 💳 تقدير فاتورة\n```json\n" + json.dumps(inv, ensure_ascii=False, indent=2) + "\n```"

    m = re.search(
        r"(?:شغّل|شغل)\s*مهمة\s+(ten_[\w]+)(?:\s+مجال\s*=\s*([\w_]+))?",
        text,
        re.I,
    )
    if m:
        tid = m.group(1)
        domain = m.group(2) or "tabular_classification"
        job = run_tenant_job(tid, domain=domain)
        return "## 🧾 نتيجة مهمة مستأجر\n```json\n" + json.dumps(
            {k: v for k, v in job.items() if k != "result_preview"},
            ensure_ascii=False,
            indent=2,
        ) + "\n```\n\n" + (job.get("result_preview") or job.get("error") or "")[:1500]

    if re.search(r"(تطوير\s*ذاتي|self.?evolution|اقترح\s*ترقية)", text, re.I):
        return propose_self_evolution()

    m = re.search(r"وافق\s*ترقية\s+(evo_[\w]+)", text, re.I)
    if m:
        return approve_evolution(m.group(1))

    return None
