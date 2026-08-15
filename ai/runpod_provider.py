# -*- coding: utf-8 -*-
"""
RunPod Provider — تشغيل تدريب NSM (SurahChain) على RunPod GPU كما على Kaggle
=============================================================================
  • credentials_status(): فحص جاهزية RUNPOD_API_KEY (+RUNPOD_TEMPLATE_ID)
  • generate_endpoint(): إنشاء endpoint GPU أو إرجاع موجود
  • push_surahchain_runpod_job(): بناء payload التدريب + دفعه async + AUTO_PUSH
  • status_runpod_job() / list_runpod_jobs() / cancel_runpod_job()
  • health_runpod_endpoint()
يدعم GraphQL API (إدارة endpoints) وREST API (jobs: /run، /status، /cancel).
التوثيق: https://docs.runpod.io/sdks/graphql/manage-endpoints
         https://docs.runpod.io/serverless/endpoints/send-requests

مفاتيح التطبيق (Streamlit Secrets / Kaggle Secrets):
  RUNPOD_API_KEY       — إجباري
  RUNPOD_TEMPLATE_ID   — إجباري (Serverless template يحتوي Python + git)
  RUNPOD_GPU_IDS       — اختياري (افتراضي AMPERE_16)
  GITHUB_TOKEN         — للرفع بعد التدريب (AUTO_PUSH)
لا يحتوي هذا الملف أي مفتاح حقيقي — كل شيء من البيئة/الأسرار.
"""
from __future__ import annotations
import base64
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import http.client
from urllib import error, request

ROOT = Path(__file__).resolve().parent.parent
JOBS_DIR = ROOT / "artifacts" / "model_training" / "runpod_jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)

GRAPHQL = "https://api.runpod.io/graphql"
REST_API = "https://api.runpod.ai/v2"

GPU_CATALOG = {
    "AMPERE_16": "RTX A4000 / A5000 — 16GB VRAM (أرخص، تدريب صغير)",
    "AMPERE_24": "RTX 3090 / A5000 — 24GB VRAM",
    "ADA_24": "RTX 4090 Ada — 24GB VRAM (سريع)",
    "AMPERE_48": "A40 — 48GB VRAM",
    "ADA_48_PRO": "RTX 6000 Ada — 48GB VRAM",
    "AMPERE_80": "A100 80GB — 80GB VRAM (الأقوى)",
    "ADA_80_PRO": "RTX 6000 Ada 80GB Pro",
}
DEFAULT_GPU = "AMPERE_16"

# صور PyTorch رسمية من runpod/pytorch (مُثبتة على Docker Hub — القديم 2.5.1-py3.11 حُذف من registry):
DEFAULT_TEMPLATE_IMAGES = [
    "runpod/pytorch:1.1.0-rc.154-cu1281-torch260-ubuntu2204",
    "runpod/pytorch:1.1.0-rc.154-cu1290-torch260-ubuntu2204",
    "runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204",
    "runpod/pytorch:1.1.0-rc.154-cu1290-torch280-ubuntu2204",
]
DEFAULT_TEMPLATE_IMAGE = DEFAULT_TEMPLATE_IMAGES[0]

# GPU ids المتاحة في حسابات RunPod العامة (gpuTypes الرسمية):
GPU_TYPE_MAP = {
    "AMPERE_16": ["NVIDIA RTX A4000 Laptop GPU", "NVIDIA RTX A4000", "NVIDIA RTX A5000", "NVIDIA RTX A4500"],
    "AMPERE_24": ["NVIDIA GeForce RTX 3090", "NVIDIA GeForce RTX 4090", "NVIDIA RTX A5000"],
    "ADA_24": ["NVIDIA GeForce RTX 4090"],
    "AMPERE_48": ["NVIDIA A40"],
    "ADA_48_PRO": ["NVIDIA RTX 6000 Ada Generation"],
    "AMPERE_80": ["NVIDIA A100 80GB PCIe", "NVIDIA A100-SXM4-80GB"],
    "ADA_80_PRO": ["NVIDIA RTX 6000 Ada Generation 80GB Pro"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _secret(name: str, default: str = "") -> str:
    return (os.environ.get(name) or "").strip() or default


UA_BROWSER = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"


def _post_json(url: str, data: Any, headers: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    headers = dict(headers)
    headers.setdefault("user-agent", UA_BROWSER)
    body = json.dumps(data).encode("utf-8")
    parsed = request.urlparse(url)
    last_err = None
    for attempt in range(3):
        try:
            conn = http.client.HTTPSConnection(parsed.hostname, timeout=timeout)
            path = parsed.path or "/"
            if parsed.query:
                path += "?" + parsed.query
            conn.request("POST", path, body=body, headers=headers)
            with conn.getresponse() as r:
                out = r.read().decode("utf-8")
            if r.status in (403, 429):
                time.sleep(2 * (attempt + 1))
                continue
            if r.status >= 400:
                return {"ok": False, "error": f"HTTP {r.status}", "raw": out[:500]}
            return json.loads(out)
        except Exception as ex:  # connection errors — retry
            last_err = ex
            time.sleep(1)
    if last_err is not None:
        raise last_err
    raise RuntimeError("runpod api: no successful response after retries")


def _get_json(url: str, headers: Dict[str, str], timeout: int = 60) -> Dict[str, Any]:
    headers = dict(headers)
    headers.setdefault("user-agent", UA_BROWSER)
    last_err = None
    for attempt in range(3):
        req = request.Request(url, headers=headers, method="GET")
        try:
            with request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except error.HTTPError as he:
            last_err = he
            if he.code in (403, 429):
                time.sleep(2 * (attempt + 1))
                continue
            raise
    raise last_err


# ----------------------------------------------------------------------------
# جاهزية المفاتيح
# ----------------------------------------------------------------------------

def credentials_status() -> Dict[str, Any]:
    key = _secret("RUNPOD_API_KEY")
    template = _secret("RUNPOD_TEMPLATE_ID")
    token = _secret("GITHUB_TOKEN")
    return {
        "provider": "runpod",
        "api_key_set": bool(key),
        "template_id_set": bool(template),
        "github_token_set": bool(token),
        "gpu": _secret("RUNPOD_GPU_IDS", DEFAULT_GPU),
        "ok": bool(key and (template or token)),
        "need": ([] if key else ["RUNPOD_API_KEY"]) + (
            [] if (template or token) else ["RUNPOD_TEMPLATE_ID (أو GITHUB_TOKEN)"]),
        "hint_ar": (
            "ضع RUNPOD_API_KEY وRUNPOD_TEMPLATE_ID في NSM Secrets. "
            "أنشئ Serverless template من https://www.runpod.io/console/serverless/user/templates "
            "يعتمد runpod/pytorch:2.5.1-py3.11-cuda12.1.1-dev مع git وcurl."
            if not (key and (template or token))
            else "جاهز — يمكن تشغيل التدريب الآن"
        ),
    }


# ----------------------------------------------------------------------------
# إدارة endpoints عبر GraphQL
# ----------------------------------------------------------------------------

def _graphql(payload: Dict[str, Any], api_key: str, retries: int = 4) -> Dict[str, Any]:
    url = GRAPHQL + "?api_key=" + request.quote(api_key, safe="")
    last_err = None
    for attempt in range(retries):
        try:
            res = _post_json(url, payload, {"content-type": "application/json"})
        except Exception as ex:  # rate limit / connection — retry
            last_err = ex
            time.sleep(2 * (attempt + 1))
            continue
        errs = (res or {}).get("errors") or []
        code = errs[0].get("extensions", {}).get("code", "") if errs else ""
        if code in ("GRAPHQL_VALIDATION_FAILED", "TOO_MANY_REQUESTS") or "rate" in code.lower():
            last_err = RuntimeError(f"graphql {code}: {errs[0].get('message')}")
            time.sleep(3 * (attempt + 1))
            continue
        return res
    raise last_err or RuntimeError("graphql: no response")


def find_template(api_key: str, image: Optional[str] = None) -> Optional[str]:
    """البحث عن template id جاهز لصورة PyTorch.

    GraphQL RunPod لا يعرّض قائمة templates قراءة (لا query ولا mutation list)،
    لذا نعتمد على template محليًا محفوظًا في: env/Secrets > RUNPOD_TEMPLATE_ID
    أو السجل المحلي ~/.nsm_runpod_templates.json الذي نكتبه عند الإنشاء.
    """
    import pathlib
    local = (pathlib.Path.home() / ".nsm_runpod_templates.json").resolve()
    if local.exists():
        try:
            saved = json.loads(local.read_text())
        except Exception:
            saved = {}
        for tpl in (saved if isinstance(saved, list) else saved.get("templates", [])):
            if isinstance(tpl, dict):
                img = (tpl.get("imageId") or "").lower()
                if image:
                    if image.lower() in img or img in image.lower():
                        return tpl["id"]
                elif "pytorch" in img or "runpod/pytorch" in img:
                    return tpl["id"]
    return None


def create_pytorch_template(api_key: str, name: str = "nsm-pytorch-dev") -> Optional[str]:
    """إنشاء Serverless template PyTorch تلقائيًا عبر saveTemplate.

    RunPod GraphQL لا يعرض templates القديمة (2.x-py3.11) في Docker Hub؛
    نستخدم أحدث صور runpod/pytorch الإنتاجية: cuXXX-torchXXX-ubuntuNNNN.
    """
    candidates = [
        "runpod/pytorch:1.1.0-rc.154-cu1281-torch260-ubuntu2204",
        "runpod/pytorch:1.1.0-rc.154-cu1290-torch260-ubuntu2204",
        "runpod/pytorch:1.1.0-rc.154-cu1281-torch280-ubuntu2204",
        "runpod/pytorch:1.1.0-rc.154-cu1290-torch280-ubuntu2204",
    ]
    import uuid
    # أولاً: أي template سابق أنشأناه NSM في السجل المحلي؟
    local_tid = find_template(api_key)
    if local_tid:
        return local_tid
    for image in candidates:
        try:
            save = _graphql({
                "query": (
                    "mutation { saveTemplate(input: { "
                    'name: "' + name + "-" + uuid.uuid4().hex[:6] + '", '
                    'imageName: "' + image + '", '
                    "env: [ { key: \"HF_HUB_DISABLE_TELEMETRY\", value: \"1\" } ], "
                    'dockerArgs: "", '
                    "containerDiskInGb: 40, "
                    "volumeInGb: 40, "
                    'volumeMountPath: "/workspace", '
                    'readme: "NSM pytorch training template", '
                    "isPublic: false "
                    "}) { id } }"
                )
            }, api_key)
            data = (save.get("data") or {}).get("saveTemplate") or {}
            tid = data.get("id")
            if tid:
                return tid
            errs = save.get("errors") or []
            if not errs:
                continue
            # IMAGE_VALIDATION_FAILED تعني أن الصورة غير موجودة — نجرب التالية
            code = errs[0].get("extensions", {}).get("code", "") if errs else ""
        except Exception:
            continue
    return None


def _remember_template(template_id: str, image: str) -> None:
    """حفظ template id محليًا في ~/.nsm_runpod_templates.json."""
    try:
        import pathlib
        local = pathlib.Path.home() / ".nsm_runpod_templates.json"
        saved = {}
        if local.exists():
            try:
                saved = json.loads(local.read_text())
            except Exception:
                saved = {}
        if not isinstance(saved, dict):
            saved = {"templates": saved if isinstance(saved, list) else []}
        seen = [t for t in saved.get("templates", []) if t.get("id") != template_id]
        seen.append({"id": template_id, "imageId": image, "name": "nsm-pytorch-dev"})
        saved["templates"] = seen
        local.write_text(json.dumps(saved, indent=2))
    except Exception:
        pass


def generate_endpoint(
    name: str = "nsm-surahchain",
    gpu_ids: Optional[str] = None,
    workers_max: int = 2,
) -> Dict[str, Any]:
    """إنشاء endpoint (أو تحديث موجود بنفس الاسم) وإرجاع id."""
    api_key = _secret("RUNPOD_API_KEY")
    if not api_key:
        return {"ok": False, "error": "RUNPOD_API_KEY غير مضبوط",
                "hint_ar": "ضعه في NSM Secrets"}
    pre = _secret("RUNPOD_ENDPOINT_ID")
    if pre:
        return {
            "ok": True, "endpoint_id": pre,
            "action": "reused",
            "gpu_ids": (gpu_ids or _secret("RUNPOD_GPU_IDS", DEFAULT_GPU)).upper(),
            "endpoint_url": f"https://api.runpod.ai/v2/{pre}",
            "msg_ar": "تم استخدام RUNPOD_ENDPOINT_ID الجاهز",
        }
    # تلقائي: البحث عن template عام (runpod/pytorch) في حساب المستخدم
    template_id = _secret("RUNPOD_TEMPLATE_ID")
    if not template_id:
        found = None
        try:
            found = find_template(api_key)
        except Exception:
            pass
        if found:
            template_id = found
        else:
            # إنشاء تلقائي: template PyTorch جديد عبر saveTemplate
            created = create_pytorch_template(api_key)
            if created:
                template_id = created
                # حفظه محليًا وفي env للجلسات القادمة
                _remember_template(template_id, DEFAULT_TEMPLATE_IMAGE)
                os.environ["RUNPOD_TEMPLATE_ID"] = template_id
            else:
                return {"ok": False, "error": "RUNPOD_TEMPLATE_ID غير مضبوط ولا يمكن إنشاء template تلقائيًا",
                        "hint_ar": "ضع RUNPOD_TEMPLATE_ID في NSM Secrets، أو أنشئ template من "
                                   "https://www.runpod.io/console/serverless/user/templates "
                                   "يعتمد صورة " + DEFAULT_TEMPLATE_IMAGE}
    gpu = (gpu_ids or _secret("RUNPOD_GPU_IDS", DEFAULT_GPU)).upper()
    if gpu not in GPU_CATALOG:
        return {"ok": False, "error": f"gpuIds غير صالح: {gpu}",
               "hint_ar": "الخيارات: " + ", ".join(sorted(GPU_CATALOG))}
    try:
        # البحث عن endpoint موجود بنفس الاسم
        list_res = _graphql({"query": "query { myself { endpoints { id name } } }"}, api_key)
        existing = [e for e in (list_res.get("data") or {}).get("myself", {}).get("endpoints") or []
                    if e.get("name") == name]
        if existing:
            eid = existing[0]["id"]
            return {"ok": True, "endpoint_id": eid, "action": "existing",
                    "endpoint_url": f"https://api.runpod.ai/v2/{eid}",
                    "msg_ar": "تم استخدام endpoint موجود"}
        save = _graphql({
            "query": (
                "mutation { saveEndpoint(input: { "
                'gpuIds: "' + gpu + '", '
                'idleTimeout: 60, '
                'locations: "", '
                'name: "' + name + '", '
                "flashBootType: FLASHBOOT, "
                'scalerType: "QUEUE_DELAY", '
                "scalerValue: 4, "
                'templateId: "' + template_id + '", '
                "workersMax: " + str(workers_max) + ", "
                "workersMin: 0 "
                "}) { id name gpuIds templateId workersMax } }"
            )
        }, api_key)
        data = (save.get("data") or {}).get("saveEndpoint") or {}
        eid = data.get("id")
        if not eid and save.get("errors"):
            return {"ok": False, "error": save["errors"][0].get("message", save["errors"][0]),
                    "raw": save}
        return {
            "ok": True,
            "endpoint_id": eid,
            "action": "created",
            "gpu_ids": gpu,
            "workers_max": workers_max,
            "endpoint_url": f"https://api.runpod.ai/v2/{eid}",
            "msg_ar": "تم إنشاء endpoint بنجاح — workersMin=0 يعني scale-to-zero",
        }
    except Exception as he:
        body = getattr(he, "read", lambda: b"")
        try:
            body = body().decode("utf-8", "ignore")[:500]
        except Exception:
            body = ""
        return {"ok": False, "error": f"HTTP: {body or str(he)}"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------
# تشغيل تدريب SurahChain (payload مدمج — لا ملفات خارجية)
# ----------------------------------------------------------------------------

_TRAIN_SCRIPT_TEMPLATE = """\
#!/usr/bin/env python3
\"\"\"NSM SurahChain on RunPod — job {job_id} (built by ai.runpod_provider)\"\"\"
from __future__ import annotations
import base64, os, subprocess, sys, time
from pathlib import Path

PRESET     = "{preset}"
SCN_N      = {n}
SCN_EPOCHS = {epochs}
SCN_BATCH  = {batch}
SCN_FRESH  = {fresh}
AUTO_PUSH  = {auto_push}
GITHUB_TOKEN = r\"{gh_token}\"
REPO       = "{repo}"
BRANCH     = "{branch}"

def main():
    out_dir = Path("/tmp/nsm_runpod_out")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "job_id.txt").write_text("{job_id}")
    print("=== CUDA check ===")
    try:
        import torch
        print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("device:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("torch check failed:", e)
    work = Path("/tmp/Neural-Service-Mesh")
    if work.exists():
        print("repo exists, pulling...")
        subprocess.run(["git", "-C", str(work), "pull", "--ff-only"], check=False)
    else:
        print("cloning repo...")
        subprocess.run([
            "git", "clone", "--depth", "1", "--branch", BRANCH,
            "https://x-access-token:{gh_token}@github.com/{repo}.git",
            str(work),
        ], check=False)
    script = work / "experiments" / "surah_chain_network" / "run_train_then_push.py"
    if not script.exists():
        # fallback: البحث عن أول سكربت تدريب متاح
        candidates = list((work / "experiments").rglob("*train*"))
        script = candidates[0] if candidates else None
        if script is None:
            raise SystemExit("run_train_then_push.py غير موجود في المستودع")
    env = dict(os.environ)
    env.update({{
        "SCN_PRESET": PRESET, "SCN_N": str(SCN_N),
        "SCN_EPOCHS": str(SCN_EPOCHS), "SCN_BATCH": str(SCN_BATCH),
        "SCN_FRESH": SCN_FRESH, "AUTO_PUSH": AUTO_PUSH,
        "GITHUB_TOKEN": GITHUB_TOKEN, "GH_TOKEN": GITHUB_TOKEN,
        "NSM_OUT_DIR": str(out_dir),
    }})
    print("=== starting training ===")
    start = time.time()
    r = subprocess.run([sys.executable, str(script)], env=env,
                       cwd=str(work), check=False)
    elapsed = time.time() - start
    print("training exited with code", r.returncode, f"({{elapsed:.1f}}s)")
    (out_dir / "exit_code.txt").write_text(str(r.returncode))
    (out_dir / "elapsed_s.txt").write_text(f"{{elapsed:.1f}}")
    if (work / "artifacts" / "model_training" / "surah_chain").exists():
        subprocess.run([
            "cp", "-r", str(work / "artifacts" / "model_training" / "surah_chain"),
            str(out_dir / "surah_chain"),
        ], check=False)
    # قائمة الملفات الناتجة
    files = [str(p) for p in out_dir.rglob("*") if p.is_file()]
    import json as _json
    (out_dir / "output_files.json").write_text(_json.dumps(files, ensure_ascii=False, indent=1))
    print("=== done. output files ===")
    print(files)
    return 0 if r.returncode == 0 else 1

if __name__ == "__main__":
    raise SystemExit(main())
"""


def generate_surahchain_runpod_payload(
    job_id: str,
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
    repo: str = "aliahmed369000000-ai/Neural-Service-Mesh",
    branch: str = "main",
    gh_token: Optional[str] = None,
) -> Dict[str, Any]:
    """يبني payload /run كاملًا (سكربت التدريب مدمج base64 داخل input)."""
    token = (gh_token or _secret("GITHUB_TOKEN")).strip()
    if not token:
        return {"ok": False, "error": "GITHUB_TOKEN غير متوفر — مطلوب للـclone وللرفع"}
    src = _TRAIN_SCRIPT_TEMPLATE.format(
        job_id=job_id, preset=preset, n=n, epochs=epochs, batch=batch,
        fresh="1" if fresh else "0", auto_push="1" if auto_push else "0",
        gh_token=token, repo=repo, branch=branch,
    )
    encoded = base64.b64encode(src.encode("utf-8")).decode("ascii")
    max_hours = max(epochs, 1) * 0.6 + 2  # تقدير تحفظي
    ttl_ms = int(min(max_hours * 3600 * 1000, 6 * 24 * 3600 * 1000))  # حتى 6 أيام
    return {
        "ok": True,
        "input": {
            "nsm_job": {
                "job_id": job_id,
                "preset": preset,
                "repo": repo,
                "branch": branch,
                "script_b64": encoded,
                "auto_push": auto_push,
            },
            "_bootstrap": (
                "import base64, json, os, subprocess, sys, time; "
                "d=os.environ.get('INPUT_NSM_JOB') or ''; "
                "cfg=json.loads(d) if d else dict(); "
                "sc=base64.b64decode(cfg.get('script_b64','')).decode(); "
                "p='/tmp/nsm_job_script.py'; open(p,'w').write(sc); "
                "r=subprocess.run([sys.executable,p],env=dict(os.environ,INPUT_NSM_JOB=d)); "
                "sys.exit(r.returncode)"
            ),
        },
        "policy": {"executionTimeout": ttl_ms, "ttl": ttl_ms + 3600000},
        "token_used": bool(token),
    }


def push_surahchain_runpod_job(
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
    endpoint_id: Optional[str] = None,
) -> Dict[str, Any]:
    """فحص الجاهزية → إنشاء/استخدام endpoint → دفع مهمة تدريب async."""
    api_key = _secret("RUNPOD_API_KEY")
    if not api_key:
        return {"ok": False, "error": "RUNPOD_API_KEY غير مضبوط",
                "need": ["RUNPOD_API_KEY", "RUNPOD_TEMPLATE_ID", "GITHUB_TOKEN"],
                "hint_ar": "ضعها في NSM Secrets ثم أعد المحاولة"}
    job_id = f"nsm_sc_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    payload = generate_surahchain_runpod_payload(
        job_id=job_id, preset=preset, n=n, epochs=epochs, batch=batch,
        fresh=fresh, auto_push=auto_push,
    )
    if not payload.get("ok"):
        return payload
    # endpoint: المستخدم يحدده في RUNPOD_ENDPOINT_ID أو ننشئ واحدًا
    eid = endpoint_id or _secret("RUNPOD_ENDPOINT_ID")
    ep_res = None
    if not eid:
        ep_res = generate_endpoint()
        if ep_res.get("ok"):
            eid = ep_res["endpoint_id"]
        else:
            return {"ok": False, "job_id": job_id, "error": ep_res.get("error"),
                    "ep": ep_res, "hint_ar": (
                        "لا يمكن إنشاء endpoint تلقائيًا — ضع RUNPOD_ENDPOINT_ID في NSM Secrets "
                        "أو تأكد من صحة RUNPOD_TEMPLATE_ID")}
    headers = {
        "authorization": "Bearer " + api_key,
        "content-type": "application/json",
    }
    body = {"input": payload["input"], "policy": payload["policy"]}
    try:
        run_res = _post_json(f"{REST_API}/{eid}/run", body, headers, timeout=90)
        req_id = run_res.get("id") or job_id
        record = {
            "ok": True,
            "job_id": req_id,
            "endpoint_id": eid,
            "preset": preset,
            "n": n,
            "epochs": epochs,
            "batch": batch,
            "auto_push": auto_push,
            "status": run_res.get("status", "IN_QUEUE"),
            "created_at": _now(),
            "raw_request": run_res,
            "msg_ar": (
                "تم دفع مهمة التدريب إلى RunPod — راقب /status. "
                "بعد الاكتمال يُرفع المستودع تلقائيًا إن وُجد GITHUB_TOKEN."
                if run_res.get("status") in (None, "IN_QUEUE")
                else f"استجابة API: {run_res.get('status')}"
            ),
        }
        (JOBS_DIR / f"{req_id}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        record.pop("raw_request", None)
        return {"ok": True, "job_id": req_id, "endpoint_id": eid,
                "endpoint_url": f"https://api.runpod.ai/v2/{eid}",
                "status": run_res.get("status", "IN_QUEUE"),
                "record": record, "ep": ep_res}
    except error.HTTPError as he:
        body_txt = ""
        try:
            body_txt = he.read().decode("utf-8", "ignore")[:500]
        except Exception:
            pass
        return {"ok": False, "job_id": job_id, "endpoint_id": eid,
                "error": f"HTTP {he.code}: {body_txt or he.reason}",
                "hint_ar": "تحقق من endpoint_id والمفتاح — انظر RunPod console للمهمة"}
    except Exception as e:
        return {"ok": False, "job_id": job_id, "endpoint_id": eid,
                "error": f"{type(e).__name__}: {e}"}


# ----------------------------------------------------------------------------
# المراقبة والإدارة
# ----------------------------------------------------------------------------

def status_runpod_job(endpoint_id: str, job_id: str) -> Dict[str, Any]:
    """GET /status/{job_id} — COMPLETED / FAILED / RUNNING / IN_QUEUE / TIMED_OUT ..."""
    api_key = _secret("RUNPOD_API_KEY")
    if not api_key:
        return {"ok": False, "error": "RUNPOD_API_KEY غير مضبوط"}
    try:
        res = _get_json(
            f"{REST_API}/{endpoint_id}/status/{job_id}",
            {"authorization": "Bearer " + api_key}, timeout=30,
        )
        rec = {"ok": True, "job_id": job_id, "endpoint_id": endpoint_id,
               "status": res.get("status", "UNKNOWN"), "raw": res, "fetched_at": _now()}
        (JOBS_DIR / f"{job_id}.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        return rec
    except error.HTTPError as he:
        return {"ok": False, "job_id": job_id, "endpoint_id": endpoint_id,
                "error": f"HTTP {he.code}: {he.reason}"}
    except Exception as e:
        return {"ok": False, "job_id": job_id, "error": f"{type(e).__name__}: {e}"}


def cancel_runpod_job(endpoint_id: str, job_id: str) -> Dict[str, Any]:
    api_key = _secret("RUNPOD_API_KEY")
    if not api_key:
        return {"ok": False, "error": "RUNPOD_API_KEY غير مضبوط"}
    try:
        _post_json(
            f"{REST_API}/{endpoint_id}/cancel/{job_id}", {},
            {"authorization": "Bearer " + api_key, "content-type": "application/json"},
            timeout=30,
        )
        return {"ok": True, "job_id": job_id, "msg_ar": "أُرسل طلب الإيقاف"}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def health_runpod_endpoint(endpoint_id: str) -> Dict[str, Any]:
    api_key = _secret("RUNPOD_API_KEY")
    if not api_key:
        return {"ok": False, "error": "RUNPOD_API_KEY غير مضبوط"}
    try:
        res = _get_json(f"{REST_API}/{endpoint_id}/health",
                        {"authorization": "Bearer " + api_key}, timeout=30)
        return {"ok": True, "endpoint_id": endpoint_id, "health": res}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def list_runpod_jobs(limit: int = 20) -> List[Dict[str, Any]]:
    out = []
    files = sorted(JOBS_DIR.glob("*.json"), key=lambda p: -p.stat().st_mtime)
    for fp in files[:limit]:
        try:
            out.append(json.loads(fp.read_text(encoding="utf-8")))
        except Exception:
            continue
    return out


def runpod_report_ar() -> str:
    """تقرير عربي موجز عن المزود والمهام."""
    cred = credentials_status()
    jobs = list_runpod_jobs()
    lines = [f"مزود RunPod — {'جاهز' if cred['ok'] else 'غير مهيّأ'}"]
    if not cred["ok"]:
        lines.append("المطلوب: " + ", ".join(cred["need"]))
    for j in jobs[:5]:
        lines.append(
            f"- {j.get('job_id')} status={j.get('status')} "
            f"preset={j.get('preset')} epochs={j.get('epochs')}"
        )
    return "\n".join(lines)
