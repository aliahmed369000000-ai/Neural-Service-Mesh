"""
NSM Notebook Lab Service — طبقة احترافية فوق notebook_engine + Kaggle
====================================================================
  • حالة الاتصال (Kaggle / GPU محلي / أسرار)
  • سجل مهام التدريب
  • قوالب جاهزة (presets)
  • تقديرات زمن تقريبية
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
LAB_DIR = ROOT / "artifacts" / "model_training" / "lab"
LAB_DIR.mkdir(parents=True, exist_ok=True)
JOBS_LOG = LAB_DIR / "jobs_history.jsonl"

PRESETS: Dict[str, Dict[str, Any]] = {
    "smoke": {
        "label_ar": "تجربة سريعة (Smoke)",
        "preset": "small",
        "n": 2000,
        "epochs": 2,
        "batch": 8,
        "d_model": 128,
        "eta_hours": "0.2–0.5",
        "desc": "للتحقق أن المسار يعمل خلال دقائق",
    },
    "small": {
        "label_ar": "Small — d=128",
        "preset": "small",
        "n": 30000,
        "epochs": 15,
        "batch": 16,
        "d_model": 128,
        "eta_hours": "2–5",
        "desc": "أخف تدريب جاد",
    },
    "medium": {
        "label_ar": "Medium — d=256 (موصى به)",
        "preset": "medium",
        "n": 60000,
        "epochs": 30,
        "batch": 24,
        "d_model": 256,
        "eta_hours": "4–12",
        "desc": "التدريب الكامل SurahChain",
    },
    "large": {
        "label_ar": "Large — d=512",
        "preset": "large",
        "n": 100000,
        "epochs": 40,
        "batch": 8,
        "d_model": 512,
        "eta_hours": "12–24+",
        "desc": "ثقيل — راقب حصة Kaggle",
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def lab_health() -> Dict[str, Any]:
    """فحص سريع لجاهزية المختبر."""
    h: Dict[str, Any] = {"ts": _now(), "checks": {}}
    # Kaggle creds
    try:
        from ai.kaggle_provider import credentials_status, ensure_kaggle_env, _kaggle_cli_available
        h["checks"]["kaggle_creds"] = credentials_status()
        ok, msg = ensure_kaggle_env()
        h["checks"]["kaggle_env"] = {"ok": ok, "msg": msg}
        h["checks"]["kaggle_cli"] = _kaggle_cli_available()
    except Exception as e:
        h["checks"]["kaggle"] = {"ok": False, "error": str(e)}
    # local GPU
    try:
        from ai.gpu_runtime import detect_device
        d = detect_device()
        h["checks"]["local_gpu"] = {
            "device": getattr(d, "device_str", str(d)),
            "cuda": getattr(d, "cuda", False),
            "name": getattr(d, "name", None),
        }
    except Exception as e:
        h["checks"]["local_gpu"] = {"error": str(e)}
    # free providers keys
    try:
        from ai.free_gpu_providers import provider_env_status
        h["checks"]["api_keys"] = provider_env_status()
    except Exception as e:
        h["checks"]["api_keys"] = {"error": str(e)}
    # secrets hints (presence only)
    h["checks"]["streamlit_kaggle_user"] = bool(os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER"))
    h["checks"]["streamlit_kaggle_key"] = bool(os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_KEY"))
    h["ready_to_launch_kaggle"] = bool(
        h["checks"].get("streamlit_kaggle_user") and h["checks"].get("streamlit_kaggle_key")
    )

    # RunPod — مزود GPU سحابي يعمل مثل Kaggle (دفع مهمة/مراقبة/رفع)
    try:
        from ai.runpod_provider import credentials_status as runpod_credentials
        h["checks"]["runpod_creds"] = runpod_credentials()
    except Exception as e:
        h["checks"]["runpod_creds"] = {"ok": False, "error": str(e)}
    h["ready_to_launch_runpod"] = bool(
        h["checks"].get("runpod_creds", {}).get("ok")
        and (os.environ.get("RUNPOD_API_KEY") or "").strip()
        and ((os.environ.get("RUNPOD_TEMPLATE_ID") or os.environ.get("RUNPOD_ENDPOINT_ID") or "").strip())
    )

    # Groq — مفتاح مجاني فائق السرعة؛ إن وُجد في البيئة فعّله تلقائيًا
    try:
        groq_key = (os.environ.get("GROQ_API_KEY") or "").strip()
        groq_ok = bool(groq_key)
        h["checks"]["groq_key"] = {"ok": groq_ok, "note": (
            "مفعّل — Groq سيُستخدم أولًا في استدعاءات LLM" if groq_ok
            else "مفتاح GROQ_API_KEY غير مضبوط")}
        if groq_ok:
            try:
                import urllib.request
                req = urllib.request.Request(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {groq_key}"})
                with urllib.request.urlopen(req, timeout=15) as r:
                    models = json.loads(r.read().decode())
                h["checks"]["groq_ping"] = {"ok": True,
                                            "n_models": len(models.get("data") or [])}
            except Exception as ge:
                h["checks"]["groq_ping"] = {"ok": False,
                                            "note": f"{type(ge).__name__}: {ge}"}
    except Exception:
        pass

    # kernel المنعزل (nb_kernel) — مسار تدريب PyTorch الآمن المعزول
    try:
        from ai.nb_kernel import _available_for_kernel
        h["checks"]["isolated_kernel"] = bool(_available_for_kernel())
    except Exception:
        h["checks"]["isolated_kernel"] = False
    return h


def append_job(entry: Dict[str, Any]) -> None:
    entry = {**entry, "recorded_at": _now()}
    with JOBS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def list_jobs(limit: int = 30) -> List[Dict[str, Any]]:
    if not JOBS_LOG.is_file():
        return []
    rows = []
    try:
        lines = JOBS_LOG.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
        for line in reversed(lines[-limit * 2 :]):
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
            if len(rows) >= limit:
                break
    except Exception:
        return []
    return rows


def refresh_job_status(job_id: str, kernel_slug: Optional[str] = None) -> Dict[str, Any]:
    """تحديث حالة مهمة من Kaggle إن أمكن."""
    out: Dict[str, Any] = {"job_id": job_id, "ok": False}
    try:
        from ai.kaggle_provider import status_kaggle_kernel
        st = status_kaggle_kernel(job_id)
        out.update(st if isinstance(st, dict) else {"raw": st})
        out["ok"] = True
    except Exception as e:
        out["error"] = str(e)
    if kernel_slug:
        try:
            import subprocess
            r = subprocess.run(
                ["kaggle", "kernels", "status", kernel_slug],
                capture_output=True,
                text=True,
                timeout=60,
            )
            out["cli_status"] = ((r.stdout or "") + (r.stderr or "")).strip()[-500:]
            out["cli_ok"] = r.returncode == 0
        except Exception as e:
            out["cli_error"] = str(e)
    return out


def launch_runpod_preset(preset_key: str, fresh: bool = True, auto_push: bool = True,
                         endpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """إطلاق تدريب SurahChain على RunPod (مثل Kaggle: دفعة + مراقبة + AUTO_PUSH)."""
    cfg = PRESETS.get(preset_key) or PRESETS["medium"]
    t0 = time.time()
    try:
        from ai.runpod_provider import push_surahchain_runpod_job
        res = push_surahchain_runpod_job(
            preset=str(cfg["preset"]),
            n=int(cfg["n"]),
            epochs=int(cfg["epochs"]),
            batch=int(cfg["batch"]),
            fresh=fresh,
            auto_push=auto_push,
            endpoint_id=endpoint_id or None,
        )
    except Exception as e:
        res = {"ok": False, "error": str(e)}
    res["preset_key"] = preset_key
    res["config"] = cfg
    res["duration_ms"] = int((time.time() - t0) * 1000)
    res["provider"] = "runpod"
    append_job({
        "type": "surahchain_runpod_launch",
        "preset_key": preset_key,
        "ok": res.get("ok"),
        "job_id": res.get("job_id"),
        "endpoint_id": res.get("endpoint_id"),
        "provider": "runpod",
        "kernel_url": res.get("endpoint_url") or "",
        "error": res.get("error"),
    })
    return res


def launch_preset(preset_key: str, fresh: bool = True, auto_push: bool = True) -> Dict[str, Any]:
    """إطلاق تدريب من قالب جاهز عبر Kaggle API."""
    cfg = PRESETS.get(preset_key) or PRESETS["medium"]
    t0 = time.time()
    try:
        from ai.kaggle_provider import start_surahchain_training_api
        res = start_surahchain_training_api(
            preset=str(cfg["preset"]),
            n=int(cfg["n"]),
            epochs=int(cfg["epochs"]),
            batch=int(cfg["batch"]),
            fresh=fresh,
            auto_push=auto_push,
        )
    except Exception as e:
        res = {"ok": False, "error": str(e)}
    res["preset_key"] = preset_key
    res["config"] = cfg
    res["duration_ms"] = int((time.time() - t0) * 1000)
    append_job({
        "type": "surahchain_launch",
        "preset_key": preset_key,
        "ok": res.get("ok"),
        "job_id": res.get("job_id"),
        "kernel_url": res.get("kernel_url"),
        "error": res.get("error") or (res.get("push") or {}).get("error"),
    })
    return res


def feature_list_ar() -> List[str]:
    return [
        "خلايا Markdown / Python / Bash / Train مع حفظ دائم",
        "قوالب SurahChain (smoke / small / medium / large)",
        "إطلاق تدريب Kaggle GPU بزر واحد عبر API",
        "سجل مهام + تحديث الحالة",
        "فحص جاهزية المفاتيح و CLI",
        "تصدير Jupyter ipynb",
        "تقدير زمن تقريبي لكل قالب",
        "AUTO_PUSH بعد نجاح التدريب (يتطلب GITHUB_TOKEN على Kaggle)",
        "كشف GPU المحلي + كتالوج مزوّدين مجانيين",
        "مزود RunPod: تشغيل SurahChain على GPU سحابي مثل Kaggle مع AUTO_PUSH",
        "Run All مع إيقاف عند أول خطأ",
        "مراقبة GPU: nvidia-smi + VRAM snapshot من تبويب البيئة",
    ]
