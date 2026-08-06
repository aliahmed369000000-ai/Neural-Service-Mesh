"""
Remote GPU Provider — واجهة موحّدة لتشغيل التدريب على GPU بعيد
==============================================================
  • LocalGPUProvider: الجهاز الحالي (Colab / سيرفرك) عبر gpu_runtime
  • ColabBridgeProvider: جلسة Colab تدفع النتائج عبر webhook/API
  • RunPodProvider: هيكل جاهز (يحتاج API key — اختياري)

لا يتضمن أتمتة متصفح Google Colab (Playwright).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

logger = logging.getLogger("RemoteGPU")

ROOT = Path(__file__).resolve().parent.parent
REMOTE_DIR = ROOT / "artifacts" / "model_training" / "remote_jobs"
REMOTE_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RemoteGPUProvider(ABC):
    name: str = "base"

    @abstractmethod
    def status(self) -> Dict[str, Any]:
        ...

    @abstractmethod
    def submit_train_csv(
        self,
        csv_path: str,
        epochs: int = 15,
        prefer: str = "auto",
        job_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """يُرجع job dict يتضمن job_id وحالة."""
        ...

    def collect_results(self, job_id: str) -> Dict[str, Any]:
        path = REMOTE_DIR / f"{job_id}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"ok": False, "error": "job not found", "job_id": job_id}


class LocalGPUProvider(RemoteGPUProvider):
    """تشغيل على نفس العملية (Colab بعد تفعيل GPU أو آلة محلية)."""

    name = "local"

    def status(self) -> Dict[str, Any]:
        try:
            from ai.gpu_runtime import detect_device, vram_snapshot

            d = detect_device(force_gpu=os.environ.get("NSM_ALLOW_GPU") == "1")
            return {
                "provider": self.name,
                "device": d.device_str,
                "reason": d.reason,
                "vram": vram_snapshot(),
                "ok": True,
            }
        except Exception as e:
            return {"provider": self.name, "ok": False, "error": str(e)}

    def submit_train_csv(
        self,
        csv_path: str,
        epochs: int = 15,
        prefer: str = "auto",
        job_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        job_id = f"local_{uuid.uuid4().hex[:10]}"
        t0 = time.time()
        os.environ.setdefault("NSM_ALLOW_GPU", "1")
        try:
            from ai.model_training_agent import train_from_csv

            result = train_from_csv(csv_path, epochs=epochs, prefer=prefer)
            elapsed = time.time() - t0
            # آخر pt
            art = ROOT / "artifacts" / "model_training"
            pts = sorted(art.glob("torch_*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
            model_path = str(pts[0].relative_to(ROOT)) if pts else None
            job = {
                "ok": True,
                "job_id": job_id,
                "provider": self.name,
                "status": "completed",
                "csv_path": csv_path,
                "epochs": epochs,
                "elapsed_s": round(elapsed, 2),
                "model_path": model_path,
                "result_preview": (result or "")[:2500],
                "meta": job_meta or {},
                "finished_at": _now(),
            }
        except Exception as e:
            job = {
                "ok": False,
                "job_id": job_id,
                "provider": self.name,
                "status": "failed",
                "error": str(e),
                "finished_at": _now(),
            }
        (REMOTE_DIR / f"{job_id}.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return job


class ColabBridgeProvider(RemoteGPUProvider):
    """
    جسر Colab: الدفتر يدفع الحزمة إلى NSM عبر webhook.
    لا يفتح المتصفح — يعتمد على خلية push داخل الدفتر.
    """

    name = "colab_bridge"

    def __init__(self, webhook_url: Optional[str] = None, secret: Optional[str] = None):
        self.webhook_url = (
            webhook_url
            or os.environ.get("NSM_REMOTE_WEBHOOK_URL")
            or ""
        ).rstrip("/")
        self.secret = secret or os.environ.get("NSM_REMOTE_WEBHOOK_SECRET") or ""

    def status(self) -> Dict[str, Any]:
        return {
            "provider": self.name,
            "webhook_configured": bool(self.webhook_url),
            "secret_set": bool(self.secret),
            "ok": True,
            "note": "شغّل خلية الدفع من دفتر Colab بعد التدريب",
        }

    def submit_train_csv(
        self,
        csv_path: str,
        epochs: int = 15,
        prefer: str = "auto",
        job_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # على Colab نفسها نستخدم LocalGPUProvider ثم push
        local = LocalGPUProvider().submit_train_csv(csv_path, epochs, prefer, job_meta)
        if local.get("ok") and self.webhook_url:
            push = push_job_package(local, self.webhook_url, self.secret)
            local["push"] = push
        return local


class RunPodProvider(RemoteGPUProvider):
    """هيكل RunPod — يتطلّب RUNPOD_API_KEY؛ بدون مفتاح يعيد تعليمات فقط."""

    name = "runpod"

    def status(self) -> Dict[str, Any]:
        key = os.environ.get("RUNPOD_API_KEY") or ""
        return {
            "provider": self.name,
            "api_key_set": bool(key),
            "ok": bool(key),
            "note": "ضع RUNPOD_API_KEY ثم نفّذ submit — التنفيذ الكامل يُضاف حسب قالب الـpod",
        }

    def submit_train_csv(
        self,
        csv_path: str,
        epochs: int = 15,
        prefer: str = "auto",
        job_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        key = os.environ.get("RUNPOD_API_KEY") or ""
        if not key:
            return {
                "ok": False,
                "provider": self.name,
                "status": "not_configured",
                "error": "RUNPOD_API_KEY غير مضبوط",
                "hint": "https://docs.runpod.io/ — أو استخدم local/colab_bridge",
            }
        # Placeholder: لا نستدعي API حقيقي بدون قالب مستخدم
        job_id = f"runpod_pending_{uuid.uuid4().hex[:8]}"
        job = {
            "ok": False,
            "job_id": job_id,
            "provider": self.name,
            "status": "stub",
            "error": "RunPod submit يحتاج قالب pod مخصص لمشروعك — الواجهة جاهزة للربط",
            "request": {
                "csv_path": csv_path,
                "epochs": epochs,
                "prefer": prefer,
                "meta": job_meta or {},
            },
            "finished_at": _now(),
        }
        (REMOTE_DIR / f"{job_id}.json").write_text(
            json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return job


def get_provider(name: Optional[str] = None) -> RemoteGPUProvider:
    n = (name or os.environ.get("NSM_REMOTE_GPU_PROVIDER") or "local").strip().lower()
    if n in ("colab", "colab_bridge", "bridge"):
        return ColabBridgeProvider()
    if n in ("runpod", "pod"):
        return RunPodProvider()
    return LocalGPUProvider()


def build_result_package(
    job: Dict[str, Any],
    copy_weights: bool = True,
) -> Dict[str, Any]:
    """حزمة نتائج قابلة للرفع (ميتا + مسار أوزان اختياري)."""
    pkg_id = f"pkg_{uuid.uuid4().hex[:10]}"
    pkg_dir = REMOTE_DIR / "packages" / pkg_id
    pkg_dir.mkdir(parents=True, exist_ok=True)
    model_rel = job.get("model_path")
    copied = None
    sha = None
    if copy_weights and model_rel:
        src = ROOT / model_rel
        if src.is_file():
            dest = pkg_dir / src.name
            shutil.copy2(src, dest)
            copied = str(dest.relative_to(ROOT))
            sha = hashlib.sha256(dest.read_bytes()).hexdigest()
    package = {
        "package_id": pkg_id,
        "created_at": _now(),
        "job": {k: v for k, v in job.items() if k != "result_preview"}
        | {"result_preview": (job.get("result_preview") or "")[:1500]},
        "weights_path": copied,
        "weights_sha256": sha,
        "source_host": os.environ.get("NSM_REMOTE_HOST_LABEL") or "unknown",
    }
    (pkg_dir / "package.json").write_text(
        json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return package


def push_job_package(
    job: Dict[str, Any],
    webhook_url: str,
    secret: str = "",
    timeout: int = 60,
) -> Dict[str, Any]:
    """POST JSON package إلى خادم NSM (أو أي webhook)."""
    package = build_result_package(job, copy_weights=False)
    # لا نرسل أوزان ثنائية في JSON — فقط ميتا + رابط/مسار
    body = json.dumps(
        {
            "type": "nsm_remote_train_result",
            "package": package,
            "secret": secret or None,
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = request.Request(
        webhook_url,
        data=body,
        headers={"Content-Type": "application/json", "User-Agent": "NSM-RemoteGPU/1.0"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return {"ok": True, "status": resp.status, "response": raw[:1000]}
    except error.HTTPError as e:
        return {"ok": False, "status": e.code, "error": e.read().decode("utf-8", errors="replace")[:500]}
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def ingest_remote_package(payload: Dict[str, Any], expected_secret: str = "") -> Dict[str, Any]:
    """يستقبل حزمة من Colab/عقدة بعيدة ويحفظها محلياً."""
    if expected_secret:
        if (payload.get("secret") or "") != expected_secret:
            return {"ok": False, "error": "invalid secret"}
    package = payload.get("package") or payload
    pid = package.get("package_id") or f"pkg_{uuid.uuid4().hex[:8]}"
    out = REMOTE_DIR / "inbox" / f"{pid}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    # فهرس
    idx_path = REMOTE_DIR / "inbox_index.json"
    idx: List[Dict[str, Any]] = []
    if idx_path.is_file():
        try:
            idx = json.loads(idx_path.read_text(encoding="utf-8"))
        except Exception:
            idx = []
    idx.insert(
        0,
        {
            "package_id": pid,
            "received_at": _now(),
            "job_id": (package.get("job") or {}).get("job_id"),
            "model_path": (package.get("job") or {}).get("model_path"),
        },
    )
    idx_path.write_text(json.dumps(idx[:100], ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "saved": str(out.relative_to(ROOT)), "package_id": pid}


def remote_status_report() -> str:
    providers = {
        "local": LocalGPUProvider().status(),
        "colab_bridge": ColabBridgeProvider().status(),
        "runpod": RunPodProvider().status(),
    }
    inbox = REMOTE_DIR / "inbox"
    n_inbox = len(list(inbox.glob("*.json"))) if inbox.is_dir() else 0
    lines = [
        "## 🌐 Remote GPU Provider",
        f"- الوارد (inbox): **{n_inbox}** حزمة",
        f"- المجلد: `artifacts/model_training/remote_jobs/`",
        "",
    ]
    for name, st in providers.items():
        lines.append(f"### {name}")
        lines.append(f"- {json.dumps(st, ensure_ascii=False)}")
    lines.append("")
    lines.append(
        "أوامر: `حالة remote gpu` · `درّب remote csv …` · "
        "Webhook: `POST /training/remote-results`"
    )
    return "\n".join(lines)


def handle_remote_gpu_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(حالة|status).{0,12}(remote\s*gpu|gpu\s*بعيد|remote)", text, re.I) or text.lower() in (
        "remote gpu",
        "حالة remote gpu",
    ):
        return remote_status_report()
    m = re.search(
        r"(?:در[ّ]?ب|train)\s*remote(?:\s*gpu)?\s*(?:csv)?\s*((?:data|artifacts)[\w./-]+\.csv)?",
        text,
        re.I,
    )
    if m or re.search(r"remote\s*train", text, re.I):
        csv_path = (m.group(1) if m and m.group(1) else "data/samples/classification_demo.csv")
        prov = get_provider()
        job = prov.submit_train_csv(csv_path, epochs=12)
        return (
            f"## 🚀 مهمة Remote GPU (`{prov.name}`)\n\n"
            + "```json\n"
            + json.dumps({k: v for k, v in job.items() if k != "result_preview"}, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            + (job.get("result_preview") or job.get("error") or "")[:2000]
        )
    return None
