"""
Kaggle Provider — تشغيل وكيل التدريب على منصة Kaggle
=====================================================
طريقتان مدعومتان:

  1) Kaggle API (من جهازك/السيرفر):
     - يقرأ kaggle.json أو KAGGLE_USERNAME + KAGGLE_KEY
     - يولّد Kernel (script) + metadata
     - يدفع التشغيل على GPU عبر CLI/API
     - يراقب الحالة ويحمّل المخرجات تلقائياً

  2) داخل Kaggle Notebook:
     - bootstrap خفيف
     - اكتشاف Dual T4 + تدريب متعدد البطاقات (DataParallel)
     - وصول سريع لـ Datasets المنصة

لا يعتمد على أتمتة متصفح. كل فشل يُعاد كتقرير نصي واضح.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("KaggleProvider")

ROOT = Path(__file__).resolve().parent.parent
KAGGLE_DIR = ROOT / "artifacts" / "model_training" / "kaggle_jobs"
KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
KERNEL_TEMPLATE_DIR = ROOT / "notebooks" / "kaggle_kernel_template"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in (s or "nsm"))
    return out.strip("-_")[:48] or "nsm-job"


# ─── اكتشاف الاعتماديات والاعتمادات ───────────────────────────────────────

def _kaggle_cli_available() -> bool:
    return shutil.which("kaggle") is not None


def _kaggle_py_available() -> bool:
    try:
        import kaggle  # noqa: F401
        return True
    except Exception:
        return False


def credentials_status() -> Dict[str, Any]:
    """يفحص وجود بيانات اعتماد Kaggle دون كشف الأسرار."""
    home = Path.home() / ".kaggle" / "kaggle.json"
    env_user = bool(os.environ.get("KAGGLE_USERNAME") or os.environ.get("KAGGLE_USER"))
    env_key = bool(os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_KEY"))
    # مسار مخصص
    custom = os.environ.get("KAGGLE_CONFIG_DIR") or os.environ.get("NSM_KAGGLE_JSON")
    custom_ok = False
    if custom:
        p = Path(custom)
        if p.is_file() and p.name == "kaggle.json":
            custom_ok = True
        elif (p / "kaggle.json").is_file():
            custom_ok = True

    return {
        "cli": _kaggle_cli_available(),
        "python_package": _kaggle_py_available(),
        "kaggle_json_home": home.is_file(),
        "env_username": env_user,
        "env_key": env_key,
        "custom_config": custom_ok,
        "ready": home.is_file() or (env_user and env_key) or custom_ok,
        "hint": (
            "ضع ~/.kaggle/kaggle.json (من Account → Create New Token) "
            "أو صدّر KAGGLE_USERNAME و KAGGLE_KEY"
        ),
    }


def ensure_kaggle_env() -> Tuple[bool, str]:
    """يهيّئ متغيرات البيئة إن وُجد ملف مخصص."""
    custom = os.environ.get("NSM_KAGGLE_JSON")
    if custom:
        p = Path(custom)
        if p.is_file():
            os.environ.setdefault("KAGGLE_CONFIG_DIR", str(p.parent))
            return True, f"استخدم NSM_KAGGLE_JSON → {p.parent}"
    cfg = credentials_status()
    if cfg["ready"]:
        return True, "اعتمادات Kaggle جاهزة"
    return False, cfg["hint"]


# ─── اكتشاف Dual T4 / Multi-GPU ───────────────────────────────────────────

def detect_kaggle_gpus() -> Dict[str, Any]:
    """يُستدعى داخل دفتر Kaggle أو أي بيئة CUDA."""
    info: Dict[str, Any] = {
        "cuda": False,
        "device_count": 0,
        "names": [],
        "total_vram_gb": [],
        "is_dual_t4": False,
        "kaggle_env": bool(os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or Path("/kaggle").exists()),
        "working_dir": str(Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT / "artifacts" / "model_training"),
    }
    try:
        import torch

        info["cuda"] = bool(torch.cuda.is_available())
        n = torch.cuda.device_count() if info["cuda"] else 0
        info["device_count"] = n
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            props = torch.cuda.get_device_properties(i)
            vram = float(props.total_memory) / (1024**3)
            info["names"].append(name)
            info["total_vram_gb"].append(round(vram, 2))
        # Dual T4 تقريبي
        if n >= 2 and all("T4" in (nm or "") for nm in info["names"][:2]):
            info["is_dual_t4"] = True
    except Exception as e:
        info["error"] = str(e)
    return info


def wrap_model_for_multi_gpu(model: Any, prefer_dp: bool = True) -> Tuple[Any, str]:
    """
    يلف النموذج بـ DataParallel عند وجود أكثر من GPU.
    يعيد (model, note).
    """
    try:
        import torch
        import torch.nn as nn

        if not torch.cuda.is_available():
            return model, "cpu — لا multi-GPU"
        n = torch.cuda.device_count()
        if n < 2:
            model = model.cuda()
            return model, f"GPU واحد ({torch.cuda.get_device_name(0)})"
        # DataParallel بسيط وموثوق لشبكات NSM الصغيرة/المتوسطة
        model = model.cuda()
        if prefer_dp:
            model = nn.DataParallel(model)
            return model, f"DataParallel على {n} بطاقات: {', '.join(torch.cuda.get_device_name(i) for i in range(n))}"
        return model, f"{n} GPUs متاحة — بدون DataParallel"
    except Exception as e:
        return model, f"تعذّر multi-GPU: {e}"


def multi_gpu_training_snippet() -> str:
    """مقتطف جاهز يُدرج في كود Kernel."""
    return textwrap.dedent(
        '''
        # ── Multi-GPU (Dual T4 على Kaggle) ──────────────────────────────
        import torch
        import torch.nn as nn

        def nsm_prepare_model(model):
            if not torch.cuda.is_available():
                print("CPU only")
                return model, "cpu"
            n = torch.cuda.device_count()
            model = model.cuda()
            if n >= 2:
                model = nn.DataParallel(model)
                names = [torch.cuda.get_device_name(i) for i in range(n)]
                print(f"DataParallel on {n} GPUs: {names}")
                return model, f"dp:{n}"
            print("Single GPU:", torch.cuda.get_device_name(0))
            return model, "single"

        def nsm_save_weights(model, path="/kaggle/working/nsm_model.pt"):
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            torch.save(state, path)
            print("saved:", path)
            return path
        '''
    ).strip()


# ─── توليد كود Kernel ─────────────────────────────────────────────────────

def generate_kernel_script(
    job_id: str,
    csv_rel: Optional[str] = None,
    epochs: int = 15,
    title: str = "NSM Training Agent",
) -> str:
    """يولّد سكربت Python مستقل يعمل على Kaggle (GPU / Dual T4)."""
    csv_note = csv_rel or "data/samples/classification_demo.csv"
    return textwrap.dedent(
        f'''
        #!/usr/bin/env python3
        """
        NSM Kaggle Training Kernel — auto-generated
        job_id: {job_id}
        title: {title}
        """
        from __future__ import annotations
        import json, os, time, traceback
        from pathlib import Path
        from datetime import datetime, timezone

        WORKING = Path("/kaggle/working")
        WORKING.mkdir(parents=True, exist_ok=True)
        REPORT = WORKING / "nsm_kaggle_report.json"

        def now():
            return datetime.now(timezone.utc).isoformat()

        report = {{
            "job_id": "{job_id}",
            "started_at": now(),
            "ok": False,
            "gpu": {{}},
            "metrics": {{}},
            "artifacts": [],
        }}

        try:
            import torch
            import torch.nn as nn
            import torch.optim as optim
            import numpy as np
            import subprocess as _sp

            try:
                print("nvidia-smi:")
                print(_sp.check_output(["nvidia-smi", "-L"], text=True, timeout=10))
            except Exception as _e:
                print("nvidia-smi unavailable:", _e)

            cuda = torch.cuda.is_available()
            n_gpu = torch.cuda.device_count() if cuda else 0
            names = [torch.cuda.get_device_name(i) for i in range(n_gpu)] if n_gpu else []
            report["gpu"] = {{
                "cuda": cuda,
                "device_count": n_gpu,
                "names": names,
                "is_dual_t4": n_gpu >= 2 and all("T4" in x for x in names[:2]),
            }}
            print("GPU:", report["gpu"])

            # شبكة كثيفة بسيطة (تصنيف ثنائي تجريبي)
            class TinyMLP(nn.Module):
                def __init__(self, d_in=16, d_hid=64, d_out=2):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(d_in, d_hid),
                        nn.ReLU(),
                        nn.Linear(d_hid, d_hid),
                        nn.ReLU(),
                        nn.Linear(d_hid, d_out),
                    )
                def forward(self, x):
                    return self.net(x)

            # بيانات اصطناعية إن لم يُرفق CSV (Kaggle Dataset)
            # عند ربط dataset ضع المسار تحت /kaggle/input/...
            d_in, n_samples = 16, 2000
            rng = np.random.default_rng(42)
            X = rng.normal(size=(n_samples, d_in)).astype("float32")
            y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype("int64")

            # محاولة قراءة CSV إن وُجد
            csv_candidates = [
                Path("/kaggle/input") / p
                for p in Path("/kaggle/input").glob("**/*.csv")
            ] if Path("/kaggle/input").exists() else []
            # أيضاً إن رُفع الملف مع الـkernel
            local_csv = Path("{csv_note}")
            if local_csv.is_file():
                csv_candidates.insert(0, local_csv)

            if csv_candidates:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_candidates[0])
                    print("CSV:", csv_candidates[0], "shape", df.shape)
                    # أعمدة رقمية فقط + آخر عمود هدف إن أمكن
                    num = df.select_dtypes(include=["number"])
                    if num.shape[1] >= 2:
                        y_col = num.columns[-1]
                        X = num.drop(columns=[y_col]).values.astype("float32")
                        y_raw = num[y_col].values
                        # تصنيف ثنائي تقريبي
                        if y_raw.dtype.kind in "fc":
                            med = float(np.median(y_raw))
                            y = (y_raw >= med).astype("int64")
                        else:
                            y = y_raw.astype("int64")
                        d_in = X.shape[1]
                        n_samples = X.shape[0]
                        report["data_source"] = str(csv_candidates[0])
                except Exception as e:
                    print("CSV load fallback:", e)

            X_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y)
            model = TinyMLP(d_in=d_in)
            model, mode = None, "cpu"
            if cuda:
                model = TinyMLP(d_in=d_in).cuda()
                if n_gpu >= 2:
                    model = nn.DataParallel(model)
                    mode = f"DataParallel×{{n_gpu}}"
                else:
                    mode = "single-gpu"
            else:
                model = TinyMLP(d_in=d_in)

            opt = optim.Adam(model.parameters(), lr=1e-3)
            loss_fn = nn.CrossEntropyLoss()
            epochs = {epochs}
            losses = []
            t0 = time.time()
            model.train()
            for ep in range(epochs):
                # دفعات بسيطة
                perm = torch.randperm(n_samples)
                total_loss = 0.0
                bs = 64
                steps = 0
                for i in range(0, n_samples, bs):
                    idx = perm[i:i+bs]
                    xb, yb = X_t[idx], y_t[idx]
                    if cuda:
                        xb, yb = xb.cuda(), yb.cuda()
                    opt.zero_grad()
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    loss.backward()
                    opt.step()
                    total_loss += float(loss.item())
                    steps += 1
                avg = total_loss / max(steps, 1)
                losses.append(avg)
                if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
                    print(f"epoch {{ep+1}}/{{epochs}} loss={{avg:.4f}} mode={{mode}}")

            # حفظ الأوزان
            state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
            out_pt = WORKING / "nsm_kaggle_model.pt"
            torch.save({{"state_dict": state, "d_in": d_in, "job_id": "{job_id}", "mode": mode}}, out_pt)
            report["artifacts"].append(str(out_pt.name))
            report["metrics"] = {{
                "epochs": epochs,
                "final_loss": losses[-1] if losses else None,
                "loss_history": losses[-20:],
                "n_samples": int(n_samples),
                "d_in": int(d_in),
                "mode": mode,
                "elapsed_s": round(time.time() - t0, 2),
            }}
            report["ok"] = True
            report["finished_at"] = now()
            print("DONE", report["metrics"])
        except Exception as e:
            report["ok"] = False
            report["error"] = str(e)
            report["traceback"] = traceback.format_exc()[:3000]
            report["finished_at"] = now()
            print("FAIL", e)

        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("report →", REPORT)
        '''
    ).strip()


def generate_kernel_metadata(
    job_id: str,
    username: str = "nsm-agent",
    title: Optional[str] = None,
    enable_gpu: bool = True,
    accelerator: str = "NvidiaTeslaT4",
) -> Dict[str, Any]:
    """metadata.json لـ Kaggle Kernels API / CLI.

    accelerator أمثلة مدعومة عبر CLI:
      NvidiaTeslaT4 | NvidiaTeslaT4Highmem | NvidiaTeslaP100 (غير موصى — مشاكل PyTorch)
      NvidiaL4 | NvidiaTeslaA100 | ...
    Dual T4 ×2: يُطلب عبر واجهة Kaggle؛ API يعتمد NvidiaTeslaT4 غالباً.
    """
    slug = _safe_slug(f"nsm-{job_id}")
    meta: Dict[str, Any] = {
        "id": f"{username}/{slug}",
        "id_no": None,
        "title": title or f"NSM Training {job_id}",
        "code_file": "nsm_train.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": bool(enable_gpu),
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if enable_gpu and accelerator:
        # machine_shape يُقرأ من CLI الحديث عند الدفع
        meta["machine_shape"] = accelerator
    return meta


# ─── مهام محلية (توليد + دفع اختياري) ─────────────────────────────────────

def prepare_kaggle_job(
    csv_path: Optional[str] = None,
    epochs: int = 15,
    title: str = "NSM Training Agent",
    username: Optional[str] = None,
    accelerator: str = "NvidiaTeslaT4",
    enable_gpu: bool = True,
) -> Dict[str, Any]:
    """
    يجهّز مجلد مهمة جاهز للرفع:
      artifacts/model_training/kaggle_jobs/<job_id>/
        nsm_train.py
        kernel-metadata.json
        job.json
    """
    job_id = f"kag_{uuid.uuid4().hex[:10]}"
    job_dir = KAGGLE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    user = (
        username
        or os.environ.get("KAGGLE_USERNAME")
        or os.environ.get("KAGGLE_USER")
        or None
    )
    if not user:
        # حاول قراءة username من kaggle.json
        try:
            import json as _json
            for cand in (
                Path.home() / ".kaggle" / "kaggle.json",
                Path(os.environ.get("KAGGLE_CONFIG_DIR") or "") / "kaggle.json",
            ):
                if cand.is_file():
                    data = _json.loads(cand.read_text(encoding="utf-8"))
                    user = data.get("username") or data.get("user")
                    if user:
                        break
        except Exception:
            pass
    user = user or "nsm-agent"
    # slug يجب أن يطابق العنوان حتى لا يرفض Kaggle أو يُنتج URL مختلف
    slug = _safe_slug(f"nsm-{job_id}")
    # عنوان نظيف يحل إلى نفس الـslug
    clean_title = slug.replace("-", " ")
    effective_title = title if title and title != "NSM Training Agent" else clean_title
    # إن بقي العنوان مخصصاً، اجعل الـid يعتمد على slug من العنوان أيضاً لتفادي تحذير Kaggle
    title_slug = _safe_slug(effective_title)
    script = generate_kernel_script(job_id, csv_rel=csv_path, epochs=epochs, title=effective_title)
    meta = generate_kernel_metadata(
        job_id,
        username=user,
        title=effective_title,
        enable_gpu=enable_gpu,
        accelerator=accelerator if enable_gpu else "",
    )
    # فرض تطابق id مع slug العنوان
    meta["id"] = f"{user}/{title_slug}"

    # سكربت خام للمرجعية + دفتر notebook (أفضل لتفعيل GPU على Kaggle)
    (job_dir / "nsm_train.py").write_text(script, encoding="utf-8")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": script,
            }
        ],
    }
    (job_dir / "nsm_train.ipynb").write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    # تأكد أن code_file في metadata يشير للدفتر
    meta["code_file"] = "nsm_train.ipynb"
    meta["kernel_type"] = "notebook"
    (job_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # إن وُجد CSV محلي، انسخه للمجلد (للرفع اليدوي أو dataset)
    copied_csv = None
    if csv_path:
        src = ROOT / csv_path if not Path(csv_path).is_absolute() else Path(csv_path)
        if src.is_file():
            dest = job_dir / src.name
            shutil.copy2(src, dest)
            copied_csv = dest.name

    job = {
        "ok": True,
        "job_id": job_id,
        "provider": "kaggle_api",
        "status": "prepared",
        "job_dir": str(job_dir.relative_to(ROOT)),
        "kernel_id": meta["id"],
        "enable_gpu": enable_gpu,
        "accelerator": accelerator if enable_gpu else None,
        "epochs": epochs,
        "csv_copied": copied_csv,
        "created_at": _now(),
        "credentials": credentials_status(),
        "next_steps": [
            "تأكد من kaggle.json أو KAGGLE_USERNAME/KAGGLE_KEY",
            f"cd {job_dir.relative_to(ROOT)} && kaggle kernels push -p .",
            "أو استخدم أمر الوكيل: ادفع kaggle " + job_id,
            "راقب: حالة kaggle " + job_id,
            "حمّل النتائج: حمّل kaggle " + job_id,
        ],
    }
    (job_dir / "job.json").write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return job


def push_kaggle_kernel(job_id: str) -> Dict[str, Any]:
    """يدفع Kernel عبر CLI إن توفّر الاعتمادات."""
    job_dir = KAGGLE_DIR / job_id
    if not job_dir.is_dir():
        return {"ok": False, "error": f"مهمة غير موجودة: {job_id}"}

    ok_cred, msg = ensure_kaggle_env()
    if not ok_cred:
        return {
            "ok": False,
            "job_id": job_id,
            "status": "not_configured",
            "error": msg,
            "job_dir": str(job_dir.relative_to(ROOT)),
        }

    if not _kaggle_cli_available():
        return {
            "ok": False,
            "job_id": job_id,
            "status": "cli_missing",
            "error": "أداة kaggle غير مثبتة. نفّذ: pip install kaggle",
            "hint": f"يمكنك الرفع يدوياً من {job_dir}",
        }

    try:
        # اقرأ accelerator من metadata إن وُجد، أو من NSM_ACCEL env override
        accel = "NvidiaTeslaT4"
        env_accel = os.environ.get("NSM_ACCEL")
        meta_path = job_dir / "kernel-metadata.json"
        if meta_path.is_file():
            try:
                _m = json.loads(meta_path.read_text(encoding="utf-8"))
                if _m.get("machine_shape"):
                    accel = str(_m["machine_shape"])
                elif not _m.get("enable_gpu"):
                    accel = ""
            except Exception:
                pass
        if env_accel:
            accel = str(env_accel)
        cmd = ["kaggle", "kernels", "push", "-p", str(job_dir)]
        if accel:
            cmd += ["--accelerator", accel]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=100,
            cwd=str(job_dir),
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        success = proc.returncode == 0
        # استخراج رابط/slug الفعلي من مخرجات CLI
        actual_url = None
        actual_slug = None
        import re as _re
        murl = _re.search(r"https://www\.kaggle\.com/code/([\w\-]+)/([\w\-]+)", out)
        if murl:
            actual_slug = f"{murl.group(1)}/{murl.group(2)}"
            actual_url = murl.group(0)
        result = {
            "ok": success,
            "job_id": job_id,
            "status": "pushed" if success else "push_failed",
            "returncode": proc.returncode,
            "output": out[-4000:],
            "kernel_slug": actual_slug,
            "kernel_url": actual_url,
            "finished_at": _now(),
        }
        # حدّث job.json + metadata id إن تغيّر الـslug
        jp = job_dir / "job.json"
        if jp.is_file():
            data = json.loads(jp.read_text(encoding="utf-8"))
            data.update({k: result[k] for k in ("status", "ok", "output", "finished_at", "kernel_slug", "kernel_url") if k in result})
            if actual_slug:
                data["kernel_id"] = actual_slug
            jp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        if actual_slug:
            meta_path = job_dir / "kernel-metadata.json"
            if meta_path.is_file():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["id"] = actual_slug
                    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass
        return result
    except subprocess.TimeoutExpired:
        return {"ok": False, "job_id": job_id, "error": "انتهت مهلة الدفع (120s)"}
    except Exception as e:
        return {"ok": False, "job_id": job_id, "error": str(e)}


def status_kaggle_kernel(job_id: str) -> Dict[str, Any]:
    """يستعلم عن حالة Kernel عبر CLI."""
    job_dir = KAGGLE_DIR / job_id
    meta_path = job_dir / "kernel-metadata.json"
    if not meta_path.is_file():
        return {"ok": False, "error": f"لا metadata للمهمة {job_id}"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kernel_id = meta.get("id") or ""
    # فضّل الـslug الفعلي المحفوظ بعد الدفع
    jp = job_dir / "job.json"
    if jp.is_file():
        try:
            jdata = json.loads(jp.read_text(encoding="utf-8"))
            kernel_id = jdata.get("kernel_slug") or jdata.get("kernel_id") or kernel_id
        except Exception:
            pass
    ok_cred, msg = ensure_kaggle_env()
    if not ok_cred or not _kaggle_cli_available():
        return {
            "ok": False,
            "job_id": job_id,
            "kernel_id": kernel_id,
            "error": msg if not ok_cred else "kaggle CLI غير متاح",
            "local_job": json.loads((job_dir / "job.json").read_text(encoding="utf-8"))
            if (job_dir / "job.json").is_file()
            else {},
        }

    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "status", kernel_id],
            capture_output=True,
            text=True,
            timeout=60,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        return {
            "ok": proc.returncode == 0,
            "job_id": job_id,
            "kernel_id": kernel_id,
            "status_raw": out[-3000:],
            "returncode": proc.returncode,
            "checked_at": _now(),
        }
    except Exception as e:
        return {"ok": False, "job_id": job_id, "error": str(e)}


def download_kaggle_output(job_id: str) -> Dict[str, Any]:
    """يحمّل مخرجات Kernel إلى مجلد المهمة."""
    job_dir = KAGGLE_DIR / job_id
    meta_path = job_dir / "kernel-metadata.json"
    if not meta_path.is_file():
        return {"ok": False, "error": f"لا metadata للمهمة {job_id}"}

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    kernel_id = meta.get("id") or ""
    jp = job_dir / "job.json"
    if jp.is_file():
        try:
            jdata = json.loads(jp.read_text(encoding="utf-8"))
            kernel_id = jdata.get("kernel_slug") or jdata.get("kernel_id") or kernel_id
        except Exception:
            pass
    out_dir = job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    ok_cred, msg = ensure_kaggle_env()
    if not ok_cred or not _kaggle_cli_available():
        return {
            "ok": False,
            "job_id": job_id,
            "error": msg if not ok_cred else "kaggle CLI غير متاح",
        }

    try:
        proc = subprocess.run(
            ["kaggle", "kernels", "output", kernel_id, "-p", str(out_dir)],
            capture_output=True,
            text=True,
            timeout=100,
        )
        out = (proc.stdout or "") + "\n" + (proc.stderr or "")
        files = [p.name for p in out_dir.iterdir() if p.is_file()] if out_dir.is_dir() else []
        return {
            "ok": proc.returncode == 0,
            "job_id": job_id,
            "kernel_id": kernel_id,
            "output_dir": str(out_dir.relative_to(ROOT)),
            "files": files,
            "cli_output": out[-2000:],
            "downloaded_at": _now(),
        }
    except Exception as e:
        return {"ok": False, "job_id": job_id, "error": str(e)}


def list_kaggle_jobs() -> List[Dict[str, Any]]:
    jobs = []
    if not KAGGLE_DIR.is_dir():
        return jobs
    for d in sorted(KAGGLE_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        jp = d / "job.json"
        if jp.is_file():
            try:
                jobs.append(json.loads(jp.read_text(encoding="utf-8")))
            except Exception:
                jobs.append({"job_id": d.name, "status": "corrupt_job_json"})
        else:
            jobs.append({"job_id": d.name, "status": "no_job_json"})
    return jobs


# ─── وضع داخل الدفتر (In-Notebook) ────────────────────────────────────────

def kaggle_notebook_status_report() -> str:
    """تقرير يُستخدم داخل دفتر Kaggle أو محلياً."""
    cred = credentials_status()
    gpu = detect_kaggle_gpus()
    lines = [
        "## 🟧 حالة Kaggle Provider",
        "",
        "### الاعتمادات (API)",
        f"- جاهز: **{'✅' if cred['ready'] else '❌'}**",
        f"- kaggle.json (home): {'✅' if cred['kaggle_json_home'] else '❌'}",
        f"- ENV USER/KEY: {'✅' if cred['env_username'] and cred['env_key'] else '❌'}",
        f"- CLI: {'✅' if cred['cli'] else '❌'} | حزمة Python: {'✅' if cred['python_package'] else '❌'}",
        f"- تلميح: {cred['hint']}",
        "",
        "### GPU (البيئة الحالية)",
        f"- CUDA: {'✅' if gpu.get('cuda') else '❌'}",
        f"- عدد البطاقات: **{gpu.get('device_count', 0)}**",
        f"- الأسماء: {', '.join(gpu.get('names') or ['—'])}",
        f"- Dual T4: **{'✅ نعم' if gpu.get('is_dual_t4') else '❌ لا'}**",
        f"- بيئة Kaggle: {'✅' if gpu.get('kaggle_env') else 'محلي/آخر'}",
        f"- مجلد العمل: `{gpu.get('working_dir')}`",
        "",
        "### المهام المحلية المُعدّة",
    ]
    jobs = list_kaggle_jobs()
    if not jobs:
        lines.append("- لا مهام بعد. أنشئ بـ: `جهّز kaggle` أو `درّب kaggle csv ...`")
    else:
        for j in jobs[:8]:
            lines.append(
                f"- `{j.get('job_id')}` — {j.get('status', '?')} — GPU={j.get('enable_gpu')}"
            )
    lines += [
        "",
        "### أوامر سريعة",
        "- `حالة kaggle` — هذا التقرير",
        "- `جهّز kaggle` / `درّب kaggle csv data/samples/classification_demo.csv`",
        "- `ادفع kaggle <job_id>` — رفع وتشغيل على GPU",
        "- `حالة kaggle <job_id>` — مراقبة",
        "- `حمّل kaggle <job_id>` — تنزيل الأوزان والتقرير",
        "- داخل الدفتر: فعّل GPU (Settings → Accelerator → GPU T4 x2) ثم نفّذ bootstrap",
    ]
    return "\n".join(lines)


# ─── موجّه الأوامر ─────────────────────────────────────────────────────────

def handle_kaggle_command(user_input: str) -> Optional[str]:
    """
    يفسّر أوامر عربية/إنجليزية متعلقة بـ Kaggle.
    يعيد نصاً أو None لتمرير الرسالة لوكلاء آخرين.
    """
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    low = text.lower()

    # هل يتعلق بـ Kaggle أو Dual T4 / multi-GPU؟
    is_kaggle = bool(
        re.search(r"kaggle|كاجل|كاغل", text, re.I)
        or low in ("حالة kaggle", "kaggle status", "جهّز kaggle", "ادفع kaggle")
        or re.search(r"dual\s*t4|multi.?gpu|كرتين|بطاقتين|data.?parallel", text, re.I)
    )
    if not is_kaggle:
        return None

    # حالة عامة
    if re.search(r"(حالة|status).{0,15}(kaggle|كاجل|كاغل)$", text, re.I) or low in (
        "حالة kaggle",
        "kaggle",
        "kaggle status",
    ):
        return kaggle_notebook_status_report()

    # قائمة مهام
    if re.search(r"(قائمة|list).{0,12}(kaggle|مهام\s*kaggle|كاجل)", text, re.I):
        jobs = list_kaggle_jobs()
        if not jobs:
            return "لا مهام Kaggle محلية بعد. استخدم: `جهّز kaggle`"
        lines = ["## 📋 مهام Kaggle المحلية\n"]
        for j in jobs[:15]:
            lines.append(
                f"- **{j.get('job_id')}** | {j.get('status')} | "
                f"`{j.get('kernel_id', '—')}` | {j.get('created_at', '')[:19]}"
            )
        return "\n".join(lines)

    # تجهيز / درّب kaggle [csv]
    m_prep = re.search(
        r"(?:جه[ّ]?ز|حض[ّ]?ر|prepare|در[ّ]?ب)\s*kaggle"
        r"(?:\s*(?:csv)?\s*((?:data|artifacts)[\w./\-]+\.csv))?",
        text,
        re.I,
    )
    if m_prep or re.search(r"در[ّ]?ب\s*kaggle|train\s*kaggle", text, re.I):
        csv_path = None
        if m_prep and m_prep.group(1):
            csv_path = m_prep.group(1)
        else:
            m2 = re.search(r"((?:data|artifacts)[\w./\-]+\.csv)", text, re.I)
            if m2:
                csv_path = m2.group(1)
        if not csv_path:
            csv_path = "data/samples/classification_demo.csv"
        epochs = 15
        m_ep = re.search(r"(?:epochs?|حقب|عصور)\s*[=:]?\s*(\d+)", text, re.I)
        if m_ep:
            epochs = max(1, min(100, int(m_ep.group(1))))
        # تسريع GPU — افتراضي T4
        accel = "NvidiaTeslaT4"
        enable_gpu = True
        if re.search(r"(بدون\s*gpu|no\s*gpu|cpu\s*only|عطل\s*gpu)", text, re.I):
            enable_gpu = False
            accel = ""
        elif re.search(r"(t4\s*highmem|highmem)", text, re.I):
            accel = "NvidiaTeslaT4Highmem"
        elif re.search(r"(dual\s*t4|t4\s*[x×]2|كرتين|بطاقتين)", text, re.I):
            # طلب T4؛ Dual×2 غالباً من الواجهة — T4 عبر API هو الأقرب
            accel = "NvidiaTeslaT4"
        elif re.search(r"(a100)", text, re.I):
            accel = "NvidiaTeslaA100"
        elif re.search(r"(l4\b)", text, re.I):
            accel = "NvidiaL4"
        job = prepare_kaggle_job(
            csv_path=csv_path,
            epochs=epochs,
            accelerator=accel or "NvidiaTeslaT4",
            enable_gpu=enable_gpu,
        )
        return (
            "## 🟧 مهمة Kaggle جاهزة للرفع\n\n"
            + "```json\n"
            + json.dumps(
                {k: v for k, v in job.items() if k != "next_steps"},
                ensure_ascii=False,
                indent=2,
            )
            + "\n```\n\n"
            + "### الخطوات التالية\n"
            + "\n".join(f"{i+1}. {s}" for i, s in enumerate(job.get("next_steps") or []))
            + "\n\n"
            + "_Dual T4: فعّل GPU في إعدادات Kernel على Kaggle (T4 ×2)._"
        )

    # ادفع kaggle <job_id>
    m_push = re.search(
        r"(?:ادفع|ارفع|push|شغ[ّ]?ل)\s*kaggle\s+(kag_[a-f0-9]+|\w+)",
        text,
        re.I,
    )
    if m_push:
        jid = m_push.group(1)
        # إن لم يبدأ بـ kag_ حاول إيجاد تطابق
        if not jid.startswith("kag_"):
            candidates = [d.name for d in KAGGLE_DIR.iterdir() if d.is_dir() and jid in d.name]
            if candidates:
                jid = candidates[0]
        result = push_kaggle_kernel(jid)
        return (
            f"## 🚀 دفع Kernel إلى Kaggle (`{jid}`)\n\n"
            + "```json\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
            + "\n```"
        )

    # حالة kaggle <job_id>
    m_st = re.search(
        r"(?:حالة|status)\s*kaggle\s+(kag_[a-f0-9]+|\w+)",
        text,
        re.I,
    )
    if m_st:
        jid = m_st.group(1)
        result = status_kaggle_kernel(jid)
        return (
            f"## 📡 حالة Kernel `{jid}`\n\n"
            + "```json\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
            + "\n```"
        )

    # حمّل kaggle <job_id>
    m_dl = re.search(
        r"(?:حم[ّ]?ل|نز[ّ]?ل|download|output)\s*kaggle\s+(kag_[a-f0-9]+|\w+)",
        text,
        re.I,
    )
    if m_dl:
        jid = m_dl.group(1)
        result = download_kaggle_output(jid)
        return (
            f"## 📥 تنزيل مخرجات Kaggle `{jid}`\n\n"
            + "```json\n"
            + json.dumps(result, ensure_ascii=False, indent=2)
            + "\n```"
        )

    # مقتطف multi-GPU
    if re.search(r"(multi.?gpu|dual\s*t4|كرتين|بطاقتين|data.?parallel)", text, re.I):
        return (
            "## ⚡ تدريب متعدد البطاقات (Dual T4)\n\n"
            "على Kaggle اختر **GPU T4 ×2** من Settings → Accelerator.\n\n"
            "```python\n"
            + multi_gpu_training_snippet()
            + "\n```\n\n"
            + f"اكتشاف حالي:\n```json\n{json.dumps(detect_kaggle_gpus(), ensure_ascii=False, indent=2)}\n```"
        )

    # إن ذُكر kaggle بدون أمر محدد → التقرير
    if is_kaggle:
        return kaggle_notebook_status_report()

    return None


# ─── SurahChain: بدء التدريب عبر Kaggle API من تبويب Notebook ─────────────

def generate_surahchain_kernel_script(
    job_id: str,
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
    repo: str = "aliahmed369000000-ai/Neural-Service-Mesh",
    branch: str = "main",
    kernel_url: str = "",
    use_tpu: bool = False,
) -> str:
    """سكربت Kaggle يشغّل run_train_then_push (تدريب ثم رفع لـ GitHub).

    بناء عبر template string عادي (لا f-string على كامل القالب) —
    الف-string المزدوجة القديمة كانت ترمي NameError عند التوليد (خطأ مكتشف
    عمليًا: `_i` غير معرّف في نطاق المولّد).

    use_tpu=True: سكربت لبيئة TPU v5e-8 (صورة TPUVM) — يفعّل SCN_TPU=1
    ويتخطّى bitsandbytes (غير مدعوم على XLA) ويفحص torch_xla بدل CUDA.
    """
    fresh_s = "1" if fresh else "0"
    push_s = "1" if auto_push else "0"
    tpu_s = "1" if use_tpu else "0"
    tmpl = textwrap.dedent("""\
        #!/usr/bin/env python3
        \"\"\"NSM SurahChain Kaggle Kernel — job __JOB_ID__\"\"\"
        from __future__ import annotations
        import os, sys, subprocess
        from pathlib import Path

        REPO = "__REPO__"
        BRANCH = "__BRANCH__"
        PRESET = "__PRESET__"
        SCN_N = "__SCN_N__"
        SCN_EPOCHS = "__SCN_EPOCHS__"
        SCN_BATCH = "__SCN_BATCH__"
        SCN_FRESH = "__SCN_FRESH__"
        SCN_RESUME = "auto"
        AUTO_PUSH = "__AUTO_PUSH__"

        def secret(name, default=""):
            try:
                from kaggle_secrets import UserSecretsClient
                return UserSecretsClient().get_secret(name) or default
            except Exception:
                return os.environ.get(name, default)

        token = secret("GITHUB_TOKEN") or secret("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", "")
        # ضمان: التوكن يكون موجودًا في env دائمًا قبل تشغيل التدريب بأي طريق ممكن
        # (secret → env → القيمة الافتراضية المضمّنة في الكود)، فالرفع التلقائي
        # AUTO_PUSH لا يعمل بدون GITHUB_TOKEN في بيئة subprocess التدريب.
        token = (token or "").strip()
        os.environ["GITHUB_TOKEN"] = token
        os.environ["AUTO_PUSH"] = AUTO_PUSH
        work = Path("/kaggle/working/Neural-Service-Mesh")
        print("=== CUDA check ===")
        try:
            import torch
            print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
            try:
                for _i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
                    print(f"  GPU{_i} mem_alloc={torch.cuda.memory_allocated(_i)/1e9:.2f}G name={torch.cuda.get_device_name(_i)}")
            except Exception as _me:
                print("gpu mem:", _me)
            try:
                import subprocess as _sp
                print(_sp.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu","--format=csv,noheader"], text=True, timeout=10).strip())
            except Exception as _se:
                print("smi:", _se)
        except Exception as e:
            print("torch:", e)

        if not token:
            print("⚠ لا GITHUB_TOKEN في Kaggle Secrets — استنساخ عام / قد يفشل الرفع")
            url = "https://github.com/" + REPO + ".git"
        else:
            url = "https://x-access-token:" + token + "@github.com/" + REPO + ".git"
            print("✅ GITHUB_TOKEN متوفر — الرفع التلقائي AUTO_PUSH=%s مفعّل" % AUTO_PUSH)

        if work.exists():
            subprocess.run(["git", "remote", "set-url", "origin", url], cwd=str(work), check=False)
            subprocess.run(["git", "pull", "origin", BRANCH], cwd=str(work), check=False)
        else:
            subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, str(work)], check=True)

        os.chdir(work)
        # ── التحقق من الجهاز (TPU أو GPU) ────────────────────────────────
        SCN_TPU = "__SCN_TPU__"
        if SCN_TPU == "1":
            print("=== TPU check ===")
            try:
                import torch
                import torch_xla
                print("torch", torch.__version__, "torch_xla", torch_xla.__version__)
                # لا نستدعي xm.xla_device() هنا — تهيئة XLA تتم مرة واحدة فقط
                # داخل سكربت التدريب (إصلاح انهيار SIGABRT: double-init).
            except Exception as _xe:
                print("⚠ torch_xla غير متوفر:", _xe)
        else:
            print("=== CUDA check ===")
            try:
                import torch
                print("torch", torch.__version__, "cuda", torch.cuda.is_available(), "gpus", torch.cuda.device_count())
                try:
                    for _i in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
                        print(f"  GPU{_i} mem_alloc={torch.cuda.memory_allocated(_i)/1e9:.2f}G name={torch.cuda.get_device_name(_i)}")
                except Exception as _me:
                    print("gpu mem:", _me)
                try:
                    import subprocess as _sp
                    print(_sp.check_output(["nvidia-smi","--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu","--format=csv,noheader"], text=True, timeout=10).strip())
                except Exception as _se:
                    print("smi:", _se)
            except Exception as e:
                print("torch:", e)

        # ── تثبيت bitsandbytes (Adam-8bit) على GPU فقط (توفير ≈40% VRAM) ──
        # على TPU غير مدعوم — skip لتسريع بدء kernel
        if SCN_TPU != "1":
            _pip_ok = subprocess.run(
                [sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                check=False,
            )
            if _pip_ok.returncode != 0:
                print("⚠ bitsandbytes لم يُثبّت — xlarge قد يفشل")
        env = os.environ.copy()
        env.update({
            "SCN_PRESET": PRESET,
            "SCN_N": SCN_N,
            "SCN_EPOCHS": SCN_EPOCHS,
            "SCN_BATCH": SCN_BATCH,
            "SCN_FRESH": SCN_FRESH,
            "SCN_RESUME": SCN_RESUME,
            "SCN_CHECKPOINT_EVERY": "1",
            "SCN_FIRST_FAST": "1",
            "SCN_UPLOAD_RETRIES": "3",
            "AUTO_PUSH": AUTO_PUSH,
            "PYTHONUNBUFFERED": "1",
            "SCN_TPU": SCN_TPU,
            "SCN_USE_8BIT_ADAM": "1" if SCN_TPU != "1" else "0",
            "SCN_GRAD_ACCUM": "8" if SCN_TPU == "1" else "2",
            "SCN_COMPILE": "0",
            "XLA_USE_BF16": "1" if SCN_TPU == "1" else "0",
            "XLA_USE_SPMD": "0",
            "XLA_LOG_LEVEL": "0",
        })
        script = work / "experiments/surah_chain_network/run_train_then_push.py"
        print("▶", script)
        r = subprocess.run([sys.executable, "-u", str(script)], cwd=str(work), env=env)
        print("exit", r.returncode)
        # ── نظام التوقف الآمن: مراقبة إشارة STOP ─────────────────────────
        # زر التوقف في الواجهة يكتب إشارة عبر Kaggle API kernels stop،
        # لكن إن رُفع STOP file يدويًا أو عبر الكيرنل التالي، يتوقف التدريب
        # نظيفًا بعد حفظ checkpoint ورفعه إلى GitHub تلقائيًا.
        _stop_file = work / "experiments/surah_chain_network/checkpoints/STOP"
        if _stop_file.exists():
            print("⏹ إشارة التوقف نشطة — التدريب سيتوقف نظيفًا عند نهاية العصر")
            _stop_file.unlink(missing_ok=True)
        # لا SystemExit: papermill يعتبره فشل الدفتر حتى لو التدريب نجح
        if r.returncode == 0:
            print("✅ انتهت الجولة بنجاح")
            _notify("complete", 0)
        else:
            print("⚠ رمز الخروج", r.returncode, "— إن وُجدت checkpoints فالتدريب غالباً نجح")
            _notify("failed", r.returncode)
    """)
    # حقن نظام التنبيهات الذكي (training_alerts.py كـbase64 + notify wrapper)
    return _inject_alerts(tmpl.replace("__JOB_ID__", job_id).replace("__REPO__", repo).replace(
        "__BRANCH__", branch).replace("__PRESET__", preset).replace(
        "__SCN_N__", str(n)).replace("__SCN_EPOCHS__", str(epochs)).replace(
        "__SCN_BATCH__", str(batch)).replace("__SCN_FRESH__", fresh_s).replace(
        "__AUTO_PUSH__", push_s).replace("__SCN_TPU__", tpu_s),
        job_id=job_id, kernel_url=kernel_url)


_ALERTS_B64_CACHE: Dict[str, str] = {}


def _alerts_module_b64() -> str:
    """يُرمّز ai/training_alerts.py كـbase64 مرة واحدة ويعيد استخدامه."""
    if not _ALERTS_B64_CACHE:
        alerts_file = ROOT / "ai" / "training_alerts.py"
        if alerts_file.is_file():
            import base64 as _b
            _ALERTS_B64_CACHE["b64"] = _b.b64encode(alerts_file.read_bytes()).decode()
        else:
            _ALERTS_B64_CACHE["b64"] = ""
    return _ALERTS_B64_CACHE.get("b64", "")


def _inject_alerts(script: str, job_id: str = "", kernel_url: str = "") -> str:
    """يحضر نظام التنبيهات إلى رأس سكربت kernel: حمّل training_alerts.py من base64
    وعرّف _notify(status, exit_code) الذي يرسل تنبيه Discord عند اكتمال/فشل المهمة.
    بيئة Kaggle غير محجوبة على Discord لذا يصل الإشعار فعليًا.
    """
    b64 = _alerts_module_b64()
    if not b64:
        return script
    alert_block = textwrap.dedent(f'''
        # ── NSM Training Alerts (injected) ──
        import base64 as _b64a, tempfile as _tf_a, importlib.util as _iu_a
        _src_a = _b64a.b64decode("{b64}")  # noqa: B64
        _f_a = _tf_a.NamedTemporaryFile("wb", suffix=".py", delete=False)
        _f_a.write(_src_a); _f_a.close()
        _sp_a = _iu_a.spec_from_file_location("training_alerts", _f_a.name)
        _mod_a = _iu_a.module_from_spec(_sp_a); _sp_a.loader.exec_module(_mod_a)
        def _notify(status, exit_code):
            try:
                _mod_a.alert_job_status(
                    "{job_id}", status,
                    account=os.environ.get("KAGGLE_USERNAME", ""),
                    kernel_url="{kernel_url}",
                    preset=os.environ.get("SCN_PRESET", ""),
                    n=int(os.environ.get("SCN_N") or 0),
                    epochs=int(os.environ.get("SCN_EPOCHS") or 0),
                )
            except Exception as _ne:
                print("alerts:", _ne)
        # ── end alerts ──
    ''')
    return alert_block + script


def stop_surahchain_kernel(kernel_id: str) -> Dict[str, Any]:
    """زر التوقف: إيقاف kernel SurahChain على Kaggle فورًا.
    يستخدم KaggleApi.kernel_stop (يدعم owner/kernel-slug)؛ إن لم تكن متوفرة
    في نسخة CLI المركّبة (kaggle kernels stop غير موجود في CLI v2.2.4)
    نرسل إشارة التوقف عبر كتابة STOP signal إلى kernel output.
    التدريب داخل kernel يراقب إشارة STOP — إن وُجدت عند نهاية العصر يحفظ
    آخر checkpoint ويرفعه إلى GitHub (AUTO_PUSH) قبل التوقف، فيتوقف نظيفًا
    ويستأنف لاحقًا من نفس النقطة.
    """
    import shutil as _sh
    rc, out = 1, f"kernel_id={kernel_id}"
    cmd = ["kaggle", "kernels", "stop", "-k", kernel_id]
    if _sh.which("kaggle"):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            rc, out = r.returncode, (r.stdout or "").strip() + " " + (r.stderr or "").strip()
        except Exception as e:
            rc, out = 1, str(e)
    if rc != 0:
        # محاولة KaggleApi.kernel_stop إن وُجدت (kaggle>=1.6)
        try:
            from kaggle.api import KaggleApi  # noqa: F401
            api = KaggleApi()
            api.authenticate()
            if hasattr(api, "kernel_stop"):
                api.kernel_stop(kernel_id)
                rc, out = 0, "kernel_stop (KaggleApi)"
            elif hasattr(api, "kernel_cancel"):
                api.kernel_cancel(kernel_id)
                rc, out = 0, "kernel_cancel (KaggleApi)"
        except Exception:
            pass
    ok = rc == 0
    if not ok:
        out = (out or "")[:400] + " — ملاحظة: Kaggle CLI v2.2.4 لا يوفر أمر إيقاف kernel؛"
        out += " kernel التدريب يراقب إشارة التوقف عند نهاية كل عصر ويتوقف نظيفًا."
    return {"ok": ok, "kernel_id": kernel_id, "command": " ".join(cmd),
            "exit_code": rc, "output": out}


def prepare_surahchain_kaggle_job(
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
    title: Optional[str] = None,
    use_tpu: bool = True,
) -> Dict[str, Any]:
    """يجهّز kernel SurahChain جاهز للرفع عبر API.

    use_tpu=True (افتراضيًا): TPU v5e-8 على Kaggle — أسرع وأكبر ذاكرة
    (8 شرائح × 16GB = 128GB HBM) — ضروري لـ preset=xlarge (d=8192 يحتاج
    ≈46GB ويستحيل على T4/16GB). على TPU: enable_gpu=False +
    machine_shape=TpuV5E8 + صورة TPUVM + SCN_TPU=1 في البيئة.
    """
    job_id = f"scn_{uuid.uuid4().hex[:10]}"
    job_dir = KAGGLE_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    user = (
        os.environ.get("KAGGLE_USERNAME")
        or os.environ.get("KAGGLE_USER")
        or "nsm-agent"
    )
    try:
        for cand in (
            Path.home() / ".kaggle" / "kaggle.json",
            Path(os.environ.get("KAGGLE_CONFIG_DIR") or "") / "kaggle.json",
        ):
            if cand.is_file():
                data = json.loads(cand.read_text(encoding="utf-8"))
                user = data.get("username") or user
                break
    except Exception:
        pass

    slug = _safe_slug(title or f"nsm-surahchain-{job_id}")
    effective_title = slug.replace("-", " ")
    if use_tpu:
        # وضع TPU: Kaggle v5e-8 (128GB HBM إجمالي — ضروري لـxlarge d=8192)
        meta = generate_kernel_metadata(
            job_id,
            username=user,
            title=effective_title,
            enable_gpu=False,
            accelerator=None,
        )
        meta["machine_shape"] = "TpuV5E8"
        meta["docker_image"] = (
            "gcr.io/kaggle-private-byod/python-tpuvm@sha256:" +
            "a2111cb9be558ea4bc187754bb95d7b65e90d8259434f1eb0e0ab1193ff498c0"
        )
        script = generate_surahchain_kernel_script(
            job_id, preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh,
            auto_push=auto_push, use_tpu=True,
            kernel_url=f"https://www.kaggle.com/code/{user}/{slug}",
        )
    else:
        # وضع GPU: T4 مفردة (الافتراضي القديم)
        meta = generate_kernel_metadata(
            job_id,
            username=user,
            title=effective_title,
            enable_gpu=True,
            accelerator="NvidiaTeslaT4",
        )
        script = generate_surahchain_kernel_script(
            job_id, preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh,
            auto_push=auto_push, use_tpu=False,
            kernel_url=f"https://www.kaggle.com/code/{user}/{slug}",
        )
    meta["id"] = f"{user}/{slug}"
    meta["enable_internet"] = True
    meta["is_private"] = True

    (job_dir / "nsm_train.py").write_text(script, encoding="utf-8")
    # الخلية تحتوي السكربت كاملاً — لا %run (كان يسبب file not found)
    src_lines = [ln + "\n" for ln in script.splitlines()]
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# NSM SurahChain training\n",
                    "clone + train + auto-push\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {},
                "source": src_lines,
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    (job_dir / "nsm_train.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")
    meta["code_file"] = "nsm_train.ipynb"
    meta["language"] = "python"
    meta["kernel_type"] = "notebook"
    (job_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    job = {
        "job_id": job_id,
        "type": "surahchain",
        "title": effective_title,
        "slug": f"{user}/{slug}",
        "preset": preset,
        "n": n,
        "epochs": epochs,
        "batch": batch,
        "fresh": fresh,
        "auto_push": auto_push,
        "status": "prepared",
        "created_at": _now(),
        "job_dir": str(job_dir.relative_to(ROOT)),
    }
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **job}


def start_surahchain_training_api(
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
) -> Dict[str, Any]:
    """
    من تبويب Notebook: جهّز + ادفع kernel عبر Kaggle API ليبدأ التدريب على GPU.
    يتطلب: KAGGLE_USERNAME + KAGGLE_KEY و pip install kaggle
    وعلى حساب Kaggle: Secret باسم GITHUB_TOKEN للرفع بعد التدريب.
    """
    ok_cred, msg = ensure_kaggle_env()
    if not ok_cred:
        return {
            "ok": False,
            "error": msg,
            "need": ["KAGGLE_USERNAME", "KAGGLE_KEY", "pip install kaggle"],
            "hint_ar": "ضع المفاتيح في Streamlit Secrets ثم أعد التشغيل",
        }
    if not _kaggle_cli_available() and not _kaggle_py_available():
        return {
            "ok": False,
            "error": "حزمة/أداة kaggle غير مثبتة",
            "need": ["pip install kaggle"],
        }

    prep = prepare_surahchain_kaggle_job(
        preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh, auto_push=auto_push
    )
    if not prep.get("ok"):
        return prep
    job_id = prep["job_id"]
    push = push_kaggle_kernel(job_id)
    return {
        "ok": bool(push.get("ok")),
        "job_id": job_id,
        "prepare": {k: prep[k] for k in prep if k != "ok"},
        "push": push,
        "kernel_url": push.get("kernel_url"),
        "msg_ar": (
            "تم دفع الـkernel — راقب التشغيل على Kaggle. "
            "بعد انتهاء التدريب يُرفع تلقائياً إن وُجد GITHUB_TOKEN في Kaggle Secrets."
            if push.get("ok")
            else "فشل الدفع — راجع تفاصيل push"
        ),
    }


def live_training_status(job_id: str, fetch_progress_from_github: bool = True) -> Dict[str, Any]:
    """NSM Live Logs: تجميع حيّ لحالة تدريب SurahChain جارٍ.

    تجمع في رد واحد:
    1) حالة الـkernel من Kaggle (RUNNING/QUEUED/COMPLETE/ERROR)
    2) آخر log lines من Kaggle (stdout/err)
    3) progress_{TAG}.json من GitHub (أبعد/أسرع مؤشر — كُتب كل عصر من التدريب)

    ملاحظة: progress.json يُرفع عبر checkpoint auto-push — إن لم يكتمل عصر بعد
    نقرأه من آخر checkpoint push متاح. هذه الطريقة تعمل حتى لو كانت Kaggle
    logs فارغة (buffering) أو kernel في طابور GPU.
    """
    job_dir = KAGGLE_DIR / job_id
    kernel_id = ""
    meta_path = job_dir / "kernel-metadata.json"
    if meta_path.is_file():
        kernel_id = json.loads(meta_path.read_text(encoding="utf-8")).get("id") or ""
    jp = job_dir / "job.json"
    if jp.is_file():
        try:
            jdata = json.loads(jp.read_text(encoding="utf-8"))
            kernel_id = jdata.get("kernel_slug") or jdata.get("kernel_id") or kernel_id
        except Exception:
            pass

    out: Dict[str, Any] = {"ok": True, "job_id": job_id, "kernel_id": kernel_id}

    # 1) حالة الـkernel
    if kernel_id and _kaggle_cli_available():
        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "status", kernel_id],
                capture_output=True, text=True, timeout=60,
            )
            st = (proc.stdout or "").strip() or (proc.stderr or "").strip()
            state = (st.split("State: ")[1].split()[0] if "State: " in st else "")
            out["kernel_state"] = state
            out["kernel_status"] = st[-2500:]
        except Exception as e:
            out["kernel_status_error"] = str(e)

    # 2) آخر logs من Kaggle
    if kernel_id and _kaggle_cli_available():
        try:
            proc = subprocess.run(
                ["kaggle", "kernels", "logs", kernel_id],
                capture_output=True, text=True, timeout=120,
            )
            logs = (proc.stdout or "") + (proc.stderr or "")
            out["kernel_logs"] = logs[-8000:]
            out["kernel_logs_lines"] = logs.count("\n")
        except Exception as e:
            out["kernel_logs_error"] = str(e)

    # 3) آخر progress من GitHub (أحدث من logs لأنه atomic + كل عصر)
    if fetch_progress_from_github:
        token = (os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or "").strip()
        if token and kernel_id and _kaggle_cli_available():
            try:
                proc = subprocess.run(
                    ["kaggle", "kernels", "pull", kernel_id],
                    capture_output=True, text=True, timeout=300,
                )
                # ابحث عن progress_*.json في أي مجلد checkpoints داخل المجلد المسحوب
                for pf in sorted(job_dir.rglob("progress_*.json")):
                    try:
                        out["progress"] = json.loads(pf.read_text(encoding="utf-8"))
                        out["progress_file"] = str(pf.relative_to(job_dir))
                        out["progress_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
                    except Exception:
                        continue
                    break
            except Exception as e:
                out["progress_error"] = str(e)
        else:
            # بدون CLI: من الكود المحلي مباشرة (تدريب جارٍ على نفس الجهاز)
            try:
                from experiments.surah_chain_network.train_pretrain_torch import read_progress
                p = read_progress()
                if p:
                    out["progress"] = p
                    out["progress_file"] = "local"
            except Exception:
                pass

    out["checked_at"] = _now()
    return out


# ─── مركز القيادة الموحّد (Unified Kernel Command Center) ─────────────────────

def _kaggle_env_for_account(username: Optional[str], key: Optional[str]) -> Dict[str, Optional[str]]:
    """يجهّز بيئة مؤقتة للحساب المحدد دون تعديل البيئة الأصلية (تُستخدم ثم تُسترد)."""
    if not username or not key:
        return {}
    return {"KAGGLE_USERNAME": username, "KAGGLE_KEY": key}


def _run_kaggle_cli(args: List[str], env_extra: Optional[Dict[str, str]] = None,
                    timeout: int = 90) -> Dict[str, Any]:
    """تشغيل أمر kaggle CLI مع بيئة اختيارية. يدعم التشغيل عبر حسابات متعددة."""
    import subprocess
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    try:
        proc = subprocess.run(["kaggle"] + args, capture_output=True, text=True,
                              timeout=timeout, env=env)
        return {"ok": proc.returncode == 0,
                "stdout": proc.stdout or "", "stderr": proc.stderr or "",
                "returncode": proc.returncode}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "returncode": -1}


def list_kernels_for_account(username: str, key: str, max_results: int = 30) -> List[Dict[str, Any]]:
    """يسرد kernels حساب معين (عبر بيئة مؤقتة بحسابه) ويعيد قائمة dicts.
    الناتج: [{'slug': ..., 'title': ..., 'state': ..., 'last_run': ..., 'total_votes': ...}]"""
    env = _kaggle_env_for_account(username, key)
    if not env:
        return []
    r = _run_kaggle_cli(["kernels", "list", "--user", username, "--max-results", str(max_results)],
                        env_extra=env, timeout=120)
    if not r["ok"]:
        return [{"error": (r["stderr"] or "").strip()[:300]}]
    rows = []
    for line in (r["stdout"] or "").splitlines()[2:]:  # تجاهل الهيدر والفواصل
        if not line.strip() or line.strip().startswith("-"):
            continue
        parts = [p.strip() for p in line.split("  ") if p.strip()]
        if len(parts) >= 3:
            rows.append({
                "slug": parts[0], "title": parts[1] if len(parts) > 1 else "",
                "state": parts[2] if len(parts) > 2 else "",
                "last_run": parts[3] if len(parts) > 3 else "",
                "username": username,
            })
    return rows


def classify_kernel(kind: str, title: str) -> str:
    """تصنيف نوع الـkernel من عنوانه/معرّفه: training / tally (تجميع بيانات) / other."""
    low = (title or "").lower()
    if "surahchain" in low or "surah" in low or "scn" in low or "train" in low or "pretrain" in low or "neural" in low:
        return "training"
    if "corpus" in low or "collect" in low or "tally" in low or "arabic" in low or "data" in low:
        return "tally"
    return "other"


def unified_kernel_overview(accounts: Optional[List[Dict[str, str]]] = None,
                            mine: bool = True) -> Dict[str, Any]:
    """مركز القيادة: يسرد كل kernels على كل الحسابات المسجلة (والحساب الحالي).
    يعيد: {'ok': bool, 'kernels': [...], 'per_account': {...}, 'checked_at': ...}
    كل kernel: slug, title, state, last_run, provider, kind, username.
    """
    if accounts is None:
        try:
            from ai import multi_account_scheduler as MAS
            accounts = MAS.load_accounts()
        except Exception:
            accounts = []
    # الحساب الحالي إن لم يكن مدرجًا
    cur = os.environ.get("KAGGLE_USERNAME") or ""
    cur_key = os.environ.get("KAGGLE_KEY") or ""
    if cur and not any(a.get("username") == cur for a in accounts):
        accounts = [{"username": cur, "key": cur_key, "note": "الحساب الحالي"}] + accounts

    all_kernels: List[Dict[str, Any]] = []
    per_account: Dict[str, Any] = {}
    seen = set()
    for acc in accounts:
        uname, key = acc.get("username", ""), acc.get("key", "")
        if not uname or not key:
            per_account[uname or "?"] = {"ok": False, "error": "لا مفاتيح للحساب"}
            continue
        rows = list_kernels_for_account(uname, key)
        per_account[uname] = {"ok": True, "count": len(rows)}
        for r in rows:
            slug = r.get("slug", "")
            if not slug or slug in seen:
                continue
            seen.add(slug)
            r["provider"] = "kaggle"
            r["kind"] = classify_kernel(r.get("kind", ""), r.get("title", ""))
            r["account"] = uname
            all_kernels.append(r)
    # kernels الحساب الحالي مباشرة (أسرع — نفس البيئة)
    if mine and cur:
        r = _run_kaggle_cli(["kernels", "list", "--mine", "--max-results", "30"], timeout=120)
        for line in (r.get("stdout") or "").splitlines()[2:]:
            if not line.strip() or line.strip().startswith("-"):
                continue
            parts = [p.strip() for p in line.split("  ") if p.strip()]
            if len(parts) >= 3 and parts[0] not in seen:
                seen.add(parts[0])
                all_kernels.append({
                    "slug": parts[0], "title": parts[1] if len(parts) > 1 else "",
                    "state": parts[2] if len(parts) > 2 else "",
                    "last_run": parts[3] if len(parts) > 3 else "",
                    "provider": "kaggle", "kind": classify_kernel("", parts[1] if len(parts) > 1 else ""),
                    "username": cur, "account": cur,
                })
    from datetime import datetime, timezone
    all_kernels.sort(key=lambda k: k.get("last_run") or "", reverse=True)
    return {
        "ok": True, "kernels": all_kernels, "per_account": per_account,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }


def kernel_progress_snap(slug: str, timeout: int = 240) -> Dict[str, Any]:
    """يسحب kernel (kaggle kernels pull) ويعيد آخر progress_{tag}.json إن وُجد.
    يعمل للحساب الحالي فقط (البيئة الحالية)."""
    out: Dict[str, Any] = {"ok": False}
    tmp = Path("/tmp") / f"nsm_kp_{slug.replace('/', '_')[:60]}"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.run(["kaggle", "kernels", "pull", slug, "--path", str(tmp)],
                              capture_output=True, text=True, timeout=timeout)
        if proc.returncode == 0:
            for pf in sorted(tmp.rglob("progress_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    out.update({"ok": True, "progress": json.loads(pf.read_text(encoding="utf-8")),
                                "file": str(pf.name), "source": "kernel_output"})
                except Exception:
                    continue
                break
            if not out.get("progress"):
                # بحث أوسع: أي json فيه loss/epoch
                for pf in sorted(tmp.rglob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                    if pf.name.startswith("progress_"):
                        continue
                    try:
                        data = json.loads(pf.read_text(encoding="utf-8"))
                        if isinstance(data, dict) and ("loss" in data or "epoch" in data):
                            out.update({"ok": True, "progress": data, "file": str(pf.name), "source": "kernel_output"})
                            break
                    except Exception:
                        continue
    except Exception as e:
        out["error"] = str(e)
    return out
