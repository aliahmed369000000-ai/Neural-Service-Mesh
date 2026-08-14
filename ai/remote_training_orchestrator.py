"""
Remote Training Orchestrator — تحكم موحّد بمنصات التدريب البعيد
================================================================
يدير دورة حياة تدريب الشبكات العصبية عبر:

  • Kaggle  — API كامل (تجهيز Kernel / دفع GPU / مراقبة / تنزيل)
  • Google Colab — داخل الدفتر + جسر webhook (بدون أتمتة متصفح)
  • Local GPU — نفس الجهاز (Colab runtime أو سيرفرك)

الوكيل لا يفتح متصفحاً ولا يخالف شروط Colab؛ التحكم البرمجي فقط.
"""
from __future__ import annotations

import json
import logging
import os
import textwrap
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("RemoteTrainingOrchestrator")

ROOT = Path(__file__).resolve().parent.parent
ORCH_DIR = ROOT / "artifacts" / "model_training" / "orchestrator"
ORCH_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ═══════════════════════════════════════════════════════════════════════════
# 1) كود تدريب فعّال (يُحقن في Kaggle / Colab)
# ═══════════════════════════════════════════════════════════════════════════

def efficient_nn_training_source(
    job_id: str,
    epochs: int = 20,
    use_amp: bool = True,
    prefer_multi_gpu: bool = True,
) -> str:
    """
    سكربت تدريب شبكة كثيفة بكفاءة:
      - اكتشاف GPU / Dual GPU + DataParallel
      - Mixed Precision (AMP) عند توفر CUDA
      - batch size تكيفي حسب VRAM
      - AdamW + cosine annealing
      - early stopping بسيط
      - حفظ أفضل أوزان + تقرير JSON
    """
    return textwrap.dedent(
        f'''
        #!/usr/bin/env python3
        """NSM Efficient NN Training — job {job_id}"""
        from __future__ import annotations
        import json, math, os, time, traceback
        from pathlib import Path
        from datetime import datetime, timezone

        def _now():
            return datetime.now(timezone.utc).isoformat()

        # مجلد العمل: Kaggle أو Colab أو محلي
        if Path("/kaggle/working").exists():
            WORK = Path("/kaggle/working")
        elif Path("/content").exists():
            WORK = Path("/content/nsm_train_out")
            WORK.mkdir(parents=True, exist_ok=True)
        else:
            WORK = Path("artifacts/model_training/orchestrator") / "{job_id}"
            WORK.mkdir(parents=True, exist_ok=True)

        REPORT = WORK / "nsm_efficient_report.json"
        report = {{
            "job_id": "{job_id}",
            "started_at": _now(),
            "ok": False,
            "platform_hints": {{
                "kaggle": Path("/kaggle").exists(),
                "colab": Path("/content").exists() or bool(os.environ.get("COLAB_GPU") is not None or os.environ.get("COLAB_RELEASE_TAG")),
            }},
            "gpu": {{}},
            "metrics": {{}},
            "artifacts": [],
        }}

        try:
            import numpy as np
            import torch
            import torch.nn as nn
            import torch.optim as optim
            from torch.cuda.amp import autocast, GradScaler

            # ── GPU discovery ──────────────────────────────────────────
            cuda = torch.cuda.is_available()
            n_gpu = torch.cuda.device_count() if cuda else 0
            names = [torch.cuda.get_device_name(i) for i in range(n_gpu)] if n_gpu else []
            vram = []
            for i in range(n_gpu):
                try:
                    props = torch.cuda.get_device_properties(i)
                    vram.append(round(props.total_memory / (1024**3), 2))
                except Exception:
                    vram.append(None)
            report["gpu"] = {{
                "cuda": cuda,
                "device_count": n_gpu,
                "names": names,
                "vram_gb": vram,
                "is_dual_t4": n_gpu >= 2 and all("T4" in (n or "") for n in names[:2]),
            }}
            print("GPU:", report["gpu"])

            try:
                import subprocess
                print(subprocess.check_output(["nvidia-smi", "-L"], text=True, timeout=8))
            except Exception as e:
                print("nvidia-smi:", e)

            # ── بيانات (CSV إن وُجد وإلا اصطناعية) ─────────────────────
            d_in, n_samples, n_classes = 32, 4000, 2
            X = y = None
            csv_paths = []
            for root in (Path("/kaggle/input"), Path("/content"), Path("data/samples"), Path(".")):
                if root.exists():
                    csv_paths.extend(root.rglob("*.csv"))
            if csv_paths:
                try:
                    import pandas as pd
                    df = pd.read_csv(csv_paths[0])
                    num = df.select_dtypes(include=["number"])
                    if num.shape[1] >= 2:
                        y_col = num.columns[-1]
                        X = num.drop(columns=[y_col]).values.astype("float32")
                        y_raw = num[y_col].values
                        if y_raw.dtype.kind in "fc":
                            med = float(np.median(y_raw))
                            y = (y_raw >= med).astype("int64")
                        else:
                            uniq = {{v: i for i, v in enumerate(sorted(set(y_raw.tolist())))}}
                            y = np.array([uniq[v] for v in y_raw], dtype="int64")
                            n_classes = max(2, len(uniq))
                        d_in = X.shape[1]
                        n_samples = X.shape[0]
                        report["data_source"] = str(csv_paths[0])
                        print("CSV", csv_paths[0], X.shape)
                except Exception as e:
                    print("CSV fallback:", e)
            if X is None:
                rng = np.random.default_rng(42)
                X = rng.normal(size=(n_samples, d_in)).astype("float32")
                y = (X[:, 0] + 0.4 * X[:, 1] > 0).astype("int64")
                report["data_source"] = "synthetic"

            # تطبيع
            mu, sigma = X.mean(0), X.std(0) + 1e-6
            X = (X - mu) / sigma

            # ── نموذج ──────────────────────────────────────────────────
            class EfficientMLP(nn.Module):
                def __init__(self, d_in, n_classes, width=128):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(d_in, width),
                        nn.GELU(),
                        nn.BatchNorm1d(width),
                        nn.Dropout(0.1),
                        nn.Linear(width, width),
                        nn.GELU(),
                        nn.BatchNorm1d(width),
                        nn.Dropout(0.1),
                        nn.Linear(width, n_classes),
                    )
                def forward(self, x):
                    return self.net(x)

            model = EfficientMLP(d_in, n_classes)
            mode = "cpu"
            if cuda:
                model = model.cuda()
                if {str(prefer_multi_gpu)} and n_gpu >= 2:
                    model = nn.DataParallel(model)
                    mode = f"DataParallel×{{n_gpu}}"
                else:
                    mode = "single-gpu"

            # batch حسب VRAM
            if cuda and vram and vram[0]:
                bs = 256 if vram[0] >= 14 else (128 if vram[0] >= 7 else 64)
            else:
                bs = 64
            if n_samples < bs:
                bs = max(8, n_samples // 2)

            X_t = torch.from_numpy(X)
            y_t = torch.from_numpy(y)
            # split
            n_val = max(1, int(0.15 * n_samples))
            perm = torch.randperm(n_samples)
            val_idx, tr_idx = perm[:n_val], perm[n_val:]

            opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            epochs = {epochs}
            sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))
            loss_fn = nn.CrossEntropyLoss()
            use_amp = {str(use_amp)} and cuda
            scaler = GradScaler(enabled=use_amp)

            best_val = float("inf")
            best_state = None
            patience, bad = 5, 0
            hist = []
            t0 = time.time()
            model.train()

            for ep in range(epochs):
                model.train()
                order = tr_idx[torch.randperm(len(tr_idx))]
                total, steps = 0.0, 0
                for i in range(0, len(order), bs):
                    idx = order[i:i+bs]
                    xb, yb = X_t[idx], y_t[idx]
                    if cuda:
                        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    with autocast(enabled=use_amp):
                        logits = model(xb)
                        loss = loss_fn(logits, yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    total += float(loss.item())
                    steps += 1
                sched.step()
                train_loss = total / max(steps, 1)

                # val
                model.eval()
                with torch.no_grad():
                    xb = X_t[val_idx]
                    yb = y_t[val_idx]
                    if cuda:
                        xb, yb = xb.cuda(), yb.cuda()
                    with autocast(enabled=use_amp):
                        vloss = float(loss_fn(model(xb), yb).item())
                        pred = model(xb).argmax(-1)
                        acc = float((pred == yb).float().mean().item())
                hist.append({{"ep": ep+1, "train": round(train_loss, 5), "val": round(vloss, 5), "acc": round(acc, 4)}})
                if ep % max(1, epochs // 5) == 0 or ep == epochs - 1:
                    print(f"epoch {{ep+1}}/{{epochs}} train={{train_loss:.4f}} val={{vloss:.4f}} acc={{acc:.3f}} mode={{mode}} amp={{use_amp}}")

                if vloss < best_val - 1e-4:
                    best_val = vloss
                    bad = 0
                    state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
                    best_state = {{k: v.detach().cpu().clone() for k, v in state.items()}}
                else:
                    bad += 1
                    if bad >= patience:
                        print("early stop at", ep+1)
                        break

            out_pt = WORK / "nsm_efficient_model.pt"
            torch.save({{
                "state_dict": best_state,
                "d_in": d_in,
                "n_classes": n_classes,
                "mu": mu.tolist(),
                "sigma": sigma.tolist(),
                "job_id": "{job_id}",
                "mode": mode,
            }}, out_pt)

            report["artifacts"].append(out_pt.name)
            report["metrics"] = {{
                "epochs_ran": len(hist),
                "best_val_loss": best_val,
                "final": hist[-1] if hist else None,
                "history_tail": hist[-15:],
                "n_samples": int(n_samples),
                "d_in": int(d_in),
                "batch_size": bs,
                "mode": mode,
                "amp": use_amp,
                "elapsed_s": round(time.time() - t0, 2),
            }}
            report["ok"] = True
            report["finished_at"] = _now()
            print("DONE", report["metrics"])
        except Exception as e:
            report["ok"] = False
            report["error"] = str(e)
            report["traceback"] = traceback.format_exc()[:4000]
            report["finished_at"] = _now()
            print("FAIL", e)

        REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("report →", REPORT)
        '''
    ).strip()


def colab_mission_cells(job_id: str, epochs: int = 20, webhook_url: str = "") -> str:
    """نص خلايا جاهزة للصق في Google Colab."""
    webhook = webhook_url or os.environ.get("NSM_REMOTE_WEBHOOK_URL") or ""
    secret = os.environ.get("NSM_REMOTE_WEBHOOK_SECRET") or ""
    train_src = efficient_nn_training_source(job_id, epochs=epochs)
    return textwrap.dedent(
        f'''
        # ═══ NSM Colab Mission — {job_id} ═══
        # 1) Runtime → Change runtime type → GPU (T4)
        # 2) نفّذ الخلايا بالترتيب

        # --- خلية 1: استنساخ وتهيئة ---
        !git clone -q https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git || true
        %cd Neural-Service-Mesh
        %run scripts/colab_bootstrap.py

        # --- خلية 2: تدريب فعّال ---
        import os
        os.environ["NSM_ALLOW_GPU"] = "1"
        code = r"""{train_src}"""
        exec(compile(code, "nsm_efficient_train.py", "exec"))

        # --- خلية 3 (اختياري): دفع النتائج لخادمك ---
        # os.environ["NSM_REMOTE_WEBHOOK_URL"] = "{webhook}"
        # os.environ["NSM_REMOTE_WEBHOOK_SECRET"] = "{secret}"
        # %run scripts/colab_result_push.py --csv data/samples/classification_demo.csv --epochs {epochs}
        '''
    ).strip()


# ═══════════════════════════════════════════════════════════════════════════
# 2) حالة المنصات
# ═══════════════════════════════════════════════════════════════════════════

def platforms_status() -> str:
    lines = ["## 🌐 حالة منصات التدريب البعيد", ""]

    # Kaggle
    try:
        from ai.kaggle_provider import credentials_status, list_kaggle_jobs, detect_kaggle_gpus

        kc = credentials_status()
        jobs = list_kaggle_jobs()
        lines += [
            "### 🟧 Kaggle",
            f"- API جاهز: **{'✅' if kc.get('ready') else '❌'}** | CLI: {'✅' if kc.get('cli') else '❌'}",
            f"- مهام محلية: **{len(jobs)}**",
            f"- أوامر: `درّب kaggle` · `ادفع kaggle <id>` · `حالة kaggle <id>` · `حمّل kaggle <id>`",
            "",
        ]
    except Exception as e:
        lines += ["### 🟧 Kaggle", f"- خطأ تحميل: {e}", ""]

    # Colab / Remote webhook
    webhook = os.environ.get("NSM_REMOTE_WEBHOOK_URL") or ""
    lines += [
        "### 🟦 Google Colab",
        f"- Webhook مضبوط: **{'✅' if webhook else '❌'}** (`NSM_REMOTE_WEBHOOK_URL`)",
        "- التحكم: داخل الدفتر (bootstrap) + دفع نتائج عبر webhook — **بدون أتمتة متصفح**",
        "- أوامر: `مهمة colab` · `خلايا colab` · `حالة remote gpu`",
        "",
    ]

    # Local
    try:
        from ai.gpu_runtime import detect_device, device_report_md

        d = detect_device(force_gpu=os.environ.get("NSM_ALLOW_GPU") == "1")
        lines += [
            "### 💻 الجهاز الحالي",
            f"- device: `{d.device_str}` | cuda_available={d.cuda} | {d.name or '—'}",
            f"- السبب: {d.reason}",
            "",
        ]
    except Exception as e:
        lines += ["### 💻 الجهاز الحالي", f"- {e}", ""]

    lines += [
        "### كفاءة التدريب",
        "- AMP (mixed precision) + DataParallel عند تعدد GPU",
        "- AdamW + CosineAnnealing + early stopping",
        "- batch تكيفي حسب VRAM",
        "",
        "أمر موحّد: `درّب بعيد kaggle` أو `درّب بعيد colab` أو `حالة المنصات`",
    ]
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# 3) إرسال مهام
# ═══════════════════════════════════════════════════════════════════════════

def dispatch_kaggle_efficient(
    epochs: int = 20,
    csv_path: Optional[str] = None,
    push: bool = False,
    accelerator: str = "NvidiaTeslaT4",
) -> Dict[str, Any]:
    """يولّد مهمة Kaggle بكود التدريب الفعّال، ويدفع اختيارياً."""
    from ai.kaggle_provider import (
        prepare_kaggle_job,
        push_kaggle_kernel,
        generate_kernel_metadata,
        _safe_slug,
        KAGGLE_DIR,
        ensure_kaggle_env,
        credentials_status,
    )

    # استخدم prepare ثم استبدل السكربت/الدفتر بالنسخة الفعّالة
    job = prepare_kaggle_job(
        csv_path=csv_path or "data/samples/classification_demo.csv",
        epochs=epochs,
        title="nsm efficient nn train",
        accelerator=accelerator,
        enable_gpu=True,
    )
    job_id = job["job_id"]
    job_dir = ROOT / job["job_dir"]
    src = efficient_nn_training_source(job_id, epochs=epochs)

    (job_dir / "nsm_train.py").write_text(src, encoding="utf-8")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True},
        },
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {"id": "nsm_train"},
                "outputs": [],
                "source": src,
            }
        ],
    }
    (job_dir / "nsm_train.ipynb").write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

    meta_path = job_dir / "kernel-metadata.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["code_file"] = "nsm_train.ipynb"
        meta["kernel_type"] = "notebook"
        meta["enable_gpu"] = True
        meta["machine_shape"] = accelerator
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    job["training_mode"] = "efficient_nn"
    job["epochs"] = epochs
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

    if push:
        push_res = push_kaggle_kernel(job_id)
        job["push"] = push_res
        job["status"] = push_res.get("status", job.get("status"))
    return job


def prepare_colab_mission(epochs: int = 20) -> Dict[str, Any]:
    """يحفظ مهمة Colab محلياً (خلايا + سكربت) للتشغيل اليدوي داخل الدفتر."""
    job_id = f"colab_{uuid.uuid4().hex[:10]}"
    job_dir = ORCH_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    src = efficient_nn_training_source(job_id, epochs=epochs)
    cells = colab_mission_cells(job_id, epochs=epochs)
    (job_dir / "efficient_train.py").write_text(src, encoding="utf-8")
    (job_dir / "colab_cells.md").write_text(cells, encoding="utf-8")
    job = {
        "ok": True,
        "job_id": job_id,
        "provider": "colab_mission",
        "status": "prepared",
        "job_dir": str(job_dir.relative_to(ROOT)),
        "epochs": epochs,
        "created_at": _now(),
        "instructions": [
            "افتح Google Colab وفعّل GPU (T4)",
            f"الصق محتوى artifacts/model_training/orchestrator/{job_id}/colab_cells.md",
            "أو: clone المستودع ثم %run scripts/colab_bootstrap.py ثم نفّذ efficient_train.py",
            "اختياري: اضبط NSM_REMOTE_WEBHOOK_URL لدفع النتائج",
        ],
    }
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return job


# ═══════════════════════════════════════════════════════════════════════════
# 4) موجّه الأوامر
# ═══════════════════════════════════════════════════════════════════════════

def handle_orchestrator_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None

    # حالة المنصات
    if re.search(
        r"(حالة|status).{0,12}(المنصات|platforms|remote\s*platforms)|حالة\s*المنصات|platforms\s*status",
        text,
        re.I,
    ) or text.lower() in ("حالة المنصات", "platforms"):
        return platforms_status()

    # درّب بعيد kaggle [وادفع]
    m_k = re.search(
        r"(?<![اأ])(?:در[ّ]?ب|train)\s*(?:بعيد|remote)?\s*kaggle|(?<![اأ])(?:در[ّ]?ب|train)\s*kaggle\s*(?:فع[ّ]?ال|efficient)",
        text,
        re.I,
    )
    if m_k or re.search(r"efficient\s*kaggle|kaggle\s*efficient", text, re.I):
        epochs = 20
        m_ep = re.search(r"(?:epochs?|حقب|عصور)\s*[=:]?\s*(\d+)", text, re.I)
        if m_ep:
            epochs = max(1, min(100, int(m_ep.group(1))))
        do_push = bool(re.search(r"وادفع|وادفعه|and\s*push|push\s*now|ادفع", text, re.I))
        csv_path = None
        m_csv = re.search(r"((?:data|artifacts)[\w./\-]+\.csv)", text, re.I)
        if m_csv:
            csv_path = m_csv.group(1)
        job = dispatch_kaggle_efficient(epochs=epochs, csv_path=csv_path, push=do_push)
        return (
            "## 🚀 تدريب فعّال على Kaggle\n\n"
            + "```json\n"
            + json.dumps({k: v for k, v in job.items() if k not in ("next_steps",)}, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            + (
                "_تم الدفع — راقب بـ `حالة kaggle " + job["job_id"] + "`_"
                if do_push
                else f"_للدفع: `ادفع kaggle {job['job_id']}`_"
            )
        )

    # مهمة / خلايا colab
    if re.search(r"(مهمة|cells?|خلايا|mission)\s*colab|colab\s*(مهمة|mission|cells?)|جه[ّ]?ز\s*colab", text, re.I):
        epochs = 20
        m_ep = re.search(r"(?:epochs?|حقب|عصور)\s*[=:]?\s*(\d+)", text, re.I)
        if m_ep:
            epochs = max(1, min(100, int(m_ep.group(1))))
        job = prepare_colab_mission(epochs=epochs)
        cells_path = Path(job["job_dir"]) / "colab_cells.md"
        cells_preview = ""
        full = ROOT / cells_path
        if full.is_file():
            cells_preview = full.read_text(encoding="utf-8")[:2500]
        return (
            "## 🟦 مهمة Google Colab جاهزة\n\n"
            + "```json\n"
            + json.dumps(job, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            + "### معاينة الخلايا\n```python\n"
            + cells_preview
            + "\n```"
        )

    # درّب بعيد colab
    if re.search(r"(?:در[ّ]?ب|train)\s*(?:بعيد|remote)?\s*colab", text, re.I):
        job = prepare_colab_mission(epochs=20)
        return (
            "## 🟦 تدريب بعيد عبر Colab\n\n"
            "لا يمكن تشغيل Colab من API رسمي مثل Kaggle بدون متصفح.\n"
            "الوكيل جهّز **مهمة كاملة** للصق داخل الدفتر:\n\n"
            + "```json\n"
            + json.dumps(job, ensure_ascii=False, indent=2)
            + "\n```\n\n"
            "1. Colab → Runtime → GPU (T4)\n"
            f"2. افتح `{job['job_dir']}/colab_cells.md` والصق الخلايا\n"
            "3. اختياري: Webhook لاستلام الأوزان على سيرفرك"
        )

    # خطة كفاءة
    if re.search(r"(خطة|plan).{0,10}(كفاءة|efficient|تدريب\s*فع[ّ]?ال)", text, re.I):
        return textwrap.dedent(
            """
            ## ⚡ خطة تدريب شبكات بكفاءة (NSM)

            | التقنية | الفائدة |
            |---------|---------|
            | GPU T4 / Dual T4 | تسريع 10×–50× عن CPU |
            | AMP (fp16) | توفير VRAM + سرعة |
            | DataParallel | استغلال كرتين على Kaggle |
            | AdamW + Cosine | استقرار أفضل من SGD الثابت |
            | Early stopping | وقف الهدر عند توقف التحسن |
            | Batch تكيفي | تفادي OOM |

            **Kaggle (تحكم API كامل):**
            `درّب بعيد kaggle وادفع epochs 30`

            **Colab (داخل الدفتر + webhook):**
            `مهمة colab` ثم نفّذ الخلايا بعد تفعيل GPU

            **محلي:**
            `درّب شبكة torch` أو `حالة gpu`
            """
        ).strip()

    return None
