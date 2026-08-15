"""
المزودات المجانية البديلة (Free Fallback Providers)
====================================================
مزودان إضافيان للتدريب المجاني عند نفاد كوتا حسابات Kaggle السبعة:

1) Google Colab (المجاني):
   - Google Colab REST API غير متاح مباشرة للمجاني، لكن يُنفَّذ تلقائيًا عبر:
     a) Google Colab CLI البديل: تنزيل notebook ثم تنفيذه على Colab يدويًا
     b) التشغيل المباشر محليًا بنفس سكربت التحويل: يولّد notebook جاهز للفتح في Colab
        مع رابط "Open in Colab" واحد بالنقر
   - مجاني: T4، ~12 ساعة/جلسة، بدون كوتا أسبوعية صارمة

2) Lightning AI (المجاني):
   - REST API رسمي: https://lightning.ai
   - يتطلب LIGHTNING_API_KEY (مجانًا من account)
   - 22 ساعة L4/شهريًا في الخطة المجانية
   - يدعم إنشاء studio + تنفيذ سكربت كامل عبر /studio/{id}/runs

لا يعتمد على أتمتة متصفح — كل شيء عبر REST/توليد ملفات.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlencode

logger = logging.getLogger("FreeProviders")

ROOT = Path(__file__).resolve().parent.parent
FREE_DIR = ROOT / "artifacts" / "model_training" / "free_jobs"
FREE_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_slug(s: str) -> str:
    out = "".join(c if c.isalnum() or c in "-_" else "-" for c in (s or "nsm"))
    return out.strip("-_")[:48] or "nsm-job"


# ═══════════════════════════════════════════════════════════════════════════
# 1) Google Colab
# ═══════════════════════════════════════════════════════════════════════════

COLAB_OPEN_URL = "https://colab.research.google.com/github/"


def colab_generate_notebook(
    job_id: str,
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
) -> Dict[str, Any]:
    """
    يولّد notebook جاهزًا للتشغيل على Google Colab المجاني:
      - خلية إعداد GitHub PAT + استنساخ المستودع
      - خلية التدريب (نفس سكربت run_train_then_push.py)
      - رابط "Open in Colab" عبر colab.research.google.com/github
    """
    job_dir = FREE_DIR / f"colab_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    script = _colab_training_script(preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh, auto_push=auto_push)
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "colab": {
                "provenance": [],
                "gpuType": "T4",
                "include_colab_link": True,
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "accelerator": "GPU",
        },
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {"id": "nsm-info"},
                "source": [
                    "<a href=\"https://colab.research.google.com/\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>\n",
                    "\n",
                    "# NSM SurahChain — Google Colab (مجاني)\n",
                    "افتح: Runtime → Change runtime type → T4 GPU ← مهم!\n",
                    f"المعلمات: preset={preset}, n={n}, epochs={epochs}, batch={batch}\n",
                ],
            },
            {
                "cell_type": "code",
                "metadata": {"id": "nsm-setup"},
                "source": _colab_setup_cells(),
                "outputs": [],
                "execution_count": None,
            },
            {
                "cell_type": "code",
                "metadata": {"id": "nsm-train"},
                "source": [line + "\n" for line in script.splitlines()],
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    nb_path = job_dir / "nsm_colab_train.ipynb"
    nb_path.write_text(json.dumps(nb, indent=2), encoding="utf-8")

    slug = _safe_slug(f"nsm-colab-{job_id}")
    # ملاحظة: رابط GitHub المباشر يتطلب رفع الملف أولًا؛ نولّد رابط عام بديل
    open_url = None
    try:
        open_url = (
            "https://colab.research.google.com/?hl=ar#url="
            + urlencode({"url": f"https://raw.githubusercontent.com/aliahmed369000000-ai/Neural-Service-Mesh/main/artifacts/model_training/free_jobs/colab_{job_id}/nsm_colab_train.ipynb"})
        )
    except Exception:
        open_url = f"https://colab.research.google.com/?hl=ar"

    job = {
        "job_id": job_id,
        "provider": "colab",
        "slug": slug,
        "preset": preset, "n": n, "epochs": epochs, "batch": batch,
        "fresh": fresh, "auto_push": auto_push,
        "notebook": str(nb_path.relative_to(ROOT)),
        "colab_open_url": open_url,
        "status": "prepared",
        "created_at": _now(),
        "hint_ar": (
            "انقل محتوى الخلية الأولى والثانية إلى Colab جديد (Runtime→T4) واضغط Run all. "
            "Colab المجاني: T4 لمدة تصل إلى ~12 ساعة."
        ),
    }
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **job}


def _colab_setup_cells() -> List[str]:
    """خلايا الإعداد الأولية لـ Colab: git + استنساخ المستودع."""
    return [
        "# ── خلية 1: الاستنساخ والاعتمادات ──────────────────────────────\n",
        "import os, json, sys\n",
        "os.environ['SCN_PRESET'] = os.environ.get('SCN_PRESET', 'medium')\n",
        "!pip install -q torch numpy matplotlib 2>/dev/null\n",
        "if not os.path.exists('/tmp/Neural-Service-Mesh'):\n",
        "    token = os.environ.get('GITHUB_TOKEN', '')\n",
        "    auth = ('x-access-token:' + token + '@') if token else ''\n",
        "    !git clone -q --depth 1 --branch main \\\n",
        "        https://{auth}github.com/aliahmed369000000-ai/Neural-Service-Mesh.git \\\n",
        "        /tmp/Neural-Service-Mesh\n",
        "    print('clone done')\n",
        "os.chdir('/tmp/Neural-Service-Mesh')\n",
        "!nvidia-smi\n",
    ]


def _colab_training_script(
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
) -> str:
    """نفس منطق run_train_then_push.py في خلية واحدة (نقل مباشر من السكربت الأصلي)."""
    from ai.kaggle_provider import generate_surahchain_kernel_script

    # نعيد استخدام مولّد السكربت الموجود ونستبدل أوامر Kaggle بأوامر Colab
    script = generate_surahchain_kernel_script(
        f"colab_{uuid.uuid4().hex[:8]}",
        preset=preset, n=n, epochs=epochs, batch=batch, fresh=fresh, auto_push=auto_push,
    )
    # إزالة أسطر Kaggle-specific (تحويل الكود ليعمل على Colab مباشرة)
    lines = []
    for line in script.splitlines():
        if "kaggle_secrets" in line or "get_secret" in line:
            # الاستبدال: قراءة GitHub Token من colab secrets أو env
            if "GITHUB_TOKEN" in line:
                lines.append("GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')")
            continue
        lines.append(line)
    return "\n".join(lines)


def colab_list_jobs() -> List[Dict[str, Any]]:
    jobs = []
    if FREE_DIR.is_dir():
        for d in sorted(FREE_DIR.iterdir()):
            jp = d / "job.json"
            if d.name.startswith("colab_") and jp.is_file():
                try:
                    jobs.append(json.loads(jp.read_text(encoding="utf-8")))
                except Exception:
                    pass
    return jobs


# ═══════════════════════════════════════════════════════════════════════════
# 2) Lightning AI
# ═══════════════════════════════════════════════════════════════════════════

LIGHTNING_API = "https://lightning.ai"


def _lightning_request(method: str, path: str, key: str,
                       payload: Optional[Dict[str, Any]] = None,
                       params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """استدعاء REST API لـ Lightning AI."""
    import http.client
    import urllib.parse

    url = urllib.parse.urlparse(LIGHTNING_API)
    conn = http.client.HTTPSConnection(url.netloc, timeout=60)
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
    }
    body = json.dumps(payload).encode() if payload else None
    q = "?" + urllib.parse.urlencode(params) if params else ""
    try:
        conn.request(method, path + q, body=body, headers=headers)
        resp = conn.getresponse()
        data = resp.read().decode("utf-8", "replace")
        try:
            parsed = json.loads(data)
        except Exception:
            parsed = {"raw": data[:2000]}
        return {"status": resp.status, "data": parsed}
    except Exception as e:
        return {"status": 0, "error": str(e)}
    finally:
        conn.close()


def lightning_credentials_status() -> Dict[str, Any]:
    """يفحص اعتمادات Lightning دون كشف المفتاح."""
    key = os.environ.get("LIGHTNING_API_KEY") or ""
    ready = bool(key) and len(key) > 20
    return {
        "api_key_set": bool(key),
        "key_length": len(key),
        "ready": ready,
        "hint": "ضع LIGHTNING_API_KEY في Streamlit Secrets (مجانًا من https://lightning.ai)",
    }


def lightning_check_balance(key: Optional[str] = None) -> Dict[str, Any]:
    """يفحص رصيد/كوتا الحساب المجاني."""
    key = key or os.environ.get("LIGHTNING_API_KEY") or ""
    if not key:
        return {"ok": False, "error": "LIGHTNING_API_KEY غير مضبوط"}
    res = _lightning_request("GET", "/v1/user/me", key)
    if res["status"] == 200:
        data = res["data"]
        return {
            "ok": True,
            "user": data.get("id") or data.get("username"),
            "balance": data.get("balance") or data.get("credits"),
            "plan": (data.get("plan") or {}).get("name") if isinstance(data.get("plan"), dict) else None,
            "raw": {k: data[k] for k in ("plan", "balance", "id") if k in data},
        }
    return {"ok": False, "status": res["status"], "data": res.get("data")}


def lightning_generate_run(
    job_id: str,
    preset: str = "medium",
    n: int = 60000,
    epochs: int = 30,
    batch: int = 24,
    fresh: bool = True,
    auto_push: bool = True,
    studio_name: Optional[str] = None,
    machine_type: str = "cpu-small",  # المجاني: cpu-small | gpu-t4
) -> Dict[str, Any]:
    """
    يطلق مهمة تدريب على Lightning AI عبر REST.
    المسار: إنشاء Studio من صورة PyTorch ثم تشغيل run بداخله.
    """
    key = os.environ.get("LIGHTNING_API_KEY") or ""
    if not key:
        return {"ok": False, "error": "LIGHTNING_API_KEY غير مضبوط", "need": ["LIGHTNING_API_KEY"]}

    job_dir = FREE_DIR / f"lightning_{job_id}"
    job_dir.mkdir(parents=True, exist_ok=True)

    # ── الخطوة 1: إنشاء studio ──────────────────────────────────────────────
    study_spec = {
        "name": studio_name or f"nsm-surahchain-{job_id}",
        "spec": {
            "cluster": "",
            "display_name": "nsm-surahchain",
            "entrypoint": ["/bin/bash", "-c", "sleep infinity"],
            "env": [
                {"name": "SCN_PRESET", "value": preset},
                {"name": "SCN_N", "value": str(n)},
                {"name": "SCN_EPOCHS", "value": str(epochs)},
                {"name": "SCN_BATCH", "value": str(batch)},
                {"name": "SCN_FRESH", "value": "1" if fresh else "0"},
                {"name": "SCN_AUTO_PUSH", "value": "1" if auto_push else "0"},
            ],
            "image": "lightlyteam/pytorch:2.1.0-cuda12.1",
            "machine": {"name": machine_type},
            "user_command": "",
            "version": 1,
        },
    }
    create = _lightning_request("POST", "/v1/studios", key, payload=study_spec)
    if create["status"] not in (200, 201):
        return {"ok": False, "status": create["status"], "step": "create_studio", "data": create.get("data")}

    studio_id = create["data"].get("id") or ""
    if not studio_id:
        return {"ok": False, "step": "create_studio", "data": create.get("data")}

    # ── الخطوة 2: إطلاق run داخل الـstudio ───────────────────────────────────
    run_spec = {
        "spec": {
            "display_name": f"surahchain-{job_id}",
            "entrypoint": [
                "/bin/bash", "-lc",
                (
                    "cd /home && git clone -q --depth 1 --branch main "
                    "https://github.com/aliahmed369000000-ai/Neural-Service-Mesh.git && "
                    "cd Neural-Service-Mesh && "
                    "python experiments/surah_chain_network/run_train_then_push.py"
                ),
            ],
        },
    }
    run = _lightning_request("POST", f"/v1/studios/{studio_id}/runs", key, payload=run_spec)
    if run["status"] not in (200, 201):
        return {"ok": False, "status": run["status"], "step": "create_run", "studio_id": studio_id, "data": run.get("data")}

    run_id = run["data"].get("id") or ""
    job = {
        "job_id": job_id,
        "provider": "lightning",
        "studio_id": studio_id,
        "run_id": run_id,
        "preset": preset, "n": n, "epochs": epochs, "batch": batch,
        "machine_type": machine_type,
        "status": "launched",
        "studio_url": f"https://lightning.ai/{os.environ.get('LIGHTNING_USER', '')}" if False else f"{LIGHTNING_API}/studios",
        "created_at": _now(),
    }
    (job_dir / "job.json").write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, **job}


def lightning_run_status(studio_id: str, run_id: str) -> Dict[str, Any]:
    """حالة run داخل studio على Lightning."""
    key = os.environ.get("LIGHTNING_API_KEY") or ""
    if not key:
        return {"ok": False, "error": "LIGHTNING_API_KEY غير مضبوط"}
    res = _lightning_request("GET", f"/v1/studios/{studio_id}/runs/{run_id}", key)
    if res["status"] != 200:
        return {"ok": False, "status": res["status"], "data": res.get("data")}
    data = res["data"]
    status = (data.get("status") or {}).get("phase") if isinstance(data.get("status"), dict) else data.get("status")
    return {"ok": True, "status": status, "data": data}


def lightning_list_jobs() -> List[Dict[str, Any]]:
    jobs = []
    if FREE_DIR.is_dir():
        for d in sorted(FREE_DIR.iterdir()):
            jp = d / "job.json"
            if d.name.startswith("lightning_") and jp.is_file():
                try:
                    jobs.append(json.loads(jp.read_text(encoding="utf-8")))
                except Exception:
                    pass
    return jobs


# ─── واجهة موحدة ─────────────────────────────────────────────────────────────

def free_providers_status() -> Dict[str, Any]:
    """حالة المزودين المجانيين (كوتا + اعتمادات)."""
    return {
        "colab": {
            "name": "Google Colab",
            "cost": "مجاني",
            "accelerator": "T4",
            "quota": "~12 ساعة/جلسة (بدون حد أسبوعي صارم)",
            "setup_needed": "فتح notebook في colab.research.google.com",
            "jobs_count": len(colab_list_jobs()),
        },
        "lightning": {
            "name": "Lightning AI",
            "cost": "مجاني (22 ساعة L4/شهر)",
            "credentials": lightning_credentials_status(),
            "jobs_count": len(lightning_list_jobs()),
        },
    }
