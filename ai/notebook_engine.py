"""
NSM Training Notebook Engine — خلايا مثل Colab/Kaggle
=====================================================
- خلايا: markdown | code (python) | bash | train
- جلسات محفوظة JSON
- تنفيذ محلي آمن + خطط إرسال لـ GPU/Kaggle/Remote
- لا يقطع الجلسة تلقائياً (persistent sessions على القرص)
"""
from __future__ import annotations

import json
import os
import subprocess
import textwrap
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "artifacts" / "model_training" / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

_MAX_OUTPUT = 50_000
_DEFAULT_TIMEOUT = 120


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Cell:
    id: str
    type: str = "code"  # markdown | code | bash | train
    source: str = ""
    outputs: List[Dict[str, Any]] = field(default_factory=list)
    execution_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: str = "idle"  # idle | running | ok | error

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Cell":
        return Cell(
            id=d.get("id") or uuid.uuid4().hex[:8],
            type=d.get("type") or "code",
            source=d.get("source") or "",
            outputs=list(d.get("outputs") or []),
            execution_count=d.get("execution_count"),
            metadata=dict(d.get("metadata") or {}),
            status=d.get("status") or "idle",
        )


@dataclass
class Notebook:
    id: str
    name: str
    cells: List[Cell] = field(default_factory=list)
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    kernel: str = "python3"
    provider: str = "local"  # local | kaggle | colab | modal | lightning | huggingface | runpod | vast | generic_gpu

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "cells": [c.to_dict() for c in self.cells],
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
            "kernel": self.kernel,
            "provider": self.provider,
        }

    @staticmethod
    def from_dict(d: dict) -> "Notebook":
        return Notebook(
            id=d.get("id") or uuid.uuid4().hex[:10],
            name=d.get("name") or "Untitled",
            cells=[Cell.from_dict(c) for c in (d.get("cells") or [])],
            created_at=d.get("created_at") or _now(),
            updated_at=d.get("updated_at") or _now(),
            metadata=dict(d.get("metadata") or {}),
            kernel=d.get("kernel") or "python3",
            provider=d.get("provider") or "local",
        )

    def path(self) -> Path:
        return NB_DIR / f"{self.id}.json"


def list_notebooks() -> List[dict]:
    rows = []
    for p in sorted(NB_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            rows.append({
                "id": d.get("id"),
                "name": d.get("name"),
                "cells": len(d.get("cells") or []),
                "updated_at": d.get("updated_at"),
                "provider": d.get("provider"),
            })
        except Exception:
            continue
    return rows


def load_notebook(nb_id: str) -> Optional[Notebook]:
    p = NB_DIR / f"{nb_id}.json"
    if not p.is_file():
        return None
    return Notebook.from_dict(json.loads(p.read_text(encoding="utf-8")))


def save_notebook(nb: Notebook) -> Path:
    nb.updated_at = _now()
    path = nb.path()
    path.write_text(json.dumps(nb.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    return path



def import_ipynb(path: str | Path, name: Optional[str] = None) -> Notebook:
    """استيراد دفتر Jupyter/Kaggle (.ipynb) إلى مختبر NSM."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells: List[Cell] = []
    for c in raw.get("cells") or []:
        src = c.get("source") or ""
        if isinstance(src, list):
            src = "".join(src)
        ctype = c.get("cell_type") or "code"
        if ctype == "markdown":
            ntype = "markdown"
        else:
            # كشف bash / train تقريبي
            if src.lstrip().startswith("!") or "subprocess" in src[:200]:
                ntype = "code"  # نبقي code لأن ! لا يعمل محلياً كما في IPython
            else:
                ntype = "code"
        cell = Cell(
            id=uuid.uuid4().hex[:8],
            type=ntype,
            source=src,
            execution_count=c.get("execution_count"),
            metadata={"from_ipynb": True, **(c.get("metadata") or {})},
        )
        # لا ننسخ مخرجات ضخمة
        cells.append(cell)
    title = name or (raw.get("metadata") or {}).get("nsm", {}).get("name") or path.stem
    nb = Notebook(id=uuid.uuid4().hex[:10], name=title, cells=cells, provider="kaggle")
    save_notebook(nb)
    return nb


def _surahchain_kaggle_cells() -> List[Cell]:
    """نفس منطق دفتر Kaggle notebookf207055113 — بدون توكنات مضمّنة."""
    return [
        Cell(
            id=uuid.uuid4().hex[:8],
            type="markdown",
            source=textwrap.dedent(
                """                # SurahChain 114 — تدريب (نمط Kaggle)

                ## قبل التشغيل على Kaggle
                1. **Settings → Accelerator → GPU T4** (أو Dual T4)
                2. **Settings → Internet → ON**
                3. **Add-ons → Secrets** → `GITHUB_TOKEN` (صلاحية repo)
                4. عدّل خلية **الإعدادات**
                5. **Save Version → Save & Run All** — يكمل في الخلفية

                ## في NSM Notebook (محلي / Streamlit)
                - نفّذ الخلايا بالترتيب (▶ أو Run All)
                - خلية التدريب تشغّل `run_train_then_push.py`: **تدريب ثم رفع تلقائي** عند النجاح\n                - للتدريب الثقيل على GPU: Kaggle Save & Run All بنفس السكربت
                - **لا تضع مفاتيح في الخلايا** — استخدم Secrets / متغيرات البيئة
                """
            ),
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="code",
            source=textwrap.dedent(
                """                # =========================
                # إعدادات — عدّل هنا فقط
                # =========================
                SCN_PRESET = "medium"   # small | medium | large
                SCN_N = 60000
                SCN_EPOCHS = 30
                SCN_BATCH = 24           # إن نفدت الذاكرة: 16 أو 8
                SCN_FRESH = True
                SCN_COMPILE = True
                SCN_QK_NORM = True
                SCN_GATED_ATTN = True
                AUTO_PUSH = True

                REPO = "aliahmed369000000-ai/Neural-Service-Mesh"
                BRANCH = "main"

                print("=" * 50)
                print("preset:", SCN_PRESET, "N:", SCN_N, "epochs:", SCN_EPOCHS)
                print("batch:", SCN_BATCH, "fresh:", SCN_FRESH)
                print("=" * 50)
                """
            ),
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="code",
            source=textwrap.dedent(
                """                import sys, torch
                print("Python:", sys.version.split()[0])
                print("torch:", torch.__version__)
                print("CUDA:", torch.cuda.is_available(), "| GPUs:", torch.cuda.device_count())
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        print(f"  GPU{i}:", torch.cuda.get_device_name(i))
                else:
                    print("تحذير: لا GPU محلياً — على Kaggle فعّل Accelerator")
                """
            ),
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="code",
            source=textwrap.dedent(
                """                import os
                # التوكن من البيئة فقط — لا تكتب ghp_ هنا
                GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN") or ""
                if not GITHUB_TOKEN:
                    print("⚠ لا GITHUB_TOKEN في البيئة — الاستنساخ الخاص قد يفشل")
                else:
                    print("✓ GITHUB_TOKEN موجود (مخفي)")
                print("REPO ready for clone on Kaggle working dir")
                print("CWD tip: on Kaggle use /kaggle/working/")
                """
            ),
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="bash",
            source="python experiments/surah_chain_network/prepare_pretrain_data.py 2>&1 | tail -30",
            metadata={"note": "تحضير بيانات — يحتاج SCN_N في البيئة"},
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="code",
            source=textwrap.dedent(
                """                import os
                # اضبط من خلية الإعدادات يدوياً أو عبر globals إن نُفذت في نفس الجلسة
                import subprocess, sys
                from pathlib import Path as _P
                for k, v in {
                    "SCN_PRESET": "medium",
                    "SCN_N": "60000",
                    "SCN_EPOCHS": "30",
                    "SCN_BATCH": "24",
                    "SCN_FRESH": "1",
                    "SCN_COMPILE": "1",
                    "SCN_QK_NORM": "1",
                    "SCN_GATED_ATTN": "1",
                    "SCN_CHAIN_SCALE": "1",
                    "AUTO_PUSH": "1",
                }.items():
                    os.environ.setdefault(k, v)
                for name in ("SCN_PRESET", "SCN_N", "SCN_EPOCHS", "SCN_BATCH", "SCN_FRESH", "AUTO_PUSH"):
                    if name in globals():
                        val = globals()[name]
                        os.environ[name] = ("1" if val else "0") if isinstance(val, bool) else str(val)
                print("▶ تدريب ثم رفع تلقائي | AUTO_PUSH=", os.environ.get("AUTO_PUSH"))
                script = _P("experiments/surah_chain_network/run_train_then_push.py")
                # على Kaggle: قد تحتاج --skip-prepare إن نُفّذت خلية التحضير
                r = subprocess.run([sys.executable, str(script)], cwd=str(_P(".").resolve()))
                print("exit", r.returncode)
                if r.returncode != 0:
                    raise SystemExit(r.returncode)
                print("✅ انتهى التدريب والرفع التلقائي")
                """
            ),
            metadata={"type_hint": "train"},
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="code",
            source=textwrap.dedent(
                """                from pathlib import Path
                import json
                exp = Path("experiments/surah_chain_network")
                ckpt = exp / "checkpoints"
                print("--- الملفات ---")
                if ckpt.is_dir():
                    for p in sorted(ckpt.glob("*")):
                        if p.is_file():
                            print(f"  {p.name}: {p.stat().st_size/1e6:.2f} MB")
                    for state in list(ckpt.glob("pretrain_state_*.json")) + list(ckpt.glob("pretrain_torch_state.json")):
                        try:
                            d = json.loads(state.read_text(encoding="utf-8"))
                            print(f"\n[{state.name}]")
                            for k in ("best_loss", "epochs_completed", "preset", "step"):
                                if k in d:
                                    print(f"  {k}:", d[k])
                        except Exception as e:
                            print(state, e)
                else:
                    print("لا مجلد checkpoints بعد")
                """
            ),
        ),
        Cell(
            id=uuid.uuid4().hex[:8],
            type="markdown",
            source=textwrap.dedent(
                """                ## بعد الاستيقاظ (Kaggle)
                1. تأكد أن Version حالتها **Success**
                2. على GitHub راجع `best_loss` في `pretrain_state_*.json`
                3. للاستكمال: `SCN_FRESH = False` ثم Save & Run All

                هذه الجولة تشغّل بنية **114** وتقيس الـloss (هدف بحثي).
                """
            ),
        ),
    ]



def create_notebook(name: str = "NSM Training Lab", template: str = "training") -> Notebook:
    nb = Notebook(id=uuid.uuid4().hex[:10], name=name)
    if template == "training":
        nb.cells = [
            Cell(
                id=uuid.uuid4().hex[:8],
                type="markdown",
                source=(
                    "# 🧪 NSM Training Lab\n"
                    "دفتر تدريب بأسلوب **Colab / Kaggle**: خلايا Markdown وCode وBash.\n"
                    "- التنفيذ المحلي الآن · الإرسال لـ GPU/Kaggle لاحقاً عبر المزوّد.\n"
                    "- الجلسة **محفوظة على القرص** — لا تُمسح بانقطاع المتصفح."
                ),
            ),
            Cell(
                id=uuid.uuid4().hex[:8],
                type="code",
                source=textwrap.dedent(
                    """\
                    import sys, platform
                    print("Python", sys.version.split()[0])
                    print("Platform", platform.platform())
                    try:
                        import torch
                        print("PyTorch", torch.__version__, "CUDA", torch.cuda.is_available())
                        if torch.cuda.is_available():
                            print("GPU", torch.cuda.get_device_name(0))
                    except Exception as e:
                        print("torch:", e)
                    try:
                        from ai.gpu_runtime import detect_device
                        d = detect_device()
                        print("NSM device:", d)
                    except Exception as e:
                        print("gpu_runtime:", e)
                    """
                ),
            ),
            Cell(
                id=uuid.uuid4().hex[:8],
                type="bash",
                source="pwd && ls -la ai | head -20 && git status -sb 2>/dev/null | head -5",
            ),
            Cell(
                id=uuid.uuid4().hex[:8],
                type="train",
                source=textwrap.dedent(
                    """\
                    # خلية تدريب — مثال خفيف (CPU/GPU حسب التوفر)
                    # استبدل بسكربت التدريب الفعلي: run_training_loop.sh أو train_*.py
                    print("Training cell placeholder — ربط مع run_training_loop / Kaggle provider")
                    import os
                    print("NSM_ALLOW_GPU =", os.environ.get("NSM_ALLOW_GPU", "not set"))
                    """
                ),
                metadata={"epochs": 5, "provider_hint": "local"},
            ),
            Cell(
                id=uuid.uuid4().hex[:8],
                type="markdown",
                source=(
                    "### 🚀 مزوّدو GPU\n"
                    "| مزوّد | الاستخدام |\n|------|----------|\n"
                    "| local | تنفيذ هنا |\n"
                    "| **kaggle** | مجاني ~30س/أسبوع + API |\n"
                    "| **modal** | رصيد مجاني + API tokens |\n"
                    "| **lightning** | رصيد شهري + API |\n"
                    "| **huggingface** | HF_TOKEN |\n"
                    "| colab | سريع وقد ينقطع |\n"
                    "| runpod / vast | مستقر (مدفوع غالباً) |\n\n"
                    "كتالوج كامل: `ai/free_gpu_providers.py` — مفاتيح في Streamlit Secrets."
                ),
            ),
        ]
    elif template in ("surahchain", "kaggle_surahchain", "notebookf207055113"):
        nb.cells = _surahchain_kaggle_cells()
        nb.provider = "kaggle"
        nb.metadata["source_notebook"] = "aliahmedmo/notebookf207055113"
    else:
        nb.cells = [Cell(id=uuid.uuid4().hex[:8], type="code", source="print('hello NSM')")]
    save_notebook(nb)
    return nb


def add_cell(nb: Notebook, cell_type: str = "code", source: str = "", index: Optional[int] = None) -> Cell:
    cell = Cell(id=uuid.uuid4().hex[:8], type=cell_type, source=source)
    if index is None or index >= len(nb.cells):
        nb.cells.append(cell)
    else:
        nb.cells.insert(max(0, index), cell)
    save_notebook(nb)
    return cell


def delete_cell(nb: Notebook, cell_id: str) -> bool:
    before = len(nb.cells)
    nb.cells = [c for c in nb.cells if c.id != cell_id]
    save_notebook(nb)
    return len(nb.cells) < before


def move_cell(nb: Notebook, cell_id: str, direction: int) -> None:
    ids = [c.id for c in nb.cells]
    if cell_id not in ids:
        return
    i = ids.index(cell_id)
    j = i + direction
    if j < 0 or j >= len(nb.cells):
        return
    nb.cells[i], nb.cells[j] = nb.cells[j], nb.cells[i]
    save_notebook(nb)


def _truncate(s: str) -> str:
    if len(s) > _MAX_OUTPUT:
        return s[:_MAX_OUTPUT] + "\n... [truncated]"
    return s


def _exec_python(source: str, timeout: int) -> Dict[str, Any]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    try:
        r = subprocess.run(
            ["python3", "-c", source],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
            env=env,
        )
        return {
            "ok": r.returncode == 0,
            "stdout": _truncate(r.stdout or ""),
            "stderr": _truncate(r.stderr or ""),
            "exit_code": r.returncode,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"timeout {timeout}s", "exit_code": 124,
                "duration_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "exit_code": 1,
                "duration_ms": int((time.time() - t0) * 1000)}


def _exec_bash(source: str, timeout: int) -> Dict[str, Any]:
    t0 = time.time()
    try:
        r = subprocess.run(
            source,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(ROOT),
        )
        return {
            "ok": r.returncode == 0,
            "stdout": _truncate(r.stdout or ""),
            "stderr": _truncate(r.stderr or ""),
            "exit_code": r.returncode,
            "duration_ms": int((time.time() - t0) * 1000),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "stdout": "", "stderr": f"timeout {timeout}s", "exit_code": 124,
                "duration_ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "exit_code": 1,
                "duration_ms": int((time.time() - t0) * 1000)}


def detect_compute() -> Dict[str, Any]:
    info: Dict[str, Any] = {"providers": {}}
    try:
        from ai.gpu_runtime import detect_device
        d = detect_device()
        info["local"] = {
            "device": getattr(d, "device_str", str(d)),
            "cuda": getattr(d, "cuda", False),
            "name": getattr(d, "name", None),
            "vram_gb": getattr(d, "total_vram_gb", None),
            "reason": getattr(d, "reason", ""),
        }
    except Exception as e:
        info["local"] = {"error": str(e), "cuda": False}
    for mod, key in (
        ("ai.kaggle_provider", "kaggle"),
        ("ai.remote_gpu_provider", "remote_gpu"),
        ("connectors.kaggle_training_connector", "kaggle_connector"),
        ("ai.free_gpu_providers", "free_gpu_catalog"),
    ):
        try:
            __import__(mod)
            info["providers"][key] = {"available": True, "module": mod}
        except Exception as e:
            info["providers"][key] = {"available": False, "error": str(e)[:120]}
    try:
        from ai.free_gpu_providers import list_free_gpu_providers, provider_env_status
        info["free_gpu_catalog"] = list_free_gpu_providers(include_paid=True)
        info["api_keys_status"] = provider_env_status()
    except Exception as e:
        info["free_gpu_catalog_error"] = str(e)
    return info


def plan_remote_run(nb: Notebook, provider: str) -> Dict[str, Any]:
    """خطة إرسال للمزوّد — لا تشغّل تدريباً ثقيلاً هنا إن لم تُضبط المفاتيح."""
    provider = (provider or "local").lower()
    code_cells = [c.source for c in nb.cells if c.type in ("code", "train", "bash")]
    plan = {
        "ok": True,
        "provider": provider,
        "notebook_id": nb.id,
        "notebook_name": nb.name,
        "n_executable_cells": len(code_cells),
        "steps": [],
        "env_hints": [],
    }
    if provider == "local":
        plan["steps"] = ["execute cells sequentially via notebook_engine.run_cell"]
    elif provider == "kaggle":
        plan["steps"] = [
            "تأكد من KAGGLE_USERNAME + KAGGLE_KEY",
            "استخدم ai/kaggle_provider أو connectors/kaggle_training_connector",
            "ارفع kernel من notebooks/ أو صدّر هذا الدفتر",
        ]
        plan["env_hints"] = ["KAGGLE_USERNAME", "KAGGLE_KEY"]
        try:
            from ai import kaggle_provider as kp
            plan["module_ok"] = True
            if hasattr(kp, "_kaggle_cli_available"):
                plan["kaggle_cli"] = kp._kaggle_cli_available()
        except Exception as e:
            plan["module_ok"] = False
            plan["error"] = str(e)
    elif provider == "colab":
        plan["steps"] = [
            "افتح notebooks/*Colab*.ipynb في Google Colab",
            "scripts/colab_bootstrap.py لربط المستودع",
            "Colab قد ينقطع — للديمومة استخدم Kaggle أو RunPod/Vast",
        ]
    elif provider in ("modal", "lightning", "huggingface", "hf"):
        try:
            from ai.free_gpu_providers import plan_for_provider
            pid = "huggingface" if provider == "hf" else provider
            detailed = plan_for_provider(pid, notebook_id=nb.id)
            plan["steps"] = detailed.get("steps") or []
            plan["env"] = detailed.get("env")
            plan["ready_to_submit"] = detailed.get("ready_to_submit")
            plan["provider_meta"] = detailed.get("provider")
        except Exception as e:
            plan["ok"] = False
            plan["error"] = str(e)
    elif provider in ("runpod", "vast", "generic_gpu", "remote"):
        plan["steps"] = [
            "اضبط مفاتيح المزوّد في Secrets",
            "ai/remote_gpu_provider.py لإرسال مهمة",
            "ارفع artifacts/model_training/notebooks/{id}.json أو سكربت التدريب",
        ]
        try:
            from ai import remote_gpu_provider as rgp
            plan["module_ok"] = True
            plan["module"] = "ai.remote_gpu_provider"
        except Exception as e:
            plan["module_ok"] = False
            plan["error"] = str(e)
        try:
            from ai.free_gpu_providers import plan_for_provider
            plan["catalog"] = plan_for_provider(provider if provider in ("runpod", "vast") else "runpod", notebook_id=nb.id)
        except Exception:
            pass
    else:
        # أي id من الكتالوج
        try:
            from ai.free_gpu_providers import plan_for_provider
            detailed = plan_for_provider(provider, notebook_id=nb.id)
            if detailed.get("ok"):
                plan["steps"] = detailed.get("steps") or []
                plan["env"] = detailed.get("env")
                plan["ready_to_submit"] = detailed.get("ready_to_submit")
                plan["provider_meta"] = detailed.get("provider")
            else:
                plan["ok"] = False
                plan["error"] = detailed.get("error") or f"مزوّد غير معروف: {provider}"
        except Exception as e:
            plan["ok"] = False
            plan["error"] = f"مزوّد غير معروف: {provider} ({e})"
    return plan


def run_cell(nb: Notebook, cell_id: str, timeout: int = _DEFAULT_TIMEOUT) -> Cell:
    cell = next((c for c in nb.cells if c.id == cell_id), None)
    if not cell:
        raise KeyError(cell_id)
    cell.status = "running"
    cell.outputs = []
    save_notebook(nb)

    if cell.type == "markdown":
        cell.status = "ok"
        cell.outputs = [{"type": "markdown", "data": cell.source}]
        save_notebook(nb)
        return cell

    if cell.type == "bash":
        res = _exec_bash(cell.source, timeout)
    else:
        # code + train
        res = _exec_python(cell.source, timeout)

    cell.execution_count = (cell.execution_count or 0) + 1
    cell.status = "ok" if res.get("ok") else "error"
    cell.outputs = [{
        "type": "stream",
        "stdout": res.get("stdout", ""),
        "stderr": res.get("stderr", ""),
        "exit_code": res.get("exit_code"),
        "duration_ms": res.get("duration_ms"),
    }]
    save_notebook(nb)
    return cell


def run_all(nb: Notebook, timeout: int = _DEFAULT_TIMEOUT, stop_on_error: bool = True) -> List[dict]:
    results = []
    for cell in nb.cells:
        if cell.type == "markdown":
            continue
        try:
            c = run_cell(nb, cell.id, timeout=timeout)
            results.append({"id": c.id, "status": c.status})
            if c.status == "error" and stop_on_error:
                break
        except Exception as e:
            results.append({"id": cell.id, "status": "error", "error": str(e)})
            if stop_on_error:
                break
    return results


def export_ipynb(nb: Notebook) -> dict:
    """تصدير شبه Jupyter nbformat 4."""
    cells_out = []
    for c in nb.cells:
        if c.type == "markdown":
            cells_out.append({
                "cell_type": "markdown",
                "metadata": {},
                "source": c.source.splitlines(keepends=True) or [""],
            })
        else:
            cells_out.append({
                "cell_type": "code",
                "execution_count": c.execution_count,
                "metadata": {"nsm_type": c.type, **(c.metadata or {})},
                "source": c.source.splitlines(keepends=True) or [""],
                "outputs": [],
            })
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "nsm": {"id": nb.id, "provider": nb.provider, "name": nb.name},
        },
        "cells": cells_out,
    }
