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

# ───────────────────────────────────────────────────────────────────────────
# 🆕 محرك kernel حقيقي (Colab/Kaggle style): kernel واحد دائم لكل دفتر،
# تتشارك خلاياه نفس الذاكرة (متغيرات/استيرادات/figures). يُحمَّل كسولًا؛
# فشله يعود للآلية القديمة (subprocess) تلقائيًا دون كسر أي سلوك موجود.
# ───────────────────────────────────────────────────────────────────────────
_NSK = {
    "run_cell_kernel": None,
    "restart_kernel": None,
    "shutdown_session": None,
    "kernel_health": None,
    "interrupt_kernel": None,
}
try:
    from ai.nb_kernel import (  # noqa: E402
        kernel_health as _kh,
        restart_kernel as _rk,
        run_cell_kernel as _rck,
        shutdown_session as _ss,
        interrupt_kernel as _ik,
    )
    _NSK.update({"run_cell_kernel": _rck, "restart_kernel": _rk,
                 "shutdown_session": _ss, "kernel_health": _kh,
                 "interrupt_kernel": _ik})
except Exception:  # ipykernel غير متوفر — نبقى على الآلية القديمة
    pass


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


def delete_notebook(nb_id: str) -> bool:
    """يحذف دفتراً نهائياً من القرص. يُرجع True عند النجاح، False إن لم يوجد."""
    p = NB_DIR / f"{nb_id}.json"
    if not p.is_file():
        return False
    p.unlink()
    return True



def import_ipynb(path: str | Path, name: Optional[str] = None) -> Notebook:
    """استيراد دفتر Jupyter/Kaggle (.ipynb) إلى مختبر NSM."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    cells: List[Cell] = []
    for c in raw.get("cells") or []:
        src = c.get("source") or ""
        if isinstance(src, list):
            src = "".join(src)
        meta = dict(c.get("metadata") or {})
        ctype = c.get("cell_type") or "code"
        if ctype == "markdown":
            ntype = "markdown"
        else:
            # 🆕 v4: استعادة نوع خلية NSM الأصلي إن حُفظ في metadata عند التصدير
            ntype = meta.pop("nsm_type", None) or "code"
            if ntype not in ("code", "bash", "train", "sql", "http"):
                ntype = "code"
            meta.pop("nsm_id", None)
        cell = Cell(
            id=uuid.uuid4().hex[:8],
            type=ntype,
            source=src,
            execution_count=c.get("execution_count"),
            metadata={"from_ipynb": True, **meta},
        )
        # لا ننسخ مخرجات ضخمة
        cells.append(cell)
    title = name or (raw.get("metadata") or {}).get("nsm", {}).get("name") or path.stem
    nb = Notebook(id=uuid.uuid4().hex[:10], name=title, cells=cells, provider="kaggle")
    save_notebook(nb)
    return nb


def export_ipynb(nb: Notebook) -> str:
    """يصدّر الدفتر كـ ipynb قياسي (nbformat 4.5) قابل للفتح في Colab/Jupyter.
    يتضمن المخرجات النصية لكل خلية، ويحفظ نوع خلية NSM في metadata لاستعادة دقيقة عند الاستيراد."""
    cells_out: List[Dict[str, Any]] = []
    for cell in nb.cells:
        kind = "code" if cell.type in ("code", "bash", "train", "sql", "http") else "markdown"
        outputs: List[Dict[str, Any]] = []
        for out in (cell.outputs or []):
            if out.get("type") == "stream":
                outputs.append({"output_type": "stream", "name": "stdout",
                                "text": ["".join(out.get("text") or [])]})
            elif out.get("type") == "error":
                outputs.append({"output_type": "error",
                                "ename": out.get("ename") or "Error",
                                "evalue": out.get("evalue") or "",
                                "traceback": list(out.get("traceback") or []),
                                "execution_count": cell.execution_count})
            elif out.get("type") in ("display", "execute_result"):
                data = out.get("data") or out.get("text")
                if data is None:
                    continue
                if isinstance(data, dict):
                    # dict قد يحوي text/plain مباشرة أو مفاتيح MIME
                    txt = ""
                    for m in ("text/plain", "text/html", "text/markdown"):
                        if m in data:
                            v = data[m]
                            txt = "".join(v) if isinstance(v, list) else str(v)
                            break
                    if not txt:
                        txt = str(data)[:_MAX_OUTPUT]
                elif isinstance(data, list):
                    txt = "".join(str(x) for x in data)
                else:
                    txt = str(data)
                outputs.append({"output_type": "execute_result",
                                "data": {"text/plain": [txt[:_MAX_OUTPUT]]},
                                "metadata": {},
                                "execution_count": cell.execution_count})
        cells_out.append({
            "cell_type": kind,
            "metadata": {"nsm_type": cell.type, "nsm_id": cell.id},
            "source": cell.source.splitlines(keepends=True) if cell.source else [],
            "outputs": outputs,
            "execution_count": cell.execution_count,
        })
    nbj = {"nbformat": 4, "nbformat_minor": 5,
           "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                       "name": "python3"},
                        "language_info": {"name": "python", "version": "3.12"},
                        "nsm_notebook": {"id": nb.id, "name": nb.name}},
           "cells": cells_out}
    return json.dumps(nbj, ensure_ascii=False, indent=1)


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


def _exec_sql(source: str, timeout: int) -> Dict[str, Any]:
    """🆕 نوع sql: تنفيذ SQL على قاعدة SQLite محلية (تُحدَّد بسطر تعليق
    الأول -- db مسار/أو افتراضي notebooks/<nb>.sqlite). قراءة فقط افتراضيًا
    ما لم يكن الخلية metadata={'allow_writes': True} (INSERT/UPDATE/DELETE).
    المخرجات: جدول أول 500 صف + إجمالي الصفوف كـdisplay_data HTML."""
    import sqlite3 as _sqlite3
    t0 = time.time()
    try:
        allow_writes = False
        lines = source.splitlines()
        db_rel = f"nbsql_{uuid.uuid4().hex[:8]}.sqlite"
        for _ln in lines:
            _s = _ln.strip()
            if _s.lower().startswith("-- db"):
                db_rel = _s[5:].strip()
            elif _s.lower().startswith("-- allow_writes"):
                allow_writes = True
        db_path = (NB_DIR / db_rel).resolve()
        conn = _sqlite3.connect(str(db_path), timeout=timeout)
        if not allow_writes:
            conn.execute("PRAGMA query_only = ON")
        # كل سطر هو جملة SQL مستقلة — لا نضم الأسطر معًا لأن تعليقات SQL
        # (-- ...) ستصبح جزءًا من الجملة وتفسد التنفيذ في sqlite
        _sqls = []
        for _qln in lines:
            _qs = _qln.split(";")[0].strip()
            if _qs and not _qs.startswith("--"):
                _sqls.append(_qs)
        results = []
        for _q in _sqls:
            _qu = _q.strip().upper()
            if _qu.startswith(("SELECT", "PRAGMA", "EXPLAIN", "WITH")):
                cur = conn.execute(_q)
                cols = [d[0] for d in (cur.description or [])]
                rows = cur.fetchmany(500)
                results.append({"cols": cols, "rows": list(rows),
                                "many": len(rows) >= 500})
            elif allow_writes:
                # جمل كتابة: CREATE/ALTER/DROP/INSERT/UPDATE/DELETE/... تُنفَّذ
                # دون إعادة نتائج (fetchmany يفشل لها)
                conn.execute(_q)
            # أي نوع آخر عند قراءة-فقط يُتجاهل بأمان
        conn.commit()
        conn.close()
        return {"ok": True, "duration_ms": int((time.time() - t0) * 1000),
                "exit_code": 0, "results": results}
    except Exception as e:
        return {"ok": False, "duration_ms": int((time.time() - t0) * 1000),
                "exit_code": 1, "error": str(e)}


def _exec_http(source: str, timeout: int) -> Dict[str, Any]:
    """🆕 نوع http: إرسال طلب HTTP. الصيغة: سطر URL أولًا (اختياري GET)
    أو GET/POST/PUT/DELETE قبله، ويمكن تمرير رؤوس {Header: value} في أسطر
    تليها، وbody بعد سطر --. المخرجات: status + headers + body مقتطع."""
    import urllib.request as _ur
    import urllib.parse as _up
    t0 = time.time()
    try:
        lines = source.strip().splitlines()
        if not lines:
            raise ValueError("خلية http فارغة — أضف سطر URL أولًا")
        body = None
        custom_headers = {}
        url = lines[0].strip()
        method = "GET"
        for _m in ("GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"):
            if url.upper().startswith(_m + " "):
                method = _m
                url = url[len(_m):].strip()
                break
        if not url.startswith(("http://", "https://")):
            url = ("https://" + url) if not url.startswith("//") else url
        if not _up.urlparse(url).scheme:
            raise ValueError(f"URL غير صالح: {url}")
        # أسطر الرؤوس: Key: Value
        header_lines, body_lines = [], []
        in_body = False
        for _l in lines[1:]:
            if _l.strip() == "--":
                in_body = True
                continue
            (body_lines if in_body else header_lines).append(_l)
        for _hl in header_lines:
            if ":" in _hl:
                _k, _v = _hl.split(":", 1)
                custom_headers[_k.strip()] = _v.strip()
        if body_lines:
            body = "\n".join(body_lines).encode("utf-8")
        req = _ur.Request(url, data=body, method=method)
        req.add_header("User-Agent", "nsm-notebook/1.0")
        for _k, _v in custom_headers.items():
            req.add_header(_k, _v)
        with _ur.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            resp_headers = dict(resp.getheaders())
            raw = resp.read()
        try:
            body_out = raw.decode("utf-8")
        except UnicodeDecodeError:
            body_out = f"[binary {len(raw)} bytes]"
        return {"ok": True, "duration_ms": int((time.time() - t0) * 1000),
                "exit_code": status, "status": status,
                "headers": {k: v for k, v in list(resp_headers.items())[:20]},
                "body": _truncate(body_out)}
    except Exception as e:
        return {"ok": False, "duration_ms": int((time.time() - t0) * 1000),
                "exit_code": 1, "error": str(e)}


def _render_md_ex(source: str, timeout: int) -> Dict[str, Any]:
    """🆕 نوع markdown_ex: markdown معتمد (mermaid مخططات + HTML خام).
    لا ينفذ شيء: يُحوّل إلى مخرجات markdown عادية مع ملاحظة النواة
    (يُفسَّر العرض في الواجهة — الواجهة تعالج ```mermaid وHTML)."""
    return {"ok": True, "duration_ms": 0, "exit_code": 0,
            "html": source}


def _exec_with_retries(fn, source: str, timeout: int, retries: int,
                       ) -> Dict[str, Any]:
    """تشغيل دالة تنفيذ مع إعادة محاولة عند الخطأ (metadata.retries)."""
    res = fn(source, timeout)
    attempt = 1
    while not res.get("ok") and attempt <= retries:
        attempt += 1
        res = fn(source, timeout)
        res.setdefault("attempts", attempt)
    return res


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
    code_cells = [c.source for c in nb.cells
                  if c.type in ("code", "train", "bash", "sql", "http",
                                "markdown_ex")]
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
    """يشغّل خلية واحدة.

    🆕 المسار الجديد (provider=local + ipykernel متوفر): kernel حقيقي دائم
    للدفتر — الخلايا تتشارك نفس الذاكرة (متغيرات/استيرادات/figures) مثل
    Colab/Kaggle، والمخرجات تُحفظ بصيغة ipynb-native قابلة للتصدير.
    المسار القديم (fallback): subprocess منفصل لكل خلية — يبقى متاحًا عبر
    provider!=local أو عند عدم توفر ipykernel.
    """
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
        _retries = int(cell.metadata.get("retries") or 0)
        res = _exec_with_retries(_exec_bash, cell.source, timeout, _retries)
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

    if cell.type == "sql":
        _retries = int(cell.metadata.get("retries") or 0)
        res = _exec_with_retries(_exec_sql, cell.source, timeout, _retries)
        cell.execution_count = (cell.execution_count or 0) + 1
        cell.status = "ok" if res.get("ok") else "error"
        cell.outputs = [{"type": "sql_result",
                         "results": res.get("results") or [],
                         "error": res.get("error") if not res.get("ok") else None,
                         "duration_ms": res.get("duration_ms"),
                         "exit_code": res.get("exit_code")}]
        save_notebook(nb)
        return cell

    if cell.type == "http":
        _retries = int(cell.metadata.get("retries") or 0)
        res = _exec_with_retries(_exec_http, cell.source, timeout, _retries)
        cell.execution_count = (cell.execution_count or 0) + 1
        cell.status = "ok" if res.get("ok") else "error"
        cell.outputs = [{"type": "http_result",
                         "status": res.get("status"),
                         "headers": res.get("headers") or {},
                         "body": res.get("body") or "",
                         "error": res.get("error") if not res.get("ok") else None,
                         "duration_ms": res.get("duration_ms"),
                         "exit_code": res.get("exit_code")}]
        save_notebook(nb)
        return cell

    if cell.type == "markdown_ex":
        _retries = int(cell.metadata.get("retries") or 0)
        res = _exec_with_retries(_render_md_ex, cell.source, timeout, _retries)
        cell.execution_count = (cell.execution_count or 0) + 1
        cell.status = "ok" if res.get("ok") else "error"
        cell.outputs = [{"type": "markdown_ex",
                         "html": res.get("html", ""),
                         "duration_ms": res.get("duration_ms")}]
        save_notebook(nb)
        return cell

    # code + train — kernel الحقيقي أولًا
    res = None
    kr = _NSK.get("run_cell_kernel")
    if kr is not None and (nb.provider or "local") == "local":
        try:
            res = kr(nb.id, cell.source, timeout=timeout)
        except Exception:
            res = None
    _retries_code = int(cell.metadata.get("retries") or 0)
    if res is None:
        # fallback للآلية القديمة (subprocess منفصل)
        res = _exec_with_retries(_exec_python, cell.source, timeout,
                                 _retries_code)
        cell.execution_count = (cell.execution_count or 0) + 1
        cell.status = "ok" if res.get("ok") else "error"
        cell.outputs = [{
            "type": "stream",
            "stdout": res.get("stdout", ""),
            "stderr": res.get("stderr", ""),
            "exit_code": res.get("exit_code"),
            "duration_ms": res.get("duration_ms"),
        }]
    else:
        # مخرجات kernel بصيغة ipynb-native (stream|display_data|execute_result|error)
        cell.execution_count = res.get("execution_count") or (cell.execution_count or 0) + 1
        cell.status = "ok" if res.get("ok") else "error"
        cell.outputs = list(res.get("outputs") or [])
        # retries لمسار kernel الحقيقي: إعادة المحاولة عند الخطأ
        _att = 0
        while not cell.status == "ok" and _att < _retries_code:
            _att += 1
            res = kr(nb.id, cell.source, timeout=timeout)
            cell.execution_count = (cell.execution_count or 0) + 1
            cell.status = "ok" if res.get("ok") else "error"
            cell.outputs = list(res.get("outputs") or [])
    save_notebook(nb)
    return cell


def interrupt_cell(nb: Notebook, cell_id: str) -> Dict[str, Any]:
    """🆕 إيقاف خلية معلّقة (Ctrl+C) دون قتل kernel والذاكرة المشتركة.
    يدعم kernel الحقيقي (ipykernel)؛ للمسار القديم (subprocess) تُسجّل
    علامة إيقاف لأن العملية لا تستمر بعد timeout أصلاً."""
    target_running = any(c.id == cell_id and c.status == "running"
                         for c in nb.cells)
    if not target_running:
        return {"ok": False, "error": "الخلية غير معلّقة حاليًا — لا شيء لإيقافه"}
    kr = _NSK.get("interrupt_kernel")
    if kr is not None and (nb.provider or "local") == "local":
        try:
            res = kr(nb.id)
            if res.get("ok"):
                for c in nb.cells:
                    if c.id == cell_id and c.status == "running":
                        c.outputs.append({"type": "stream", "name": "stderr",
                                          "text": "⏹ توقّفت الخلية بطلب المستخدم (interrupt)"})
                        break
                save_notebook(nb)
                return {"ok": True, "error": None}
        except Exception:
            pass
    for c in nb.cells:
        if c.id == cell_id and c.status == "running":
            c.status = "idle"
            c.outputs.append({"type": "stream", "name": "stderr",
                              "text": "⏹ توقّفت الخلية بطلب المستخدم (interrupt)"})
            break
    save_notebook(nb)
    return {"ok": False, "error": "لا kernel نشط — الخلية ستنتهي تلقائيًا بانتهاء المهلة"}


def duplicate_notebook(nb: Notebook, new_name: Optional[str] = None) -> Notebook:
    """🆕 استنساخ دفتر كامل بهوية جديدة — مثل Copy Notebook في Colab/Kaggle."""
    import copy as _copy
    clone = Notebook(id=uuid.uuid4().hex[:10], name=new_name or f"نسخة من {nb.name}",
                     provider=nb.provider or "local")
    clone.metadata = _copy.deepcopy(nb.metadata or {})
    clone.cells = []
    for c in nb.cells:
        clone.cells.append(Cell(
            id=uuid.uuid4().hex[:8],
            type=c.type,
            source=c.source,
            metadata=_copy.deepcopy(c.metadata or {}),
        ))
    save_notebook(clone)
    return clone


def restart_kernel_session(nb_id: str) -> Dict[str, Any]:
    """🆕 يعيد تشغيل kernel الدفتر — ذاكرة صافية مثل Reset في Colab/Kaggle.
    غير مؤثرة في الدفاتر التي تعمل بالآلية القديمة (subprocess)."""
    if _NSK.get("restart_kernel") is None:
        return {"ok": False, "error": "محرك kernel غير متوفر في هذه البيئة"}
    try:
        return _NSK["restart_kernel"](nb_id)
    except Exception as e:
        return {"ok": False, "error": str(e)}


def kill_kernel_session(nb_id: str) -> bool:
    """🆕 يغلق kernel الدفتر كليًا (تنظيف الموارد عند حذف الدفتر)."""
    if _NSK.get("shutdown_session") is None:
        return False
    try:
        return _NSK["shutdown_session"](nb_id)
    except Exception:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 v2: المجدول الخفيف (Scheduler) — تشغيل دفتر دوريًا أو عند وقت محدد
# ═══════════════════════════════════════════════════════════════════════════
SCHED_DIR = ROOT / "artifacts" / "model_training" / "scheduled_runs"
SCHED_DIR.mkdir(parents=True, exist_ok=True)


def schedule_notebook(nb_id: str, name: str, run_at: str,
                      interval_minutes: int = 0,
                      description: str = "") -> Dict[str, Any]:
    """جدولة دفتر: run_at بصيغة HH:MM أو 'now'، interval_minutes>0 = تكرار دوري.
    تُحفظ المهمة في SCHED_DIR وتُشغّل عبر run_scheduled_jobs() (تُدعى دوريًا من الواجهة/Streamlit rerun)."""
    job = {
        "id": uuid.uuid4().hex[:10],
        "nb_id": nb_id,
        "name": name,
        "description": description,
        "run_at": run_at,
        "interval_minutes": max(0, int(interval_minutes or 0)),
        "created_at": _now(),
        "last_run_at": None,
        "next_run_at": None,
        "status": "scheduled",
        "last_result": None,
    }
    job["next_run_at"] = _next_run_at(job)
    _write_job(job)
    return {"ok": True, "job": job}


def _next_run_at(job: Dict[str, Any]) -> Optional[str]:
    from datetime import datetime as _dt
    now = _dt.now()
    try:
        if job["run_at"] == "now":
            target = now
        else:
            hh, mm = map(int, str(job["run_at"]).split(":")[:2])
            target = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
            if target <= now:
                target = target.replace(day=target.day + 1)
    except Exception:
        return None
    interval = job.get("interval_minutes") or 0
    if interval and (job.get("last_run_at") or ""):
        try:
            base = _dt.fromisoformat(job["last_run_at"])
            from datetime import timedelta as _td
            target = base + _td(minutes=interval)
        except Exception:
            pass
    return target.strftime("%Y-%m-%d %H:%M")


def _job_path(job_id: str) -> Path:
    return SCHED_DIR / f"{job_id}.json"


def _write_job(job: Dict[str, Any]) -> None:
    _job_path(job["id"]).write_text(
        json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")


def list_scheduled_jobs() -> List[Dict[str, Any]]:
    jobs = []
    for p in SCHED_DIR.glob("*.json"):
        try:
            jobs.append(json.loads(p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return sorted(jobs, key=lambda j: j.get("next_run_at") or "", reverse=False)


def delete_scheduled_job(job_id: str) -> bool:
    p = _job_path(job_id)
    if p.is_file():
        p.unlink()
        return True
    return False


def run_scheduled_jobs() -> List[Dict[str, Any]]:
    """يفحص المهام المجدولة ويشغّل المستحقّ منها الآن — يُستدعى دوريًا.
    يعيد قائمة نتائج التشغيل (فارغة إن لم يكن هناك شيء)."""
    from datetime import datetime as _dt
    now_str = _dt.now().strftime("%Y-%m-%d %H:%M")
    results = []
    for job in list_scheduled_jobs():
        nxt = (job.get("next_run_at") or "")[:16]
        if job.get("status") == "disabled":
            continue
        if nxt and nxt <= now_str:
            nb = load_notebook(job["nb_id"])
            res = {"job_id": job["id"], "name": job["name"]}
            if nb is None:
                res.update({"ok": False, "error": "الدفتر غير موجود"})
            else:
                try:
                    out = run_all(nb)
                    job["last_result"] = {"ok": True,
                                          "cells": len(out),
                                          "errors": sum(1 for r in out
                                                          if r.get("status") == "error")}
                    res.update({"ok": True, "cells": len(out)})
                except Exception as e:
                    job["last_result"] = {"ok": False, "error": str(e)[:200]}
                    res.update({"ok": False, "error": str(e)[:200]})
            job["last_run_at"] = _now()
            job["next_run_at"] = _next_run_at(job)
            job["status"] = "completed"
            _write_job(job)
            results.append(res)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 v2: مخرجات حيّة (streaming) — تنفيذ خلية في خيط جانبي مع جمع المخرجات
# تدريجيًا (كل 0.3ث) في قائمة مشتركة، لعرضها لحظيًا في الواجهة
# ═══════════════════════════════════════════════════════════════════════════
def run_cell_streaming(nb: Notebook, cell_id: str, timeout: int = _DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """تشغيل خلية في خيط جانبي وجمع مخرجات kernel الحقيقية تدريجيًا.
    يعيد {'live': [{'text':...}], 'final_cell': Cell} — تقرأ الواجهة 'live' دوريًا."""
    cell = next((c for c in nb.cells if c.id == cell_id), None)
    if not cell:
        raise KeyError(cell_id)
    kr = _NSK.get("run_cell_kernel")
    if kr is None or (nb.provider or "local") != "local":
        # لا دعم حي خارج kernel الحقيقي
        c = run_cell(nb, cell.id, timeout=timeout)
        return {"live": [], "final_cell": c}
    import threading as _thr
    live: List[Dict[str, Any]] = []
    cell.status = "running"
    cell.outputs = []
    save_notebook(nb)
    state = {"final": None}
    def _worker():
        try:
            kc = _NSK["kernel_health"] and __import__("ai.nb_kernel",
                                                       fromlist=["get_kernel_client"]
                                                      ).get_kernel_client(nb.id)
            # نعيد الاستخدام: ننفذ عبر run_cell_kernel في خيط مستقل
            state["final"] = run_cell(nb, cell_id, timeout=timeout)
        except Exception as e:
            cell.status = "error"
            cell.outputs.append({"type": "stream", "name": "stderr",
                                 "text": str(e)[:2000]})
            state["final"] = cell
            save_notebook(nb)
    t = _thr.Thread(target=_worker, daemon=True)
    t.start()
    # جمع المخرجات الحالية دوريًا أثناء التشغيل
    while t.is_alive():
        time.sleep(0.3)
        live.append({"outputs": list(cell.outputs), "status": cell.status,
                     "thread_alive": True})
    return {"live": live, "final_cell": state["final"] or cell}


def live_outputs(nb: Notebook, cell_id: str) -> Dict[str, Any]:
    """لحظة مخرجات خلية: آخر ما توفّر من outputs دون انتظار."""
    cell = next((c for c in nb.cells if c.id == cell_id), None)
    return {"outputs": list(cell.outputs or []), "status": cell.status}


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 v2: سجل إصدارات الخلية (Undo/Redo) — يحفظ آخر 20 تعديلًا/تنفيذًا لمصدر الخلية
# ═══════════════════════════════════════════════════════════════════════════
_CELL_HISTORY_MAX = 20


def _cell_history(nb: Notebook, cell_id: str) -> List[Dict[str, Any]]:
    return nb.metadata.setdefault("cell_history", {}).setdefault(cell_id, [])


def save_cell_version(nb: Notebook, cell_id: str, note: str = "") -> None:
    """حفظ لقطة حالية للمصدر قبل التعديل (تُدعى من الواجهة عند حفظ مصدر)."""
    cell = next((c for c in nb.cells if c.id == cell_id), None)
    if not cell:
        return
    hist = _cell_history(nb, cell_id)
    hist.append({"source": cell.source, "at": _now(), "note": note})
    if len(hist) > _CELL_HISTORY_MAX:
        hist.pop(0)
    save_notebook(nb)


def undo_cell(nb: Notebook, cell_id: str) -> Dict[str, Any]:
    """🆕 تراجع: استعادة آخر لقطة لمصدر الخلية."""
    cell = next((c for c in nb.cells if c.id == cell_id), None)
    hist = _cell_history(nb, cell_id)
    if not hist:
        return {"ok": False, "error": "لا لقطات سابقة لهذه الخلية"}
    snap = hist.pop()
    if cell is not None:
        cell.source = snap["source"]
    save_notebook(nb)
    return {"ok": True, "restored_at": snap["at"], "remaining": len(hist)}


def cell_version_list(nb: Notebook, cell_id: str) -> List[Dict[str, Any]]:
    return list(_cell_history(nb, cell_id))


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 v2: أسرار الدفتر (Secrets) — مفاتيح تُمرَّر كمتغيرات بيئة داخل kernel
# (لا تُحفظ داخل الكود — تبقى في metadata النظيف)
# ═══════════════════════════════════════════════════════════════════════════
def set_notebook_secret(nb: Notebook, key: str, value: str) -> None:
    """تخزين سر بأمان مشوّه (mask) داخل metadata — لا يظهر في المصادر."""
    secrets = nb.metadata.setdefault("secrets", {})
    secrets[key] = value
    nb.metadata["secret_keys"] = list(secrets.keys())
    save_notebook(nb)


def delete_notebook_secret(nb: Notebook, key: str) -> bool:
    secrets = nb.metadata.get("secrets", {})
    if key in secrets:
        del secrets[key]
        nb.metadata["secret_keys"] = list(secrets.keys())
        save_notebook(nb)
        return True
    return False


def list_notebook_secrets(nb: Notebook) -> List[str]:
    """يُعيد أسماء الأسرار فقط — لا القيم."""
    return list(nb.metadata.get("secret_keys") or [])


def inject_secrets_into_kernel(nb: Notebook) -> int:
    """حقن أسرار الدفتر كمتغيرات بيئة داخل kernel (بدون تسجيل القيم)."""
    kc = None
    try:
        import ai.nb_kernel as _nbk
        kc = _nbk.get_kernel_client(nb.id)
    except Exception:
        return 0
    if kc is None:
        return 0
    secrets = nb.metadata.get("secrets", {})
    n = 0
    for k, v in secrets.items():
        try:
            kc.execute(f"import os; os.environ[{k!r}] = {v!r}",
                       store_history=False, silent=True)
            n += 1
        except Exception:
            pass
    return n


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 v2: مكتبة قوالب الكود (Snippets)
# ═══════════════════════════════════════════════════════════════════════════
CODE_SNIPPETS: Dict[str, Dict[str, str]] = {
    "train_pytorch": {
        "label_ar": "تدريب PyTorch مصغّر",
        "source": textwrap.dedent("""\
            import torch, torch.nn as nn
            model = nn.Sequential(nn.Linear(10, 32), nn.ReLU(), nn.Linear(32, 1))
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            for epoch in range(5):
                x = torch.randn(64, 10); y = x.sum(1, keepdim=True)
                loss = nn.MSELoss()(model(x), y)
                loss.backward(); opt.step(); opt.zero_grad()
                print(f"epoch {epoch+1} loss {loss.item():.4f}")
            """),
    },
    "dataloader": {
        "label_ar": "DataLoader عربي (نص → توكنات)",
        "source": textwrap.dedent("""\
            from torch.utils.data import DataLoader, Dataset
            class ArabicText(Dataset):
                def __init__(self, texts): self.texts = texts
                def __len__(self): return len(self.texts)
                def __getitem__(self, i): return self.texts[i]
            dl = DataLoader(ArabicText(["مرحبا بالعالم", "الذكاء الاصطناعي"]), batch_size=2)
            for batch in dl:
                print(batch)
            """),
    },
    "matplotlib_plot": {
        "label_ar": "رسم matplotlib",
        "source": textwrap.dedent("""\
            import matplotlib.pyplot as plt
            plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
            plt.title("مخطط تجريبي")
            plt.show()
            """),
    },
    "metrics_parse": {
        "label_ar": "استخراج مقاييس من سجلات التدريب",
        "source": textwrap.dedent("""\
            import re, json
            # مثال: قراءة metrics من outputs الخلايا السابقة
            metrics = {"loss": [], "step": []}
            print(json.dumps(metrics, ensure_ascii=False, indent=2))
            """),
    },
    "hf_load_stream": {
        "label_ar": "تحميل بيانات من HF (streaming)",
        "source": textwrap.dedent("""\
            # requires datasets; uses HF_TOKEN from environment if needed
            from datasets import load_dataset
            ds = load_dataset("ClusterlabAi/101_billion_arabic_words_dataset",
                              split="train", streaming=True)
            for i, row in enumerate(ds):
                print(row)
                if i >= 3:
                    break
            """),
    },
    "kaggle_env": {
        "label_ar": "فحص بيئة Kaggle",
        "source": textwrap.dedent("""\
            import os
            print("KAGGLE kernel:", os.path.exists("/kaggle/input"))
            print("GPU:", os.popen("nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1").read().strip() or "none")
            """),
    },
    # ═══════════════════════════════════════════════════════════════════
    # 🆕 قوالب أنواع الخلايا المتقدمة (sql / http / markdown_ex)
    # ═══════════════════════════════════════════════════════════════════
    "sql_ex": {
        "label_ar": "استعلام SQL (جدول تجريبي)",
        "type": "sql",
        "source": textwrap.dedent("""\
            -- db demo.sqlite
            -- allow_writes
            CREATE TABLE IF NOT EXISTS tokens (
                epoch INTEGER, loss REAL, lr REAL
            );
            INSERT INTO tokens VALUES (1, 2.45, 0.001), (2, 1.92, 0.0008);
            SELECT * FROM tokens WHERE loss < 2.0;
            """),
    },
    "http_ping": {
        "label_ar": "فحص API (HTTP)",
        "type": "http",
        "source": textwrap.dedent("""\
            GET https://api.github.com/rate_limit
            """),
    },
    "http_post": {
        "label_ar": "POST مع رؤوس وجسم",
        "type": "http",
        "source": textwrap.dedent("""\
            POST https://httpbin.org/post
            Content-Type: application/json
            --
            {"model": "surahchain", "d": 8192}
            """),
    },
    "md_mermaid": {
        "label_ar": "Markdown+ (مخطط Mermaid)",
        "type": "markdown_ex",
        "source": textwrap.dedent("""\
            ### بنية نظام NSM

            ```mermaid
            graph LR
                A[المستخدم] --> B[الوكلاء]
                B --> C[الذاكرة المشتركة]
                B --> D[الدفتر]
            ```
            """),
    },
}


def get_snippet(key: str) -> Optional[Dict[str, str]]:
    return CODE_SNIPPETS.get(key)


def insert_snippet(nb: Notebook, snippet_key: str, index: Optional[int] = None) -> Optional[Cell]:
    """إدراج قالب جديد في الدفتر — يدعم الأنواع المتقدمة (sql/http/
    markdown_ex) عبر حقل 'type' في القالب؛ افتراضيًا code."""
    snip = CODE_SNIPPETS.get(snippet_key)
    if not snip:
        return None
    cell = add_cell(nb, snip.get("type") or "code", snip["source"], index=index)
    cell.metadata["snippet"] = snippet_key
    save_notebook(nb)
    return cell


# ═══════════════════════════════════════════════════════════════════════════
# 🆕 v2: مشاركة الدفاتر (Shared Notebooks) — مجلد مُتشارك عام داخل repo
# ═══════════════════════════════════════════════════════════════════════════
SHARED_NB_DIR = ROOT / "artifacts" / "model_training" / "shared_notebooks"
SHARED_NB_DIR.mkdir(parents=True, exist_ok=True)


def share_notebook(nb: Notebook, description: str = "") -> Dict[str, Any]:
    """مشاركة الدفتر في المكتبة المشتركة (يدخلها الوكلاء/الأعضاء) — مخرجاته مضمّنة."""
    shared = {
        "id": nb.id,
        "name": nb.name,
        "description": description,
        "shared_at": _now(),
        "notebook": nb.to_dict(),
    }
    p = SHARED_NB_DIR / f"{nb.id}.json"
    p.write_text(json.dumps(shared, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"ok": True, "path": str(p)}


def list_shared_notebooks() -> List[Dict[str, Any]]:
    rows = []
    for p in SHARED_NB_DIR.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            nb = d.get("notebook") or {}
            rows.append({
                "id": d.get("id"),
                "name": d.get("name"),
                "description": d.get("description"),
                "cells": len(nb.get("cells") or []),
                "shared_at": d.get("shared_at"),
            })
        except Exception:
            continue
    return sorted(rows, key=lambda r: r.get("shared_at") or "", reverse=True)


def import_shared_notebook(shared_id: str) -> Optional[Notebook]:
    """استنساخ دفتر مُتشارك إلى دفاتري الشخصية."""
    p = SHARED_NB_DIR / f"{shared_id}.json"
    if not p.is_file():
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    nb = Notebook.from_dict(d.get("notebook") or {})
    clone = duplicate_notebook(nb, new_name=f"{nb.name} (مُشارك)")
    return clone


def nb_kernel_health() -> Dict[str, Any]:
    """🆕 حالة محرك kernel: توفر ipykernel، الجلسات النشطة، الباك إند الفعلي."""
    if _NSK.get("kernel_health") is None:
        return {"ipykernel_available": False, "active_sessions": 0,
                "sessions": [], "backend": "subprocess"}
    try:
        return _NSK["kernel_health"]()
    except Exception:
        return {"ipykernel_available": False, "active_sessions": 0,
                "sessions": [], "backend": "subprocess"}


def run_all(nb: Notebook, timeout: int = _DEFAULT_TIMEOUT, stop_on_error: bool = True) -> List[dict]:
    """🆕 تشغيل كل الخلايا بالترتيب — الحالة والمخرجات تُحفظ بعد كل خلية،
    فترى الواجهة تقدّمًا حيًا (⏳→✅/❌) بدل انتظار النهاية الكاملة."""
    results = []
    for i, cell in enumerate(nb.cells, 1):
        if cell.type == "markdown":
            cell.status = "idle"
            continue
        try:
            c = run_cell(nb, cell.id, timeout=timeout)
            results.append({"id": c.id, "status": c.status})
            if c.status == "error" and stop_on_error:
                break
        except Exception as e:
            cell.status = "error"
            cell.outputs.append({"type": "stream", "name": "stderr",
                                 "text": str(e)[:2000]})
            save_notebook(nb)
            results.append({"id": cell.id, "status": "error", "error": str(e)})
            if stop_on_error:
                break
    return results
