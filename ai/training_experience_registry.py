"""ai/training_experience_registry.py — سجل تجارب التدريب المركزي لمشروع NSM

يلتقط كل كيرنل تدريب منتهٍ (كاملًا أو فاشلًا) ويحفظ تجربة علمية قابلة
للمقارنة: الإعدادات (preset / d_model / N / epochs / batch / الجهاز)،
النتيجة (نجاح/فشل + سبب الفشل)، ومؤشرات الأداء (أفضل خسارة/آخر خسارة
إن وُجدت في progress JSON المرفوعة على GitHub).

الفائدة العملية:
  - عند اكتمال أي كيرنل نضيف صفًا في السجل بدل أن يضيع تاريخ التجارب.
  - دالة compare_runs تقارن الإعدادات والخسائر بين التجارب لاتخاذ
    قرارات الضبط (مثل: هل xlarge يستحق الوقت الإضافي؟).
  - read_kernel_experience تقرأ من GitHub (progress_{TAG}.json +
    checkpoints state) + Kaggle API بدون أي مفاتيح للتاريخ.

التخزين محلي JSONL في artifacts/model_training/lab/experiments.jsonl
(نفس مكان jobs_history.jsonl) — ويمكن لاحقًا رفعه للمستودع تلقائيًا.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ai.training_alerts import ALERTS_DIR, _load_log, _save_log  # noqa: E402 — نفس نمط الوحدات المجاورة

_HERE = Path(__file__).resolve().parent
ROOT = _HERE.parent
EXPERIMENTS_PATH = ROOT / "artifacts" / "model_training" / "lab" / "experiments.jsonl"

REPO = "aliahmed369000000-ai/Neural-Service-Mesh"
BRANCH = "main"
CHECKPOINTS_RAW = (
    f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/experiments/surah_chain_network/checkpoints/"
)
REPO_RAW = (
    f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/experiments/surah_chain_network/"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_experiments() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if EXPERIMENTS_PATH.exists():
        for line in EXPERIMENTS_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def _save_experiments(rows: List[Dict[str, Any]]) -> None:
    EXPERIMENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf-8",
    )


def _http_get_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        import urllib.request

        req = urllib.request.Request(url, headers={"User-Agent": "nsm-training-registry/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def kernel_state_from_github(tag: str = "d8192_s1p0") -> Dict[str, Any]:
    """يقرأ progress_{TAG}.json و state من GitHub raw (لا يحتاج مفاتيح)."""
    data: Dict[str, Any] = {"tag": tag}
    progress = _http_get_json(REPO_RAW + f"progress_{tag}.json")
    if progress:
        data["progress"] = progress
        if isinstance(progress, list):
            losses = [e.get("loss") for e in progress if isinstance(e, dict) and e.get("loss") is not None]
            if losses:
                data["first_loss"] = losses[0]
                data["last_loss"] = losses[-1]
                data["best_loss"] = min(losses)
                data["epochs_recorded"] = len(losses)
        elif isinstance(progress, dict):
            for key in ("epoch", "loss", "best_loss", "steps"):
                if key in progress:
                    data[key] = progress[key]
    state = _http_get_json(CHECKPOINTS_RAW + "pretrain_torch_state.json")
    if state and isinstance(state, dict):
        data["state"] = {k: state.get(k) for k in ("epoch", "step", "best_loss", "total_loss") if k in state}
    return data


def record_training_run(
    kernel_slug: str,
    preset: str = "",
    d_model: int = 0,
    n: int = 0,
    epochs: int = 0,
    batch: int = 0,
    device: str = "",
    status: str = "",
    failure_reason: str = "",
    kernel_url: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """يسجّل تجربة تدريب واحدة في السجل المركزي."""
    entry: Dict[str, Any] = {
        "recorded_at": _now(),
        "kernel_slug": kernel_slug,
        "kernel_url": kernel_url or f"https://www.kaggle.com/code/{kernel_slug}",
        "preset": preset,
        "d_model": d_model,
        "n": n,
        "epochs": epochs,
        "batch": batch,
        "device": device,
        "status": status,
        "failure_reason": failure_reason,
    }
    if d_model and not preset:
        entry["preset"] = "xlarge" if d_model >= 8192 else ("large" if d_model >= 2048 else ("medium" if d_model >= 512 else "small"))
    # إثراء من GitHub إن أمكن
    tag = (extra or {}).get("tag")
    github = kernel_state_from_github(tag) if tag else {}
    entry["github"] = github
    rows = _load_experiments()
    rows.append(entry)
    _save_experiments(rows)
    # تنبيه عبر نظام التنبيهات القائم
    if status in ("complete", "failed"):
        try:
            from ai.training_alerts import alert_job_status

            alert_job_status(
                job_id=kernel_slug,
                status=status,
                kernel_url=entry["kernel_url"],
                preset=preset,
                n=n,
                epochs=epochs,
            )
        except Exception:
            pass
    return entry


def compare_runs(keys: Optional[List[str]] = None, by: str = "best_loss") -> Dict[str, Any]:
    """مقارنة التجارب المسجلة — يعيد جدول مقارنة + الخلاصة.

    by ∈ {best_loss, last_loss, epochs_recorded, duration}
    """
    rows = _load_experiments()
    finished = [r for r in rows if r.get("status") == "complete"]
    result: Dict[str, Any] = {
        "total_runs": len(rows),
        "completed": len(finished),
        "failed": len(rows) - len(finished),
        "table": [],
    }
    for r in rows:
        gh = r.get("github") or {}
        losses = gh.get("progress") if isinstance(gh.get("progress"), list) else None
        prog_dict = gh.get("progress") if isinstance(gh.get("progress"), dict) else {}
        row = {
            "kernel": r.get("kernel_slug"),
            "preset": r.get("preset"),
            "d_model": r.get("d_model"),
            "n": r.get("n"),
            "epochs": r.get("epochs"),
            "device": r.get("device"),
            "status": r.get("status"),
            "first_loss": gh.get("first_loss") if losses else (prog_dict.get("loss") or None),
            "last_loss": gh.get("last_loss") if losses else (prog_dict.get("loss") or None),
            "best_loss": gh.get("best_loss") if losses else (prog_dict.get("best_loss") or None),
            "epochs_recorded": gh.get("epochs_recorded") if losses else (prog_dict.get("epoch") or None),
            "failure": r.get("failure_reason"),
        }
        result["table"].append(row)
    result["by"] = by
    return result


def registry_summary() -> Dict[str, Any]:
    """ملخص السجل للوحات Streamlit."""
    rows = _load_experiments()
    completed = [r for r in rows if r.get("status") == "complete"]
    failed = [r for r in rows if r.get("status") == "failed"]
    return {
        "total": len(rows),
        "completed": len(completed),
        "failed": len(failed),
        "last_run": rows[-1] if rows else None,
    }


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "summary"
    if cmd == "summary":
        print(json.dumps(registry_summary(), ensure_ascii=False, indent=2))
    elif cmd == "compare":
        print(json.dumps(compare_runs(), ensure_ascii=False, indent=2))
    elif cmd == "github":
        tag = sys.argv[2] if len(sys.argv) > 2 else "d8192_s1p0"
        print(json.dumps(kernel_state_from_github(tag), ensure_ascii=False, indent=2))
