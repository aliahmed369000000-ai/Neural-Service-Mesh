"""
Autonomous Train DevOps — تعديل معلمات عند ثبات الخسارة
======================================================
يقرأ ckg_train_state_v3.json، يكتشف plateau، يكتب ckg_train_hparams_override.json
(يقرأه train_batch_v3.py). لا يشغّل تدريباً ثقيلاً إلا بضوء أخضر من نموذج العالم.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_FILE = ROOT / "ckg_train_state_v3.json"
OVERRIDE_FILE = ROOT / "ckg_train_hparams_override.json"
OUT = ROOT / "artifacts" / "model_training" / "train_devops"
OUT.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_train_state() -> Dict[str, Any]:
    if not STATE_FILE.is_file():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_plateau(loss_tail: List[float], window: int = 12, eps: float = 0.02) -> Dict[str, Any]:
    tail = [float(x) for x in (loss_tail or []) if x is not None]
    if len(tail) < window:
        return {"plateau": False, "reason": "not_enough_points", "n": len(tail)}
    recent = tail[-window:]
    early = recent[: window // 2]
    late = recent[window // 2 :]
    mean_early = statistics.mean(early)
    mean_late = statistics.mean(late)
    improvement = mean_early - mean_late
    plateau = improvement < eps
    return {
        "plateau": plateau,
        "mean_early": round(mean_early, 4),
        "mean_late": round(mean_late, 4),
        "improvement": round(improvement, 4),
        "eps": eps,
    }


def propose_hparams(state: Dict[str, Any], plateau: Dict[str, Any]) -> Dict[str, Any]:
    """يقترح LR أصغر و/أو pack أصغر عند الثبات."""
    last_lr = float(state.get("last_lr") or 1e-4)
    pack = int(state.get("last_pack_size") or 40)
    packs = int(state.get("last_packs_per_run") or 4)
    if plateau.get("plateau"):
        new_lr = max(1e-6, last_lr * 0.5)
        new_pack = max(8, pack // 2) if pack > 16 else pack
        note = "plateau_detected_reduce_lr"
    else:
        new_lr = last_lr
        new_pack = pack
        note = "continue"
    return {
        "lr_max": new_lr,
        "lr_min": max(1e-7, new_lr * 0.08),
        "pack_size": new_pack,
        "packs_per_run": packs,
        "note": note,
        "from_state_lr": last_lr,
        "from_state_pack": pack,
    }


def apply_override(hparams: Dict[str, Any]) -> Dict[str, Any]:
    payload = {
        **hparams,
        "updated_at": _now(),
        "source": "autonomous_train_devops",
    }
    OVERRIDE_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def devops_cycle(run_train: bool = False) -> Dict[str, Any]:
    state = load_train_state()
    plateau = detect_plateau(state.get("loss_history_tail") or [])
    hparams = propose_hparams(state, plateau)
    applied = apply_override(hparams) if plateau.get("plateau") else {"skipped": True, **hparams}
    safety = {}
    try:
        from world_model.environment_model import EnvironmentModel
        safety = EnvironmentModel(model_dir=str(ROOT / "world_model")).assess_training_safety(
            "autonomous_train_devops", estimated_vram_mb=6144
        )
    except Exception as e:
        safety = {"error": str(e), "green_light": False}
    train_cmd = None
    if run_train and safety.get("green_light") and plateau.get("plateau"):
        train_cmd = (
            f"NSM_PACK_SIZE={hparams['pack_size']} "
            f"NSM_PACKS_PER_RUN={hparams['packs_per_run']} "
            f"python3 train_batch_v3.py"
        )
        # لا تنفيذ تلقائي في الطلب المتزامن — يُرجع الأمر فقط
    report = {
        "at": _now(),
        "state_snapshot": {
            "runs": state.get("runs"),
            "global_step": state.get("global_step"),
            "last_lr": state.get("last_lr"),
            "loss_tail_n": len(state.get("loss_history_tail") or []),
        },
        "plateau": plateau,
        "override": applied,
        "safety": safety,
        "train_cmd_suggested": train_cmd,
        "policy": "no_blind_weight_replace",
    }
    path = OUT / f"devops_{int(datetime.now().timestamp())}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["path"] = str(path.relative_to(ROOT))
    return report


def handle_train_devops_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(devops\s*تدريب|تدريب\s*ذاتي\s*معلمات|ضبط\s*معلمات\s*تدريب|plateau\s*train)", text, re.I):
        return None
    r = devops_cycle(run_train=bool(re.search(r"شغ[ّل]|run", text, re.I)))
    return "## 🛠️ Train DevOps ذاتي\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2)[:3500] + "\n```"
