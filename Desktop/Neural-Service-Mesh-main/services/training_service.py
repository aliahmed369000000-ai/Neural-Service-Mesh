from __future__ import annotations

from typing import Dict, Any
from pathlib import Path
import json

from services.backend import get_trainer


def get_train_status() -> Dict[str, Any]:
    trainer = get_trainer()
    try:
        return trainer.stats()
    except Exception:
        # fallback to DB stats
        try:
            return trainer.db.stats()
        except Exception:
            return {}


def get_matrix() -> Dict[str, Any]:
    # Try to read neural weights on disk (checkpoints/neural_weights.npy)
    try:
        import numpy as _np
        p = Path("./checkpoints/neural_weights.npy")
        if not p.exists():
            return {"error": "no_weights"}
        w = _np.load(str(p), allow_pickle=False)
        return {
            "shape": list(w.shape),
            "train_steps": int(getattr(w, "_train_steps", 0)),
            "last_loss": float(getattr(w, "_last_loss", 0.0)),
            "weight_stats": {
                "min": round(float(w.min()), 6),
                "max": round(float(w.max()), 6),
                "mean": round(float(w.mean()), 6),
                "std": round(float(w.std()), 6),
            },
            "dimensions": {
                "0_IMPORTANCE":  "أهمية المعلومة",
                "1_CERTAINTY":   "درجة اليقين",
                "2_ABSTRACTION": "مستوى التجريد",
                "3_DOMAIN":      "رمز المجال",
                "4_CONNECTIVITY":"كثافة العلاقات",
                "5_TEMPORALITY": "الزمنية",
                "6_NOVELTY":     "جِدَّة المعلومة",
            },
        }
    except Exception:
        return {"error": "no_weights_or_error"}


def get_status() -> Dict[str, Any]:
    trainer = get_trainer()
    # Provide a minimal status used by the dashboard
    meta = getattr(trainer.ckg, "_data", {}).get("_meta", {})
    started = meta.get("saved_at")
    return {"started_at": started, "timestamp": started}


def get_train_audit() -> Dict[str, Any]:
    trainer = get_trainer()
    db_stats = {}
    try:
        db_stats = trainer.db.stats()
    except Exception:
        pass

    ckg = trainer.ckg

    from pathlib import Path
    weights_path = Path("./checkpoints/neural_weights.npy")

    cursor_path = Path("./data/quran_training_cursor.json")
    cursor = {}
    if cursor_path.exists():
        try:
            with open(cursor_path, encoding="utf-8") as f:
                cursor = json.load(f)
        except Exception:
            cursor = {}

    return {
        "training_steps": db_stats.get("total_items_trained", 0),
        "training_sessions": db_stats.get("completed_sessions", 0),
        "training_by_domain": db_stats.get("by_domain", {}),
        "recent_avg_loss": db_stats.get("recent_avg_loss", None),
        "concepts": ckg.concept_count() if hasattr(ckg, "concept_count") else len(getattr(ckg, "_data", {}).get("concepts", {})),
        "relations": ckg.relation_count() if hasattr(ckg, "relation_count") else len(getattr(ckg, "_data", {}).get("relations", {})),
        "weights_saved": weights_path.exists(),
        "weights_path": str(weights_path) if weights_path.exists() else None,
        "quran_training_cursor": cursor,
    }
