"""
AI Research & Discovery — ابتكار خوارزميات ودمج النماذج
=======================================================
  • بحث دوال تنشيط / معادلات خسارة معدَّلة عبر محاكاة سريعة
  • Model Merging: دمج أوزان نموذجين (متوسط / task-arithmetic خفيف)
بدون ادعاء اكتشاف علمي منشور — مسار تجريبي منهجي قابل للتكرار.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("ResearchDiscovery")

ROOT = Path(__file__).resolve().parent.parent
RES_DIR = ROOT / "artifacts" / "model_training" / "scientist" / "research"
RES_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── دوال تنشيط مرشّحة للبحث ───────────────────────────────────────────────

def act_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def act_gelu(x: np.ndarray) -> np.ndarray:
    # تقريب tanh لـ GELU
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def act_swish(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40, 40)))


def act_mish(x: np.ndarray) -> np.ndarray:
    sp = np.log1p(np.exp(np.clip(x, -40, 40)))
    return x * np.tanh(sp)


def act_nsm_softpeak(x: np.ndarray, a: float = 1.2) -> np.ndarray:
    """تنشيط تجريبي: مزيج swish + حدّ ناعم — مرشّح للاختبار لا اكتشاف منشور."""
    s = x / (1.0 + np.exp(-np.clip(a * x, -40, 40)))
    return s + 0.05 * np.tanh(x)


ACTIVATIONS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "relu": act_relu,
    "gelu": act_gelu,
    "swish": act_swish,
    "mish": act_mish,
    "nsm_softpeak": act_nsm_softpeak,
}


def _mlp_train_eval(
    act_fn: Callable[[np.ndarray], np.ndarray],
    n: int = 600,
    epochs: int = 40,
    lr: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 8))
    y = (X[:, 0] * 0.7 + np.sin(X[:, 1]) > 0).astype(np.float64)
    W1 = rng.normal(0, 0.2, size=(8, 16))
    b1 = np.zeros(16)
    W2 = rng.normal(0, 0.2, size=(16,))
    b2 = 0.0
    for _ in range(epochs):
        h = act_fn(X @ W1 + b1)
        z = h @ W2 + b2
        p = 1 / (1 + np.exp(-np.clip(z, -30, 30)))
        err = (p - y) / n
        # تدرج تقريبي عبر perturbation بسيط على الطبقة الأخيرة
        dW2 = h.T @ err
        db2 = float(err.sum())
        # تدرج طبقة أولى تقريبي
        dh = np.outer(err, W2)
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)
        W2 -= lr * dW2
        b2 -= lr * db2
        W1 -= lr * dW1
        b1 -= lr * db1
    h = act_fn(X @ W1 + b1)
    p = 1 / (1 + np.exp(-np.clip(h @ W2 + b2, -30, 30)))
    acc = float(((p >= 0.5) == y).mean())
    loss = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
    return {"accuracy": acc, "loss": loss}


@dataclass
class DiscoveryReport:
    ok: bool
    ranking: List[Dict[str, Any]]
    best_activation: str
    best_score: float
    notes_ar: str
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        lines = [
            "## 🔬 اكتشاف بحثي — دوال التنشيط / محاكاة",
            f"- الأفضل: **{self.best_activation}** (score={self.best_score:.4f})",
            "",
            "### الترتيب",
        ]
        for i, r in enumerate(self.ranking, 1):
            lines.append(
                f"{i}. `{r['name']}` — acc={r['accuracy']:.3f} loss={r['loss']:.4f}"
            )
        lines += ["", self.notes_ar]
        return "\n".join(lines)


def discover_activations(seed: int = 0) -> DiscoveryReport:
    ranking = []
    for name, fn in ACTIVATIONS.items():
        m = _mlp_train_eval(fn, seed=seed)
        ranking.append({"name": name, **m, "score": m["accuracy"] - 0.1 * m["loss"]})
    ranking.sort(key=lambda x: x["score"], reverse=True)
    best = ranking[0]
    notes = (
        f"المحاكاة تفضّل `{best['name']}` على هذه المهمة الاصطناعية. "
        "هذا مسار تجريبي داخلي — أي اعتماد إنتاجي يتطلب معياراً أوسع وإعادة تدريب كاملة. "
        "يمكن دمج الدالة الفائزة في دورة البحث الفائق التالية."
    )
    report = DiscoveryReport(
        ok=True,
        ranking=ranking,
        best_activation=best["name"],
        best_score=float(best["score"]),
        notes_ar=notes,
    )
    out = RES_DIR / f"activation_search_{int(time.time())}.json"
    out.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (out.with_suffix(".md")).write_text(report.to_markdown(), encoding="utf-8")
    return report


def merge_state_dicts(
    a: Dict[str, np.ndarray],
    b: Dict[str, np.ndarray],
    alpha: float = 0.5,
    mode: str = "linear",
) -> Dict[str, np.ndarray]:
    """
    دمج أوزان:
      linear: (1-α)·A + α·B
      task_arith: A + α·(B-A)  (إزاحة مهمة مبسطة)
    """
    alpha = float(np.clip(alpha, 0.0, 1.0))
    keys = [k for k in a.keys() if k in b and np.shape(a[k]) == np.shape(b[k])]
    out: Dict[str, np.ndarray] = {}
    for k in keys:
        wa, wb = np.asarray(a[k], dtype=np.float64), np.asarray(b[k], dtype=np.float64)
        if mode == "task_arith":
            out[k] = (wa + alpha * (wb - wa)).astype(np.float32)
        else:
            out[k] = ((1 - alpha) * wa + alpha * wb).astype(np.float32)
    return out


def merge_demo(alpha: float = 0.5, mode: str = "linear") -> Dict[str, Any]:
    rng = np.random.default_rng(3)
    a = {"W": rng.normal(size=(32, 16)).astype(np.float32), "b": rng.normal(size=(16,)).astype(np.float32)}
    b = {"W": a["W"] + rng.normal(scale=0.05, size=a["W"].shape).astype(np.float32),
         "b": a["b"] + rng.normal(scale=0.02, size=a["b"].shape).astype(np.float32)}
    merged = merge_state_dicts(a, b, alpha=alpha, mode=mode)
    # مقياس تقارب
    delta = float(np.mean((merged["W"] - a["W"]) ** 2))
    out = RES_DIR / f"merged_{mode}_{int(time.time())}.npz"
    np.savez_compressed(out, **merged)
    report = {
        "ok": True,
        "mode": mode,
        "alpha": alpha,
        "keys": list(merged.keys()),
        "mse_from_a": delta,
        "output": str(out.relative_to(ROOT)),
        "narrative_ar": (
            f"دُمجت أوزان A/B بوضع `{mode}` وα={alpha}. "
            "الدمج يوفّر نموذجاً هجيناً دون إعادة تدريب كامل — يُنصح بتقييم سريع ثم fine-tune خفيف."
        ),
    }
    (RES_DIR / f"merge_report_{int(time.time())}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report


def handle_research_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(اكتشف|ابحث\s*تنشيط|activation\s*search|دوال\s*تنشيط|ابتكار\s*خوارزم)", text, re.I):
        r = discover_activations()
        return r.to_markdown()
    if re.search(r"(دمج\s*نماذج|model\s*merg|ادمج\s*الأوزان|دمج\s*أدمغة)", text, re.I):
        mode = "task_arith" if re.search(r"task|إزاحة", text, re.I) else "linear"
        alpha = 0.5
        m = re.search(r"alpha\s*[=:]?\s*([0-9.]+)|α\s*[=:]?\s*([0-9.]+)", text, re.I)
        if m:
            alpha = float(m.group(1) or m.group(2))
        rep = merge_demo(alpha=alpha, mode=mode)
        return (
            "## 🧠 دمج النماذج (Model Merging)\n\n"
            + "```json\n"
            + json.dumps(rep, ensure_ascii=False, indent=2)
            + "\n```"
        )
    return None
