"""
Federated Learning — تدريب موزّع مع الحفاظ على الخصوصية
======================================================
محاكاة FedAvg:
  • كل عميل يدرّب محلياً على بياناته الخاصة (لا تُرفع العينات)
  • يُرسل تحديثات الأوزان فقط إلى المنسّق
  • المنسّق يدمج المتوسط المرجّح ويعيد النموذج العام

مناسب لسيناريوهات الطب/البنوك حيث تبقى البيانات على الأجهزة الطرفية.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("FederatedLearning")

ROOT = Path(__file__).resolve().parent.parent
FED_DIR = ROOT / "artifacts" / "model_training" / "architect" / "federated"
FED_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ClientUpdate:
    client_id: str
    n_samples: int
    weights: Dict[str, np.ndarray]
    local_loss: float
    local_acc: float


@dataclass
class FederatedReport:
    ok: bool
    n_clients: int
    rounds: int
    global_acc_history: List[float]
    final_acc: float
    privacy_note_ar: str
    created_at: str = field(default_factory=_now)
    output_path: str = ""

    def to_markdown(self) -> str:
        hist = ", ".join(f"{a:.3f}" for a in self.global_acc_history[-8:])
        return "\n".join(
            [
                "## 🔐 تقرير التعلم الموحّد (Federated Learning)",
                f"- العملاء: **{self.n_clients}** | الجولات: **{self.rounds}**",
                f"- دقة النموذج العام النهائية: **{self.final_acc:.1%}**",
                f"- مسار الدقة: {hist}",
                f"- المخرج: `{self.output_path}`",
                "",
                "### الخصوصية",
                self.privacy_note_ar,
            ]
        )


def _init_weights(d_in: int = 12, width: int = 24) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "W1": rng.normal(0, 0.1, size=(d_in, width)).astype(np.float64),
        "b1": np.zeros(width),
        "W2": rng.normal(0, 0.1, size=(width, 1)).astype(np.float64),
        "b2": np.zeros(1),
    }


def _forward(X: np.ndarray, w: Dict[str, np.ndarray]) -> np.ndarray:
    h = np.tanh(X @ w["W1"] + w["b1"])
    z = h @ w["W2"] + w["b2"]
    return 1 / (1 + np.exp(-np.clip(z.ravel(), -30, 30)))


def _local_train(
    X: np.ndarray,
    y: np.ndarray,
    w: Dict[str, np.ndarray],
    epochs: int = 8,
    lr: float = 0.05,
) -> Tuple[Dict[str, np.ndarray], float, float]:
    """SGD محلي — لا يغادر X,y الجهاز."""
    w = {k: v.copy() for k, v in w.items()}
    n = len(y)
    for _ in range(epochs):
        p = _forward(X, w)
        err = (p - y) / max(n, 1)
        h = np.tanh(X @ w["W1"] + w["b1"])
        # تدرجات تقريبية
        dW2 = h.T @ err.reshape(-1, 1)
        db2 = err.sum(keepdims=True)
        dh = np.outer(err, w["W2"].ravel()) * (1 - h**2)
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)
        w["W2"] -= lr * dW2
        w["b2"] -= lr * db2.ravel()
        w["W1"] -= lr * dW1
        w["b1"] -= lr * db1
    p = _forward(X, w)
    loss = float(-(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9)).mean())
    acc = float(((p >= 0.5).astype(np.float64) == y).mean())
    return w, loss, acc


def _fedavg(updates: List[ClientUpdate]) -> Dict[str, np.ndarray]:
    total = sum(u.n_samples for u in updates) or 1
    keys = updates[0].weights.keys()
    avg = {}
    for k in keys:
        acc = None
        for u in updates:
            piece = u.weights[k] * (u.n_samples / total)
            acc = piece if acc is None else acc + piece
        avg[k] = acc
    return avg


def run_federated(
    n_clients: int = 5,
    rounds: int = 6,
    samples_per_client: int = 120,
    local_epochs: int = 6,
    seed: int = 7,
) -> FederatedReport:
    rng = np.random.default_rng(seed)
    d_in = 12
    global_w = _init_weights(d_in=d_in)
    # بيانات غير متجانسة لكل عميل (non-IID خفيف)
    client_data = []
    for c in range(n_clients):
        bias = rng.normal(scale=0.3, size=d_in)
        X = rng.normal(size=(samples_per_client, d_in)) + bias
        y = (X[:, 0] + 0.4 * X[:, 1] + rng.normal(scale=0.1, size=samples_per_client) > 0).astype(np.float64)
        client_data.append((X, y))

    # مجموعة اختبار مركزية للتقييم فقط (في الواقع تُستبدل بتقييم محلي)
    X_test = rng.normal(size=(400, d_in))
    y_test = (X_test[:, 0] + 0.4 * X_test[:, 1] > 0).astype(np.float64)

    history = []
    for r in range(rounds):
        updates: List[ClientUpdate] = []
        # مشاركة جزئية للعملاء (simulates availability)
        active = list(range(n_clients))
        rng.shuffle(active)
        active = active[: max(2, int(0.8 * n_clients))]
        for ci in active:
            X, y = client_data[ci]
            local_w, loss, acc = _local_train(X, y, global_w, epochs=local_epochs)
            updates.append(
                ClientUpdate(
                    client_id=f"client_{ci}",
                    n_samples=len(y),
                    weights=local_w,
                    local_loss=loss,
                    local_acc=acc,
                )
            )
            # لا نُبقي X,y في التحديث — الخصوصية
        global_w = _fedavg(updates)
        p = _forward(X_test, global_w)
        gacc = float(((p >= 0.5).astype(np.float64) == y_test).mean())
        history.append(gacc)
        logger.info("fed round %s acc=%.3f clients=%s", r + 1, gacc, len(updates))

    out = FED_DIR / f"fedavg_{int(time.time())}.npz"
    np.savez(out, **global_w)
    report = FederatedReport(
        ok=True,
        n_clients=n_clients,
        rounds=rounds,
        global_acc_history=history,
        final_acc=history[-1] if history else 0.0,
        output_path=str(out.relative_to(ROOT)),
        privacy_note_ar=(
            "لم تُنقل عيّنات خام بين العملاء والمنسّق — فقط أوزان محدّثة. "
            "هذا يقلّل خطر تسريب البيانات الحساسة (طب/بنوك). "
            "للإنتاج يُفضل إضافة Diffie-Hellman/secure aggregation وضوضاء تفاضلية."
        ),
    )
    meta = {
        **{k: v for k, v in asdict(report).items() if k != "privacy_note_ar"},
        "privacy_note_ar": report.privacy_note_ar,
        "history": history,
    }
    (FED_DIR / f"fed_report_{int(time.time())}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (FED_DIR / f"fed_report_{int(time.time())}.md").write_text(report.to_markdown(), encoding="utf-8")
    return report


def handle_federated_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(اتحاد\w*|federat\w*|خصوصي\w*|fedavg|تدريب\s*موز\w*|تدريب\s*اتحاد\w*|بدون\s*رفع\s*بيانات)",
        text,
        re.I,
    ):
        return None
    n_clients, rounds = 5, 6
    m = re.search(r"(\d+)\s*(?:عملاء|عميل|clients?)", text, re.I)
    if m:
        n_clients = max(2, min(20, int(m.group(1))))
    m2 = re.search(r"(\d+)\s*(?:جولات|جولة|rounds?)", text, re.I)
    if m2:
        rounds = max(2, min(30, int(m2.group(1))))
    report = run_federated(n_clients=n_clients, rounds=rounds)
    return report.to_markdown()
