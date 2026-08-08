"""
AI Red Teaming (دفاعي) — اختبار أمني ذاتي للنماذج
=================================================
الغرض: اكتشاف نقاط الضعف في **نماذجنا** وتحصينها — لا هجوم على أنظمة الغير.

  • تقييم حساسية النموذج لاضطراب المدخلات (FGSM-like مبسّط)
  • محاكاة تسميم بيانات محلي وقياس تدهور الدقة
  • تحصين: إعادة تدريب قصير على عينات خصومية / تنظيف

كل التجارب على بيانات اصطناعية أو مجموعات محلية يملكها المشروع.
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

logger = logging.getLogger("ModelRedTeam")

ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = ROOT / "artifacts" / "model_training" / "scientist" / "security"
SEC_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(z, -40, 40)))


def _train_linear(X: np.ndarray, y: np.ndarray, epochs: int = 50, lr: float = 0.1) -> np.ndarray:
    w = np.zeros(X.shape[1])
    for _ in range(epochs):
        p = _sigmoid(X @ w)
        w -= lr * (X.T @ (p - y)) / max(len(y), 1)
    return w


def _acc(w: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    p = _sigmoid(X @ w)
    return float(((p >= 0.5).astype(np.float64) == y).mean())


@dataclass
class RedTeamReport:
    ok: bool
    clean_acc: float
    adv_acc: float
    poisoned_acc: float
    hardened_acc: float
    attack_success_rate: float
    findings_ar: List[str] = field(default_factory=list)
    harden_steps_ar: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        lines = [
            "## 🛡️ تقرير الأمن السيبراني للنموذج (Red Team دفاعي)",
            f"- دقة نظيفة: **{self.clean_acc:.1%}**",
            f"- تحت اضطراب خصومي: **{self.adv_acc:.1%}**",
            f"- بعد تسميم بيانات محاكي: **{self.poisoned_acc:.1%}**",
            f"- بعد التحصين: **{self.hardened_acc:.1%}**",
            f"- معدل نجاح الهجوم التقريبي: **{self.attack_success_rate:.1%}**",
            "",
            "### اكتشافات",
        ]
        for f in self.findings_ar:
            lines.append(f"- {f}")
        lines += ["", "### تحصين تلقائي مقترح"]
        for s in self.harden_steps_ar:
            lines.append(f"- {s}")
        lines += [
            "",
            "_هذا الاختبار دفاعي على بيانات محلية/اصطناعية فقط — لا يُستخدم ضد أنظمة خارجية._",
        ]
        return "\n".join(lines)


def run_red_team(
    n: int = 500,
    eps: float = 0.15,
    poison_rate: float = 0.1,
    seed: int = 11,
) -> RedTeamReport:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 10))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float64)
    w = _train_linear(X, y)
    clean = _acc(w, X, y)

    # اضطراب خصومي مبسّط باتجاه التدرج
    p = _sigmoid(X @ w)
    grad = (p - y).reshape(-1, 1) * X
    X_adv = X + eps * np.sign(grad)
    adv = _acc(w, X_adv, y)
    asr = max(0.0, clean - adv)

    # تسميم: قلب تسميات جزء من العينات وإعادة تدريب
    y_poison = y.copy()
    n_poison = int(poison_rate * n)
    idx = rng.choice(n, size=n_poison, replace=False)
    y_poison[idx] = 1.0 - y_poison[idx]
    w_p = _train_linear(X, y_poison)
    poisoned = _acc(w_p, X, y)

    # تحصين: تدريب على مزيج نظيف + اضطرابات خفيفة
    X_h = np.vstack([X, X + 0.05 * rng.normal(size=X.shape)])
    y_h = np.concatenate([y, y])
    w_h = _train_linear(X_h, y_h, epochs=60)
    hardened = _acc(w_h, X_adv, y)

    findings = []
    if asr > 0.1:
        findings.append("النموذج حسّاس لاضطراب المدخلات — دقة الخصوم منخفضة بوضوح.")
    else:
        findings.append("مقاومة مقبولة للاضطراب البسيط على هذه المهمة.")
    if clean - poisoned > 0.08:
        findings.append("التسميم المحلي أضرّ بالتعميم — يلزم رصد جودة التسميات في خطوط الإمداد.")
    else:
        findings.append("تأثير التسميم محدود نسبياً على هذا الحجم.")

    harden = [
        "أعد التدريب بعينات مضطربة خفيفاً (adversarial training).",
        "فعّل تدقيق التسميات / outlier detection قبل الدمج في CKG أو CSV.",
        "راقب انحرافاً مفاجئاً في loss على شريحة ثابتة من البيانات النظيفة.",
    ]

    report = RedTeamReport(
        ok=True,
        clean_acc=clean,
        adv_acc=adv,
        poisoned_acc=poisoned,
        hardened_acc=hardened,
        attack_success_rate=asr,
        findings_ar=findings,
        harden_steps_ar=harden,
    )
    path = SEC_DIR / f"redteam_{int(time.time())}.json"
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (path.with_suffix(".md")).write_text(report.to_markdown(), encoding="utf-8")
    return report


def handle_security_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(red\s*team|أمن\s*نموذج|اختراق\s*ذات|تسميم|poison|تحصين|خصوم|adversar)",
        text,
        re.I,
    ):
        return None
    r = run_red_team()
    return r.to_markdown()
