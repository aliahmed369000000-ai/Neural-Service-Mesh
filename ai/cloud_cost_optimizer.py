"""
Cloud Cost Optimizer — إدارة مالية ذاتية للتدريب
================================================
  • تقدير تكلفة ساعة GPU عبر مزوّدين (أسعار مرجعية + تحديث اختياري)
  • قرار عائد/تكلفة: هل زيادة الميزانية تبرَّر بتحسن الدقة المتوقع؟
  • توصية Spot / أرخص مسار (Kaggle المجاني → Spot → On-demand)

لا ينفّذ شراء سحابي فعلي بدون مفاتيح — يصدر قراراً وتوصية قابلة للتنفيذ.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("CloudCostOptimizer")

ROOT = Path(__file__).resolve().parent.parent
COST_DIR = ROOT / "artifacts" / "model_training" / "scientist" / "cost"
COST_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# أسعار مرجعية تقريبية USD/hour (تُحدَّث يدوياً أو عبر API لاحق)
REFERENCE_GPU_PRICES: Dict[str, Dict[str, float]] = {
    "kaggle_t4": {"on_demand": 0.0, "spot": 0.0, "note": "حصة مجانية أسبوعية محدودة"},
    "colab_t4": {"on_demand": 0.0, "spot": 0.0, "note": "مجاني متقطع / Colab Pro منفصل"},
    "vast_rtx3090": {"on_demand": 0.25, "spot": 0.15, "note": "تقريبي سوقي"},
    "aws_g4dn_xlarge": {"on_demand": 0.526, "spot": 0.16, "note": "T4 تقريبي"},
}


@dataclass
class CostDecision:
    ok: bool
    action: str  # continue | stop | switch_spot | use_free_tier
    reason_ar: str
    estimated_cost_usd: float
    expected_acc_gain: float
    cost_per_point: float
    recommendations: List[str] = field(default_factory=list)
    price_table: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        lines = [
            "## 💰 قرار التكلفة / العائد",
            f"- القرار: **{self.action}**",
            f"- تكلفة مقدّرة: **${self.estimated_cost_usd:.3f}**",
            f"- تحسن دقة متوقع: **{self.expected_acc_gain:.3%}**",
            f"- تكلفة لكل نقطة دقة: **${self.cost_per_point:.2f}**" if self.cost_per_point < 1e6 else "- تكلفة/نقطة: غير معرّفة",
            f"- السبب: {self.reason_ar}",
            "",
            "### توصيات",
        ]
        for r in self.recommendations:
            lines.append(f"- {r}")
        lines += ["", "### أسعار مرجعية (USD/ساعة)"]
        for k, v in self.price_table.items():
            lines.append(f"- `{k}`: on_demand=${v.get('on_demand')} spot=${v.get('spot')} — {v.get('note', '')}")
        return "\n".join(lines)


def estimate_training_cost(
    gpu_key: str = "vast_rtx3090",
    hours: float = 2.0,
    use_spot: bool = True,
) -> Dict[str, float]:
    row = REFERENCE_GPU_PRICES.get(gpu_key, {"on_demand": 0.5, "spot": 0.3})
    rate = float(row["spot"] if use_spot else row["on_demand"])
    return {
        "gpu": gpu_key,
        "hours": hours,
        "rate_usd_h": rate,
        "total_usd": rate * hours,
        "use_spot": use_spot,
    }


def decide_roi(
    current_acc: float = 0.82,
    expected_acc: float = 0.83,
    extra_budget_usd: float = 5.0,
    min_gain_per_dollar: float = 0.002,
    prefer_free: bool = True,
) -> CostDecision:
    """
    إن كان التحسن لكل دولار أقل من العتبة → أوقف.
    """
    gain = max(0.0, expected_acc - current_acc)
    cpp = (extra_budget_usd / gain) if gain > 1e-9 else 1e9
    gain_per_dollar = gain / max(extra_budget_usd, 1e-9)
    table = dict(REFERENCE_GPU_PRICES)

    recs: List[str] = []
    if prefer_free:
        recs.append("ابدأ بـ Kaggle Dual T4 / Colab ضمن الحصة المجانية قبل أي إنفاق.")
    # أرخص spot
    spot_opts = sorted(
        ((k, v["spot"]) for k, v in REFERENCE_GPU_PRICES.items() if v["spot"] > 0),
        key=lambda x: x[1],
    )
    if spot_opts:
        recs.append(f"أرخص Spot مرجعي الآن: `{spot_opts[0][0]}` ≈ ${spot_opts[0][1]}/ساعة.")

    if gain_per_dollar < min_gain_per_dollar:
        action = "stop"
        reason = (
            f"صرف ${extra_budget_usd:.2f} لربح دقة {gain:.2%} غير مجدٍ "
            f"(عتبة {min_gain_per_dollar:.3%} لكل دولار)."
        )
        recs.append("أوقف التدريب الإضافي أو خفّض epochs / استخدم ضغط النموذج بدل الميزانية.")
    elif prefer_free and gain < 0.02:
        action = "use_free_tier"
        reason = "التحسن المتوقع محدود — استنفد الحصة المجانية أولاً."
    else:
        action = "switch_spot" if spot_opts else "continue"
        reason = (
            f"التحسن {gain:.2%} مقابل ${extra_budget_usd:.2f} يفوق العتبة المالية. "
            "فضّل Spot لتقليل التكلفة."
        )
        recs.append("راقب السعر كل ساعة؛ إن ارتفع Spot > on_demand×0.7 انتقل أو أوقف.")

    d = CostDecision(
        ok=True,
        action=action,
        reason_ar=reason,
        estimated_cost_usd=float(extra_budget_usd),
        expected_acc_gain=float(gain),
        cost_per_point=float(cpp if gain > 1e-9 else 0.0),
        recommendations=recs,
        price_table=table,
    )
    path = COST_DIR / f"cost_decision_{int(time.time())}.json"
    path.write_text(json.dumps(asdict(d), ensure_ascii=False, indent=2), encoding="utf-8")
    (path.with_suffix(".md")).write_text(d.to_markdown(), encoding="utf-8")
    return d


def cheapest_path(hours: float = 3.0) -> str:
    rows = []
    for k, v in REFERENCE_GPU_PRICES.items():
        rate = min(v["on_demand"], v["spot"]) if v["on_demand"] or v["spot"] else 0.0
        rows.append((k, rate * hours, rate, v.get("note", "")))
    rows.sort(key=lambda x: x[1])
    lines = ["## 📉 أرخص مسارات التدريب", ""]
    for k, total, rate, note in rows[:8]:
        lines.append(f"- `{k}`: ${total:.2f} لـ {hours}h (rate=${rate}/h) — {note}")
    lines.append("")
    lines.append("الوكيل يوصي: **مجاني (Kaggle/Colab) → Spot رخيص → On-demand**.")
    return "\n".join(lines)


def handle_cost_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(
        r"(تكلفة|cost|ميزاني|spot|عائد|roi|أرخص|سعر\s*gpu|مالية\s*تدريب|مسار\s*رخيص)",
        text,
        re.I,
    ):
        return None
    if re.search(r"(أرخص|cheapest|مسار\s*رخيص)", text, re.I):
        return cheapest_path()
    # استخراج أرقام اختيارية
    budget = 5.0
    cur, exp = 0.82, 0.83
    m = re.search(r"ميزاني[ة]?\s*[=:]?\s*(\d+(?:\.\d+)?)", text)
    if m:
        budget = float(m.group(1))
    m = re.search(r"(\d+(?:\.\d+)?)\s*\$", text)
    if m:
        budget = float(m.group(1))
    m = re.search(r"دقة\s*[=:]?\s*(\d+(?:\.\d+)?)", text)
    # قرار افتراضي
    d = decide_roi(current_acc=cur, expected_acc=exp, extra_budget_usd=budget)
    return d.to_markdown()
