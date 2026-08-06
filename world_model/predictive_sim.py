"""
Predictive Simulation — محاكاة عواقب قبل النشر/الإنفاق
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent


def simulate_social_reaction(text: str, platform: str = "twitter") -> Dict[str, Any]:
    """تقدير تفاعل بناءً على طول النص، إشارات إيجابية/سلبية، وتاريخ تجميد."""
    text = text or ""
    pos = len(re.findall(r"(خير|أمل|علم|فائدة|شكر|نفع)", text))
    neg = len(re.findall(r"(كره|شتيم|سب|هجوم|فشل)", text))
    length = len(text)
    base = 0.45
    base += min(0.25, pos * 0.05)
    base -= min(0.35, neg * 0.08)
    if 40 <= length <= 280:
        base += 0.1
    if length > 600:
        base -= 0.05
    # منصة
    if platform in ("linkedin", "telegram"):
        base += 0.05
    if platform in ("tiktok", "instagram") and length < 80:
        base += 0.05
    try:
        from ai.social_swarm import pre_publish_check
        chk = pre_publish_check(text)
        if not chk.get("ok"):
            base = min(base, 0.15)
            risk = "crisis_gate"
        else:
            risk = "ok"
    except Exception:
        risk = "unchecked"
    engagement = float(max(0.05, min(0.95, base)))
    return {
        "platform": platform,
        "predicted_engagement": round(engagement, 3),
        "risk": risk,
        "recommend_publish": engagement >= 0.35 and risk == "ok",
        "at": datetime.now(timezone.utc).isoformat(),
    }


def simulate_budget(gpu_hours: float = 2.0, rate_usd: float = 0.35, expected_acc_gain: float = 0.01) -> Dict[str, Any]:
    cost = float(gpu_hours) * float(rate_usd)
    # قيمة تقريبية لكل نقطة دقة
    value = expected_acc_gain * 50.0  # $50 per full accuracy point heuristic
    roi = value - cost
    return {
        "gpu_hours": gpu_hours,
        "cost_usd": round(cost, 3),
        "expected_acc_gain": expected_acc_gain,
        "estimated_value_usd": round(value, 3),
        "roi_usd": round(roi, 3),
        "decision": "activate_servers" if roi > 0 else "cancel_or_use_free_tier",
        "at": datetime.now(timezone.utc).isoformat(),
    }


def full_campaign_sim(text: str, platforms: Optional[list] = None) -> Dict[str, Any]:
    platforms = platforms or ["twitter", "linkedin", "telegram"]
    per = {p: simulate_social_reaction(text, p) for p in platforms}
    avg = sum(v["predicted_engagement"] for v in per.values()) / max(len(per), 1)
    budget = simulate_budget(1.5, 0.3, 0.008)
    return {
        "avg_engagement": round(avg, 3),
        "platforms": per,
        "budget": budget,
        "go": avg >= 0.35 and budget["decision"] == "activate_servers" or avg >= 0.4,
    }


def handle_predictive_command(user_input: str):
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(محاكاة\s*حملة|simulate\s*campaign|تنبؤ\s*تفاعل|محاكاة\s*ميزاني)", text, re.I):
        return None
    m = re.search(r"(?:نص|text)[:\s]+(.+)$", text, re.I)
    body = m.group(1).strip() if m else "منشور معرفي عن الأمانة والعلم"
    if re.search(r"ميزاني", text):
        return "## 💰 محاكاة ميزانية\n```json\n" + json.dumps(simulate_budget(), ensure_ascii=False, indent=2) + "\n```"
    r = full_campaign_sim(body)
    return "## 🔮 محاكاة حملة / نموذج العالم\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2) + "\n```"
