"""
Scientist Manager — العالِم المبتكر والمدير الأمني والمالي التلقائي
================================================================
يجمع:
  1) Research & Discovery  → research_discovery
  2) Cloud Cost Optimization → cloud_cost_optimizer
  3) Red Teaming الدفاعي → model_red_team
  4) CI/CD for AI → ai_cicd

ويتكامل مع المهندس المعماري ومنصات التدريب البعيدة.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger("ScientistManager")

ROOT = Path(__file__).resolve().parent.parent
SCI_DIR = ROOT / "artifacts" / "model_training" / "scientist"
SCI_DIR.mkdir(parents=True, exist_ok=True)


def scientist_status() -> str:
    lines = [
        "## 🧬 العالِم المبتكر + المدير الأمني والمالي",
        "",
        "### القدرات",
        "1. **بحث واكتشاف** — `اكتشف تنشيط` · `دمج نماذج`",
        "2. **إدارة مالية** — `تكلفة تدريب` · `أرخص مسار` · قرار ROI تلقائي",
        "3. **أمن دفاعي** — `red team` · `تحصين نموذج` (محلي فقط)",
        "4. **CI/CD للذكاء** — `ترقية نموذج score=0.9` · `سجل نماذج`",
        "",
        "### طبقات الوكيل",
        "- تدريب وتشغيل → Model Training Agent",
        "- منصات بعيدة → Orchestrator (Kaggle/Colab)",
        "- حكم هندسي → AI Architect",
        "- علم + أمن + مال + نشر → **Scientist Manager** (هذه الطبقة)",
        "",
    ]
    mods = [
        "ai.research_discovery",
        "ai.cloud_cost_optimizer",
        "ai.model_red_team",
        "ai.ai_cicd",
    ]
    lines.append("### الوحدات")
    for m in mods:
        try:
            __import__(m)
            lines.append(f"- `{m}`: ✅")
        except Exception as e:
            lines.append(f"- `{m}`: ❌ {e}")
    return "\n".join(lines)


def run_scientist_cycle() -> str:
    parts = ["## 🧬 دورة العالِم المختصرة", ""]
    try:
        from ai.research_discovery import discover_activations, merge_demo

        parts.append(discover_activations().to_markdown())
        parts.append("")
        parts.append("### دمج تجريبي")
        parts.append(str(merge_demo(alpha=0.4)))
        parts.append("")
    except Exception as e:
        parts.append(f"بحث: {e}")
    try:
        from ai.cloud_cost_optimizer import decide_roi, cheapest_path

        parts.append(decide_roi().to_markdown())
        parts.append("")
        parts.append(cheapest_path(2.0))
        parts.append("")
    except Exception as e:
        parts.append(f"تكلفة: {e}")
    try:
        from ai.model_red_team import run_red_team

        parts.append(run_red_team().to_markdown())
        parts.append("")
    except Exception as e:
        parts.append(f"أمن: {e}")
    try:
        from ai.ai_cicd import register_challenger

        d = register_challenger("cycle_demo", 0.88)
        parts.append(d.to_markdown())
    except Exception as e:
        parts.append(f"CI/CD: {e}")
    report = "\n".join(parts)
    (SCI_DIR / "last_scientist_cycle.md").write_text(report, encoding="utf-8")
    return report


def handle_scientist_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(
        r"(عال[مِ]?\s*مبتكر|scientist|مدير\s*مال|حالة\s*العال|scientist\s*status)",
        text,
        re.I,
    ) or text.lower() in ("scientist", "عالم", "عالِم"):
        return scientist_status()

    if re.search(r"(دورة\s*العال|scientist\s*cycle|شغ[ّ]?ل\s*العال)", text, re.I):
        return run_scientist_cycle()

    for mod, fn_name in (
        ("ai.research_discovery", "handle_research_command"),
        ("ai.cloud_cost_optimizer", "handle_cost_command"),
        ("ai.model_red_team", "handle_security_command"),
        ("ai.ai_cicd", "handle_cicd_command"),
    ):
        try:
            m = __import__(mod, fromlist=[fn_name])
            fn = getattr(m, fn_name)
            r = fn(text)
            if r is not None:
                return r
        except Exception as e:
            logger.warning("%s: %s", mod, e)

    return None
