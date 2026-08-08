"""
Reasoning Traces — تفكير وتحليل عميق قبل/بعد القرار
===================================================
  • Tree-of-Thoughts خفيف: 3–5 سيناريوهات هيكلية مع تخيل الفشل
  • Self-Reflection بعد الفشل: مراجعة تُحفظ في الذاكرة المستمرة
  • أثر قرار (decision trace) قابل للقراءة بالعربية
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ReasoningTraces")

ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = ROOT / "artifacts" / "model_training" / "meta_ai" / "traces"
TRACE_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Scenario:
    name: str
    architecture: str
    risk: str
    mitigation: str
    expected_acc: float
    cost_hint: str


@dataclass
class ReasoningReport:
    ok: bool
    goal: str
    scenarios: List[Dict[str, Any]]
    chosen: Dict[str, Any]
    chain_ar: List[str]
    reflection_ar: str = ""
    created_at: str = field(default_factory=_now)

    def to_markdown(self) -> str:
        lines = [
            f"## 🧠 أثر تفكير عميق — {self.goal}",
            "",
            "### سلسلة التفكير (CoT)",
        ]
        for i, s in enumerate(self.chain_ar, 1):
            lines.append(f"{i}. {s}")
        lines += ["", "### سيناريوهات (Tree of Thoughts)"]
        for i, sc in enumerate(self.scenarios, 1):
            lines.append(
                f"**{i}. {sc['name']}** — `{sc['architecture']}`\n"
                f"   - خطر: {sc['risk']}\n"
                f"   - تخفيف: {sc['mitigation']}\n"
                f"   - دقة متوقعة: {sc['expected_acc']:.0%} · تكلفة: {sc['cost_hint']}"
            )
        lines += [
            "",
            "### القرار",
            f"- المختار: **{self.chosen.get('name')}** (`{self.chosen.get('architecture')}`)",
            f"- السبب: {self.chosen.get('why_ar', '')}",
        ]
        if self.reflection_ar:
            lines += ["", "### نقد ذاتي", self.reflection_ar]
        return "\n".join(lines)


def plan_architectures(goal: str = "تصنيف جدولي متوسط", n: int = 5) -> ReasoningReport:
    """يبني عدة سيناريوهات هيكلية قبل التدريب."""
    library = [
        Scenario(
            "شبكة ضيقة سريعة",
            "MLP 2×64",
            "تحت-ملاءمة على إشارات معقّدة",
            "زد العرض إن بقيت الخسارة عالية بعد 5 عصور",
            0.78,
            "منخفض (CPU/T4 دقائق)",
        ),
        Scenario(
            "شبكة متوازنة",
            "MLP 3×128 + Dropout0.1",
            "OOM على GPU ضعيف إن كبر الـbatch",
            "batch تكيفي + AMP",
            0.86,
            "متوسط",
        ),
        Scenario(
            "عميقة ثقيلة",
            "MLP 5×256",
            "فرط ملاءمة + بطء",
            "weight decay + early stopping",
            0.84,
            "مرتفع",
        ),
        Scenario(
            "دمج نموذجين",
            "Merge(A,B) α=0.4",
            "تعارض أوزان إن تباعدت المهام",
            "fine-tune خفيف بعد الدمج",
            0.88,
            "منخفض بعد وجود A/B",
        ),
        Scenario(
            "اتحادي خاص",
            "FedAvg 5 clients",
            "non-IID يبطئ الاندماج",
            "مزيد من الجولات + مشاركة 80% عملاء",
            0.80,
            "متوسط موزّع",
        ),
        Scenario(
            "NAS جيني قصير",
            "Neuroevolution 4 gens",
            "تكلفة تجارب متعددة",
            "ابدأ على Kaggle المجاني",
            0.90,
            "مرتفع زمنياً / رخيص مالياً إن مجاني",
        ),
    ]
    scenarios = library[: max(3, min(n, len(library)))]
    # اختيار بسيط: أعلى دقة متوقعة مع تكلفة غير مرتفعة
    ranked = sorted(
        scenarios,
        key=lambda s: s.expected_acc - (0.05 if "مرتفع" in s.cost_hint else 0.0),
        reverse=True,
    )
    chosen = ranked[0]
    chain = [
        f"الهدف: {goal} — نحتاج توازناً بين الدقة والتكلفة ومخاطر الفشل.",
        f"رُسمت {len(scenarios)} سيناريوهات هيكلية مع تخيل فشل كل مسار.",
        f"استُبعد ما تكلفته مرتفعة دون هامش دقة واضح (مثل العميقة الثقيلة إن وُجد بديل).",
        f"المسار المختار «{chosen.name}» لأن دقته المتوقعة {chosen.expected_acc:.0%} مع مخاطر قابلة للتخفيف: {chosen.mitigation}.",
        "الخطوة التالية: بحث فائق ضيق حول هذا الهيكل ثم تدريب بعيد عند الحاجة.",
    ]
    report = ReasoningReport(
        ok=True,
        goal=goal,
        scenarios=[asdict(s) for s in scenarios],
        chosen={**asdict(chosen), "why_ar": chain[3]},
        chain_ar=chain,
    )
    path = TRACE_DIR / f"plan_{int(time.time())}.json"
    path.write_text(json.dumps(asdict(report), ensure_ascii=False, indent=2), encoding="utf-8")
    (path.with_suffix(".md")).write_text(report.to_markdown(), encoding="utf-8")
    # ذاكرة مستمرة إن توفّرت
    try:
        from ai.persistent_memory import remember_experience

        remember_experience(
            kind="reasoning_plan",
            text=f"خطة لـ {goal}: اخترنا {chosen.architecture} — {chosen.mitigation}",
            meta={"goal": goal, "arch": chosen.architecture},
        )
    except Exception:
        pass
    return report


def reflect_on_failure(
    failure_summary: str,
    context: str = "",
) -> ReasoningReport:
    """نقد ذاتي بعد فشل تدريب — يُحدّث الذاكرة طويلة المدى."""
    causes = []
    fs = (failure_summary or "").lower()
    if "oom" in fs or "ذاكر" in failure_summary:
        causes.append("نفاد ذاكرة GPU — batch كبير أو نموذج أوسع من الجهاز.")
    if "loss" in fs and ("nan" in fs or "inf" in fs):
        causes.append("انفجار تدرجات — معدل تعلم مرتفع أو بيانات غير مطبّعة.")
    if "overfit" in fs or "فرط" in failure_summary:
        causes.append("فرط ملاءمة — نقص تنظيم أو بيانات قليلة.")
    if not causes:
        causes.append("سبب غير مصنّف — راجع السجلات والمقاييس والعتاد.")

    lessons = [
        "خفّض batch أو فعّل AMP عند ضغط VRAM.",
        "ابدأ بـ lr أصغر مع cosine annealing.",
        "احفظ أثر هذا الفشل في الذاكرة المتجهة لتجنّب تكراره.",
    ]
    reflection = (
        "لماذا فشلنا كمنظومة؟\n"
        + "\n".join(f"- {c}" for c in causes)
        + "\n\nدروس دائمة:\n"
        + "\n".join(f"- {l}" for l in lessons)
    )
    chain = [
        "رُصد فشل في دورة التدريب.",
        "حُلِّلت الأعراض مقابل أنماط معروفة (OOM، NaN، فرط ملاءمة).",
        "كُتبت مراجعة ذاتية وخُزّنت كخبرة دائمة.",
    ]
    report = ReasoningReport(
        ok=True,
        goal="post-failure reflection",
        scenarios=[],
        chosen={"name": "إصلاح", "architecture": "policy_update", "why_ar": causes[0]},
        chain_ar=chain,
        reflection_ar=reflection,
    )
    path = TRACE_DIR / f"reflect_{int(time.time())}.json"
    path.write_text(
        json.dumps({**asdict(report), "context": context[:2000]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (path.with_suffix(".md")).write_text(report.to_markdown(), encoding="utf-8")
    try:
        from ai.persistent_memory import remember_experience

        remember_experience(
            kind="failure_lesson",
            text=reflection[:1500],
            meta={"summary": failure_summary[:500]},
        )
    except Exception:
        pass
    return report


def handle_reasoning_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(نقد\s*ذاتي|لماذا\s*فشل|reflect|مراجع[ةه]\s*فشل)", text, re.I):
        # استخرج ملخصاً بعد : أو —
        m = re.search(r"(?:فشل|failure|لأن)[:\s\-]*(.+)$", text, re.I)
        summary = m.group(1).strip() if m else text
        return reflect_on_failure(summary).to_markdown()
    if re.search(
        r"(فكر\s*عميق|سيناريو|tree\s*of\s*thought|chain\s*of\s*thought|خطط\s*هيكلي|reasoning)",
        text,
        re.I,
    ):
        goal = "تصنيف جدولي متوسط"
        m = re.search(r"(?:ل|goal|هدف)[:\s]+(.+)$", text, re.I)
        if m:
            goal = m.group(1).strip()[:200]
        elif "ل" in text:
            parts = re.split(r"\s+ل\s+", text, maxsplit=1)
            if len(parts) == 2:
                goal = parts[1].strip()[:200]
        return plan_architectures(goal=goal).to_markdown()
    return None
