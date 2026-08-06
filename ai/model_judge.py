"""
Model Judge — LLM-as-a-Judge لتقييم نماذج التدريب
=================================================
يبني تقريراً بشرياً مفهوماً عن جودة نموذج مدرَّب:
  • مقاييس كمية (loss / accuracy / calibration تقريبية)
  • حكم نوعي عبر free_router إن توفّر، وإلا محكّم محلي قائم على قواعد
  • توصيات لدورة التدريب التالية
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger("ModelJudge")

ROOT = Path(__file__).resolve().parent.parent
JUDGE_DIR = ROOT / "artifacts" / "model_training" / "architect" / "judgements"
JUDGE_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JudgeReport:
    ok: bool
    model_id: str
    scores: Dict[str, float] = field(default_factory=dict)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    next_cycle_plan: List[str] = field(default_factory=list)
    narrative_ar: str = ""
    judge_backend: str = "local"
    created_at: str = field(default_factory=_now)
    raw_llm: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        lines = [
            f"## ⚖️ تقرير التحكيم — `{self.model_id}`",
            f"- الوقت: {self.created_at}",
            f"- الحَكَم: **{self.judge_backend}**",
            "",
            "### المقاييس",
        ]
        for k, v in self.scores.items():
            lines.append(f"- **{k}**: {v:.4f}" if isinstance(v, float) else f"- **{k}**: {v}")
        if self.strengths:
            lines += ["", "### نقاط القوة"]
            lines += [f"- {s}" for s in self.strengths]
        if self.weaknesses:
            lines += ["", "### نقاط الضعف"]
            lines += [f"- {w}" for w in self.weaknesses]
        if self.next_cycle_plan:
            lines += ["", "### خطة الدورة التالية"]
            lines += [f"{i+1}. {p}" for i, p in enumerate(self.next_cycle_plan)]
        if self.narrative_ar:
            lines += ["", "### الملخص", self.narrative_ar]
        return "\n".join(lines)


def _local_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """مقاييس بدون مكتبات ثقيلة."""
    y_true = np.asarray(y_true).astype(int).ravel()
    y_prob = np.asarray(y_prob)
    if y_prob.ndim == 1:
        # احتمال صنف 1
        pred = (y_prob >= 0.5).astype(int)
        conf = np.clip(np.where(pred == 1, y_prob, 1 - y_prob), 1e-6, 1 - 1e-6)
        acc = float((pred == y_true).mean()) if len(y_true) else 0.0
        # logloss ثنائي تقريبي
        p1 = np.clip(y_prob, 1e-6, 1 - 1e-6)
        ll = float(-(y_true * np.log(p1) + (1 - y_true) * np.log(1 - p1)).mean())
        ece = float(np.abs(conf.mean() - acc))  # تبسيط
        return {"accuracy": acc, "log_loss": ll, "confidence_gap": ece, "n": float(len(y_true))}
    # متعدد الأصناف
    pred = y_prob.argmax(axis=1)
    acc = float((pred == y_true).mean()) if len(y_true) else 0.0
    p = np.clip(y_prob, 1e-6, 1.0)
    p = p / p.sum(axis=1, keepdims=True)
    nll = float(-np.log(p[np.arange(len(y_true)), y_true]).mean()) if len(y_true) else 0.0
    conf = p.max(axis=1)
    ece = float(np.abs(conf.mean() - acc))
    return {"accuracy": acc, "nll": nll, "confidence_gap": ece, "n": float(len(y_true))}


def _rule_based_verdict(scores: Dict[str, float]) -> JudgeReport:
    acc = scores.get("accuracy", 0.0)
    gap = scores.get("confidence_gap", 0.0)
    ll = scores.get("log_loss", scores.get("nll", 1.0))
    strengths, weaknesses, plan = [], [], []
    if acc >= 0.85:
        strengths.append("دقة عالية على مجموعة التقييم.")
    elif acc >= 0.65:
        strengths.append("أداء مقبول لكن قابل للتحسين.")
    else:
        weaknesses.append("الدقة منخفضة — النموذج لم يتعلم الإشارة بقوة كافية.")
        plan.append("زيادة سعة النموذج أو عدد العصور مع early stopping.")
    if gap > 0.15:
        weaknesses.append("فجوة ثقة/دقة مرتفعة (سوء معايرة).")
        plan.append("تطبيق label smoothing أو temperature scaling بعد التدريب.")
    else:
        strengths.append("معايرة ثقة مقبولة نسبياً.")
    if ll > 0.7:
        weaknesses.append("خسارة احتمالية مرتفعة — تنبؤات غير حادة بما يكفي على العينات الصعبة.")
        plan.append("مراجعة معدل التعلم وجرّب cosine annealing + weight decay.")
    if not plan:
        plan.append("اختبر على بيانات خارج التوزيع (OOD) قبل النشر.")
        plan.append("دورة ضغط (quantization/pruning) إن كان الهدف أجهزة ضعيفة.")
    grade = "ممتاز" if acc >= 0.9 else ("جيد" if acc >= 0.75 else ("متوسط" if acc >= 0.6 else "ضعيف"))
    narrative = (
        f"تقييم محلي آلي: الدرجة **{grade}**. "
        f"الدقة {acc:.1%}، فجوة الثقة {gap:.3f}. "
        f"يُنصح بتنفيذ {len(plan)} إجراء(ات) قبل الدورة التالية."
    )
    return JudgeReport(
        ok=True,
        model_id="pending",
        scores=scores,
        strengths=strengths,
        weaknesses=weaknesses,
        next_cycle_plan=plan,
        narrative_ar=narrative,
        judge_backend="local_rules",
    )


def _llm_enrich(report: JudgeReport, context: str) -> JudgeReport:
    """يستعين بـ free_router إن وُجد لإثراء التقرير."""
    try:
        from ai.free_router import chat_free
    except Exception:
        return report
    prompt = (
        "أنت مهندس معماري خبير في تقييم نماذج التعلم الآلي. "
        "بالعربية الفصحى المبسطة، أكمل التقييم التالي بجمل قصيرة:\n"
        f"المقاييس: {json.dumps(report.scores, ensure_ascii=False)}\n"
        f"نقاط أولية: قوة={report.strengths} ضعف={report.weaknesses}\n"
        f"سياق: {context[:1500]}\n\n"
        "أجب بتنسيق JSON فقط بالمفاتيح: strengths (list), weaknesses (list), "
        "next_cycle_plan (list), narrative_ar (string)."
    )
    try:
        raw = chat_free(prompt)
        report.raw_llm = (raw or "")[:4000]
        # استخراج JSON
        import re

        m = re.search(r"\{[\s\S]*\}", raw or "")
        if m:
            data = json.loads(m.group(0))
            if isinstance(data.get("strengths"), list):
                report.strengths = [str(x) for x in data["strengths"][:8]]
            if isinstance(data.get("weaknesses"), list):
                report.weaknesses = [str(x) for x in data["weaknesses"][:8]]
            if isinstance(data.get("next_cycle_plan"), list):
                report.next_cycle_plan = [str(x) for x in data["next_cycle_plan"][:8]]
            if data.get("narrative_ar"):
                report.narrative_ar = str(data["narrative_ar"])[:2000]
            report.judge_backend = "free_router+local"
    except Exception as e:
        logger.warning("llm judge enrich failed: %s", e)
    return report


def judge_predictions(
    model_id: str,
    y_true: Any,
    y_prob: Any,
    context: str = "",
    use_llm: bool = True,
) -> JudgeReport:
    scores = _local_metrics(np.asarray(y_true), np.asarray(y_prob))
    report = _rule_based_verdict(scores)
    report.model_id = model_id
    if use_llm:
        report = _llm_enrich(report, context or f"model={model_id}")
    # حفظ
    path = JUDGE_DIR / f"judge_{model_id}_{int(time.time())}.json"
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    md_path = path.with_suffix(".md")
    md_path.write_text(report.to_markdown(), encoding="utf-8")
    return report


def judge_demo(use_llm: bool = False) -> JudgeReport:
    """تحكيم تجريبي على بيانات اصطناعية."""
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=n)
    # نموذج متوسط الجودة
    noise = rng.random(n) * 0.45
    prob = np.clip(y * 0.75 + (1 - y) * 0.25 + noise - 0.2, 0.02, 0.98)
    return judge_predictions("demo_clf", y, prob, context="تصنيف ثنائي تجريبي", use_llm=use_llm)


def handle_judge_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(حك[ّ]?م|judge|تحكيم|قي[ّ]?م\s*نموذج|تقرير\s*تقييم)", text, re.I):
        return None
    use_llm = not bool(re.search(r"بدون\s*llm|local\s*only|محلي\s*فقط", text, re.I))
    report = judge_demo(use_llm=use_llm)
    return report.to_markdown() + f"\n\n_حُفظ تحت `{JUDGE_DIR.relative_to(ROOT)}/`_"
