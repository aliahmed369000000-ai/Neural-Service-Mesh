"""
Continuous Training Agent — من صيانة إلى تدريب ذاتي مستمر
==========================================================
يراقب جودة الإجابات (verifier / heuristics) وعند الضعف:
  1) يسجّل فجوة جودة
  2) يقترح/يشغّل حلقة تدريب محلية أو مهمة Kaggle
  3) يحدّث سجل الإنتاج

لا يسحب الإنترنت بعدوانية بدون مفاتيح؛ يستخدم مصادر محلية + أوامر منظّمة.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("ContinuousTraining")

ROOT = Path(__file__).resolve().parent.parent
CT_DIR = ROOT / "artifacts" / "model_training" / "continuous"
CT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = CT_DIR / "training_triggers.jsonl"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append(event: Dict[str, Any]) -> None:
    event = {**event, "at": _now()}
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def assess_answer_quality(qa_result: Optional[Dict[str, Any]] = None, answer_text: str = "") -> Dict[str, Any]:
    """تقييم خفيف بدون فرض DeepEval."""
    score = 0.5
    reasons = []
    text = answer_text or (qa_result or {}).get("summary") or (qa_result or {}).get("answer") or ""
    if len(text.strip()) < 40:
        score -= 0.25
        reasons.append("إجابة قصيرة جداً")
    else:
        score += 0.1
    verses = (qa_result or {}).get("verses") or []
    concepts = (qa_result or {}).get("primary_concepts") or (qa_result or {}).get("concepts") or []
    if concepts:
        score += 0.15
    else:
        reasons.append("لا مفاهيم مسترجَعة")
        score -= 0.1
    if verses:
        score += 0.1
    # محاولة verifier إن توفّر
    try:
        from ai.nsm_answer_verifier import verify_answer_faithfulness
        if qa_result:
            v = verify_answer_faithfulness(qa_result)
            if isinstance(v, dict) and v.get("score") is not None:
                fs = float(v["score"])
                score = 0.5 * score + 0.5 * fs
                reasons.append(f"faithfulness={fs:.2f}")
    except Exception as e:
        reasons.append(f"verifier_skip:{e.__class__.__name__}")
    score = float(max(0.0, min(1.0, score)))
    weak = score < 0.55
    return {"score": score, "weak": weak, "reasons": reasons}


def plan_retrain(quality: Dict[str, Any], prefer_remote: bool = True) -> Dict[str, Any]:
    """خطة تدريب عند ضعف الجودة."""
    if not quality.get("weak"):
        return {"action": "none", "reason_ar": "الجودة مقبولة — لا إعادة تدريب فورية."}
    plan = {
        "action": "retrain",
        "local_epochs_boost": 20,
        "prefer_kaggle": bool(prefer_remote),
        "script": "run_training_loop.sh",
        "reason_ar": "جودة دلالية ضعيفة — رفع epochs وجلب بيانات مكمّلة محلية/بعيدة.",
        "quality": quality,
    }
    return plan


def trigger_local_training_hint() -> str:
    """لا يشغّل تدريباً ثقيلاً في طلب HTTP؛ يعيد أوامر آمنة."""
    return (
        "لتشغيل التدريب المستمر محلياً:\n"
        "```bash\nbash run_training_loop.sh\n```\n"
        "أو عبر الوكيل: `درّب بعيد kaggle وادفع` بعد `جهّز kaggle`."
    )


def run_continuous_cycle(sample_answer: str = "", prefer_remote: bool = True) -> Dict[str, Any]:
    q = assess_answer_quality(answer_text=sample_answer or "إجابة تجريبية قصيرة")
    plan = plan_retrain(q, prefer_remote=prefer_remote)
    remote = None
    if plan.get("action") == "retrain" and prefer_remote:
        try:
            from connectors.kaggle_training_connector import queue_retrain_job
            remote = queue_retrain_job(epochs=plan.get("local_epochs_boost", 20))
        except Exception as e:
            remote = {"ok": False, "error": str(e)}
    event = {"quality": q, "plan": plan, "remote": remote}
    _append(event)
    return event


def continuous_status() -> str:
    """ملخص حالة التدريب الذاتي المستمر من السجل المحلي."""
    lines = ["## 🔄 حالة التدريب الذاتي المستمر", ""]
    if not LOG_PATH.is_file():
        lines.append("لا يوجد سجل بعد. شغّل `تدريب مستمر` لتسجيل أول دورة.")
        lines.append("")
        lines.append(trigger_local_training_hint())
        return "\n".join(lines)
    try:
        rows = LOG_PATH.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    except Exception as e:
        return f"تعذّر قراءة السجل: {e}"
    lines.append(f"- عدد الأحداث المسجّلة: **{len(rows)}**")
    if rows:
        try:
            last = json.loads(rows[-1])
            q = last.get("quality") or {}
            plan = last.get("plan") or {}
            lines.append(f"- آخر وقت: `{last.get('at', '?')}`")
            lines.append(f"- جودة أخيرة: **{q.get('score', '?')}** (ضعيف={q.get('weak')})")
            if q.get("reasons"):
                lines.append(f"- أسباب: {', '.join(map(str, q.get('reasons', [])[:5]))}")
            lines.append(f"- إجراء مخطط: **{plan.get('action', '?')}** — {plan.get('reason_ar', '')}")
        except Exception:
            lines.append("- آخر حدث موجود لكن تعذّر تفصيله.")
    lines.append("")
    lines.append(trigger_local_training_hint())
    return "\n".join(lines)


def handle_continuous_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(r"(حالة|status|سجل).{0,12}(تدريب\s*مستمر|continuous)", text, re.I) or re.search(
        r"(تدريب\s*مستمر).{0,8}(حالة|status|سجل)", text, re.I
    ):
        return continuous_status()
    if not re.search(
        r"(تدريب\s*مستمر|continuous\s*train|صيان[ةه]\s*تدريب|راقب\s*جود[ةه]|اعادة\s*تدريب\s*ذاتي)",
        text,
        re.I,
    ):
        return None
    ev = run_continuous_cycle(prefer_remote=bool(re.search(r"kaggle|بعيد", text, re.I)))
    return (
        "## 🔄 وكيل التدريب الذاتي المستمر\n\n```json\n"
        + json.dumps(ev, ensure_ascii=False, indent=2)[:3500]
        + "\n```\n\n"
        + trigger_local_training_hint()
    )
