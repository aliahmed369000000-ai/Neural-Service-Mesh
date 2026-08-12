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



# ── تفعيل مستمر ───────────────────────────────────────────────────────────
_CFG = ROOT / "config" / "continuous_learning.json"
_FLAG = CT_DIR / "enabled.flag"


def is_continuous_enabled() -> bool:
    if _FLAG.is_file():
        return _FLAG.read_text(encoding="utf-8").strip() != "0"
    if _CFG.is_file():
        try:
            return bool(json.loads(_CFG.read_text(encoding="utf-8")).get("enabled", True))
        except Exception:
            pass
    return True


def enable_continuous_learning(enabled: bool = True) -> Dict[str, Any]:
    """تفعيل/إيقاف التعلّم المستمر بشكل دائم."""
    CT_DIR.mkdir(parents=True, exist_ok=True)
    _FLAG.write_text("1" if enabled else "0", encoding="utf-8")
    cfg = {
        "enabled": bool(enabled),
        "mode": "continuous" if enabled else "paused",
        "web_self_feed": bool(enabled),
        "quality_monitor": bool(enabled),
        "active_retrain_plans": bool(enabled),
        "updated_at": _now(),
    }
    try:
        _CFG.parent.mkdir(parents=True, exist_ok=True)
        if _CFG.is_file():
            old = json.loads(_CFG.read_text(encoding="utf-8"))
            old.update(cfg)
            cfg = old
        _CFG.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass
    _append({"event": "enable_toggle", "enabled": bool(enabled)})
    # اربط العلم مع AutoRuntime دون حلقة إقلاع ثقيلة
    try:
        from ai.auto_runtime import get_auto_runtime
        rt = get_auto_runtime(start=False)
        # لا نوقف كل AutoRuntime عند إيقاف التعلّم فقط — نتركه يعمل للفحوصات
        if enabled and not getattr(rt, "_running", False):
            rt.enable(True)
            rt.start()
    except Exception:
        pass
    return {"ok": True, "enabled": bool(enabled), "config": cfg}


def run_continuous_learning_pulse() -> Dict[str, Any]:
    """نبضة تعلّم مستمر كاملة: جودة + تغذية ويب + خطة فجوات."""
    if not is_continuous_enabled():
        return {"ok": False, "msg": "التعلّم المستمر متوقف", "enabled": False}
    out: Dict[str, Any] = {"ok": True, "enabled": True, "phases": {}}
    # 1) مراقبة جودة / خطة تدريب
    try:
        out["phases"]["quality_cycle"] = run_continuous_cycle(prefer_remote=False)
    except Exception as e:
        out["phases"]["quality_cycle"] = {"error": str(e)}
    # 2) تغذية من الويب
    try:
        from ai.self_feed_learner import self_learn_cycle
        out["phases"]["self_feed"] = self_learn_cycle(limit=2)
    except Exception as e:
        out["phases"]["self_feed"] = {"error": str(e)}
    # 3) حصاد فجوات → خطة إعادة تدريب (بدون استبدال أوزان تلقائي)
    try:
        from ai.active_retrain_loop import harvest_gaps, plan_retrain
        gaps = harvest_gaps()
        plan = plan_retrain(epochs=15)
        out["phases"]["active_retrain"] = {"gaps_n": gaps.get("n"), "plan_action": plan.get("action") if isinstance(plan, dict) else None}
        # احفظ الخطة إن وُجد مسار
        try:
            from ai.active_retrain_loop import OUT as AR_OUT
            AR_OUT.mkdir(parents=True, exist_ok=True)
            (AR_OUT / "last_auto_plan.json").write_text(
                json.dumps({"gaps": gaps, "plan": plan, "at": _now()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception:
            pass
    except Exception as e:
        out["phases"]["active_retrain"] = {"error": str(e)}
    _append({"event": "learning_pulse", "summary": {k: bool(v) for k, v in out["phases"].items()}})
    return out


def handle_continuous_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    # تفعيل / إيقاف
    _learn_kw = r"(?:ال)?تعل[ّ]?م\s*(?:ال)?مستمر|continuous\s*learn(?:ing)?|تدريب\s*مستمر"
    if re.search(rf"(تفعيل|شغ[ّل]?ل|enable).{{0,24}}({_learn_kw})", text, re.I) \
            or re.search(rf"({_learn_kw}).{{0,16}}(تفعيل|شغ[ّل]?ل|enable)", text, re.I) \
            or re.search(r"^(تفعيل|enable)\s*(ال)?تعل", text, re.I):
        res = enable_continuous_learning(True)
        return (
            "## ✅ تم تفعيل التعلّم المستمر\n\n"
            "يعمل تلقائياً مع **AutoRuntime** (نبضات دورية: جودة + تغذية ويب + خطط فجوات).\n\n"
            "```json\n"
            + json.dumps(res, ensure_ascii=False, indent=2)[:2000]
            + "\n```\n\n"
            "للنبضة الفورية: `نبضة تعلّم` — للحالة: `حالة تدريب مستمر`"
        )
    if re.search(r"(إيقاف|عط[ّل]ل|disable|pause).{0,20}(تعل[ّم]م\s*مستمر|التعل[ّم]م\s*المستمر|continuous\s*learn|تدريب\s*مستمر)", text, re.I):
        res = enable_continuous_learning(False)
        return "## ⏸ أُوقف التعلّم المستمر\n```json\n" + json.dumps(res, ensure_ascii=False, indent=2) + "\n```"
    if re.search(r"(حالة|status|سجل).{0,12}(تدريب\s*مستمر|continuous|تعل[ّم]م\s*مستمر)", text, re.I) or re.search(
        r"(تدريب\s*مستمر|تعل[ّم]م\s*مستمر).{0,8}(حالة|status|سجل)", text, re.I
    ):
        st = continuous_status()
        return st + f"\n\n- مفعّل الآن: **{is_continuous_enabled()}**"
    if not re.search(
        r"(تدريب\s*مستمر|continuous\s*train|صيان[ةه]\s*تدريب|راقب\s*جود[ةه]|اعادة\s*تدريب\s*ذاتي|"
        r"تعل[ّم]م\s*مستمر|نبضة\s*تعل[ّم]م)",
        text,
        re.I,
    ):
        return None
    if re.search(r"نبضة\s*تعل[ّم]م|learning\s*pulse", text, re.I):
        pulse = run_continuous_learning_pulse()
        return "## 🔄 نبضة تعلّم مستمر\n```json\n" + json.dumps(pulse, ensure_ascii=False, indent=2)[:4000] + "\n```"
    ev = run_continuous_cycle(prefer_remote=bool(re.search(r"kaggle|بعيد", text, re.I)))
    return (
        "## 🔄 وكيل التدريب الذاتي المستمر\n\n```json\n"
        + json.dumps(ev, ensure_ascii=False, indent=2)[:3500]
        + "\n```\n\n"
        + trigger_local_training_hint()
    )
