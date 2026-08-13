# -*- coding: utf-8 -*-
"""
ai/proactive_planning.py — التخطيط الجماعي الاستباقي (Proactive Planning)
═══════════════════════════════════════════════════════════════════════════
قبل أن يبدأ الفريق مهمة جديدة، يُنتج هذا المخطط «رأي جماعي استباقي»:
1. يستحضر من سجل الخبرات (TEM) ما يشبه المهمة الحالية: نجاحات سابقة،
   وإخفاقات يجب تفاديها (failure_avoid).
2. يستحضر من نظام المكافآت (Role Rewards) أفضل الأدوار للمهارات المطلوبة.
3. يجمع التوصيات في خطة استباقية واحدة: توزيع الأدوار المقترح،
   استراتيجية التنفيذ، ومحاذير (cautions) من الإخفاقات السابقة.

التكامل:
- cooperative_tasks: يُستدعى بعد _advise_roles_from_experience وقبل
  إنشاء الأدوار، وتُضاف خطته إلى ملاحظات المهمة (task._pp_plan).
- long_horizon_tasks: يُستدعى قبل بناء الخطة، مع دمج توصياته في
  عناوين الخطوات الأولى عند الاقتضاء.

التدهور: فشل كامل → كتلة _PP_OK في app_core تعطل الوحدة بصمت.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MAX_GOAL_CHARS = 200
_MAX_ADVICE_CHARS = 600

# دوال late import من app_core (تفادي circular import — نفس نمط TEM/SKB).
_app_core_for_pp = None

def _set_app_core(app_core_module: Any) -> None:
    """تُستدعى مرة عند تحميل الوحدة داخل app_core."""
    global _app_core_for_pp
    _app_core_for_pp = app_core_module


def _app_core_mod():
    global _app_core_for_pp
    if _app_core_for_pp is None:
        import app_core as _m  # noqa: E402
        _app_core_for_pp = _m
    return _app_core_for_pp


def _RR_OK() -> bool:
    return bool(getattr(_app_core_mod(), "_RR_OK", False))


def _TEM_OK() -> bool:
    return bool(getattr(_app_core_mod(), "_TEM_OK", False))


def _get_experience_log():
    fn = getattr(_app_core_mod(), "_get_experience_log", None)
    if fn is None:
        raise RuntimeError("سجل الخبرات الجماعية غير متاح")
    return fn()


def _get_role_rewards():
    fn = getattr(_app_core_mod(), "_get_role_rewards", None)
    if fn is None:
        raise RuntimeError("نظام المكافآت غير متاح")
    return fn()


def build_pre_task_plan(goal: str, skills: Optional[List[str]] = None,
                        top_k: int = 5) -> Dict[str, Any]:
    """
    خطة استباقية لمهمة جديدة قبل أي تخصيص أدوار أو بناء خطة تنفيذ.
    يعيد: {"roles_advice": [...], "strategy_notes": [...],
            "cautions": [...], "recalled_contexts": int}
    """
    goal = (goal or "")[:_MAX_GOAL_CHARS]
    out: Dict[str, Any] = {"roles_advice": [], "strategy_notes": [],
                           "cautions": [], "recalled_contexts": 0}
    if not goal.strip():
        return out

    # ── 1. استحضار الخبرات المشابهة من TEM ─────────────────────────────────
    try:
        if _TEM_OK():
            recalled = _get_experience_log().recall(goal, top_k=8)
            successes: List[Dict[str, Any]] = []
            failures: List[Dict[str, Any]] = []
            for e in recalled:
                entry = {"context": (e.get("context") or "")[:_MAX_ADVICE_CHARS],
                         "decision": (e.get("decision") or "")[:_MAX_ADVICE_CHARS],
                         "confidence": e.get("confidence", 0.0)}
                if e.get("outcome") == "failure":
                    failures.append(entry)
                else:
                    successes.append(entry)
            out["recalled_contexts"] = len(recalled)
            out["strategy_notes"] = [s["decision"] for s in successes][:3]
            out["cautions"] = [
                ("تجنّب تكرار: " + f["context"]) for f in failures][:3]
    except Exception as exc:
        logger.warning("ProactivePlan TEM recall failed: %s", exc)

    # ── 2. أفضل الأدوار للمهارات المطلوبة من نظام المكافآت ────────────────
    try:
        if _RR_OK():
            rr = _get_role_rewards()
            for sk in (skills or [])[:5]:
                tops = rr.top_roles_for_skill(sk, 3)
                for t in tops[:2]:
                    out["roles_advice"].append({
                        "skill": sk, "role": t["role"],
                        "skill_score": round(t["skill_score"], 2),
                        "xp": round(t["xp"], 1)})
    except Exception as exc:
        logger.warning("ProactivePlan role advice failed: %s", exc)

    # ── 3. توصيات عامة من أعلى خبرات الثقة ──────────────────────────────────
    try:
        if _TEM_OK() and not out["strategy_notes"]:
            best = _get_experience_log().latest(3)
            out["strategy_notes"] = [
                (b.get("decision") or "")[:_MAX_ADVICE_CHARS]
                for b in best if b.get("decision")]
    except Exception:
        pass

    return out


def plan_summary_text(plan: Dict[str, Any]) -> str:
    """نص موحّد تُحقن خطة استباقية في تقارير وسجلات المهام."""
    parts: List[str] = []
    adv = plan.get("roles_advice") or []
    if adv:
        parts.append("أدوار مقترحة: " + " ؛ ".join(
            f"{a['role']} ({a['skill']})" for a in adv[:3]))
    notes = plan.get("strategy_notes") or []
    if notes:
        parts.append("استراتيجية: " + " ؛ ".join(notes[:2]))
    caut = plan.get("cautions") or []
    if caut:
        parts.append("محاذير: " + " ؛ ".join(caut[:2]))
    return " | ".join(parts)
