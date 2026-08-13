# -*- coding: utf-8 -*-
"""
السرب المتعلم (Adaptive Swarm)
================================
طبقة تعلّم ديناميكية فوق نظام الوكلاء: تراقب الأداء التاريخي للوكلاء عبر
أحداث ناقل الأحداث، وتحوّله إلى ملف أداء موثوق لكل وكيل (نقاط 0-100)،
ثم تستخدمه لأربع قرارات حية:

1. **الترتيب**: الوكلاء المختارون للمهمة يُرتَّبون من الأعلى أداءً
   ليبدأ الأقوى ويُبنى التوليف حول ترتيبه.
2. **الاستبعاد**: وكيل فشلت أغلب مهماته تاريخيًا لا يُدعى لمهمة جديدة
   (حماية مؤقتة بدل الحذف الدائم).
3. **الوزن**: عند التوليف، تُرتَّب ردود الوكلاء في برومبت التوليف
   بحيث يقدَّم أداء الأقوى قبل الأضعف.
4. **الحجم**: عدد الوكلاء المسموح به يتكيّف مع كثافة الأحداث (سرب
   أصغر عندما يرتفع الفشل).

قواعد أمان صارمة تحفظ استقرار النظام:

- لا بيانات تاريخية كافية (أقل من حد أدنى من الأحداث) →
  الوضع المتعلم لا يتدخل إطلاقًا ويعمل النظام كالمعتاد (fallback).
- لا يُستبعد وكيل من مهمة جديدة إلا إذا كانت سجلاته كافية (مهام >= 2)
  ونسبة فشله >= 0.75.
- لا يتجاوز الاستبعاد حدودًا قصوى: على الأقل وكيل واحد يبقى دائمًا.
- نقاط الأداء تستخدم توهينًا زمنيًا: أحداث أقدم من 48 ساعة تفقد
  50% من وزنها تدريجيًا، فلا يعلّق الوكيل الجديد على سجل قديم.

القابلية للاختبار: كل الدوال نقية (events تُمرَّر كمعامل)، لا اعتماد
على Streamlit أو أي API خارجي، والتكامل مع ناقل الأحداث عبر حقن.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("adaptive_swarm")

# ──────────────────────────────────────────────── ثوابت
MIN_EVENTS_FOR_LEARNING = 4      # أقل عدد أحداث ليُفعَّل الوضع المتعلم
MIN_TASKS_FOR_EXCLUSION = 2      # أقل عدد مهام قبل السماح بالاستبعاد
EXCLUSION_FAILURE_RATE = 0.75    # نسبة فشل تستدعي الاستبعاد المؤقت
KEEP_AT_LEAST_AGENTS = 1         # لا نستبعد الجميع
MAX_ADAPTIVE_AGENTS = 3          # حد أعلى للوكلاء في المسار المتعدد
LOW_FAILURE_CAP = 2              # حد عدد المهام قبل اعتبار الوكيل «غير محمَّل»
WEIGHT_SUCCESS = 0.70            # ترجيح النجاح في نقاط الأداء
WEIGHT_SPEED = 0.20              # ترجيح السرعة
WEIGHT_STABILITY = 0.10          # ترجيح الاستقرار (قلة إعادة المحاولات)
AGE_HALF_LIFE_SECONDS = 48 * 3600  # نصف عمر الأحداث (48 ساعة)
BASE_SCORE_MISSING_DATA = 50.0   # نقاط وكيل بلا سجل (محايد)

# مفاتيح أحداث الأداء الموثوق بها من ناقل الأحداث
_SUCCESS_EVENTS = {"agent_done", "task_done", "synthesis_done",
                   "delegation_resolved", "swarm_task_done",
                   "debate_consensus", "lesson_learned"}
_FAILURE_EVENTS = {"agent_error", "task_error", "delegation_rejected",
                   "agent_failed", "swarm_task_failed"}

_ADAPTIVE_RANKED = "adaptive_ranked"
_ADAPTIVE_EXCLUDED = "adaptive_excluded"
_ADAPTIVE_REWEIGHTED = "adaptive_reweighted"


# ──────────────────────────────────────────────── نقاط الأداء

def _age_weight(ts: float, now: float) -> float:
    """توهين زمني: حدث عمره نصف العمر يفقد نصف وزنه."""
    if ts <= 0:
        return 1.0
    age = max(0.0, now - ts)
    if age <= 0:
        return 1.0
    return 0.5 ** (age / AGE_HALF_LIFE_SECONDS)


def _event_ts(event: Dict[str, Any], now: float) -> float:
    ts = event.get("ts")
    try:
        tsf = float(ts)
    except (TypeError, ValueError):
        tsf = now
    return tsf


def agent_profiles(events: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """تحويل أحداث الناقل إلى ملف أداء لكل وكيل.

    المعادلات (لكل وكيل):
        score  = 70% × success + 20% × speed + 10% × stability
        success    = مهام ناجحة / مهام إجمالية  (0 إذا لا مهام)
        speed      = 1 - min(1, duration / 60s)      (الأسرع أعلى)
        stability  = 1 - min(1, retries / (tasks + 1))
    """
    now = time.time()
    profiles: Dict[str, Dict[str, Any]] = {}

    for ev in events:
        etype = str(ev.get("event_type", ""))
        if (etype not in _SUCCESS_EVENTS
                and etype not in _FAILURE_EVENTS
                and etype != "agent_started"):
            continue
        key = ev.get("agent_id") or ev.get("agent")
        if not key:
            continue
        key = str(key)
        prof = profiles.setdefault(key, {
            "key": key,
            "tasks": 0, "done": 0, "errors": 0, "retries": 0,
            "durations": [],
            "start_count": 0,
            "latest_ts": 0.0,
        })
        ts = _event_ts(ev, now)
        w = _age_weight(ts, now)
        prof["latest_ts"] = max(prof["latest_ts"], ts)
        prof["_w_sum"] = prof.get("_w_sum", 0.0) + w

        if etype in _SUCCESS_EVENTS:
            prof["done"] = prof.get("done", 0) + 1 * w
        elif etype in _FAILURE_EVENTS:
            prof["errors"] = prof.get("errors", 0) + 1 * w
        if etype == "agent_started":
            prof["start_count"] = prof.get("start_count", 0) + 1 * w
            prof["tasks"] = prof.get("tasks", 0) + 1 * w
        if "duration_ms" in ev or "duration" in ev:
            try:
                d = float(ev.get("duration_ms") or ev.get("duration", 0))
                prof["durations"].append((d, w))
            except (TypeError, ValueError):
                pass

    # درجات موحدة 0-100
    for prof in profiles.values():
        tasks = prof.get("tasks", 0) or (prof.get("start_count", 0) or 1e-9)
        done = prof.get("done", 0.0)
        errors = prof.get("errors", 0.0)
        denom = tasks + errors or 1e-9
        success_rate = min(1.0, max(0.0, done / denom))
        durations = [d for d, w in prof.get("durations", [])]
        avg_dur_s = (sum(durations) / max(1, len(durations))) / 1000.0 if durations else 30.0
        speed = 1.0 - min(1.0, max(0.0, avg_dur_s / 60.0))
        # إعادة المحاولات تقارب: عدد البدايات فوق عدد المهام
        starts = prof.get("start_count", 0) or 1e-9
        retries = max(0.0, starts - tasks)
        stability = 1.0 - min(1.0, retries / (tasks + 1))
        score = (WEIGHT_SUCCESS * success_rate
                 + WEIGHT_SPEED * speed
                 + WEIGHT_STABILITY * stability) * 100.0
        prof.update({
            "success_rate": round(success_rate, 3),
            "avg_duration_ms": round(avg_dur_s * 1000, 1),
            "retries": round(retries, 2),
            "score": round(score, 1),
        })

    for prof in profiles.values():
        prof.pop("_w_sum", None)
    return profiles


# ──────────────────────────────────────────────── القرارات الحية

def _learning_enabled(events: List[Dict[str, Any]]) -> bool:
    return len(events) >= MIN_EVENTS_FOR_LEARNING


def rank_agents(selected: List[str],
                events: List[Dict[str, Any]]) -> List[str]:
    """إعادة ترتيب الوكلاء المختارين من الأعلى أداءً.

    عند عدم كفاية البيانات التاريخية يُعاد `selected` كما هو
    (الوضع المتعلم لا يتدخل).
    """
    if not _learning_enabled(events):
        return list(selected)
    profiles = agent_profiles(events)
    ordered = sorted(
        selected,
        key=lambda k: profiles.get(k, {}).get("score", BASE_SCORE_MISSING_DATA),
        reverse=True,
    )
    return ordered


def exclude_agents(selected: List[str],
                   events: List[Dict[str, Any]]) -> List[str]:
    """استبعاد مؤقت للوكلاء الفاشلين تاريخيًا.

    يعود القائمة بعد الاستبعاد مع ضمان بقاء وكيل واحد على الأقل.
    """
    if not _learning_enabled(events):
        return list(selected)
    profiles = agent_profiles(events)
    kept, excluded = [], []
    for key in selected:
        prof = profiles.get(key, {})
        tasks = prof.get("tasks", 0)
        failure_rate = (prof.get("errors", 0.0) / (tasks + 0.0)) if tasks else 0.0
        is_failure_prone = (tasks >= MIN_TASKS_FOR_EXCLUSION
                            and failure_rate >= EXCLUSION_FAILURE_RATE)
        if is_failure_prone:
            excluded.append((key, failure_rate, tasks))
        else:
            kept.append(key)
    if len(kept) < KEEP_AT_LEAST_AGENTS:
        # لا نستبعد الجميع: أعد الأفضل أداءً حتى يبقى واحد
        remaining = [k for k in selected if k not in [e[0] for e in excluded]]
        profs = agent_profiles(events)
        remaining.sort(
            key=lambda k: profs.get(k, {}).get("score", BASE_SCORE_MISSING_DATA),
            reverse=True,
        )
        return remaining[:KEEP_AT_LEAST_AGENTS]
    return kept


def excluded_agents(selected: List[str],
                    events: List[Dict[str, Any]]) -> List[Tuple[str, float, float]]:
    """وكلاء جرى استبعادهم (المعرف، نسبة الفشل، عدد المهام) لأغراض العرض."""
    if not _learning_enabled(events):
        return []
    profiles = agent_profiles(events)
    out = []
    for key in selected:
        prof = profiles.get(key, {})
        tasks = prof.get("tasks", 0)
        failure_rate = (prof.get("errors", 0.0) / (tasks + 0.0)) if tasks else 0.0
        if tasks >= MIN_TASKS_FOR_EXCLUSION and failure_rate >= EXCLUSION_FAILURE_RATE:
            out.append((key, round(failure_rate, 3), float(tasks)))
    return out


def weighted_synth_prompt(task: str,
                          agent_replies: Dict[str, str],
                          events: List[Dict[str, Any]],
                          titles: Optional[Dict[str, str]] = None) -> str:
    """برومبت توليف ترجيحي: يعرض ردود الوكلاء بترتيب أدائهم التاريخي.

    الأقوى أداءً يُقدَّم أولًا مع وسم وزنه، فيوجّه انتباه المدير الموحّد
    نحو الإجابة الأكثر موثوقية. عند عدم كفاية البيانات يُبنى البرومبت
    بالترتيب الافتراضي (التوليف الأصلي لم يتغير سلوكيًا).
    """
    titles = titles or {}
    profiles = agent_profiles(events)
    have_history = any(_learning_enabled(events) for _ in [events])
    if _learning_enabled(events) and any(
            k in profiles for k in agent_replies):
        keys = sorted(agent_replies.keys(),
                      key=lambda k: profiles.get(k, {}).get(
                          "score", BASE_SCORE_MISSING_DATA),
                      reverse=True)
        weight_label = lambda score: (
            "الأعلى أداءً" if score >= 70
            else ("متوسط الأداء" if score >= 45 else "الأقل أداءً")
        )
    else:
        keys = list(agent_replies.keys())
        weight_label = lambda _s: "لا بيانات كافية للترجيح"

    lines = [f"المهمة: {task.strip()}", "", "ردود الوكلاء بالترتيب المرجَّح:"]
    for i, key in enumerate(keys, 1):
        title = titles.get(key, key)
        score = profiles.get(key, {}).get("score", BASE_SCORE_MISSING_DATA)
        label = weight_label(score)
        reply = (agent_replies.get(key, "") or "").strip()
        lines.append(
            f"{i}. [{title}] (أداء تاريخي {score}/100 — {label})\n{reply}"
        )
    lines.append("")
    lines.append(
        "أولّف هذه الردود في إجابة موحّدة شاملة، مع اعتماد "
        "الأقوى أداءً تاريخيًا مرجعًا أساسيًا عند التعارض."
    )
    return "\n".join(lines)


def adaptive_max_agents(base_max: int,
                        events: List[Dict[str, Any]]) -> int:
    """تكييف سقف الوكلاء: عند ارتفاع الفشل الجماعي نُصغّر السرب."""
    if not _learning_enabled(events):
        return base_max
    profiles = agent_profiles(events)
    if not profiles:
        return base_max
    busy = [p for p in profiles.values() if p.get("tasks", 0) >= LOW_FAILURE_CAP]
    if not busy:
        return base_max
    avg_failure = sum(p.get("errors", 0.0) for p in busy) / max(1, sum(
        p.get("tasks", 0) for p in busy))
    if avg_failure >= EXCLUSION_FAILURE_RATE:
        return min(base_max, 2)
    if avg_failure >= 0.5:
        return min(base_max, base_max)
    return min(base_max, MAX_ADAPTIVE_AGENTS)


# ──────────────────────────────────────────────── التكامل مع ناقل الأحداث

def _safe_emit(emit_fn, **kwargs) -> None:
    try:
        emit_fn(**kwargs)
    except Exception as exc:  # الناقل اختياري: لا نعطّل القرار بسببه
        logger.warning("adaptive_swarm: تعذّر إطلاق حدث %s: %s",
                       kwargs.get("event_type", "?"), exc)


def announce_adaptive_ranking(emit_fn, selected: List[str],
                              profiles: Dict[str, Dict[str, Any]],
                              parent_task_id: str = "adaptive") -> None:
    """إطلاق adaptive_ranked بعد إعادة ترتيب الوكلاء."""
    _safe_emit(
        emit_fn,
        event_type=_ADAPTIVE_RANKED,
        agent_id="adaptive_swarm",
        title="السرب المتعلم",
        status="running",
        detail="ترتيب الوكلاء حسب الأداء التاريخي: "
               + " ← ".join(str(k) for k in selected),
        metadata={"order": list(selected),
                  "profiles": {k: profiles.get(k, {}).get("score", 0.0)
                               for k in selected},
                  "parent_task_id": parent_task_id},
    )


def announce_adaptive_exclusion(emit_fn,
                                excluded: List[Tuple[str, float, float]],
                                kept: List[str],
                                parent_task_id: str = "adaptive") -> None:
    """إطلاق adaptive_excluded عند استبعاد وكلاء فاشلين."""
    detail = ("استبعاد مؤقت: "
              + "، ".join(f"{k} (فشل {f:.0%})" for k, f, _ in excluded)
              + " | بقي: " + "، ".join(str(k) for k in kept))
    _safe_emit(
        emit_fn,
        event_type=_ADAPTIVE_EXCLUDED,
        agent_id="adaptive_swarm",
        title="السرب المتعلم",
        status="running",
        detail=detail,
        metadata={"excluded": [{"key": k, "failure_rate": f, "tasks": t}
                               for k, f, t in excluded],
                  "kept": list(kept),
                  "parent_task_id": parent_task_id},
    )


def announce_adaptive_reweight(emit_fn, ordered: List[str],
                               profiles: Dict[str, Dict[str, Any]],
                               parent_task_id: str = "adaptive") -> None:
    """إطلاق adaptive_reweighted عند ترجيح ردود التوليف."""
    _safe_emit(
        emit_fn,
        event_type=_ADAPTIVE_REWEIGHTED,
        agent_id="adaptive_swarm",
        title="السرب المتعلم",
        status="running",
        detail="ترجيح ردود التوليف حسب الأداء التاريخي",
        metadata={"order": list(ordered),
                  "profiles": {k: profiles.get(k, {}).get("score", 0.0)
                               for k in ordered},
                  "parent_task_id": parent_task_id},
    )


ADAPTIVE_EVENTS = {_ADAPTIVE_RANKED, _ADAPTIVE_EXCLUDED,
                   _ADAPTIVE_REWEIGHTED}
