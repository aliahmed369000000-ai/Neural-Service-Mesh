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
- نقاط الأداء تستخدم توهينًا زمنيًا قابلًا للتكوين عبر ملف
  `config/adaptive_swarm.json`: أربع صيغ توهين (أسي، نافذة صارمة،
  سيجموي، مزيج حديث/تراكمي)، تعزيز اختياري لأحداث آخر 24 ساعة في
  القرارات الحرجة (الاستبعاد تحديدًا)، توهين حسب نشاط الوكيل
  (الوكيل النشط تفقد سجلاته وزنها أسرع)، وتجميع زمني للأحداث يقلل
  التكلفة ويثبت النتائج.

القابلية للاختبار: كل الدوال نقية (events تُمرَّر كمعامل)، لا اعتماد
على Streamlit أو أي API خارجي، والتكامل مع ناقل الأحداث عبر حقن.
"""

from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("adaptive_swarm")

# ──────────────────────────────────────────────── ثوابت أساسية
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

# ──────────────────────────────────────────────── ثوابت التوهين الافتراضية
_DECAY_MODES = ("exponential", "window", "sigmoid", "blended")
_DEFAULT_DECAY_CONFIG: Dict[str, Any] = {
    "half_life_hours": 48.0,
    "decay_mode": "exponential",
    "strict_window_hours": 72.0,
    "recent_boost_hours": 24.0,
    "recent_boost_factor": 2.0,
    "activity_decay": False,
    "activity_window_hours": 72.0,
    "activity_half_life_hours": 24.0,
    "bucket_minutes": 0,
}
_decay_config_cache: Dict[str, Any] = {}

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

def _load_adaptive_config() -> Dict[str, Any]:
    """قراءة تكوين التوهين الزمني من config/adaptive_swarm.json.

    عند تعذّر القراءة يُستخدم التكوين الافتراضي (سلوك مطابق لما قبل
    هذا التحسين: أسي بنصف عمر 48 ساعة دون تعزيز أو تجميع)، فلا يفشل
    النظام بسبب ملف تهيئة غائب أو مشوه.
    """
    global _decay_config_cache
    if _decay_config_cache:
        return _decay_config_cache
    merged = dict(_DEFAULT_DECAY_CONFIG)
    try:
        root = Path(__file__).resolve().parent.parent
        cfg_path = root / "config" / "adaptive_swarm.json"
        if cfg_path.is_file():
            with open(cfg_path, encoding="utf-8") as f:
                user_cfg = json.load(f) or {}
            for key, default in merged.items():
                value = user_cfg.get(key)
                if value is None:
                    continue
                # صلاحيات بسيطة: أنماط معروفة للنصوص، أرقام موجبة للقيم
                if key == "decay_mode":
                    if value in _DECAY_MODES:
                        merged[key] = value
                elif key == "activity_decay":
                    merged[key] = bool(value)
                elif isinstance(default, (int, float)):
                    # يقبل الأرقام الصريحة (0 مسموح للتجميع) والمركّبة
                    try:
                        fvalue = float(value)
                        if isinstance(value, bool):
                            continue  # bool يُعامل رقمًا في بايثون — نرفضه
                        if fvalue >= 0 or isinstance(default, (int, float)):
                            merged[key] = fvalue
                    except (TypeError, ValueError):
                        pass
            if merged.get("recent_boost_factor", 1.0) < 1.0:
                merged["recent_boost_factor"] = 1.0
            if not merged.get("enabled", True):
                merged = dict(_DEFAULT_DECAY_CONFIG)
        _decay_config_cache = merged
    except Exception as exc:
        logger.warning("adaptive_swarm: تعذّر قراءة التهيئة — الافتراضي: %s", exc)
        merged = dict(_DEFAULT_DECAY_CONFIG)
    return merged


def _decay_config() -> Dict[str, Any]:
    """واجهة الوصول الوحيدة لتكوين التوهين (قابلة للتجاهل في الاختبارات)."""
    return _load_adaptive_config()


def reset_decay_config() -> None:
    """إفراغ ذاكرة التخزين المؤقت للتكوين — تُستدعى بعد تعديل الملف.

    موجودة أساسًا للاختبار والمراقبة: تعديل المستخدم لملف التهيئة
    يُقرأ تلقائيًا عند الحاجة إذا حُذفت الذاكرة (الخدمة تقرأ الملف
    مرة واحدة ثم تحتفظ به في الذاكرة).
    """
    global _decay_config_cache
    _decay_config_cache = {}


def format_recency(age_seconds: float) -> str:
    """صياغة عربية لعمر السجل (بالساعات أو الأيام)."""
    age_hours = age_seconds / 3600.0
    if age_hours < 1:
        return f"أقل من ساعة"
    if age_hours < 24:
        return f"{age_hours:.1f} ساعة"
    return f"{age_hours / 24:.1f} يوم"


def _age_weight(ts: float, now: float,
                cfg: Optional[Dict[str, Any]] = None) -> float:
    """توهين زمني بأربع صيغ قابلة للتبديل من التهيئة.

    - exponential: منحنى نصف العمر الكلاسيكي الحالي (افتراضي).
    - window: أحداث أقدم من strict_window_hours تُستبعد كليًا.
    - sigmoid: وزن كامل ثم سقوط حاد حول عمر الوسط (half_life).
    - blended: وزن حديث (آخر recent_boost_hours = 1.0) ممزوج بوزن
      إجمالي مخفّض؛ يحفظ الأداء المتراكم طويل الأمد.
    """
    cfg = cfg if cfg is not None else _decay_config()
    if ts <= 0:
        return 1.0
    age = max(0.0, now - ts)
    if age <= 0:
        return 1.0
    half_life = float(cfg.get("half_life_hours", 48.0)) * 3600.0
    mode = str(cfg.get("decay_mode", "exponential"))

    if mode == "window":
        window = float(cfg.get("strict_window_hours", 72.0)) * 3600.0
        return 1.0 if age <= window else 0.0
    if mode == "sigmoid":
        # سقوط حاد حول عمر الوسط: σ(−k × (age − half_life)) مع k حاد
        k = 8.0 / max(1e-9, half_life)
        return 1.0 / (1.0 + math.exp(k * (age - half_life)))
    if mode == "blended":
        # نصيب حديث كامل (آخر half_life) + نصيب تراكمي مخفّض
        recent = 1.0 - min(1.0, age / half_life) if age <= half_life else 0.0
        legacy = 0.5 ** (age / max(1e-9, half_life))
        return 0.5 * recent + 0.5 * legacy
    # exponential (الافتراضي — السلوك الأصلي)
    return 0.5 ** (age / max(1e-9, half_life))


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

    تحسينات التوهين (من config/adaptive_swarm.json):
        - decay_mode: exponential|window|sigmoid|blended
        - recent_boost: تعزيز نسبة فشل أحداث آخر 24 ساعة عند الاستبعاد
        - activity_decay: وكيل نشط حديثًا (داخل activity_window)
          تفقد سجلاته وزنها بنصف عمر نشاطي أقصر
        - bucket_minutes: تجميع المدد الزمنية في شرائح ومتوسطها
    """
    now = time.time()
    cfg = _decay_config()
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
        w = _age_weight(ts, now, cfg)
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
                prof["durations"].append((d, w, ts))
            except (TypeError, ValueError):
                pass

    # توهين النشاط: وكيل نشط حديثًا تفقد سجلاته وزنها أسرع.
    # وكيل نفّذ مهمة واحدة على الأقل داخل نافذة النشاط يُعتبر نشطًا
    # فتوزن أحداثه بمنحنى نصف عمر نشاطي أقصر (أسرع نسيانًا).
    if cfg.get("activity_decay", False):
        activity_window = float(cfg.get("activity_window_hours", 72.0)) * 3600.0
        activity_half_life = float(cfg.get("activity_half_life_hours", 24.0)) * 3600.0
        for prof in profiles.values():
            if not prof.get("_w_sum", 0.0):
                continue
            latest = prof.get("latest_ts", 0.0)
            is_recently_active = latest > 0 and (now - latest) <= activity_window
            if is_recently_active:
                scale = 0.5 ** (activity_window / max(1e-9, activity_half_life))
                prof["_w_sum"] = prof.get("_w_sum", 0.0) * scale
                prof["done"] = prof.get("done", 0.0) * scale
                prof["errors"] = prof.get("errors", 0.0) * scale
                prof["start_count"] = prof.get("start_count", 0.0) * scale
                prof["tasks"] = prof.get("tasks", 0.0) * scale
                prof["durations"] = [
                    (d, w * scale, dts) for d, w, dts in prof.get("durations", [])]
                prof["_activity_decay_applied"] = True

    # درجات موحدة 0-100
    for prof in profiles.values():
        tasks = prof.get("tasks", 0) or (prof.get("start_count", 0) or 1e-9)
        done = prof.get("done", 0.0)
        errors = prof.get("errors", 0.0)
        denom = tasks + errors or 1e-9
        success_rate = min(1.0, max(0.0, done / denom))
        durations = prof.get("durations", [])
        if cfg.get("bucket_minutes", 0) and durations:
            # 🆕 تجميع زمني: تُجمع المدد في شرائح زمنية (مثلًا ساعة)
            # بناءً على طابع حدثها، ويُحتسب متوسط مرجّح لكل شريحة
            # ثم متوسط الشرائح — يقلل أثر تكرار الأحداث المتفجرة
            # ويثبت النتيجة.
            bucket_sec = float(cfg["bucket_minutes"]) * 60.0
            buckets: Dict[int, List[Tuple[float, float]]] = {}
            for d, w, dts in durations:
                bucket_key = int(dts // bucket_sec)
                buckets.setdefault(bucket_key, []).append((d, w))
            per_bucket = []
            for items in buckets.values():
                w_total = sum(w for _, w in items) or 1e-9
                avg = sum(d * w for d, w in items) / w_total
                per_bucket.append(avg)
            avg_dur_s = (sum(per_bucket) / max(1, len(per_bucket))) / 1000.0
            prof["bucketed"] = True
        else:
            durations = [d for d, w, _ in prof.get("durations", [])]
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

    # 🆕 تعزيز الأحداث الحديثة (recent boost) في القرارات الحرجة:
    # وكيل نشاطه الأخير ضمن recent_boost_hours يُمنح عامل تعزيز
    # محدودًا عند احتساب نسبة الفشل للاستبعاد — لا يغيّر الدرجة العامة.
    boost_hours = float(cfg.get("recent_boost_hours", 24.0)) * 3600.0
    boost_factor = float(cfg.get("recent_boost_factor", 2.0))
    for prof in profiles.values():
        latest = prof.get("latest_ts", 0.0)
        age_of_latest = (now - latest) if latest > 0 else 0.0
        prof["recency_age_s"] = max(0.0, age_of_latest)
        prof["_recent_boost"] = (
            min(boost_factor, max(1.0, 1.0 + (boost_factor - 1.0)
                                  * max(0.0, 1.0 - age_of_latest / max(1e-9, boost_hours))))
            if age_of_latest <= boost_hours else 1.0)

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
        # تعزيز الأحداث الحديثة: نسبة فشل مرجّحة لأحداث آخر 24 ساعة
        errors = prof.get("errors", 0.0) * prof.get("_recent_boost", 1.0)
        failure_rate = (errors / (tasks + 0.0)) if tasks else 0.0
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
        errors = prof.get("errors", 0.0) * prof.get("_recent_boost", 1.0)
        failure_rate = (errors / (tasks + 0.0)) if tasks else 0.0
        if tasks >= MIN_TASKS_FOR_EXCLUSION and failure_rate >= EXCLUSION_FAILURE_RATE:
            out.append((key, round(failure_rate, 3), float(tasks)))
    return out


def decay_curve_summary(sample_ages_hours: Optional[List[float]] = None,
                        cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """ملخص منحنى التوهين النشط لأغراض العرض والقياس.

    يعيد الأوزان لأعمار نموذجية (بالساعات) وفق صيغة التوهين الحالية
    في التهيئة، ليتمكن المستخدم والاختبارات من رؤية شكل المنحنى
    ومقارنة الصيغ الأربع دون تشغيل المحاكاة.
    """
    cfg = cfg if cfg is not None else _decay_config()
    now = time.time()
    ages = sample_ages_hours or [0.5, 2, 6, 12, 24, 48, 72, 120, 168]
    weights = {}
    for age_h in ages:
        ts = now - age_h * 3600.0
        weights[age_h] = round(_age_weight(ts, now, cfg), 4)
    return {
        "mode": str(cfg.get("decay_mode", "exponential")),
        "half_life_hours": float(cfg.get("half_life_hours", 48.0)),
        "recent_boost_hours": float(cfg.get("recent_boost_hours", 24.0)),
        "strict_window_hours": float(cfg.get("strict_window_hours", 72.0)),
        "activity_decay": bool(cfg.get("activity_decay", False)),
        "weights_by_age_hours": weights,
    }


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
