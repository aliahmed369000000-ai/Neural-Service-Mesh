# -*- coding: utf-8 -*-
"""طبقة قياس زمن الاستجابة (Performance Profiling) — NSM.

توفر تزيين `measure_latency` يقيس زمن تنفيذ أي دالة ويخزّن العينات في
session_state (بمفتاح `_nsm_perf_samples`) مع حد أقصى، ويرسل عينة على
ناقل أحداث الوكلاء كحدث من نوع `perf_sample` لتظهر في المراقبة الحية.

وظائف إحصائية: `perf_stats` تحسب p50/p90/p95/المتوسط/الأقصى لآخر N عينة،
و`perf_slowest` ترتّب الدوال حسب أبطأ زمن p95. كل ذلك يعمل محليًا بالكامل
دون أي مفتاح API، ويمكن الاختبار خارج سياق Streamlit (يعمل بصمت).

إعداد: Manus AI — البند 1 من خارطة التطوير (قياس زمن الاستجابة).
"""
from __future__ import annotations

import functools
import statistics
from typing import Any, Callable, Dict, List, Optional

MAX_PERF_SAMPLES = 300  # حد عينات القياس في الجلسة الواحدة
PERF_SAMPLES_KEY = "_nsm_perf_samples"
PERF_EVENT_TYPE = "perf_sample"


def _state() -> Optional[Dict[str, Any]]:
    """session_state إن وجدت، وإلا None (اختبارات خارج Streamlit)."""
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return None


def get_perf_samples(limit: int = 50) -> List[Dict[str, Any]]:
    """آخر `limit` عينة قياس مرتبة من الأحدث إلى الأقدم."""
    state = _state()
    if state is None:
        return []
    samples = state.get(PERF_SAMPLES_KEY, [])
    limit = max(1, int(limit))
    tail = samples[-limit:]
    return list(reversed(tail))


def clear_perf_samples() -> None:
    """يمسح سجل عينات القياس."""
    state = _state()
    if state is not None:
        state[PERF_SAMPLES_KEY] = []


def _record_sample(sample: Dict[str, Any]) -> None:
    """يخزّن العينة في session_state ويرسلها على ناقل الأحداث."""
    state = _state()
    if state is None:
        return
    samples = state.setdefault(PERF_SAMPLES_KEY, [])
    samples.append(sample)
    if len(samples) > MAX_PERF_SAMPLES:
        state[PERF_SAMPLES_KEY] = samples[-MAX_PERF_SAMPLES:]
    # حدث على ناقل الأحداث لتظهر في المراقبة الحية وضمن نظام التنبيهات.
    try:
        from ai.agent_event_bus import emit_event
        emit_event(
            PERF_EVENT_TYPE,
            agent_id="profiler",
            title=f"⏱ {sample['func']}",
            status="done",
            detail=f"z={sample['ms']}ms (p50={sample['p50']}ms)",
            metadata={
                "func": sample["func"],
                "ms": sample["ms"],
                "count": sample.get("count", 0),
                "p50": sample["p50"],
                "p90": sample["p90"],
                "p95": sample["p95"],
            },
        )
    except Exception:
        # خارج سياق Streamlit أو مع حدث مفقود: نفشل بصمت.
        pass


def _percentile(values: List[float], pct: float) -> float:
    """المئة المئوية بالترقيم الخطي (معيار NumPy الافتراضي) مع حماية من
    القوائم القصيرة — quantiles(values, n=100) غير دقيقة للقوائم القصيرة
    (مثلاً [100, 200] تعطي 99 نقطة تجعل p95 = 285 خارج النطاق!)."""
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return float(s[lo] + frac * (s[hi] - s[lo]))


def perf_stats(samples: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """إحصاءات p50/p90/p95 للمتوسط العام + لكل دالة على حدة.

    `samples` إن لم تُمرَّر تُقرأ تلقائيًا من الجلسة (آخر 300 عينة).
    """
    rows = samples if samples is not None else list(
        reversed(_state().get(PERF_SAMPLES_KEY, [])) if _state() else []
    )
    ms_all = [float(r["ms"]) for r in rows]
    by_func: Dict[str, List[float]] = {}
    for r in rows:
        by_func.setdefault(str(r.get("func", "unknown")), []).append(float(r["ms"]))
    funcs = [
        {
            "func": name,
            "count": len(vals),
            "avg_ms": round(sum(vals) / len(vals), 1),
            "p50_ms": round(_percentile(vals, 50), 1),
            "p90_ms": round(_percentile(vals, 90), 1),
            "p95_ms": round(_percentile(vals, 95), 1),
            "max_ms": round(max(vals), 1),
        }
        for name, vals in sorted(by_func.items(), key=lambda kv: len(kv[1]), reverse=True)
    ]
    return {
        "sample_count": len(ms_all),
        "avg_ms": round(sum(ms_all) / len(ms_all), 1) if ms_all else 0.0,
        "p50_ms": round(_percentile(ms_all, 50), 1),
        "p90_ms": round(_percentile(ms_all, 90), 1),
        "p95_ms": round(_percentile(ms_all, 95), 1),
        "max_ms": round(max(ms_all), 1) if ms_all else 0.0,
        "by_func": funcs,
    }


def perf_slowest(top: int = 5, samples: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """أبطأ الدوال حسب p95 — المرشح الأول لتحسين الأداء."""
    stats = perf_stats(samples)
    return sorted(stats["by_func"], key=lambda f: f["p95_ms"], reverse=True)[:max(1, int(top))]


def measure_latency(label: str) -> Callable[..., Callable[..., Any]]:
    """تزيين يقيس زمن تنفيذ الدالة ويسجّل عينة p95 متراكمة (نافذة آخر 60 عينة للدالة).

    الاستخدام:

        @measure_latency("search_knowledge")
        def search_knowledge(query: str) -> Dict: ...

    العينات تُجمَّع لكل دالة حتى 60 عينة (نافذة زمنية ثابتة الحجم) ثم تُحفظ
    عينات p50/p90/p95 المحسوبة في السجل العام.
    """
    if not isinstance(label, str) or not label:
        raise ValueError("تسمية القياس يجب أن تكون نصًا غير فارغ")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        window: List[float] = []

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            import time as _time
            t0 = _time.perf_counter()
            try:
                result = fn(*args, **kwargs)
            finally:
                elapsed_ms = (_time.perf_counter() - t0) * 1000.0
            window.append(elapsed_ms)
            if len(window) > 60:
                window.pop(0)
            _record_sample({
                "func": label,
                "ms": round(elapsed_ms, 2),
                "count": len(window),
                "p50": round(_percentile(window, 50), 2),
                "p90": round(_percentile(window, 90), 2),
                "p95": round(_percentile(window, 95), 2),
            })
            return result

        wrapper.__perf_label__ = label  # type: ignore[attr-defined]
        return wrapper

    return decorator
