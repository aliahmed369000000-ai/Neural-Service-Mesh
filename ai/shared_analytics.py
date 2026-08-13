# -*- coding: utf-8 -*-
"""
ai/shared_analytics.py — التحليلات التشاركية وتقارير أداء الوكلاء
==================================================================
تجميع البيانات من المصادر الدائمة للنظام (ناقل الأحداث، سجل السرب،
سجل التوجيه، الذاكرة الجماعية، مهام الخلفية) لإنتاج تقرير أداء شامل
مع درجة عامة وتوصيات تحسين تلقائية بالعربية.

التصميم:
- SharedAnalyticsReporter: وحدة صرفة لا تعتمد على streamlit ولا على مفاتيح API.
  كل مصدر بيانات قابل للحقن (dependency injection) عبر `DataSources` ليظل
  كل مسار قابلًا للاختبار في بيئة محاكاة.
- التوصيات تُولَّد من قواعد حتمية على المقاييس (لا LLM اختياريًا) —
  لا يُختلق أي رقم أو توصية بلا دليل في البيانات.
- fallback آمن: أي مصدر متعذّر يتحوّل إلى سجل فارغ ولا يعطّل التقرير.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ai.shared_analytics")

# ── عتبات التحليل ──────────────────────────────────────────────────
FAILURE_RATE_WARNING = 0.25      # نسبة فشل الوكلاء فوقها تنبيه
SLOW_MEDIAN_MS = 8000            # متوسط المدة فوقه اختناق
FAILOVER_RATE_WARNING = 0.20     # نسبة failover فوقها تقلب مزود
QUALITY_LOW = 0.5                # متوسط جودة الدروس تحته توصية
MIN_EVENTS_FOR_REPORT = 3        # أقل عدد أحداث لتقرير ذكي


class DataSources:
    """حاوية دوال قراءة المصادر — قابلة للحقن للاختبار."""

    def __init__(
        self,
        get_events: Optional[Callable] = None,
        get_swarm_summary: Optional[Callable] = None,
        get_swarm_recent: Optional[Callable] = None,
        get_route_log: Optional[Callable] = None,
        get_memory_summary: Optional[Callable] = None,
        get_memory_lessons: Optional[Callable] = None,
        get_bg_status: Optional[Callable] = None,
        get_bg_tasks: Optional[Callable] = None,
    ):
        self.get_events = get_events
        self.get_swarm_summary = get_swarm_summary
        self.get_swarm_recent = get_swarm_recent
        self.get_route_log = get_route_log
        self.get_memory_summary = get_memory_summary
        self.get_memory_lessons = get_memory_lessons
        self.get_bg_status = get_bg_status
        self.get_bg_tasks = get_bg_tasks


def _safe_call(fn, default):
    try:
        return fn() if fn is not None else default
    except Exception as exc:
        logger.warning("shared_analytics: تعذّر مصدر بيانات: %s", exc)
        return default


def default_data_sources() -> DataSources:
    """يولّد DataSources مرتبطة بالمصادر الحقيقية للنظام."""
    return DataSources(
        get_events=lambda: _events_bus("get_events")(250),
        get_swarm_summary=lambda: _swarm_store("summary")(),
        get_swarm_recent=lambda: _swarm_store("get_recent")(20),
        get_route_log=lambda: _route_store()(),
        get_memory_summary=lambda: _memory("summary")(),
        get_memory_lessons=lambda: _memory("lessons_list")(50),
        get_bg_status=lambda: _bg_mgr("status")(),
        get_bg_tasks=lambda: _bg_mgr("list_tasks")(30),
    )


def _events_bus(name):
    from ai.agent_event_bus import get_events
    return get_events if name == "get_events" else lambda: []


def _swarm_store(method):
    from ai.swarm_history_store import get_default_swarm_store
    return getattr(get_default_swarm_store(), method)


def _route_store():
    from ai.route_log_store import get_recent
    return get_recent


def _memory(method):
    from ai.collective_memory import get_collective_memory
    return getattr(get_collective_memory(), method)


def _bg_mgr(method):
    from ai.background_tasks import get_background_task_manager
    return getattr(get_background_task_manager(), method)


# ══════════════════════════════════════════════════════════════════
#  المقاييس
# ══════════════════════════════════════════════════════════════════

def agent_performance(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    """أداء كل وكيل من سجل ناقل الأحداث."""
    agents: Dict[str, Dict[str, Any]] = {}
    for e in events:
        aid = (e.get("agent_id") or "orchestrator").strip()
        row = agents.setdefault(aid, {
            "agent_id": aid, "title": e.get("title", ""), "starts": 0,
            "done": 0, "errors": 0, "durations": [], "retries": 0,
        })
        et = e.get("event_type", "")
        if et in ("agent_started", "task_started", "synthesis_started",
                  "delegation_started", "debate_started", "bg_task_started",
                  "bg_task_running"):
            row["starts"] += 1
            if row["starts"] > 1:
                row["retries"] += 1
        if et in ("agent_done", "task_done", "synthesis_done",
                  "delegation_resolved", "debate_consensus", "bg_task_done"):
            row["done"] += 1
        if et in ("agent_error", "task_error", "bg_task_failed",
                  "delegation_rejected"):
            row["errors"] += 1
        d = e.get("duration_ms")
        if d is not None:
            try:
                row["durations"].append(float(d))
            except (TypeError, ValueError):
                pass
    result = []
    for row in agents.values():
        durs = row["durations"]
        done_total = row["done"] + row["errors"]
        result.append({
            "agent_id": row["agent_id"],
            "title": row["title"],
            "tasks": row["starts"],
            "done": row["done"],
            "errors": row["errors"],
            "retries": row["retries"],
            "avg_ms": round(sum(durs) / len(durs), 1) if durs else 0.0,
            "max_ms": round(max(durs), 1) if durs else 0.0,
            "failure_rate": round(row["errors"] / done_total, 3) if done_total else 0.0,
        })
    result.sort(key=lambda r: r["tasks"], reverse=True)
    avg_fail = (
        sum(r["failure_rate"] for r in result) / len(result)
        if result else 0.0
    )
    return {"agents": result, "average_failure_rate": round(avg_fail, 3)}


def swarm_health(summary_fn, recent_fn) -> Dict[str, Any]:
    """صحة السرب من السجل الدائم."""
    summary = _safe_call(summary_fn, {})
    if not isinstance(summary, dict):
        summary = {}
    total = summary.get("total_swarms", 0) or 0
    swarms = _safe_call(recent_fn, []) or []
    success_per = []
    if not isinstance(swarms, list):
        swarms = []
    statuses = []
    for s in swarms:
        if isinstance(s, dict):
            statuses.append(s.get("status", "unknown"))
            t = s.get("total_tasks", 0) or 0
            ok = s.get("success_count", 0) or 0
            if t > 0:
                success_per.append(ok / t)
    return {
        "total_swarms": total,
        "by_status": summary if summary else {
            "done": statuses.count("done"),
            "partial": statuses.count("partial"),
            "failed": statuses.count("failed"),
        },
        "average_task_success_rate": round(sum(success_per) / len(success_per), 3) if success_per else 0.0,
        "recent_samples": len(swarms),
    }


def routing_quality(recent_fn) -> Dict[str, Any]:
    """جودة قرارات التوجيه من السجل الدائم."""
    rows = _safe_call(recent_fn, []) or []
    rows = [r for r in rows if isinstance(r, dict)]
    if not rows:
        return {"sample": 0, "avg_latency_ms": 0.0, "success_rate": 0.0,
                "failover_rate": 0.0, "avg_quality_score": 0.0, "top_categories": []}
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    qualities = [r["quality_score"] for r in rows if r.get("quality_score") is not None]
    top = sorted({(r.get("category", "عام"), r.get("cat_icon", "")) for r in rows},
                 key=lambda c: sum(1 for r in rows if r.get("category") == c[0]),
                 reverse=True)[:5]
    return {
        "sample": len(rows),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0.0,
        "success_rate": round(sum(1 for r in rows if r.get("success")) / len(rows), 3),
        "failover_rate": round(sum(1 for r in rows if r.get("failover")) / len(rows), 3),
        "avg_quality_score": round(sum(qualities) / len(qualities), 3) if qualities else 0.0,
        "top_categories": [f"{icon} {name}" for name, icon in top],
    }


def bg_health(status_fn, tasks_fn) -> Dict[str, Any]:
    """صحة مهام الخلفية."""
    status = _safe_call(status_fn, {}) or {}
    tasks = _safe_call(tasks_fn, []) or []
    return {
        "total": status.get("total", 0),
        "done": status.get("done", 0),
        "running": status.get("running", 0),
        "failed": status.get("failed", 0),
        "recent_failed_ids": [t.get("task_id") for t in tasks if t.get("status") == "failed"][:5],
    }


def memory_health(summary_fn, lessons_fn) -> Dict[str, Any]:
    """صحة الذاكرة الجماعية."""
    summary = _safe_call(summary_fn, {}) or {}
    lessons = _safe_call(lessons_fn, []) or []
    quals = [l.get("quality", 0) or 0 for l in lessons if l.get("quality") is not None]
    return {
        "total_lessons": summary.get("total_lessons", len(lessons)),
        "avg_lesson_quality": round(sum(quals) / len(quals), 3) if quals else 0.0,
        "domains": summary.get("domains", {}),
        "top_domain": next(iter(summary.get("domains", {}) or {}), ""),
    }


# ══════════════════════════════════════════════════════════════════
#  الدرجة العامة والتوصيات
# ══════════════════════════════════════════════════════════════════

def composite_score(perf: Dict, swarm: Dict, routing: Dict,
                    bg: Dict, memory: Dict) -> Dict[str, Any]:
    """درجة عامة 0-100 مع توزيعها حسب المصادر (قواعد حتمية)."""
    components: Dict[str, float] = {}

    # أداء الوكلاء 40% — من معدل الفشل ومتوسط المدة
    fail = perf.get("average_failure_rate", 0.0) or 0.0
    agents = perf.get("agents", []) or []
    slow_agents = sum(1 for a in agents if a.get("avg_ms", 0) >= SLOW_MEDIAN_MS)
    perf_score = max(0.0, 100 * (1 - fail) - 5 * slow_agents)
    components["agents"] = round(min(100, perf_score), 1)

    # السرب 25%
    sr = swarm.get("average_task_success_rate", 0.0) or 0.0
    components["swarm"] = round(100 * sr, 1)

    # التوجيه 20% — من نسبة النجاح وجودة الرد
    qs = routing.get("avg_quality_score", 0.0) or 0.0
    # توحيد quality_score إلى 0-100 إن كان في مقياس آخر متعارف عليه (1-5)
    norm_q = qs * 20 if qs <= 5 else qs
    components["routing"] = round(min(100, (100 * (routing.get("success_rate", 0.0) or 0.0) + norm_q) / 2), 1)

    # الخلفية 10%
    bg_done = bg.get("done", 0) or 0
    bg_failed = bg.get("failed", 0) or 0
    bg_total = bg_done + bg_failed
    components["background"] = round(100 * bg_done / bg_total, 1) if bg_total else 100.0

    # الذاكرة 5%
    q = memory.get("avg_lesson_quality", 0.0) or 0.0
    # الجودة في [-1, 1] — نوحد إلى 0-100
    components["memory"] = round(50 + 50 * q, 1)

    total = (components["agents"] * 0.4 + components["swarm"] * 0.25
             + components["routing"] * 0.2 + components["background"] * 0.1
             + components["memory"] * 0.05)
    grade = (
        "ممتاز" if total >= 85 else
        "جيد جدًا" if total >= 70 else
        "جيد" if total >= 55 else
        "يحتاج تحسين" if total >= 40 else "ضعيف"
    )
    return {"total": round(total, 1), "grade": grade, "components": components}


def recommendations(perf: Dict, swarm: Dict, routing: Dict,
                    bg: Dict, memory: Dict, score: Dict) -> List[Dict[str, Any]]:
    """توصيات عربية حتمية مبنية على أدلة في المقاييس."""
    recs: List[Dict[str, Any]] = []
    agents = perf.get("agents", []) or []

    # 1. وكيل بمعدل فشل مرتفع
    for a in agents:
        if a.get("failure_rate", 0) >= FAILURE_RATE_WARNING:
            recs.append({
                "severity": "critical",
                "title": f"وكيل متكرر الفشل: {a.get('title', a.get('agent_id'))}",
                "detail": (f"نسبة فشل {a['failure_rate'] * 100:.0f}% على {a['tasks']} مهمة — "
                           f"راجع برومبته وحدود المزود لديه"),
                "agent": a.get("agent_id", ""),
            })

    # 2. اختناقات مدة
    for a in agents:
        if a.get("avg_ms", 0) >= SLOW_MEDIAN_MS and a.get("tasks", 0) >= 2:
            recs.append({
                "severity": "warning",
                "title": f"اختناق مدة: {a.get('title', a.get('agent_id'))}",
                "detail": f"متوسط المدة {a['avg_ms']:.0f} مللي ثانية — فكّر بتبسيط برومبته",
                "agent": a.get("agent_id", ""),
            })

    # 3. إعادة محاولات مفرطة
    for a in agents:
        if a.get("retries", 0) >= 3:
            recs.append({
                "severity": "warning",
                "title": f"إعادة محاولات مفرطة: {a.get('title', a.get('agent_id'))}",
                "detail": f"{a['retries']} إعادة محاولة — تحقق من استقرار المزود",
                "agent": a.get("agent_id", ""),
            })

    # 4. صحة السرب
    sr = swarm.get("average_task_success_rate", 0.0) or 0.0
    if swarm.get("total_swarms", 0) and sr < 0.6:
        recs.append({
            "severity": "critical",
            "title": "معدل نجاح مهام السرب منخفض",
            "detail": f"{sr * 100:.0f}% من المهام الفرعية تنجح — راجع قواعد التفكيك",
            "agent": "swarm",
        })

    # 5. تقلب المزود في التوجيه
    fr = routing.get("failover_rate", 0.0) or 0.0
    if routing.get("sample", 0) and fr >= FAILOVER_RATE_WARNING:
        recs.append({
            "severity": "warning",
            "title": "تقلب مزوّد النموذج",
            "detail": f"{fr * 100:.0f}% من الطلبات انتقلت لمزوّد بديل — فعّل مزودًا احتياطيًا دائمًا",
            "agent": "router",
        })

    # 6. جودة التوجيه المنخفضة
    qs = routing.get("avg_quality_score", 0.0) or 0.0
    if routing.get("sample", 0) and 0 < qs <= QUALITY_LOW:
        recs.append({
            "severity": "warning",
            "title": "جودة الردود تحت العتبة",
            "detail": f"متوسط {qs:.2f} — فعّل التقييم الذاتي والتعزيز الجماعي",
            "agent": "router",
        })

    # 7. مهام خلفية فاشلة
    if bg.get("failed", 0) >= 2:
        recs.append({
            "severity": "critical",
            "title": "تكرار فشل مهام الخلفية",
            "detail": f"{bg['failed']} مهام فاشلة — تحقق من السجل (data/background_tasks.db)",
            "agent": "background",
        })

    # 8. جودة الدروس الجماعية
    mq = memory.get("avg_lesson_quality", 0.0) or 0.0
    if memory.get("total_lessons", 0) and mq < QUALITY_LOW:
        recs.append({
            "severity": "warning",
            "title": "جودة الدروس الجماعية منخفضة",
            "detail": f"متوسط الجودة {mq:.2f} — راجع تصنيف الدروس وصوّت على الضعيف",
            "agent": "memory",
        })

    # 9. عدم كفاية البيانات
    if not any(r["agent"] for r in recs) and not agents and not routing.get("sample"):
        recs.append({
            "severity": "info",
            "title": "لا بيانات كافية بعد",
            "detail": "نفّذ مهام على الوكلاء لتوليد تقرير تحليلات كامل",
            "agent": "",
        })
        return recs

    # 10. إيجابية عامة
    if not recs:
        recs.append({
            "severity": "info",
            "title": "الشبكة بحالة صحية",
            "detail": f"الدرجة {score.get('total', 0)} — استمر بالمراقبة الدورية",
            "agent": "",
        })
    recs.sort(key=lambda r: {"critical": 0, "warning": 1, "info": 2}.get(r["severity"], 3))
    return recs


# ══════════════════════════════════════════════════════════════════
#  التقرير الشامل
# ══════════════════════════════════════════════════════════════════

class SharedAnalyticsReporter:
    """منشئ تقارير التحليلات التشاركية — singleton اختياري."""

    def __init__(self, sources: Optional[DataSources] = None):
        self.sources = sources or default_data_sources()

    def report(self) -> Dict[str, Any]:
        perf = agent_performance(_safe_call(self.sources.get_events, []))
        swarm = swarm_health(self.sources.get_swarm_summary, self.sources.get_swarm_recent)
        routing = routing_quality(self.sources.get_route_log)
        bg = bg_health(self.sources.get_bg_status, self.sources.get_bg_tasks)
        memory = memory_health(self.sources.get_memory_summary, self.sources.get_memory_lessons)
        score = composite_score(perf, swarm, routing, bg, memory)
        recs = recommendations(perf, swarm, routing, bg, memory, score)
        return {
            "score": score,
            "agents": perf,
            "swarm": swarm,
            "routing": routing,
            "background": bg,
            "memory": memory,
            "recommendations": recs,
            "generated_at": __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M"),
        }


_reporter: Optional[SharedAnalyticsReporter] = None


def get_shared_analytics_reporter(sources: Optional[DataSources] = None) -> SharedAnalyticsReporter:
    global _reporter
    if _reporter is None:
        _reporter = SharedAnalyticsReporter(sources)
    return _reporter


__all__ = [
    "DataSources",
    "SharedAnalyticsReporter",
    "get_shared_analytics_reporter",
    "default_data_sources",
    "agent_performance",
    "swarm_health",
    "routing_quality",
    "bg_health",
    "memory_health",
    "composite_score",
    "recommendations",
    "FAILURE_RATE_WARNING",
    "SLOW_MEDIAN_MS",
    "FAILOVER_RATE_WARNING",
    "QUALITY_LOW",
]
