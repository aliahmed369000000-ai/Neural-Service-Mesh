"""سجل أحداث حيّ لمراقبة تفاعل الوكلاء داخل جلسة Streamlit.

لا يعتمد على قاعدة بيانات أو مزود خارجي؛ الأحداث تخص جلسة المستخدم الحالية فقط،
وتُقصّ تلقائياً حتى لا تتضخم ذاكرة الصفحة أثناء المحادثات الطويلة.
"""
from __future__ import annotations

from datetime import datetime, timezone
from time import time
from collections import Counter
from typing import Any, Dict, List, Optional

EVENTS_KEY = "_nsm_agent_live_events"
MAX_EVENTS = 250


def _state():
    try:
        import streamlit as st
        return st.session_state
    except Exception:
        return None


def emit_event(
    event_type: str,
    *,
    agent_id: str = "",
    title: str = "",
    status: str = "running",
    detail: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """يسجل حدثاً ويعيده؛ يفشل بصمت خارج سياق Streamlit."""
    now = time()
    event = {
        "timestamp": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
        "ts": now,
        "event_type": str(event_type),
        "agent_id": str(agent_id or ""),
        "title": str(title or event_type),
        "status": str(status or "running"),
        "detail": str(detail or ""),
        "metadata": dict(metadata or {}),
        "duration_ms": None,
    }
    state = _state()
    if state is not None:
        events = list(state.get(EVENTS_KEY, []))
        if event_type in {"agent_done", "agent_error", "task_done", "task_error", "synthesis_done"}:
            for previous in reversed(events):
                if (
                    previous.get("agent_id") == event["agent_id"]
                    and previous.get("event_type") in {"agent_started", "task_started", "synthesis_started"}
                    and previous.get("ts") is not None
                ):
                    event["duration_ms"] = round(max(0.0, now - float(previous["ts"])) * 1000, 1)
                    break
        events.append(event)
        state[EVENTS_KEY] = events[-MAX_EVENTS:]
    return event


def get_events(limit: int = 80) -> List[Dict[str, Any]]:
    state = _state()
    if state is None:
        return []
    return list(state.get(EVENTS_KEY, []))[-max(1, int(limit)):]


def clear_events() -> None:
    state = _state()
    if state is not None:
        state[EVENTS_KEY] = []


def analyze_alerts(
    events: Optional[List[Dict[str, Any]]] = None,
    *,
    slow_threshold_ms: float = 12000,
    stale_threshold_s: float = 45,
) -> List[Dict[str, Any]]:
    """يستخرج تنبيهات قابلة للعرض من سجل الأحداث الحالي."""
    rows = list(events if events is not None else get_events(250))
    alerts: List[Dict[str, Any]] = []
    now = time()
    failures = Counter(row.get("agent_id") or "orchestrator" for row in rows if row.get("status") == "error")
    latest: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        latest[row.get("agent_id") or "orchestrator"] = row
    for row in rows:
        agent = row.get("title") or row.get("agent_id") or "المدير"
        if row.get("status") == "error":
            alerts.append({"severity": "critical", "title": f"فشل في {agent}", "detail": row.get("detail") or "حدث خطأ غير موصوف", "event": row})
        duration = row.get("duration_ms")
        if duration is not None and float(duration) >= slow_threshold_ms:
            severity = "critical" if float(duration) >= slow_threshold_ms * 2 else "warning"
            alerts.append({"severity": severity, "title": f"استجابة بطيئة من {agent}", "detail": f"زمن التنفيذ {float(duration):.0f} مللي ثانية", "event": row})
        agent_key = row.get("agent_id") or "orchestrator"
        if (
            row.get("status") == "running"
            and latest.get(agent_key) is row
            and row.get("ts") is not None
            and now - float(row["ts"]) >= stale_threshold_s
        ):
            alerts.append({"severity": "warning", "title": f"اختناق محتمل في {agent}", "detail": f"ما زال قيد التنفيذ منذ {now - float(row['ts']):.0f} ثانية", "event": row})
    for agent_id, count in failures.items():
        if count >= 2:
            label = next((r.get("title") for r in rows if r.get("agent_id") == agent_id), agent_id)
            alerts.append({"severity": "critical", "title": f"تكرار فشل {label}", "detail": f"سُجلت {count} أخطاء في السجل الحالي", "event": {"agent_id": agent_id}})
    running = [r for r in latest.values() if r.get("status") == "running"]
    if len(running) >= 3:
        alerts.append({"severity": "warning", "title": "ازدحام في التنفيذ", "detail": f"يوجد {len(running)} وكلاء في حالة تشغيل متزامنة", "event": {}})
    return alerts


def performance_summary(events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    rows = list(events if events is not None else get_events(250))
    durations = [float(r["duration_ms"]) for r in rows if r.get("duration_ms") is not None]
    return {
        "count": len(durations),
        "avg_ms": round(sum(durations) / len(durations), 1) if durations else 0.0,
        "max_ms": round(max(durations), 1) if durations else 0.0,
        "last_ms": round(durations[-1], 1) if durations else 0.0,
    }


def current_agent_states(events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """يبني آخر حالة معروفة لكل وكيل من سجل الأحداث."""
    states: Dict[str, Dict[str, Any]] = {}
    for event in events if events is not None else get_events():
        agent_id = event.get("agent_id") or "orchestrator"
        states[agent_id] = event
    return states


__all__ = ["emit_event", "get_events", "clear_events", "current_agent_states", "analyze_alerts", "performance_summary"]
