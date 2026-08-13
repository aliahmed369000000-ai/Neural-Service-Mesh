"""سجل أحداث حيّ لمراقبة تفاعل الوكلاء داخل جلسة Streamlit.

لا يعتمد على قاعدة بيانات أو مزود خارجي؛ الأحداث تخص جلسة المستخدم الحالية فقط،
وتُقصّ تلقائياً حتى لا تتضخم ذاكرة الصفحة أثناء المحادثات الطويلة.
"""
from __future__ import annotations

from datetime import datetime, timezone
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
    event = {
        "timestamp": datetime.now(timezone.utc).astimezone().strftime("%H:%M:%S"),
        "event_type": str(event_type),
        "agent_id": str(agent_id or ""),
        "title": str(title or event_type),
        "status": str(status or "running"),
        "detail": str(detail or ""),
        "metadata": dict(metadata or {}),
    }
    state = _state()
    if state is not None:
        events = list(state.get(EVENTS_KEY, []))
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


def current_agent_states(events: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Dict[str, Any]]:
    """يبني آخر حالة معروفة لكل وكيل من سجل الأحداث."""
    states: Dict[str, Dict[str, Any]] = {}
    for event in events if events is not None else get_events():
        agent_id = event.get("agent_id") or "orchestrator"
        states[agent_id] = event
    return states


__all__ = ["emit_event", "get_events", "clear_events", "current_agent_states"]
