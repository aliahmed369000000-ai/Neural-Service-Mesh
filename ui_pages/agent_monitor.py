"""لوحة مراقبة حيّة لتفاعل agents داخل جلسة Streamlit."""
from __future__ import annotations

from app_core import *  # noqa: F401,F403


def _status_label(status: str) -> str:
    return {
        "running": "⏳ يعمل",
        "done": "✅ اكتمل",
        "error": "❌ فشل",
        "waiting": "⏸️ ينتظر",
    }.get(status, status or "—")


def _severity_label(severity: str) -> str:
    return {
        "critical": "🚨 حرج",
        "warning": "⚠️ تحذير",
        "info": "ℹ️ معلومات",
    }.get(severity, severity or "تنبيه")


def render_agent_monitor() -> None:
    """يعرض آخر حالة معروفة لكل وكيل وسجل الأحداث الزمني للجلسة."""
    from ai.agent_event_bus import (
        analyze_alerts,
        clear_events,
        current_agent_states,
        get_events,
        performance_summary,
    )

    st.markdown(
        """
        <div style="text-align:center;padding:1rem 0 0.5rem">
            <div style="font-size:1.6rem;font-weight:900;color:var(--gold)">📡 مراقبة الوكلاء لحظة بلحظة</div>
            <div style="color:var(--text-muted);font-size:.86rem;direction:rtl">
                تتبع التوجيه، التفويض، تشغيل الوكلاء، التوليف، والأخطاء داخل جلسة المحادثة الحالية.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_refresh, col_clear, col_limit = st.columns([1, 1, 1.4])
    with col_refresh:
        if st.button("🔄 تحديث الآن", key="agent_monitor_refresh", use_container_width=True):
            st.rerun()
    with col_clear:
        if st.button("🧹 مسح السجل", key="agent_monitor_clear", use_container_width=True):
            clear_events()
            st.rerun()
    with col_limit:
        limit = st.slider("عدد الأحداث", 10, 120, 50, key="agent_monitor_limit")
    threshold_cols = st.columns(2)
    with threshold_cols[0]:
        slow_threshold_ms = st.number_input(
            "عتبة البطء (مللي ثانية)", min_value=1000, max_value=120000,
            value=12000, step=1000, key="agent_slow_threshold_ms",
        )
    with threshold_cols[1]:
        stale_threshold_s = st.number_input(
            "عتبة الاختناق (ثانية)", min_value=5, max_value=600,
            value=45, step=5, key="agent_stale_threshold_s",
        )

    events = get_events(limit)
    states = current_agent_states(events)
    running = sum(1 for row in states.values() if row.get("status") == "running")
    completed = sum(1 for row in states.values() if row.get("status") == "done")
    failures = sum(1 for row in events if row.get("status") == "error")
    alerts = analyze_alerts(
        events,
        slow_threshold_ms=float(slow_threshold_ms),
        stale_threshold_s=float(stale_threshold_s),
    )
    performance = performance_summary(events)

    if alerts:
        st.markdown("#### 🚨 التنبيهات الفورية")
        for alert in reversed(alerts[-8:]):
            message = f"**{_severity_label(alert['severity'])}: {alert['title']}** — {alert['detail']}"
            if alert["severity"] == "critical":
                st.error(message)
            elif alert["severity"] == "warning":
                st.warning(message)
            else:
                st.info(message)
    else:
        st.success("✅ لا توجد أخطاء أو اختناقات تتجاوز العتبات الحالية.")

    metric_cols = st.columns(6)
    metric_cols[0].metric("الأحداث", len(events))
    metric_cols[1].metric("وكلاء نشطون", running)
    metric_cols[2].metric("مكتملة", completed)
    metric_cols[3].metric("أخطاء", failures, delta_color="inverse")
    metric_cols[4].metric("متوسط الزمن", f"{performance['avg_ms']:.0f} ms")
    metric_cols[5].metric("أقصى زمن", f"{performance['max_ms']:.0f} ms")

    if not events:
        st.info("لا توجد أحداث بعد. نفّذ مهمة من تبويب «منسّق الوكلاء» أو «الوكيل الموحّد» لتظهر هنا.")
        return

    st.markdown("#### الحالة الحالية لكل وكيل")
    cards = st.columns(min(4, max(1, len(states))))
    for card, (agent_id, row) in zip(cards, states.items()):
        with card:
            title = row.get("title") or agent_id
            st.markdown(f"**{title}**")
            st.caption(f"`{agent_id}` · {_status_label(row.get('status', ''))}")
            st.caption(f"آخر تحديث: {row.get('timestamp', '—')}")

    st.markdown("#### السجل الزمني للتفاعل")
    table = []
    for row in reversed(events):
        table.append({
            "الوقت": row.get("timestamp", "—"),
            "الحدث": row.get("event_type", "—"),
            "الوكيل": row.get("title") or row.get("agent_id") or "المدير",
            "الحالة": _status_label(row.get("status", "")),
            "زمن التنفيذ": f"{row['duration_ms']:.0f} ms" if row.get("duration_ms") is not None else "—",
            "التفاصيل": row.get("detail", ""),
        })
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_agent_live_trace(target) -> None:
    """يرسم نسخة مختصرة داخل مسار التنفيذ وتُحدّث بعد كل مرحلة."""
    from ai.agent_event_bus import analyze_alerts, get_events

    events = get_events(24)
    with target.container():
        st.markdown("#### 📡 التنفيذ الحي")
        if not events:
            st.caption("بانتظار بدء الأحداث...")
            return
        live_alerts = analyze_alerts(events, slow_threshold_ms=12000, stale_threshold_s=45)
        for alert in reversed(live_alerts[-3:]):
            text = f"{_severity_label(alert['severity'])}: {alert['title']} — {alert['detail']}"
            if alert["severity"] == "critical":
                st.error(text)
            else:
                st.warning(text)
        for row in reversed(events[-10:]):
            status = _status_label(row.get("status", ""))
            title = row.get("title") or row.get("agent_id") or "المدير"
            detail = f" — {row['detail']}" if row.get("detail") else ""
            st.caption(f"`{row.get('timestamp', '—')}` {status} **{title}** · {row.get('event_type', 'event')}{detail}")
