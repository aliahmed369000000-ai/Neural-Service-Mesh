"""لوحة مراقبة حيّة لتفاعل agents داخل جلسة Streamlit."""
from __future__ import annotations

from app_core import *  # noqa: F401,F403
from typing import Any, Dict

from ui_components import render_agent_cards, render_alert_cards, render_kpi_cards, render_section_header


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

    render_section_header(
        "مراقبة الوكلاء لحظة بلحظة",
        "التوجيه · التفويض · التنفيذ · التوليف",
        live=True,
    )
    st.caption("تتبّع حالة الشبكة العصبية داخل جلسة المحادثة الحالية بوضوح وفي الوقت الفعلي.")

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

    render_section_header("صحة الشبكة", "مؤشرات مباشرة من ناقل الأحداث")
    if alerts:
        render_alert_cards(alerts)
    else:
        st.success("لا توجد أخطاء أو اختناقات تتجاوز العتبات الحالية.")

    render_kpi_cards([
        {"label": "الأحداث", "value": len(events), "note": "في السجل الحالي", "accent": "var(--nsm-indigo)"},
        {"label": "وكلاء نشطون", "value": running, "note": "قيد التنفيذ", "accent": "var(--nsm-cyan)"},
        {"label": "مكتملة", "value": completed, "note": "دورات ناجحة", "accent": "#86efac"},
        {"label": "أخطاء", "value": failures, "note": "تحتاج مراجعة", "accent": "var(--nsm-danger)"},
        {"label": "متوسط الزمن", "value": f"{performance['avg_ms']:.0f} ms", "note": "زمن الاستجابة", "accent": "var(--nsm-amber)"},
        {"label": "أقصى زمن", "value": f"{performance['max_ms']:.0f} ms", "note": "أبطأ دورة", "accent": "#c084fc"},
    ])

    if not events:
        st.info("لا توجد أحداث بعد. نفّذ مهمة من تبويب «منسّق الوكلاء» أو «الوكيل الموحّد» لتظهر هنا.")
        return

    render_section_header("الحالة الحالية لكل وكيل", f"{len(states)} عقدة متصلة")
    render_agent_cards(states)

    render_section_header("السجل الزمني للتفاعل", "آخر الأحداث بالترتيب العكسي")
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

    render_delegation_chain(events)
    render_debate_journal(events)


DELEGATION_EVENTS = ("delegation_requested", "delegation_rejected", "delegation_started", "delegation_resolved")


def _delegation_status_label(status: str) -> str:
    return {
        "pending": "📤 طلب معلق",
        "running": "⚙️ قيد التنفيذ",
        "done": "✅ اكتمل",
        "fail": "❌ فشل",
        "error": "❌ مرفوض",
        "rejected": "🚫 مرفوض",
    }.get(status, status or "—")


def _delegation_icon(event_type: str) -> str:
    return {
        "delegation_requested": "📤",
        "delegation_rejected": "🚫",
        "delegation_started": "⚙️",
        "delegation_resolved": "📥",
    }.get(event_type, "🔗")


def render_delegation_chain(events) -> None:
    """يعرض سلسلة التفويض بين الوكلاء: مَن فوّض إلى مَن وبأي نتيجة."""
    chain = [e for e in events if e.get("event_type") in DELEGATION_EVENTS]
    render_section_header("سلسلة التفويض", f"{len(chain)} حدث تعاون بين الوكلاء")
    if not chain:
        st.caption("لا توجد طلبات تفويض في السجل الحالي. التفويض يظهر عندما يطلب وكيل متخصص مهمة فرعية من زميله بصيغة «⤴ DELEGATE»." )
        return
    # تجميع بالوكيل المفوَّض إليه مع آخر حالة معروفة له
    by_delegate: Dict[str, Dict[str, Any]] = {}
    for e in chain:
        dk = e.get("metadata", {}).get("delegate_key") or e.get("agent_id", "")
        entry = by_delegate.setdefault(dk, {
            "delegate_key": dk,
            "delegate_title": e.get("title"),
            "delegator_key": e.get("metadata", {}).get("delegator_key", ""),
            "delegator_title": "",
            "subtask": "",
            "events": [],
        })
        entry["events"].append(e)
        entry["delegate_title"] = e.get("title") or entry["delegate_title"]
        entry["delegator_key"] = e.get("metadata", {}).get("delegator_key", "") or entry["delegator_key"]
    table = []
    for entry in reversed(list(by_delegate.values())):
        evts = entry["events"]
        last = evts[-1]
        meta_last = last.get("metadata", {})
        delegator_label = (
            meta_last.get("delegator_title", "")
            or entry["delegator_key"]
            or (evts[0].get("metadata", {}).get("delegator_title", ""))
        )
        table.append({
            "الوكيل المفوَّض إليه": entry["delegate_title"] or entry["delegate_key"],
            "المفوِّض": delegator_label or entry["delegator_key"] or "—",
            "المهمة الفرعية": last.get("detail", "—") or "—",
            "المرحلة": f"{_delegation_icon(last.get('event_type', ''))} {last.get('event_type', '—').replace('delegation_', '')}",
            "الحالة": _delegation_status_label(last.get("status", "")),
            "الوقت": last.get("timestamp", "—"),
            "زمن التنفيذ": f"{last['duration_ms']:.0f} ms" if last.get("duration_ms") is not None else "—",
        })
    st.dataframe(table, use_container_width=True, hide_index=True)
    for e in reversed(chain):
        dk = e.get("metadata", {}).get("delegate_key", "")
        delegator = e.get("metadata", {}).get("delegator_key", "")
        arrow = "➜" if e.get("event_type") in ("delegation_started", "delegation_resolved", "delegation_rejected") else "📤"
        parties = f"{delegator} {arrow} {dk}" if delegator and dk else (e.get("title") or e.get("agent_id", ""))
        detail = f" — {e['detail']}" if e.get("detail") else ""
        st.caption(f"`{e.get('timestamp', '—')}` {_delegation_icon(e.get('event_type', ''))} **{parties}** · {e.get('status', '')}{detail}")


DEBATE_EVENTS = ("debate_started", "debate_argument", "debate_round_done", "debate_consensus", "debate_abandoned")

STANCE_BADGES = {
    "agree": "#86efac",
    "disagree": "var(--nsm-danger)",
    "enhance": "var(--nsm-cyan)",
    "partial": "var(--nsm-amber)",
}


def _debate_status_label(status: str) -> str:
    return {
        "running": "⏳ نقاش جارٍ",
        "done": "✅ اكتمل",
        "error": "❌ مُلغي",
        "agreed": "✅ مؤيد",
        "disagreed": "❌ معترض",
        "enhanced": "💡 مُثرٍ",
    }.get(status, status or "—")


def _debate_icon(event_type: str) -> str:
    return {
        "debate_started": "💬",
        "debate_argument": "✍️",
        "debate_round_done": "🔚",
        "debate_consensus": "🤝",
        "debate_abandoned": "🚫",
    }.get(event_type, "💬")


def render_debate_journal(events) -> None:
    """يعرض جلسات سرب المناقشة: من ساهم، بموقفه، وفي أي جولة."""
    journal = [e for e in events if e.get("event_type") in DEBATE_EVENTS]
    render_section_header("سرب المناقشة", f"{len(journal)} حدث من جولات النقاش بين الوكلاء")
    if not journal:
        st.caption("لا توجد جلسات نقاش في السجل الحالي. النقاش يظهر عند تفعيل «سرب المناقشة» في تبويب «🐝 السرب الذكي» مع مهمتين ناجحتين أو أكثر.")
        return
    table = []
    for e in reversed(journal):
        meta = e.get("metadata", {}) or {}
        stance = meta.get("stance", "")
        stance_label = meta.get("stance_label", "")
        badge_html = ""
        if stance:
            color = STANCE_BADGES.get(stance, "var(--text-muted)")
            label = stance_label or meta.get("stance", "")
            badge_html = (f'<span style="display:inline-block;padding:.15rem .5rem;'
                          f'border-radius:999px;font-size:.7rem;font-weight:700;'
                          f'color:#fff;background:{color}">{label}</span>')
        table.append({
            "الوقت": e.get("timestamp", "—"),
            "الحدث": _debate_icon(e.get("event_type", "")) + " " + e.get("event_type", "—").replace("debate_", ""),
            "الوكيل": e.get("title") or e.get("agent_id") or "—",
            "الجولة": meta.get("round_index", "—") if isinstance(meta.get("round_index"), int) else "—",
            "الموقف": badge_html or "—",
            "التفاصيل": (e.get("detail") or "")[:140],
        })
    st.dataframe(table, use_container_width=True, hide_index=True)
    started = next((e for e in journal if e.get("event_type") == "debate_started"), None)
    consensus = next((e for e in journal if e.get("event_type") == "debate_consensus"), None)
    if consensus:
        c = (consensus.get("metadata", {}) or {}).get("consensus", {}) or {}
        st.success(f"🤝 **القرار الموحّد:** {c.get('verdict', '—')}")
    elif started:
        st.warning("💬 المناقشة بدأت لكن لم يُعتمد قرار موحّد بعد (قد تكون قيد التنفيذ أو أُلغيت).")
    for e in reversed(journal):
        meta = e.get("metadata", {}) or {}
        if e.get("event_type") == "debate_argument":
            stance = meta.get("stance", "")
            label = meta.get("stance_label", stance)
            target = meta.get("target_agent", "")
            color = STANCE_BADGES.get(stance, "var(--text-muted)")
            target_txt = f" ➜ {target}" if target else ""
            st.caption(
                f"`{e.get('timestamp', '—')}` ✍️ **{e.get('agent_id', '')}** "
                f'<span style="display:inline-block;padding:.1rem .45rem;border-radius:999px;'
                f'font-size:.68rem;font-weight:700;color:#fff;background:{color}">{label}</span>'
                f"{target_txt} · {e.get('detail', '')}"
            )
        else:
            st.caption(f"`{e.get('timestamp', '—')}` {_debate_icon(e.get('event_type', ''))} {e.get('title', '')} · {e.get('status', '')}")


def render_agent_live_trace(target) -> None:
    """يرسم نسخة مختصرة داخل مسار التنفيذ وتُحدّث بعد كل مرحلة."""
    from ai.agent_event_bus import analyze_alerts, get_events

    events = get_events(24)
    with target.container():
        render_section_header("التنفيذ الحي", "تحديثات مرحلية داخل المهمة", live=True)
        if not events:
            st.caption("بانتظار بدء الأحداث...")
            return
        live_alerts = analyze_alerts(events, slow_threshold_ms=12000, stale_threshold_s=45)
        if live_alerts:
            render_alert_cards(live_alerts, limit=3)
        for row in reversed(events[-10:]):
            status = _status_label(row.get("status", ""))
            title = row.get("title") or row.get("agent_id") or "المدير"
            detail = f" — {row['detail']}" if row.get("detail") else ""
            st.caption(f"`{row.get('timestamp', '—')}` {status} **{title}** · {row.get('event_type', 'event')}{detail}")
