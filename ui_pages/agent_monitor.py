"""لوحة مراقبة حيّة لتفاعل agents داخل جلسة Streamlit."""
from __future__ import annotations

from html import escape as _escape

from app_core import *  # noqa: F401,F403
from typing import Any, Dict

from ui_components import render_agent_cards, render_alert_cards, render_kpi_cards, render_section_header
from ai.telemetry_store import TelemetryStore


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

    telemetry_days = st.select_slider("النطاق الزمني", options=["اليوم", "7 أيام", "30 يومًا", "كل السجل"], value="7 أيام", key="agent_telemetry_days")
    route_filter = st.selectbox("تصفية المسار", ["الكل"], key="agent_telemetry_route")
    import time as _time
    _days = {"اليوم": 1, "7 أيام": 7, "30 يومًا": 30}.get(telemetry_days)
    _store = TelemetryStore()
    _persistent = _store.query(since=_time.time() - _days * 86400 if _days else None, route=route_filter, limit=limit)
    events = get_events(limit)
    # عند توفر telemetry دائم، اجعله مصدر العرض؛ وإلا نستخدم سجل الجلسة الحالي.
    if _persistent:
        events = [{"timestamp": _time.strftime("%Y-%m-%d %H:%M:%S", _time.localtime(x["created_at"])), "agent_id": x["agent"], "title": x["agent"], "event_type": x["route"], "status": "done" if x["status"] == "success" else x["status"], "duration_ms": x["latency_ms"], "metadata": x.get("metadata", {})} for x in reversed(_persistent)]
    states = current_agent_states(events)
    running = sum(1 for row in states.values() if row.get("status") == "running")
    completed = sum(1 for row in states.values() if row.get("status") == "done")
    failures = sum(1 for row in events if row.get("status") == "error")
    alerts = analyze_alerts(events, slow_threshold_ms=float(slow_threshold_ms), stale_threshold_s=float(stale_threshold_s))
    for message in _store.alerts(since=_time.time() - _days * 86400 if _days else None, latency_ms=float(slow_threshold_ms)):
        alerts.append({"severity": "warning", "title": "تنبيه telemetry دائم", "detail": message})
    performance = performance_summary(events)
    _latest = events[-1] if events else {}
    _latest_agent = _escape(str(_latest.get("title") or _latest.get("agent_id") or "لا يوجد نشاط بعد"))
    _latest_event = _escape(str(_latest.get("event_type") or "بانتظار أول حدث"))
    _latest_status = _escape(_status_label(str(_latest.get("status", "waiting"))))
    st.markdown(
        f'<div class="nsm-agent-monitor-hero">'
        f'<div><div class="nsm-agent-monitor-eyebrow">● مراقبة حيّة داخل الجلسة</div>'
        f'<div class="nsm-agent-monitor-title">خريطة الوكلاء في نظرة واحدة</div>'
        f'<div class="nsm-agent-monitor-copy">تابع الوكلاء النشطين، آخر الأحداث، وأزمنة التنفيذ من دون مغادرة صفحة المراقبة.</div></div>'
        f'<div class="nsm-agent-monitor-chip">آخر نشاط: {_latest_agent} · {_latest_status} · {_latest_event}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

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

    # ملخص تشغيلي سريع: يوضح أين تذهب الطلبات وكيف يتغير الأداء.
    from collections import Counter
    route_counts = Counter(
        str((row.get("metadata") or {}).get("route") or row.get("event_type") or "غير مصنف")
        for row in events
    )
    route_counts = dict(route_counts.most_common(8))
    chart_cols = st.columns([1.15, 1])
    with chart_cols[0]:
        render_section_header("توزيع المسارات", "أكثر أنواع العمل ظهورًا في السجل")
        if route_counts:
            st.bar_chart(route_counts, height=180, use_container_width=True)
        else:
            st.caption("ستظهر المسارات بعد تنفيذ أول مهمة.")
    with chart_cols[1]:
        render_section_header("آخر لقطة تشغيل", "ملخص سريع للحالة الحالية")
        st.markdown(
            f'<div class="nsm-dashboard-panel" dir="rtl">'
            f'<div class="nsm-dashboard-panel-title">{_latest_agent}</div>'
            f'<div class="nsm-dashboard-panel-copy">الحالة: <strong>{_latest_status}</strong><br>'
            f'الحدث: <code>{_latest_event}</code><br>'
            f'زمن آخر دورة: <strong>{performance["last_ms"]:.0f} ms</strong></div></div>',
            unsafe_allow_html=True,
        )

    if not events:
        st.info("لا توجد أحداث بعد. نفّذ مهمة من تبويب «منسّق الوكلاء» أو «الوكيل الموحّد» لتظهر هنا.")
        return

    _filter_labels = {"all": "كل الحالات", "running": "⏳ يعمل", "done": "✅ اكتمل", "error": "❌ فشل", "waiting": "⏸️ ينتظر"}
    _status_filter = st.selectbox(
        "تصفية الوكلاء حسب الحالة",
        options=list(_filter_labels),
        format_func=lambda value: _filter_labels.get(value, value),
        key="agent_monitor_status_filter",
    ) or "all"
    _visible_states = {
        agent_id: row for agent_id, row in states.items()
        if _status_filter == "all" or row.get("status") == _status_filter
    }
    _visible_events = [
        row for row in events
        if _status_filter == "all" or row.get("status") == _status_filter
    ]

    render_section_header("الحالة الحالية لكل وكيل", f"{len(_visible_states)} من {len(states)} عقدة متصلة")
    if _visible_states:
        render_agent_cards(_visible_states)
    else:
        st.info("لا يوجد وكيل بالحالة المحددة في السجل الحالي.")

    render_section_header("السجل الزمني للتفاعل", "آخر الأحداث بالترتيب العكسي")
    table = []
    for row in reversed(_visible_events):
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
    render_collective_memory_panel()
    render_background_tasks_panel()
    render_shared_analytics_panel()
    render_adaptive_swarm_panel()
    render_failure_learning_panel()
    render_custom_alerts_panel()
    render_auto_actions_panel()
    render_perf_panel()
    render_swarm_runner_panel()
    render_long_horizon_panel()
    render_collaborative_panel()
    render_skb_panel()
    render_tem_panel()
    render_ltg_panel()
    render_par_panel()


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
ADAPTIVE_EVENTS = ("adaptive_ranked", "adaptive_excluded", "adaptive_reweighted")

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


def render_collective_memory_panel() -> None:
    """لوحة الدروس المستفادة جماعيًا بين الوكلاء عبر الجلسات (ذاكرة دائمة)."""
    from ai.collective_memory import get_collective_memory

    try:
        _cm = get_collective_memory()
        summary = _cm.summary()
        lessons = _cm.lessons_list(limit=20)
    except Exception:
        st.caption("الذاكرة الجماعية غير متاحة حاليًا (فشل التحميل).")
        return

    total = summary.get("total_lessons", 0)
    render_section_header(
        "الذاكرة الجماعية",
        f"{total} درسًا مستفادًا — يتشاركها الوكلاء بين الجلسات",
        live=False,
    )
    if total == 0:
        st.caption(
            "لا توجد دروس مستفادة بعد. تُسجّل تلقائيًا من نتائج مهام "
            "السرب الناجحة والفاشلة، ثم تُحقن ضمن برومبت المدير الموحّد "
            "عند المهام الجديدة ذات الصلة."
        )
        return

    col1, col2 = st.columns(2)
    with col1:
        _top = summary.get("top_lessons", []) or []
        if _top:
            for t in _top[:3]:
                st.success(f"📗 {t.get('lesson', '')}")
        _domains = summary.get("domains", {}) or {}
        if _domains:
            st.caption("**المجالات:** " + " · ".join(
                f"{d} ({c})" for d, c in _domains.items()
            ))
    with col2:
        render_section_header("أحدث الدروس", f"{min(len(lessons), 10)} دروس")
        for lesson in lessons[:10]:
            q = lesson.get("quality", 0) or 0
            hits = lesson.get("task_hits", 0) or 0
            fails = lesson.get("task_fails", 0) or 0
            bar = "✅" * max(0, min(5, int((q + 1) / 2 * 5))) or "·"
            st.caption(
                f"`[{lesson.get('domain', 'عام')}]` "
                f'<span style="color:var(--text-muted)">{bar}</span> '
                f"({hits} نجاح / {fails} فشل) — {str(lesson.get('lesson', ''))[:110]}"
            )


def render_background_tasks_panel() -> None:
    """لوحة مهام الخلفية: تتبع المهام الثقيلة دون حجز الواجهة."""
    try:
        from ai.background_tasks import get_background_task_manager
        _btm = get_background_task_manager()
        status = _btm.status()
        tasks = _btm.list_tasks(limit=30)
    except Exception:
        st.caption("مهام الخلفية غير متاحة حاليًا (فشل التحميل).")
        return
    total = status.get("total", 0)
    render_section_header(
        "مهام الخلفية",
        f"{total} مهمة — تُنفَّذ دون حجز الواجهة مع إشعارات فورية",
        live=True,
    )
    if total == 0:
        st.caption(
            "لا توجد مهام خلفية بعد. فعّل «⚡ تنفيذ في الخلفية» في الوكيل الموحّد "
            "لتنفيذ المهام الثقيلة دون انتظار."
        )
        return
    _scols = st.columns(4)
    with _scols[0]:
        st.metric("✅ مكتملة", status.get("done", 0))
    with _scols[1]:
        st.metric("⏳ قيد التنفيذ", status.get("running", 0))
    with _scols[2]:
        st.metric("📥 معلقة", status.get("pending", 0))
    with _scols[3]:
        st.metric("❌ فشلت", status.get("failed", 0))
    if not tasks:
        return
    table = []
    for t in tasks:
        table.append({
            "المهمة": t.get("title", ""),
            "المعرف": t.get("task_id", ""),
            "المسار": t.get("route", "—") or "—",
            "الحالة": {
                "pending": "📥 معلقة",
                "running": "⏳ قيد التنفيذ",
                "done": "✅ اكتملت",
                "failed": "❌ فشلت",
                "cancelled": "🚫 أُلغيت",
            }.get(t.get("status"), t.get("status", "—")),
            "بدأت": t.get("started_at", t.get("created_at", "—")),
            "انتهت": t.get("finished_at", "—") or "—",
            "المدة (ms)": round(t.get("duration_ms", 0) or 0, 1),
        })
    st.dataframe(table, use_container_width=True, hide_index=True)


def render_shared_analytics_panel() -> None:
    """لوحة التحليلات التشاركية: تقرير أداء شامل مع درجة عامة وتوصيات تلقائية."""
    try:
        from ai.shared_analytics import get_shared_analytics_reporter
        report = get_shared_analytics_reporter().report()
    except Exception:
        st.caption("التحليلات التشاركية غير متاحة حاليًا (فشل التحميل).")
        return

    score = report.get("score", {})
    render_section_header(
        "التحليلات التشاركية",
        f"الدرجة الإجمالية: {score.get('total', 0)}/100 — {score.get('grade', '—')} · "
        f"أُحدِثت: {report.get('generated_at', '—')}",
        live=True,
    )
    components = score.get("components", {})
    _cols = st.columns(5)
    with _cols[0]:
        st.metric("الوكلاء (40%)", f"{components.get('agents', 0):.0f}/100")
    with _cols[1]:
        st.metric("السرب (25%)", f"{components.get('swarm', 0):.0f}/100")
    with _cols[2]:
        st.metric("التوجيه (20%)", f"{components.get('routing', 0):.0f}/100")
    with _cols[3]:
        st.metric("الخلفية (10%)", f"{components.get('background', 0):.0f}/100")
    with _cols[4]:
        st.metric("الذاكرة (5%)", f"{components.get('memory', 0):.0f}/100")

    # توصيات التحسين
    recs = report.get("recommendations", [])
    if recs:
        st.subheader("🔧 توصيات التحسين")
        render_alert_cards(
            [
                {
                    "severity": r.get("severity", "info"),
                    "title": r.get("title", ""),
                    "detail": r.get("detail", ""),
                    "timestamp": "",
                }
                for r in recs
            ],
            limit=8,
        )

    # أداء الوكلاء
    agents = report.get("agents", {}).get("agents", [])
    if agents:
        st.subheader("أداء الوكلاء")
        agent_table = [
            {
                "الوكيل": a.get("title", a.get("agent_id", "—")),
                "المهام": a.get("tasks", 0),
                "اكتملت": a.get("done", 0),
                "فشلت": a.get("errors", 0),
                "إعادة محاولة": a.get("retries", 0),
                "متوسط المدة (ms)": a.get("avg_ms", 0),
                "أعلى مدة (ms)": a.get("max_ms", 0),
                "نسبة الفشل": f"{a.get('failure_rate', 0) * 100:.0f}%",
            }
            for a in agents
        ]
        st.dataframe(agent_table, use_container_width=True, hide_index=True)

    # ملخصات المصادر
    _dcols = st.columns(4)
    with _dcols[0]:
        sw = report.get("swarm", {})
        st.metric("السرب (إجمالي)", sw.get("total_swarms", 0))
        st.caption(f"نسبة نجاح المهام الفرعية: {sw.get('average_task_success_rate', 0) * 100:.0f}%")
    with _dcols[1]:
        rt = report.get("routing", {})
        st.metric("سجل التوجيه", rt.get("sample", 0))
        st.caption(f"متوسط المدة: {rt.get('avg_latency_ms', 0):.0f}ms · الفشل البديل: {rt.get('failover_rate', 0) * 100:.0f}%")
    with _dcols[2]:
        bg = report.get("background", {})
        st.metric("الخلفية مكتملة", bg.get("done", 0))
        st.caption(f"فشلت: {bg.get('failed', 0)}")
    with _dcols[3]:
        mem = report.get("memory", {})
        st.metric("الدروس الجماعية", mem.get("total_lessons", 0))
        st.caption(f"متوسط الجودة: {mem.get('avg_lesson_quality', 0):.2f}")


def render_adaptive_swarm_panel() -> None:
    """لوحة السرب الذكي المتعلم: ترتيب الوكلاء ديناميكيًا حسب أدائهم التاريخي."""
    from ai.adaptive_swarm import (
        ADAPTIVE_EVENTS as _ADAPTIVE_EVENTS,
        agent_profiles,
        exclude_agents,
        excluded_agents,
        format_recency,
        rank_agents,
        decay_curve_summary,
    )
    from ai.agent_event_bus import get_events
    events = get_events(100)
    profiles = agent_profiles(events)
    render_section_header(
        "السرب المتعلم",
        f"ترتيب ديناميكي حسب الأداء التاريخي — {len(profiles)} وكيلًا في ملف الأداء",
        live=True,
    )
    if not profiles:
        st.caption(
            "لا توجد بيانات أداء كافية بعد. يُبنى ملف الأداء من الأحداث التاريخية "
            "(النجاح 70% + السرعة 20% + الاستقرار 10%)، وكلما نفّذ الوكلاء مهامًا أكثر "
            "أصبح الترتيب والاستبعاد أدق."
        )
    else:
        try:
            all_agents = list(profiles.keys())
            ordered = rank_agents(all_agents, events)
            _scols = st.columns(min(len(ordered), 4))
            for i, key in enumerate(ordered[:4]):
                with _scols[i]:
                    p = profiles.get(key, {})
                    st.metric(f"{i + 1}. {key}", f"{p.get('score', 0):.0f}/100",
                              delta=f"{p.get('tasks', 0)} مهمة")
        except Exception:
            pass
        rows = []
        for key, p in profiles.items():
            rows.append({
                "الوكيل": key,
                "الدرجة": f"{p.get('score', 0):.0f}/100",
                "المهام": int(p.get('tasks', 0) or 0),
                "نسبة النجاح": f"{p.get('success_rate', 0) * 100:.0f}%",
                "متوسط المدة (ms)": round(p.get("avg_duration_ms", 0) or 0, 0),
                "عمر السجل": format_recency(p.get("recency_age_s", 0) or 0),
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)
        try:
            _sel = list(profiles.keys())
            _new_selected = exclude_agents(_sel, events)
            _exc_keys = [k for k in _sel if k not in _new_selected]
        except Exception:
            _new_selected = _sel
            _exc_keys = []
        if _exc_keys:
            exc_rows = []
            for key in _exc_keys:
                prof = profiles.get(key, {})
                tasks = prof.get("tasks", 0)
                rate = prof.get("errors", 0) / (tasks + 0.0) if tasks else 0.0
                exc_rows.append({"الوكيل المستبعد": key,
                                 "نسبة الفشل": f"{rate * 100:.0f}%",
                                 "المهام": int(tasks or 0),
                                 "السبب": "استبعاد مؤقت حتى يحسّن الأداء"})
            st.warning("⛔ وكلاء مستبعدون مؤقتًا من السرب")
            st.dataframe(exc_rows, use_container_width=True, hide_index=True)
        if _new_selected and len(_new_selected) != len(_sel):
            st.caption(
                f"التطبيق الحالي للاستبعاد: يبقى {len(_new_selected)} "
                f"وكيلًا نشطًا من أصل {len(_sel)} بعد تطبيق الاستبعاد المؤقت."
            )
    adaptive = [e for e in get_events(100) if e.get("event_type") in _ADAPTIVE_EVENTS]
    if adaptive:
        st.caption(f"آخر تحديث من السرب المتعلم: {adaptive[-1].get('event_type', '')} — {adaptive[-1].get('timestamp', '')}")
    try:
        curve = decay_curve_summary()
        weights_note = " · ".join(
            f"@{h:.0f}h={w:.2f}" for h, w in curve.get("weights_by_age_hours", {}).items())
        activity_hint = " (توهين نشاطي مفعّل)" if curve.get("activity_decay") else ""
        st.caption(
            f"صيغة التوهين النشطة: {curve.get('mode', 'exponential')}"
            f" — نصف العمر {curve.get('half_life_hours', 48):.0f} ساعة"
            f"{activity_hint} · الأوزان النموذجية: {weights_note}"
        )
    except Exception:
        pass


def render_failure_learning_panel() -> None:
    """لوحة تعلّم الأخطاء الجماعي: دروس الفشل المحفوظة وآخر أحداث التسجيل."""
    from ai.failure_learning import FAILURE_LEARNING_EVENTS as _FL_EVENTS
    from ai.agent_event_bus import get_events

    events = get_events(100)
    render_section_header(
        "تعلّم الأخطاء",
        "دروس تحذيرية من أخطاء الوكلاء — يستفيد منها الجميع",
        live=True,
    )
    # الدروس التحذيرية من الذاكرة الجماعية
    _memory_rows = []
    _memory = None
    try:
        from ai.collective_memory import get_collective_memory
        _memory = get_collective_memory()
        for w in _memory.lessons_list(limit=10):
            if (w.get("quality") or 0.0) < 0.0:
                _memory_rows.append({
                    "المجال": w.get("domain", "عام"),
                    "الدرس": w.get("lesson", "")[:160],
                    "الوكيل المصدر": w.get("source_agent") or "—",
                    "الجودة": f"{w.get('quality', 0):.2f}",
                    "مرات الفشل": int(w.get("task_fails", 0) or 0),
                })
    except Exception:
        pass
    if not _memory_rows:
        st.caption(
            "لا توجد دروس تحذيرية بعد. عند فشل وكيل، يُصنَّف الخطأ نمطيًا "
            "(انتهاء مهلة، حد استخدام، خطأ برمجي...) ويُخزَّن كدرس تحذيري "
            "في الذاكرة الجماعية، فيُحقَن قبل توليف مهام لاحقة في نفس المجال."
        )
    else:
        st.dataframe(_memory_rows, use_container_width=True, hide_index=True)
    fl_events = [e for e in events if e.get("event_type") in _FL_EVENTS]
    if fl_events:
        st.caption(
            f"آخر نشاط من تعلّم الأخطاء: {fl_events[-1].get('event_type', '')}"
            f" — {fl_events[-1].get('timestamp', '')}"
        )


def render_swarm_runner_panel() -> None:
    """واجهة تشغيل السرب مباشرة من لوحة المراقبة: سؤال نصه
    وتشغيل حقيقي لوكيل السرب مع عرض الإجابة وآخر الأحداث حوله."""
    _bot = None
    try:
        from ai.agent_categories import UnifiedAgentChat
        _bot = UnifiedAgentChat
    except Exception:
        _bot = None
    if "swarm_runner_bot" not in st.session_state:
        try:
            st.session_state["swarm_runner_bot"] = UnifiedAgentChat()
        except Exception:
            st.session_state["swarm_runner_bot"] = None
    if "swarm_runner_msgs" not in st.session_state:
        st.session_state["swarm_runner_msgs"] = []

    render_section_header(
        "تشغيل السرب",
        "جرّب السرب مباشرة — الترتيب والاستبعاد والتوليف أمامك",
        live=True,
    )
    st.caption(
        "اكتب سؤالًا وسيرشّد المدير الموحّد الوكلاء بالترتيب الأذكى، يستبعد "
        "الفاشلين المتكررين، ويولّف إجاباتهم في جواب واحد — كل خطوة تُطلق على "
        "الناقل وتظهر في لوحات المراقبة أعلاه."
    )

    if _bot is None:
        st.caption(
            "⚠️ لم تُحمَّل وحدة الوكلاء الموحدة — تشغيل السرب غير متاح في هذه "
            "الجلسة."
        )
        return

    # ── صندوق الإدخال ──────────────────────────────────────────────────
    run_q = st.text_input(
        "سؤالك للسرب",
        placeholder="اكتب سؤالًا وشغّل — ستشاهد السرب يعمل خطوة بخطوة",
        key="swarm_runner_input",
        label_visibility="collapsed",
    )
    col_run, col_hist = st.columns([1, 4])
    with col_run:
        run_clicked = st.button(
            "▶ شغّل السرب", key="swarm_runner_go", use_container_width=True
        )
        if run_clicked:
            st.session_state["_swarm_runner_pending"] = run_q.strip()
    with col_hist:
        if st.button("🧹 مسح السجل", key="swarm_runner_clear",
                     use_container_width=True):
            st.session_state["swarm_runner_msgs"] = []

    pending = st.session_state.pop("_swarm_runner_pending", None)
    if pending:
        bot = st.session_state.get("swarm_runner_bot")
        if bot is None:
            st.caption("الوكيل غير متاح — أعد تحميل الصفحة.")
            st.rerun()
        with st.spinner("السرب يعمل الآن..."):
            try:
                answer, meta = bot.chat(str(pending))
                meta = meta or {}
            except Exception as exc:
                answer = f"فشل التشغيل: {exc}"
                meta = {}
            st.session_state["swarm_runner_msgs"].append({
                "q": str(pending),
                "a": answer,
                "meta": meta,
            })
        st.rerun()

    # ── سجل الجلسات ────────────────────────────────────────────────────
    for i, msg in enumerate(
        st.session_state.get("swarm_runner_msgs", []) or []
    ):
        with st.expander(
            f"❓ {msg.get('q', '')[:80]}",
            expanded=(i == (len(st.session_state.get('swarm_runner_msgs', [])) - 1)),
        ):
            st.markdown(msg.get("a", "") or "(بدون إجابة)")
            delegated = (msg.get("meta") or {}).get("delegated_agents", [])
            if delegated:
                st.caption(
                    "الوكلاء المشاركون: " + " · ".join(str(a) for a in delegated)
                )
            route = (msg.get("meta") or {}).get("route_method", "")
            if route:
                st.caption(f"طريقة التوجيه: {route}")


def render_auto_actions_panel() -> None:
    """لوحة الإجراءات التلقائية للتنبيهات وتشخيص الوكلاء الفاشلين: تفعيل/
    تعطيل الإجراءات، فترة تبريدها، سجل الإجراءات التنفيذي، والوكلاء
    المستبعدون تلقائيًا مؤقتًا."""
    from ai.alert_auto_actions import (
        DEFAULT_ACTIONS,
        auto_excluded_agents,
        auto_unexclude,
        clear_action_log,
        execute_auto_actions,
        get_action_log,
        load_actions_config,
        save_actions_config,
    )

    render_section_header(
        "الإجراءات التلقائية",
        "استجابة آلية للتنبيهات: تشخيص نمطي واستبعاد تلقائي ودروس مسجلة",
        live=True,
    )
    actions = load_actions_config()

    # ── التحكم بالإجراءات ───────────────────────────────────────────────
    with st.expander("⚙️ تخصيص الإجراءات التلقائية", expanded=False):
        st.caption(
            "كل قاعدة تنبيه لها إجراء تلقائي قابل للتفعيل. فترة التبريد تمنع "
            "تكرار نفس الإجراء خلال النافذة الزمنية."
        )
        _aa_cols = st.columns(2)
        toggle_states: Dict[str, bool] = {}
        cd_states: Dict[str, int] = {}
        for i, (name, cfg) in enumerate(list(actions.items())):
            with _aa_cols[i % 2]:
                enabled = st.toggle(
                    cfg.get("description", name),
                    value=bool(cfg.get("enabled")),
                    key=f"autoact_{name}",
                )
                toggle_states[name] = enabled
                cd_states[name] = st.number_input(
                    f"تبريد ({cfg.get('description', name)[:28]}…)",
                    min_value=0, max_value=240, step=5,
                    value=int(cfg.get("cooldown_minutes", 15)),
                    key=f"autoact_cd_{name}",
                )
        if st.button("💾 حفظ إعدادات الإجراءات التلقائية", use_container_width=True):
            custom = {
                name: {
                    "enabled": toggle_states[name],
                    "cooldown_minutes": cd_states[name],
                }
                for name in DEFAULT_ACTIONS
            }
            if save_actions_config(custom):
                st.success("حُفظت إعدادات الإجراءات التلقائية وسينعكس تنفيذها على التنبيهات القادمة.")
            else:
                st.error("تعذر حفظ إعدادات الإجراءات — تبقى القيم الافتراضية سارية.")

    # ── الوكلاء المستبعدون تلقائيًا ─────────────────────────────────────
    excluded_map = auto_excluded_agents()
    if excluded_map:
        st.markdown("**🚫 مستبعدون تلقائيًا مؤقتًا**")
        _ex_rows = [
            {
                "الوكيل": aid,
                "استُبعد عند": info.get("excluded_at", "—"),
                "القاعدة": info.get("rule", "—"),
                "النمط": (info.get("diagnosis") or {}).get("category", "—"),
            }
            for aid, info in excluded_map.items()
        ]
        st.dataframe(_ex_rows, use_container_width=True, hide_index=True)
        _ex_cols = st.columns([1, 4])
        with _ex_cols[0]:
            if st.button("↩️ إعادة إدراج المستبعدين", key="autoact_reinclude"):
                for aid in list(excluded_map.keys()):
                    auto_unexclude(aid)
                st.rerun()
    else:
        st.caption(
            "لا يوجد وكلاء مستبعدون تلقائيًا حاليًا — عند تكرار فشل وكيلٍ "
            "مع تفعيل إجراء الاستبعاد التلقائي سيُضاف هنا مؤقتًا (١٥ دقيقة)."
        )

    # ── سجل الإجراءات ──────────────────────────────────────────────────
    st.markdown("**📋 سجل الإجراءات التلقائية**")
    _aa_log_rows = [
        {
            "#": r.get("action_id", ""),
            "الوقت": r.get("timestamp", "—"),
            "الإجراء": _ACTION_TYPE_LABEL.get(r.get("action_type", ""), ""),
            "القاعدة": r.get("rule", ""),
            "الوكيل": r.get("agent_id", "—"),
            "الحالة": r.get("status", "—"),
            "التشخيص": (r.get("diagnosis") or {}).get("diagnosis", "—")[:60],
        }
        for r in get_action_log(15)
    ]
    if not _aa_log_rows:
        st.caption(
            "السجل فارغ — تُضاف الإجراءات هنا عند إطلاق تنبيه مخصص مع إجراء "
            "مفعّل. جرّب تشغيل مهمة تُفشل وكيلًا لترى التشخيص التلقائي."
        )
    else:
        st.dataframe(_aa_log_rows, use_container_width=True, hide_index=True)
        _log_cols = st.columns([1, 4])
        with _log_cols[0]:
            if st.button("🧹 مسح سجل الإجراءات", key="autoact_log_clear"):
                clear_action_log()
                st.rerun()


_ACTION_TYPE_LABEL = {
    "diagnose": "تشخيص",
    "diagnose_and_lesson": "تشخيص + درس",
    "auto_exclude_and_diagnose": "استبعاد تلقائي + تشخيص",
    "lesson_and_escalate": "درس جماعي + تصعيد",
    "performance_lesson": "درس أداء",
    "exclude_followup": "متابعة استبعاد",
    "throttle_hint": "تلميح تخفيف",
}


def render_custom_alerts_panel() -> None:
    """لوحة التنبيهات القابلة للتخصيص: عتبات يمكن للمستخدم تعديلها وحفظها
    في ملف التكوين، مع سجل تنبيهات مركزي وسجل مباشر من آخر الأحداث."""
    from ai.alert_config import (
        DEFAULT_RULES,
        check_alert_rules,
        clear_alert_log,
        get_alert_log,
        get_rules,
        reset_rules_cache,
        save_custom_rules,
    )
    from ai.agent_event_bus import get_events
    events = get_events(120)
    render_section_header(
        "التنبيهات المخصصة",
        "عتبات تنبيه يحددها المستخدم مع سجل تنبيهات مركزي",
        live=True,
    )
    rules = get_rules()

    # ── تعديل العتبات وحفظها ────────────────────────────────────────────
    with st.expander("⚙️ تخصيص عتبات التنبيه", expanded=False):
        st.caption(
            "عدّل أي قيمة ثم اضغط حفظ — تُخزَّن في config/alert_rules.json "
            "وتُطبق فورًا على التحليل القادم. يُمنع تكرار نفس التنبيه خلال "
            "فترة التبريد الخاصة به."
        )
        _ed_cols = st.columns(2)
        slow_rule = rules.get("slow_response", DEFAULT_RULES["slow_response"])
        with _ed_cols[0]:
            ed_slow = st.number_input(
                "عتبة البطء (مللي ثانية)",
                min_value=1000, max_value=300000, step=1000,
                value=int(float(slow_rule.get("threshold_ms", 12000))),
                key="alert_ed_slow_ms",
            )
            ed_slow_cd = st.number_input(
                "تبريد البطء (دقيقة)",
                min_value=0, max_value=240, step=5,
                value=int(slow_rule.get("cooldown_minutes", 15)),
                key="alert_ed_slow_cd",
            )
        rep_rule = rules.get("repeated_errors", DEFAULT_RULES["repeated_errors"])
        with _ed_cols[1]:
            ed_rep = st.number_input(
                "أخطاء متتالية للتنبيه",
                min_value=1, max_value=20, step=1,
                value=int(rep_rule.get("min_errors", 2)),
                key="alert_ed_rep",
            )
            ed_rep_cd = st.number_input(
                "تبريد الأخطاء المتكررة (دقيقة)",
                min_value=0, max_value=240, step=5,
                value=int(rep_rule.get("cooldown_minutes", 30)),
                key="alert_ed_rep_cd",
            )
        swarm_rule = rules.get(
            "swarm_failure_rate", DEFAULT_RULES["swarm_failure_rate"]
        )
        deg_rule = rules.get("agent_degraded", DEFAULT_RULES["agent_degraded"])
        _ed_cols2 = st.columns(3)
        with _ed_cols2[0]:
            ed_swarm = st.slider(
                "نسبة فشل السرب (التنبيه)",
                min_value=0.1, max_value=1.0, step=0.05,
                value=float(swarm_rule.get("failure_rate_threshold", 0.5)),
                key="alert_ed_swarm",
            )
            ed_swarm_tasks = st.number_input(
                "حد أدنى لمهام السرب",
                min_value=1, max_value=20, step=1,
                value=int(swarm_rule.get("min_tasks", 3)),
                key="alert_ed_swarm_tasks",
            )
        with _ed_cols2[1]:
            ed_deg = st.slider(
                "نسبة فشل الوكيل (تدهور)",
                min_value=0.1, max_value=1.0, step=0.05,
                value=float(deg_rule.get("failure_rate_threshold", 0.75)),
                key="alert_ed_deg",
            )
            ed_deg_tasks = st.number_input(
                "حد أدنى لمهام الوكيل",
                min_value=1, max_value=20, step=1,
                value=int(deg_rule.get("min_tasks", 2)),
                key="alert_ed_deg_tasks",
            )
        cong_rule = rules.get("congestion", DEFAULT_RULES["congestion"])
        with _ed_cols2[2]:
            ed_cong = st.number_input(
                "ازدحام (وكلاء متزامنون)",
                min_value=1, max_value=12, step=1,
                value=int(cong_rule.get("max_concurrent", 3)),
                key="alert_ed_cong",
            )
            ed_cong_cd = st.number_input(
                "تبريد الازدحام (دقيقة)",
                min_value=0, max_value=240, step=5,
                value=int(cong_rule.get("cooldown_minutes", 10)),
                key="alert_ed_cong_cd",
            )
        if st.button("💾 حفظ تخصيص التنبيهات", use_container_width=True):
            custom = {
                "slow_response": {
                    "threshold_ms": ed_slow, "cooldown_minutes": ed_slow_cd
                },
                "repeated_errors": {
                    "min_errors": ed_rep, "cooldown_minutes": ed_rep_cd
                },
                "swarm_failure_rate": {
                    "failure_rate_threshold": ed_swarm,
                    "min_tasks": ed_swarm_tasks,
                },
                "agent_degraded": {
                    "failure_rate_threshold": ed_deg, "min_tasks": ed_deg_tasks
                },
                "congestion": {
                    "max_concurrent": ed_cong, "cooldown_minutes": ed_cong_cd
                },
            }
            if save_custom_rules(custom):
                reset_rules_cache()
                st.success("حُفظ التخصيص وأُعيد تحميل القواعد — ستنعكس على التحليل التالي.")
            else:
                st.error("تعذر حفظ التخصيص — تبقى القيم الافتراضية سارية.")

    # ── التحليل بالقواعد المخصصة ────────────────────────────────────────
    excluded = []
    try:
        from ai.adaptive_swarm import excluded_agents as _excluded_agents
        from ai.agent_event_bus import current_agent_states as _states
        _states_map = _states(events)
        _agent_ids = list(_states_map.keys())
        excluded = _excluded_agents(_agent_ids, events)
    except Exception:
        pass
    custom_alerts = check_alert_rules(events, excluded_agents=excluded)
    if custom_alerts:
        render_alert_cards(custom_alerts, limit=6)
    else:
        st.caption(
            "لا توجد تنبيهات مخصصة حاليًا — جميع الأحداث داخل العتبات "
            "المحددة. عدّل العتبات أعلاه لتناسب سلوك شبكتك."
        )

    # ── سجل التنبيهات المركزي ───────────────────────────────────────────
    st.markdown("**🗂 سجل التنبيهات**")
    _log_rows = [
        {
            "#": a.get("log_id", ""),
            "الوقت": a.get("timestamp", "—"),
            "الشدة": a.get("severity", ""),
            "القاعدة": a.get("rule", ""),
            "العنوان": a.get("title", "")[:70],
            "التفاصيل": a.get("detail", "")[:100],
        }
        for a in get_alert_log(20)
    ]
    if not _log_rows:
        st.caption("السجل فارغ — تُضاف التنبيهات هنا عند تجاوز القواعد المخصصة.")
    else:
        st.dataframe(_log_rows, use_container_width=True, hide_index=True)
        _log_cols = st.columns([1, 4])
        with _log_cols[0]:
            if st.button("🧹 مسح السجل", key="alert_log_clear"):
                clear_alert_log()
                st.rerun()


def render_perf_panel() -> None:
    """لوحة قياس زمن الاستجابة: إحصاءات p50/p90/p95 لكل دالة مقاسة
    مع مرتبة أبطأ الدوال — البيانات من ai.perf_profiler (محلي بالكامل)."""
    from ai.perf_profiler import clear_perf_samples, perf_slowest, perf_stats
    render_section_header(
        "قياس الأداء",
        "زمن تنفيذ الدوال الأثقل (p50/p90/p95) — محلي بالكامل",
        live=True,
    )
    stats = perf_stats()
    col_perf_refresh, col_perf_clear = st.columns([1, 1])
    with col_perf_refresh:
        if st.button("🔄 تحديث القياسات", key="perf_refresh", use_container_width=True):
            st.rerun()
    with col_perf_clear:
        if st.button("🧹 مسح عينات القياس", key="perf_clear", use_container_width=True):
            clear_perf_samples()
            st.rerun()
    if stats["sample_count"] == 0:
        st.info(
            "لا توجد عينات قياس بعد. نفّذ بحثًا في المعرفة أو افتح التبويبات "
            "المعرفية لتظهر قياسات الدوال الأثقل هنا — وتُضاف كل عينة إلى "
            "المراقبة الحية كحدث perf_sample."
        )
        return
    render_kpi_cards([
        {"label": "عينات", "value": stats["sample_count"], "note": "في الجلسة الحالية", "accent": "var(--nsm-indigo)"},
        {"label": "متوسط", "value": f"{stats['avg_ms']:.0f} ms", "note": "لجميع الدوال", "accent": "var(--nsm-cyan)"},
        {"label": "P50", "value": f"{stats['p50_ms']:.0f} ms", "note": "الوسيط", "accent": "#86efac"},
        {"label": "P90", "value": f"{stats['p90_ms']:.0f} ms", "note": "ذيل الأداء", "accent": "var(--nsm-amber)"},
        {"label": "P95", "value": f"{stats['p95_ms']:.0f} ms", "note": "أسوأ 5%", "accent": "#c084fc"},
        {"label": "أقصى", "value": f"{stats['max_ms']:.0f} ms", "note": "أبطأ عينة", "accent": "var(--nsm-danger)"},
    ])
    st.subheader("الأداء لكل دالة")
    _perf_table = []
    for row in stats["by_func"]:
        _perf_table.append({
            "الدالة": row["func"],
            "عينات": row["count"],
            "متوسط ms": row["avg_ms"],
            "P50 ms": row["p50_ms"],
            "P90 ms": row["p90_ms"],
            "P95 ms": row["p95_ms"],
            "أقصى ms": row["max_ms"],
        })
    st.dataframe(_perf_table, use_container_width=True, hide_index=True)
    slow = perf_slowest(3)
    if slow:
        st.subheader("🐢 الأبطأ أداءً (حسب P95)")
        for row in slow:
            st.warning(
                f"**{row['func']}** — P95: {row['p95_ms']:.0f} ms من {row['count']} عينة"
                f" (متوسط {row['avg_ms']:.0f} ms)"
            )


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


# ─────────────────────────────────────────────────────────────────────
# 🆕 لوحة المهام طويلة الأمد (Long-Horizon Tasks)
# ─────────────────────────────────────────────────────────────────────

def render_long_horizon_panel() -> None:
    """لوحة المهام طويلة الأمد: إرسال طلب بحثي معمّق من اللوحة مباشرة،
    تتبع التقدم اللحظي وسجل الخطوات، وإيقاف مهمة قيد التنفيذ.
    الوصول للإنترنت متوفر لكل المهام (بحث + جلب صفحات + ملفات + بايثون محمي)."""
    try:
        from app_core import _LHT_OK, _get_lht_manager
        if not _LHT_OK:
            return
        _lht_mgr = _get_lht_manager()
    except Exception:
        return

    render_section_header(
        "المهام طويلة الأمد",
        "مهام بحثية معمّقة متعددة الخطوات مع وصول الإنترنت — تعمل في الخلفية",
        live=True,
    )

    # ── إرسال مهمة جديدة ────────────────────────────────
    col_goal, col_btn = st.columns([5, 1])
    with col_goal:
        goal = st.text_input(
            "🌐 أرسل مهمة بحثية طويلة",
            placeholder=(
                "مثال: أعدّ تقرير بحث معمّقًا عن تطورات الذكاء الاصطناعي في 2026 "
                "وإسهاماته في تعليم اللغة العربية"
            ),
            key="lht_goal_input",
        )
    with col_btn:
        submit_btn = st.button("📤 تنفيذ", key="lht_submit_btn", use_container_width=True)

    if submit_btn and (goal or "").strip():
        task = _lht_mgr.submit((goal or "").strip())
        if task is None:
            st.warning(
                "طابور المهام ممتلئ (٢ متزامنة / ٦ معلقة كحد أقصى) — "
                "أعد المحاولة بعد اكتمال مهمة أو أوقف واحدة أدناه."
            )
        else:
            st.session_state.setdefault("_lht_last_task", task.task_id)
            st.success(f"🧵 بدأت المهمة «{task.title}» (`{task.task_id}`)")
        st.rerun()

    # ── قائمة المهام ────────────────────────────────────
    try:
        tasks = _lht_mgr.list_tasks(limit=25)
    except Exception:
        st.caption("تعذّر تحميل المهام.")
        return

    if not tasks:
        st.caption(
            "لا توجد مهام طويلة بعد. اكتب موضوعًا أعلاه وادفع «📤 تنفيذ» — "
            "المهمة تُنفَّذ في الخلفية عبر خطوات: خطة ← بحث إنترنت ← جلب "
            "صفحات ← تجميع الأدلة ← تقرير نهائي مع المصادر."
        )
        return

    for t in tasks:
        status = t.get("status", "pending")
        prog = t.get("progress", 0.0) or 0.0
        label = {
            "running": "⚙️ قيد التنفيذ",
            "done": "✅ اكتملت",
            "failed": "❌ فشلت",
            "cancelled": "⛔ أُلغيت",
            "pending": "⏳ في الطابور",
        }.get(status, status)
        with st.expander(
            f"{label} · {t.get('title', '—')} (`{t.get('task_id', '')}`) "
            f"· {int(prog * 100)}% · خطوات: {len(t.get('steps', []))}",
            expanded=(status == "running"),
        ):
            if status == "running":
                st.progress(prog)
            _steps = t.get("steps", []) or []
            _cols = st.columns([4, 1])
            with _cols[0]:
                for s in _steps:
                    icon = {"running": "🔄", "done": "✅", "failed": "❌"}.get(
                        s.get("status", ""), "⬜")
                    st.caption(
                        f"{icon} {s.get('tool', '—')} — "
                        f"{(s.get('result') or '').strip()[:140].replace(chr(10), ' ')}"
                    )
            with _cols[1]:
                if status == "running":
                    if st.button("⛔ أوقف", key=f"lht_stop_{t.get('task_id')}"):
                        _lht_mgr.cancel(t.get("task_id"))
                        st.rerun()
            _meta = t.get("metadata") or {}
            st.caption(
                f"طلبات ويب: {_meta.get('fetch_count', 0)} · ملفّات كتبها الوكيل: "
                f"{_meta.get('files_written', 0)} · مدة: {t.get('elapsed_s', '—')}"
            )
            if status == "done" and t.get("result"):
                st.markdown(t["result"][:3000])


def render_collaborative_panel() -> None:
    """لوحة التعاون في المهام طويلة الأمد: إرسال مهمة مركّبة تُفكّك إلى أدوار
    وكلاء متوازية (باحثون + مدقق نتائج) عبر ناقل الأحداث المشترك، تتبع تقدم
    كل دور لحظيًا وعرض التقرير الموحد الذي يولّفه المنسّق."""
    try:
        from app_core import _COOP_OK, _get_collab_manager
        if not _COOP_OK:
            return
        _coop_mgr = _get_collab_manager()
    except Exception:
        return
    render_section_header(
        "التعاون في المهام الطويلة",
        "مهمة مركّبة ← فريق أدوار متوازية عبر الإنترنت ← مدقق نتائج ← تقرير موحد",
        live=True,
    )
    # ── إرسال مهمة تعاونية جديدة ──────────────────────────
    col_goal, col_btn = st.columns([5, 1])
    with col_goal:
        goal = st.text_input(
            "🤝 أرسل مهمة تعاونية مركّبة",
            placeholder=(
                "مثال: قارن بين الخوارزمي وابن الهيثم من حيث الإسهامات، "
                "وعدّ تقريرًا عن أثرهما على العلوم الحديثة"
            ),
            key="coop_goal_input",
        )
    with col_btn:
        submit_btn = st.button("📤 أنشئ فريقًا", key="coop_submit_btn", use_container_width=True)
    if submit_btn and (goal or "").strip():
        from ai.collaborative_tasks import detect_collaborative_request as _coop_detect
        subgoals = _coop_detect((goal or "").strip())
        if subgoals is None:
            st.warning(
                "الطلب ليس مركّبًا بما يكفي لتشكيل فريق — صُغ المهمة بوجهين "
                "أو أكثر (مثل: «قارن بين X و Y» أو «تقرير عن X ثم Y»)."
            )
        else:
            task = _coop_mgr.submit((goal or "").strip(), subgoals)
            if task is None:
                st.warning(
                    "طابور المهام التعاونية ممتلئ (مهمتان متزامنتان كحد أقصى) — "
                    "أعد المحاولة بعد اكتمال مهمة أو أوقف واحدة أدناه."
                )
            else:
                st.success(
                    f"🤝 بدأت المهمة التعاونية «{task.title}» "
                    f"(`{task.task_id}`) — الأدوار: {', '.join(subgoals)}"
                )
            st.rerun()
    # ── قائمة المهام التعاونية ──────────────────────────────
    try:
        tasks = _coop_mgr.list_tasks(limit=20)
    except Exception:
        st.caption("تعذّر تحميل المهام التعاونية.")
        return
    if not tasks:
        st.caption(
            "لا توجد مهام تعاونية بعد. اكتب مهمة مركّبة أعلاه (وجهان أو أكثر) "
            "وادفع «📤 أنشئ فريقًا» — سيُشكّل فريقًا من الأدوار المتوازية، "
            "ويتبادلون النتائج عبر ناقل الأحداث حتى التقرير الموحّد."
        )
        return
    for t in tasks:
        status = t.get("status", "pending")
        prog = t.get("progress", 0.0) or 0.0
        label = {
            "running": "⚙️ قيد التنفيذ",
            "gathering": "📊 تجميع النتائج",
            "done": "✅ اكتملت",
            "failed": "❌ فشلت",
            "cancelled": "⛔ أُلغيت",
            "pending": "⏳ في الطابور",
        }.get(status, status)
        subgoals = t.get("subgoals") or []
        roles = t.get("roles") or []
        with st.expander(
            f"{label} · {t.get('title', '—')} (`{t.get('task_id', '')}`) "
            f"· {int(prog * 100)}% · أدوار: {len(roles)}",
            expanded=(status == "running"),
        ):
            if status == "running":
                st.progress(prog)
            # مصفوفة الأدوار والتقدم اللحظي
            if roles:
                _rtbl = []
                for r in roles:
                    n = r.get("name", "—")
                    st.caption(f"🎭 **{n}** — {(r.get('goal') or '')[:120]}")
                    _rtbl.append({
                        "الدور": n,
                        "الحالة": {
                            "running": "🔄 يعمل",
                            "done": "✅ أنجز",
                            "failed": "❌ فشل",
                            "pending": "⏳ ينتظر",
                        }.get(r.get("status", ""), r.get("status", "—")),
                        "خطوات": len(r.get("steps") or []),
                        "طلبات ويب": r.get("fetch_count", 0),
                    })
                st.dataframe(_rtbl, use_container_width=True, hide_index=True)
                # سجل الخطوات الأخيرة للأدوار النشطة
                _recent = []
                for r in roles:
                    for s in (r.get("steps") or [])[-6:]:
                        _recent.append({
                            "الدور": r.get("name", "—"),
                            "الأداة": s.get("tool", "—"),
                            "المدخل": (s.get("tool_input") or "")[:80],
                            "النتيجة": (s.get("result") or "")[:120],
                            "الحالة": s.get("status", ""),
                        })
                if _recent:
                    st.caption("آخر خطوات الفريق:")
                    st.dataframe(
                        _recent[::-1][:8], use_container_width=True, hide_index=True
                    )
            _cols = st.columns([4, 1])
            with _cols[1]:
                if status in ("running", "pending"):
                    if st.button("⛔ أوقف", key=f"coop_stop_{t.get('task_id')}"):
                        _coop_mgr.cancel(t.get("task_id"))
                        st.rerun()
            st.caption(
                f"مدة: {t.get('duration_s', '—')} ثانية · أهداف فرعية: "
                f"{len(t.get('subgoals') or [])}"
            )
            if status == "done" and t.get("synthesis"):
                st.markdown((t.get("synthesis") or "")[:3500])


# ══════════════════════════════════════════════════════════════════
# لوحة المعرفة المشتركة (ناقل Qdrant)
# ══════════════════════════════════════════════════════════════════

def render_skb_panel() -> None:
    """لوحة ناقل المعرفة المشترك: إحصاءات وأحدث المعارف المتبادلة."""
    try:
        from app_core import _SKB_OK, _get_skb
        from ai.shared_knowledge import skb_latest
    except Exception:
        return
    st.markdown("---")
    st.subheader("🧠 المعرفة المشتركة (ناقل الفريق)")
    st.caption(
        "يتقاسم أدوار الفريق التعاوني نتائجهم لحظيًا: كل بحث/جلب ناجح يُشارك "
        "في الناقل، ويستحضر كل دور ما وجده الزملاء قبل بحثه — بحث دلالي "
        "عربي (bge-m3) عبر Qdrant، أو ناقل محلي احتياطي عند عدم توفره."
    )
    if not _SKB_OK:
        st.info("وحدة ناقل المعرفة غير متاحة — تعاون الفريق يعمل دون "
                "تبادل دلالي (كل دور يعمل على نتائجه فقط).")
        return
    try:
        _skb = _get_skb()
        _stats = _skb.stats()
        _recent = skb_latest(10)
    except Exception:
        st.info("تعذر الاتصال بناقل المعرفة — العمل مستمر بالفallback المحلي.")
        return
    # بطاقات الإحصاءات
    _q_active = bool(_stats.get("qdrant_active"))
    _cards = st.columns(4)
    _cards[0].metric("🌐 Qdrant",
                     "نشط" if _q_active else "محلي احتياطي")
    _cards[1].metric("📦 نقاط Qdrant",
                     _stats.get("qdrant_points") if _q_active else "—")
    _cards[2].metric("🗄️ العناصر المحلية",
                     _stats.get("local_count", 0))
    _cards[3].metric("🔠 التضمين العربي",
                     "جاهز" if _stats.get("embedder_available") else "—")
    if not _q_active:
        st.caption(
            "نقطة Qdrant المضبوطة حاليًا غير متاحة (404) — النظام يعمل "
            "بناقل محلي SQLite كامل الوظائف، ويستبدل تلقائيًا بـQdrant "
            "فور ضبط نقطة صالحة."
        )
    # أحدث المعارف المتبادلة
    if _recent:
        with st.expander("📨 أحدث المعارف المتبادلة", expanded=False):
            _rows = []
            for r in _recent:
                _rows.append({
                    "المهمة": r.get("task_id", "—"),
                    "الدور": r.get("role", "—"),
                    "الأداة": r.get("tool", "—"),
                    "الوقت": r.get("timestamp", "—")[:19],
                    "المعرفة": (r.get("text") or "")[:160],
                })
            st.dataframe(_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("لا معارف متبادلة بعد — ستظهر هنا نتائج أدوار الفريق "
                   "حال بدء مهمة تعاونية.")


# ══════════════════════════════════════════════════════════════════
# لوحة الخبرة الجماعية (سجل الخبرات المتراكم)
# ══════════════════════════════════════════════════════════════════

def render_tem_panel() -> None:
    """لوحة الخبرة الجماعية المتراكمة: إحصاءات وأحدث الخبرات المستحضرَة."""
    try:
        from app_core import _TEM_OK, _get_experience_log
        from ai.team_experience import tem_latest
    except Exception:
        return
    st.markdown("---")
    st.subheader("📚 الخبرة الجماعية المتراكمة")
    st.caption(
        "ذاكرة ذاتية جماعية مستمرة: كل مهمة تعاونية/طويلة الأمد تُسجِّل "
        "خبراتها (قرار + نتيجته الفعلية) في سجل متراكم — تستحضر الأدوار "
        "الخبرات ذات الصلة قبل التخطيط فتتجه نحو الأنجح وتتجنب الفاشل."
    )
    if not _TEM_OK:
        st.info("وحدة سجل الخبرات غير متاحة — الفريق يعمل دون ذاكرة "
                "تراكمية (سلوك سابق).")
        return
    try:
        _log = _get_experience_log()
        _stats = _log.stats()
        _recent = tem_latest(10)
    except Exception:
        st.info("تعذر فتح سجل الخبرات — العمل مستمر دون تراكم.")
        return
    # بطاقات الإحصاءات
    _cards = st.columns(4)
    _cards[0].metric("📝 إجمالي الخبرات",
                     _stats.get("total", 0))
    _cards[1].metric("✓ نجاح", _stats.get("success", 0))
    _cards[2].metric("~ جزئي", _stats.get("partial", 0))
    _cards[3].metric("✗ فشل", _stats.get("failure", 0))
    if _recent:
        with st.expander("📜 أحدث الخبرات المتراكمة", expanded=False):
            _rows = []
            for r in _recent:
                _mark = ({"success": "✓", "failure": "✗"}.get(
                    r.get("outcome", ""), "~"))
                _rows.append({
                    "الفئة": r.get("category", "—"),
                    "القرار": (r.get("decision") or "")[:180],
                    "النتيجة": f"{_mark} {r.get('outcome', '')}",
                    "الثقة": round(r.get("confidence", 0), 2),
                    "التكرار": r.get("hits", 0),
                    "الوقت": r.get("timestamp", "—")[:19],
                })
            st.dataframe(_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("لا خبرات متراكمة بعد — ستظهر هنا نتائج المهام "
                   "التعاونية وطويلة الأمد حال انتهائها.")


def render_ltg_panel() -> None:
    """لوحة الأهداف المؤسسية طويلة الأمد: الأهداف الجارية وتقدمها وتقييمها الدوري."""
    try:
        from app_core import _LTG_OK, _get_long_term_goals
    except Exception:
        return
    st.markdown("---")
    st.subheader("🎯 الأهداف طويلة الأمد")
    st.caption(
        "أهداف مؤسسية تتراكم عبر الجلسات: يقيّم النظام دوريًا (كل 24 ساعة) "
        "تقدم كل هدف تلقائيًا من سجل الخبرات المتراكمة، ويمكن تعديل التقدم "
        "يدويًا أو أرشفة الأهداف المحققة."
    )
    if not _LTG_OK:
        st.info("وحدة الأهداف طويلة الأمد غير متاحة — الفريق يعمل دون "
                "أهداف مؤسسية (سلوك سابق).")
        return
    try:
        _ltg = _get_long_term_goals()
        _goals = _ltg.list_goals()
        _stats = _ltg.stats()
    except Exception:
        st.info("تعذر فتح سجل الأهداف — العمل مستمر دون أهداف مؤسسية.")
        return
    # بطاقات الإحصاءات
    _cards = st.columns(3)
    _cards[0].metric("🎯 أهداف نشطة",
                     _stats.get("active", 0))
    _cards[1].metric("✓ محققة",
                     _stats.get("achieved", 0))
    _cards[2].metric("📊 متوسط التقدم %",
                     round(100 * (_stats.get("avg_progress", 0) or 0)))
    if _goals:
        with st.expander("📋 قائمة الأهداف", expanded=False):
            _rows = []
            for g in _goals:
                _st_mark = {"achieved": "✓", "archived": "📦"}.get(
                    g.get("status", ""), "🎯")
                _rows.append({
                    "الهدف": (g.get("title") or "—")[:60],
                    "الحالة": f"{_st_mark} {g.get('status', '')}",
                    "التقدم %": round(100 * (g.get("progress", 0) or 0)),
                    "الفئة": g.get("category", "—"),
                    "المحدث": (g.get("updated_at") or "—")[:19],
                })
            st.dataframe(_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("لا أهداف مؤسسية بعد — الفريق لا يعمل على أهداف "
                   "تراكمية. تبدأ هنا عند أول مهمة طويلة الأمد.")


def render_par_panel() -> None:
    """لوحة التفكير ما قبل الفعل: جلسات التفكير المسجلة وحكمها وثقتها."""
    try:
        from app_core import (_PAR_OK, _par_stats, _par_latest,
                              _par_learned_stats)
    except Exception:
        return
    st.markdown("---")
    st.subheader("🧠 التفكير ما قبل الفعل")
    st.caption(
        "قبل كل مهمة طويلة الأمد أو تعاونية يفكر الفريق أولًا: خطوات متوقعة، "
        "مخاطر كل خطوة، بدائل أمان، ودرجة ثقة — ثم يُصدِر حكمًا بـ\"نفذ\" "
        "أو \"راجع\". التحليل نمطي محلي بلا أي API خارجي، وتتراكم الجلسات "
        "في قاعدة محلية.")
    if not _PAR_OK:
        st.info("وحدة التفكير ما قبل الفعل غير متاحة — الفريق يعمل دون "
                "جلسات تفكير مسبقة (سلوك سابق).")
        return
    try:
        _stats = _par_stats()
    except Exception:
        st.info("تعذر فتح سجل التفكير — العمل مستمر دون جلسات تفكير "
                "مسبقة.")
        return
    _cards = st.columns(4)
    _cards[0].metric("🧠 جلسات تفكير",
                     _stats.get("reasoned", 0))
    _cards[1].metric("✓ نُفّذ بلا تعديل",
                     _stats.get("proceeded", 0))
    _cards[2].metric("⟳ عُدّل وخفّف",
                     _stats.get("revised", 0))
    _cards[3].metric("📊 متوسط الثقة",
                     round((_stats.get("avg_confidence", 0) or 0), 2))
    st.caption(
        "تُخزَّن كل جلسة تفكير مع مهمتها — تظهر تفاصيل آخر جلسة في تقرير "
        "المهمة نفسها بلوحة المهام الطويلة والتعاونية (حقل التفكير المسبق).")
    # ── 🆕 دقة التوقعات (الحلقة الاستدراكية) ────────────────────────
    try:
        _acc = _par_learned_stats()
    except Exception:
        _acc = {"learned": 0, "correct": 0, "accuracy": 0.0}
    _learned = _acc.get("learned", 0)
    _correct = _acc.get("correct", 0)
    _acc_cards = st.columns(3)
    _acc_cards[0].metric("🎯 مهام حُسمت نتائجها",
                         _learned)
    _acc_cards[1].metric("✅ توقعات صحيحة",
                         _correct)
    _acc_cards[2].metric("📈 دقة التوقعات",
                         round(float(_acc.get("accuracy", 0) or 0), 3))
    st.caption(
        "بعد اكتمال كل مهمة تربط الحلقة الاستدراكية النتيجة الفعلية "
        "(نجحت/فشلت) بجلسة التفكير السابقة لنفس المهمة، ثم يقيس: هل كان "
        "حكم «نفّذ» صحيحًا؟ هل كانت «راجع» صحيحة؟ — فتتراكم دقة تفكير "
        "الفريق وتُغذّى بها الذاكرة الحسية (سوابق معروفة النتائج).")
    # ── 🆕 معايرة الدقة التاريخية (Historical Calibration) ─────────
    _par_cal_fn = None
    _par_ra_fn = None
    try:
        from app_core import (_par_calibration as _par_cal_fn,  # noqa: F401
                              _par_role_accuracy as _par_ra_fn)
    except Exception:
        pass
    _ra = {}
    if _par_ra_fn is not None:
        try:
            _ra = _par_ra_fn()
        except Exception:
            _ra = {}
    st.markdown("**معايرة الدقة التاريخية**")
    if _ra:
        _rows = []
        for _r in _ra.values():
            _effect = 0.0
            _label = "بلا معايرة"
            if _r.get("learned", 0) >= 3:
                _a = float(_r.get("accuracy", 0) or 0)
                if _a >= 2 / 3:
                    _effect = 0.05
                    _label = "↑ ثقة +0.05 (دور دقيق تاريخًا)"
                elif _a < 0.5:
                    _effect = -0.075
                    _label = "↓ ثقة −0.075 (دور كثير الأخطاء)"
                else:
                    _label = "دقة بين الحدّين — بلا معايرة"
            _rows.append({
                "الدور": _r.get("role", ""),
                "مهام حُسمت": _r.get("learned", 0),
                "توقعات صحيحة": _r.get("correct", 0),
                "الدقة التاريخية": round(
                    float(_r.get("accuracy", 0) or 0), 3),
                "المعايرة المُستحضرة": _label,
            })
        st.dataframe(_rows, use_container_width=True, height=220)
        st.caption(
            "قبل كل جلسة تفكير يستحضر المحرك دقة التوقعات السابقة لنفس "
            "الدور، فيرفع ثقة الدور الدقيق تاريخًا بحد +0.05 ويخفّض ثقة "
            "الدور كثير الأخطاء بحد −0.075 — والمعايرة محصورة بحد صارم "
            "±0.15 ولا تحدث قبل 3 مهام مقاسة للدور.")
    else:
        st.caption(
            "لا توجد بعدُ أدوار بحسم نتائج: مع كل مهمة تكتمل وتُقاس "
            "نتيجتها، يستحضر الفريق دقة كل دور تاريخًا ويُعاير ثقته "
            "بحد آمن صارم ±0.15.")
    # ── الذاكرة الحسية للأدوار ───────────────────────────────────────
    _par_recall_fn = None
    try:
        from app_core import _par_recall as _par_recall_fn  # noqa: F401
    except Exception:
        pass
    st.markdown("**ذاكرة الأدوار الحسية**")
    if _par_recall_fn is None:
        st.caption(
            "لا توجد ذكريات حسية بعد — مع كل جلسة جديدة يبدأ الفريق "
            "بتراكم خبراته: يستحضر سوابق الدور نفسه أو الأهداف "
            "المتشابهة قبل التفكير، فيرتفع مستوى ذكائه الاستباقي "
            "مع الزمن.")
        return
    try:
        _mems = _par_recall_fn(top_k=8)
    except Exception:
        _mems = []
    if not _mems:
        st.caption(
            "لا سوابق تفكير بعد — كلما تراكمَت الجلسات، استحضرت وحدة "
            "التفكير ذكريات مماثلة لترفع الثقة عند السوابق الناجحة "
            "وتخففها عند تكرار المراجعات.")
        return
    _proc = sum(1 for _m in _mems if _m.get("verdict") == "proceed")
    st.caption(
        f"آخر {len(_mems)} جلسات مماثلة: {_proc} أمرت بـ«نفّذ» و"
        f"{len(_mems) - _proc} أمرت بـ«راجع» — يستحضرها الفريق قبل كل "
        "جلسة جديدة فيرتفع ذكاؤه الاستباقي مع الزمن.")
    with st.expander("أحدث الذكريات المستحضرة"):
        for _m in _mems[:8]:
            _v = "نفّذ" if _m.get("verdict") == "proceed" else "راجع"
            st.text(
                f"[{_v}] {_m.get('role') or '—'}: "
                f"{_m.get('goal', '')[:110]} — "
                f"ثقة {round(float(_m.get('confidence') or 0), 2)} "
                f"(تشابه {round(float(_m.get('similarity') or 0), 2)})")
    # ── التوقع المتعدد المسارات (Multi-Path Forecasting) ──────────
    _par_multi_fn = None
    try:
        from app_core import _reason_multi_task as _par_multi_fn  # noqa: F401
    except Exception:
        pass
    st.markdown("**التوقع المتعدد المسارات**")
    if _par_multi_fn is None:
        st.caption(
            "غير متاح في هذه النسخة — يمكن للفريق مقارنة عدة خطط بديلة "
            "للهدف نفسه واختيار الأعلى ثقةً تاريخًا بحد ±0.15 صارم.")
        return
    # محاكاة توضيحية فورية على أحدث هدف معروف في السجل دون كسر الحالة
    try:
        _latest = None
        try:
            from app_core import _par_latest as _par_latest_fn  # noqa: F401
            _latest = _par_latest_fn("")  # لا نجبر جلب هدف معين
        except Exception:
            _latest = None
        if _latest:
            _goal = _latest.get("goal", "")[:140]
            _paths = (
                [["ابحث في الويب عن آخر المستجدات","قارن بين النتائج"]]
                if _goal else [])
            if _paths:
                _multi = _par_multi_fn(
                    "_ui_multi",
                    _goal or "استكشاف الهدف",
                    _paths,
                    role=_latest.get("role") or "")
                if _multi and _multi.get("n_paths", 0) >= 2:
                    _cmp = []
                    for _i, _p in enumerate(_multi.get("paths", [])):
                        _cmp.append({
                            "المسار": f"{('المختار ' if _i == _multi.get('chosen_index') else '')}مسار {_i + 1}",
                            "الثقة بعد المعايرة": round(_p.get("confidence", 0), 3),
                            "الأثر التاريخي": round(
                                _multi.get("history_scores", [_multi.get("history_scores")[0]]*2)[_i], 3),
                            "الحكم": ("نفّذ" if _p.get("verdict") == "proceed" else "راجع"),
                        })
                    st.dataframe(_cmp, use_container_width=True,
                                 height=190)
                    st.caption(
                        "عند تعدد الخطط البديلة للهدف نفسه، يجري المحرك "
                        "جلسة تفكير مستقلة لكل مسار ويصنّف خطواته مقابل "
                        "السوابق المقاسة تاريخًا للدور: خطوات متشابهة مع "
                        "سوابق صحيحة ترفع المسار (+0.05) ومع سوابق خاطئة "
                        "تخفضه (−0.08) — بحد صارم ±0.15 — ثم يختار "
                        "المسار الأعلى ثقةً تاريخًا للتنفيذ.")
                    return
    except Exception:
        pass
    st.caption(
        "لا مقارنة مسارات نشطة بعد — مع كل مهمة ذات خطط بديلة يختار "
        "الفريق المسار الأعلى ثقةً تاريخًا للدور (±0.15 كحد صارم).")
    # ── حُكم الفريق على المسارات (Multi-Role Path Voting) ────────────
    _par_mrr_fn = None
    try:
        from app_core import (_reason_multi_role_task  # noqa: F401
                               as _par_mrr_fn)
    except Exception:
        pass
    st.markdown("**حُكم الفريق على المسارات**")
    if _par_mrr_fn is None:
        st.caption(
            "غير متاح في هذه النسخة — ميزة يوزع المحرك فيها الخطط "
            "البديلة على أدوار الفريق، فيقيم كل دور كل مسار من زاوية "
            "تخصصه، ثم يجتمع الفريق على المسار الأعلى توافقًا وثقةً "
            "(±0.15 كحد صارم).")
        return
    try:
        _mrr = None
        if _latest:
            _goal2 = _latest.get("goal", "")[0:140]
            if _goal2:
                _mrr = _par_mrr_fn(
                    "_ui_mrr",
                    _goal2,
                    [["ابحث في الويب عن آخر المستجدات",
                      "قارن بين النتائج"],
                     ["اجمع المصادر الموثوقة",
                      "حلل البيانات داخليا"]],
                    roles=["analyst", "writer"])
        if _mrr and _mrr.get("n_paths", 0) >= 2:
            _mcmp = []
            for _i in range(_mrr.get("n_paths", 0)):
                _chosen_tag = ("المختار "
                               if _i == _mrr.get("chosen_index") else "")
                _mcmp.append({
                    "المسار": f"{_chosen_tag}مسار {_i + 1}",
                    "ثقة الفريق": round(
                        _mrr.get("vote_scores", [0.0] * 2)[_i], 3),
                    "التوافق بين الأدوار": round(
                        _mrr.get("consensus_scores", [0.0] * 2)[_i], 3),
                })
            st.dataframe(_mcmp, use_container_width=True, height=140)
            if _mrr.get("role_judgments"):
                with st.expander("أحكام الأدوار التفصيلية"):
                    for _r, _jud in _mrr["role_judgments"].items():
                        _acc = _mrr.get("role_accuracies", {})
                        _a = _acc.get(_r, {})
                        st.text(
                            f"◂ {(_r or '—').capitalize()} "
                            f"(دقة تاريخية {round(float(
                                _a.get('accuracy', 0) or 0), 2)} "
                            f"من {_a.get('learned', 0)} مهمة) — "
                            f"أحكام المسارات: "
                            + "; ".join(
                                f"مسار {int(_k) + 1}: "
                                f"ثقة {round(float(
                                    _v.get('confidence', 0)), 2)} "
                                f"{'نفّذ' if _v.get('verdict') == 'proceed' else 'راجع'}"
                                for _k, _v in sorted(_jud.items())))
            st.caption(
                _mrr.get("chosen_role_note", "")
                + " — يوزع المحرك الخطط البديلة على أدوار متخصصة: كل "
                "دور يقيم كل مسار من زاوية تخصصه بثقته وذاكرته "
                "الحسية ومعايرته التاريخية، ثم يجمع الأحكام بوزن "
                "الثقة التاريخية للدور (±0.03 مكافأة توافق و±0.15 "
                "حد صارم) ويختار المسار الأعلى توافقًا وثقةً.")
            # ── التوقع الجماعي المتطور (Collective Resolution) ────
            _rcd = None
            try:
                from app_core import _resolve_collective_task
                _rcd = _resolve_collective_task(
                    "_ui_collective", _goal2,
                    [["ابحث في الويب عن آخر المستجدات",
                      "قارن بين النتائج"],
                     ["اجمع المصادر الموثوقة",
                      "حلل البيانات داخليا"]],
                    roles=["analyst", "writer"])
            except Exception:
                _rcd = None
            if _rcd and _rcd.get("n_merged", 0) > 0:
                st.markdown("**التوقع الجماعي المتطور**")
                _cmt = [
                    {"الخطوة": s.get("n"),
                     "الإجراء": s.get("action", ""),
                     "المسار المصدر": s.get("source_path", 0) + 1,
                     "الدور المصدر": s.get("source_role", ""),
                     "ثقة المصدر": round(
                         float(s.get("source_confidence", 0) or 0), 2),
                     "الوزن": round(
                         float(s.get("step_weight", 0) or 0), 2)}
                    for s in _rcd.get("merged_steps", [])]
                st.dataframe(_cmt,
                             use_container_width=True,
                             height=180)
                st.caption(
                    f"ثقة الخطة المدمجة {round(
                        float(_rcd.get('confidence', 0) or 0), 2)} "
                    f"— تعارض افتراضي بين الأحكام "
                    f"{round(float(_rcd.get('conflict_sim', 0) or 0) * 100)}% "
                    f"— {_rcd.get('resolution_note', '')}")
                with st.expander("تفاصيل الدمج"):
                    st.text(
                        "بعد جمع أحكام أدوار الفريق على المسارات "
                        "البديلة، يحاكي المحرك سيناريو تعارضٍ "
                        "افتراضيًا بين الأحكام (رياضيًا بلا أي نموذج "
                        "لغوي)، ثم يولّد خطة واحدة موسّعة تدمج أفضل "
                        "خطوات كل مسار: المسار الأعلى تصويتًا أولًا "
                        "ثم الخطوات غير المكررة من المسارات الأخرى "
                        "— بحد صارم +0.15 على مكافأة الدمج.")
                # ── ذاكرة التعارضات (Conflict Memory) ────
                _cs = None
                try:
                    from app_core import _conflict_stats_task
                    _cs = _conflict_stats_task()
                except Exception:
                    _cs = None
                if _cs and _cs.get("resolutions", 0) > 0:
                    _mem = _rcd.get("recalled_conflicts", []) or []
                    st.markdown("**ذاكرة التعارضات**")
                    _mem_table = [
                        {"المهمة": (_m.get("task_id", "") or "")[:20],
                         "الهدف": (_m.get("goal", "") or "")[:30],
                         "الأدوار": _m.get("roles_csv", ""),
                         "ن مسارات": _m.get("n_paths", 0),
                         "تعارض": round(
                             float(_m.get("conflict_sim", 0) or 0), 2),
                         "النتيجة": _m.get("outcome")
                         or "بانتظار القياس",
                         "صحيح": "نعم" if _m.get("was_correct") else "—"}
                        for _m in _mem[:5]]
                    if _mem_table:
                        st.dataframe(_mem_table,
                                     use_container_width=True,
                                     height=150)
                    st.caption(
                        f"تعارضات محلولة {int(_cs.get('resolutions', 0))} "
                        f"— مقاسة {int(_cs.get('measured', 0))} "
                        f"— صحيحة {int(_cs.get('correct', 0))} "
                        f"(دقة "
                        f"{round(float(_cs.get('accuracy', 0) or 0), 2)}) "
                        f"— أثر الاستحضار "
                        + str(round(float(_rcd.get('conflict_recall_effect', 0)
                                        or 0), 3)))
                    with st.expander(
                            "كيف يتعلم الفريق من تعارضاته"):
                        st.text(
                            "يسجّل المحرك كل حلّ تعارض جماعي مع "
                            "محاكاته وخطوات الدمج، ثم تربط النتيجة "
                            "الفعلية للمهمة بأحدث حلّ (proceed صحيح "
                            "فقط عند نجاح وrevise عند فشل). "
                            "التعارضات المماثلة في الهدف وانحراف "
                            "التعارض (±0.05) تُستحضر قبل أي حلّ "
                            "جديد: السوابق الصحيحة ترفع ثقة الدمج "
                            "والخاطئة تخفضها — بحد صارم ±0.15.")
                # ── منهجية NSM (سرّ عمل الوالد الموروث) ────
                _ms = None
                try:
                    from app_core import _method_stats
                    _ms = _method_stats()
                except Exception:
                    _ms = None
                if _ms and int(_ms.get("tasks", 0) or 0) > 0:
                    st.markdown("**🎓 منهجية NSM (المنهجية الموروثة)**")
                    _m_table = [
                        {"المهمة": str(_ms.get("tasks", 0)),
                         "ناجحة": str(_ms.get("tasks_ok", 0)),
                         "دقة المنهجية":
                         f"{round(float(_ms.get('accuracy', 0) or 0), 2)}",
                         "إجمالي الخطوات":
                         str(_ms.get("total_steps", 0)),
                         "نسبة فحص+تحقق": f"{round(float(_ms.get('inspect_verify_ratio', 0) or 0) * 100)}%",
                         "دروس مسجلة": str(_ms.get("lessons", 0)),
                         "دروس مطبقة":
                         str(_ms.get("lessons_applied", 0))}]
                    st.dataframe(_m_table,
                                 use_container_width=True, height=90)
                    st.caption(
                        "دورة المنهجية: خطة → فحص فعلي → تنفيذ منضبط "
                        "→ تحقق → تعلّم من الأخطاء. كلما ارتفعت نسبة "
                        "خطوات الفحص والتحقق، كلما عمل الوكيل بمنهجية "
                        "أقرب إلى الوالد.")
                    with st.expander("المبادئ السبعة الموروثة"):
                        st.text(
                            "1. التخطيط قبل التنفيذ — 2. فحص فعلي لا "
                            "تخمين — 3. تنفيذ منضبط خطوة بخطوة — "
                            "4. تحقق بعد التنفيذ (الادعاء بلا تحقق "
                            "ممنوع) — 5. تعلّم من الأخطاء (تشخيص "
                            "السبب الجذري أولاً) — 6. انضباط الإخراج "
                            "(بصراحة: ما تحقق وما لم يتحقق) — "
                            "7. الأمان أولاً (لا يُكسر شيء يعمل).")
    except Exception:
        pass
    st.caption(
        "لا حُكم فريق نشطًا بعد — مع كل مهمة ذات خطط بديلة يوزع "
        "المحرك الأحكام على أدوار الفريق ويجتمع على المسار الأعلى "
        "توافقًا وثقةً (±0.15 كحد صارم).")



