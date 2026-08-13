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
    render_collective_memory_panel()
    render_background_tasks_panel()
    render_shared_analytics_panel()
    render_adaptive_swarm_panel()
    render_failure_learning_panel()


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
