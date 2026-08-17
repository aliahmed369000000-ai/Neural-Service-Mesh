# -*- coding: utf-8 -*-
"""🧭 لوحة السرب الموحدة — Adaptive Swarm Dashboard.

تجمع حالة السرب كلها في مكان واحد:
الوكلاء الأحياء/المنتهون/الفاشلون · مهام السرب المحفوظة · المهام طويلة الأمد ·
التنبيهات القابلة للتخصيص والإجراءات التلقائية.
"""
from __future__ import annotations

import streamlit as st  # noqa: F401  (نفس نمط باقي صفحات ui_pages)
from html import escape as _escape

from app_core import *  # noqa: F401,F403
from typing import Any, Dict

from ui_components import render_agent_cards, render_alert_cards, render_kpi_cards, render_section_header


def _esc(value: Any) -> str:
    return _escape(str(value or ""))


def render_unified_swarm_dashboard() -> None:
    """لوحة السرب الموحدة: حالة كل الوكلاء والسرب في نظرة واحدة."""
    from ai.unified_swarm_dashboard import (
        agents_overview,
        apply_auto_actions,
        evaluate_alerts,
        list_alert_rules,
        list_auto_actions,
        long_horizon_status,
        swarm_status,
        toggle_auto_action,
        unified_dashboard_snapshot,
        update_alert_rule,
    )

    render_section_header(
        "🧭 لوحة السرب الموحدة",
        "الوكلاء · السرب · المهام طويلة الأمد · التنبيهات",
        live=True,
    )
    st.caption("نظرة موحدة على صحة السرب: من أي وكيل نشط إلى آخر تنبيه جرى تفعيله — مع تنبيهات قابلة للتخصيص وإجراءات تلقائية.")

    col_refresh, col_apply = st.columns([2, 1])
    with col_refresh:
        if st.button("🔄 تحديث اللوحة", key="swarm_dash_refresh", use_container_width=True):
            st.rerun()
    with col_apply:
        if st.button("⚡ طبّق الإجراءات التلقائية الآن", key="swarm_dash_apply"):
            alerts = evaluate_alerts()
            applied = apply_auto_actions(alerts)
            if applied:
                st.success(f"طُبّقت {len(applied)} إجراءات تلقائية — راجع «مراقبة حيّة» لتفاصيل الأحداث")
                st.rerun()
            else:
                st.info("لا توجد تنبيهات مطابقة لإجراءات مفعّلة حالياً.")

    snapshot = unified_dashboard_snapshot()
    agents = snapshot.get("agents", {})
    swarm = snapshot.get("swarm", {})
    lh = snapshot.get("long_horizon", {})
    perf = snapshot.get("performance", {}) if isinstance(snapshot.get("performance"), dict) else {}
    alerts = snapshot.get("alerts", []) or []

    counts = agents.get("counts", {})
    render_kpi_cards([
        {"label": "وكلاء نشطون", "value": counts.get("alive", 0), "note": "قيد التنفيذ الآن", "accent": "var(--nsm-cyan)"},
        {"label": "دورات مكتملة", "value": counts.get("done", 0), "note": "ناجحون في السجل", "accent": "#86efac"},
        {"label": "وكلاء فاشلون", "value": counts.get("failed", 0), "note": "تحتاج مراجعة", "accent": "var(--nsm-danger)"},
        {"label": "وكلاء بطيئون", "value": counts.get("slow", 0), "note": "فوق عتبة البطء", "accent": "var(--nsm-amber)"},
        {"label": "مهام السرب", "value": swarm.get("total", 0), "note": "محفوظة في السجل", "accent": "var(--nsm-indigo)"},
        {"label": "مهام طويلة الأمد", "value": sum(lh.get("counts", {}).values()) if isinstance(lh, dict) else 0, "note": "قيد الإدارة", "accent": "#c084fc"},
    ])

    # ── مؤشرات الأداء: الذاكرة ووقت الاستجابة ──────────────────
    sys_perf = perf.get("system", {}) if isinstance(perf.get("system"), dict) else {}
    rt = perf.get("response_times", {}) if isinstance(perf.get("response_times"), dict) else {}
    mem_used = sys_perf.get("memory_used_mb")
    mem_total = sys_perf.get("memory_total_mb")
    mem_note = (
        f"من إجمالي {mem_total:.0f} MB" if mem_used is not None and mem_total
        else "على هذه الآلة"
    )
    render_kpi_cards([
        {"label": "استخدام الذاكرة", "value": f"{mem_used:.0f} MB" if mem_used is not None else "—",
         "note": mem_note, "accent": "#38bdf8"},
        {"label": "نسبة الذاكرة", "value": f"{sys_perf.get('memory_percent')}%" if sys_perf.get("memory_percent") is not None else "—",
         "note": "ذاكرة النظام الكلية", "accent": "#38bdf8"},
        {"label": "ذروة RSS", "value": f"{sys_perf.get('peak_rss_mb')} MB" if sys_perf.get("peak_rss_mb") is not None else "—",
         "note": "أقصى ما استُخدم منذ البدء", "accent": "#38bdf8"},
        {"label": "حِمْل النظام (1د)", "value": sys_perf.get("load_1m", "—"),
         "note": "متوسط الحمل على المعالج", "accent": "#38bdf8"},
        {"label": "متوسط الاستجابة", "value": f"{rt.get('avg_ms')} ms" if rt.get("avg_ms") is not None else "—",
         "note": "آخر 80 حدثًا في السجل", "accent": "#facc15"},
        {"label": "أبطأ استجابة", "value": f"{rt.get('max_ms')} ms" if rt.get("max_ms") is not None else "—",
         "note": f"العتبة {rt.get('slow_ms_threshold', 12000):.0f} ms", "accent": "#facc15"},
    ])
    rt_count = rt.get("count", 0) if isinstance(rt, dict) else 0
    rt_slow = rt.get("slow_count", 0) if isinstance(rt, dict) else 0
    if rt_count and rt_slow:
        st.warning(
            f"⚠️ **{rt_slow} من {rt_count} استدعاءً** تجاوزت عتبة البطء "
            f"({rt.get('slow_ms_threshold', 12000):.0f} ms) — راجع «مراقبة حيّة».",
            icon="⚠️",
        )
    # ── وقت استجابة الوكلاء الفردي ────────────────────────────
    render_section_header("وقت الاستجابة للوكلاء", "آخر زمن تنفيذ مسجل لكل وكيل (ms)")
    agent_rows = agents.get("agents", {})
    if agent_rows:
        _resp = [
            f"**{_esc(aid)}**: {row.get('last_response_ms')} ms"
            if row.get("last_response_ms") is not None
            else f"**{_esc(aid)}**: لم يكمل بعد"
            for aid, row in agent_rows.items()
        ]
        st.markdown(" · ".join(_resp), unsafe_allow_html=True)
    else:
        st.info("لا توجد وكلاء نشطة بعد — نفّذ مهمة من أي تبويب وكيل لتظهر أوقات استجابتها هنا.")

    # ── التنبيهات ───────────────────────────────────────────────
    render_section_header("التنبيهات", "تُقيَّم وفق القواعد المخصّصة ثم تُطبَّق إجراءاتها التلقائية")
    if alerts:
        render_alert_cards(alerts)
    else:
        st.success("لا توجد تنبيهات نشطة — السرب ضمن الحدود المسموحة.")

    # ── الوكلاء ────────────────────────────────────────────────
    render_section_header("الوكلاء", "آخر حالة معروفة لكل وكيل في السجل الحالي")
    agent_rows = agents.get("agents", {})
    if agent_rows:
        render_agent_cards(agent_rows)
    else:
        st.info("لا توجد حالات وكلاء بعد — نفّذ مهمة من أي تبويب وكيل لتظهر هنا.")

    # ── السرب ──────────────────────────────────────────────────
    render_section_header("السرب", "آخر عمليات التنفيذ المحفوظة دائمًا")
    swarm_history = swarm.get("history") or []
    if swarm_history:
        for item in swarm_history[:10]:
            if not isinstance(item, dict):
                continue
            success = bool(item.get("success"))
            st.markdown(
                f"{'✅' if success else '❌'} **{_esc(item.get('goal') or item.get('title') or 'مهمة سرب')}**"
                f"{' — ' + _esc(item.get('merged') or item.get('result') or '') if item.get('merged') or item.get('result') else ''}",
                unsafe_allow_html=True,
            )
        st.caption(f"المجموع: {swarm.get('total', 0)} عملية · ناجحة: {swarm.get('successful', 0)} · فاشلة: {swarm.get('failed', 0)}")
    else:
        st.info("لا توجد عمليات سرب محفوظة بعد — نفّذ هدفاً من تبويب «السرب الذكي» لتظهر هنا.")

    # ── المهام طويلة الأمد ─────────────────────────────────────
    render_section_header("المهام طويلة الأمد", "مهام متعددة الخطوات قيد الإدارة")
    lh_tasks = lh.get("tasks") or []
    if lh_tasks:
        for task in lh_tasks[:10]:
            if not isinstance(task, dict):
                continue
            status = str(task.get("status") or "—")
            st.markdown(
                f"**{_esc(task.get('title') or task.get('goal') or task.get('id'))}** — {status}",
                unsafe_allow_html=True,
            )
        st.caption((" · ".join(f"{k}: {v}" for k, v in (lh.get("counts", {}) or {}).items())))
    else:
        st.info("لا توجد مهام طويلة الأمد مسجّلة بعد.")

    # ── قواعد التنبيهات القابلة للتخصيص ────────────────────────
    with st.expander("⚙️ تخصيص قواعد التنبيهات", expanded=False):
        _unit_map = {"slow_threshold_ms": "مللي ثانية", "stale_threshold_s": "ثانية",
                     "error_ratio": "نسبة (مثلاً 0.2 = 20%)"}
        _step_map = {"slow_threshold_ms": 1000.0, "stale_threshold_s": 5.0,
                     "error_ratio": 0.05}
        rules = list_alert_rules()
        for rule in rules:
            col_en, col_val = st.columns([1, 2])
            with col_en:
                enabled = st.toggle(f"⚡ {rule.get('label')}", value=bool(rule.get("enabled")),
                                    key=f"rule_en_{rule.get('id')}")
                if enabled != bool(rule.get("enabled")):
                    update_alert_rule(rule["id"], enabled=enabled)
                    st.rerun()
            with col_val:
                kind = str(rule.get("kind"))
                unit = _unit_map.get(kind, "نسبة")
                value = st.number_input(
                    f"العتبة ({unit})",
                    min_value=0.0001 if kind == "error_ratio" else 0.0,
                    value=float(rule.get("value") or 1.0),
                    step=_step_map.get(kind, 0.05),
                    key=f"rule_val_{rule.get('id')}",
                )
                if abs(value - float(rule.get("value") or 1.0)) > 1e-9:
                    update_alert_rule(rule["id"], value=value)
                    st.caption(f"حُدّثت قيمة «{rule.get('label')}»")
            st.caption(rule.get("description") or "")

    # ── الإجراءات التلقائية ─────────────────────────────────────
    with st.expander("🔧 الإجراءات التلقائية", expanded=False):
        for action in list_auto_actions():
            enabled = st.toggle(f"⚡ {action.get('label')}", value=bool(action.get("enabled")),
                                key=f"auto_en_{action.get('id')}")
            if enabled != bool(action.get("enabled")):
                toggle_auto_action(action["id"], enabled)
                st.caption(f"{'فُعّل' if enabled else 'أُوقف'} {action.get('label')}")
            st.caption(action.get("description") or "")
