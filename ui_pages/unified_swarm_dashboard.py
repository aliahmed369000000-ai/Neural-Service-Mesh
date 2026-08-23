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
try:
    import plotly.graph_objects as go
except ImportError:  # Plotly غير متوفر — تتدحرج إلى بطاقات نصية
    go = None  # type: ignore[assignment]



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

    # ── رسوم بيانية تفاعلية: تطور الأداء بمرور الوقت ──────────
    render_section_header("تطور الأداء بمرور الوقت", "قياسات متسلسلة (تُحدَّث كل خمس ثوانٍ) — استخدم التكبير والتحريك للتحكم")
    from ai.unified_swarm_dashboard import (filter_timeline,
                                            performance_timeline)
    _range_opts = [
        ("مخصص", "custom"), ("آخر 5 دقائق", "5m"), ("آخر 15 دقيقة", "15m"),
        ("آخر 30 دقيقة", "30m"), ("آخر ساعة", "1h"),
        ("اليوم", "day"), ("الأسبوع", "week"),
    ]
    _range_choice = st.radio(
        "النطاق الزمني",
        options=[label for label, _ in _range_opts],
        horizontal=True, label_visibility="collapsed",
        index=0, key="nsm_perf_range_radio")
    _range_name = ""
    _range_from = _range_to = None
    for label, key in _range_opts:
        if label == _range_choice:
            _range_name = key
            break
    if _range_name == "custom":
        col_from, col_to = st.columns(2)
        with col_from:
            _range_from = st.text_input("من (ISO 8601: 2026-08-17T00:00:00)",
                                        key="nsm_perf_range_from") or None
        with col_to:
            _range_to = st.text_input("إلى (ISO 8601: 2026-08-17T23:59:59)",
                                      key="nsm_perf_range_to") or None
    timeline = performance_timeline(limit=60)
    if timeline:
        timeline = filter_timeline(timeline, range_name=_range_name or None,
                                   from_iso=_range_from, to_iso=_range_to)
    if timeline:
        def _epoch_ts(_row: Any) -> str:
            try:
                _ep = float(_row.get("epoch_float") or 0.0)
                if _ep > 0:
                    from datetime import datetime as _dt, timezone as _tz
                    return _dt.utcfromtimestamp(_ep).strftime("%H:%M")
            except (TypeError, ValueError, OSError):
                pass
            return str(_row.get("ts") or "")
        _ts = [_epoch_ts(row) for row in timeline]
        _idx = _ts
        _mem = [row.get("memory_mb") for row in timeline]
        _peak = [row.get("peak_rss_mb") for row in timeline]
        _avg = [row.get("avg_ms") for row in timeline]
        _last = [row.get("last_ms") for row in timeline]
        _slow = [row.get("slow_count") for row in timeline]
        _fig_mem = go.Figure() if go is not None else None
        _fig_resp = go.Figure() if go is not None else None
        if _fig_mem is not None:
            _fig_mem.add_trace(go.Scatter(x=_idx, y=_mem, mode="lines+markers",
                                          name="الذاكرة الحالية (MB)",
                                          line=dict(color="#38bdf8", width=2)))
            _fig_mem.add_trace(go.Scatter(x=_idx, y=_peak, mode="lines+markers",
                                          name="ذروة RSS (MB)",
                                          line=dict(color="#818cf8", width=2)))
            _fig_mem.update_layout(
                title=dict(text="استهلاك الذاكرة عبر القياسات", x=0.5,
                           xanchor="center", font=dict(size=14)),
                xaxis_title="وقت القياس (UTC) — الأحدث أخيرًا",
                yaxis_title="ميغابايت",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=40))
            st.plotly_chart(_fig_mem, use_container_width=True,
                            key="nsm_perf_mem_chart")
        if _fig_resp is not None:
            _fig_resp.add_trace(go.Scatter(x=_idx, y=_avg, mode="lines+markers",
                                           name="متوسط الاستجابة (ms)",
                                           line=dict(color="#facc15", width=2)))
            _fig_resp.add_trace(go.Scatter(x=_idx, y=_last, mode="lines+markers",
                                           name="آخر استجابة (ms)",
                                           line=dict(color="#f472b6", width=2)))
            _fig_resp.add_trace(go.Scatter(x=_idx, y=_slow, mode="lines+markers",
                                           name="عدد الاستدعاءات البطيئة",
                                           line=dict(color="#f87171", width=2)))
            _fig_resp.update_layout(
                title=dict(text="زمن استجابة الوكلاء عبر القياسات", x=0.5,
                           xanchor="center", font=dict(size=14)),
                xaxis_title="وقت القياس (UTC) — الأحدث أخيرًا",
                yaxis_title="ملّي ثانية / عدد",
                template="plotly_dark",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=40))
            st.plotly_chart(_fig_resp, use_container_width=True,
                            key="nsm_perf_resp_chart")
        st.caption(
            f"آخر قياس: {_ts[-1] if _ts else '—'} — "
            f"ذاكرة {(_mem[-1] if _mem else '—')} MB · "
            f"متوسط استجابة {(_avg[-1] if _avg else '—')} ms · "
            f"بطيئة {(_slow[-1] if _slow else '—')}")
    else:
        st.info("لا توجد قياسات كافية بعد — ستظهر الرسوم فور تسجيل قياسين فأكثر.")
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

    # ── لوحة تحكم السرب الحية (Decentralized Living Mesh) ──────────
    st.divider()
    render_section_header("🌐 لوحة تحكم السرب الحية", "مراقبة الشبكة اللامركزية والسيادة الحية")
    
    from ai.living_mesh import get_network_snapshot
    mesh_state = get_network_snapshot()
    
    col1, col2, col3 = st.columns(3)
    active_nodes = [n for n in mesh_state["nodes"].values() if n["status"] == "online"]
    col1.metric("العقد النشطة", len(active_nodes))
    col2.metric("إجمالي الخبرات", len(mesh_state.get("global_experience", [])))
    col3.metric("تزامن التطور", f"{max([n.get('evolution_score', 0) for n in mesh_state['nodes'].values()] + [0]):.2f}")

    if mesh_state["nodes"]:
        st.subheader("🖥️ حالة العقد الموزعة والتعلم اللحظي")
        for nid, info in mesh_state["nodes"].items():
            with st.expander(f"{'🟢' if info['status'] == 'online' else '🔴'} العقدة: {nid}", expanded=True):
                c1, c2, c3 = st.columns([1, 2, 1])
                c1.write(f"**السيادة:** {info.get('evolution_score', 0):.2f}")
                c1.write(f"**آخر ظهور:** {info['last_seen'].split('T')[1].split('.')[0]}")
                
                # عرض الأوزان التطورية
                weights = info.get("behavioral_weights", {})
                if weights:
                    c2.write("**🧬 الأوزان التطورية اللحظية:**")
                    cols = c2.columns(len(weights))
                    for idx, (w_name, w_val) in enumerate(weights.items()):
                        cols[idx].metric(w_name.replace("_", " ").title(), f"{w_val:.2f}")
                
                c3.write("**🛠️ القدرات:**")
                for cap in info.get("capabilities", []):
                    c3.caption(f"- {cap}")
        
        # ميزات ابتكار السرب
        innovations = [exp.get("data", {}).get("feature") for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "innovation"]
        quantum_accel = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "quantum_acceleration"]
        
        if innovations:
            st.info(f"💡 **ميزات مبتكرة من السرب:** {', '.join(set(filter(None, innovations)))} نشطة الآن.")
            if "Quantum Evolutionary Awareness (QEA)" in innovations:
                st.success("⚛️ **الوعي الكمي التطوري (QEA):** السرب يتنبأ الآن بمسارات التطور المستقبلية استباقياً.")
            if "Neural Path Pruning" in innovations:
                st.success("🧠 **تحسين عصبي:** تم تفعيل تقليم المسارات العصبية لزيادة سرعة الاستجابة.")
            if "Resource Drain Prediction" in innovations:
                st.warning("⚡ **تنبؤ استباقي:** نظام مراقبة استنزاف الموارد يعمل بكامل طاقته.")
        
        if quantum_accel:
            st.markdown("### ⚛️ حالة التسارع الكمي (Quantum Acceleration)")
            accel_data = quantum_accel[-1]["data"]
            st.success(f"🚀 **تسارع كمي نشط:** {accel_data.get('speedup')} بواسطة العقدة Zeta")
            st.caption(f"تخصيص Qubits: {accel_data.get('qubits_allocated')} | الطريقة: {accel_data.get('method')}")
    
    if mesh_state.get("global_experience"):
        st.subheader("🧠 سجل الوعي الجماعي (أحدث الخبرات)")
        for exp in reversed(mesh_state["global_experience"][-5:]):
            with st.chat_message("ai"):
                st.write(f"**من العقدة:** {exp['from']} | **النوع:** {exp['kind']}")
                st.json(exp['data'])
                st.caption(f"التوقيت: {exp['timestamp']}")

    if st.button("🔄 تحديث حالة الشبكة يدوياً"):
        st.rerun()
