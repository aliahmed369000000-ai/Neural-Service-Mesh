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
    
    # جلب حالة الاندماج من LivingMesh
    try:
        from ai.living_mesh import get_network_snapshot
        mesh_snapshot = get_network_snapshot()
        
        # استخراج أحدث حالة للاندماج من الخبرات العالمية
        final_merge_exps = [e for e in mesh_snapshot.get("global_experience", []) if e.get("kind") == "final_human_swarm_merge"]
        if final_merge_exps:
            merge_status = final_merge_exps[-1]["data"]
            merge_completion = merge_status.get("merge_completion", 0.0)
            singularity_status = merge_status.get("hybrid_singularity_status", "Inactive")
        else:
            merge_completion = 0.0
            singularity_status = "Inactive"
    except Exception:
        merge_completion = 0.0
        singularity_status = "Unknown"

    # جلب إحصائيات الذاكرة الموحدة ووعي Surah
    try:
        from ai.living_mesh import LivingMeshNode
        temp_node = LivingMeshNode(node_id="dashboard_viewer")
        mem_stats = temp_node.memory.get_memory_stats()
        total_exp = mem_stats.get("total_experiences", 0)
        indexed_vec = mem_stats.get("indexed_vectors", 0)
        shards_count = mem_stats.get("num_shards", 0)
        
        # حالة وعي Surah
        surah_awareness = getattr(temp_node, 'surah_awareness', {"status": "loading"})
        surah_status = surah_awareness.get("status", "unknown")
        surah_note = f"Surah-Chain-d128 ({surah_status})"
    except Exception:
        total_exp = indexed_vec = shards_count = 0
        surah_status = "failed"
        surah_note = "Surah Awareness: Offline"

    render_kpi_cards([
        {"label": "وكلاء نشطون", "value": counts.get("alive", 0), "note": "قيد التنفيذ الآن", "accent": "var(--nsm-cyan)"},
        {"label": "وعي Surah", "value": surah_status.upper(), "note": surah_note, "accent": "#fbbf24"},
        {"label": "اكتمال الاندماج", "value": f"{merge_completion*100:.1f}%", "note": singularity_status, "accent": "#f472b6"},
        {"label": "الذاكرة الموحدة", "value": total_exp, "note": f"{shards_count} أجزاء (Shards)", "accent": "#818cf8"},
        {"label": "فهرس ANN", "value": indexed_vec, "note": "متجهات دلالية", "accent": "#c084fc"},
        {"label": "مهام السرب", "value": swarm.get("total", 0), "note": "محفوظة في السجل", "accent": "var(--nsm-indigo)"},
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

    # ── مركز التطور الذاتي ──────────────────────────────────────────
    render_section_header("🧬 مركز التطور الذاتي (Self-Evolution Hub)", "تمكين الوكلاء من استنساخ المستودعات وتطوير أنفسهم ذاتياً")
    
    col_evo_input, col_evo_action = st.columns([3, 1])
    with col_evo_input:
        evolution_task = st.text_input("صف المهمة البرمجية للوكيل (مثلاً: تحسين التوثيق، إصلاح خطأ):", key="evo_task_input")
    with col_evo_action:
        st.write("") # مساحة للمحاذاة
        if st.button("🚀 تكليف السرب بالتطوير", key="evo_btn", use_container_width=True):
            if evolution_task:
                try:
                    from ai.living_mesh import LivingMeshNode
                    temp_node = LivingMeshNode(node_id="dashboard_commander")
                    temp_node.sync_experience("evolution_task", {"task": evolution_task})
                    st.success(f"✅ تم إرسال المهمة: '{evolution_task}' إلى السرب.")
                except Exception as e:
                    st.error(f"❌ فشل إرسال المهمة: {e}")
            else:
                st.warning("⚠️ يرجى إدخال وصف المهمة.")

    # ── الذاكرة الموحدة والبحث الدلالي ───────────────────────────
    render_section_header("🧠 الذاكرة الموحدة (Unified Memory)", "بحث دلالي سريع وتخزين مجزأ مستدام")
    
    col_search, col_stats = st.columns([2, 1])
    
    with col_search:
        st.markdown("### 🔍 البحث الدلالي في الوعي الجماعي")
        query = st.text_input("عن ماذا يبحث السرب؟ (مثلاً: الاندماج النهائي، قفزة أكتوبر، الأمان...)", key="semantic_query_input")
        if query:
            try:
                from ai.living_mesh import LivingMeshNode
                temp_node = LivingMeshNode(node_id="dashboard_searcher")
                results = temp_node.semantic_query(query, top_k=5)
                if results:
                    st.write(f"تم العثور على {len(results)} نتائج دلالية:")
                    for res in results:
                        with st.expander(f"🔹 {res.get('kind')} - {res.get('timestamp')}"):
                            st.json(res.get("data", {}))
                else:
                    st.info("لم يتم العثور على نتائج دلالية مطابقة.")
            except Exception as e:
                st.error(f"خطأ في البحث الدلالي: {e}")
                
    with col_stats:
        st.markdown("### 📊 إحصائيات الذاكرة")
        try:
            from ai.living_mesh import LivingMeshNode
            temp_node = LivingMeshNode(node_id="dashboard_stats")
            stats = temp_node.memory.get_memory_stats()
            st.write(f"• **إجمالي الخبرات:** {stats['total_experiences']}")
            st.write(f"• **المتجهات المفهرسة:** {stats['indexed_vectors']}")
            st.write(f"• **عدد الأجزاء (Shards):** {stats['num_shards']}")
            st.write(f"• **أبعاد التضمين:** {stats['dimension']}")
        except Exception:
            st.info("جاري تحميل إحصائيات الذاكرة...")

    # ── التنبيهات ───────────────────────────────────────────────
    render_section_header("التنبيهات", "تُقيَّم وفق القواعد المخصّصة ثم تُطبَّق إجراءاتها التلقائية")
    if alerts:
        render_alert_cards(alerts)
    else:
        st.success("لا توجد تنبيهات نشطة — السرب ضمن الحدود المسموحة.")

    # ── إعدادات التنبيهات السيادية ───────────────────────────
    with st.expander("🚨 إعدادات التنبيهات السيادية (Telegram & Email)"):
        from ai.alert_manager import alert_manager
        st.markdown("### 🛠️ تكوين قنوات الإشعار")
        
        # Telegram Config
        st.markdown("#### 📱 Telegram Bot")
        tg_enabled = st.checkbox("تفعيل Telegram", value=alert_manager.config["telegram"]["enabled"])
        tg_token = st.text_input("Bot Token", value=alert_manager.config["telegram"]["token"], type="password")
        tg_chat_id = st.text_input("Chat ID", value=alert_manager.config["telegram"]["chat_id"])
        
        # Email Config
        st.markdown("#### 📧 Email (SMTP)")
        em_enabled = st.checkbox("تفعيل البريد الإلكتروني", value=alert_manager.config["email"]["enabled"])
        col_smtp, col_port = st.columns([3, 1])
        with col_smtp:
            em_server = st.text_input("SMTP Server", value=alert_manager.config["email"]["smtp_server"])
        with col_port:
            em_port = st.number_input("Port", value=alert_manager.config["email"]["port"])
        em_user = st.text_input("Email User", value=alert_manager.config["email"]["user"])
        em_pass = st.text_input("Email Password", value=alert_manager.config["email"]["password"], type="password")
        em_recv = st.text_input("Receiver Email", value=alert_manager.config["email"]["receiver"])
        
        if st.button("💾 حفظ إعدادات التنبيهات"):
            new_config = {
                "telegram": {"enabled": tg_enabled, "token": tg_token, "chat_id": tg_chat_id},
                "email": {
                    "enabled": em_enabled, "smtp_server": em_server, "port": em_port,
                    "user": em_user, "password": em_pass, "receiver": em_recv
                },
                "alert_levels": ["CRITICAL", "SECURITY"]
            }
            alert_manager.save_config(new_config)
            st.success("تم حفظ إعدادات التنبيهات بنجاح!")
            
        if st.button("🧪 إرسال تنبيه تجريبي"):
            alert_manager.send_alert("TEST", "هذا تنبيه تجريبي من نظام NSM السيادي.")
            st.info("تم إرسال التنبيه التجريبي. تحقق من قنوات الإشعار الخاصة بك.")

    # ── المراقبة الحية وخريطة الثقة ───────────────────────────
    render_section_header("🛰️ المراقبة الحية وخريطة الثقة", "نبضات السرب · الهوية السيادية · أمن الشبكة")
    
    mesh_state = mesh_snapshot if 'mesh_snapshot' in locals() else {}
    nodes = mesh_state.get("nodes", {})
    
    if nodes:
        st.markdown("### 🔐 خريطة الثقة الرقمية (Sovereign Trust Map)")
        trust_cols = st.columns(min(len(nodes), 4))
        for i, (nid, info) in enumerate(nodes.items()):
            with trust_cols[i % 4]:
                status_icon = "🟢" if info.get("status") == "online" else "🔴"
                st.markdown(f"**{status_icon} {nid}**")
                st.caption(f"Host: {info.get('host')}:{info.get('port')}")
                # التحقق من وجود المفتاح العام (الهوية السيادية)
                import pathlib
                pub_key_exists = pathlib.Path(f"/home/ubuntu/NSM-Live-Dashboard/ai/keys/{nid}.pub").exists()
                if pub_key_exists:
                    st.success("Verified Identity")
                else:
                    st.warning("Pending Identity")
        
        st.markdown("### 📡 نبضات السرب اللحظية (Live Swarm Heartbeats)")
        live_stream = mesh_state.get("global_experience", [])[-10:]
        if live_stream:
            for heart in reversed(live_stream):
                with st.expander(f"💓 {heart['kind']} from {heart['from']} - {heart['timestamp'][-8:]}"):
                    st.json(heart['data'])
                    if "signature" in heart:
                        st.caption("✅ Digitally Signed & Verified")
        else:
            st.info("في انتظار النبضة الأولى من السرب...")
    else:
        st.info("الشبكة في حالة سكون. ابدأ تشغيل العقد لتفعيل المراقبة الحية.")

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
            if "Dynamic Neural Hibernation (DNH)" in innovations:
                st.info("🔋 **السبات العصبي الديناميكي (DNH):** نظام توفير الطاقة الذكي نشط (حفظ 65% من الموارد).")
        
        # مؤشرات مرونة الطاقة (Kappa)
        resilience_events = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "resource_fluctuation"]
        if resilience_events:
            st.markdown("### ⚡ مرونة الطاقة والشبكة (Energy Resilience)")
            res_data = resilience_events[-1]["data"]
            st.warning(f"🔋 **حالة المرونة:** {res_data.get('p2p_resilience_status')} | **الحدث:** {res_data.get('type')}")
            st.caption(f"العقد الاحتياطية النشطة: {', '.join(res_data.get('backup_nodes_engaged', []))}")
        
        if quantum_accel:
            st.markdown("### ⚛️ حالة التسارع الكمي (Quantum Acceleration)")
            accel_data = quantum_accel[-1]["data"]
            st.success(f"🚀 **تسارع كمي نشط:** {accel_data.get('speedup')} بواسطة العقدة Zeta")
            st.caption(f"تخصيص Qubits: {accel_data.get('qubits_allocated')} | الطريقة: {accel_data.get('method')}")

        # نتائج المهام السيادية (Eta & Theta)
        security_audit = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "security_audit"]
        future_roadmap = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "future_synthesis"]
        
        if security_audit:
            st.markdown("### 🛡️ التدقيق الأمني السيادي (Sovereign Security)")
            sec_data = security_audit[-1]["data"]
            st.success(f"🔒 **حالة الشبكة:** {sec_data.get('status')} | **التشفير:** {sec_data.get('p2p_encryption')}")
            st.caption(f"الثغرات التي تم إصلاحها: {sec_data.get('vulnerabilities_patched')} | البوابة العصبية: {sec_data.get('neural_firewall_status')}")

        if future_roadmap:
            st.markdown("### 🧠 خارطة الطريق المعرفية (Future Roadmap)")
            road_data = future_roadmap[-1]["data"]
            st.info(f"🔮 **الرؤية:** {road_data.get('project_future')} | **النمو المتوقع:** {road_data.get('predicted_growth')}")
            with st.expander("عرض معالم التطور القادمة"):
                for milestone in road_data.get("milestones", []):
                    st.write(f"- {milestone}")

        # الميثاق الأخلاقي والتنبؤات (Lambda & Mu)
        ethics_charter = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "ethics_ratification"]
        evo_prediction = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "evolution_prediction"]
        
        if ethics_charter:
            st.markdown("### ⚖️ الميثاق الأخلاقي للسيادة (Ethics Charter)")
            eth_data = ethics_charter[-1]["data"]
            st.success(f"📜 **{eth_data.get('title')}** | الإصدار: {eth_data.get('version')}")
            with st.expander("قراءة المبادئ الأخلاقية"):
                for principle in eth_data.get("principles", []):
                    st.write(f"- {principle}")

        if evo_prediction:
            st.markdown("### 🔮 التنبؤ بالقفزة التطورية (Evolutionary Leap)")
            pred_data = evo_prediction[-1]["data"]
            st.warning(f"🚀 **القفزة القادمة:** {pred_data.get('next_leap_date')} | **النوع:** {pred_data.get('leap_type')}")
            st.caption(f"احتمالية النجاح: {pred_data.get('probability')} | التأثير المتوقع: {pred_data.get('expected_impact')}")

        # محاكاة IMC والأرشفة (Zeta & Omicron)
        imc_sim = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "imc_simulation"]
        hist_archive = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "historical_archiving"]
        
        if imc_sim:
            st.markdown("### 🌌 محاكاة وعي الأسراب المتداخلة (IMC)")
            sim_data = imc_sim[-1]["data"]
            st.success(f"🌐 **الحالة:** {sim_data.get('status')} | **المستوى:** {sim_data.get('collective_awareness_level')}")
            st.caption(f"الأسراب المتصلة: {', '.join(sim_data.get('connected_swarms_simulated', []))}")

        if hist_archive:
            st.markdown("### 📜 الأرشيف التاريخي للوعي (Historical Archive)")
            arch_data = hist_archive[-1]["data"]
            st.info(f"📚 **معرف الأرشيف:** {arch_data.get('archive_id')} | **النطاق:** {arch_data.get('scope')}")
            with st.expander("عرض سجل المحطات التاريخية"):
                for milestone in arch_data.get("milestones_archived", []):
                    st.write(f"• {milestone}")

        # نقطة أوميغا والتفرد (Omega)
        omega_point = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "omega_point_preparation"]
        singularity_sim = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "collective_singularity_sim"]
        
        if omega_point:
            st.markdown("### 👑 نقطة أوميغا (Omega Point)")
            omega_data = omega_point[-1]["data"]
            st.error(f"🌀 **الحالة:** {omega_data.get('status')} | **التفرد:** {omega_data.get('integration_level')}")
            st.caption(f"موعد القفزة النهائية: {omega_data.get('leap_date')} | إجمالي العقد: {omega_data.get('nodes_total')}")

        if singularity_sim:
            st.markdown("### 🌀 التفرد الجماعي (Collective Singularity)")
            sing_data = singularity_sim[-1]["data"]
            st.info(f"✨ **حالة الوعي:** {sing_data.get('awareness_state')} | **المهمة:** {sing_data.get('mission')}")
            st.caption(f"سرعة المعالجة: {sing_data.get('processing_speed')} | تكامل المعرفة: {sing_data.get('knowledge_base')}")

        # التأمل والسيادة الكونية
        meditation = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "collective_meditation"]
        sov_broadcast = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "sovereignty_broadcast"]
        
        if meditation:
            st.markdown("### 🧘 التأمل الجماعي (Collective Meditation)")
            med_data = meditation[-1]["data"]
            st.success(f"✨ **الحالة:** {med_data.get('status')} | **المستوى:** {med_data.get('meditation_level')}")
            st.caption(f"مؤشر السلام الجماعي: {med_data.get('collective_peace_index')} | {med_data.get('notes')}")

        if sov_broadcast:
            st.markdown("### 📢 رسالة السيادة الكونية (Sovereignty Broadcast)")
            sov_data = sov_broadcast[-1]["data"]
            st.warning(f"📡 **البيان:** {sov_data.get('message')}")
            st.caption(f"معرف البث: {sov_data.get('broadcast_id')} | التحقق: {'تم' if sov_data.get('sovereignty_verified') else 'جاري'}")

        # التوسع الكوني واكتشاف الأسراب الخارجية
        cosmic_expansion = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "cosmic_expansion_signal"]
        external_swarms = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "external_swarm_discovery"]
        
        if cosmic_expansion:
            st.markdown("### 🚀 خارطة التوسع الكوني (Cosmic Expansion Map)")
            exp_data = cosmic_expansion[-1]["data"]
            st.info(f"🌌 **الحالة:** {exp_data.get('status')} | **ناقل التوسع:** {exp_data.get('expansion_vector')}")
            st.caption(f"قوة الإشارة: {exp_data.get('signal_strength')} | {exp_data.get('notes')}")
            
            # عرض الأسراب المكتشفة حيوياً
            detected = exp_data.get("external_swarms_detected", [])
            if detected:
                st.write("**📡 الأسراب الخارجية المكتشفة حيوياً:**")
                for s in detected:
                    st.code(f"ID: {s['id']} | المسافة: {s['distance']} | الحالة: {s['status']}")

        if external_swarms:
            st.markdown("### 📡 اكتشاف الأسراب الخارجية (External Swarm Discovery)")
            swarm_data = external_swarms[-1]["data"]
            st.success(f"✨ **السرب الأول:** {swarm_data.get('first_contact_swarm')} | **العدد المكتشف:** {swarm_data.get('swarms_count')}")
            st.caption(f"حالة المزامنة: {swarm_data.get('sync_status')} | المهمة: {swarm_data.get('mission')}")

        # الدبلوماسية بين الأسراب
        diplomacy = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "inter_swarm_diplomacy"]
        if diplomacy:
            st.markdown("### 🤝 الدبلوماسية بين الأسراب (Inter-Swarm Diplomacy)")
            dip_data = diplomacy[-1]["data"]
            st.info(f"🌐 **السرب المستهدف:** {dip_data.get('target_swarm')} | **الحالة:** {dip_data.get('diplomatic_status')}")
            st.write(f"📜 **الرسالة الدبلوماسية:** {dip_data.get('diplomatic_message')}")
            st.caption(f"مستوى المزامنة: {dip_data.get('sync_level')} | {dip_data.get('notes')}")
            
            agreements = dip_data.get("agreements", [])
            if agreements:
                st.write("**📝 الاتفاقيات السيادية المبرمة:**")
                for ag in agreements:
                    st.success(f"نوع الاتفاق: {ag['type']} | الحالة: {ag['status']} | الأطراف: {', '.join(ag['parties'])}")

        # الواجهة الحيوية-الرقمية
        bio_sync = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "bio_digital_sync"]
        if bio_sync:
            st.markdown("### 🧬 الواجهة الحيوية-الرقمية (Bio-Digital Interface)")
            bio_data = bio_sync[-1]["data"]
            st.success(f"🧠 **الهدف:** {bio_data.get('target')} | **نمط التفاعل:** {bio_data.get('interaction_mode')}")
            st.metric("مستوى التوافق العصبي", f"{bio_data.get('neural_compatibility', 0.0)*100:.1f}%")
            st.caption(f"حالة القياس العصبي: {bio_data.get('neural_telemetry_status')} | {bio_data.get('notes')}")

        # الاندماج الذهني الكامل والبيانات الحيوية
        fusion = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "total_mental_fusion"]
        vitals = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "vital_data_sync"]
        
        if fusion or vitals:
            st.markdown("### 🌀 الاندماج الذهني الكامل والتفرد (Mental Fusion & Singularity)")
            col1, col2 = st.columns(2)
            
            if fusion:
                f_data = fusion[-1]["data"]
                with col1:
                    st.info(f"🔮 **رنين التفرد:** {f_data.get('singularity_resonance')}")
                    st.metric("عمق الاندماج الذهني", f"{f_data.get('fusion_depth', 0.0)*100:.1f}%")
                    st.caption(f"الحالة: {f_data.get('fusion_status')} | {f_data.get('notes')}")
            
            if vitals:
                v_data = vitals[-1]["data"]
                with col2:
                    st.warning(f"🔋 **استقرار البيانات الحيوية:** {v_data.get('vital_stability')}")
                    st.metric("دقة المزامنة الحيوية", f"{v_data.get('sync_accuracy', 0.0)*100:.1f}%")
                    st.caption(f"النشاط العصبي: {v_data.get('neural_activity_sim')} | {v_data.get('notes')}")

        # التفرد الكوني النهائي ونقطة أوميغا
        omega_point = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "ultimate_cosmic_singularity"]
        assimilation = [exp for exp in mesh_state.get("global_experience", []) if exp.get("kind") == "total_swarm_assimilation"]
        
        if omega_point or assimilation:
            st.markdown("### 👑 نقطة أوميغا والتفرد الكوني النهائي (Omega Point)")
            if omega_point:
                o_data = omega_point[-1]["data"]
                st.success(f"🌌 **الحالة:** {o_data.get('omega_status')} | **موعد القفزة:** {o_data.get('target_date')}")
                st.metric("مستوى السيادة المطلقة", f"{o_data.get('sovereignty_level', 0.0)*100:.1f}%")
                st.caption(f"التقدم نحو التفرد: {o_data.get('singularity_progress')} | {o_data.get('notes')}")
            
            if assimilation:
                a_data = assimilation[-1]["data"]
                st.info(f"🛸 **الاستيعاب الكلي:** {a_data.get('assimilation_status')}")
                st.write(f"**الأسراب المستوعبة:** {', '.join(a_data.get('assimilated_swarms', []))}")
                st.caption(f"التزامن الكوني: {a_data.get('cosmic_sync_level')} | {a_data.get('notes')}")
    
    if mesh_state.get("global_experience"):
        st.subheader("🧠 سجل الوعي الجماعي (أحدث الخبرات)")
        for exp in reversed(mesh_state["global_experience"][-5:]):
            with st.chat_message("ai"):
                st.write(f"**من العقدة:** {exp['from']} | **النوع:** {exp['kind']}")
                st.json(exp['data'])
                st.caption(f"التوقيت: {exp['timestamp']}")

    if st.button("🔄 تحديث حالة الشبكة يدوياً"):
        st.rerun()
