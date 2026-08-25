# -*- coding: utf-8 -*-
"""🧭 لوحة السرب الموحدة — Adaptive Swarm Dashboard.

تجمع حالة السرب كلها في مكان واحد:
الوكلاء الأحياء/المنتهون/الفاشلون · مهام السرب المحفوظة · المهام طويلة الأمد ·
التنبيهات القابلة للتخصيص والإجراءات التلقائية.
"""
from __future__ import annotations

import streamlit as st  # noqa: F401
from html import escape as _escape

from app_core import *  # noqa: F401,F403
from typing import Any, Dict, List

from ui_components import render_agent_cards, render_alert_cards, render_kpi_cards, render_section_header
try:
    import plotly.graph_objects as go
except ImportError:
    go = None

def _esc(value: Any) -> str:
    return _escape(str(value or ""))

def render_unified_swarm_dashboard() -> None:
    """لوحة السرب الموحدة: حالة كل الوكلاء والسرب في نظرة واحدة."""
    from ai.unified_swarm_dashboard import (
        agents_overview,
        apply_auto_actions,
        evaluate_alerts,
        unified_dashboard_snapshot,
    )

    render_section_header(
        "🧭 لوحة السرب الموحدة",
        "الوكلاء · السرب · المهام طويلة الأمد · التنبيهات",
        live=True,
    )
    
    # تهيئة عقدة العرض لجلب البيانات الحية
    try:
        from ai.living_mesh import LivingMeshNode
        node = LivingMeshNode(node_id="dashboard_viewer")
    except Exception:
        node = None

    col_refresh, col_apply = st.columns([2, 1])
    with col_refresh:
        if st.button("🔄 تحديث اللوحة", key="swarm_dash_refresh", use_container_width=True):
            st.rerun()
    with col_apply:
        if st.button("⚡ طبّق الإجراءات التلقائية الآن", key="swarm_dash_apply"):
            alerts = evaluate_alerts()
            applied = apply_auto_actions(alerts)
            if applied:
                st.success(f"طُبّقت {len(applied)} إجراءات تلقائية")
                st.rerun()

    snapshot = unified_dashboard_snapshot()
    agents = snapshot.get("agents", {})
    counts = agents.get("counts", {})
    
    # جلب إحصائيات الذاكرة الموحدة ووعي Surah
    if node:
        mem_stats = node.memory.get_memory_stats()
        total_exp = mem_stats.get("total_experiences", 0)
        indexed_vec = mem_stats.get("indexed_vectors", 0)
        shards_count = mem_stats.get("num_shards", 0)
        surah_awareness = getattr(node, 'surah_awareness', {"status": "loading"})
        surah_status = surah_awareness.get("status", "unknown")
    else:
        total_exp = indexed_vec = shards_count = 0
        surah_status = "offline"

    render_kpi_cards([
        {"label": "وكلاء نشطون", "value": counts.get("alive", 0), "note": "قيد التنفيذ الآن", "accent": "var(--nsm-cyan)"},
        {"label": "وعي Surah", "value": surah_status.upper(), "note": "Surah-Chain-d128", "accent": "#fbbf24"},
        {"label": "الذاكرة الموحدة", "value": total_exp, "note": f"{shards_count} أجزاء", "accent": "#818cf8"},
        {"label": "فهرس ANN", "value": indexed_vec, "note": "متجهات دلالية", "accent": "#c084fc"},
    ])

    # ── مركز التطور الذاتي ──────────────────────────────────────────
    st.divider()
    render_section_header("🧬 مركز التطور الذاتي (Self-Evolution Hub)", "تمكين الوكلاء من تطوير أنفسهم ذاتياً")
    
    evolution_task = st.text_input("صف المهمة البرمجية (مثلاً: تحسين التوثيق، إصلاح خطأ):", key="evo_task_input")
    if st.button("🚀 تكليف السرب بالتطوير", key="evo_btn"):
        if evolution_task and node:
            node.sync_experience("evolution_task", {"task": evolution_task})
            st.success("✅ تم إرسال مهمة التطوير للسرب بنجاح!")

    # ── صندوق الأدوات العالمي ──────────────────────────────────────────
    st.divider()
    render_section_header("🛠️ صندوق الأدوات العالمي (Global Toolbox)", "الأدوات المتاحة للوكلاء لتنفيذ المهام")
    
    if node:
        available_tools = node.toolbox.list_tools()
        import pandas as pd
        tools_df = pd.DataFrame(available_tools)
        st.table(tools_df[['name', 'category', 'description']])
        
        st.write("#### ⚙️ تنفيذ أداة برمجياً")
        col_tool, col_args = st.columns([1, 2])
        with col_tool:
            selected_tool = st.selectbox("اختر الأداة", options=[t['name'] for t in available_tools])
        with col_args:
            tool_args = st.text_input("المعاملات (JSON)", value="{}")
            
        if st.button("▶️ تشغيل الأداة في السرب"):
            try:
                import json
                args = json.loads(tool_args)
                node.sync_experience("tool_request", {"tool_name": selected_tool, "args": args})
                st.info(f"تم إرسال طلب تنفيذ '{selected_tool}' إلى الشبكة الموزعة.")
            except Exception as e:
                st.error(f"خطأ في المعاملات: {e}")
    else:
        st.error("صندوق الأدوات غير متاح حالياً.")
