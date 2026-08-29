"""إعدادات مركزية آمنة لجميع وكلاء NSM."""
from __future__ import annotations
import json
import streamlit as st
from ai.telemetry_store import TelemetryStore
from ui_components import render_kpi_cards, render_section_header


def render_agent_settings() -> None:
    store = TelemetryStore()
    render_section_header("إعدادات الوكلاء", "تحكم مركزي بعتبات الأداء والأولوية والتنبيهات", live=False)
    rows = store.list_agent_settings()
    render_kpi_cards([
        {"label": "وكلاء مخصصون", "value": len(rows), "note": "إعدادات محفوظة", "accent": "var(--nsm-indigo)"},
        {"label": "التنبيهات المفعلة", "value": sum(bool(r["notifications_enabled"]) for r in rows), "note": "من الإعدادات الحالية", "accent": "var(--nsm-cyan)"},
        {"label": "متوسط البطء", "value": f"{sum(r['slow_threshold_ms'] for r in rows)/len(rows):.0f} ms" if rows else "5000 ms", "note": "عتبة افتراضية", "accent": "var(--nsm-amber)"},
    ])
    st.caption("يمكنك تعديل عدة وكلاء دفعة واحدة. القيم تُراجع وتُحصر تلقائياً قبل الحفظ.")
    if not rows:
        st.info("لا توجد إعدادات مخصصة بعد. أضف أول وكيل من النموذج أدناه.")
    agents = [r["agent"] for r in rows]
    with st.form("central_agent_settings_form", clear_on_submit=False):
        names = st.text_input("أسماء الوكلاء", value=", ".join(agents), help="افصل الأسماء بفواصل لإدارة عدة وكلاء")
        columns = st.columns(4)
        with columns[0]: slow = st.number_input("عتبة البطء (ms)", 100, 120000, int(rows[0]["slow_threshold_ms"]) if rows else 5000, 100)
        with columns[1]: error = st.slider("معدل الأخطاء", .01, 1.0, float(rows[0]["error_rate_threshold"]) if rows else .25, .01)
        with columns[2]: priority = st.selectbox("الأولوية", ["critical", "warning", "info"], index=["critical", "warning", "info"].index(rows[0]["priority"]) if rows else 1)
        with columns[3]: enabled = st.toggle("التنبيهات مفعلة", value=bool(rows[0]["notifications_enabled"]) if rows else True)
        submitted = st.form_submit_button("حفظ الإعدادات للجميع", type="primary", use_container_width=True)
    if submitted:
        targets = [x.strip() for x in names.split(",") if x.strip()][:100]
        if not targets:
            st.error("أدخل اسماً واحداً على الأقل.")
        else:
            for agent in targets:
                store.save_agent_settings(agent=agent, slow_threshold_ms=slow, error_rate_threshold=error, priority=priority, notifications_enabled=enabled)
            st.success(f"تم حفظ إعدادات {len(targets)} وكيل.")
            st.rerun()
    rows = store.list_agent_settings()
    if rows:
        st.dataframe(rows, use_container_width=True, hide_index=True)
        payload = json.dumps(rows, ensure_ascii=False, indent=2)
        st.download_button("تصدير إعدادات JSON", payload, "nsm-agent-settings.json", "application/json", use_container_width=True)
        uploaded = st.file_uploader("استيراد إعدادات JSON", type=["json"], help="يجب أن يحتوي الملف على قائمة إعدادات وكلاء")
        if uploaded is not None and st.button("تطبيق الملف المستورد", key="apply_agent_settings_import"):
            try:
                imported = json.loads(uploaded.getvalue().decode("utf-8"))
                if not isinstance(imported, list) or len(imported) > 100: raise ValueError
                for item in imported:
                    if not isinstance(item, dict) or not item.get("agent"): raise ValueError
                    store.save_agent_settings(agent=item["agent"], slow_threshold_ms=item.get("slow_threshold_ms", 5000), error_rate_threshold=item.get("error_rate_threshold", .25), priority=item.get("priority", "warning"), notifications_enabled=item.get("notifications_enabled", True))
                st.success("تم استيراد الإعدادات والتحقق منها.")
                st.rerun()
            except (ValueError, UnicodeDecodeError, json.JSONDecodeError):
                st.error("ملف JSON غير صالح أو بنيته غير مدعومة.")
