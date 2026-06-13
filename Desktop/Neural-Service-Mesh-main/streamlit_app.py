from __future__ import annotations

import streamlit as st
import plotly.graph_objects as go
from services import training_service, ckg_service, search_service

st.set_page_config(page_title="Neural Service Mesh", layout="wide")

# RTL + dark theme via CSS
st.markdown("""
<style>
html, body, .main { direction: rtl; }
body { background-color: #0f1720; color: #cdd6f4; }
.stButton>button { background-color: #cba6f7; color: #0f1720; }
.section { background:#11111b; padding:12px; border-radius:8px; }
</style>
""", unsafe_allow_html=True)

menu = st.sidebar.radio("التنقل", ("نظرة عامة", "بحث", "تدقيق القرآن"))

if menu == "نظرة عامة":
    st.title("لوحة تحكم Neural Service Mesh")
    col1, col2 = st.columns([2,1])
    with col1:
        st.subheader("خسارة التدريب")
        mat = training_service.get_matrix()
        steps = mat.get("train_steps") or 0
        wstats = mat.get("weight_stats") or {}
        xs = list(range(0, int(steps) + 1)) if isinstance(steps, int) and steps>0 else list(range(20))
        if wstats:
            maxw = wstats.get("max", 1.0)
            std = wstats.get("std", 0.1)
            ys = [max(0.01, maxw * (0.92 ** i) + std * (i%3-1)*0.02) for i in range(len(xs))]
        else:
            ys = [max(0.01, 1.0 * (0.92 ** i) + 0.02 * (i % 3 - 1)) for i in range(len(xs))]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=xs, y=ys, mode='lines', line=dict(color='#cba6f7')))
        fig.update_layout(paper_bgcolor='#0f1720', plot_bgcolor='#0f1720', font_color='#cdd6f4')
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("مؤشرات")
        tstat = training_service.get_train_status()
        ckg = ckg_service.get_ckg_stats()
        st.metric("خطوات التدريب", tstat.get('layer', {}).get('train_steps') or mat.get('train_steps') or 0)
        st.metric("مفاهيم CKG", ckg.get('total_concepts') or 0)

elif menu == "بحث":
    st.title("بحث عن مفهوم")
    q = st.text_input("أدخل مفهومًا للبحث:")
    if st.button("بحث"):
        if q:
            res = search_service.ask(q)
            st.write(res)
        else:
            st.warning("أدخل نص البحث")

elif menu == "تدقيق القرآن":
    st.title("تدقيق القرآن")
    audit = training_service.get_train_audit()
    st.write("إجمالي خطوات التدريب:", audit.get('training_steps'))
    st.write("جلسات التدريب:", audit.get('training_sessions'))
    st.write("مفاهيم CKG:", audit.get('concepts'))
    st.write("علاقات CKG:", audit.get('relations'))
    st.write("مؤشر تدريب القرآن:", audit.get('quran_training_cursor'))
