"""ملفات تعريف وكلاء NSM."""
import streamlit as st
from ai.telemetry_store import TelemetryStore
from ui_components import render_kpi_cards, render_section_header

def render_agent_profiles():
    store = TelemetryStore(); profiles = store.list_agent_profiles(); settings = store.list_agent_settings()
    names = sorted({x['agent'] for x in profiles + settings}) or ['default']
    render_section_header('ملفات تعريف الوكلاء', 'هوية كل وكيل وقدراته وحالته التشغيلية', live=False)
    render_kpi_cards([
        {'label':'ملفات التعريف','value':len(profiles),'note':'محفوظة','accent':'var(--nsm-indigo)'},
        {'label':'مفعلة','value':sum(bool(x['enabled']) for x in profiles),'note':'جاهزة للعمل','accent':'var(--nsm-cyan)'},
        {'label':'متوقفة','value':sum(not bool(x['enabled']) for x in profiles),'note':'تحتاج مراجعة','accent':'var(--nsm-danger)'},
    ])
    selected = st.selectbox('اختر الوكيل', names, key='profile_agent')
    current = store.get_agent_profile(selected)
    with st.form('agent_profile_form'):
        description = st.text_area('الوصف', value=current['description'], max_chars=500)
        capabilities_text = st.text_input('القدرات', value=', '.join(current['capabilities']), help='افصل القدرات بفواصل')
        enabled = st.toggle('الوكيل مفعّل', value=bool(current['enabled']))
        save = st.form_submit_button('حفظ ملف الوكيل', type='primary', use_container_width=True)
    if save:
        caps = [x.strip() for x in capabilities_text.split(',') if x.strip()]
        store.save_agent_profile(agent=selected, description=description, capabilities=caps, enabled=enabled)
        st.success('تم حفظ ملف الوكيل.'); st.rerun()
    if profiles:
        render_section_header('دليل الوكلاء', 'نظرة سريعة على الملفات والحالة')
        for profile in profiles:
            status = 'مفعّل' if profile['enabled'] else 'متوقف'
            st.markdown(f"**{profile['agent']}** · {status}\n\n{profile['description'] or 'لا يوجد وصف'}\n\nالقدرات: {', '.join(profile['capabilities']) or 'غير محددة'}")
            st.divider()
