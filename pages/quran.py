"""
pages/quran.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_quran():
    """تبويب القرآن الكريم."""
    st.markdown('<div class="section-header">📖 القرآن الكريم في النظام</div>', unsafe_allow_html=True)

    quran_index = load_quran_index()
    ayat        = load_all_quran_ayat()
    roots       = load_arabic_roots()

    # إحصاءات
    col1, col2, col3 = st.columns(3)
    with col1: metric_card(f"{quran_index.get('total_ayat', len(ayat)):,}", "آية محملة")
    with col2: metric_card(f"{quran_index.get('total_surahs', 114)}", "سورة")
    with col3: metric_card(f"{len(roots):,}", "مفهوم مستخرج")

    st.markdown("")

    # أكثر المفاهيم تكراراً
    st.markdown('<div class="section-header">🔝 أكثر المفاهيم تكراراً في القرآن</div>', unsafe_allow_html=True)

    # فلترة الجذور ذات المعنى
    filtered = {k: v for k, v in roots.items()
                if len(normalize_arabic(k)) >= 3
                and v.get("frequency", 0) > 50
                and normalize_arabic(k) not in {
                    "من", "في", "على", "إلى", "عن", "مع", "الا", "ومن",
                    "وان", "بهۦ", "بما", "وما", "الذ", "وقا", "وله"
                }}

    top_concepts = sorted(filtered.items(), key=lambda x: x[1].get("frequency", 0), reverse=True)[:20]

    if top_concepts:
        # رسم بياني
        try:
            import plotly.graph_objects as go
            names = [v.get("top_token", k) for k, v in top_concepts[:15]]
            freqs = [v.get("frequency", 0) for _, v in top_concepts[:15]]

            _theme = THEMES.get(st.session_state.get("ui_theme", "dark"), THEMES["dark"])
            fig = go.Figure(go.Bar(
                x=freqs,
                y=names,
                orientation='h',
                marker_color=_theme["gold"],
                text=freqs,
                textposition='outside',
                textfont=dict(color=_theme["text"]),
            ))
            fig.update_layout(
                height=450,
                margin=dict(l=20, r=60, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color=_theme["text"]),
                yaxis=dict(autorange="reversed", color=_theme["text"], gridcolor=_theme["border"]),
                xaxis=dict(color=_theme["text"], gridcolor=_theme["border"]),
                xaxis_title="التكرار",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for k, v in top_concepts[:10]:
                token = v.get("top_token", k)
                freq  = v.get("frequency", 0)
                st.markdown(f"**{token}**: {freq:,} مرة")
    else:
        st.info("لم تُكتشف مفاهيم بعد. يحتاج النظام إلى تدريب إضافي.")

    # بحث داخل القرآن
    st.markdown('<div class="section-header">🔍 البحث في آيات القرآن</div>', unsafe_allow_html=True)
    quran_q = st.text_input("بحث قرآن", placeholder="ابحث عن كلمة أو مفهوم...", key="quran_search",
                             label_visibility="collapsed")
    if quran_q.strip():
        matches = search_quran_for_concept(quran_q.strip(), ayat, max_results=20)
        if matches:
            st.success(f"وُجد {len(matches)} آية تحتوي على «{quran_q}»")
            for ayah in matches:
                surah = ayah.get("surah", "")
                verse = ayah.get("ayah", "")
                text  = ayah.get("text", "")
                st.markdown(f"""
                <div class="quran-verse">
                    {text}
                    <div class="verse-ref">سورة {surah}، الآية {verse}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"لم يُعثر على «{quran_q}» في الآيات المحملة.")
