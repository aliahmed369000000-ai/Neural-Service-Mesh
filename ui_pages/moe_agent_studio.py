"""
ui_pages/moe_agent_studio.py
============================
استوديو MoE + نمو الوكيل — واجهة عملية للتصنيف والصحة والتنفيذ الآمن.
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403


def render_moe_agent_studio():
    st.markdown(
        '<div class="section-header">🧩 استوديو MoE ونمو الوكيل</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "تصنيف هرمي · صحة الخبراء · دورة نمو آمنة (بدون shell حر أو نشر تلقائي)"
    )
    st.info(
        "💡 ابدأ بـ **تصنيف سؤال** أو **صحة MoE**. من الوكيل الموحّد اكتب: `مساعدة` أو `كيف حال النظام`."
    )

    tab_moe, tab_growth, tab_help = st.tabs(
        ["🧩 Hierarchical MoE", "🌱 نمو الوكيل", "ℹ️ أوامر"]
    )

    with tab_moe:
        _render_moe_panel()

    with tab_growth:
        _render_growth_panel()

    with tab_help:
        st.markdown(
            """
### أوامر نصية (من المحادثة أو الوكيل الموحّد)

| الأمر | الوظيفة |
|--------|---------|
| `صنّف: ...` | تصنيف + ثقة + خبراء |
| `صحة moe` | تقرير صحة النظام |
| `إحصاء moe` | إحصاء التصنيفات المستمرة |
| `حالة نمو الوكيل` | ذاكرة وخبرات |
| `خطة: <هدف>` | تخطيط فقط |
| `نفّذ بأمان: <هدف>` | تنفيذ أدوات آمنة |
| `طوّر الوكيل` | فحص + تكامل + اختبارات محدودة |

في **المحادثة المعرفية** يظهر شريط توجيه MoE أعلى الإجابة عند تفعيل الطبقة.
"""
        )


def _render_moe_panel():
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("🩺 صحة MoE", use_container_width=True, key="moe_ui_health"):
            try:
                from ai.moe_ckg_bridge import get_moe_bridge

                st.session_state["_moe_ui_health"] = get_moe_bridge().health_report()
            except Exception as e:
                st.session_state["_moe_ui_health"] = f"خطأ: {e}"
    with col_b:
        if st.button("📈 إحصاء مستمر", use_container_width=True, key="moe_ui_stats"):
            try:
                from ai.moe_continual import stats_report

                st.session_state["_moe_ui_stats"] = stats_report()
            except Exception as e:
                st.session_state["_moe_ui_stats"] = f"خطأ: {e}"

    if st.session_state.get("_moe_ui_health"):
        st.markdown(st.session_state["_moe_ui_health"])
    if st.session_state.get("_moe_ui_stats"):
        st.markdown(st.session_state["_moe_ui_stats"])

    st.markdown("---")
    # 🆕 كان st.subheader (عنوان Streamlit الافتراضي بلا CSS الثيم المخصص —
    # لا يقع تحت محدّد .stMarkdown h3 المُوحَّد لبقية عناوين المستوى الفرعي
    # بالتطبيق، فيظهر بحجم/خط مختلفَين). استبدلته بنفس صياغة "### " المستخدمة
    # لعناوين المستوى نفسه بصفحات أخرى (مثال: aiaas_console.py، ultraplinian.py).
    st.markdown("### 🏷️ صنّف سؤالاً")
    q = st.text_input(
        "نص السؤال",
        placeholder="مثال: ما حكم الصلاة في المذهب الشافعي؟",
        key="moe_ui_q",
    )
    adapt = st.checkbox("تكيّف خفيف عند ثقة منخفضة", value=True, key="moe_ui_adapt")
    if st.button("تصنيف", type="primary", use_container_width=True, key="moe_ui_cls"):
        if not (q or "").strip():
            st.warning("أدخل نصاً للتصنيف")
        else:
            try:
                from ai.moe_continual import classify_and_adapt

                r = classify_and_adapt(q.strip(), adapt=adapt)
                st.session_state["_moe_ui_cls"] = r
            except Exception as e:
                st.error(f"تعذّر التصنيف: {e}")

    r = st.session_state.get("_moe_ui_cls")
    if isinstance(r, dict):
        c1, c2, c3 = st.columns(3)
        c1.metric("الفئة", r.get("top", "—"))
        conf = r.get("confidence")
        c2.metric("الثقة", f"{float(conf):.0%}" if conf is not None else "—")
        c3.metric("المصدر", r.get("source", "—"))
        experts = r.get("experts") or []
        if experts:
            st.markdown("**خبراء مقترحون:** " + " · ".join(f"`{e}`" for e in experts[:6]))
        alts = r.get("alternatives") or []
        if alts:
            st.markdown(
                "**بدائل:** "
                + " · ".join(f"`{a.get('category')}` ({a.get('weight')})" for a in alts)
            )
        if r.get("adapted"):
            st.info("تم تكيّف خفيف لراوتر الفئات بعد ثقة منخفضة.")


def _render_growth_panel():
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🌱 حالة النمو", use_container_width=True, key="gr_status"):
            from ai.agent_growth_loop import growth_status

            st.session_state["_gr_out"] = growth_status()
    with c2:
        if st.button("🧠 الخبرات", use_container_width=True, key="gr_exp"):
            from ai.agent_growth_loop import list_experiences

            st.session_state["_gr_out"] = list_experiences()
    with c3:
        if st.button("🚀 طوّر الوكيل", use_container_width=True, key="gr_dev"):
            with st.spinner("دورة نمو آمنة..."):
                from ai.agent_growth_loop import develop_agent_once

                st.session_state["_gr_out"] = develop_agent_once()

    st.markdown("---")
    goal = st.text_area(
        "هدف للتخطيط أو التنفيذ الآمن",
        placeholder="افحص المشروع وشغّل اختبارات",
        key="gr_goal",
        height=80,
    )
    b1, b2 = st.columns(2)
    with b1:
        if st.button("🗺️ خطة فقط", use_container_width=True, key="gr_plan"):
            if not (goal or "").strip():
                st.warning("أدخل هدفاً")
            else:
                from ai.agent_growth_loop import run_safe_mission

                st.session_state["_gr_out"] = run_safe_mission(goal.strip(), execute=False)
    with b2:
        if st.button("⚙️ نفّذ بأمان", type="primary", use_container_width=True, key="gr_run"):
            if not (goal or "").strip():
                st.warning("أدخل هدفاً")
            else:
                with st.spinner("تنفيذ أدوات آمنة..."):
                    from ai.agent_growth_loop import run_safe_mission

                    st.session_state["_gr_out"] = run_safe_mission(goal.strip(), execute=True)

    if st.session_state.get("_gr_out"):
        st.markdown(st.session_state["_gr_out"])
