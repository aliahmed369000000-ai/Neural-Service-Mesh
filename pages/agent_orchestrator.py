"""
pages/agent_orchestrator.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب 🤝 منسّق الوكلاء — توزيع مهمة واحدة على وكلاء Agents Hub الفعليين
# ══════════════════════════════════════════════════════════════════════════
def render_agent_orchestrator():
    """يوجّه مهمة/سؤال المستخدم تلقائياً إلى وكيل أو أكثر من وكلاء
    "🤖 وكلاء AI" الفعليين (نفس جلسات session_state وذاكرة المحادثة
    المستخدَمة في تبويب Agents Hub)، ثم يعرض ردودهم، مع توليف اختياري
    لإجابة موحّدة. يطبّق نمط Multi-Agent Systems: تفويض مهمة رئيسية إلى
    وكلاء متخصصين ثم تجميع نتائجهم عبر وكيل "منسّق"."""
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <span style="font-size:2rem">🤝</span>
        <div style="font-size:1.5rem;font-weight:900;color:var(--gold)">
            منسّق الوكلاء
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;direction:rtl">
            وزّع مهمتك تلقائياً على وكلاء "🤖 وكلاء AI" المتخصصين، ثم احصل على إجابة موحّدة
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _AGENTS_HUB_OK or not _ORCHESTRATOR_OK:
        st.error("⚠️ تعذّر تحميل وحدات الوكلاء (ai/agent_categories.py أو ai/godmode.py).")
        return

    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl">اكتب مهمة أو سؤالاً مركّباً، وسيُحدَّد تلقائياً '
        'أنسب وكيل/وكلاء من تبويب "🤖 وكلاء AI" للإجابة عليه — بنفس ذاكرة محادثتهم الفعلية. '
        'يمكنك أيضاً اختيار الوكلاء يدوياً.</p>',
        unsafe_allow_html=True,
    )

    manual = st.multiselect(
        "اختر وكلاء يدوياً (اختياري — إن تُرك فارغاً يتم التوجيه التلقائي):",
        options=CATEGORY_ORDER,
        format_func=lambda k: f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}",
        key="orch_manual_agents",
    )

    task = st.text_area(
        "المهمة أو السؤال:",
        placeholder="مثال: راجع خطة إطلاق ميزة جديدة من ناحية الأتمتة والتحليل والمخاطر",
        key="orch_task_input",
        height=100,
    )

    synth = st.checkbox("🧩 وَلِّف الردود في إجابة واحدة موحّدة", value=True, key="orch_synth")

    exec_mode = st.radio(
        "نمط التنفيذ:",
        options=["parallel", "sequential"],
        format_func=lambda m: (
            "⚡ متوازٍ — كل وكيل يجيب على المهمة الأصلية بشكل مستقل"
            if m == "parallel" else
            "🔗 متسلسل — كل وكيل يبني على ردود الوكلاء السابقين (سير عمل أعمق)"
        ),
        index=0,
        key="orch_exec_mode",
        help=(
            "متوازٍ: أسرع، مناسب لمهام مستقلة (مثال: تحليل من زوايا مختلفة).\n"
            "متسلسل: كل وكيل يرى ردود من سبقه قبل أن يضيف رأيه — مناسب لسير "
            "عمل تراكمي (مثال: بحث ← تحليل ← توصية)."
        ),
    )

    if st.button("🚀 نفّذ عبر الوكلاء", type="primary", key="orch_run") and task.strip():
        if manual:
            selected, route_method = manual, "manual"
        else:
            selected, route_method, _route_scores = route_query_verbose(
                task.strip(), AGENT_CATEGORIES, max_agents=2
            )
        if not selected:
            st.warning("لم يتم تحديد أي وكيل مناسب تلقائياً. اختر وكلاء يدوياً من القائمة أعلاه.")
        else:
            mode_label = "🔗 متسلسل" if exec_mode == "sequential" else "⚡ متوازٍ"
            route_label = {
                "manual":  "🖐️ اختيار يدوي",
                "keyword": "🔤 مطابقة كلمات مفتاحية",
                "llm":     "🧠 توجيه دلالي عبر LLM",
                "default": "⚙️ افتراضي عام (لا تطابق واضح)",
            }.get(route_method, route_method)
            st.caption(
                f"نمط التنفيذ: {mode_label} — التوجيه: {route_label} — الوكلاء المُفعَّلون: "
                + "، ".join(
                    f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}" for k in selected
                )
            )
            responses: Dict[str, str] = {}
            failed_keys: set = set()
            final_answer: Optional[str] = None
            for key in selected:
                cat = AGENT_CATEGORIES[key]
                bot_key = f"agent_bot_{cat.key}"
                if bot_key not in st.session_state:
                    st.session_state[bot_key] = CategoryAgentChat(cat.key)
                bot = st.session_state[bot_key]

                # ── النمط المتسلسل: يُرفَق ملخّص ردود الوكلاء السابقين
                # بنص المهمة، بحيث يبني كل وكيل على ما سبقه (سير عمل حقيقي
                # بدل مجرد ردود متوازية منفصلة). النمط المتوازي يمرّر
                # المهمة الأصلية فقط لكل وكيل، بدون أي تعديل. ──
                if exec_mode == "sequential" and responses:
                    prior = "\n\n".join(
                        f"[{AGENT_CATEGORIES[k].title}]\n{v}"
                        for k, v in responses.items() if k not in failed_keys
                    )
                    agent_input = (
                        f"{task.strip()}\n\n"
                        f"── ردود وكلاء سابقين في نفس سير العمل (ابنِ عليها، لا تكررها) ──\n"
                        f"{prior}"
                    )
                else:
                    agent_input = task.strip()

                _orch_skel_ph = st.empty()
                with _orch_skel_ph.container():
                    st.caption(f"⟳ {cat.title} يعمل على المهمة...")
                    _skeleton(lines=3)
                # 🆕 إعادة محاولة واحدة عند فشل الاستدعاء الأول (فشل عابر:
                # مزوّد LLM بطيء، تحميل أول مرة، إلخ) — بنفس روح إعادة
                # المحاولة المضافة أصلاً للسرب الذكي (SwarmCoordinator).
                resp, _ok = None, False
                for _attempt in range(2):
                    try:
                        resp = bot.chat(agent_input, source="orchestrator")
                        _ok = True
                        break
                    except Exception as _orch_err:
                        resp = f"⚠️ خطأ: {_orch_err}"
                if not _ok:
                    failed_keys.add(key)
                _orch_skel_ph.empty()
                responses[key] = resp
                # 🆕 شارة جودة موحّدة لكل رد وكيل (نفس ميزة تبويب "🤖 وكلاء AI"
                # ووحدة إعادة التوليد التلقائي المدمجة الآن في CategoryAgentChat).
                _q_label = ""
                if _ok and hasattr(bot, "last_quality_badge"):
                    try:
                        _qb = bot.last_quality_badge()
                        _q_label = f" — {_qb}" if _qb else ""
                    except Exception:
                        _q_label = ""
                with st.expander(f"{cat.emoji} {cat.title}{_q_label}", expanded=not synth):
                    st.markdown(resp)
                    _copy_button(resp, key=f"orch_{key}")

            valid_responses = {k: v for k, v in responses.items() if k not in failed_keys}
            if synth and responses:
                if not valid_responses:
                    st.warning("⚠️ فشل كل الوكلاء المُفعَّلين — لا يوجد ما يُولَّف.")
                else:
                    combined_input = "\n\n".join(
                        f"[{AGENT_CATEGORIES[k].title}]\n{v}" for k, v in valid_responses.items()
                    )
                    _synth_skel_ph = st.empty()
                    with _synth_skel_ph.container():
                        st.caption("⟳ يجري توليف الإجابة النهائية...")
                        _skeleton(lines=4)
                    try:
                        from ai.llm_fallback import LLMFallback
                        _llm = LLMFallback()
                        _synth_result = _llm.generate(
                            query=f"السؤال الأصلي: {task.strip()}\n\nردود الوكلاء:\n{combined_input}",
                            system_prompt=COORDINATOR_SYSTEM_PROMPT,
                        )
                        final = _synth_result.text
                    except Exception as _synth_err:
                        final = f"⚠️ تعذّر التوليف: {_synth_err}"
                    final_answer = final
                    _synth_skel_ph.empty()
                    if failed_keys:
                        st.caption(
                            "⚠️ استُبعِد من التوليف: "
                            + "، ".join(AGENT_CATEGORIES[k].title for k in failed_keys)
                        )
                    st.toast("✅ تم توليف الإجابة الموحّدة", icon="✅")
                    _final_q_label = ""
                    try:
                        from ai.response_quality import score_response as _score_final
                        _fq = _score_final(final, query=task.strip())
                        _final_q_label = f" — 🔎 {_fq.as_percent()}٪ {_fq.label}"
                    except Exception:
                        pass  # تقييم إضافي وغير حرج — لا يمنع عرض الإجابة نفسها
                    st.markdown(
                        f'<div class="section-header">✅ الإجابة الموحّدة{_final_q_label}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(final)
                    _copy_button(final, key="orch_final")

            # ── تصدير النتيجة الكاملة (ردود كل الوكلاء + التوليف إن وُجد) ──
            # كان لا يوجد سوى زر نسخ لكل رد على حدة — أي فقدان للنتيجة عند
            # تحديث الصفحة، رغم أنها قد تكون نتاج عدة استدعاءات LLM.
            if responses:
                _orch_export_lines = [f"# نتيجة منسّق الوكلاء\n\n**المهمة:** {task.strip()}\n"]
                _orch_export_lines.append(f"**نمط التنفيذ:** {mode_label} · **التوجيه:** {route_label}\n")
                for _ek, _ev in responses.items():
                    _ecat = AGENT_CATEGORIES[_ek]
                    _estatus = " ⚠️ (فشل)" if _ek in failed_keys else ""
                    _orch_export_lines.append(f"## {_ecat.emoji} {_ecat.title}{_estatus}\n\n{_ev}\n")
                if final_answer:
                    _orch_export_lines.append(f"## ✅ الإجابة الموحّدة\n\n{final_answer}\n")
                st.download_button(
                    "⬇️ تصدير النتيجة الكاملة",
                    data="\n".join(_orch_export_lines).encode("utf-8"),
                    file_name="نتيجة_منسق_الوكلاء.md",
                    mime="text/markdown",
                    key="orch_export_full",
                )
