"""
ui_pages/agent_orchestrator.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة
from ai.agent_event_bus import emit_event
from ui_pages.agent_monitor import render_agent_live_trace


# ══════════════════════════════════════════════════════════════════════════
# تبويب 🤝 منسّق الوكلاء — توزيع مهمة واحدة على وكلاء Agents Hub الفعليين
# ══════════════════════════════════════════════════════════════════════════
def render_agent_orchestrator():
    """يوجّه مهمة/سؤال المستخدم تلقائياً إلى وكيل أو أكثر من وكلاء
    "🤖 وكلاء AI" الفعليين (نفس جلسات session_state وذاكرة المحادثة
    المستخدَمة في تبويب Agents Hub)، ثم يعرض ردودهم، مع توليف اختياري
    لإجابة موحّدة. يطبّق نمط Multi-Agent Systems: تفويض مهمة رئيسية إلى
    وكلاء متخصصين ثم تجميع نتائجهم عبر وكيل "منسّق"."""
    # 🆕 تبسيط: استُبدلت اللافتة الزخرفية (أيقونة كبيرة + عنوان بخط كبير)
    # بعنوان section-header القياسي — بنفس منهجية باقي صفحات NSM. الغرض من
    # الصفحة موضّح أصلاً في صندوق st.info الظاهر فوقها مباشرة من التبويب
    # الأب (مجموعة "🤖 الوكلاء")، فلا داعٍ لتكرار نفس المعنى بلافتة أكبر.
    st.markdown('<div class="section-header">🤝 منسّق الوكلاء</div>', unsafe_allow_html=True)
    st.caption('وزّع مهمتك تلقائياً على وكلاء "🤖 وكلاء AI" المتخصصين، ثم احصل على إجابة موحّدة')

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

    # 🆕 كان أقصى عدد وكلاء للتوجيه التلقائي مثبَّتاً في الكود (max_agents=2)،
    # فأي مهمة مركّبة تحتاج فعلياً 3+ وكلاء (مثال: بحث + تحليل + برمجة) كانت
    # تُقتَص تلقائياً لوكيلين فقط دون أي طريقة للمستخدم للتحكم بذلك. الآن
    # قابل للتعديل (1 إلى إجمالي عدد الفئات المتاحة)، ويُعطَّل تلقائياً عند
    # الاختيار اليدوي لأنه غير ذي صلة في تلك الحالة.
    max_agents = st.slider(
        "🎚️ أقصى عدد وكلاء للتوجيه التلقائي:",
        min_value=1,
        max_value=len(CATEGORY_ORDER),
        value=2,
        key="orch_max_agents",
        disabled=bool(manual),
        help="يُستخدَم فقط عند التوجيه التلقائي (بدون اختيار يدوي أعلاه). ارفعه للمهام المركّبة التي تحتاج أكثر من وكيلين.",
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
        _live_trace = st.empty()
        emit_event(
            "task_started",
            agent_id="agent_orchestrator",
            title="منسّق الوكلاء",
            status="running",
            detail="بدأ تنفيذ المهمة من الواجهة",
        )
        render_agent_live_trace(_live_trace)
        _route_scores: Dict[str, int] = {}
        if manual:
            selected, route_method = manual, "manual"
        else:
            selected, route_method, _route_scores = route_query_verbose(
                task.strip(), AGENT_CATEGORIES, max_agents=max_agents
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
            emit_event(
                "route_selected",
                agent_id="agent_orchestrator",
                title="منسّق الوكلاء",
                status="running",
                detail=f"{route_label} · تم اختيار {len(selected)} وكيل",
                metadata={"selected": list(selected), "route_method": route_method},
            )
            render_agent_live_trace(_live_trace)
            # 🆕 شفافية التوجيه: route_query_verbose كان أصلاً يُرجِع نقاط تطابق
            # الكلمات المفتاحية لكل فئة (_route_scores) لكن الواجهة لم تكن تعرضها
            # إطلاقاً — المستخدم لا يعرف لماذا اختير وكيل معيّن أو مدى قرب/بُعد
            # المنافسين. تُعرض هنا فقط عند التوجيه التلقائي (لا معنى لها يدوياً).
            if route_method in ("keyword", "llm", "default") and _route_scores:
                _sorted_scores = sorted(
                    _route_scores.items(), key=lambda kv: kv[1], reverse=True
                )
                with st.expander("🔎 تفصيل نقاط التوجيه", expanded=False):
                    for _sk, _sv in _sorted_scores:
                        _mark = "✅" if _sk in selected else "▫️"
                        st.caption(
                            f"{_mark} {AGENT_CATEGORIES[_sk].emoji} "
                            f"{AGENT_CATEGORIES[_sk].title} — {_sv} تطابق"
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

                emit_event(
                    "agent_started",
                    agent_id=cat.key,
                    title=cat.title,
                    status="running",
                    detail="بدأ الوكيل تنفيذ المهمة",
                )
                render_agent_live_trace(_live_trace)
                _orch_skel_ph = st.empty()
                with _orch_skel_ph.container():
                    st.caption(f"⟳ {cat.title} يعمل على المهمة...")
                    _skeleton(lines=3)
                # 🆕 نظام التقييم الذاتي (Self-Reflection): يراجع فشل الوكيل
                # ويصنّف سببه ويعيد المحاولة باستراتيجية مصححة تلقائياً حتى
                # استنفاد دورات التقييم — بنفس الروح المضافة في UnifiedAgentChat.
                from ai.agent_reflection import ReflectionContext, reflecting_call
                _orch_ref_ctx = ReflectionContext()
                resp, _ok = None, False
                try:
                    def _orch_call() -> str:
                        return bot.chat(agent_input, source="orchestrator")
                    def _orch_retry(_att: int, info: dict) -> None:
                        _strategy_note = {
                            "retry_with_backoff": "إعادة المحاولة بعد انتظار قصير",
                            "switch_provider_hint": "محاولة عبر مسار مزوّد بديل",
                            "simplify_prompt": "تبسيط الطلب وإعادة المحاولة",
                        }.get(info.get("strategy", ""), "إعادة المحاولة")
                        emit_event(
                            "agent_started",
                            agent_id=cat.key,
                            title=cat.title,
                            status="running",
                            detail=f"إعادة محاولة ذاتية: {_strategy_note}",
                        )
                        render_agent_live_trace(_live_trace)
                    resp = reflecting_call(cat.key, cat.title, _orch_call, _orch_ref_ctx, on_retry=_orch_retry)
                    _ok = True
                except Exception as _orch_err:
                    resp = f"⚠️ خطأ: {_orch_err}"
                if not _ok:
                    failed_keys.add(key)
                    emit_event("agent_error", agent_id=cat.key, title=cat.title, status="error", detail=str(resp)[:180])
                else:
                    emit_event("agent_done", agent_id=cat.key, title=cat.title, status="done", detail="اكتمل رد الوكيل")
                _orch_skel_ph.empty()
                responses[key] = resp
                render_agent_live_trace(_live_trace)
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
                    emit_event(
                        "synthesis_started",
                        agent_id="agent_orchestrator",
                        title="منسّق الوكلاء",
                        status="running",
                        detail="يولّف الردود في إجابة واحدة",
                    )
                    render_agent_live_trace(_live_trace)
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
                    emit_event(
                        "synthesis_done",
                        agent_id="agent_orchestrator",
                        title="منسّق الوكلاء",
                        status="done",
                        detail="اكتمل التوليف النهائي",
                    )
                    emit_event(
                        "task_done",
                        agent_id="agent_orchestrator",
                        title="منسّق الوكلاء",
                        status="done",
                        detail="اكتملت المهمة من الواجهة",
                    )
                    render_agent_live_trace(_live_trace)
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
