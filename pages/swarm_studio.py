"""
pages/swarm_studio.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة




# ══════════════════════════════════════════════════════════════════════════
# تبويب 🐝 السرب الذكي — AgentFactory + SwarmCoordinator (تنفيذ حقيقي)
# ══════════════════════════════════════════════════════════════════════════
def render_swarm_studio():
    """
    واجهة فعلية لنظام الوكلاء الوظيفي (ai/agent_factory.py +
    ai/swarm_coordinator.py): تفكيك هدف معقّد ديناميكياً عبر PlanningAgent
    حقيقي، ثم توزيعه على الأدوار المتخصصة (Research/Translation/Review/
    Planning/Monitor/Optimization/Coding) وتنفيذها فعلياً عبر محرك
    NSMAgent (نفس محرك تبويب 💬 المحادثة)، مع عرض حي لنتيجة كل مهمة.
    """
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <span style="font-size:2rem">🐝</span>
        <div style="font-size:1.5rem;font-weight:900;color:var(--gold)">
            السرب الذكي — Multi-Agent Swarm
        </div>
        <div style="color:var(--text-muted);font-size:0.85rem;direction:rtl">
            هدف واحد ← تفكيك تلقائي ← تنفيذ فعلي متوازٍ عبر عدة وكلاء متخصصين
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not _SWARM_OK:
        st.error("⚠️ تعذّر تحميل نظام السرب. تأكد من وجود ai/agent_factory.py و ai/swarm_coordinator.py.")
        return

    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl">اكتب هدفاً — بسيطاً أو معقداً — وسيُفكِّكه '
        '<b>PlanningAgent</b> حقيقياً إلى مهام فرعية، ثم يوزّعها <b>SwarmCoordinator</b> على '
        'الوكلاء المناسبين وينفذها فعلياً (وليس محاكاة) عبر نفس محرك المحادثة.</p>',
        unsafe_allow_html=True,
    )

    # ── mesh bundle حقيقي بمستوى العملية (وليس session_state فقط) — يربط
    # AgentFactory + SwarmCoordinator بنفس NodeRegistry/ScoringEngine/
    # MemoryEngine/NodeReputationEngine/SystemDNA المشتركة لكل الـmesh ──
    from core.mesh_bundle import get_mesh_bundle
    _mesh = get_mesh_bundle()
    factory = _mesh.agent_factory
    coordinator = _mesh.coordinator

    with st.expander("📋 الأدوار المتاحة في الكتالوج"):
        for role in AgentFactory.available_roles():
            spec = AGENT_CATALOGUE[role]
            st.markdown(
                f"**{role}** — {spec['description']}  \n"
                f"القدرات: `{', '.join(spec['capabilities'])}`"
            )

    goal = st.text_area(
        "🎯 الهدف:",
        placeholder="مثال: ابحث عن أحدث تطورات الذكاء الاصطناعي، لخّصها، وراجع جودة الملخص",
        key="swarm_goal_input",
        height=90,
    )
    extra_context = st.text_area(
        "📎 سياق/بيانات إضافية (اختياري — نص خام يُمرَّر لكل مهمة فرعية):",
        key="swarm_context_input",
        height=70,
    )
    use_planner = st.toggle(
        "🧠 تفكيك ديناميكي عبر PlanningAgent (إن أُطفئ: قواعد كلمات مفتاحية ثابتة فقط)",
        value=True,
        key="swarm_use_planner",
    )
    col_opt1, col_opt2 = st.columns(2)
    with col_opt1:
        retry_failed = st.toggle(
            "🔁 إعادة محاولة المهام الفاشلة تلقائياً (مرة واحدة، بوكيل جديد)",
            value=True,
            key="swarm_retry_failed",
        )
    with col_opt2:
        synthesize = st.toggle(
            "🧩 وَلِّف نتائج المهام في إجابة نهائية واحدة موحّدة",
            value=True,
            key="swarm_synthesize",
        )

    if st.button("🚀 نفّذ عبر السرب", type="primary", key="swarm_run") and goal.strip():
        data = {"content": extra_context.strip()} if extra_context.strip() else {}
        _swarm_skeleton_ph = st.empty()
        with _swarm_skeleton_ph.container():
            st.caption("⟳ السرب يعمل — تفكيك الهدف وتنفيذ المهام الفرعية...")
            _skeleton(kind="cards")
            _skeleton(lines=4)
        result = coordinator.execute(
            goal.strip(),
            data=data,
            use_planner=use_planner,
            retry_failed=retry_failed,
            synthesize=synthesize,
        )
        _swarm_skeleton_ph.empty()

        # 🆕 تغذية النتيجة الحقيقية إلى ScoringEngine + MemoryEngine +
        # NodeReputationEngine + SystemDNA (كانت هذه المحركات موجودة
        # ومكتوبة لكن غير مربوطة بأي نتيجة تنفيذ فعلية من قبل).
        try:
            _mesh.record_swarm_result(result)
        except Exception as _mesh_err:
            logger.warning(f"mesh_bundle.record_swarm_result failed: {_mesh_err}")

        status_emoji = {"done": "✅", "partial": "🟡", "failed": "❌"}.get(result.status, "❔")
        st.toast(
            f"{status_emoji} السرب انتهى: {result.success_count}/{len(result.tasks)} مهمة نجحت",
            icon=status_emoji,
        )
        st.markdown(
            f'<div class="section-header">{status_emoji} حالة السرب: {result.status} '
            f"({result.success_count}/{len(result.tasks)} مهمة نجحت)</div>",
            unsafe_allow_html=True,
        )

        for _ti, task in enumerate(result.tasks):
            icon = "✅" if task.status == "done" else ("❌" if task.status == "failed" else "⏳")
            _task_result_text = (task.result or {}).get("result_text", "")
            # 🆕 شارة جودة موحّدة لكل نتيجة مهمة (نفس ميزة تبويب "🤖 وكلاء AI"
            # ومنسّق الوكلاء) — تُحسب فقط للمهام الناجحة ذات نص نتيجة فعلي.
            _task_q_label = ""
            if task.status == "done" and _task_result_text:
                try:
                    from ai.response_quality import score_response as _score_task
                    _tq = _score_task(_task_result_text, query=task.sub_goal)
                    _task_q_label = f" — 🔎 {_tq.as_percent()}٪ {_tq.label}"
                except Exception:
                    pass  # تقييم إضافي وغير حرج — لا يمنع عرض نتيجة المهمة نفسها
            with st.expander(
                f"{icon} {task.sub_goal} — [{task.required_capability}] "
                f"({task.duration_ms or 0:.0f} ms){_task_q_label}",
                expanded=(task.status == "failed"),
            ):
                st.caption(f"الوكيل: {task.assigned_agent_id or '—'}")
                if _task_result_text:
                    st.markdown(_task_result_text)
                    _copy_button(_task_result_text, key=f"swarm_task_{_ti}")
                elif task.error:
                    st.warning(task.error)
                else:
                    st.caption("لا توجد نتيجة (لم يُسنَد وكيل لهذه المهمة).")

        _synthesis = (result.merged_output or {}).get("synthesis")
        if _synthesis:
            _synth_q_label = ""
            try:
                from ai.response_quality import score_response as _score_synth
                _sq = _score_synth(_synthesis, query=goal.strip())
                _synth_q_label = f" — 🔎 {_sq.as_percent()}٪ {_sq.label}"
            except Exception:
                pass  # تقييم إضافي وغير حرج — لا يمنع عرض التوليف نفسه
            st.markdown(
                f'<div class="section-header">✅ الإجابة الموحّدة{_synth_q_label}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(_synthesis)
            _copy_button(_synthesis, key="swarm_synthesis")
        elif synthesize:
            st.info("⚠️ لم يتم توليف إجابة موحّدة (لا توجد مهام ناجحة، أو تعذّر استدعاء LLM).")

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-header">📊 ملخص الوكلاء (AgentFactory)</div>',
                    unsafe_allow_html=True)
        _fs = factory.summary()
        st.markdown(f"""
        <div class="bento-grid">
            <div class="metric-card">
                <div class="metric-value">{_fs['total_agents']:,}</div>
                <div class="metric-label">إجمالي الوكلاء</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_fs['active_agents']:,}</div>
                <div class="metric-label">نشط الآن</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_fs['retired_agents']:,}</div>
                <div class="metric-label">متقاعد</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_fs['total_spawned']:,}</div>
                <div class="metric-label">إجمالي المُولَّد</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if _fs.get("role_distribution"):
            st.markdown(
                " ".join(
                    f'<span class="badge badge-blue">{role}: {count}</span>'
                    for role, count in _fs["role_distribution"].items()
                ),
                unsafe_allow_html=True,
            )
        with st.popover("🧹 تقليم الوكلاء ضعيفي الأداء"):
            st.caption(
                "يُقاعِد (retire) أي وكيل نفّذ 5 مهام على الأقل وكان "
                "متوسط أدائه أقل من الحد المحدَّد — لتفادي تكدّس وكلاء "
                "فاشلين تُختار من بينهم مهام مستقبلية عن طريق الخطأ."
            )
            prune_min_score = st.slider(
                "حد الأداء الأدنى", min_value=0.1, max_value=0.9, value=0.5,
                step=0.05, key="swarm_prune_min_score",
            )
            if st.button("🧹 نفّذ التقليم الآن", key="swarm_prune_btn"):
                retired_ids = factory.prune_underperformers(min_score=prune_min_score)
                if retired_ids:
                    st.success(f"تمت مقاعدة {len(retired_ids)} وكيل ضعيف الأداء.")
                else:
                    st.info("لا يوجد وكلاء تنطبق عليهم شروط التقليم حالياً.")
                st.rerun()
    with col_b:
        st.markdown('<div class="section-header">📊 ملخص السرب (SwarmCoordinator)</div>',
                    unsafe_allow_html=True)
        _cs = coordinator.summary()
        with st.popover("🕸️ حالة الـmesh الكاملة (Registry + Scoring + Memory + Reputation + DNA)"):
            st.caption(
                "هذه بيانات حقيقية من core.registry + ai.scoring_engine + "
                "ai.memory_engine + ai.reputation_engine + ai.system_dna، "
                "مبنية من نتائج تنفيذ السرب الفعلية أعلاه (وليست عرضاً منفصلاً)."
            )
            st.json(_mesh.summary())
        st.markdown(f"""
        <div class="bento-grid">
            <div class="metric-card">
                <div class="metric-value">{_cs['total_swarms']:,}</div>
                <div class="metric-label">إجمالي عمليات السرب</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['done']:,}</div>
                <div class="metric-label">✅ ناجحة بالكامل</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['partial']:,}</div>
                <div class="metric-label">🟡 نجاح جزئي</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['failed']:,}</div>
                <div class="metric-label">❌ فاشلة</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['active_agents']:,}</div>
                <div class="metric-label">وكلاء نشطون الآن</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{_cs['max_agents']:,}</div>
                <div class="metric-label">الحد الأقصى المسموح</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    hist = coordinator.history(limit=5)
    if hist:
        with st.expander("🕓 آخر 5 عمليات سرب"):
            for h in reversed(hist):
                st.markdown(f"**{h['goal']}** — {h['status']} ({h['success_count']}/{h['total_tasks']})")
