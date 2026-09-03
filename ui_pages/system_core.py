"""
ui_pages/system_core.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة

import os
from pathlib import Path

from ui_components import (
    render_alert_cards,
    render_health_cards,
    render_provider_cards,
    render_section_header,
    render_system_events,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _process_memory_mb():
    """قراءة استهلاك العملية الحالية دون اعتماد إضافي أو اتصال خارجي."""
    try:
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return round(float(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _provider_status():
    """يعرض وجود إعدادات المزود فقط؛ لا يطبع مفاتيح ولا يختبر الشبكة."""
    providers = [
        ("OpenRouter", "OPENROUTER_API_KEY", "مسار النماذج المتعددة"),
        ("OpenAI-compatible", "OPENAI_API_KEY", "مسار الذكاء الاصطناعي الأساسي"),
        ("Kaggle Scheduler", "NSM_KAGGLE_ACCOUNTS_JSON", "جدولة التدريب السحابي"),
        ("Discord Alerts", "DISCORD_BOT_TOKEN", "التنبيهات الاجتماعية"),
    ]
    return [
        {
            "name": name,
            "ready": bool(os.environ.get(env_name)),
            "detail": detail if os.environ.get(env_name) else "لا يوجد إعداد في البيئة",
        }
        for name, env_name, detail in providers
    ]


def _render_system_center() -> None:
    """لوحة قراءة فقط تجمع صحة NSM والتنبيهات دون تغيير أي مسار تشغيل."""
    try:
        snapshot = system_snapshot()
    except Exception as exc:
        snapshot = {
            "score": 0.0,
            "path_ok": 0,
            "path_total": 0,
            "sections": {},
            "error": str(exc),
        }

    sections = snapshot.get("sections") or {}
    score = float(snapshot.get("score") or 0.0)
    score_tone = "ok" if score >= .75 else "warn" if score >= .5 else "bad"

    try:
        from ai.background_tasks import get_background_task_manager
        background = get_background_task_manager().status()
    except Exception:
        background = {"total": 0, "pending": 0, "running": 0, "done": 0, "failed": 0}

    try:
        from ai.training_alerts import alerts_summary, list_alerts
        alert_summary = alerts_summary()
        alerts = list_alerts(limit=6)
    except Exception:
        alert_summary = {"total": 0, "by_severity": {}, "discord_enabled": False}
        alerts = []

    try:
        from ai.agent_event_bus import get_events
        events = get_events(limit=40)
    except Exception:
        events = []

    memory_mb = _process_memory_mb()
    path_ok = int(snapshot.get("path_ok") or 0)
    path_total = int(snapshot.get("path_total") or 0)
    failed_tasks = int(background.get("failed") or 0)
    active_tasks = int(background.get("running") or 0) + int(background.get("pending") or 0)
    critical_alerts = int((alert_summary.get("by_severity") or {}).get("critical", 0))
    warning_alerts = int((alert_summary.get("by_severity") or {}).get("warning", 0))

    st.markdown(
        '<div class="section-header">📡 مركز الحالة والإشعارات</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl;margin-top:-.35rem">'
        'لوحة قراءة موحّدة تجمع صحة المكوّنات، الذاكرة، المهام الخلفية، مزودي الذكاء الاصطناعي، وآخر التنبيهات دون تغيير مسارات التشغيل.'
        '</p>',
        unsafe_allow_html=True,
    )
    refresh_col, note_col = st.columns([1, 4])
    with refresh_col:
        if st.button("↻ تحديث الحالة", key="system_center_refresh"):
            st.rerun()
    with note_col:
        st.caption("البيانات تُقرأ من الحالة الحالية للعملية، ولا تُرسل أسراراً أو تُجري طلبات شبكة تلقائية.")

    render_health_cards([
        {
            "icon": "🫀",
            "label": "الصحة الكلية",
            "value": f"{score:.0%}",
            "note": f"{path_ok}/{path_total} مكوّنات حرجة متاحة",
            "tone": score_tone,
        },
        {
            "icon": "🧠",
            "label": "ذاكرة العملية",
            "value": f"{memory_mb:.1f} MB" if memory_mb is not None else "غير متاح",
            "note": "قياس RSS محلي للعملية الحالية",
            "tone": "info",
        },
        {
            "icon": "⚙️",
            "label": "المهام الخلفية",
            "value": str(active_tasks),
            "note": f"نشطة أو معلّقة · {failed_tasks} فاشلة",
            "tone": "warn" if failed_tasks else "ok",
        },
        {
            "icon": "🔔",
            "label": "التنبيهات",
            "value": str(alert_summary.get("total", 0)),
            "note": f"{critical_alerts} حرجة · {warning_alerts} تحذير",
            "tone": "bad" if critical_alerts else "warn" if warning_alerts else "ok",
        },
    ])

    left, right = st.columns([1.12, .88])
    with left:
        st.markdown(
            '<div class="nsm-system-panel-title"><span>🔔 آخر التنبيهات</span>'
            f'<span class="badge badge-{"red" if critical_alerts else "green"}">{len(alerts)} معروض</span></div>',
            unsafe_allow_html=True,
        )
        if alerts:
            render_alert_cards(alerts, limit=6)
        else:
            st.markdown('<div class="nsm-empty-state">لا توجد تنبيهات مسجلة حالياً.</div>', unsafe_allow_html=True)

        st.markdown('<div class="nsm-system-panel-title" style="margin-top:1rem"><span>🛰️ آخر أحداث النظام</span></div>', unsafe_allow_html=True)
        render_system_events(events, limit=8)

    with right:
        st.markdown('<div class="nsm-system-panel-title"><span>🔌 مزودو الذكاء الاصطناعي</span></div>', unsafe_allow_html=True)
        render_provider_cards(_provider_status())

        training = sections.get("training") or {}
        bridge = sections.get("bridge") or {}
        moe = sections.get("moe") or {}
        st.markdown('<div class="nsm-system-panel-title" style="margin-top:1rem"><span>🧩 ملخص الوحدات</span></div>', unsafe_allow_html=True)
        render_health_cards([
            {"icon": "🧬", "label": "MoE", "value": "جاهز" if moe.get("ok") else "تنبيه", "note": str(moe.get("detail", "لا توجد بيانات"))[:70], "tone": "ok" if moe.get("ok") else "warn"},
            {"icon": "🎓", "label": "التدريب", "value": "متاح" if training.get("ok") else "تنبيه", "note": str(training.get("detail", "لا توجد بيانات"))[:70], "tone": "ok" if training.get("ok") else "warn"},
            {"icon": "🌉", "label": "جسر الوكلاء", "value": "متصل" if bridge.get("ok") else "جزئي", "note": str(bridge.get("detail", "لا توجد بيانات"))[:70], "tone": "ok" if bridge.get("ok") else "warn"},
            {"icon": "📦", "label": "المشروع", "value": str((sections.get("project") or {}).get("tests", "—")), "note": "اختبارات مرصودة في المشروع", "tone": "info"},
        ])


# ══════════════════════════════════════════════════════════════════════════
# تبويب ⚙️ النظام الداخلي — النواة العصبية + الوعي الذاتي + مخطط الأهداف
# ══════════════════════════════════════════════════════════════════════════
def render_system_core():
    """ربط الوحدات الداخلية الأساسية بالواجهة."""
    _render_system_center()
    st.markdown('<div class="section-header">⚙️ النظام الداخلي — Neural Core & Intelligence</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<p style="color:var(--text-muted);direction:rtl">هذا التبويب يعرض الوحدات الداخلية للنظام: '
        'النواة العصبية، الوعي الذاتي، مخطط الأهداف، والمفكر الفوقي.</p>',
        unsafe_allow_html=True,
    )

    core_tabs = st.tabs([
        "🧠 النواة العصبية",
        "👁️ الوعي الذاتي",
        "🎯 مخطط الأهداف",
        "🔬 التحليل اللغوي",
        "🌐 بحث الويب المباشر",
        "🌍 التغذية من العالم",
        "⚡ الأداء (كاش الأسئلة)",
    ])

    # ══════════════════ 1. النواة العصبية ══════════════════
    with core_tabs[0]:
        st.markdown('<div class="section-header">🧠 النواة العصبية (Neural Core)</div>',
                    unsafe_allow_html=True)
        if not _NEURAL_CORE_OK:
            st.error("⚠️ تعذّر تحميل NeuralCore — تأكد من تثبيت numpy.")
        else:
            try:
                # ── النواة الحية المشتركة (نفس singleton الذي يستخدمه ──
                # ReasoningPipeline فعلياً في مسار الاستدلال الحي، بنفس
                # مسار الحفظ models/neural_core. أي تدريب هنا يُحدِّث
                # نفس الكائن الحي بالذاكرة، ونفس الملف عند الحفظ.
                from ai.neural_core import get_default_core, DEFAULT_INPUT_DIM, \
                    DEFAULT_HIDDEN_DIMS, DEFAULT_OUTPUT_DIM
                _nc_path = "models/neural_core"
                _nc = get_default_core(
                    _nc_path,
                    input_dim=DEFAULT_INPUT_DIM,
                    hidden_dims=list(DEFAULT_HIDDEN_DIMS),
                    output_dim=DEFAULT_OUTPUT_DIM,
                )
                _nc_info = _nc.get_info()

                if os.path.exists(os.path.join(_nc_path, "network.json")):
                    st.caption(f"📂 النواة الحية — مُحمَّلة من `{_nc_path}` (نفس النواة التي يستخدمها الاستدلال الحقيقي)")
                else:
                    st.caption("🆕 نواة جديدة (لا يوجد ملف محفوظ بعد) — L1 المدروسة 784×784 محمّلة تلقائياً")

                col_nc1, col_nc2, col_nc3, col_nc4 = st.columns(4)
                with col_nc1:
                    metric_card(_nc_info.get("total_parameters", "—"), "إجمالي المعاملات")
                with col_nc2:
                    metric_card(_nc_info.get("train_steps", 0), "خطوات التدريب")
                with col_nc3:
                    metric_card(len(_nc_info.get("architecture", [])), "عدد الطبقات")
                with col_nc4:
                    mem_size = _nc_info.get("memory_size", 0)
                    metric_card(mem_size, "حجم الذاكرة الترابطية")

                st.markdown("")
                col_left, col_right = st.columns(2)
                with col_left:
                    st.markdown("**معمارية الشبكة:**")
                    arch = _nc_info.get("architecture", [])
                    for i, layer in enumerate(arch):
                        st.markdown(f"""
                        <div class="root-item">
                            <span class="badge badge-blue">طبقة {i+1}</span>
                            &nbsp;{layer.get('type','—')} &nbsp;
                            <span class="badge badge-purple">{layer.get('input_dim','?')} → {layer.get('output_dim','?')}</span>
                            &nbsp;<small>{layer.get('activation','')}</small>
                        </div>
                        """, unsafe_allow_html=True)

                with col_right:
                    st.markdown("**حالة النواة:**")
                    last_loss = _nc_info.get("last_loss")
                    best_loss = _nc_info.get("best_loss")
                    lr        = _nc_info.get("learning_rate", 0.01)
                    st.markdown(f"""
                    <div class="root-item">
                        <strong>معدل التعلم:</strong> {lr}<br>
                        <strong>آخر خسارة:</strong> {f"{last_loss:.6f}" if last_loss else "لا يوجد"}<br>
                        <strong>أفضل خسارة:</strong> {f"{best_loss:.6f}" if best_loss else "لا يوجد"}
                    </div>
                    """, unsafe_allow_html=True)

                # اختبار تمرير أمامي
                st.markdown("")
                st.markdown("**اختبار التمرير الأمامي:**")
                import numpy as np
                _test_input = np.random.randn(784)
                _output = _nc.forward(_test_input)
                _out_str = "، ".join(f"{v:.4f}" for v in _output)
                st.code(f"مدخل: متجه عشوائي (784 بُعد)\nمخرج (4 فئات): [{_out_str}]", language="text")
                st.success("✅ النواة العصبية تعمل بشكل صحيح")

                # ── تدريب فعلي من التجارب الحقيقية (بدون تخزين بيانات خام) ──
                st.markdown("---")
                st.markdown("**🎓 تدريب من التجارب الحقيقية (Experience Replay)**")
                st.caption(
                    "يتدرّب على حلقات حقيقية من استخدام النظام الفعلي "
                    "(memory/experience.db) عبر train_step() + evolve_if_plateau() — "
                    "تحديث أوزان ونمو هيكلي فعلي، **بدون** تخزين أي متجهات خام "
                    "بالذاكرة الترابطية."
                )
                _replay_strategy = st.selectbox(
                    "استراتيجية الاختيار:",
                    ["الأحدث (recent)", "الأعلى جودة (top)", "متنوعة (diverse)"],
                    key="nc_replay_strategy",
                )
                if st.button("🎓 ابدأ التدريب الآن", key="nc_train_btn"):
                    try:
                        from ai.experience_trainer import ExperienceTrainer
                        from ai.experience_store import EpisodeStore
                        _params_before = _nc_info.get("total_parameters", 0)
                        _store = EpisodeStore()
                        _trainer = ExperienceTrainer(core=_nc, store=_store)
                        if _replay_strategy.startswith("الأعلى"):
                            _report = _trainer.replay_top(limit=20)
                        elif _replay_strategy.startswith("متنوعة"):
                            _report = _trainer.replay_diverse(limit=20)
                        else:
                            _report = _trainer.replay_recent(limit=20)

                        if _report.episodes_used == 0:
                            st.warning(
                                "⚠️ لا توجد تجارب حقيقية محفوظة بعد (0 حلقة) في "
                                "memory/experience.db — النواة تتعلم تلقائياً من "
                                "الاستخدام الحقيقي للنظام (أسئلة حقيقية عبر "
                                "ReasoningPipeline)، لا يوجد بعد ما تتدرّب عليه."
                            )
                        else:
                            _params_after = _nc.get_info().get("total_parameters", 0)
                            _grew = _params_after > _params_before
                            st.success(
                                f"✅ تدرّبت على {_report.episodes_used} حلقة حقيقية — "
                                f"الخسارة: {_report.avg_loss_before:.6f} → {_report.avg_loss_after:.6f}"
                            )
                            if _grew:
                                st.info(
                                    f"📈 النواة توسّعت فعلياً: {_params_before:,} → "
                                    f"{_params_after:,} معامل (نمو هيكلي بسبب ركود الخسارة)"
                                )
                    except Exception as _train_err:
                        st.error(f"فشل التدريب: {_train_err}")

                st.markdown("")
                if st.button("💾 حفظ الأوزان فقط (بدون بيانات خام)", key="nc_save_ckpt"):
                    try:
                        _saved_path = None
                        try:
                            from ai.rollback_guard import CheckpointGuard
                            _guard = CheckpointGuard(asset="neural_core_weights")
                            _last_loss = _nc.get_info().get("last_loss")
                            _guard_files = [
                                f"{_nc_path}/network.json",
                                f"{_nc_path}/core_state.json",
                            ]

                            def _do_nc_save():
                                nonlocal _saved_path
                                _saved_path = _nc.save(_nc_path, include_memory=False)
                                return _saved_path

                            if _last_loss is not None:
                                _decision = _guard.guarded_update(
                                    files=_guard_files,
                                    update_fn=_do_nc_save,
                                    eval_fn=lambda: -float(_last_loss),
                                    tolerance=-0.05,
                                    label=f"حفظ يدوي (last_loss={_last_loss:.6f})",
                                )
                                if _decision.rolled_back:
                                    st.error(
                                        f"⚠️ رُفض الحفظ تلقائياً وأُعيدت الأوزان السابقة "
                                        f"(محمي بـ RollbackGuard) — جودة الحفظ الجديد "
                                        f"({-_decision.new_score:.6f} خسارة) أسوأ من "
                                        f"المحفوظة سابقاً ({-_decision.old_score:.6f} خسارة)."
                                    )
                                else:
                                    st.success(
                                        f"✅ تم حفظ الأوزان بأمان (محمي من التراجع) → "
                                        f"`{_saved_path}`"
                                    )
                            else:
                                # لا يوجد last_loss بعد (لم يُدرَّب النموذج في هذه
                                # الجلسة) — لا أساس مقارنة، فنأخذ لقطة احتياطية
                                # يدوية فقط قبل الحفظ العادي تحسباً لأي مشكلة.
                                _guard.snapshot(_guard_files, label="حفظ يدوي (بدون last_loss)")
                                _do_nc_save()
                                st.success(
                                    f"✅ تم حفظ الأوزان والحالة الهيكلية فقط → "
                                    f"`{_saved_path}` (أُخذت لقطة احتياطية أولاً)"
                                )
                        except ImportError:
                            # rollback_guard غير متاح لأي سبب — احفظ عادياً كما
                            # كان يحدث قبل هذا الربط، بدون حماية.
                            _saved_path = _nc.save(_nc_path, include_memory=False)
                            st.success(f"✅ تم حفظ الأوزان والحالة الهيكلية فقط → `{_saved_path}`")
                    except Exception as _save_err:
                        st.error(f"فشل الحفظ: {_save_err}")

            except Exception as _nc_err:
                st.error(f"خطأ في NeuralCore: {_nc_err}")

    # ══════════════════ 2. الوعي الذاتي ══════════════════
    with core_tabs[1]:
        st.markdown('<div class="section-header">👁️ الوعي الذاتي (Self-Awareness Engine)</div>',
                    unsafe_allow_html=True)
        if not _SELF_AWARE_OK:
            st.error("⚠️ تعذّر تحميل SelfAwarenessEngine.")
        else:
            try:
                _ckg   = load_ckg()
                _roots = load_arabic_roots()
                _ep    = get_episodic_stats()
                _ckpt  = load_latest_checkpoint()

                _sa_engine = SelfAwarenessEngine()
                _report    = _sa_engine.introspect()
                _rd = _report.to_dict()
                # إثراء التقرير ببيانات CKG المحلية
                if _rd.get("node_count", 0) == 0:
                    _rd["node_count"] = len(_ckg.get("concepts", {}))
                if _rd.get("edge_count", 0) == 0:
                    _rd["edge_count"] = len(_ckg.get("relations", {}))

                # مقاييس رئيسية
                score = _rd.get("system_health_score", 0.0)
                readiness = _rd.get("phase7_readiness", 0.0)
                col_sa1, col_sa2, col_sa3 = st.columns(3)
                with col_sa1:
                    metric_card(f"{score:.0%}", "درجة صحة النظام")
                with col_sa2:
                    metric_card(f"{readiness:.0%}", "جاهزية Phase 7")
                with col_sa3:
                    metric_card(_rd.get("node_count", 0), "عدد العقد (المفاهيم)")

                st.markdown("")

                # الأهداف الحالية
                objectives = _rd.get("current_objectives", [])
                if objectives:
                    st.markdown('<div class="section-header">🎯 الأهداف الحالية</div>',
                                unsafe_allow_html=True)
                    for obj in objectives:
                        st.markdown(f"""
                        <div class="root-item">
                            <span style="font-size:1.1rem">🎯</span> {obj}
                        </div>
                        """, unsafe_allow_html=True)

                # القدرات المعروفة
                capabilities = _rd.get("known_capabilities", [])
                if capabilities:
                    st.markdown('<div class="section-header">✅ القدرات المعروفة</div>',
                                unsafe_allow_html=True)
                    caps_html = " ".join(
                        f'<span class="badge badge-green" style="margin:3px;font-size:0.85rem">{c}</span>'
                        for c in capabilities
                    )
                    st.markdown(caps_html, unsafe_allow_html=True)

                # الرؤى والتوصيات
                insights = _rd.get("insights", [])
                if insights:
                    st.markdown('<div class="section-header">💡 رؤى النظام</div>',
                                unsafe_allow_html=True)
                    for ins in insights:
                        st.info(ins)

                # شريط الصحة
                st.markdown("")
                st.markdown(f"**درجة الصحة الكلية:** {score:.0%}")
                st.progress(score)
                st.markdown(f"**جاهزية Phase 7:** {readiness:.0%}")
                st.progress(readiness)

            except Exception as _sa_err:
                st.error(f"خطأ في Awareness Engine: {_sa_err}")

            # ── اللغة الذاتية (SelfNarrative) ────────────────────────────
            st.markdown("")
            st.markdown('<div class="section-header">📖 اللغة الذاتية (Self-Narrative)</div>',
                        unsafe_allow_html=True)
            if not _SELF_NARRATIVE_OK:
                st.caption("⚠️ تعذّر تحميل SelfNarrative.")
            else:
                _sn_engine = _get_self_narrative()
                if _sn_engine is not None:
                    st.markdown(f"*{_sn_engine.get_identity_statement()}*")
                    _sn_log = _sn_engine.get_narrative_log(8)
                    if _sn_log:
                        for _ne in _sn_log:
                            st.markdown(
                                f"<div class='root-item' style='font-size:0.85rem'>"
                                f"{_ne['summary']}"
                                f"</div>", unsafe_allow_html=True,
                            )
                    else:
                        st.caption("لا أحداث مسجلة بعد — تحدّث مع NSM أو شغّل التغذية من العالم.")

    # ══════════════════ 3. مخطط الأهداف ══════════════════
    with core_tabs[2]:
        st.markdown('<div class="section-header">🎯 مخطط الأهداف (Goal Planner)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">حدّد هدفاً بالعربية وسيبني النظام خطة تنفيذ تلقائية.</p>',
            unsafe_allow_html=True,
        )

        if not _GOAL_PLANNER_OK:
            st.error("⚠️ تعذّر تحميل GoalPlanner.")
        else:
            _gp_goal = st.text_input(
                "اكتب هدفك:",
                value=st.session_state.get("gp_goal", ""),
                placeholder="مثال: تلخيص مفاهيم سورة البقرة",
                key="gp_goal_input",
            )
            st.session_state["gp_goal"] = _gp_goal

            _gp_run = st.button("🎯 بناء خطة التنفيذ", type="primary", key="gp_run")

            if _gp_run and _gp_goal.strip():
                with st.spinner("⟳ يبني النظام خطة التنفيذ..."):
                    try:
                        _planner = GoalPlanner()
                        _plan = _planner.plan(_gp_goal.strip())
                        if _plan is None:
                            st.warning("لم يُمكن بناء خطة لهذا الهدف — لا توجد عقد كافية في السجل.")
                        else:
                            _plan_d = _plan.to_dict()

                            st.markdown('<div class="section-header">📋 خطة التنفيذ</div>',
                                        unsafe_allow_html=True)

                            _p_cols = st.columns(3)
                            with _p_cols[0]:
                                metric_card(f"{_plan_d.get('confidence', 0):.0%}", "درجة الثقة")
                            with _p_cols[1]:
                                metric_card(len(_plan_d.get("path", [])), "عدد الخطوات")
                            with _p_cols[2]:
                                metric_card(_plan_d.get("status", "—"), "الحالة")

                            _path = _plan_d.get("path", [])
                            if _path:
                                st.markdown("")
                                st.markdown("**مسار التنفيذ:**")
                                for _step_i, _step in enumerate(_path):
                                    st.markdown(f"""
                                    <div class="root-item">
                                        <span class="badge badge-blue">خطوة {_step_i+1}</span>
                                        &nbsp;<strong>{_step}</strong>
                                    </div>
                                    """, unsafe_allow_html=True)

                            _reasoning = _plan_d.get("reasoning", [])
                            if _reasoning:
                                with st.expander("🔍 تفاصيل المنطق"):
                                    for _r in _reasoning:
                                        st.markdown(f"- {_r}")

                    except Exception as _gp_err:
                        st.error(f"خطأ في GoalPlanner: {_gp_err}")

    # ══════════════════ 4. التحليل اللغوي ══════════════════
    with core_tabs[3]:
        st.markdown('<div class="section-header">🔬 محرك اللغة العربية (ArabicNLP)</div>',
                    unsafe_allow_html=True)
        if not _ARABIC_NLP_OK:
            st.error("⚠️ تعذّر تحميل ArabicNLPEngine.")
        else:
            _nlp_input = st.text_area(
                "أدخل نصاً عربياً للتحليل:",
                placeholder="مثال: الصبر مفتاح الفرج، والإيمان نور يهدي القلوب إلى الحق.",
                height=100,
                key="nlp_core_input",
            )
            _nlp_run = st.button("🔬 حلّل النص", type="primary", key="nlp_core_run")

            if _nlp_run and _nlp_input.strip():
                with st.spinner("⟳ يحلل النص..."):
                    try:
                        _nlp_e  = get_arabic_engine(ckg=load_ckg())
                        _res    = _nlp_e.analyse(_nlp_input.strip())
                        _fv     = _res.feature_vector

                        st.markdown("**متجه الخصائص (Feature Vector):**")
                        _fv_col1, _fv_col2, _fv_col3, _fv_col4 = st.columns(4)
                        with _fv_col1:
                            st.metric("نسبة الأفعال", f"{_fv.verb_score:.0%}")
                            st.metric("نسبة الأسماء", f"{_fv.noun_score:.0%}")
                        with _fv_col2:
                            st.metric("تعقيد الجذور", f"{_fv.root_complexity:.0%}")
                            st.metric("أنماط الصرف", f"{_fv.morpho_pattern_score:.0%}")
                        with _fv_col3:
                            st.metric("الكثافة الدلالية", f"{_fv.semantic_concept_score:.0%}")
                            st.metric("تماسك السياق", f"{_fv.context_score:.0%}")
                        with _fv_col4:
                            st.metric("التعقيد النحوي", f"{_fv.syntactic_complexity:.0%}")
                            st.metric("طول المتجه", len(_fv.to_list()))

                        st.markdown("")

                        # الطبقة النحوية
                        _syn = _res.syntactic
                        if _syn.tokens:
                            st.markdown('<div class="section-header">📝 الطبقة النحوية</div>',
                                        unsafe_allow_html=True)
                            _tok_html = " ".join(
                                f'<span class="badge badge-{"blue" if t.is_verb else "purple" if t.is_noun else "amber"}" style="margin:3px;padding:4px 10px;font-size:0.9rem" title="{"فعل" if t.is_verb else "اسم" if t.is_noun else "أداة"}">{t.surface}</span>'
                                for t in _syn.tokens[:30]
                            )
                            st.markdown(_tok_html, unsafe_allow_html=True)
                            st.caption("🔵 فعل | 🟣 اسم | 🟡 أداة/حرف")

                        # الطبقة الصرفية
                        _morph = _res.morphological
                        if _morph.roots_found:
                            st.markdown('<div class="section-header">🌿 الطبقة الصرفية</div>',
                                        unsafe_allow_html=True)
                            _roots_html = " ".join(
                                f'<span class="badge badge-green" style="margin:3px">√ {r}</span>'
                                for r in _morph.roots_found[:15]
                            )
                            st.markdown(_roots_html, unsafe_allow_html=True)

                        # الطبقة الدلالية
                        _sem = _res.semantic
                        if hasattr(_sem, "concepts_found") and _sem.concepts_found:
                            st.markdown('<div class="section-header">💡 المفاهيم الدلالية</div>',
                                        unsafe_allow_html=True)
                            _con_html = " ".join(
                                f'<span class="badge badge-purple" style="margin:3px">{c}</span>'
                                for c in _sem.concepts_found[:15]
                            )
                            st.markdown(_con_html, unsafe_allow_html=True)

                    except Exception as _nlp_err2:
                        st.error(f"خطأ في التحليل: {_nlp_err2}")

    # ══════════════════ 5. بحث الويب المباشر ══════════════════
    with core_tabs[4]:
        st.markdown('<div class="section-header">🌐 بحث الويب الحقيقي (DuckDuckGo)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">بحث حقيقي في الإنترنت بدون مفتاح API — '
            'يستخدم DuckDuckGo ويُرجع نتائج فعلية.</p>',
            unsafe_allow_html=True,
        )

        if not _WEB_SEARCH_OK:
            st.error("⚠️ تعذّر تحميل web_search_tool.")
        else:
            _ws_direct_q = st.text_input(
                "ابحث في الإنترنت:",
                placeholder="مثال: أحدث نماذج الذكاء الاصطناعي 2026، أو: ما هو الإسلام؟",
                key="ws_direct_input",
            )
            _ws_direct_n = st.slider("عدد النتائج", 3, 10, 5, key="ws_direct_n")
            _ws_direct_btn = st.button("🔍 ابحث الآن", type="primary", key="ws_direct_btn",
                                        use_container_width=True)

            if _ws_direct_btn and _ws_direct_q.strip():
                with st.spinner("⟳ يبحث في الإنترنت..."):
                    _ws_out = _web_search(_ws_direct_q.strip(), max_results=_ws_direct_n)

                st.markdown('<div class="section-header">📋 النتائج</div>', unsafe_allow_html=True)
                st.markdown(f"""
                <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                            padding:1.2rem 1.5rem;direction:rtl;line-height:2.0;
                            white-space:pre-wrap;font-size:0.95rem;border:1px solid var(--border)">
                {_ws_out}
                </div>
                """, unsafe_allow_html=True)

                st.download_button(
                    "⬇️ تحميل النتائج",
                    data=_ws_out,
                    file_name="web_search_results.txt",
                    mime="text/plain",
                    key="ws_download",
                )

    # ══════════════════ 6. التغذية من العالم الخارجي ══════════════════
    with core_tabs[5]:
        st.markdown('<div class="section-header">🌍 التغذية من العالم (World Feed)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted);direction:rtl">يسحب بيانات حقيقية من مصادر '
            'عامة بلا مفتاح API (arXiv، Hacker News، TechCrunch)، يمررها عبر الجهاز المناعي '
            'وموتور الجودة، ولا يقبل إلا ما يجتاز الفحصين — العناصر المقبولة تُسجَّل كذاكرة '
            'إيبيسودية حقيقية.</p>',
            unsafe_allow_html=True,
        )

        if not _WORLD_FEED_OK:
            st.error("⚠️ تعذّر تحميل WorldFeed/QualityEngine/ImmuneSystem.")
        else:
            _wf = _get_world_feed()
            if _wf is None:
                st.error("⚠️ تعذّر تهيئة WorldFeed.")
            else:
                _wf_stats = _wf.get_feed_stats()

                col_wf1, col_wf2, col_wf3, col_wf4 = st.columns(4)
                with col_wf1:
                    metric_card(_wf_stats["total_fetched"], "إجمالي المسحوب")
                with col_wf2:
                    metric_card(_wf_stats["total_accepted"], "مقبول (جودة+مناعة)")
                with col_wf3:
                    metric_card(_wf_stats["total_rejected"], "مرفوض (جودة منخفضة)")
                with col_wf4:
                    metric_card(_wf_stats["total_blocked"], "محجوب (مناعة)")

                st.markdown("")
                _wf_running = _wf_stats["running"]
                _wf_status_badge = "🟢 يعمل بالخلفية" if _wf_running else "⚪ متوقف"
                st.markdown(f"**الحالة:** {_wf_status_badge}  ·  **الدورات:** {_wf_stats['cycles']}")

                col_wfb1, col_wfb2, col_wfb3 = st.columns(3)
                with col_wfb1:
                    if not _wf_running:
                        if st.button("▶️ ابدأ الاستطلاع التلقائي", key="wf_start",
                                      use_container_width=True):
                            _wf.start(interval_s=300.0)
                            st.rerun()
                    else:
                        if st.button("⏹️ أوقف", key="wf_stop", use_container_width=True):
                            _wf.stop()
                            st.rerun()
                with col_wfb2:
                    if st.button("🔄 اسحب الآن يدوياً", key="wf_poll_once",
                                  use_container_width=True):
                        with st.spinner("⟳ يسحب من المصادر الحقيقية..."):
                            _wf.poll_once()
                        st.rerun()
                with col_wfb3:
                    st.caption(f"المصادر النشطة: {len(_wf_stats['sources'])}")

                st.markdown('<div class="section-header">📡 المصادر</div>', unsafe_allow_html=True)
                for _src in _wf_stats["sources"]:
                    _src_icon = "✅" if _src["enabled"] else "⛔"
                    st.markdown(
                        f"{_src_icon} **{_src['name']}** — "
                        f"نجح: {_src['fetch_count']} · فشل: {_src['error_count']}"
                    )

                _wf_recent = _wf.get_recent(10)
                if _wf_recent:
                    st.markdown('<div class="section-header">📥 آخر العناصر المقبولة</div>',
                                unsafe_allow_html=True)
                    for _item in reversed(_wf_recent):
                        st.markdown(
                            f"""<div class="root-item">
                            <b>{_item.get('title') or '(بلا عنوان)'}</b><br/>
                            <span style="color:var(--text-muted);font-size:0.85rem">
                            المصدر: {_item.get('source','')} · الجودة: {_item.get('quality_score',0):.1f}
                            </span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                else:
                    st.info("لا عناصر مقبولة بعد — اضغط «اسحب الآن يدوياً» لتجربة السحب الحقيقي.")

    # ══════════════════ 7. الأداء (كاش الأسئلة) ══════════════════
    with core_tabs[6]:
        st.markdown('<div class="section-header">⚡ كاش الأسئلة المتكررة (Agent Cache)</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted);direction:rtl">يخزّن ai/agent_performance_cache.py '
            'إجابات المسار الرمزي القطعي فقط (بلا توليد حر، بلا سياق محادثة سابقة) لمدة 30 دقيقة — '
            'يسرّع الأسئلة المتكررة (أسئلة FAQ شائعة، إعادة تحميل الصفحة) دون أي تأثير على دقة أو حداثة '
            'الإجابات ذات السياق أو التوليد الحر.</p>',
            unsafe_allow_html=True,
        )
        try:
            from knowledge.qa_engine import _get_agent_cache
            _ac = _get_agent_cache()
        except Exception:
            _ac = None

        if _ac is None:
            st.warning("⚠️ طبقة الكاش غير محمَّلة بعد (تُحمَّل تلقائياً عند أول سؤال في هذه الجلسة).")
        else:
            _ac_stats = _ac.stats()
            col_ac1, col_ac2, col_ac3, col_ac4 = st.columns(4)
            with col_ac1:
                metric_card(_ac_stats["size"], "عناصر مخزَّنة حالياً")
            with col_ac2:
                metric_card(_ac_stats["hit_rate"], "نسبة الإصابة (Hit Rate)")
            with col_ac3:
                metric_card(_ac_stats["hits"], "إجابات مُسرَّعة من الكاش")
            with col_ac4:
                metric_card(_ac_stats["misses"], "أسئلة جديدة (تُحسَب من الصفر)")

            st.caption(f"الحد الأقصى: {_ac_stats['max_size']} عنصر · مدة الصلاحية: {_ac_stats['ttl_seconds']//60} دقيقة")

            if st.button("🗑️ إفراغ الكاش الآن", key="ac_clear"):
                _cleared = _ac.clear()
                st.success(f"تم إفراغ {_cleared} عنصراً من الكاش.")
                st.rerun()
