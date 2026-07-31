"""
ui_pages/training.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة




def render_training():
    """تبويب التدريب."""
    # ── 📊 إحصاءات النظام المعرفي — انتقلت هنا من الصفحة الرئيسية ──
    _roots       = load_arabic_roots()
    _ckg_overview = load_ckg()
    _quran_index = load_quran_index()
    _training_ov = load_training_summary()
    _checkpoint  = load_latest_checkpoint()
    _episodic    = get_episodic_stats()

    _concepts_count  = len(_ckg_overview.get("concepts", {}))
    _relations_count = len(_ckg_overview.get("relations", {}))
    _meaningful_roots = sum(1 for k in _roots if len(k) >= 3 and _roots[k].get("frequency", 0) >= 5)
    _train_steps_ov = _training_ov.get("train_steps", 0)

    # آخر تحديث — وقت مطلق + وقت نسبي ("منذ...") لملاحظة الحيوية بلمحة
    _saved_at = _checkpoint.get("saved_at", "")
    _last_update = "غير محدد"
    _last_update_relative = ""
    if _saved_at:
        try:
            _dt = datetime.fromisoformat(_saved_at.replace("Z", "+00:00"))
            _last_update = _dt.strftime("%Y-%m-%d %H:%M") + " UTC"
            _now = datetime.now(_dt.tzinfo) if _dt.tzinfo else datetime.utcnow()
            _delta_sec = max(0, (_now - _dt).total_seconds())
            if _delta_sec < 60:
                _last_update_relative = "منذ لحظات"
            elif _delta_sec < 3600:
                _last_update_relative = f"منذ {int(_delta_sec // 60)} دقيقة"
            elif _delta_sec < 86400:
                _last_update_relative = f"منذ {int(_delta_sec // 3600)} ساعة"
            else:
                _last_update_relative = f"منذ {int(_delta_sec // 86400)} يوم"
        except Exception:
            _last_update = _saved_at[:19]

    st.markdown(
        '<div class="section-header">📊 إحصاءات النظام المعرفي <span class="live-dot"></span></div>',
        unsafe_allow_html=True,
    )

    _last_label_ov = f"آخر تحديث · {_last_update_relative}" if _last_update_relative else "آخر تحديث"
    st.markdown(f"""
    <div class="bento-grid">
        <div class="metric-card bento-featured">
            <div class="metric-value" data-count-target="{_concepts_count}">{_concepts_count:,}</div>
            <div class="metric-label">مفهوم في CKG</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_relations_count}">{_relations_count:,}</div>
            <div class="metric-label">علاقة معرفية</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_meaningful_roots}">{_meaningful_roots:,}</div>
            <div class="metric-label">جذر عربي مكتشف</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_train_steps_ov}">{_train_steps_ov:,}</div>
            <div class="metric-label">خطوة تدريب</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_quran_index.get('total_ayat', 6236)}">{_quran_index.get('total_ayat', 6236):,}</div>
            <div class="metric-label">آية قرآنية محملة</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_quran_index.get('total_surahs', 114)}">{_quran_index.get('total_surahs', 114)}</div>
            <div class="metric-label">سورة كريمة</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" data-count-target="{_episodic.get('episodic', 0)}">{_episodic.get('episodic', 0):,}</div>
            <div class="metric-label">ذكرى تجريبية</div>
        </div>
        <div class="metric-card">
            <div class="metric-value metric-value--wrap">{_last_update}</div>
            <div class="metric-label">{_last_label_ov}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # عدّاد متحرك من 0 حتى القيمة الفعلية — نفس أسلوب حقن JS المضمون
    # (components.html بدل st.markdown الذي لا يُنفَّذ فيه <script> إطلاقاً)
    st.components.v1.html("""
    <script>
    (function() {
        const doc = window.parent.document;
        const counters = doc.querySelectorAll('.metric-value[data-count-target]');
        counters.forEach(function(el) {
            if (el.dataset.nsmAnimated) return;
            el.dataset.nsmAnimated = "1";
            const target = parseInt(el.getAttribute('data-count-target'), 10) || 0;
            const duration = 900;
            const start = performance.now();
            function tick(now) {
                const p = Math.min(1, (now - start) / duration);
                const eased = 1 - Math.pow(1 - p, 3);
                el.textContent = Math.round(eased * target).toLocaleString('en-US');
                if (p < 1) requestAnimationFrame(tick);
                else el.textContent = target.toLocaleString('en-US');
            }
            requestAnimationFrame(tick);
        });
    })();
    </script>
    """, height=0)

    st.markdown("")
    training   = load_training_summary()
    checkpoint = load_latest_checkpoint()
    ckg        = load_ckg()

    train_steps  = training.get("train_steps", 0)
    last_loss    = training.get("last_loss", 0.0)
    total_params = training.get("total_parameters", 0)
    ckg_size     = len(ckg.get("concepts", {}))
    _is_active   = train_steps > 0

    _hdr_col, _btn_col, _save_col = st.columns([4.2, 1, 1.5])
    with _hdr_col:
        _pill = (
            '<span class="status-pill status-pill--active"><span class="status-pill-dot"></span>نشط</span>'
            if _is_active else
            '<span class="status-pill status-pill--idle"><span class="status-pill-dot"></span>لم يبدأ بعد</span>'
        )
        st.markdown(f'<div class="section-header">🎓 حالة التدريب {_pill}</div>', unsafe_allow_html=True)
    with _btn_col:
        if st.button("🔄 تحديث", key="training_refresh_btn", use_container_width=True):
            load_training_summary.clear()
            load_latest_checkpoint.clear()
            load_ckg.clear()
            st.rerun()
    with _save_col:
        if st.button("💾 حفظ Checkpoint", key="save_checkpoint_btn", use_container_width=True,
                      disabled=not _CHECKPOINT_OK, help="يحفظ حالة CKG + الذاكرة الإيبيسودية الحقيقية الآن"):
            with st.spinner("💾 جارٍ حفظ الحالة الحقيقية..."):
                _saved_path = save_real_checkpoint()
            if _saved_path:
                st.toast(f"✅ تم الحفظ: {os.path.basename(_saved_path)}", icon="💾")
                load_latest_checkpoint.clear()
                st.rerun()
            else:
                st.toast("⚠️ تعذّر الحفظ — راجع السجلّات", icon="⚠️")

    if _GITHUB_SYNC_OK:
        _gh_status = _github_sync.status()
        if _gh_status.get("token_set"):
            if _gh_status.get("push_count", 0) > 0:
                if _gh_status.get("last_push_ok"):
                    st.caption(f"🔗 GitHub sync: آخر رفع ناجح ({_gh_status.get('last_push_ts', '')})")
                else:
                    st.caption(f"🔗 GitHub sync: آخر محاولة فشلت — {_gh_status.get('last_push_msg', '')}")
            else:
                st.caption("🔗 GitHub sync: جاهز (لم يُنفَّذ أي رفع بعد)")

    if not _is_active:
        st.markdown("""
        <div class="training-empty">
            <div class="training-empty-icon">🌱</div>
            <div class="training-empty-text">
                لم تُسجَّل أي خطوة تدريب بعد على هذه النسخة. بمجرد تشغيل دورة تدريب
                (<code>ai/knowledge_trainer.py</code> أو <code>ai/continual_learner.py</code>)
                ستظهر هنا خطوات التدريب، قيمة الخسارة، ونقاط الحفظ فور توفّرها.
            </div>
        </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card(f"{train_steps:,}", "خطوات التدريب")
    with col2:
        _loss_display = f"{last_loss:.2e}" if last_loss else "—"
        metric_card(_loss_display, "آخر خسارة (Loss)")
    with col3:
        metric_card(f"{total_params:,}" if total_params else "—", "معامل في الشبكة")
    with col4:
        metric_card(f"{ckg_size:,}", "مفهوم في CKG")

    st.markdown("")

    # معلومات الـ Checkpoint
    saved_at = checkpoint.get("saved_at", "")
    if saved_at:
        st.markdown('<div class="section-header">💾 آخر نقطة حفظ</div>', unsafe_allow_html=True)
        try:
            dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
            st.info(f"تم الحفظ في: **{dt.strftime('%Y-%m-%d الساعة %H:%M:%S')} UTC**")
        except Exception:
            st.info(f"تم الحفظ في: {saved_at}")

        state = checkpoint.get("state", {})
        if state:
            st.markdown('<div class="section-header">🧠 محتوى نقطة الحفظ</div>', unsafe_allow_html=True)
            module_labels = {
                "neural_weights":  ("⚙️", "الأوزان العصبية"),
                "deep_network":    ("🧬", "الشبكة العميقة"),
                "dynamic_layer":   ("🔀", "الطبقة الديناميكية"),
                "episodic_memory": ("💭", "الذاكرة التجريبية"),
                "world_model":     ("🌍", "نموذج العالم"),
                "system_dna":      ("🧿", "الحمض النووي للنظام"),
                "self_awareness":  ("👁️", "الوعي الذاتي"),
                "knowledge_keys":  ("📚", "مفاهيم CKG"),
                "meta":            ("📋", "البيانات الوصفية"),
            }
            _chips_html = '<div class="module-chip-grid">'
            for module_name in state.keys():
                _icon, _label = module_labels.get(module_name, ("✅", module_name))
                _chips_html += (
                    f'<div class="module-chip"><span class="module-chip-dot"></span>'
                    f'<span>{_icon} {_label}</span></div>'
                )
            _chips_html += '</div>'
            st.markdown(_chips_html, unsafe_allow_html=True)

    # معلومات التدريب التفصيلية
    if training:
        st.markdown("")
        st.markdown('<div class="section-header">📐 بنية الشبكة العصبية</div>', unsafe_allow_html=True)
        arch = training.get("architecture", "")
        if arch:
            st.markdown('<div class="arch-card">', unsafe_allow_html=True)
            st.code(arch, language=None)
            st.markdown('</div>', unsafe_allow_html=True)

        avg_loss = training.get("avg_recent_loss", 0)
        lr       = training.get("learning_rate", 0)
        col_a, col_b = st.columns(2)
        with col_a:
            _avg_display = f"`{avg_loss:.2e}`" if avg_loss else "`—`"
            st.markdown(f"**متوسط الخسارة الأخيرة:** {_avg_display}")
        with col_b:
            st.markdown(f"**معدل التعلم:** `{lr}`" if lr else "**معدل التعلم:** `—`")

    # ── [NSM Router Bridge] تبويب التوجيه الذكي ──────────────────────────
    st.markdown("")
    render_nsm_routing()



def render_nsm_routing():
    """لوحة NSM Mesh — توجيه دلالي + self-healing + سجل حي + إثبات تعلم."""
    st.markdown(
        '<div class="section-header">🕸️ NSM Mesh — الشبكة الذكية الحية '
        '<span class="live-dot"></span></div>',
        unsafe_allow_html=True,
    )

    if not _NSM_BRIDGE_OK or not _nsm_bridge:
        st.warning("⚠️ NSM Router Bridge غير مُفعَّل.")
        return

    # ════════════════════════════════════════════════════════════════════════
    # [A] درجات العقد الثلاث — أداء تاريخي فوري
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("#### 📡 درجات العقد الحية")
    node_scores = _nsm_bridge.get_node_scores_for_display()
    n_cols = st.columns(len(node_scores))
    for col, ns in zip(n_cols, node_scores):
        score = ns["connection_score"]
        sc = "var(--emerald)" if score >= 70 else ("var(--gold)" if score >= 45 else "var(--text-muted)")
        bar_w = int(score)
        with col:
            st.markdown(f"""
            <div class="metric-card" style="text-align:center;padding:0.9rem">
                <div style="font-size:1.3rem;margin-bottom:0.25rem">{ns["label"]}</div>
                <div class="metric-value" style="color:{sc};font-size:2.2rem;line-height:1">
                    {score:.1f}
                </div>
                <div class="metric-label" style="margin-bottom:0.5rem">/ 100</div>
                <div style="background:var(--surface);border-radius:6px;height:6px;overflow:hidden;margin-bottom:0.6rem">
                    <div style="background:{sc};width:{bar_w}%;height:6px;border-radius:6px"></div>
                </div>
                <div style="font-size:0.78rem;color:var(--text-muted);direction:rtl;line-height:1.8">
                    🔁 <strong>{ns["total_runs"]}</strong> تشغيل &nbsp;
                    ✅ <strong>{ns["success_rate"]:.0f}%</strong> &nbsp;
                    ⏱️ <strong>{int(ns["avg_latency_ms"])}ms</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # [B] سجل التوجيه الحي — آخر القرارات في هذه الجلسة
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("")
    st.markdown("#### 🔴 سجل التوجيه الحي")

    _rlog = st.session_state.get("nsm_route_log", [])
    if not _rlog and _ROUTE_LOG_DB_OK:
        # استرجاع الذاكرة التراكمية من SQLite عند عدم وجود سجل في الجلسة الحالية
        _rlog = _rlog_get_recent(limit=100)
        if _rlog:
            st.session_state["nsm_route_log"] = _rlog
    if not _rlog:
        st.info("📋 سيظهر سجل التوجيه هنا فور إرسال أول رسالة في تبويب المحادثة.")
    else:
        # إحصاءات سريعة
        _total_req     = len(_rlog)
        _failover_reqs = sum(1 for r in _rlog if r.get("failover"))
        _ok_reqs       = sum(1 for r in _rlog if r.get("success"))
        _avg_lat       = sum(r.get("latency_ms", 0) for r in _rlog) / max(_total_req, 1)

        _sc1, _sc2, _sc3, _sc4 = st.columns(4)
        with _sc1: metric_card(_total_req, "طلبات في الجلسة")
        with _sc2: metric_card(f"{_ok_reqs/_total_req*100:.0f}%", "معدل النجاح")
        with _sc3: metric_card(_failover_reqs, "إعادة توجيه تلقائي")
        with _sc4: metric_card(f"{_avg_lat:.0f}ms", "متوسط الاستجابة")

        st.markdown("")

        # توزيع الفئات الدلالية
        if _NSM_SEMANTIC_OK and _nsm_semantic:
            _cat_counts: dict = {}
            for r in _rlog:
                _cat = r.get("category", "general")
                _cat_counts[_cat] = _cat_counts.get(_cat, 0) + 1
            if _cat_counts:
                _cat_html = '<div style="direction:rtl;margin-bottom:0.8rem">'
                _cat_html += '<span style="font-size:0.8rem;color:var(--text-muted)">توزيع الاستعلامات: </span>'
                for _cat, _cnt in sorted(_cat_counts.items(), key=lambda x: -x[1]):
                    _lbl = _nsm_semantic.CATEGORY_LABELS.get(_cat, ("💬", _cat))
                    _cat_html += (
                        f'<span class="badge badge-blue" style="margin:2px">'
                        f'{_lbl[0]} {_lbl[1]}: {_cnt}</span>'
                    )
                _cat_html += '</div>'
                st.markdown(_cat_html, unsafe_allow_html=True)

        # توزيع العقد المختارة
        _node_counts: dict = {}
        for r in _rlog:
            _nd = r.get("node", "?").replace("nsm:", "")
            _node_counts[_nd] = _node_counts.get(_nd, 0) + 1
        _node_html = '<div style="direction:rtl;margin-bottom:1rem">'
        _node_html += '<span style="font-size:0.8rem;color:var(--text-muted)">العقد المختارة: </span>'
        for _nd, _cnt in sorted(_node_counts.items(), key=lambda x: -x[1]):
            _node_html += f'<span class="badge badge-amber" style="margin:2px">{_nd}: {_cnt}</span>'
        _node_html += '</div>'
        st.markdown(_node_html, unsafe_allow_html=True)

        # جدول آخر 20 قراراً
        _last20 = list(reversed(_rlog[-20:]))
        _rows_html = ""
        for r in _last20:
            _ico    = r.get("cat_icon", "💬")
            _cat    = r.get("category", "general")
            _nd     = r.get("node", "?").replace("nsm:", "")
            _lat    = r.get("latency_ms", 0)
            _ok     = r.get("success", False)
            _fo     = r.get("failover", False)
            _ok_ico = "✅" if _ok else "❌"
            _fo_badge = '<span class="badge badge-purple" style="font-size:0.65rem">↩️ failover</span>' if _fo else ""
            _q      = r.get("query", "")
            _ts     = r.get("ts", "")
            _qs     = r.get("quality_score")
            _qs_badge = ""
            if _qs is not None:
                _qs_cls = "badge-green" if _qs >= 70 else ("badge-amber" if _qs >= 40 else "badge-purple")
                _qs_badge = f'<span class="badge {_qs_cls}" style="font-size:0.65rem">⭐ {_qs:.0f}</span>'
            _rows_html += f"""
            <div class="root-item" style="direction:rtl;display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;padding:0.4rem 0.7rem">
                <span style="color:var(--text-muted);font-size:0.72rem;min-width:60px">{_ts}</span>
                <span>{_ico}</span>
                <span class="badge badge-blue">{_cat}</span>
                <span class="badge badge-amber">{_nd}</span>
                <span style="font-size:0.8rem;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{_q}</span>
                <span style="min-width:55px;font-size:0.78rem;color:var(--text-muted)">{_lat}ms</span>
                <span>{_ok_ico}</span>
                {_qs_badge}
                {_fo_badge}
            </div>"""
        st.markdown(
            f'<div style="max-height:380px;overflow-y:auto;border:1px solid var(--border);'
            f'border-radius:12px;padding:0.3rem">{_rows_html}</div>',
            unsafe_allow_html=True,
        )

        if st.button("🗑 مسح سجل التوجيه", key="clear_route_log"):
            st.session_state["nsm_route_log"] = []
            if _ROUTE_LOG_DB_OK:
                _rlog_clear_all()
            st.rerun()

        # ────────────────────────────────────────────────────────────────
        # [B.1] رؤى Meta-Reasoner — تحليل تأملي حقيقي فوق سجل التوجيه
        # ────────────────────────────────────────────────────────────────
        if _META_REASONER_OK and _ROUTE_LOG_DB_OK:
            with st.expander("🧠 رؤى Meta-Reasoner (تحليل تأملي لسجل التوجيه)"):
                _reasoner = _get_meta_reasoner()
                if _reasoner is None:
                    st.caption("⚠️ تعذّر تهيئة MetaReasoner.")
                else:
                    _insights = _reasoner.reflect()
                    if not _insights:
                        st.caption("لا توجد أنماط تستحق التنبيه بعد — يحتاج المزيد من طلبات التوجيه المسجَّلة.")
                    else:
                        _badge_by_type = {
                            "warning": "badge-purple", "opportunity": "badge-amber",
                            "pattern": "badge-blue", "lesson": "badge-green",
                        }
                        for _ins in _insights[:8]:
                            _cls = _badge_by_type.get(_ins.insight_type, "badge-blue")
                            st.markdown(f"""
                            <div class="root-item" style="direction:rtl;padding:0.6rem 0.8rem;margin-bottom:0.4rem">
                                <span class="badge {_cls}">{_ins.insight_type}</span>
                                <strong style="margin-right:0.4rem">{_ins.title}</strong>
                                <div style="font-size:0.82rem;color:var(--text-muted);margin-top:0.3rem">{_ins.body}</div>
                            </div>
                            """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════════════
    # [C] إثبات التعلم — prove_learning + learning_curve
    # ════════════════════════════════════════════════════════════════════════
    st.markdown("")
    st.markdown("#### 🎓 إثبات التعلم التراكمي")

    report = _nsm_bridge.get_learning_report()
    if "error" in report:
        st.warning(f"⚠️ {report['error']}")
    else:
        proof   = report.get("proof", {})
        verdict = proof.get("verdict", "insufficient_data")
        _vmap   = {
            "learning_confirmed":       ("✅", "var(--emerald)", "تعلّم مؤكَّد"),
            "learning_in_progress":     ("🔄", "var(--gold)",   "التعلم قيد التقدم"),
            "learning_not_yet_evident": ("⏳", "var(--text-muted)", "بيانات غير كافية بعد"),
            "insufficient_data":        ("⏳", "var(--text-muted)", "بيانات غير كافية بعد"),
        }
        _ico, _vc, _vl = _vmap.get(verdict, ("❓", "var(--text-muted)", verdict))
        st.markdown(
            f'<div class="metric-card" style="border-right:4px solid {_vc};direction:rtl;padding:0.9rem 1.1rem">'
            f'<span style="font-size:1.3rem">{_ico}</span>'
            f' <strong style="color:{_vc}">{_vl}</strong>'
            f'<p style="margin:0.4rem 0 0;color:var(--text-muted);font-size:0.85rem">'
            f'{proof.get("message","")}</p></div>',
            unsafe_allow_html=True,
        )

        _evs = proof.get("evidence", [])
        if _evs:
            st.markdown("")
            for ev in _evs:
                st.markdown(f"- {ev}")

        _metrics = proof.get("metrics", {})
        if _metrics.get("total_executions", 0) > 0:
            st.markdown("")
            _lm1, _lm2, _lm3, _lm4 = st.columns(4)
            with _lm1: metric_card(_metrics.get("total_executions", 0), "تشغيلات تراكمية")
            with _lm2: metric_card(f"{_metrics.get('success_rate',0)*100:.1f}%", "معدل النجاح")
            with _lm3: metric_card(f"{_metrics.get('learning_improvement_pct',0):+.1f}%", "تحسّن vs الأساس")
            with _lm4:
                _tn = proof.get("top_nodes", [])
                metric_card(_tn[0].get("name","—") if _tn else "—", "أكثر العقد ثقةً")

        # منحنى التعلم
        _curve  = report.get("curve", {})
        _pts    = _curve.get("data_points", [])
        _trend  = _curve.get("trend", "insufficient_data")
        _tmap   = {"improving": "📈 تحسّن", "degrading": "📉 تراجع",
                   "stable": "➡️ مستقر", "insufficient_data": "⏳ غير كافٍ"}
        if _pts:
            st.markdown("")
            st.caption(f"منحنى التعلم — الاتجاه: {_tmap.get(_trend, _trend)}")
            try:
                import pandas as _pd
                _df = _pd.DataFrame(_pts)
                if "avg_connection_score" in _df.columns:
                    st.line_chart(_df.set_index("index")["avg_connection_score"],
                                  use_container_width=True)
            except Exception:
                for _dp in _pts[-8:]:
                    st.text(f"[{_dp.get('index','')}] {_dp.get('avg_connection_score','?'):.1f}")
        else:
            st.caption("📊 منحنى التعلم سيظهر بعد تراكم بيانات كافية.")

        # سمعة العقد
        _rep = report.get("reputation", [])
        if _rep:
            st.markdown("")
            st.markdown("#### 🏅 سمعة العقد")
            _tier_c = {"platinum":"var(--emerald)","gold":"var(--gold)",
                       "silver":"#aaa","bronze":"#cd7f32","unrated":"var(--text-muted)"}
            for _nd in _rep:
                _t  = _nd.get("tier","unrated")
                _tc = _tier_c.get(_t,"var(--text-muted)")
                st.markdown(
                    f'<div class="root-item" style="direction:rtl">'
                    f'<strong>{_nd.get("name",_nd.get("node_id","?"))}</strong>'
                    f' <span class="badge badge-blue" style="background:{_tc};color:#fff">{_t}</span>'
                    f' <span class="badge badge-amber">سمعة: {_nd.get("reputation_score",0):.1f}</span>'
                    f' <span class="badge badge-blue">نجاح: {_nd.get("success_rate",0)*100:.0f}%</span>'
                    f' <span style="color:var(--text-muted);font-size:0.78rem">'
                    f'({_nd.get("total_runs",0)} تشغيل)</span></div>',
                    unsafe_allow_html=True,
                )

    # ════════════════════════════════════════════════════════════════════════
    # [D] معلومات التوجيه الدلالي
    # ════════════════════════════════════════════════════════════════════════
    if _NSM_SEMANTIC_OK and _nsm_semantic:
        st.markdown("")
        with st.expander("🧠 كيف يعمل التوجيه الدلالي؟", expanded=False):
            st.markdown("""
**الصيغة:** `درجة_مركَّبة = 65% × ScoringEngine_التاريخي + 35% × تحيُّز_دلالي`

| الفئة | العقدة المُفضَّلة | السبب |
|---|---|---|
| 🕌 عربي/إسلامي | NSM Agent | مُدرَّب ومُخصَّص للعربية والمعرفة الإسلامية |
| 💻 برمجة | OpenRouter (GPT-4o/Claude) | نماذج الكود الأقوى |
| ✍️ إبداعي | OpenRouter | إبداع أغنى مع نماذج كبيرة |
| 🔍 تحليل | OpenRouter | تحليل أعمق مع سياق أوسع |
| 💬 عام | NSM Agent | الافتراضي المُحسَّن للعربية |

⚡ **Failover:** إذا فشل المسار الأول — يُعاد التوجيه تلقائياً للتالي مع تسجيل الفشل.
            """)

    st.caption("🔁 كل رسالة في المحادثة تُحدِّث هذه اللوحة تلقائياً.")
