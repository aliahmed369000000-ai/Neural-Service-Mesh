"""
ui_pages/memory.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_memory():
    """تبويب الذاكرة."""
    st.markdown('<div class="section-header">🧠 حالة الذاكرة</div>', unsafe_allow_html=True)

    episodic = get_episodic_stats()
    ckg      = load_ckg()
    roots    = load_arabic_roots()

    concepts_count  = len(ckg.get("concepts", {}))
    relations_count = len(ckg.get("relations", {}))

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(episodic.get("episodic", 0), "ذاكرة تجريبية")
    with col2: metric_card(concepts_count, "ذاكرة دلالية (مفاهيم)")
    with col3: metric_card(relations_count, "علاقات مستنتجة")
    with col4: metric_card(len(roots), "جذر عربي مفهرس")

    st.markdown("")
    st.markdown('<div class="section-header">📁 تفاصيل الذاكرة الدلالية (CKG)</div>', unsafe_allow_html=True)

    concepts_db = ckg.get("concepts", {})
    if concepts_db:
        # عرض أقوى المفاهيم
        sorted_concepts = sorted(
            concepts_db.items(),
            key=lambda x: x[1].get("frequency", 0),
            reverse=True
        )[:15]

        for cname, cdata in sorted_concepts:
            freq     = cdata.get("frequency", 0)
            cluster  = cdata.get("cluster", "غير مصنّف")
            strength = cdata.get("strength", 0.0)
            sources  = cdata.get("sources", [])
            st.markdown(f"""
            <div class="root-item">
                <strong>{cname}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{cluster}</span>
                <span class="badge badge-blue">تكرار: {freq}</span>
                <span class="badge badge-amber">قوة: {strength:.2f}</span>
                <br><small style="color:var(--text-muted)">المصادر: {', '.join(sources[:3]) if sources else 'غير محددة'}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("الذاكرة الدلالية (CKG) فارغة حالياً. قم بتشغيل دورة تدريب في Colab لملئها.")

    # ── أنواع العلاقات في CKG ────────────────────────────────────────────
    relations_db = ckg.get("relations", {})
    if relations_db:
        st.markdown("")
        st.markdown('<div class="section-header">🔗 أنواع العلاقات في الذاكرة الدلالية</div>', unsafe_allow_html=True)

        rel_type_counter = Counter(r.get("relation_type", "غير محدد") for r in relations_db.values())
        type_labels = {
            "co_occurrence":    "تزامن في الآية",
            "semantic":         "علاقة دلالية (نفس المجموعة)",
            "thematic_cluster": "تجمّع موضوعي (تشارك سور)",
            "root_link":        "ربط بجذر عربي",
            "narrative_sequence": "تسلسل سردي (قصص الأنبياء)",
            "episodic_rule":    "قاعدة من الذاكرة التجريبية",
        }
        badges = " ".join(
            f'<span class="badge badge-blue" style="margin:3px">{type_labels.get(t, t)}: {n}</span>'
            for t, n in rel_type_counter.most_common()
        )
        st.markdown(badges, unsafe_allow_html=True)

    # ── ملامح السور (Surah Thematic Profiles) ───────────────────────────
    surah_profiles = ckg.get("surah_profiles", {})
    if surah_profiles:
        st.markdown("")
        st.markdown('<div class="section-header">📖 ملامح السور الموضوعية</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:var(--text-muted)">تم بناء ملامح موضوعية لـ {len(surah_profiles)} سورة '
            f'بناءً على المفاهيم الأكثر ظهوراً في كل سورة.</p>',
            unsafe_allow_html=True,
        )

        surah_options = sorted(surah_profiles.keys(), key=lambda x: int(x))
        chosen_surah = st.selectbox(
            "اختر سورة لعرض ملامحها:",
            options=surah_options,
            format_func=lambda s: f"سورة {s}",
            key="surah_profile_select",
        )
        if chosen_surah:
            profile = surah_profiles.get(chosen_surah, [])
            badges = " ".join(
                f'<span class="badge badge-purple" style="margin:3px">{p["concept"]} ({p["weight"]})</span>'
                for p in profile
            )
            st.markdown(badges, unsafe_allow_html=True)

    # حالة قاعدة البيانات
    st.markdown("")
    st.markdown('<div class="section-header">💾 حالة قواعد البيانات</div>', unsafe_allow_html=True)
    db_path = MEMORY_DIR / "episodic.db"
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        st.markdown(f'<span class="health-ok">✅ قاعدة الذاكرة التجريبية: متصلة ({size_kb:.1f} KB)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="health-err">❌ قاعدة الذاكرة التجريبية: غير موجودة</span>', unsafe_allow_html=True)

    # ── إحصاءات الذاكرة التجريبية للأسئلة والأجوبة ──────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">📊 إحصاءات ذاكرة الأسئلة والأجوبة</div>', unsafe_allow_html=True)

    try:
        qa_stats = get_memory_stats(db_path)
    except Exception:
        qa_stats = {"total_episodes": 0, "common_concepts": [], "recent_episodes": [], "avg_confidence": 0.0}

    qcol1, qcol2 = st.columns(2)
    with qcol1: metric_card(qa_stats["total_episodes"], "إجمالي الحلقات المخزّنة")
    with qcol2: metric_card(f"{qa_stats['avg_confidence']:.0%}", "متوسط درجة الثقة")

    if qa_stats["total_episodes"] > 0:
        # أكثر المفاهيم تكراراً في الأسئلة
        st.markdown("**أكثر المفاهيم ظهوراً في الأسئلة:**")
        if qa_stats["common_concepts"]:
            badges = " ".join(
                f'<span class="badge badge-blue" style="margin:2px">{c} ({n})</span>'
                for c, n in qa_stats["common_concepts"][:8]
            )
            st.markdown(badges, unsafe_allow_html=True)

        # أحدث الحلقات
        st.markdown("")
        st.markdown("**أحدث الأسئلة:**")
        for ep in qa_stats["recent_episodes"][:5]:
            ts = ep.get("timestamp", "")[:19].replace("T", " ")
            st.markdown(f"""
            <div class="root-item">
                <strong>{ep['question']}</strong>
                <span class="badge badge-amber">ثقة: {ep['confidence']:.0%}</span>
                <br><small style="color:var(--text-muted)">{ts} UTC</small>
            </div>
            """, unsafe_allow_html=True)

        # ── التوحيد (Consolidation) ──
        st.markdown("")
        st.markdown('<div class="section-header">🧬 توحيد الذاكرة (Consolidation)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">يستخرج هذا الإجراء أزواج المفاهيم المتكررة في الأسئلة السابقة، '
            'ويولّد منها قواعد دلالية، ويضيفها كعلاقات جديدة في الذاكرة الدلالية (CKG) '
            'دون حذف أو تعديل أي علاقة موجودة.</p>',
            unsafe_allow_html=True,
        )

        if st.button("🧬 تشغيل التوحيد الآن", key="consolidate_btn"):
            ckg_path = KNOWLEDGE_DIR / "cognitive_graph.json"
            with st.spinner("يتم تحليل الحلقات واستخراج القواعد الدلالية..."):
                ckg_full = load_json(ckg_path) or {"concepts": {}, "relations": {}}
                cons_result = consolidate_memory(db_path, ckg_full, ckg_path, min_co_occurrence=2)
            st.success(
                f"تم التحليل: {cons_result['pairs_analyzed']} زوج مفاهيم، "
                f"{cons_result['new_rules']} قاعدة جديدة، "
                f"{cons_result['new_relations']} علاقة جديدة في CKG."
            )
            load_json.clear()
            load_ckg.clear()

        rules = get_semantic_rules(db_path, limit=10)
        if rules:
            st.markdown("**القواعد الدلالية المستخرجة:**")
            for r in rules:
                st.markdown(f"""
                <div class="root-item">
                    {r['rule_text']}
                    <span class="badge badge-purple">ثقة: {r['confidence']:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("لا توجد أسئلة محفوظة بعد. استخدم تبويب «الأسئلة والأجوبة» لبدء بناء الذاكرة التجريبية.")

    # ── القوانين المكتسبة تلقائياً (MemoryConsolidator) ──────────────────
    # طبقة منفصلة عن قسم "توحيد الذاكرة" أعلاه: تلك يدوية وتُنتج علاقات CKG
    # من أسئلة المستخدم، بينما هذه تعمل تلقائياً بالخلفية كل 15 دقيقة فوق
    # EpisodicMemoryEngine الحقيقية (get_strongest_memories) وتحوّل الأنماط
    # المتكررة (مصدر الحلقة، نطاق قيمة الهدف الرقمي) إلى "قوانين مكتسبة".
    if _CONSOLIDATOR_OK and _EPISODIC_OK:
        st.markdown("")
        st.markdown('<div class="section-header">⚖️ قوانين الذاكرة المكتسبة تلقائياً</div>', unsafe_allow_html=True)
        _consolidator = _get_memory_consolidator()
        if _consolidator is None:
            st.caption("⚠️ تعذّر تشغيل MemoryConsolidator.")
        else:
            _mc_summary = _consolidator.summary()
            _mcol1, _mcol2, _mcol3 = st.columns(3)
            with _mcol1: metric_card(_mc_summary["total_laws"], "قوانين مكتسبة")
            with _mcol2: metric_card(_mc_summary["total_episodes_freed"], "حلقات مُحرَّرة")
            with _mcol3: metric_card(_mc_summary["local_patterns_tracked"], "أنماط قيد الرصد")

            if st.button("⚖️ تشغيل دورة دمج الآن", key="consolidate_laws_btn"):
                with st.spinner("يفحص الذاكرة الإيبيسودية عن أنماط متكررة..."):
                    _mc_report = _consolidator.consolidate()
                st.success(
                    f"فُحصت {_mc_report['episodes_scanned']} حلقة، "
                    f"{_mc_report['new_laws']} قانون جديد، "
                    f"{_mc_report['updated_laws']} قانون محدَّث."
                )

            _laws = _consolidator.get_consolidated_laws()
            if _laws:
                st.markdown("**القوانين المكتسبة (مرتبة بالثقة):**")
                for _law in _laws[:8]:
                    st.markdown(f"""
                    <div class="root-item">
                        {_law['description']}
                        <span class="badge badge-green">ثقة: {_law['confidence']:.0%}</span>
                        <span class="badge badge-blue">×{_law['occurrence_count']}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.caption(f"لا توجد قوانين بعد — يحتاج نمط للتكرار {_consolidator._threshold} مرات على الأقل.")

    # ── سجل المحادثات المحفوظة (nsm_memory.py — SQLite) ──────────────────
    st.markdown("")
    st.markdown('<div class="section-header">📜 سجل المحادثات المحفوظة</div>', unsafe_allow_html=True)
    try:
        from nsm_memory import _LongTermStore as _NSMLongTermStore
        _mem_store = _NSMLongTermStore()
        _all_sessions = _mem_store.list_sessions(limit=100)
    except Exception as _mem_err:
        _mem_store = None
        _all_sessions = []
        st.caption(f"⚠️ تعذّر تحميل سجل المحادثات: {_mem_err}")

    if _mem_store is not None:
        if not _all_sessions:
            st.info("لا توجد محادثات محفوظة بعد. ابدأ محادثة في تبويب «💬 المحادثة».")
        else:
            _sess_labels = {
                s["session_id"]: f"{s['session_id']} · {s['turns']} رسالة · "
                                 f"{datetime.fromtimestamp(s['last_ts']).strftime('%Y-%m-%d %H:%M') if s.get('last_ts') else ''}"
                for s in _all_sessions
            }
            _mem_col1, _mem_col2 = st.columns([2, 1])
            with _mem_col1:
                _chosen_session = st.selectbox(
                    "اختر جلسة لاستعراض محادثاتها",
                    options=list(_sess_labels.keys()),
                    format_func=lambda k: _sess_labels.get(k, k),
                    key="mem_browse_session",
                )
            with _mem_col2:
                _mem_search = st.text_input(
                    "🔎 ابحث داخل هذه الجلسة", key="mem_browse_search", placeholder="كلمة مفتاحية..."
                )

            _turns = _mem_store.list_recent_turns(limit=200, session_id=_chosen_session)
            if _mem_search.strip():
                _needle = _mem_search.strip().lower()
                _turns = [t for t in _turns if _needle in t["user"].lower() or _needle in t["bot"].lower()]

            st.caption(f"عدد الأدوار المعروضة: {len(_turns)}")
            for _t in _turns[:50]:
                _ts_str = datetime.fromtimestamp(_t["ts"]).strftime("%Y-%m-%d %H:%M") if _t.get("ts") else ""
                st.markdown(f"""
                <div class="root-item">
                    <span class="badge badge-blue">👤 {_t['user'][:200]}</span><br>
                    <span class="badge badge-purple" style="margin-top:4px">🧠 {_t['bot'][:300]}</span>
                    <br><small style="color:var(--text-muted)">{_ts_str} · {_t.get('topic') or 'بدون موضوع'}</small>
                </div>
                """, unsafe_allow_html=True)
