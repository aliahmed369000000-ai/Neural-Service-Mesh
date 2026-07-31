"""
ui_pages/search.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_search():
    """تبويب البحث المعرفي — قلب النظام."""
    st.markdown('<div class="section-header">🔍 البحث المعرفي</div>', unsafe_allow_html=True)
    st.markdown("ابحث عن أي مفهوم وسيظهر لك ما يعرفه النظام عنه:")

    default_q = st.session_state.get("search_query", "")
    query = st.text_input(
        "",
        value=default_q,
        placeholder="اكتب مفهوماً... مثل: الصبر، الجاذبية، التوبة، العلم",
        key="main_search",
        label_visibility="collapsed",
    )

    # أمثلة سريعة
    st.markdown("**أمثلة:**")
    ex_cols = st.columns(6)
    examples = ["الصبر", "الرحمة", "العلم", "الجاذبية", "العدل", "الإيمان"]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex
                st.session_state["search_query"] = ex

    st.markdown("---")

    if not query.strip():
        st.info("اكتب مفهوماً في خانة البحث أعلاه لاستكشاف قاعدة المعرفة.")
        return

    # تنفيذ البحث
    with st.spinner("🔍 جارٍ البحث في قاعدة المعرفة..."):
        result = search_knowledge(query.strip())

    if not result["found"]:
        st.warning(f"لم يُعثر على معلومات كافية عن «{query}» حتى الآن. يتعلم النظام بشكل مستمر!")
        return

    # ── عرض النتائج ──────────────────────────────────────────────────────

    # بطاقة المفهوم الرئيسية — بنداء واحد متكامل (كانت مقسّمة على 3 نداءات
    # منفصلة سابقاً، ما يجعل Streamlit يرسم كل جزء كعنصر DOM مستقل، فلا
    # تلتف البطاقة فعلياً حول محتواها البصري رغم تطابق المظهر ظاهرياً)
    cdata = result["concept_data"]
    _stats_html = ""
    if cdata:
        _stats_html = f"""
        <div class="concept-stats">
            <div class="concept-stat"><span class="concept-stat-label">التصنيف</span><span class="concept-stat-value">{cdata.get('cluster', 'غير مصنّف')}</span></div>
            <div class="concept-stat"><span class="concept-stat-label">التكرار</span><span class="concept-stat-value">{cdata.get('frequency', 0):,} مرة</span></div>
            <div class="concept-stat"><span class="concept-stat-label">قوة المفهوم</span><span class="concept-stat-value">{cdata.get('strength', 0.0):.2%}</span></div>
        </div>
        """
    st.markdown(f"""
    <div class="concept-card">
        <div class="concept-name">💡 {result['query']}</div>
        {_stats_html}
    </div>
    """, unsafe_allow_html=True)

    # ── المفاهيم المرتبطة ────────────────────────────────────────────────
    related_concepts = []
    if result["ckg_related"]:
        related_concepts = result["ckg_related"]
    elif result["root_matches"]:
        related_concepts = [m[0] for m in result["root_matches"] if m[0] != query]

    if related_concepts:
        st.markdown('<div class="section-header">🔗 المفاهيم المرتبطة</div>', unsafe_allow_html=True)
        tags_html = ""
        for concept in related_concepts[:12]:
            tags_html += f'<span class="related-tag">{concept}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── العلاقات من CKG ──────────────────────────────────────────────────
    if result["ckg_relations"]:
        st.markdown('<div class="section-header">↔️ العلاقات المعرفية</div>', unsafe_allow_html=True)
        for rel in result["ckg_relations"][:6]:
            rel_type = rel.get("type", "مرتبط")
            weight   = rel.get("weight", 0)
            target   = rel.get("target", "")
            badge_color = "badge-blue"
            st.markdown(f"""
            <div class="root-item">
                <span class="badge {badge_color}">{rel_type}</span>
                &nbsp;→&nbsp; <strong>{target}</strong>
                &nbsp;&nbsp; <small style="color:var(--text-muted)">قوة: {weight:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    # ── الإشارات القرآنية ────────────────────────────────────────────────
    quran_matches = result["quran_matches"]
    if quran_matches:
        st.markdown(f'<div class="section-header">📖 الإشارات القرآنية ({len(quran_matches)} آية)</div>', unsafe_allow_html=True)
        for ayah in quran_matches[:6]:
            surah = ayah.get("surah", "")
            verse = ayah.get("ayah", "")
            text  = ayah.get("text", "")
            st.markdown(f"""
            <div class="quran-verse">
                {text}
                <div class="verse-ref">سورة {surah}، الآية {verse}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(quran_matches) > 6:
            with st.expander(f"عرض {len(quran_matches) - 6} آية إضافية"):
                for ayah in quran_matches[6:]:
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
        st.markdown('<div class="section-header">📖 الإشارات القرآنية</div>', unsafe_allow_html=True)
        st.info("لم يُعثر على آيات مباشرة لهذا المفهوم بهذه الصياغة. جرّب مرادفاً أو جذر الكلمة.")

    # ── المصادر ودرجة الثقة ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 تفاصيل البحث</div>', unsafe_allow_html=True)
    col_src, col_conf = st.columns(2)
    with col_src:
        sources = result["sources"] or ["الجذور العربية"]
        st.markdown(f"**المصادر:** {' ، '.join(sources)}")
    with col_conf:
        conf = result["confidence"]
        st.markdown(f"**درجة الثقة:** {conf:.0%}")
        st.progress(conf)

    # ── الجذور المرتبطة من الجذور العربية ────────────────────────────────
    root_matches = result["root_matches"]
    if root_matches:
        with st.expander("🌿 الجذور العربية المكتشفة"):
            for token, freq in root_matches[:10]:
                st.markdown(f"""
                <div class="root-item">
                    <strong>{token}</strong>
                    <span class="badge badge-green" style="float:left">تكرار: {freq:,}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── تحليل اللغة العربية (ArabicNLP) ─────────────────────────────────
    if _ARABIC_NLP_OK and query.strip():
        with st.expander("🔬 التحليل اللغوي العميق (ArabicNLP)"):
            try:
                _nlp_engine = get_arabic_engine(ckg=load_ckg())
                _analysis   = _nlp_engine.analyse(query.strip())
                _fv         = _analysis.feature_vector
                col_n1, col_n2, col_n3 = st.columns(3)
                with col_n1:
                    st.metric("نسبة الأفعال", f"{_fv.verb_score:.0%}")
                    st.metric("نسبة الأسماء", f"{_fv.noun_score:.0%}")
                with col_n2:
                    st.metric("تعقيد الجذور", f"{_fv.root_complexity:.0%}")
                    st.metric("التعقيد النحوي", f"{_fv.syntactic_complexity:.0%}")
                with col_n3:
                    st.metric("الكثافة الدلالية", f"{_fv.semantic_concept_score:.0%}")
                    st.metric("تماسك السياق", f"{_fv.context_score:.0%}")
                if _analysis.syntactic.tokens:
                    _pos_badge = {"verb": "blue", "noun": "purple", "adj": "purple", "particle": "amber"}
                    _tokens_html = " ".join(
                        f'<span class="badge badge-{_pos_badge.get(t.pos, "amber")}" style="margin:2px" '
                        f'title="جذر: {t.root or "—"} · وزن: {t.wazn or "—"}">{t.raw}</span>'
                        for t in _analysis.syntactic.tokens[:20]
                    )
                    st.markdown(f"**الرموز المُحلَّلة:** {_tokens_html}", unsafe_allow_html=True)
                if _analysis.morphological.unique_roots:
                    st.markdown(f"**الجذور المكتشفة:** `{'، '.join(_analysis.morphological.unique_roots[:8])}`")
            except Exception as _nlp_err:
                st.caption(f"تعذّر التحليل: {_nlp_err}")

    # ── بحث الويب الحقيقي ────────────────────────────────────────────────
    if _WEB_SEARCH_OK:
        st.markdown("")
        st.markdown('<div class="section-header">🌐 بحث في الإنترنت</div>', unsafe_allow_html=True)
        _ws_cols = st.columns([3, 1])
        with _ws_cols[0]:
            _ws_q = st.text_input(
                "ابحث في الويب",
                value=query.strip() if query.strip() else "",
                placeholder="اكتب ما تريد البحث عنه في الإنترنت...",
                key="web_search_query",
                label_visibility="collapsed",
            )
        with _ws_cols[1]:
            _ws_btn = st.button("🌐 ابحث", key="web_search_btn", use_container_width=True)

        if _ws_btn and _ws_q.strip():
            with st.spinner("⟳ جارٍ البحث في الإنترنت (DuckDuckGo)..."):
                _ws_result = _web_search(_ws_q.strip(), max_results=6)
            st.markdown(f"""
            <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                        padding:1rem 1.4rem;direction:rtl;line-height:1.9;
                        white-space:pre-wrap;font-size:0.93rem;border:1px solid #1e3a5f">
            {_ws_result}
            </div>
            """, unsafe_allow_html=True)

    # ── بحث حقيقي عن الصور (Unsplash) ───────────────────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">🖼️ بحث عن الصور</div>', unsafe_allow_html=True)
    try:
        from ai.image_search_tool import image_search_safe as _img_search
        _IMG_SEARCH_OK = True
    except Exception as _img_imp_err:
        _IMG_SEARCH_OK = False
        st.caption(f"⚠️ تعذّر تحميل أداة بحث الصور: {_img_imp_err}")

    if _IMG_SEARCH_OK:
        _is_cols = st.columns([3, 1])
        with _is_cols[0]:
            _is_q = st.text_input(
                "ابحث عن صور",
                placeholder="مثال: مسجد، طبيعة، خط عربي...",
                key="image_search_query",
                label_visibility="collapsed",
            )
        with _is_cols[1]:
            _is_btn = st.button("🖼️ ابحث", key="image_search_btn", use_container_width=True)

        if _is_btn and _is_q.strip():
            with st.spinner("⟳ جارٍ البحث عن الصور (Unsplash)..."):
                _is_result = _img_search(_is_q.strip(), max_results=9)

            if not _is_result["ok"]:
                st.error(f"❌ {_is_result['error']}")
            else:
                _is_images = _is_result["results"]
                _is_grid = st.columns(3)
                for _i, _img in enumerate(_is_images):
                    with _is_grid[_i % 3]:
                        st.image(_img["thumb_url"] or _img["url"], use_container_width=True)
                        _cap = _img["description"] or "بدون وصف"
                        st.caption(f"📷 {_cap}")
                        if _img.get("author"):
                            _author_line = f"[{_img['author']}]({_img['author_url']})" if _img.get("author_url") else _img["author"]
                            st.caption(f"بواسطة {_author_line}", unsafe_allow_html=False)
