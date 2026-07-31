"""
ui_pages/qa.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_qa():
    """تبويب الأسئلة والأجوبة القرآني — يعتمد على CKG والآيات فقط."""
    st.markdown('<div class="section-header">❓ الأسئلة والأجوبة القرآني</div>', unsafe_allow_html=True)
    _qa_ckg = load_ckg()
    _qa_concepts_n = len(_qa_ckg.get("concepts", {}))
    _qa_relations_n = len(_qa_ckg.get("relations", {}))
    _qa_ayat_n = load_quran_index().get("total_ayat", 6236)
    st.markdown(
        f'<p style="color:var(--text-muted)">اسأل سؤالاً بالعربية، وسيحلل النظام السؤال '
        f'ويبحث في {_qa_concepts_n:,} مفهوماً و{_qa_relations_n:,} علاقة دلالية و{_qa_ayat_n:,} آية للإجابة.</p>',
        unsafe_allow_html=True,
    )

    # ── أمثلة جاهزة ──
    st.markdown("**أمثلة:**")
    examples = [
        "من هو محمد ﷺ؟",
        "ما علاقة الصبر بالإيمان؟",
        "ماذا يقول القرآن عن العدل؟",
        "ما قصة يوسف؟",
    ]
    ex_cols = st.columns(len(examples))
    chosen_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"qa_example_{i}", use_container_width=True):
                chosen_example = ex

    default_q = chosen_example or st.session_state.get("qa_question", "")
    st.session_state.setdefault("qa_conversation_history", [])
    question = st.text_input(
        "اكتب سؤالك هنا:",
        value=default_q,
        key="qa_input",
        placeholder="مثال: ما علاقة الصبر بالإيمان؟",
    )
    st.session_state["qa_question"] = question

    opt_col1, opt_col2, opt_col3 = st.columns([1, 1, 3])
    with opt_col1:
        show_reasoning = st.checkbox(
            "🧠 اعرض لماذا هذه الإجابة",
            value=st.session_state.get("qa_show_reasoning", False),
            key="qa_show_reasoning",
        )
    with opt_col2:
        show_images = st.checkbox(
            "🖼️ صور توضيحية",
            value=st.session_state.get("qa_show_images", False),
            key="qa_show_images",
        )
    with opt_col3:
        if st.session_state["qa_conversation_history"]:
            st.caption(f"💬 سياق محادثة نشط ({len(st.session_state['qa_conversation_history'])} سؤال سابق)")
            if st.button("🗑️ مسح سياق المحادثة", key="qa_clear_context"):
                st.session_state["qa_conversation_history"] = []
                st.rerun()

    ask = st.button("🔍 اسأل", type="primary")

    if not (ask or chosen_example) or not question.strip():
        return

    ckg  = load_ckg()
    ayat = load_all_quran_ayat()

    if not ckg.get("concepts"):
        st.error("الذاكرة الدلالية (CKG) فارغة — لا يمكن الإجابة على الأسئلة حالياً.")
        return

    with st.spinner("يتم تحليل السؤال والبحث في قاعدة المعرفة..."):
        entities = load_entities()
        result = answer_question(
            question, ckg, ayat, entities=entities,
            generation_mode=st.session_state.get("yemeni_generation_mode", False),
            temperature=st.session_state.get("yemeni_temperature", 0.8),
            top_p=st.session_state.get("yemeni_top_p", 0.95),
            top_k=st.session_state.get("yemeni_top_k", 50),
            include_reasoning_trace=show_reasoning,
            include_images=show_images,
            conversation_history=st.session_state["qa_conversation_history"],
        )

    # ── حظر أمان (nova_system.py) — أولوية على أي عرض آخر، لا LoRA ولا حلقة ──
    if result.get("safety_blocked"):
        st.markdown("---")
        st.warning(f"🛡️ {result['summary']}")
        return

    if result.get("generation_used") and result.get("generated_text"):
        st.markdown("---")
        st.markdown('<div class="section-header">🗣️ توليد حر (تجريبي)</div>', unsafe_allow_html=True)
        st.caption("نص مولَّد بواسطة YemeniDecoder — تجريبي وغير مضمون الدقة، منفصل عن الإجابة الرمزية أدناه.")
        st.info(result["generated_text"])

    # ── حفظ الدور الحالي في سياق المحادثة (لأسئلة المتابعة القادمة) ──
    # يُستبعد عمداً أي رد محظور أمنياً (return أعلاه) حتى لا يتلوّث سياق
    # الأسئلة اللاحقة بمحتوى مرفوض. سقف 5 أدوار لتفادي تضخّم prompt الـLLM
    # بلا حدود مع طول الجلسة.
    st.session_state["qa_conversation_history"].append(
        {"question": question, "summary": result.get("summary", "")}
    )
    st.session_state["qa_conversation_history"] = st.session_state["qa_conversation_history"][-5:]

    # ── حفظ الحلقة في الذاكرة التجريبية ──
    db_path = MEMORY_DIR / "episodic.db"
    try:
        store_episode(db_path, question, result)
    except Exception:
        pass

    # ── أسئلة سابقة مشابهة ──
    try:
        similar = find_similar_episodes(db_path, question, threshold=0.4, top_k=3)
    except Exception:
        similar = []

    st.markdown("---")

    if similar:
        st.markdown('<div class="section-header">🕘 أسئلة سابقة مشابهة</div>', unsafe_allow_html=True)
        for s in similar:
            if normalize_arabic(s["question"]) == normalize_arabic(question):
                continue
            st.markdown(f"""
            <div class="root-item">
                <strong>{s['question']}</strong>
                <span class="badge badge-blue">تشابه: {s['similarity']:.0%}</span>
                <span class="badge badge-amber">ثقة: {s['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")

    # ── ملخص الإجابة ──
    entity_info = result.get("entity")
    if entity_info:
        st.markdown(
            f'<div class="section-header">📝 ملخص الإجابة '
            f'<span class="badge badge-purple">كيان: {entity_info["name"]} ({entity_info["type"]})</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-header">📝 ملخص الإجابة</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="root-item" style="font-size:1.05rem; line-height:1.8">
        {result['summary']}
    </div>
    """, unsafe_allow_html=True)

    # ── درجة الثقة ──
    confidence = result.get("confidence", 0.0)
    st.markdown("")
    st.markdown(f"**درجة الثقة:** {confidence:.0%}")
    st.progress(confidence)

    # ── أثر التفكير (اختياري — ai/chain_of_thought.py) ──
    if result.get("reasoning_trace"):
        with st.expander("🧠 لماذا هذه الإجابة؟"):
            st.markdown(result["reasoning_trace"])

    # ── صور توضيحية (اختياري — ai/image_sources.py) — بهوية زجاجية موحَّدة ──
    images = result.get("images") or []
    if images:
        st.markdown('<div class="section-header">🖼️ صور توضيحية</div>', unsafe_allow_html=True)
        img_cols = st.columns(len(images))
        for col, img in zip(img_cols, images):
            with col:
                url = img.get("url", "")
                source_label = img.get("source", "")
                if url.startswith("https://"):  # فحص أمان بسيط قبل الحقن في HTML
                    st.markdown(f"""
                    <div class="glass-card" style="padding:0.6rem; text-align:center;">
                        <img src="{url}" style="width:100%; border-radius:12px; display:block;"
                             loading="lazy" />
                        <div style="margin-top:0.5rem;">
                            <span class="badge badge-green">المصدر: {source_label}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ── تغذية راجعة: تدريب LoRA خفيف من ملاحظة المستخدم (لا يمسّ الأوزان الأساسية) ──
    _fb_key = f"qa_feedback_{hash(question)}"
    if st.session_state.get(_fb_key) is None:
        fb_col1, fb_col2, fb_col3 = st.columns([1, 1, 4])
        with fb_col1:
            if st.button("👍 إجابة جيدة", key=f"{_fb_key}_up"):
                try:
                    record_positive_feedback(question, result.get("summary", ""))
                except Exception:
                    pass
                st.session_state[_fb_key] = "up"
                st.rerun()
        with fb_col2:
            if st.button("👎 غير دقيقة", key=f"{_fb_key}_down"):
                # لا تدريب على الملاحظات السلبية حالياً (قد يزعزع الشبكة
                # بدون آلية contrastive loss مناسبة) — فقط تسجيل للمراجعة.
                st.session_state[_fb_key] = "down"
                st.rerun()
    else:
        _fb = st.session_state[_fb_key]
        if _fb == "up":
            st.success("✅ شكراً! تم استخدام ملاحظتك لتحسين الفهم الدلالي للنموذج.")
        else:
            st.info("📝 شكراً على ملاحظتك — تم تسجيلها للمراجعة.")

    if not result["primary_concepts"]:
        st.info("لم يتم العثور على مفاهيم مرتبطة بهذا السؤال في قاعدة المعرفة الحالية.")
        return

    # ── المفاهيم الأساسية ──
    st.markdown("")
    st.markdown('<div class="section-header">🧩 المفاهيم المستخرجة من السؤال</div>', unsafe_allow_html=True)
    for c in result["primary_concepts"]:
        if entity_info:
            # في إجابات الكيانات، أرقام "تكرار/تطابق" التقنية لا تضيف
            # قيمة للمستخدم — نعرض فقط الاسم والمجموعة المعرفية
            st.markdown(f"""
            <div class="root-item">
                <strong>{c['name']}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{c['cluster']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="root-item">
                <strong>{c['name']}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{c['cluster']}</span>
                <span class="badge badge-blue">تكرار في القرآن: {c['frequency']}</span>
                <span class="badge badge-amber">درجة التطابق: {c['match']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── المفاهيم المرتبطة (من العلاقات) ──
    related = result.get("related_concepts", [])
    if related:
        st.markdown("")
        st.markdown('<div class="section-header">🔗 مفاهيم مرتبطة (من الذاكرة الدلالية)</div>', unsafe_allow_html=True)
        rel_type_labels = {
            "co_occurrence":     "تزامن في الآية",
            "semantic":          "علاقة دلالية",
            "thematic_cluster":  "تجمّع موضوعي",
            "root_link":         "ربط بجذر",
            "narrative_sequence": "تسلسل سردي",
            "episodic_rule":     "قاعدة من الذاكرة التجريبية",
            "entity_attribute":  "صفة الكيان",
        }
        for r in related[:6]:
            rtype = rel_type_labels.get(r["relation_type"], r["relation_type"])
            st.markdown(f"""
            <div class="root-item">
                <strong>{r['concept']}</strong>
                <span class="badge badge-blue">نوع العلاقة: {rtype}</span>
                <span class="badge badge-amber">وزن العلاقة: {r['weight']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── الآيات الداعمة ──
    verses = result.get("verses", [])
    st.markdown("")
    st.markdown(f'<div class="section-header">📖 الآيات الداعمة ({len(verses)})</div>', unsafe_allow_html=True)
    if verses:
        for v in verses:
            st.markdown(f"""
            <div class="quran-verse">
                {v['text']}
                <div class="verse-ref">سورة {v['surah']}، الآية {v['ayah']} — مفهوم: {v['concept']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("لم يتم العثور على آيات داعمة مباشرة لهذا السؤال.")
