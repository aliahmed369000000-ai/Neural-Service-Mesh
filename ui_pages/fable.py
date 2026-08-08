"""
ui_pages/fable.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب 🎭 إبداع — السرد الإبداعي التفاعلي وتوليد الشعر
# ══════════════════════════════════════════════════════════════════════════
def render_fable():
    """تبويب القصص التفاعلية والشعر — مبني فوق نفس LLMFallback المستخدم
    في المحادثة (Anthropic أولاً ثم بقية المزوّدين المجانية)."""

    st.markdown('<div class="section-header">🎭 إبداع — السرد الإبداعي العربي</div>',
                unsafe_allow_html=True)

    if not _FABLE_OK:
        st.error("⚠️ تعذّر تحميل محرك السرد الإبداعي (ai/fable_engine.py). "
                  "تأكد من رفع الملف إلى مجلد ai/.")
        return

    st.markdown(
        '<p style="color:var(--text-muted)">اختر وضع القصة والراوي، وابدأ حكاية تفاعلية '
        'تتطور حسب اختياراتك، أو اطلب قصيدة على أحد بحور الشعر العربي.</p>',
        unsafe_allow_html=True,
    )

    # ── تهيئة محرك السرد مرة واحدة لكل جلسة Streamlit ──
    if "fable_engine" not in st.session_state:
        fb = _FableLLMFallback(model_key="fable")
        st.session_state.fable_engine = FableEngine(
            llm_fallback=fb, db_path=str(MEMORY_DIR / "fable.db")
        )
        st.session_state.fable_chapter = None   # آخر فصل مُولَّد

    engine = st.session_state.fable_engine

    story_tab, poem_tab, explainer_tab, shorts_tab, edit_tab, library_tab = st.tabs(
        ["📖 قصة تفاعلية", "🪶 توليد شعر", "🎬 وثائقي (سيناريو)", "⚡ Shorts فيديو", "✂️ تعديل فيديو", "📚 مكتبة القصص"]
    )

    # ══════════════════ قصة تفاعلية ══════════════════
    with story_tab:
        st.caption("🎨 اختر الوضع والراوي ثم ابدأ — يمكنك توجيه القصة بالخيارات في كل فصل.")
        cur = st.session_state.fable_chapter

        if cur is None:
            c1, c2 = st.columns(2)
            with c1:
                mode = st.selectbox(
                    "وضع القصة",
                    list(STORY_MODES.keys()),
                    index=list(STORY_MODES.keys()).index(FABLE_DEFAULT_MODE),
                    format_func=lambda m: f"{STORY_MODES[m]['emoji']} {m} — {STORY_MODES[m]['desc']}",
                )
            with c2:
                character = st.selectbox(
                    "الراوي / الأسلوب",
                    list(CHARACTERS.keys()),
                    index=list(CHARACTERS.keys()).index(FABLE_DEFAULT_CHARACTER),
                    format_func=lambda c: f"{CHARACTERS[c]['emoji']} {c} — {CHARACTERS[c]['style']}",
                )
            target_value = None
            if mode == "قصص إسلامية تربوية":
                target_value = st.selectbox(
                    "🕌 القيمة المستهدفة",
                    ISLAMIC_VALUES,
                    help="اختر القيمة أو الخُلق الذي تريد أن تتعلّمه القصة للطفل — "
                         "يمكنك أيضاً إضافة تفاصيل حرة في الحقل أدناه.",
                )
            seed = st.text_input(
                "فكرة مبدئية (اختياري):" if target_value is None
                else "تفاصيل إضافية عن القصة (اختياري):",
                placeholder="مثال: قصة عن تاجر يبحث عن كنز مفقود في الصحراء" if target_value is None
                else "مثال: طفل يجد محفظة نقود في الحديقة",
            )
            if st.button("✨ ابدأ القصة", type="primary"):
                _story_skel_ph = st.empty()
                with _story_skel_ph.container():
                    _skeleton(lines=6)
                effective_seed = seed
                if target_value is not None:
                    effective_seed = (
                        f"اكتب قصة تُعلّم الطفل قيمة «{target_value}»."
                        + (f" تفاصيل إضافية: {seed.strip()}" if seed.strip() else "")
                    )
                try:
                    chapter = engine.start_story(mode=mode, character=character, seed_idea=effective_seed)
                except Exception as e:  # noqa: BLE001
                    _story_skel_ph.empty()
                    st.error(f"⚠️ تعذّر بدء القصة، حاول مرة أخرى. (تفصيل تقني: {e})")
                else:
                    st.session_state.fable_chapter = chapter
                    st.rerun()
        else:
            # ── عرض الفصل الحالي ──
            mode_info = STORY_MODES.get(cur.mode, {})
            char_info = CHARACTERS.get(cur.character, {})
            st.markdown(
                f'<span class="badge badge-purple">{mode_info.get("emoji","")} {cur.mode}</span> '
                f'<span class="badge badge-blue">{char_info.get("emoji","")} {cur.character}</span> '
                f'<span class="badge badge-green">المزوّد: {cur.provider}</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"""
            <div class="root-item" style="font-size:1.05rem; line-height:2; text-align:right; direction:rtl">
                {cur.text}
            </div>
            """, unsafe_allow_html=True)
            _cc1, _cc2, _cc3 = st.columns(3)
            with _cc1:
                _copy_button(cur.text, key="fable_chapter")
            with _cc2:
                try:
                    _full_story_rows = engine.memory.get_history(cur.session_id, limit=500)
                    _full_story_text = "\n\n".join(
                        r["content"] for r in _full_story_rows if r["role"] == "narration"
                    )
                except Exception:  # noqa: BLE001
                    _full_story_text = cur.text
                st.download_button(
                    "⬇️ تحميل القصة كاملة",
                    data=_full_story_text,
                    file_name="قصتي.txt",
                    mime="text/plain",
                    key="fable_story_download",
                    use_container_width=True,
                )
            with _cc3:
                if _PDF_EXPORT_OK:
                    try:
                        _story_pdf_bytes = _story_to_pdf(
                            title="قصتي", mode=cur.mode, character=cur.character,
                            full_text=_full_story_text,
                        )
                    except Exception as e:  # noqa: BLE001
                        _story_pdf_bytes = None
                        st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                    if _story_pdf_bytes:
                        st.download_button(
                            "📄 تحميل PDF",
                            data=_story_pdf_bytes,
                            file_name="قصتي.pdf",
                            mime="application/pdf",
                            key="fable_story_pdf_download",
                            use_container_width=True,
                        )

            if cur.error:
                st.caption(f"⚠️ ملاحظة تقنية: {cur.error}")

            st.markdown("**ماذا يحدث بعد ذلك؟**")
            cols = st.columns(len(cur.choices) or 1)
            chosen = None
            for i, choice in enumerate(cur.choices):
                with cols[i]:
                    if st.button(choice, key=f"fable_choice_{i}", use_container_width=True):
                        chosen = choice

            custom_choice = st.text_input("أو اكتب مسارك الخاص:", key="fable_custom_choice")
            if st.button("➡️ تابع") and custom_choice.strip():
                chosen = custom_choice.strip()

            if chosen:
                _story_cont_skel_ph = st.empty()
                with _story_cont_skel_ph.container():
                    _skeleton(lines=6)
                try:
                    next_chapter = engine.continue_story(cur.session_id, chosen)
                except Exception as e:  # noqa: BLE001
                    _story_cont_skel_ph.empty()
                    st.error(f"⚠️ تعذّر متابعة القصة، حاول مرة أخرى. (تفصيل تقني: {e})")
                else:
                    st.session_state.fable_chapter = next_chapter
                    st.session_state.fable_qc_result = None
                    st.rerun()

            st.markdown("---")
            st.markdown("**أوامر سريعة:**")
            qc_cols = st.columns(4)
            if cur.mode == "قصص إسلامية تربوية":
                quick_labels = ["أضف عبرة", "صف المكان", "أضف حواراً", "لخّص"]
            else:
                quick_labels = ["أنشد بيتاً", "صف المكان", "أضف حواراً", "لخّص"]
            for i, label in enumerate(quick_labels):
                with qc_cols[i]:
                    if st.button(f"⚡ {label}", key=f"fable_qc_{i}", use_container_width=True):
                        with st.spinner("..."):
                            try:
                                qc_result = engine.quick_command(cur.session_id, label)
                                st.session_state.fable_qc_result = (label, qc_result.text, None)
                            except Exception as e:  # noqa: BLE001
                                st.session_state.fable_qc_result = (label, "", str(e))
                        st.rerun()

            _qc = st.session_state.get("fable_qc_result")
            if _qc:
                _qc_label, _qc_text, _qc_err = _qc
                if _qc_err:
                    st.error(f"⚠️ تعذّر تنفيذ «{_qc_label}»، حاول مرة أخرى. (تفصيل تقني: {_qc_err})")
                else:
                    st.markdown(
                        f'<span class="badge badge-blue">⚡ {_qc_label}</span>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"""
                    <div class="root-item" style="text-align:right; direction:rtl">
                        {_qc_text}
                    </div>
                    """, unsafe_allow_html=True)
                    _copy_button(_qc_text, key="fable_qc")

            if st.button("🔄 قصة جديدة"):
                st.session_state.fable_chapter = None
                st.session_state.fable_qc_result = None
                st.rerun()

    # ══════════════════ توليد شعر ══════════════════
    with poem_tab:
        st.markdown("**اطلب قصيدة قصيرة على أحد بحور الشعر العربي:**")
        topic = st.text_input("موضوع القصيدة:", placeholder="مثال: الوفاء، الوطن، الصحراء ليلاً")
        meter = st.selectbox(
            "البحر الشعري",
            list(ARABIC_METERS.keys()),
            format_func=lambda m: f"{m} — {ARABIC_METERS[m]['وصف']}",
        )
        def _run_poem_generation(_topic: str, _meter: str):
            _poem_skel_ph = st.empty()
            with _poem_skel_ph.container():
                _skeleton(lines=5)
            try:
                poem = engine.generate_poem(_topic, meter=_meter)
            except Exception as e:  # noqa: BLE001
                _poem_skel_ph.empty()
                st.session_state.fable_poem_result = None
                st.session_state.fable_poem_error = str(e)
            else:
                _poem_skel_ph.empty()
                st.session_state.fable_poem_result = poem
                st.session_state.fable_poem_error = None
                st.session_state.fable_poem_topic = _topic
                st.session_state.fable_poem_meter = _meter
                st.session_state.fable_poem_audio = None
                st.session_state.fable_poem_audio_error = None

        if st.button("🪶 أنشئ القصيدة", type="primary"):
            if not topic.strip():
                st.warning("⚠️ الرجاء كتابة موضوع القصيدة أولاً.")
            else:
                _run_poem_generation(topic.strip(), meter)

        _poem_err = st.session_state.get("fable_poem_error")
        if _poem_err:
            st.error(f"⚠️ تعذّر توليد القصيدة، حاول مرة أخرى. (تفصيل تقني: {_poem_err})")

        poem = st.session_state.get("fable_poem_result")
        if poem is not None:
            st.toast("✅ القصيدة جاهزة", icon="🪶")
            st.markdown(f"""
            <div class="root-item" style="font-size:1.1rem; line-height:2.1; text-align:center; direction:rtl">
                {poem.text}
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"المزوّد: {poem.provider}")
            _pc1, _pc2, _pc3, _pc4, _pc5 = st.columns(5)
            with _pc1:
                _copy_button(poem.text, key="fable_poem")
            with _pc2:
                st.download_button(
                    "⬇️ تحميل كملف نصي",
                    data=poem.text,
                    file_name="قصيدة.txt",
                    mime="text/plain",
                    key="fable_poem_download",
                    use_container_width=True,
                )
            with _pc5:
                if _PDF_EXPORT_OK:
                    try:
                        _poem_pdf_bytes = _poem_to_pdf(
                            title="قصيدتي", topic=topic, meter=meter, poem_text=poem.text,
                        )
                    except Exception as e:  # noqa: BLE001
                        _poem_pdf_bytes = None
                        st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                    if _poem_pdf_bytes:
                        st.download_button(
                            "📄 تحميل PDF",
                            data=_poem_pdf_bytes,
                            file_name="قصيدتي.pdf",
                            mime="application/pdf",
                            key="fable_poem_pdf_download",
                            use_container_width=True,
                        )
            with _pc3:
                if st.button("🔄 أعد التوليد", key="fable_poem_regenerate", use_container_width=True):
                    _run_poem_generation(
                        st.session_state.get("fable_poem_topic", topic.strip() or "موضوع حر"),
                        st.session_state.get("fable_poem_meter", meter),
                    )
                    st.rerun()
            with _pc4:
                if st.button("🔊 استمع", key="fable_poem_listen", use_container_width=True, disabled=not _TTS_OK):
                    with st.spinner("⟳ جارٍ تحويل القصيدة لصوت..."):
                        try:
                            _poem_tts = _TTSEngineCls().synthesize(poem.text)
                        except Exception as e:  # noqa: BLE001
                            st.session_state.fable_poem_audio = None
                            st.session_state.fable_poem_audio_error = str(e)
                        else:
                            if _poem_tts.ok:
                                import base64 as _b64_poem
                                st.session_state.fable_poem_audio = (
                                    _b64_poem.b64encode(_poem_tts.audio_bytes).decode("ascii"),
                                    _poem_tts.format,
                                )
                                st.session_state.fable_poem_audio_error = None
                            else:
                                st.session_state.fable_poem_audio = None
                                st.session_state.fable_poem_audio_error = _poem_tts.error or "تعذّر توليد الصوت"

            _poem_audio_err = st.session_state.get("fable_poem_audio_error")
            if _poem_audio_err:
                st.error(f"⚠️ تعذّر توليد الصوت. (تفصيل تقني: {_poem_audio_err})")

            _poem_audio = st.session_state.get("fable_poem_audio")
            if _poem_audio:
                _a_b64, _a_fmt = _poem_audio
                st.markdown(
                    f'<audio controls style="width:100%;margin-top:0.5rem" '
                    f'src="data:audio/{_a_fmt};base64,{_a_b64}"></audio>',
                    unsafe_allow_html=True,
                )

    # ══════════════════ وثائقي (سيناريو Explainer) ══════════════════
    with explainer_tab:
        st.markdown(
            '<p style="color:var(--text-muted)">يولّد سيناريو وثائقياً مُقسّماً إلى مشاهد '
            '(نص السرد + توجيه مرئي مقترح لكل مشهد) — فكرة مستوحاة من أدوات '
            'مثل Higgsfield Explainer. <strong>ملاحظة:</strong> NSM لا يملك '
            'نموذج توليد فيديو فعلي، لذا الناتج هنا نص سيناريو فقط جاهز '
            'لتُغذّى به يدوياً أي أداة توليد فيديو خارجية.</p>',
            unsafe_allow_html=True,
        )
        topic = st.text_input(
            "موضوع الوثائقي:",
            placeholder="مثال: تاريخ طريق الحرير، كيف تعمل الأقمار الصناعية",
            key="explainer_topic",
        )
        minutes = st.slider("المدة المستهدفة (دقائق)", min_value=1, max_value=10, value=5)

        if st.button("🎬 أنشئ السيناريو", type="primary"):
            if not topic.strip():
                st.warning("⚠️ الرجاء كتابة موضوع الوثائقي أولاً.")
                st.session_state.explainer_script = None
            else:
                with st.spinner("يُجري بحثاً ويكتب السيناريو..."):
                    try:
                        st.session_state.explainer_script = engine.generate_explainer(
                            topic.strip(), target_minutes=minutes
                        )
                        st.session_state.explainer_error = None
                    except Exception as e:  # noqa: BLE001
                        st.session_state.explainer_script = None
                        st.session_state.explainer_error = str(e)

        _explainer_err = st.session_state.get("explainer_error")
        if _explainer_err:
            st.error(f"⚠️ تعذّر إنشاء السيناريو، حاول مرة أخرى. (تفصيل تقني: {_explainer_err})")

        script = st.session_state.get("explainer_script")
        if script is not None:
            st.markdown(f"### {script.title}")
            st.caption(
                f"عدد المشاهد: {len(script.segments)} · "
                f"إجمالي المدة التقديرية: ~{script.total_seconds // 60} دقيقة "
                f"({script.total_seconds} ثانية) · المزوّد: {script.provider}"
            )
            if script.error:
                st.caption(f"⚠️ ملاحظة تقنية: {script.error}")

            for seg in script.segments:
                st.markdown(f"""
                <div class="root-item" style="text-align:right; direction:rtl">
                    <span class="badge badge-purple">المشهد {seg.index}</span>
                    <span class="badge badge-amber">~{seg.est_seconds} ثانية</span>
                    <p style="margin-top:0.5rem"><strong>السرد:</strong> {seg.narration}</p>
                    <p style="color:var(--text-muted)"><strong>🎥 اللقطة المقترحة:</strong> {seg.visual_notes}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 النص الكامل للسرد (لنسخه إلى أداة التعليق الصوتي)"):
                st.text_area("النص الكامل:", value=script.full_narration, height=200)
                _ec1, _ec2 = st.columns(2)
                with _ec1:
                    st.download_button(
                        "⬇️ تحميل السيناريو كملف نصي",
                        data=script.full_narration,
                        file_name=f"{(script.title or 'سيناريو')[:40]}.txt",
                        mime="text/plain",
                        key="explainer_download",
                        use_container_width=True,
                    )
                with _ec2:
                    if _PDF_EXPORT_OK:
                        try:
                            _explainer_pdf_bytes = _script_to_pdf(
                                title=script.title, format_label=script.format,
                                segments=[
                                    {"index": s.index, "narration": s.narration,
                                     "visual_notes": s.visual_notes, "est_seconds": s.est_seconds}
                                    for s in script.segments
                                ],
                                total_seconds=script.total_seconds,
                            )
                        except Exception as e:  # noqa: BLE001
                            _explainer_pdf_bytes = None
                            st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                        if _explainer_pdf_bytes:
                            st.download_button(
                                "📄 تحميل السيناريو PDF",
                                data=_explainer_pdf_bytes,
                                file_name=f"{(script.title or 'سيناريو')[:40]}.pdf",
                                mime="application/pdf",
                                key="explainer_pdf_download",
                                use_container_width=True,
                            )

    # ══════════════════ ⚡ Shorts (فيديو قصير عمودي) ══════════════════
    with shorts_tab:
        st.markdown(
            '<p style="color:var(--text-muted)">يحوّل نصاً أو موضوعاً إلى فيديو '
            'قصير عمودي (~دقيقة) بسرد صوتي ونص حركي — مع أنماط إبداعية، '
            'مولّد محلي يعمل بدون مفاتيح، ورندر mp4 داخل المشروع.</p>',
            unsafe_allow_html=True,
        )
        # قوالب أفكار سريعة
        st.caption("قوالب سريعة")
        _tpls = [
            "5 حقائق مذهلة عن الفضاء",
            "كيف تبدأ يومك بطاقة؟",
            "قصة نجاح في 60 ثانية",
            "ما لا تعرفه عن الذكاء الاصطناعي",
        ]
        _tc = st.columns(len(_tpls))
        for _i, _tp in enumerate(_tpls):
            with _tc[_i]:
                if st.button(_tp, key=f"shorts_tpl_{_i}", use_container_width=True):
                    st.session_state["shorts_source"] = _tp
                    st.rerun()
        source_text = st.text_area(
            "الصق مصدرك أو اكتب الموضوع:",
            placeholder="مثال: فقرة من مقال، ملخص بحث، أو مجرد فكرة موضوع قصير",
            key="shorts_source",
            height=120,
        )
        c_style, c_sec, c_off = st.columns([2, 1.2, 1])
        with c_style:
            try:
                from ai.fable_engine import SHORTS_STYLES, DEFAULT_SHORTS_STYLE
                _style_labels = [f"{v['emoji']} {k}" for k, v in SHORTS_STYLES.items()]
                _style_keys = list(SHORTS_STYLES.keys())
                _si = _style_keys.index(DEFAULT_SHORTS_STYLE) if DEFAULT_SHORTS_STYLE in _style_keys else 0
                _picked = st.selectbox("النمط الإبداعي", _style_labels, index=_si, key="shorts_style_sel")
                shorts_style = _style_keys[_style_labels.index(_picked)]
            except Exception:
                shorts_style = "حقائق سريعة"
                st.selectbox("النمط الإبداعي", ["حقائق سريعة"], key="shorts_style_sel_fb")
        with c_sec:
            target_sec = st.slider("المدة (ث)", min_value=20, max_value=90, value=60, step=5, key="shorts_sec")
        with c_off:
            force_offline = st.toggle("مولّد محلي", value=False, help="بدون مفاتيح API — إبداع فوري")

        if st.button("⚡ أنشئ سيناريو Shorts", type="primary"):
            if not source_text.strip():
                st.warning("⚠️ الرجاء لصق نص أو كتابة موضوع أولاً.")
            else:
                with st.spinner("يُلخّص ويكتب لقطات سريعة..."):
                    try:
                        st.session_state.shorts_script = engine.generate_short(
                            source_text.strip(),
                            target_seconds=target_sec,
                            style=shorts_style,
                            force_offline=force_offline,
                        )
                        st.session_state.shorts_error = None
                        st.session_state.shorts_mp4 = None
                    except Exception as e:  # noqa: BLE001
                        st.session_state.shorts_script = None
                        st.session_state.shorts_error = str(e)

        _shorts_err = st.session_state.get("shorts_error")
        if _shorts_err:
            st.error(f"⚠️ تعذّر إنشاء سيناريو Shorts، حاول مرة أخرى. (تفصيل تقني: {_shorts_err})")

        short = st.session_state.get("shorts_script")
        if short is not None:
            st.markdown(f"### {short.title}")
            st.caption(
                f"عدد اللقطات: {len(short.segments)} · "
                f"إجمالي المدة التقديرية: ~{short.total_seconds} ثانية · "
                f"المزوّد: {short.provider} · "
                f"النمط: {st.session_state.get('shorts_style_sel', '—')}"
            )
            if short.error:
                st.caption(f"⚠️ ملاحظة تقنية: {short.error}")

            for seg in short.segments:
                st.markdown(f"""
                <div class="root-item" style="text-align:right; direction:rtl">
                    <span class="badge badge-purple">لقطة {seg.index}</span>
                    <span class="badge badge-amber">~{seg.est_seconds} ثانية</span>
                    <p style="margin-top:0.5rem"><strong>السرد:</strong> {seg.narration}</p>
                    <p style="color:var(--text-muted)"><strong>🎞️ رسم متحرك مقترح:</strong> {seg.visual_notes}</p>
                </div>
                """, unsafe_allow_html=True)

            with st.expander("📋 النص الكامل للسرد"):
                st.text_area("النص الكامل:", value=short.full_narration, height=150, key="shorts_full_text")
                _sc1, _sc2 = st.columns(2)
                with _sc1:
                    st.download_button(
                        "⬇️ تحميل السيناريو كملف نصي",
                        data=short.full_narration,
                        file_name=f"{(short.title or 'shorts')[:40]}.txt",
                        mime="text/plain",
                        key="shorts_download",
                        use_container_width=True,
                    )
                with _sc2:
                    if _PDF_EXPORT_OK:
                        try:
                            _shorts_pdf_bytes = _script_to_pdf(
                                title=short.title, format_label=short.format,
                                segments=[
                                    {"index": s.index, "narration": s.narration,
                                     "visual_notes": s.visual_notes, "est_seconds": s.est_seconds}
                                    for s in short.segments
                                ],
                                total_seconds=short.total_seconds,
                            )
                        except Exception as e:  # noqa: BLE001
                            _shorts_pdf_bytes = None
                            st.caption(f"⚠️ تعذّر تجهيز PDF: {e}")
                        if _shorts_pdf_bytes:
                            st.download_button(
                                "📄 تحميل السيناريو PDF",
                                data=_shorts_pdf_bytes,
                                file_name=f"{(short.title or 'shorts')[:40]}.pdf",
                                mime="application/pdf",
                                key="shorts_pdf_download",
                                use_container_width=True,
                            )

            st.divider()
            st.markdown("#### 🎬 رندر الفيديو الفعلي (mp4)")

            _VOICE_OPTIONS = {
                "🎙️ افتراضي (تلقائي حسب المزوّد المتاح)": "",
                "👨 حامد — سعودي (Edge, مجاني)": "ar-SA-HamedNeural",
                "👩 زارية — سعودية (Edge, مجاني)": "ar-SA-ZariyahNeural",
                "👨 شاكر — مصري (Edge, مجاني)": "ar-EG-ShakirNeural",
                "👩 سلمى — مصرية (Edge, مجاني)": "ar-EG-SalmaNeural",
                "👨 حمدان — إماراتي (Edge, مجاني)": "ar-AE-HamdanNeural",
                "👩 فاطمة — إماراتية (Edge, مجاني)": "ar-AE-FatimaNeural",
                "✨ Kore — Gemini TTS (يتطلب GOOGLE_API_KEY)": "Kore",
            }
            selected_voice_label = st.selectbox(
                "🗣️ اختر الصوت",
                options=list(_VOICE_OPTIONS.keys()),
                key="shorts_voice_select",
                help="الأصوات المجانية (Edge) لا تحتاج أي مفتاح API. صوت Gemini يحتاج GOOGLE_API_KEY في البيئة.",
            )
            selected_voice = _VOICE_OPTIONS[selected_voice_label]

            _hf_key_present = bool(os.getenv("HIGGSFIELD_API_KEY", "").strip())
            use_cinematic_bg = st.checkbox(
                "🎥 خلفيات سينمائية حقيقية (بدل التدرّج اللوني الافتراضي)",
                value=False,
                key="shorts_cinematic_bg_toggle",
                help="يستبدل الخلفية المتدرّجة الافتراضية بخلفية فيديو حقيقية لكل مشهد.",
            )
            pro_cols = st.columns(2)
            with pro_cols[0]:
                professional_mode = st.checkbox(
                    "✨ جودة احترافية (موصى بها)",
                    value=True,
                    key="shorts_pro_mode",
                    help="ترميز أعلى (CRF 14) · شريط تقدّم · بطاقة ختامية · انتقالات أنعم · صوت أوضح",
                )
            with pro_cols[1]:
                use_bg_music = st.checkbox(
                    "🎵 سجادة صوتية محيطية خفيفة",
                    value=True,
                    key="shorts_bg_music",
                    help="نغمات محيطية مولَّدة محلياً تحت السرد (ليست أغنية). عطّلها إن فضّلت صمتاً تاماً.",
                )
            cinematic_provider = "higgsfield"
            if use_cinematic_bg:
                _shorts_provider_options = [
                    "🆓 Wan2.1 مجاني ⚡ Running on Zero (GPU حقيقي مجاني)",
                    "💳 Higgsfield (مدفوع — أسرع وأدق، بجودة National Geographic)"
                    + ("" if _hf_key_present else " 🔒"),
                ]
                _shorts_provider_label = st.radio(
                    "المزوّد",
                    options=_shorts_provider_options,
                    index=0,
                    key="shorts_cinematic_provider_radio",
                    horizontal=True,
                    help=(
                        "🆓 Wan2.1 مجاني: يشتغل فعلياً على GPU A100 مجاني عبر "
                        "Hugging Face ZeroGPU (مساحات مُوسومة رسمياً \"Running "
                        "on Zero\" على Hugging Face — ليست محاكاة)، بدون أي "
                        "تكلفة وبدون أي مفتاح إلزامي. أبطأ بكثير (طابور GPU "
                        "مشترك) وقد يتعطّل أحياناً؛ عند فشله يتراجع تلقائياً "
                        "للخلفية المتدرّجة لنفس المشهد فقط. HF_TOKEN اختياري "
                        "لتحسين حد الاستخدام. (يُجرَّب LTX-Video أولاً ثم "
                        "Wan2.2 ثم Wan2.1 تلقائياً حتى ينجح أحدها.)"
                        "\n\n💳 Higgsfield: مزوّد مدفوع، يستهلك رصيدك لكل "
                        "مشهد، أسرع وأدق. يتطلب HIGGSFIELD_API_KEY."
                        + ("" if _hf_key_present else " (🔒 المفتاح غير موجود بالبيئة حالياً)")
                    ),
                )
                cinematic_provider = (
                    "wan_free" if "Wan2.1" in _shorts_provider_label else "higgsfield"
                )
                if cinematic_provider == "wan_free":
                    st.markdown(
                        '<div style="margin:0.3rem 0 0.6rem;">'
                        '<span class="badge badge-green">🟢 Running on Zero</span> '
                        '<span class="badge badge-blue" style="margin-right:6px;">'
                        "GPU A100 مجاني حقيقي — Hugging Face ZeroGPU</span></div>",
                        unsafe_allow_html=True,
                    )
                    _render_wan_free_status_widget("shorts")
                elif not _hf_key_present:
                    st.warning(
                        "⚠️ HIGGSFIELD_API_KEY غير موجود بالبيئة حالياً — أضِفه "
                        "بإعدادات Secrets، أو اختر «🆓 Wan2.1 مجاني» بالأعلى "
                        "لمتابعة العمل بدون أي مفتاح."
                    )

            if st.button("🎬 أنشئ الفيديو الآن", type="primary", key="shorts_render_video_btn"):
                try:
                    _spinner_msg = (
                        "⏳ يولّد السرد الصوتي والخلفيات السينمائية ثم يركّب الفيديو... "
                        "قد يستغرق عدة دقائق"
                        if use_cinematic_bg else
                        "⏳ يولّد السرد الصوتي ثم يركّب الفيديو... قد يستغرق دقيقة"
                    )
                    with st.spinner(_spinner_msg):
                        mp4_bytes = engine.render_video(
                            short, voice=selected_voice,
                            use_cinematic_backgrounds=use_cinematic_bg,
                            cinematic_provider=cinematic_provider,
                            wan_skip_spaces=st.session_state.get("shorts_wan_dead_spaces"),
                            professional_mode=professional_mode,
                            use_background_music=use_bg_music if professional_mode else use_bg_music,
                            music_volume=0.09 if use_bg_music else 0.0,
                        )
                    st.session_state.shorts_mp4 = mp4_bytes
                    st.success("✅ تم إنتاج فيديو Shorts" + (" بجودة احترافية" if professional_mode else ""))
                except Exception as e:  # noqa: BLE001
                    st.error(f"⚠️ فشل رندر الفيديو: {e}")

            mp4_bytes = st.session_state.get("shorts_mp4")
            if mp4_bytes:
                st.video(mp4_bytes)
                _shorts_dl_cols = st.columns(2)
                with _shorts_dl_cols[0]:
                    st.download_button(
                        "⬇️ تحميل الفيديو (mp4)",
                        data=mp4_bytes,
                        file_name=f"{short.title[:40] or 'short'}.mp4",
                        mime="video/mp4",
                        key="shorts_download_mp4",
                    )
                with _shorts_dl_cols[1]:
                    # ── ملف ترجمة SRT — راجع نفس الشرح بتبويب Higgsfield
                    #    Explainer أعلاه (ai.video_engine.build_srt).
                    try:
                        from ai.video_engine import build_srt as _shorts_build_srt
                        _shorts_srt_text = _shorts_build_srt(short)
                        st.download_button(
                            "📝 تحميل الترجمة (SRT)",
                            data=_shorts_srt_text.encode("utf-8"),
                            file_name=f"{short.title[:40] or 'short'}.srt",
                            mime="text/srt",
                            key="shorts_download_srt",
                        )
                    except Exception as _srt_err:  # noqa: BLE001
                        logger.debug("تعذّر بناء ملف SRT لـShorts: %s", _srt_err)

                st.markdown("---")
                st.markdown("#### 📤 مشاركة اجتماعية فعلية (رفع الفيديو)")
                try:
                    from ai.social_platforms import YouTubeAdapter, TikTokAdapter
                except ImportError as e:  # noqa: BLE001
                    st.caption(f"⚠️ تعذّر تحميل محولات المشاركة: {e}")
                else:
                    yt = YouTubeAdapter()
                    tk = TikTokAdapter()
                    share_cols = st.columns(2)

                    # ── يوتيوب ──
                    with share_cols[0]:
                        st.markdown("**▶️ YouTube**")
                        yt_ready = yt.is_configured() and yt._can_write()
                        if not yt_ready:
                            missing = yt.missing_env() or yt.write_env
                            st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(missing))
                        else:
                            yt_title = st.text_input(
                                "العنوان:", value=short.title[:100], key="yt_upload_title"
                            )
                            yt_privacy = st.selectbox(
                                "الخصوصية:", ["private", "unlisted", "public"],
                                key="yt_upload_privacy",
                            )
                            if st.button("▶️ ارفع على يوتيوب", key="yt_upload_btn", use_container_width=True):
                                try:
                                    with st.spinner("⏳ يرفع الفيديو على يوتيوب (Resumable Upload)..."):
                                        video_id = yt.upload_video(
                                            mp4_bytes,
                                            title=yt_title,
                                            description=short.full_narration[:4500],
                                            privacy_status=yt_privacy,
                                        )
                                    st.success(f"✅ تم الرفع! الرابط: https://youtu.be/{video_id}")
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"⚠️ فشل الرفع على يوتيوب: {e}")

                    # ── تيك توك ──
                    with share_cols[1]:
                        st.markdown("**🎵 TikTok**")
                        tk_ready = tk.is_configured()
                        if not tk_ready:
                            st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(tk.missing_env()))
                        else:
                            st.caption(
                                "ℹ️ التطبيقات غير المدقَّقة من TikTok تنشر كـ«خاص بحسابك فقط» "
                                "(مسودة للمراجعة) حتى يجتاز التطبيق مراجعة TikTok الرسمية للنشر العام."
                            )
                            tk_title = st.text_input(
                                "العنوان:", value=short.title[:150], key="tk_upload_title"
                            )
                            if st.button("🎵 ارفع على تيك توك", key="tk_upload_btn", use_container_width=True):
                                try:
                                    with st.spinner("⏳ يرفع الفيديو على تيك توك..."):
                                        publish_id = tk.upload_video(mp4_bytes, title=tk_title)
                                    st.success(
                                        f"✅ تم إرسال الفيديو (publish_id: {publish_id}) — "
                                        "افتح تطبيق TikTok للتأكد من ظهوره ضمن المسودات/المنشورات."
                                    )
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"⚠️ فشل الرفع على تيك توك: {e}")

    # ══════════════════ مكتبة القصص المحفوظة ══════════════════
    # ══════════════════ تعديل فيديو ══════════════════
    with edit_tab:
        try:
            from ui_pages.video_editor_ui import render_video_editor
            render_video_editor()
        except Exception as _ve_err:
            st.error(f"واجهة تعديل الفيديو: {_ve_err}")

    with library_tab:
        st.markdown(
            '<p style="color:var(--text-muted)">كل قصة تفاعلية تُحفظ تلقائياً في قاعدة بيانات SQLite محلية '
            '(<code>memory/fable.db</code>) — هذه الواجهة تستعرضها.</p>',
            unsafe_allow_html=True,
        )

        try:
            sessions = engine.memory.list_recent_sessions(limit=100)
        except Exception as e:  # noqa: BLE001
            sessions = []
            st.error(f"⚠️ تعذّر قراءة مكتبة القصص: {e}")

        if not sessions:
            st.info(
                "📭 لا توجد قصص محفوظة بعد. ابدأ قصة من تبويب «📖 قصة تفاعلية» "
                "وستظهر هنا تلقائياً بمجرد إنشاء الفصل الأول."
            )
        else:
            _lib_modes_present = sorted({s["mode"] for s in sessions if s["mode"]})
            _lib_filter = st.multiselect(
                "🔎 فلترة حسب النمط:",
                options=_lib_modes_present,
                format_func=lambda m: f"{STORY_MODES.get(m, {}).get('emoji', '📖')} {m}",
                key="lib_mode_filter",
                placeholder="كل الأنماط",
            )
            if _lib_filter:
                sessions = [s for s in sessions if s["mode"] in _lib_filter]

            st.caption(f"📚 عدد القصص المعروضة: {len(sessions)}")
            for sess in sessions:
                session_id = sess["session_id"]
                mode = sess["mode"]
                character = sess["character"]
                mode_info = STORY_MODES.get(mode, {})
                char_info = CHARACTERS.get(character, {})
                try:
                    created_label = datetime.fromtimestamp(sess["created_at"]).strftime("%Y-%m-%d %H:%M")
                except Exception:  # noqa: BLE001
                    created_label = ""

                try:
                    preview_text, chapter_count = engine.memory.get_narration_preview(session_id)
                except Exception:  # noqa: BLE001
                    preview_text, chapter_count = "", 0
                preview = (
                    (preview_text[:90] + "…") if preview_text and len(preview_text) > 90
                    else (preview_text or "(لا يوجد نص بعد)")
                )

                header = (
                    f"{mode_info.get('emoji', '📖')} {mode} · "
                    f"{char_info.get('emoji', '')} {character} — {created_label}"
                )
                with st.expander(header):
                    st.caption(f"🆔 {session_id} · عدد الفصول: {chapter_count}")
                    st.markdown(
                        f"<p style='direction:rtl; text-align:right; color:var(--text-muted)'>{preview}</p>",
                        unsafe_allow_html=True,
                    )

                    view_key = f"lib_expand_{session_id}"
                    confirm_key = f"lib_confirm_delete_{session_id}"
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        if st.button("📖 عرض القصة كاملة", key=f"lib_view_btn_{session_id}", use_container_width=True):
                            st.session_state[view_key] = not st.session_state.get(view_key, False)
                    with col_b:
                        if st.button("▶️ استأنف هذه القصة", key=f"lib_resume_btn_{session_id}", use_container_width=True):
                            try:
                                last_narration = engine.memory.get_last_narration(session_id)
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ تعذّر تحميل القصة. (تفصيل تقني: {e})")
                            else:
                                st.session_state.fable_chapter = FableChapter(
                                    session_id=session_id,
                                    text=last_narration,
                                    choices=[],
                                    mode=mode,
                                    character=character,
                                    provider="محفوظ من المكتبة",
                                )
                                st.success("✅ تم تحميل القصة — افتح تبويب «📖 قصة تفاعلية» للمتابعة منها.")
                                st.rerun()
                    with col_c:
                        if st.button("🗑️ حذف", key=f"lib_delete_btn_{session_id}", use_container_width=True):
                            st.session_state[confirm_key] = True

                    if st.session_state.get(confirm_key):
                        st.warning("⚠️ هل أنت متأكد من حذف هذه القصة نهائياً؟ لا يمكن التراجع عن هذا الإجراء.")
                        _dc1, _dc2 = st.columns(2)
                        with _dc1:
                            if st.button("✅ نعم، احذفها نهائياً", key=f"lib_confirm_yes_{session_id}", use_container_width=True):
                                try:
                                    engine.memory.delete_session(session_id)
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"⚠️ تعذّر حذف القصة. (تفصيل تقني: {e})")
                                else:
                                    st.session_state[confirm_key] = False
                                    st.success("✅ تم حذف القصة.")
                                    st.rerun()
                        with _dc2:
                            if st.button("إلغاء", key=f"lib_confirm_no_{session_id}", use_container_width=True):
                                st.session_state[confirm_key] = False
                                st.rerun()

                    if st.session_state.get(view_key):
                        try:
                            history_rows = engine.memory.get_history(session_id, limit=500)
                            narrations = [r["content"] for r in history_rows if r["role"] == "narration"]
                            full_text = "\n\n".join(narrations) if narrations else "(لا يوجد نص محفوظ)"
                        except Exception as e:  # noqa: BLE001
                            full_text = f"⚠️ تعذّر تحميل النص الكامل. (تفصيل تقني: {e})"
                        st.markdown(f"""
                        <div class="root-item" style="text-align:right; direction:rtl; line-height:2">
                            {full_text}
                        </div>
                        """, unsafe_allow_html=True)

        st.divider()
        st.markdown('<div class="section-header">🎬 سيناريوهات Shorts/الوثائقي المحفوظة</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<p style="color:var(--text-muted)">كل سيناريو مولَّد من تبويبَي 🎤 وثائقي و🎬 Shorts '
            'يُحفظ تلقائياً هنا (بدون الصوت/الفيديو) — يمكنك إعادة استخدامه لرندر فيديو جديد '
            'دون توليد سيناريو جديد (يوفّر استدعاء LLM).</p>',
            unsafe_allow_html=True,
        )
        try:
            shorts_history = engine.memory.list_recent_shorts(limit=30)
        except Exception as e:  # noqa: BLE001
            shorts_history = []
            st.error(f"⚠️ تعذّر قراءة سيناريوهات Shorts المحفوظة: {e}")

        if not shorts_history:
            st.info("📭 لا توجد سيناريوهات محفوظة بعد. أنشئ واحداً من تبويب «🎤 وثائقي» أو «🎬 Shorts».")
        else:
            for sh_row in shorts_history:
                sh_id = sh_row["id"]
                sh_emoji = "🎬" if sh_row["format"] == "شورت" else "🎤"
                try:
                    sh_created = datetime.fromtimestamp(sh_row["created_at"]).strftime("%Y-%m-%d %H:%M")
                except Exception:  # noqa: BLE001
                    sh_created = ""
                sh_header = f"{sh_emoji} {sh_row['title']} · ~{sh_row['total_seconds']} ثانية — {sh_created}"
                with st.expander(sh_header):
                    if sh_row["source_excerpt"]:
                        st.caption(f"المصدر: {sh_row['source_excerpt'][:150]}")
                    sh_col_a, sh_col_b = st.columns(2)
                    with sh_col_a:
                        if st.button("📂 استخدم هذا السيناريو", key=f"lib_shorts_use_{sh_id}", use_container_width=True):
                            try:
                                _segs_data = json.loads(sh_row["segments_json"])
                                _rebuilt_segments = [
                                    ExplainerSegment(
                                        index=s["index"], narration=s["narration"],
                                        visual_notes=s["visual_notes"], est_seconds=s["est_seconds"],
                                    ) for s in _segs_data
                                ]
                                st.session_state.shorts_script = ExplainerScript(
                                    topic=sh_row["source_excerpt"], title=sh_row["title"],
                                    segments=_rebuilt_segments, provider="محفوظ من المكتبة",
                                    format=sh_row["format"],
                                )
                                st.session_state.shorts_mp4 = None  # فيديو جديد يحتاج رندر من جديد
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ تعذّر تحميل السيناريو. (تفصيل تقني: {e})")
                            else:
                                st.success("✅ تم تحميل السيناريو — افتح تبويب «🎬 Shorts» لرندر الفيديو.")
                                st.rerun()
                    with sh_col_b:
                        if st.button("🗑️ حذف", key=f"lib_shorts_delete_{sh_id}", use_container_width=True):
                            try:
                                engine.memory.delete_short(sh_id)
                            except Exception as e:  # noqa: BLE001
                                st.error(f"⚠️ تعذّر الحذف. (تفصيل تقني: {e})")
                            else:
                                st.success("✅ تم الحذف.")
                                st.rerun()
