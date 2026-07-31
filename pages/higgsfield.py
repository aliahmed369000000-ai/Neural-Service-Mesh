"""
pages/higgsfield.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_higgsfield():
    """
    تبويب 🎬 Higgsfield Explainer — وثائقي AI حتى 10 دقائق.
    Pipeline (محرك مجاني بالكامل — بدون أي اعتماد على مزوّد مدفوع):
    LLMFallback الداخلي لـNSM (بحث + سرد) → FableEngine.generate_explainer
    (سيناريو مُقسّم مشاهد) → TTSEngine (صوت مجاني: Edge TTS/gTTS، أو Gemini
    TTS إن توفّر مفتاح) → VideoEngine (رندر mp4 فعلي بخلفيات متحركة
    وترجمات Kinetic Captions). خلفيات سينمائية حقيقية عبر Higgsfield تبقى
    متاحة فقط كخيار اختياري (opt-in) معطَّل افتراضياً، تماماً كما في
    تبويب ⚡ Shorts.
    """
    # ── استيراد المحرك (نفس محرك السرد/الفيديو المجاني المستخدم في
    #    تبويب 🎭 إبداع، بدل ai.higgsfield_engine المدفوع) ──────────────
    try:
        from ai.llm_fallback import LLMFallback as _HFLLMFallback
        from ai.fable_engine import FableEngine
    except Exception as _hf_err:
        st.error(f"⚠️ تعذّر تحميل محرك السيناريو/الفيديو: {_hf_err}")
        return

    if "hf_fable_engine" not in st.session_state:
        _hf_fb = _HFLLMFallback(model_key="fable")
        st.session_state.hf_fable_engine = FableEngine(
            llm_fallback=_hf_fb, db_path=str(MEMORY_DIR / "fable.db")
        )
    engine = st.session_state.hf_fable_engine

    # ── رأس الصفحة ────────────────────────────────────────────────────
    st.markdown("""
    <div style="direction:rtl; text-align:right">
        <h2 style="margin-bottom:0.25rem">🎬 Higgsfield Explainer</h2>
        <p style="color:var(--text-muted); font-size:0.95rem; margin-top:0">
            أنشئ فيديو وثائقياً من أي موضوع — حتى 10 دقائق — سيناريو
            وصوت وفيديو mp4 فعلي، <strong>مجاناً بالكامل</strong> (بدون
            أي مفتاح API مدفوع مطلوب).
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    # ── لوحة الإعداد ──────────────────────────────────────────────────
    col_l, col_r = st.columns([2, 1])
    with col_l:
        topic = st.text_input(
            "🎯 موضوع الوثائقي:",
            placeholder="مثال: نشوء الحضارة الإسلامية في الأندلس، كيف تعمل الثقوب السوداء...",
            key="hf_topic",
        )
    with col_r:
        minutes = st.slider(
            "⏱️ المدة المستهدفة (دقائق):",
            min_value=1, max_value=10, value=5,
            key="hf_minutes",
        )

    # ── معلومات Pipeline ───────────────────────────────────────────────
    with st.expander("ℹ️ كيف يعمل الـ Pipeline؟", expanded=False):
        st.markdown("""
        <div style="direction:rtl; text-align:right; font-size:0.9rem">
        <ol>
            <li><strong>🔍 محرك البحث/السرد الداخلي لـNSM</strong> — يبحث
                في المعلومات ويكتب سيناريو المشاهد (نص السرد + توجيه مرئي
                مقترح لكل مشهد)</li>
            <li><strong>🔊 TTSEngine</strong> — يحوّل السرد لصوت فعلي
                (Edge TTS مجاني بدون مفتاح، أو gTTS احتياطياً، أو
                Gemini TTS إن توفّر مفتاح)</li>
            <li><strong>🎬 VideoEngine</strong> — يركّب فيديو mp4 فعلي
                (خلفية متحركة + ترجمات متحركة كلمة-بكلمة) — كل ذلك
                محلياً بدون أي مزوّد خارجي مدفوع</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── زر الإنشاء (السيناريو) ────────────────────────────────────────
    generate_btn = st.button(
        "🎬 أنشئ السيناريو",
        type="primary",
        use_container_width=True,
        disabled=not bool(topic and topic.strip()),
        key="hf_generate_btn",
    )

    if generate_btn:
        if not topic.strip():
            st.warning("أدخل موضوع الوثائقي أولاً.")
        else:
            with st.spinner("⟳ يُجري بحثاً ويكتب السيناريو..."):
                try:
                    st.session_state.hf_script = engine.generate_explainer(
                        topic.strip(), target_minutes=minutes
                    )
                    st.session_state.hf_error = None
                    st.session_state.hf_mp4 = None
                except Exception as e:  # noqa: BLE001
                    # لا نمسح hf_script السابق هنا عمداً: لو كان لدى
                    # المستخدم سيناريو ناجح سابقاً وحاول توليد موضوع جديد
                    # ففشلت المحاولة (شبكة/مزوّد LLM مؤقتاً)، يبقى السيناريو
                    # القديم ظاهراً بدل أن يفقده بلا داعٍ.
                    logger.exception("فشل توليد سيناريو Higgsfield Explainer: %s", e)
                    st.session_state.hf_error = str(e)

    _hf_err = st.session_state.get("hf_error")
    if _hf_err:
        st.error(f"⚠️ تعذّر إنشاء السيناريو، حاول مرة أخرى. (تفصيل تقني: {_hf_err})")

    script = st.session_state.get("hf_script")
    if script is not None:
        _render_hf_result(script)



def _render_hf_result(script):
    """يعرض نتائج Higgsfield Explainer (سيناريو + رندر فيديو مجاني)."""
    segments = script.segments

    # ── ملخص ──────────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("📽️ عدد المشاهد", len(segments))
    total_min = script.total_seconds // 60
    total_sec = script.total_seconds % 60
    c2.metric("⏱️ المدة الإجمالية", f"~{total_min}د {total_sec}ث")
    c3.metric("✍️ مزوّد السرد", script.provider or "—")

    if script.error:
        st.caption(f"⚠️ ملاحظة تقنية: {script.error}")

    st.markdown("---")

    # ── بطاقات المشاهد ────────────────────────────────────────────────
    st.markdown(
        f'<h3 style="direction:rtl; text-align:right">📜 مشاهد الوثائقي — {script.title}</h3>',
        unsafe_allow_html=True,
    )
    _full_script_text = "\n\n".join(
        f"[المشهد {s.index}]\n{s.narration}" for s in segments
    )
    _copy_button(_full_script_text, key="hf_full_script", label="📋 نسخ السيناريو كاملاً")

    for seg in segments:
        with st.expander(
            f"🎬 المشهد {seg.index}  (~{seg.est_seconds}ث)",
            expanded=(seg.index == 1),
        ):
            st.markdown(
                f"""
                <div style="direction:rtl; text-align:right; line-height:1.8">
                <p style="margin-top:0.25rem">
                    <strong>🔊 السرد الصوتي:</strong><br>{seg.narration}
                </p>
                <p style="color:var(--text-muted); font-size:0.9rem">
                    <strong>🎥 التوجيه المرئي:</strong> {seg.visual_notes or "—"}
                </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── تصدير النص الكامل للسرد ──────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 النص الكامل للسرد (للتعليق الصوتي)"):
        st.text_area(
            "نص السرد:",
            value=script.full_narration,
            height=300,
            key="hf_full_narration",
        )
        st.download_button(
            "⬇️ تحميل السيناريو كملف نصي",
            data=script.full_narration,
            file_name=f"{(script.title or 'وثائقي')[:40]}.txt",
            mime="text/plain",
            key="hf_script_download",
        )

    # ── رندر الفيديو الفعلي (mp4) — مجاني بالكامل ─────────────────────
    st.markdown("---")
    st.markdown("#### 🎬 رندر الفيديو الفعلي (mp4) — مجاني")

    _HF_VOICE_OPTIONS = {
        "🎙️ افتراضي (تلقائي حسب المزوّد المتاح)": "",
        "👨 حامد — سعودي (Edge, مجاني)": "ar-SA-HamedNeural",
        "👩 زارية — سعودية (Edge, مجاني)": "ar-SA-ZariyahNeural",
        "👨 شاكر — مصري (Edge, مجاني)": "ar-EG-ShakirNeural",
        "👩 سلمى — مصرية (Edge, مجاني)": "ar-EG-SalmaNeural",
        "👨 حمدان — إماراتي (Edge, مجاني)": "ar-AE-HamdanNeural",
        "👩 فاطمة — إماراتية (Edge, مجاني)": "ar-AE-FatimaNeural",
        "✨ Kore — Gemini TTS (يتطلب GOOGLE_API_KEY)": "Kore",
    }
    _hf_voice_label = st.selectbox(
        "🗣️ اختر الصوت",
        options=list(_HF_VOICE_OPTIONS.keys()),
        key="hf_voice_select",
        help="الأصوات المجانية (Edge) لا تحتاج أي مفتاح API.",
    )
    _hf_voice = _HF_VOICE_OPTIONS[_hf_voice_label]

    _hf_key_present = bool(os.getenv("HIGGSFIELD_API_KEY", "").strip())
    _hf_use_cinematic_bg = st.checkbox(
        "🎥 خلفيات سينمائية حقيقية (اختياري)",
        value=False,
        key="hf_cinematic_bg_toggle",
        help="بدل الخلفية المتدرّجة المجانية الافتراضية، يولّد خلفية فيديو حقيقية لكل مشهد.",
    )
    _hf_cinematic_provider = "higgsfield"
    if _hf_use_cinematic_bg:
        _hf_provider_label = st.radio(
            "المزوّد",
            options=[
                "💳 Higgsfield (مدفوع — أسرع وأدق)",
                "🆓 Wan2.1 مجاني ⚡ Running on Zero (GPU حقيقي مجاني)",
            ],
            key="hf_cinematic_provider_radio",
            horizontal=True,
            help=(
                "Higgsfield: يستهلك رصيدك بالمزوّد، يتطلب HIGGSFIELD_API_KEY."
                + ("" if _hf_key_present else " (المفتاح غير موجود بالبيئة حالياً)")
                + "\n\nWan2.1 مجاني: نموذج مفتوح المصدر يشتغل فعلياً على "
                "GPU A100 مجاني عبر Hugging Face ZeroGPU (مساحات مُوسومة "
                "رسمياً \"Running on Zero\" على Hugging Face — ليست محاكاة) "
                "— أبطأ بكثير (طابور GPU مشترك) وقد يتعطّل أحياناً؛ عند "
                "فشله يتراجع تلقائياً للخلفية المتدرّجة لنفس المشهد فقط. "
                "HF_TOKEN اختياري لتحسين حد الاستخدام."
                "\n\nملاحظة: يُجرَّب LTX-Video أولاً (أسرع)، ثم Wan2.2، ثم "
                "Wan2.1 — تلقائياً وبالترتيب حتى ينجح أحدها."
            ),
        )
        _hf_cinematic_provider = "wan_free" if "Wan2.1" in _hf_provider_label else "higgsfield"
        if _hf_cinematic_provider == "wan_free":
            st.markdown(
                '<div style="margin:0.3rem 0 0.6rem;">'
                '<span class="badge badge-green">🟢 Running on Zero</span> '
                '<span class="badge badge-blue" style="margin-right:6px;">'
                "GPU A100 مجاني حقيقي — Hugging Face ZeroGPU</span></div>",
                unsafe_allow_html=True,
            )
            _render_wan_free_status_widget("hf_explainer")

    _pexels_key_present = bool(os.getenv("PEXELS_API_KEY", "").strip())
    st.caption(
        ("🖼️ صور خلفية حقيقية مجانية (Pexels) مفعَّلة تلقائياً بدل التدرّج اللوني الفارغ."
         if _pexels_key_present else
         "💡 تلميح: أضِف PEXELS_API_KEY (مجاني بالكامل — تسجيل فوري عبر "
         "pexels.com/api) لاستبدال التدرّج اللوني الفارغ بصور خلفية حقيقية "
         "تطابق كل مشهد، بدون أي تكلفة.")
    )

    _hf_use_music = st.checkbox(
        "🎵 موسيقى خلفية هادئة (مجانية، مولَّدة تلقائياً — اختياري)",
        value=False,
        key="hf_bg_music_toggle",
        help=(
            "سجادة صوتية محيطية هادئة بلا لحن أو إيقاع واضح، تُولَّد "
            "داخلياً بدون أي ملف موسيقى خارجي أو مزوّد مدفوع — منخفضة "
            "جداً تحت السرد الصوتي فقط. مُعطَّلة افتراضياً لأن بعض "
            "الجمهور بالمحتوى المعرفي الإسلامي يُفضّل عدم وجود موسيقى "
            "إطلاقاً — فعّلها فقط إن كانت مناسبة لجمهورك."
        ),
    )
    _hf_music_volume = 0.10
    if _hf_use_music:
        _hf_music_volume = st.slider(
            "🔊 حجم الموسيقى النسبي",
            min_value=0.03, max_value=0.25, value=0.10, step=0.01,
            key="hf_bg_music_volume",
            help="منخفض = بالكاد يُلاحَظ تحت السرد. مرتفع = أوضح لكن قد يزاحم الصوت.",
        )

    if st.button("🎬 أنشئ الفيديو الآن", type="primary", key="hf_render_video_btn"):
        try:
            _hf_spinner_msg = (
                "⏳ يولّد السرد الصوتي والخلفيات السينمائية ثم يركّب الفيديو... "
                "قد يستغرق عدة دقائق"
                if _hf_use_cinematic_bg else
                "⏳ يولّد السرد الصوتي ثم يركّب الفيديو... قد يستغرق دقيقة"
            )
            with st.spinner(_hf_spinner_msg):
                engine = st.session_state.hf_fable_engine
                mp4_bytes = engine.render_video(
                    script, voice=_hf_voice,
                    use_cinematic_backgrounds=_hf_use_cinematic_bg,
                    cinematic_provider=_hf_cinematic_provider,
                    use_background_music=_hf_use_music,
                    music_volume=_hf_music_volume,
                    wan_skip_spaces=st.session_state.get("hf_explainer_wan_dead_spaces"),
                )
            st.session_state.hf_mp4 = mp4_bytes
            st.success("✅ تم إنتاج الفيديو")
        except MemoryError:
            # لا نسجّل traceback هنا عمداً — العملية غالباً تكون بالفعل
            # بذاكرة شبه ممتلئة، وتسجيل traceback ثقيل إضافي قد يزيد
            # الضغط سوءاً في هذه اللحظة تحديداً.
            st.error(
                "⚠️ نفدت الذاكرة أثناء الرندر — جرّب مدة أقصر (دقيقتين-3 "
                "بدل 10) أو عطّل «الخلفيات السينمائية الحقيقية» إن كانت مفعّلة."
            )
        except Exception as e:  # noqa: BLE001
            # نسجّل التتبّع الكامل بسجلات السيرفر (يظهر بلوحة Streamlit
            # Cloud logs) — سابقاً كان يُعرَض str(e) فقط للمستخدم وتُفقَد
            # بقية تفاصيل الخطأ نهائياً، ما يصعّب تشخيص أعطال الإنتاج.
            logger.exception("فشل رندر فيديو Higgsfield Explainer: %s", e)
            _err_name = type(e).__name__
            if "Timeout" in _err_name or "timed out" in str(e).lower():
                st.error(
                    "⚠️ انتهت مهلة الانتظار أثناء الرندر (غالباً بسبب جلب "
                    "خلفية سينمائية/صورة خارجية بطيئة الاستجابة) — جرّب "
                    "مرة أخرى، أو عطّل «الخلفيات السينمائية الحقيقية»."
                )
            else:
                st.error(
                    f"⚠️ فشل رندر الفيديو ({_err_name}) — تم تسجيل تفاصيل "
                    f"الخطأ. جرّب مرة أخرى أو بمدة أقصر. (رسالة تقنية: {e})"
                )

    mp4_bytes = st.session_state.get("hf_mp4")
    if mp4_bytes:
        st.video(mp4_bytes)
        _hf_dl_cols = st.columns(2)
        with _hf_dl_cols[0]:
            st.download_button(
                "⬇️ تحميل الفيديو (mp4)",
                data=mp4_bytes,
                file_name=f"{(script.title or 'documentary')[:40]}.mp4",
                mime="video/mp4",
                key="hf_download_mp4",
            )
        with _hf_dl_cols[1]:
            # ── ملف ترجمة SRT منفصل — مبني من نفس بيانات توقيت الترجمات
            #    المستخدمة أصلاً بالفيديو (راجع ai.video_engine.build_srt)،
            #    مفيد لمنصات تتطلب ترجمة منفصلة أو لإتاحة المحتوى لضعاف
            #    السمع. أي فشل بالبناء لا يُسقِط الصفحة — فقط يُخفي الزر.
            try:
                from ai.video_engine import build_srt as _hf_build_srt
                _hf_srt_text = _hf_build_srt(script)
                st.download_button(
                    "📝 تحميل الترجمة (SRT)",
                    data=_hf_srt_text.encode("utf-8"),
                    file_name=f"{(script.title or 'documentary')[:40]}.srt",
                    mime="text/srt",
                    key="hf_download_srt",
                )
            except Exception as _srt_err:  # noqa: BLE001
                logger.debug("تعذّر بناء ملف SRT لـHiggsfield Explainer: %s", _srt_err)

        # ── مشاركة اجتماعية فعلية (رفع الفيديو) ─────────────────────
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

            with share_cols[0]:
                st.markdown("**▶️ YouTube**")
                yt_ready = yt.is_configured() and yt._can_write()
                if not yt_ready:
                    missing = yt.missing_env() or yt.write_env
                    st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(missing))
                else:
                    yt_title = st.text_input(
                        "العنوان:", value=script.title[:100], key="hf_yt_upload_title"
                    )
                    yt_privacy = st.selectbox(
                        "الخصوصية:", ["private", "unlisted", "public"],
                        key="hf_yt_upload_privacy",
                    )
                    if st.button("▶️ ارفع على يوتيوب", key="hf_yt_upload_btn", use_container_width=True):
                        try:
                            with st.spinner("⏳ يرفع الفيديو على يوتيوب (Resumable Upload)..."):
                                video_id = yt.upload_video(
                                    mp4_bytes,
                                    title=yt_title,
                                    description=script.full_narration[:4500],
                                    privacy_status=yt_privacy,
                                )
                            st.success(f"✅ تم الرفع! الرابط: https://youtu.be/{video_id}")
                        except Exception as e:  # noqa: BLE001
                            st.error(f"⚠️ فشل الرفع على يوتيوب: {e}")

            with share_cols[1]:
                st.markdown("**🎵 TikTok**")
                tk_ready = tk.is_configured()
                if not tk_ready:
                    st.caption("⚙️ غير مُهيّأ — أضِف بالبيئة (Secrets): " + "، ".join(tk.missing_env()))
                else:
                    st.caption(
                        "ℹ️ التطبيقات غير المدقَّقة من TikTok تنشر كـ«خاص بحسابك فقط» "
                        "(مسودة للمراجعة) حتى يجتاز التطبيق مراجعة TikTok الرسمية."
                    )
                    tk_title = st.text_input(
                        "العنوان:", value=script.title[:150], key="hf_tk_upload_title"
                    )
                    if st.button("🎵 ارفع على تيك توك", key="hf_tk_upload_btn", use_container_width=True):
                        try:
                            with st.spinner("⏳ يرفع الفيديو على تيك توك..."):
                                publish_id = tk.upload_video(mp4_bytes, title=tk_title)
                            st.success(
                                f"✅ تم إرسال الفيديو (publish_id: {publish_id}) — "
                                "افتح تطبيق TikTok للتأكد من ظهوره ضمن المسودات/المنشورات."
                            )
                        except Exception as e:  # noqa: BLE001
                            st.error(f"⚠️ فشل الرفع على تيك توك: {e}")
