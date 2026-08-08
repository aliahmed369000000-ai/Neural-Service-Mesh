"""
ui_pages/video_editor_ui.py — واجهة أدوات تعديل الفيديو
"""
from __future__ import annotations

from pathlib import Path

from app_core import *  # noqa: F401,F403


def render_video_editor():
    st.markdown(
        '<div class="section-header">✂️ أدوات تعديل الفيديو</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "قص · دمج · كتم · استخراج صوت · 9:16 للشورتس · سرعة · ضغط · صورة مصغّرة — "
        "بدون أوامر shell حرة."
    )

    try:
        from ai import video_editor as ve
    except Exception as e:
        st.error(f"تعذّر تحميل ai/video_editor.py: {e}")
        return

    tools = ve.available_tools()
    c1, c2, c3 = st.columns(3)
    c1.metric("ffmpeg", "✅" if tools["ffmpeg"] else "❌")
    c2.metric("ffprobe", "✅" if tools["ffprobe"] else "❌")
    c3.metric("moviepy", "✅" if tools["moviepy"] else "❌")
    if not tools["ffmpeg"] and not tools["moviepy"]:
        st.warning("ثبّت ffmpeg أو moviepy لتفعيل الأدوات.")
        return

    uploaded = st.file_uploader(
        "ارفع فيديو (mp4/mov/webm)",
        type=["mp4", "mov", "webm", "mkv"],
        key="vedit_upload",
    )
    # خيار استخدام آخر Shorts من الجلسة
    use_session = False
    if st.session_state.get("shorts_mp4"):
        use_session = st.checkbox("استخدم آخر فيديو Shorts من هذه الجلسة", value=False)

    work_path = None
    if use_session and st.session_state.get("shorts_mp4"):
        tmp = Path("/tmp") / f"nsm_session_short_{id(st.session_state)}.mp4"
        tmp.write_bytes(st.session_state["shorts_mp4"])
        work_path = tmp
    elif uploaded is not None:
        tmp = Path("/tmp") / f"nsm_upload_{uploaded.name}"
        tmp.write_bytes(uploaded.getvalue())
        work_path = tmp

    if work_path is None:
        st.info("ارفع ملفاً أو أنشئ Shorts أولاً ثم عد هنا.")
        return

    st.video(str(work_path))

    if st.button("ℹ️ معلومات الفيديو", key="vedit_probe"):
        try:
            info = ve.probe(work_path)
            st.markdown(ve.format_probe_report(info))
            st.session_state["vedit_last_info"] = info
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    op = st.selectbox(
        "العملية",
        [
            "✨ تحسين ذكي بالذكاء",
            "📐 رفع الدقة (Upscale)",
            "🎯 إعادة ترميز عالية الجودة (ffmpeg)",
            "قص (trim)",
            "تحجيم عمودي 9:16",
            "كتم الصوت",
            "استخراج صوت",
            "تغيير السرعة",
            "ضغط",
            "صورة مصغّرة",
            "دمج مع فيديو ثانٍ",
        ],
        key="vedit_op",
    )

    result_path = None
    try:
        if op == "📐 رفع الدقة (Upscale)":
            target = st.selectbox(
                "الهدف",
                ["1080p", "2x", "720p", "1440p", "4k", "shorts"],
                index=0,
                format_func=lambda x: {
                    "2x": "مضاعفة ×2 (نسب أصلية)",
                    "720p": "1280×720",
                    "1080p": "Full HD 1920×1080",
                    "1440p": "QHD 2560×1440",
                    "4k": "4K 3840×2160",
                    "shorts": "شورتس عمودي 1080×1920",
                }[x],
                key="vedit_up_target",
            )
            crf = st.slider("CRF (أقل = أوضح)", 14, 22, 16, key="vedit_up_crf")
            use_ai = st.checkbox(
                "استخدم نماذج AI (Real-ESRGAN مجاني عبر Hugging Face)",
                value=False,
                key="vedit_up_ai",
                help="أفضل للفيديو القصير ≤12 ثانية. يحتاج شبكة وقد ينتظر طابور ZeroGPU. عند الفشل يتراجع لـ Lanczos تلقائياً.",
            )
            st.caption(
                "محلي: تنظيف → Lanczos → حدة. "
                "مع AI: Real-ESRGAN على الإطارات ثم تجميع (قصير فقط)."
            )
            if st.button("رفع الدقة الآن", type="primary", key="vedit_up_run"):
                with st.spinner(
                    "يرفع الدقة بنماذج AI…" if use_ai else "يرفع الدقة محلياً…"
                ):
                    result_path = ve.upscale(
                        work_path, target=target, crf=crf, use_ai=use_ai
                    )

        elif op == "🎯 إعادة ترميز عالية الجودة (ffmpeg)":
            level = st.selectbox(
                "مستوى الجودة",
                ["archive", "high", "balanced"],
                index=1,
                format_func=lambda x: {
                    "archive": "أرشفة (CRF 14, أبطأ)",
                    "high": "عالية (CRF 16)",
                    "balanced": "متوازنة (CRF 18)",
                }[x],
                key="vedit_q_level",
            )
            visual = st.selectbox(
                "تحسين بصري",
                ["soft", "strong", "none"],
                index=0,
                format_func=lambda x: {"soft": "خفيف", "strong": "قوي", "none": "ترميز فقط"}[x],
                key="vedit_q_vis",
            )
            if st.button("تطبيق جودة ffmpeg", type="primary", key="vedit_qboost"):
                with st.spinner("إعادة ترميز…"):
                    result_path = ve.quality_boost(work_path, level=level, visual=visual)

        elif op == "✨ تحسين ذكي بالذكاء":
            try:
                from ai.video_ai_enhance import list_presets, format_presets_help
                st.markdown(format_presets_help())
                presets = list_presets()
                labels = ["تلقائي (auto)", "احترافي (pro)"] + [
                    f"{p['label']} ({p['id']})" for p in presets
                ]
                ids = ["auto", "pro"] + [p["id"] for p in presets]
                pick = st.selectbox("وضع التحسين", labels, index=0, key="vedit_ai_mode")
                mode = ids[labels.index(pick)]
                crf = st.slider("جودة الترميز CRF", 14, 24, 17, key="vedit_ai_crf")
            except Exception as e:
                st.warning(str(e))
                mode, crf = "clarity", 17
            prefer_ai = st.checkbox(
                "جرّب نموذج AI على إطار عيّنة (Real-ESRGAN)",
                value=False,
                key="vedit_ai_hf",
            )
            if st.button("تطبيق التحسين الذكي", type="primary", key="vedit_ai_run"):
                with st.spinner("يحسّن الفيديو…"):
                    result_path = ve.ai_enhance(
                        work_path, mode=mode, crf=crf, prefer_hf=prefer_ai
                    )

        elif op == "قص (trim)":
            c_a, c_b = st.columns(2)
            with c_a:
                start = st.number_input("من (ث)", min_value=0.0, value=0.0, step=0.5)
            with c_b:
                end = st.number_input("إلى (ث)", min_value=0.0, value=5.0, step=0.5)
            if st.button("تطبيق القص", type="primary"):
                with st.spinner("قص…"):
                    result_path = ve.trim(work_path, start=start, end=end if end > start else None)

        elif op == "تحجيم عمودي 9:16":
            if st.button("تحويل للشورتس 1080×1920", type="primary"):
                with st.spinner("تحجيم…"):
                    result_path = ve.to_shorts_vertical(work_path)

        elif op == "كتم الصوت":
            if st.button("كتم", type="primary"):
                with st.spinner("كتم…"):
                    result_path = ve.mute(work_path)

        elif op == "استخراج صوت":
            fmt = st.selectbox("الصيغة", ["mp3", "wav", "m4a"])
            if st.button("استخراج", type="primary"):
                with st.spinner("استخراج…"):
                    result_path = ve.extract_audio(work_path, fmt=fmt)

        elif op == "تغيير السرعة":
            factor = st.slider("المعامل", 0.5, 2.0, 1.25, 0.05)
            if st.button("تطبيق السرعة", type="primary"):
                with st.spinner("معالجة…"):
                    result_path = ve.change_speed(work_path, factor=factor)

        elif op == "ضغط":
            crf = st.slider("CRF (أعلى = أصغر حجماً)", 18, 32, 28)
            if st.button("ضغط", type="primary"):
                with st.spinner("ضغط…"):
                    result_path = ve.compress(work_path, crf=crf)

        elif op == "صورة مصغّرة":
            at = st.number_input("عند الثانية", min_value=0.0, value=1.0, step=0.5)
            if st.button("استخراج إطار", type="primary"):
                with st.spinner("إطار…"):
                    result_path = ve.thumbnail(work_path, at_seconds=at)

        elif op == "دمج مع فيديو ثانٍ":
            up2 = st.file_uploader("الفيديو الثاني", type=["mp4", "mov", "webm"], key="vedit_up2")
            if up2 and st.button("دمج", type="primary"):
                p2 = Path("/tmp") / f"nsm_upload2_{up2.name}"
                p2.write_bytes(up2.getvalue())
                with st.spinner("دمج…"):
                    result_path = ve.concat([work_path, p2])

    except Exception as e:
        st.error(f"فشلت العملية: {e}")
        result_path = None

    if result_path is not None and Path(result_path).is_file():
        st.success(f"✅ تم: `{result_path}`")
        suffix = Path(result_path).suffix.lower()
        data = Path(result_path).read_bytes()
        if suffix in (".mp4", ".webm", ".mov"):
            st.video(str(result_path))
        elif suffix in (".jpg", ".jpeg", ".png"):
            st.image(str(result_path))
        elif suffix in (".mp3", ".wav", ".m4a", ".aac"):
            st.audio(str(result_path))
        st.download_button(
            "⬇️ تحميل الناتج",
            data=data,
            file_name=Path(result_path).name,
            mime="application/octet-stream",
            key="vedit_dl",
        )
        # إتاحة استخدام الناتج كمصدر Shorts لاحق
        if suffix == ".mp4":
            if st.button("استخدم الناتج كـ Shorts الجلسة"):
                st.session_state["shorts_mp4"] = data
                st.success("تم حفظه في جلسة Shorts")
