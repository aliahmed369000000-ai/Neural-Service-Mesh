"""
pages/advanced_api.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ═══════════════════════════════════════════════════════════════════════════
# تبويب API متقدمة
# ═══════════════════════════════════════════════════════════════════════════

def render_advanced_api():
    """تبويب API متقدمة — Web Search · تحليل الصور · JSON منظّم"""

    st.markdown('<div class="section-header">🔬 API متقدمة — Anthropic Claude</div>', unsafe_allow_html=True)

    # ── فحص توفّر المفتاح ────────────────────────────────────────────────
    try:
        from ai.anthropic_advanced import AnthropicAdvanced
        from ai.llm_fallback import ANTHROPIC_MODELS
        _test_client = AnthropicAdvanced()
        _has_key = _test_client.available
    except Exception as _imp_err:
        st.error(f"⚠️ تعذّر تحميل وحدة API المتقدمة: {_imp_err}")
        return

    if not _has_key:
        st.warning(
            "🔑 **ANTHROPIC_API_KEY غير موجود** — أضفه في Secrets لتفعيل هذا التبويب.\n\n"
            "الأدوات المتاحة هنا: Web Search · تحليل الصور · استخراج JSON منظّم"
        )
        st.info("💡 بعد إضافة المفتاح، اضغط **R** لإعادة تشغيل التطبيق.")
        return

    # ── اختيار النموذج ────────────────────────────────────────────────────
    st.markdown("#### ⚙️ إعدادات")
    col_m, col_t = st.columns([2, 1])
    with col_m:
        model_choice = st.selectbox(
            "النموذج",
            options=list(ANTHROPIC_MODELS.values()),
            index=0,
            format_func=lambda m: {
                "claude-sonnet-4-6":         "⚡ Sonnet 4-6 (الافتراضي)",
                "claude-opus-4-8":           "💎 Opus 4-8 (الأقوى)",
                "claude-haiku-4-5-20251001": "🚀 Haiku 4-5 (الأسرع)",
                "claude-sonnet-4-20250514":  "🔒 Sonnet Stable",
            }.get(m, m),
            key="adv_model",
        )
    with col_t:
        max_tokens = st.slider("الحد الأقصى للتوكنات", 256, 2048, 800, 128, key="adv_max_tokens")

    client = AnthropicAdvanced(model=model_choice, max_tokens=max_tokens)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    # الأقسام الثلاثة
    # ══════════════════════════════════════════════════════════════════════
    sec1, sec2, sec3, sec4 = st.tabs(
        ["🌐 بحث الويب", "🖼️ تحليل الصور", "📐 JSON منظّم", "🔌 MCP Servers"]
    )

    # ────────────────────────────────────────────────────────────────────
    # القسم 1 — Web Search Tool
    # ────────────────────────────────────────────────────────────────────
    with sec1:
        st.markdown("""
        <div style="background:color-mix(in srgb, #38bdf8 12%, var(--surface2));border:1px solid color-mix(in srgb, #38bdf8 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🌐 Web Search Tool</strong><br>
            <small>يُفعّل أداة البحث في الويب المدمجة في Anthropic API —
            النموذج يقرر بنفسه متى وكيف يبحث ثم يدمج النتائج في إجابته.</small>
        </div>
        """, unsafe_allow_html=True)

        ws_query = st.text_area(
            "سؤالك (سيبحث النموذج في الويب تلقائياً)",
            placeholder="مثال: ما آخر إصدارات نماذج Anthropic Claude؟\nأو: ما أحدث أخبار الذكاء الاصطناعي اليوم؟",
            height=100, key="ws_query",
        )
        ws_system = st.text_input(
            "تعليمات النظام (اختياري)",
            value="أجب بالعربية الفصحى بشكل مختصر ومنظّم.",
            key="ws_system",
        )

        if st.button("🔍 ابحث وأجب", key="ws_run", use_container_width=True, type="primary"):
            if not ws_query.strip():
                st.warning("أدخل سؤالك أولاً.")
            else:
                with st.spinner("⟳ يبحث النموذج في الويب..."):
                    result = client.ask_with_search(ws_query.strip(), system=ws_system.strip())

                if result.error:
                    st.error(f"❌ خطأ: {result.error}")
                else:
                    st.markdown("#### 📝 الإجابة")
                    st.markdown(f"""
                    <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                                padding:1rem 1.4rem;direction:rtl;line-height:1.9;
                                white-space:pre-wrap;font-size:0.97rem">
                    {result.text or "لا توجد إجابة نصية."}
                    </div>
                    """, unsafe_allow_html=True)

                    if result.tool_calls:
                        with st.expander(f"🔧 أدوات استُخدمت ({len(result.tool_calls)})"):
                            for tc in result.tool_calls:
                                st.json(tc)

                    if result.tool_results:
                        with st.expander(f"📦 نتائج البحث الخام ({len(result.tool_results)})"):
                            for tr in result.tool_results:
                                st.text(tr[:800])

                    cols = st.columns(3)
                    cols[0].metric("نموذج", result.model.split("-")[-1] if result.model else "—")
                    cols[1].metric("زمن الاستجابة", f"{result.latency_ms:.0f} ms")
                    cols[2].metric("توكنات الإخراج", result.output_tokens)

    # ────────────────────────────────────────────────────────────────────
    # القسم 2 — تحليل الصور
    # ────────────────────────────────────────────────────────────────────
    with sec2:
        st.markdown("""
        <div style="background:color-mix(in srgb, #c084fc 14%, var(--surface2));border:1px solid color-mix(in srgb, #c084fc 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🖼️ تحليل الصور</strong><br>
            <small>ارفع صورة (JPEG · PNG · GIF · WebP) واطرح سؤالاً عنها —
            النموذج سيحلّلها ويجيب بالعربية.</small>
        </div>
        """, unsafe_allow_html=True)

        img_file = st.file_uploader(
            "ارفع صورة", type=["jpg", "jpeg", "png", "gif", "webp"], key="img_upload"
        )
        img_question = st.text_area(
            "سؤالك عن الصورة",
            placeholder="مثال: صِف ما تراه في هذه الصورة.\nأو: هل تحتوي على نص؟ اقرأه.",
            height=90, key="img_question",
        )

        if img_file:
            st.image(img_file, caption="الصورة المرفوعة", use_container_width=False, width=350)

        if st.button("🔍 حلّل الصورة", key="img_run", use_container_width=True, type="primary"):
            if not img_file:
                st.warning("ارفع صورة أولاً.")
            elif not img_question.strip():
                st.warning("أدخل سؤالك أولاً.")
            else:
                mime_map = {
                    "jpg": "image/jpeg", "jpeg": "image/jpeg",
                    "png": "image/png", "gif": "image/gif", "webp": "image/webp",
                }
                ext = img_file.name.rsplit(".", 1)[-1].lower()
                media_type = mime_map.get(ext, "image/jpeg")
                img_bytes = img_file.read()

                with st.spinner("⟳ يحلّل النموذج الصورة..."):
                    answer = client.ask_with_image(
                        img_question.strip(), img_bytes, media_type,
                        system="أجب بالعربية الفصحى.",
                    )

                st.markdown("#### 📝 تحليل النموذج")
                st.markdown(f"""
                <div style="background:var(--surface2);color:var(--text);border-radius:10px;
                            padding:1rem 1.4rem;direction:rtl;line-height:1.9;
                            white-space:pre-wrap;font-size:0.97rem">
                {answer or "لم يُنتج النموذج إجابة."}
                </div>
                """, unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────────────────
    # القسم 3 — JSON منظّم
    # ────────────────────────────────────────────────────────────────────
    with sec3:
        st.markdown("""
        <div style="background:color-mix(in srgb, #34d399 14%, var(--surface2));border:1px solid color-mix(in srgb, #34d399 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>📐 استخراج JSON منظّم</strong><br>
            <small>اطلب من النموذج إجابة JSON خالصة — مناسب لاستخراج البيانات
            وتحليل النصوص وبناء APIs.</small>
        </div>
        """, unsafe_allow_html=True)

        json_query = st.text_area(
            "طلبك",
            placeholder="مثال: استخرج من النص التالي: الاسم والعمر والمهنة.\nأو: أعطني قائمة بأسماء الخلفاء الراشدين مع تواريخ خلافتهم.",
            height=110, key="json_query",
        )
        json_schema = st.text_input(
            "وصف البنية المطلوبة (اختياري)",
            placeholder='مثال: { "name": "string", "year": "number" }',
            key="json_schema",
        )

        if st.button("⚙️ استخرج JSON", key="json_run", use_container_width=True, type="primary"):
            if not json_query.strip():
                st.warning("أدخل طلبك أولاً.")
            else:
                with st.spinner("⟳ يولّد النموذج JSON..."):
                    data = client.ask_json(
                        json_query.strip(),
                        json_schema_hint=json_schema.strip(),
                    )

                if data is None:
                    st.error("❌ فشل تحليل JSON — قد لا يدعم النموذج هذا الطلب بصيغة JSON خالصة.")
                    raw_text = client.ask(json_query.strip())
                    if raw_text:
                        st.markdown("**الرد الخام:**")
                        st.code(raw_text, language="text")
                else:
                    st.success("✅ JSON مُستخرَج بنجاح")
                    st.json(data)

                    import json as _json
                    json_str = _json.dumps(data, ensure_ascii=False, indent=2)
                    st.download_button(
                        "⬇️ تحميل JSON",
                        data=json_str,
                        file_name="nsm_output.json",
                        mime="application/json",
                        key="json_download",
                    )

    # ────────────────────────────────────────────────────────────────────
    # القسم 4 — MCP Servers (Model Context Protocol)
    # ────────────────────────────────────────────────────────────────────
    with sec4:
        st.markdown("""
        <div style="background:color-mix(in srgb, #f87171 14%, var(--surface2));border:1px solid color-mix(in srgb, #f87171 40%, var(--border));border-radius:10px;
                    padding:0.9rem 1.2rem;direction:rtl;margin-bottom:1rem;color:var(--text)">
            <strong>🔌 MCP Servers (Model Context Protocol)</strong><br>
            <small>يتصل النموذج مباشرة بخوادم MCP بعيدة (Google Drive، Gmail، Google
            Calendar، Canva، Figma، أو أي خادم MCP آخر) وينفّذ أدواتها الفعلية أثناء
            توليد الرد. يتطلب أن يكون الحساب المرتبط مصرّحاً (OAuth) لكل خادم حسب
            سياسته الخاصة.</small>
        </div>
        """, unsafe_allow_html=True)

        MCP_PRESETS = {
            "Google Drive":   "https://drivemcp.googleapis.com/mcp/v1",
            "Gmail":          "https://gmailmcp.googleapis.com/mcp/v1",
            "Google Calendar": "https://calendarmcp.googleapis.com/mcp/v1",
            "Canva":          "https://mcp.canva.com/mcp",
            "Figma":          "https://mcp.figma.com/mcp",
        }
        mcp_chosen = st.multiselect(
            "اختر خوادم MCP جاهزة للتفعيل",
            options=list(MCP_PRESETS.keys()),
            key="mcp_servers_choice",
        )
        mcp_custom_url = st.text_input(
            "أو أضف رابط خادم MCP مخصّص (اختياري)",
            placeholder="https://example.com/mcp",
            key="mcp_custom_url",
        )
        mcp_query = st.text_area(
            "سؤالك/طلبك",
            placeholder="مثال: لخّص آخر ملف في Google Drive باسم يحتوي 'تفسير'.",
            height=110, key="mcp_query",
        )

        if st.button("🔌 نفّذ عبر MCP", key="mcp_run", use_container_width=True, type="primary"):
            servers = [
                {"type": "url", "url": MCP_PRESETS[name], "name": name}
                for name in mcp_chosen
            ]
            if mcp_custom_url.strip():
                servers.append({"type": "url", "url": mcp_custom_url.strip(), "name": "مخصّص"})

            if not mcp_query.strip():
                st.warning("أدخل سؤالك أولاً.")
            elif not servers:
                st.warning("اختر خادم MCP واحداً على الأقل أو أضف رابطاً مخصصاً.")
            else:
                with st.spinner("⟳ يتصل بخوادم MCP..."):
                    mcp_result = client.ask_with_mcp(mcp_query.strip(), servers)

                if mcp_result.error:
                    st.error(f"❌ {mcp_result.error}")
                else:
                    st.success("✅ تم")
                    if mcp_result.text:
                        st.markdown(mcp_result.text)
                    if mcp_result.tool_calls:
                        with st.expander(f"🔧 استدعاءات الأدوات ({len(mcp_result.tool_calls)})"):
                            for tc in mcp_result.tool_calls:
                                st.json(tc)
                    if mcp_result.tool_results:
                        with st.expander(f"📄 نتائج الأدوات ({len(mcp_result.tool_results)})"):
                            for tr in mcp_result.tool_results:
                                st.code(tr[:2000])

    # ── ملاحظة ختامية ───────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "هذه الأدوات تستخدم `ai/anthropic_advanced.py` — مستخلصة من Claude.ai System Prompt (That.md). "
        "كل استدعاء يُرسَل مباشرة إلى Anthropic API."
    )
