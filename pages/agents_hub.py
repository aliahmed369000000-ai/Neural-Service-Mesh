"""
pages/agents_hub.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



def render_agents_hub():
    """يعرض تبويباً فرعياً مستقلاً لكل فئة من وكلاء الذكاء الاصطناعي المتخصصين."""

    if not _AGENTS_HUB_OK:
        st.error("⚠️ تعذّر تحميل وكلاء AI. تأكد من وجود ai/agent_categories.py.")
        return

    st.markdown("### 🤖 وكلاء AI المتخصصون")
    st.caption("كل فئة لها وكيلها الخاص، بذاكرة محادثة مستقلة، ومزوّد LLM نفسه المُستخدَم في المشروع.")

    # CSS مشترك لكل فقاعات المحادثة داخل هذا التبويب (نفس أسلوب تبويب المحادثة)
    if not st.session_state.get("_nsm_agents_hub_css_injected"):
        st.session_state["_nsm_agents_hub_css_injected"] = True
        st.markdown("""
    <style>
    @keyframes agentBubbleIn {
        from {opacity:0;transform:translateY(6px);}
        to   {opacity:1;transform:translateY(0);}
    }
    .agent-user {display:flex;justify-content:flex-end;margin:0.5rem 0;animation:agentBubbleIn .25s ease-out;}
    .agent-user .bbl {
        background:linear-gradient(135deg,var(--gold),var(--emerald));
        color:var(--bg);padding:0.7rem 1.05rem;border-radius:18px 18px 4px 18px;
        max-width:85%;font-size:0.96rem;line-height:1.7;text-align:right;direction:rtl;
        box-shadow:0 3px 12px var(--shadow);white-space:pre-wrap;word-break:break-word;
        font-weight:600;
    }
    .agent-bot {display:flex;justify-content:flex-start;margin:0.5rem 0;gap:0.5rem;align-items:flex-start;animation:agentBubbleIn .25s ease-out;}
    .agent-bot .bbl {
        background:var(--surface2);
        color:var(--text);padding:0.7rem 1.05rem;border-radius:18px 18px 18px 4px;
        max-width:85%;font-size:0.96rem;line-height:1.8;text-align:right;direction:rtl;
        border:1px solid var(--border);box-shadow:0 2px 8px var(--shadow);
        white-space:pre-wrap;word-break:break-word;
    }
    .agent-box {
        height:48vh;min-height:320px;max-height:520px;overflow-y:auto;padding:1rem;
        background:var(--bg);border-radius:16px;border:1px solid var(--border);margin-bottom:0.8rem;
        scroll-behavior:smooth;box-shadow:inset 0 0 20px var(--shadow);
    }
    .agent-badge {
        display:inline-block;background:var(--surface);border:1px solid var(--border);border-radius:20px;
        padding:0.15rem 0.65rem;font-size:0.72rem;color:var(--gold);direction:rtl;
    }
    </style>
    """, unsafe_allow_html=True)

    labels = [
        f"{AGENT_CATEGORIES[k].emoji} {AGENT_CATEGORIES[k].title}" for k in CATEGORY_ORDER
    ]
    sub_tabs = st.tabs(labels)

    for i, key in enumerate(CATEGORY_ORDER):
        with sub_tabs[i]:
            _render_agent_page(AGENT_CATEGORIES[key])



def _render_agent_page(category):
    """يعرض صفحة وكيل واحد: محادثة معزولة + أسئلة سريعة خاصة بفئته."""
    import html as _html

    bot_key  = f"agent_bot_{category.key}"
    msg_key  = f"agent_msgs_{category.key}"
    cnt_key  = f"agent_count_{category.key}"

    if bot_key not in st.session_state:
        st.session_state[bot_key] = CategoryAgentChat(category.key)
        st.session_state[msg_key] = []
        st.session_state[cnt_key] = 0

    bot = st.session_state[bot_key]

    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.markdown(f"#### {category.emoji} {category.title}")
        st.caption(category.subtitle)
    with col_s:
        st.metric("رسائل الجلسة", st.session_state[cnt_key])

    web_toggle = st.toggle(
        "🌐 بحث حقيقي في الويب قبل الرد",
        value=getattr(category, "web_enabled", False),
        key=f"agent_web_{category.key}",
        help="يفعّل بحثاً فعلياً عبر DuckDuckGo قبل توليد الرد، بغض النظر عن الفئة.",
    )

    box_id = f"agent-chat-box-{category.key}"
    html_out = f'<div class="agent-box" id="{box_id}">'
    if not st.session_state[msg_key]:
        html_out += (
            f'<div style="text-align:center;color:var(--text-muted);padding:2rem 1rem">'
            f'{category.emoji}<br><br>ابدأ محادثتك مع وكيل {category.title}</div>'
        )
    else:
        for _mi, msg_tuple in enumerate(st.session_state[msg_key]):
            role, text, badge = msg_tuple[0], msg_tuple[1], msg_tuple[2]
            ts = msg_tuple[3] if len(msg_tuple) > 3 else ""
            ts_html = f'<div class="bbl-ts">{ts}</div>' if ts else ""
            safe = _html.escape(text).replace("\n", "<br>")
            if role == "user":
                html_out += f'<div class="agent-user"><div class="bbl">{safe}{ts_html}</div></div>'
            else:
                badge_html = f'<div class="agent-badge">{badge}</div>' if badge else ""
                bbl_id = f"{box_id}-msg-{_mi}"
                html_out += (
                    f'<div class="agent-bot"><span style="font-size:1.3rem;margin-top:3px">'
                    f'{category.emoji}</span><div class="bbl">{badge_html}'
                    f'<div id="{bbl_id}">{safe}</div>'
                    f'<button class="copy-btn" title="نسخ الرد" style="margin-top:0.4rem"'
                    f' onclick="var t=document.getElementById(\'{bbl_id}\').innerText;'
                    f"navigator.clipboard.writeText(t).then(function(){{"
                    f"var b=event.currentTarget;var old=b.textContent;b.textContent='✓ تم النسخ';"
                    f"setTimeout(function(){{b.textContent=old;}},1300);}});\">📋 نسخ</button>"
                    f'{ts_html}</div></div>'
                )
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)
    st.components.v1.html(f"""
    <script>
    (function() {{
        function scrollToBottom() {{
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('{box_id}');
            if (box) {{ box.scrollTop = box.scrollHeight; return true; }}
            return false;
        }}
        let attempts = 0;
        const tryScroll = () => {{
            attempts++;
            if (!scrollToBottom() && attempts < 10) {{ setTimeout(tryScroll, 60); }}
        }};
        tryScroll();
    }})();
    </script>
    """, height=0)

    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        user_input = st.text_area(
            label="سؤالك", placeholder=f"اسأل وكيل {category.title}…",
            key=f"agent_input_{category.key}", label_visibility="collapsed", height=88,
        )
    with c2:
        send = st.button("➤ إرسال", key=f"agent_send_{category.key}", use_container_width=True)

    # ── مشاركة ملف مع الوكيل (اختياري): نص، PDF، أو صورة (عبر OCR) ──
    _uploader_types = ["txt", "py", "md", "json", "csv", "log", "yaml", "yml", "pdf"]
    if _OCR_OK:
        _uploader_types += ["png", "jpg", "jpeg"]
    uploaded_file = st.file_uploader(
        "📎 أرفق ملفاً ليطّلع عليه الوكيل قبل الرد — نص/PDF" + (
            "/صورة (OCR)" if _OCR_OK else ""
        ) + " (اختياري)",
        type=_uploader_types,
        key=f"agent_file_{category.key}",
    )
    _MAX_FILE_CHARS = 6000
    file_context, file_label = "", ""
    if uploaded_file is not None:
        _extracted = _extract_file(uploaded_file)
        if _extracted is None:
            st.warning(f"⚠️ الملف أكبر من {MAX_FILE_MB}MB — لم يُرفَع.")
        elif _extracted.get("is_image"):
            _ocr_text = _ocr_image_text(_extracted.get("raw_bytes", b""))
            if _ocr_text:
                file_context = _ocr_text[:_MAX_FILE_CHARS]
                file_label = f"🖼️ {uploaded_file.name} (نص مستخرَج بـ OCR)"
                st.caption(f"{file_label} — سيُرسَل مع رسالتك التالية للوكيل.")
            else:
                st.caption(f"🖼️ {uploaded_file.name} — لم يُستخرَج نص من الصورة (قد تكون بلا نص واضح).")
        else:
            _raw_text = (_extracted.get("text_content") or "").strip()
            if _raw_text:
                _truncated = len(_raw_text) > _MAX_FILE_CHARS
                file_context = _raw_text[:_MAX_FILE_CHARS]
                file_label = f"📎 {uploaded_file.name}" + (" (مقتطع للطول)" if _truncated else "")
                st.caption(f"{file_label} — سيُرسَل محتواه مع رسالتك التالية للوكيل.")

    if category.quick_prompts:
        st.markdown("**⚡ أسئلة سريعة:**")
        qcols = st.columns(len(category.quick_prompts))
        for i, q in enumerate(category.quick_prompts):
            with qcols[i]:
                if st.button(q, key=f"agent_q_{category.key}_{i}", use_container_width=True):
                    st.session_state[f"_agent_pending_{category.key}"] = q

    col_clear, col_export = st.columns(2)
    with col_clear:
        if st.button("🗑 مسح المحادثة", key=f"agent_clear_{category.key}", use_container_width=True):
            st.session_state[msg_key] = []
            st.session_state[cnt_key] = 0
            bot.clear_history()
            st.rerun()
    with col_export:
        if st.session_state[msg_key]:
            _export_lines = [f"# محادثة مع وكيل {category.title}\n"]
            for _m in st.session_state[msg_key]:
                _role, _text = _m[0], _m[1]
                _ts = _m[3] if len(_m) > 3 else ""
                _who = "أنت" if _role == "user" else category.title
                _export_lines.append(f"**{_who}** _{_ts}_\n\n{_text}\n\n---\n")
            st.download_button(
                "⬇️ تصدير المحادثة", data="\n".join(_export_lines).encode("utf-8"),
                file_name=f"محادثة_{category.key}.md", mime="text/markdown",
                key=f"agent_export_{category.key}", use_container_width=True,
            )
        else:
            st.button("⬇️ تصدير المحادثة", disabled=True, use_container_width=True,
                       key=f"agent_export_disabled_{category.key}", help="لا توجد رسائل بعد")

    if st.session_state[msg_key]:
        with st.expander(f"📜 سجل الجلسة ({st.session_state[cnt_key]} تبادل)"):
            for _m in st.session_state[msg_key]:
                _role, _text = _m[0], _m[1]
                _ts = _m[3] if len(_m) > 3 else ""
                _tag = "🧑" if _role == "user" else category.emoji
                _preview = _text if len(_text) <= 140 else _text[:140] + "…"
                st.caption(f"{_tag} `{_ts}` — {_preview}")

    def _process(text: str):
        if not text.strip():
            return
        _ts_now = datetime.now().strftime("%H:%M")
        _display_text = text.strip()
        if file_label:
            _display_text = f"{_display_text}\n\n{file_label}"
        st.session_state[msg_key].append(("user", _display_text, "", _ts_now))

        _safety_msg = _nsm_safety_gate(text.strip())
        if _safety_msg:
            st.session_state[msg_key].append(("bot", _safety_msg, "🛡️ فحص أمان", datetime.now().strftime("%H:%M")))
            st.session_state[cnt_key] += 1
            st.rerun()
            return

        _query = text.strip()
        if file_context:
            _query = (
                f"محتوى الملف المرفق ({uploaded_file.name if uploaded_file else 'ملف'}):\n"
                f"```\n{file_context}\n```\n\nسؤال/طلب المستخدم:\n{text.strip()}"
            )

        with st.spinner(f"⟳ {category.title} يفكّر..."):
            response = bot.chat(_query, force_web=web_toggle, source="hub")
        badge = bot.last_provider_badge()
        try:
            from ai.response_quality import score_response
            _q = score_response(response, query=text.strip())
            badge = f"{badge} · 🔎 {_q.as_percent()}٪ {_q.label}" if badge else f"🔎 {_q.as_percent()}٪ {_q.label}"
        except Exception:
            pass  # تقييم الجودة إضافي وغير حرج — أي فشل فيه لا يجب أن يُسقِط الرد نفسه
        st.session_state[msg_key].append(("bot", response, badge, datetime.now().strftime("%H:%M")))
        st.session_state[cnt_key] += 1
        st.rerun()

    if send and user_input:
        _process(user_input)

    pending_key = f"_agent_pending_{category.key}"
    if pending_key in st.session_state:
        q = st.session_state[pending_key]
        del st.session_state[pending_key]
        _process(q)
