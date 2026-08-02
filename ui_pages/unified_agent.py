"""
ui_pages/unified_agent.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة




# ══════════════════════════════════════════════════════════════════════════
def render_unified_agent():
    """🎯 الوكيل الموحّد: واجهة محادثة واحدة مستمرة، توجّه كل رسالة تلقائياً
    خلف الكواليس لأنسب متخصص من AGENT_CATEGORIES (نفس منطق route_query_verbose
    المستخدَم أصلاً في 🤝 منسّق الوكلاء)، لكن بذاكرة مشتركة عبر كل الرسائل
    بدل عزل كل فئة بذاكرتها الخاصة — تجربة "وكيل واحد ذكي" حقيقية، بدل خلط
    كل الـ System Prompts في وكيل عام واحد (يُضعف دقة كل تخصص)."""
    import html as _html

    if not _AGENTS_HUB_OK:
        st.error("⚠️ تعذّر تحميل الوكيل الموحّد. تأكد من وجود ai/agent_categories.py.")
        return

    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
        <span style="font-size:2rem">🎯</span>
        <div style="font-size:1.5rem;font-weight:900;color:var(--gold)">الوكيل الموحّد</div>
        <div style="color:var(--text-muted);font-size:0.85rem;direction:rtl">
            محادثة واحدة مستمرة — كل رسالة تُوجَّه تلقائياً خلف الكواليس لأنسب متخصص،
            بذاكرة مشتركة تحافظ على سياق المحادثة عبر كل المواضيع
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.get("_nsm_ua_css_injected"):
        st.session_state["_nsm_ua_css_injected"] = True
        st.markdown("""
    <style>
    @keyframes uaBubbleIn { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:translateY(0);} }
    .ua-user {display:flex;justify-content:flex-end;margin:0.5rem 0;animation:uaBubbleIn .25s ease-out;}
    .ua-user .bbl {
        background:linear-gradient(135deg,var(--gold),var(--emerald));
        color:var(--bg);padding:0.7rem 1.05rem;border-radius:18px 18px 4px 18px;
        max-width:85%;font-size:0.96rem;line-height:1.7;text-align:right;direction:rtl;
        box-shadow:0 3px 12px var(--shadow);white-space:pre-wrap;word-break:break-word;font-weight:600;
    }
    .ua-bot {display:flex;justify-content:flex-start;margin:0.5rem 0;gap:0.5rem;align-items:flex-start;animation:uaBubbleIn .25s ease-out;}
    .ua-bot .bbl {
        background:var(--surface2);color:var(--text);padding:0.7rem 1.05rem;border-radius:18px 18px 18px 4px;
        max-width:85%;font-size:0.96rem;line-height:1.8;text-align:right;direction:rtl;
        border:1px solid var(--border);box-shadow:0 2px 8px var(--shadow);white-space:pre-wrap;word-break:break-word;
    }
    .ua-box {
        height:48vh;min-height:320px;max-height:520px;overflow-y:auto;padding:1rem;
        background:var(--bg);border-radius:16px;border:1px solid var(--border);margin-bottom:0.8rem;
        scroll-behavior:smooth;box-shadow:inset 0 0 20px var(--shadow);
    }
    .ua-badge {
        display:inline-block;background:var(--surface);border:1px solid var(--border);border-radius:20px;
        padding:0.15rem 0.65rem;font-size:0.72rem;color:var(--gold);direction:rtl;
    }
    </style>
    """, unsafe_allow_html=True)

    if "unified_agent_bot" not in st.session_state:
        st.session_state.unified_agent_bot = UnifiedAgentChat()
        st.session_state.unified_agent_msgs = []  # (role, text, badge, ts)
        st.session_state.unified_agent_count = 0

    bot = st.session_state.unified_agent_bot

    col_t, col_s = st.columns([3, 1])
    with col_t:
        st.caption("مثال: اسأل سؤالاً برمجياً ثم اسأل سؤالاً تحليلياً في نفس المحادثة — الذاكرة تبقى مشتركة.")
    with col_s:
        st.metric("رسائل الجلسة", st.session_state.unified_agent_count)

    web_toggle = st.toggle(
        "🌐 بحث حقيقي في الويب قبل الرد",
        value=False, key="unified_agent_web",
        help="يفعّل بحثاً فعلياً عبر DuckDuckGo قبل توليد الرد، أياً كان المتخصص المُختار.",
    )

    box_id = "unified-agent-chat-box"
    html_out = f'<div class="ua-box" id="{box_id}">'
    if not st.session_state.unified_agent_msgs:
        html_out += (
            '<div style="text-align:center;color:var(--text-muted);padding:2rem 1rem">'
            '🎯<br><br>اكتب أي سؤال — سيُوجَّه تلقائياً لأنسب متخصص خلف الكواليس</div>'
        )
    else:
        for _mi, msg_tuple in enumerate(st.session_state.unified_agent_msgs):
            role, text, badge = msg_tuple[0], msg_tuple[1], msg_tuple[2]
            ts = msg_tuple[3] if len(msg_tuple) > 3 else ""
            ts_html = f'<div class="bbl-ts">{ts}</div>' if ts else ""
            safe = _html.escape(text).replace("\n", "<br>")
            if role == "user":
                html_out += f'<div class="ua-user"><div class="bbl">{safe}{ts_html}</div></div>'
            else:
                badge_html = f'<div class="ua-badge">{badge}</div>' if badge else ""
                bbl_id = f"{box_id}-msg-{_mi}"
                html_out += (
                    f'<div class="ua-bot"><div class="bbl">{badge_html}'
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
            label="سؤالك", placeholder="اسأل أي شيء — سيُوجَّه تلقائياً لأنسب متخصص…",
            key="unified_agent_input", label_visibility="collapsed", height=88,
        )
    with c2:
        send = st.button("➤ إرسال", key="unified_agent_send", use_container_width=True)

    if send and user_input.strip():
        ts1 = datetime.now().strftime("%H:%M")
        st.session_state.unified_agent_msgs.append(("user", user_input.strip(), "", ts1))
        with st.spinner("⟳ يُوجَّه للمتخصص الأنسب ويولّد الرد..."):
            response, meta = bot.chat(user_input.strip(), force_web=web_toggle)
        badge = f"{meta.get('category_emoji', '🤖')} {meta.get('category_title', '')}"
        # 🆕 شارة جودة موحّدة (نفس ميزة تبويب "🤖 وكلاء AI")، معروضة الآن
        # أيضاً في الوكيل الموحّد — تُضاف فقط إن توفّر تقييم فعلاً.
        _qb = meta.get("quality_badge", "")
        if _qb:
            badge = f"{badge} · {_qb}"
        ts2 = datetime.now().strftime("%H:%M")
        st.session_state.unified_agent_msgs.append(("bot", response, badge, ts2))
        st.session_state.unified_agent_count += 1
        st.rerun()

    _ua_msgs = st.session_state.unified_agent_msgs
    _ua_last_is_bot = bool(_ua_msgs) and _ua_msgs[-1][0] == "bot"
    col_clear, col_regen, col_export = st.columns(3)
    with col_clear:
        if st.button("🗑 مسح المحادثة", key="unified_agent_clear", use_container_width=True):
            st.session_state.unified_agent_msgs = []
            st.session_state.unified_agent_count = 0
            bot.clear_history()
            st.rerun()
    with col_regen:
        if st.button("🔄 إعادة توليد آخر رد", key="unified_agent_regenerate",
                      use_container_width=True, disabled=not _ua_last_is_bot):
            _last_user_text = None
            for _m in reversed(_ua_msgs[:-1]):
                if _m[0] == "user":
                    _last_user_text = _m[1]
                    break
            if _last_user_text:
                st.session_state.unified_agent_msgs.pop()  # إزالة الرد القديم فقط
                with st.spinner("⟳ يُوجَّه للمتخصص الأنسب ويولّد الرد..."):
                    _r_response, _r_meta = bot.chat(_last_user_text, force_web=web_toggle)
                _r_badge = f"{_r_meta.get('category_emoji', '🤖')} {_r_meta.get('category_title', '')}"
                _r_qb = _r_meta.get("quality_badge", "")
                if _r_qb:
                    _r_badge = f"{_r_badge} · {_r_qb}"
                st.session_state.unified_agent_msgs.append(
                    ("bot", _r_response, _r_badge, datetime.now().strftime("%H:%M"))
                )
                st.rerun()
    with col_export:
        if st.session_state.unified_agent_msgs:
            _export_lines = ["# محادثة مع الوكيل الموحّد\n"]
            for _m in st.session_state.unified_agent_msgs:
                _role, _text = _m[0], _m[1]
                _badge = _m[2] if len(_m) > 2 else ""
                _ts = _m[3] if len(_m) > 3 else ""
                _who = "أنت" if _role == "user" else (_badge or "الوكيل")
                _export_lines.append(f"**{_who}** _{_ts}_\n\n{_text}\n\n---\n")
            st.download_button(
                "⬇️ تصدير المحادثة", data="\n".join(_export_lines).encode("utf-8"),
                file_name="محادثة_الوكيل_الموحد.md", mime="text/markdown",
                key="unified_agent_export", use_container_width=True,
            )
        else:
            st.button("⬇️ تصدير المحادثة", disabled=True, use_container_width=True,
                       key="unified_agent_export_disabled", help="لا توجد رسائل بعد")
