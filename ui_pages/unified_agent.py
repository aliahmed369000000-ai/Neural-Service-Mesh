"""
ui_pages/unified_agent.py — واجهة الوكيل الموحّد (مُعاد تنسيقها بصرياً)
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403


def render_unified_agent():
    """🎯 الوكيل الموحّد: محادثة واحدة + توجيه تلقائي + ذاكرة مشتركة."""
    import html as _html

    if not _AGENTS_HUB_OK:
        st.error("⚠️ تعذّر تحميل الوكيل الموحّد. تأكد من وجود ai/agent_categories.py.")
        return

    if not st.session_state.get("_nsm_ua_css_injected"):
        st.session_state["_nsm_ua_css_injected"] = True
        st.markdown(
            """
<style>
@keyframes uaBubbleIn { from{opacity:0;transform:translateY(6px);} to{opacity:1;transform:translateY(0);} }
.ua-user {display:flex;justify-content:flex-end;margin:0.55rem 0;animation:uaBubbleIn .25s ease-out;}
.ua-user .bbl {
    background:linear-gradient(135deg,var(--gold),var(--emerald));
    color:#fff;padding:0.75rem 1.1rem;border-radius:18px 18px 4px 18px;
    max-width:86%;font-size:0.96rem;line-height:1.75;text-align:right;direction:rtl;
    box-shadow:0 6px 18px var(--gold-soft);white-space:pre-wrap;word-break:break-word;font-weight:600;
}
.ua-bot {display:flex;justify-content:flex-start;margin:0.55rem 0;gap:0.5rem;align-items:flex-start;animation:uaBubbleIn .25s ease-out;}
.ua-bot .bbl {
    background:var(--surface);color:var(--text);padding:0.75rem 1.1rem;border-radius:18px 18px 18px 4px;
    max-width:86%;font-size:0.96rem;line-height:1.85;text-align:right;direction:rtl;
    border:1px solid var(--border);box-shadow:0 4px 14px var(--shadow);white-space:pre-wrap;word-break:break-word;
}
.ua-box {
    height:min(52vh,560px);min-height:300px;overflow-y:auto;padding:1.05rem;
    background:var(--bg);border-radius:18px;border:1px solid var(--border);margin-bottom:0.85rem;
    scroll-behavior:smooth;
}
.ua-badge {
    display:inline-block;background:var(--gold-soft);border:1px solid var(--border);border-radius:20px;
    padding:0.18rem 0.7rem;font-size:0.72rem;color:var(--gold);direction:rtl;font-weight:700;margin-bottom:0.35rem;
}
.ua-bbl-ts {font-size:0.68rem;color:var(--text-muted);margin-top:0.35rem;opacity:.85;}
.ua-toolbar {
    display:flex;flex-wrap:wrap;gap:0.45rem;align-items:center;justify-content:space-between;
    margin:0.4rem 0 0.75rem;direction:rtl;
}
.ua-empty {
    text-align:center;color:var(--text-muted);padding:2.4rem 1rem;direction:rtl;
}
.ua-empty .big {font-size:2rem;margin-bottom:0.6rem;}
.ua-empty .hint {font-size:0.9rem;line-height:1.7;max-width:420px;margin:0 auto;}
</style>
            """,
            unsafe_allow_html=True,
        )

    # ── رأس منظم ──
    st.markdown(
        """
<div class="nsm-hero-panel">
  <div class="nsm-hero-title">🎯 الوكيل الموحّد — مدير مشروعك الشخصي</div>
  <p class="nsm-hero-sub">
    أنا الواجهة الواحدة التي تجمع كل شيء: أفكر، وأقرر متى أبحث في الويب، ومتى أولّد صوراً،
    ومتى أكتب كوداً، ومتى أطلق الوكلاء المتخصصين تحتي. هم ينفّذون المهام وأنا أجمع النتائج
    وأتحمل المسؤولية النهائية عن الجواب. أعطني الهدف فقط.
  </p>
  <div class="nsm-chip-row">
    <span class="nsm-chip nsm-chip--accent">مدير واحد</span>
    <span class="nsm-chip">تفويض ذكي</span>
    <span class="nsm-chip">توليف نهائي</span>
    <span class="nsm-chip">أوامر مشروع حقيقية</span>
    <span class="nsm-chip">بحث ويب</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    if "unified_agent_bot" not in st.session_state:
        st.session_state.unified_agent_bot = UnifiedAgentChat()
        st.session_state.unified_agent_msgs = []
        st.session_state.unified_agent_count = 0

    bot = st.session_state.unified_agent_bot

    # ── شريط أدوات علوي ──
    c_metric, c_web, c_help = st.columns([1.2, 2.2, 2])
    with c_metric:
        st.metric("رسائل الجلسة", st.session_state.unified_agent_count)
    with c_web:
        web_toggle = st.toggle(
            "🌐 بحث ويب قبل الرد",
            value=False,
            key="unified_agent_web",
            help="بحث DuckDuckGo قبل التوليد، أياً كان المتخصص.",
        )
    with c_help:
        with st.expander("أوامر سريعة", expanded=False):
            st.markdown(
                """
**للمستخدم**
- `مساعدة` — ماذا أستطيع؟
- `كيف حال النظام` — لمحة سريعة
- `صنّف: ...` — توجيه MoE
- `افحص المشروع` / `شغّل الاختبارات`
- `حالة نمو الوكيل`

**للمطوّر**
- `فحص ckg` · `ماذا بعد` · `حالة gpu`
                """
            )

    # ── اقتراحات سريعة ──
    st.caption("اقتراحات")
    _suggestions = [
        "مساعدة",
        "كيف حال النظام",
        "صنّف: ما حكم الصلاة؟",
        "افحص المشروع",
        "حالة نمو الوكيل",
    ]
    scols = st.columns(len(_suggestions))
    for i, q in enumerate(_suggestions):
        with scols[i]:
            if st.button(q, key=f"ua_sug_{i}", use_container_width=True):
                st.session_state["_ua_pending"] = q

    # ── صندوق المحادثة ──
    box_id = "unified-agent-chat-box"
    html_out = f'<div class="ua-box" id="{box_id}">'
    if not st.session_state.unified_agent_msgs:
        html_out += (
            '<div class="ua-empty">'
            '<div class="big">🎯</div>'
            '<div class="hint">أعطني الهدف فقط — أنا أقرر الأدوات والوكلاء المناسبين، أجمع النتائج، وأتحمل المسؤولية النهائية عن الجواب.</div>'
            "</div>"
        )
    else:
        for _mi, msg_tuple in enumerate(st.session_state.unified_agent_msgs):
            role, text, badge = msg_tuple[0], msg_tuple[1], msg_tuple[2]
            ts = msg_tuple[3] if len(msg_tuple) > 3 else ""
            ts_html = f'<div class="ua-bbl-ts">{ts}</div>' if ts else ""
            safe = _html.escape(text).replace("\n", "<br>")
            if role == "user":
                html_out += f'<div class="ua-user"><div class="bbl">{safe}{ts_html}</div></div>'
            else:
                badge_html = f'<div class="ua-badge">{_html.escape(badge)}</div>' if badge else ""
                bbl_id = f"{box_id}-msg-{_mi}"
                html_out += (
                    f'<div class="ua-bot"><div class="bbl">{badge_html}'
                    f'<div id="{bbl_id}">{safe}</div>'
                    f'<button class="copy-btn" title="نسخ" style="margin-top:0.4rem"'
                    f' onclick="var t=document.getElementById(\'{bbl_id}\').innerText;'
                    f"navigator.clipboard.writeText(t).then(function(){{"
                    f"var b=event.currentTarget;var old=b.textContent;b.textContent='✓';"
                    f"setTimeout(function(){{b.textContent=old;}},1200);}});\">📋</button>"
                    f"{ts_html}</div></div>"
                )
    html_out += "</div>"
    st.markdown(html_out, unsafe_allow_html=True)
    st.components.v1.html(
        f"""
<script>
(function() {{
  function scrollToBottom() {{
    const doc = window.parent ? window.parent.document : document;
    const box = doc.getElementById('{box_id}');
    if (box) {{ box.scrollTop = box.scrollHeight; return true; }}
    return false;
  }}
  let n = 0;
  const tick = () => {{ n++; if (!scrollToBottom() && n < 12) setTimeout(tick, 50); }};
  tick();
}})();
</script>
        """,
        height=0,
    )

    # ── إدخال ──
    c1, c2 = st.columns([5, 1.15], gap="small")
    with c1:
        user_input = st.text_area(
            label="سؤالك",
            placeholder="مثال: مساعدة · صنّف: ... · افحص المشروع — اسأل أو اكتب أمراً… مثال: فحص ckg",
            key="unified_agent_input",
            label_visibility="collapsed",
            height=90,
        )
    with c2:
        send = st.button("➤ إرسال", key="unified_agent_send", use_container_width=True, type="primary")

    def _process(text: str, add_user: bool = True) -> None:
        text = (text or "").strip()
        if not text:
            return
        if add_user:
            st.session_state.unified_agent_msgs.append(
                ("user", text, "", datetime.now().strftime("%H:%M"))
            )
        with st.spinner("⟳ أفكر وأختار الفريق المناسب…"):
            response, meta = bot.chat(text, force_web=web_toggle)
        # شارة المدير الموحّد أو وكيل المشروع تأخذ الأولوية
        route = meta.get("route_method") or ""
        if route == "project_bridge" or meta.get("category_key") == "master_orchestrator":
            badge = meta.get("provider_badge") or f"{meta.get('category_emoji', '🎯')} {meta.get('category_title', 'المدير الموحّد')}"
        else:
            badge = f"{meta.get('category_emoji', '🤖')} {meta.get('category_title', '')}"
        qb = meta.get("quality_badge", "")
        if qb:
            badge = f"{badge} · {qb}"
        st.session_state.unified_agent_msgs.append(
            ("bot", response, badge, datetime.now().strftime("%H:%M"))
        )
        st.session_state.unified_agent_count += 1
        st.rerun()

    if send and user_input.strip():
        _process(user_input)
    elif "_ua_pending" in st.session_state:
        _pq = st.session_state.pop("_ua_pending", None)
        if _pq:
            _process(_pq)

    _ua_msgs = st.session_state.unified_agent_msgs
    _ua_last_is_bot = bool(_ua_msgs) and _ua_msgs[-1][0] == "bot"
    col_clear, col_regen, col_export = st.columns(3)
    with col_clear:
        if st.button("🗑 مسح", key="unified_agent_clear", use_container_width=True):
            st.session_state.unified_agent_msgs = []
            st.session_state.unified_agent_count = 0
            bot.clear_history()
            st.rerun()
    with col_regen:
        if st.button(
            "🔄 إعادة توليد",
            key="unified_agent_regenerate",
            use_container_width=True,
            disabled=not _ua_last_is_bot,
        ):
            last_user = None
            for m in reversed(_ua_msgs[:-1]):
                if m[0] == "user":
                    last_user = m[1]
                    break
            if last_user:
                st.session_state.unified_agent_msgs.pop()
                _process(last_user, add_user=False)
    with col_export:
        if st.session_state.unified_agent_msgs:
            lines = ["# محادثة الوكيل الموحّد\n"]
            for m in st.session_state.unified_agent_msgs:
                role, text = m[0], m[1]
                badge = m[2] if len(m) > 2 else ""
                ts = m[3] if len(m) > 3 else ""
                who = "أنت" if role == "user" else (badge or "الوكيل")
                lines.append(f"**{who}** _{ts}_\n\n{text}\n\n---\n")
            st.download_button(
                "⬇️ تصدير",
                data="\n".join(lines).encode("utf-8"),
                file_name="محادثة_الوكيل_الموحد.md",
                mime="text/markdown",
                key="unified_agent_export",
                use_container_width=True,
            )
        else:
            st.button("⬇️ تصدير", disabled=True, use_container_width=True, key="unified_agent_export_disabled")
