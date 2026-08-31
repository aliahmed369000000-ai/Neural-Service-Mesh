"""
ui_pages/chat.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب المحادثة الذكية
# ══════════════════════════════════════════════════════════════════════════
def render_chat():
    """تبويب المحادثة الذكية مع ذاكرة السياق"""

    if not _NSM_CHAT_OK:
        st.error(
            "⚠️ تعذّر تحميل NSM Chat. تأكد من وجود nsm_chat.py أو nsm_chat_plus.py "
            "و nsm_memory.py في جذر المشروع (nsm_embedding.npz اختياري — يعمل النظام بدونه)."
        )
        return

    # تهيئة النموذج مرة واحدة
    if "nsm_bot" not in st.session_state:
        with st.spinner("⟳ تحميل محرك المحادثة..."):
            st.session_state.nsm_bot = NSMChat(system_prompt=NSM_SYSTEM_PROMPT)
        st.session_state.nsm_messages = []
        st.session_state.nsm_count    = 0
        # معرّف دائم لهذه الجلسة (uuid4، يُولَّد مرة واحدة فقط) — يُستخدم
        # لربط رسائل هذه الجلسة ببعضها في memory/chat_history.db، عشان
        # يمكن الرجوع للمحادثة (أو معرفة من ردّ أولاً) حتى بعد انتهاء
        # الجلسة الحيّة. انظر ai/chat_history_store.py.
        st.session_state.nsm_chat_session_id = str(_uuid.uuid4())

    bot = st.session_state.nsm_bot

    # CSS خاص بالمحادثة
    # أداء: نص ثابت لا يعتمد على أي متغيّر، لكنه كان يُحقن من جديد عبر
    # st.markdown في *كل* rerun لهذا التبويب (كل رسالة تُرسَل، كل تقييم
    # 👍/👎، إلخ)، فيتراكم <style> مكرر في DOM ويُبطئ الصفحة تدريجياً مع
    # طول المحادثة. الحقن الآن يحدث مرة واحدة فقط لكل جلسة.
    if not st.session_state.get("_nsm_chat_css_injected"):
        st.session_state["_nsm_chat_css_injected"] = True
        st.markdown("""
    <style>
    @keyframes bubbleIn {
        from {opacity:0;transform:translateY(8px) scale(0.985);}
        to   {opacity:1;transform:translateY(0) scale(1);}
    }
    .chat-user {display:flex;justify-content:flex-end;margin:0.55rem 0;animation:bubbleIn .32s cubic-bezier(.22,.9,.35,1);}
    .chat-user .bbl {
        background:linear-gradient(135deg,var(--gold),var(--emerald));
        color:var(--bg);padding:0.75rem 1.15rem;
        border-radius:18px 18px 4px 18px;max-width:85%;
        font-size:0.98rem;line-height:1.75;text-align:right;direction:rtl;
        box-shadow:0 3px 12px var(--shadow);
        white-space:pre-wrap;word-break:break-word;
        font-weight:600;
    }
    .chat-nsm {display:flex;justify-content:flex-start;margin:0.55rem 0;gap:0.55rem;align-items:flex-start;animation:bubbleIn .32s cubic-bezier(.22,.9,.35,1);}
    .chat-nsm .bbl {
        background:var(--surface2);
        color:var(--text);padding:0.75rem 1.15rem;
        border-radius:18px 18px 18px 4px;max-width:85%;
        font-size:0.98rem;line-height:1.85;text-align:right;direction:rtl;
        border:1px solid var(--border);
        box-shadow:0 2px 8px var(--shadow);
        white-space:pre-wrap;word-break:break-word;
    }
    .chat-nsm .bbl code {
        background:var(--surface);color:var(--emerald);padding:0.15rem 0.4rem;
        border-radius:4px;font-size:0.88rem;font-family:monospace;
        white-space:pre-wrap;
    }
    .chat-nsm .bbl pre {
        background:var(--surface);border:1px solid var(--border);border-radius:8px;
        padding:0.8rem;overflow-x:auto;margin:0.5rem 0;
        font-size:0.85rem;color:var(--text-muted);
        white-space:pre;
    }
    .copy-btn {
        display:inline-block;margin-top:0.55rem;padding:0.28rem 0.7rem;
        font-size:0.74rem;font-weight:600;color:var(--text-muted);
        background:var(--surface);border:1px solid var(--border);
        border-radius:10px;cursor:pointer;transition:all .15s ease;
        direction:rtl;font-family:inherit;
    }
    .copy-btn:hover { color:var(--gold);border-color:var(--gold);}
    .copy-btn:active { transform:scale(0.96); }
    .ctx-tag {
        display:inline-block;background:var(--surface);border:1px solid var(--border);
        border-radius:20px;padding:0.18rem 0.7rem;font-size:0.72rem;
        color:var(--gold);margin-bottom:0.45rem;direction:rtl;
    }
    .chat-box {
        height:62vh;min-height:420px;max-height:680px;
        overflow-y:auto;padding:1.1rem;
        background:var(--bg);border-radius:18px;
        border:1px solid var(--border);margin-bottom:0.9rem;
        scroll-behavior:smooth;
        -webkit-overflow-scrolling:touch;
        overscroll-behavior:contain;
        box-shadow:inset 0 0 24px var(--shadow);
    }
    .chat-box::-webkit-scrollbar{width:5px;}
    .chat-box::-webkit-scrollbar-track{background:var(--bg);}
    .chat-box::-webkit-scrollbar-thumb{background:var(--border);border-radius:6px;}
    .chat-box::-webkit-scrollbar-thumb:hover{background:var(--gold);}
    .typing-indicator {
        display:inline-block;color:var(--gold);font-size:0.85rem;
        animation:pulse 1.2s infinite;
    }
    @keyframes pulse{0%,100%{opacity:.4;}50%{opacity:1;}}

    /* ── مؤشر "يكتب الآن" بنقاط متتابعة + توهّج حول أيقونة NSM ── */
    .typing-wrap { display:flex; align-items:center; gap:0.6rem; }
    .thinking-ring {
        width:34px; height:34px; border-radius:50%;
        display:flex; align-items:center; justify-content:center;
        font-size:1.15rem;
        background:var(--surface2); border:1px solid var(--border);
        box-shadow:0 0 0 0 var(--gold-soft);
        animation:nsmThinkRing 1.6s ease-out infinite;
        flex-shrink:0;
    }
    @keyframes nsmThinkRing {
        0%   { box-shadow:0 0 0 0 var(--gold-soft); }
        70%  { box-shadow:0 0 0 9px rgba(0,0,0,0); }
        100% { box-shadow:0 0 0 0 rgba(0,0,0,0); }
    }
    .typing-dots { display:inline-flex; gap:4px; align-items:center; padding:0.55rem 0.9rem;
        background:var(--surface2); border:1px solid var(--border); border-radius:18px 18px 18px 4px; }
    .typing-dots span {
        width:7px; height:7px; border-radius:50%;
        background:var(--gold); display:inline-block;
        animation:nsmDotBounce 1.1s ease-in-out infinite;
    }
    .typing-dots span:nth-child(2) { animation-delay:.15s; background:var(--emerald); }
    .typing-dots span:nth-child(3) { animation-delay:.3s; }
    @keyframes nsmDotBounce {
        0%, 60%, 100% { transform:translateY(0); opacity:.55; }
        30% { transform:translateY(-5px); opacity:1; }
    }
    @media (prefers-reduced-motion: reduce) {
        .thinking-ring, .typing-dots span { animation:none !important; }
    }

    /* ── توقيت الرسائل (يظهر بأسفل كل فقاعة) ── */
    .bbl-ts {
        font-size: 0.68rem;
        color: var(--text-muted);
        opacity: 0.75;
        margin-top: 0.3rem;
        direction: ltr;
        text-align: left;
    }
    .chat-user .bbl-ts { color: rgba(0,0,0,0.55); text-align: right; }
    .bbl-footer { display: flex; align-items: center; justify-content: space-between; gap: 0.5rem; margin-top: 0.2rem; }
    .bbl-footer .bbl-ts { margin-top: 0; }

    /* ── زر عائم "النزول لآخر رسالة" — يظهر فقط عند التمرير لأعلى بعيداً
       عن نهاية المحادثة (يُتحكّم بإظهاره/إخفائه عبر JS بالأسفل) ── */
    .chat-box-wrap { position: relative; }
    .scroll-bottom-btn {
        position: absolute;
        bottom: 1.1rem;
        left: 50%;
        transform: translateX(-50%) translateY(8px);
        width: 38px; height: 38px;
        border-radius: 50%;
        border: 1px solid var(--border);
        background: var(--surface2);
        color: var(--gold);
        font-size: 1.1rem;
        box-shadow: 0 6px 18px var(--shadow);
        cursor: pointer;
        display: flex; align-items: center; justify-content: center;
        opacity: 0;
        pointer-events: none;
        transition: opacity .2s ease, transform .2s ease;
        z-index: 5;
    }
    .scroll-bottom-btn.visible { opacity: 1; pointer-events: auto; transform: translateX(-50%) translateY(0); }
    .scroll-bottom-btn:hover { border-color: var(--gold); }

    /* ── تنسيق st.chat_message الأصلي (يُستخدم فقط أثناء بث الرد حرفاً
       بحرف قبل أن يُطوى داخل .chat-box المخصص بعد rerun) — بدون هذا
       التنسيق يظهر بمظهر Streamlit الافتراضي غير المرتبط بصرياً بهوية
       الشات، ما يسبب "قفزة" بصرية واضحة لحظة انتهاء البث. ────────────── */
    [data-testid="stChatMessage"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px 18px 18px 4px !important;
        padding: 0.75rem 1.15rem !important;
        margin: 0.55rem 0 0.9rem !important;
        box-shadow: 0 2px 8px var(--shadow);
        direction: rtl !important;
        max-width: 85%;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageContent"],
    [data-testid="stChatMessage"] p {
        color: var(--text) !important;
        text-align: right !important;
        direction: rtl !important;
        font-size: 0.98rem !important;
        line-height: 1.85 !important;
    }
    [data-testid="stChatMessage"] [data-testid="stChatMessageAvatarAssistant"],
    [data-testid="stChatMessage"] [class*="Avatar"] {
        background: var(--accent-grad) !important;
        border-radius: 10px !important;
    }

    /* ── بطاقة تعريف المحادثة ── */
    .chat-hero {
        display:grid;grid-template-columns:auto 1fr;gap:0.85rem;align-items:center;
        padding:0.95rem 1.05rem;margin:0.2rem 0 1rem;
        border:1px solid var(--border);border-radius:18px;
        background:linear-gradient(135deg,var(--surface2),var(--bg-soft));
        box-shadow:0 5px 18px var(--shadow);
    }
    .chat-hero-icon {
        width:46px;height:46px;border-radius:14px;display:flex;align-items:center;
        justify-content:center;font-size:1.45rem;background:var(--accent-grad);
        box-shadow:0 5px 16px var(--gold-soft);
    }
    .chat-hero-title { color:var(--text);font-size:0.98rem;font-weight:800;line-height:1.5; }
    .chat-hero-subtitle { color:var(--text-muted);font-size:0.79rem;line-height:1.7;margin-top:0.1rem; }
    .chat-hero-pills { grid-column:2;display:flex;flex-wrap:wrap;gap:0.38rem;margin-top:-0.25rem; }
    .chat-hero-pill {
        display:inline-flex;align-items:center;gap:0.2rem;padding:0.18rem 0.52rem;
        border-radius:999px;background:var(--surface);border:1px solid var(--border);
        color:var(--text-muted);font-size:0.68rem;white-space:nowrap;
    }
    .chat-composer-hint {
        display:flex;justify-content:space-between;gap:0.6rem;flex-wrap:wrap;
        color:var(--text-muted);font-size:0.72rem;margin:0.15rem 0 0.42rem;
    }
    .chat-composer-hint span { padding:0.18rem 0.45rem;border-radius:7px;background:var(--surface);border:1px solid var(--border); }

    /* ── استجابة الجوال ── */
    /* طبقة المحادثة الاحترافية: حاوية أنظف، رموز أوضح، وفصل بصري بين الأدوار */
    .chat-box { background:linear-gradient(180deg,rgba(15,23,42,.88),rgba(10,14,23,.96)); border-color:rgba(45,212,191,.18); }
    .chat-user .bbl { background:linear-gradient(135deg,#6d5dfc,#2dd4bf); color:#071018; border:1px solid rgba(255,255,255,.16); box-shadow:0 8px 24px rgba(45,212,191,.13); }
    .chat-nsm .bbl { background:rgba(255,255,255,.055); border-color:rgba(148,163,184,.2); box-shadow:0 7px 22px rgba(0,0,0,.16); }
    .chat-nsm::before { content:"NSM"; display:grid; place-items:center; flex:0 0 auto; width:30px; height:30px; border-radius:10px; color:#071018; background:linear-gradient(135deg,#2dd4bf,#6d5dfc); font-size:.58rem; font-weight:950; letter-spacing:.04em; box-shadow:0 5px 14px rgba(45,212,191,.16); }
    .chat-user::after { content:"أنت"; order:-1; align-self:flex-end; margin-bottom:.35rem; color:var(--text-muted); font-size:.65rem; font-weight:800; }
    .chat-nsm .bbl, .chat-user .bbl { transition:transform .16s ease,box-shadow .16s ease; }
    .chat-nsm .bbl:hover, .chat-user .bbl:hover { transform:translateY(-1px); }
    .chat-hero { border-color:rgba(45,212,191,.2); background:linear-gradient(135deg,rgba(109,93,252,.18),rgba(15,23,42,.76) 60%,rgba(45,212,191,.1)); }
    .chat-hero-title { font-size:1.08rem; }
    .chat-hero-pill { background:rgba(255,255,255,.055); border-color:rgba(148,163,184,.2); }
    .chat-composer-hint span { border-color:rgba(45,212,191,.18); background:rgba(45,212,191,.06); }
    [data-testid="stChatInput"] { margin-top:.25rem; }
    [data-testid="stChatInput"] > div { border:1px solid rgba(45,212,191,.28) !important; border-radius:16px !important; background:rgba(15,23,42,.9) !important; box-shadow:0 10px 28px rgba(0,0,0,.16); }
    [data-testid="stChatInput"] textarea { min-height:48px !important; padding:.8rem 3rem .8rem .9rem !important; direction:rtl; }
    [data-testid="stChatInput"] button { border-radius:11px !important; background:linear-gradient(135deg,#6d5dfc,#2dd4bf) !important; color:#071018 !important; }
    .chat-box-wrap { filter:drop-shadow(0 10px 24px rgba(0,0,0,.08)); }

    @media (max-width: 640px) {
        .chat-box {
            height:56vh;min-height:320px;max-height:520px;
            padding:0.8rem;border-radius:14px;
        }
        .chat-user .bbl, .chat-nsm .bbl {
            max-width:92%;font-size:0.92rem;padding:0.65rem 0.9rem;
        }
        .chat-nsm .bbl { line-height:1.7; }
        [data-testid="stChatMessage"] { max-width: 92%; }
        .chat-hero { grid-template-columns:auto 1fr;padding:0.8rem;border-radius:14px; }
        .chat-hero-icon { width:40px;height:40px;font-size:1.2rem;border-radius:12px; }
        .chat-hero-pills { grid-column:1 / -1;margin-top:0; }
        .chat-composer-hint { justify-content:flex-start; }
    }
    
    /* ── زر النجمة (الإشارة المرجعية) ── */
    .bookmark-btn {
        display:inline-block;margin-top:0.55rem;padding:0.28rem 0.7rem;
        font-size:0.85rem;color:var(--text-muted);
        background:transparent;border:1px solid var(--border);
        border-radius:10px;cursor:pointer;transition:all .15s ease;
        direction:rtl;font-family:inherit;margin-left:0.5rem;
    }
    .bookmark-btn:hover { color:var(--gold);border-color:var(--gold);}
    .bookmark-btn.bookmarked { color:var(--gold);background:var(--gold-soft);border-color:var(--gold);}
    .bookmark-btn:active { transform:scale(0.96); }

    /* ── 🆕 الحزمة 1: شارة أداء last_metadata (زمن/مزود/إعادة توجيه) ── */
    .nsm-perf-meta {
        display:inline-flex;gap:0.35rem;align-items:center;flex-wrap:wrap;
        font-size:0.7rem;color:var(--text-muted);margin-bottom:0.35rem;
        padding:0.15rem 0.6rem;border-radius:999px;
        background:var(--surface);border:1px dashed var(--border);
        direction:ltr;
    }

    </style>
    """, unsafe_allow_html=True)

    # رأس التبويب
    col_t, col_s = st.columns([3,1])
    with col_t:
        st.markdown('<div class="section-header">💬 المحادثة الذكية</div>', unsafe_allow_html=True)
        _mode = "🤖 LLM · Cloudflare / Gemini / Groq"
        st.caption(f"يتذكر السياق · {_mode} · الذكاء في الأوزان")
    with col_s:
        ctx = bot.context_info()
        if ctx:
            st.markdown(f'<div class="ctx-tag">📎 {ctx}</div>', unsafe_allow_html=True)
        st.metric("رسائل الجلسة", st.session_state.nsm_count)

    st.markdown("""
    <div class="chat-hero">
        <div class="chat-hero-icon">🧠</div>
        <div>
            <div class="chat-hero-title">مساحتك للحوار والفهم المتدرّج</div>
            <div class="chat-hero-subtitle">اسأل بحرية، تابع السياق، واستكشف الإجابة من أكثر من زاوية دون مغادرة المحادثة.</div>
        </div>
        <div class="chat-hero-pills">
            <span class="chat-hero-pill">🧠 فهم السياق</span>
            <span class="chat-hero-pill">📚 شبكة معرفية</span>
            <span class="chat-hero-pill">📋 نسخ الرد</span>
            <span class="chat-hero-pill">⌘+Enter إرسال</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── تهيئة حالة الملفات المرفَقة (تُستخدم لاحقاً في قسم الكتابة) ───────
    if "chat_pending_files" not in st.session_state:
        st.session_state["chat_pending_files"] = []
    if "chat_uploader_version" not in st.session_state:
        st.session_state["chat_uploader_version"] = 0

    # ── 🔍 بحث ضمن المحادثة الحالية — يفلتر العرض فقط، لا يمسّ السجل
    # المحفوظ (nsm_messages يبقى كاملاً؛ هذا يؤثر فقط على ما يُبنى في
    # html أدناه). مفيد بالمحادثات الطويلة لإيجاد رسالة سابقة بسرعة. ──
    _chat_search_query = ""
    if st.session_state.get("nsm_messages"):
        _chat_search_query = st.text_input(
            "بحث في المحادثة", key="nsm_chat_search",
            placeholder="🔍 ابحث ضمن هذه المحادثة...",
            label_visibility="collapsed",
        ).strip()

    # ══════════════════════════════════════════════════════════════════════
    # 📂 الحزمة 2: إدارة الجلسات المتعددة — استعادة محادثات سابقة محفوظة
    # في memory/chat_history.db (عبر ai/chat_history_store.py) — التدهور
    # آمن كامل: أي فشل استيراد/قراءة لا يمنع عرض المحادثة الحيّة إطلاقًا.
    # ══════════════════════════════════════════════════════════════════════
    _CHS_OK = False
    _chs_list_sessions = None
    _chs_get_session_messages = None
    try:
        from ai import chat_history_store as _CHS_MOD
        _chs_list_sessions = _CHS_MOD.list_sessions
        _chs_get_session_messages = _CHS_MOD.get_session_messages
        _CHS_OK = True
    except Exception:
        _CHS_OK = False
    _session_loaded_id = st.session_state.get("_nsm_session_loaded_id", "")
    if _CHS_OK:
        with st.expander("📂 إدارة الجلسات السابقة", expanded=False):
            _ses_cols = st.columns([1, 3])
            with _ses_cols[0]:
                st.caption(f"جلسة حية: `{st.session_state.nsm_chat_session_id[:8]}…`")
                if st.button("➕ محادثة جديدة", key="nsm_new_session_btn", use_container_width=True,
                             help="يولّد معرّف جلسة جديدة ويبدأ من الصفر — الرسائل الحالية محفوظة في قاعدة البيانات بلا حذف"):
                    st.session_state.nsm_chat_session_id = str(_uuid.uuid4())
                    st.session_state.nsm_messages = []
                    st.session_state.nsm_count = 0
                    st.session_state["_nsm_session_loaded_id"] = ""
                    st.session_state["_nsm_chat_summary"] = ""
                    st.rerun()
                if st.button("🧹 تنظيف الجلسات (أقدم من 30 يومًا)", key="nsm_prune_sessions_btn", use_container_width=True):
                    try:
                        _CHS_MOD.delete_sessions_older_than(30)
                        st.success("✅ تم تنظيف الجلسات القديمة")
                    except Exception as _prune_exc:
                        st.warning(f"⚠️ تعذّر التنظيف: {_prune_exc}")
                    st.rerun()
            with _ses_cols[1]:
                try:
                    _all_hist_sessions = _chs_list_sessions(limit=50) or []
                    _hist_items = []
                    for _hs in _all_hist_sessions:
                        _hs_id = str(_hs.get("session_id", ""))
                        _hs_cnt = _hs.get("message_count", 0)
                        _hs_last = str(_hs.get("last_at", ""))[:16]
                        _hs_label = f"{_hs_id[:8]}… · {_hs_cnt} رسالة · آخرها {_hs_last}" if _hs_last else f"{_hs_id[:8]}… · {_hs_cnt} رسالة"
                        _hist_items.append((_hs_label, _hs_id))
                    if not _hist_items:
                        st.caption("لا توجد جلسات سابقة محفوظة بعد")
                    else:
                        _sel_idx = st.selectbox(
                            "اختر جلسة لاستعادة محادثتها",
                            options=range(len(_hist_items)),
                            format_func=lambda _j: _hist_items[_j][0],
                            key="nsm_session_selector",
                        )
                        _restore_cols = st.columns(2)
                        with _restore_cols[0]:
                            if st.button("📥 استعادة هذه الجلسة", use_container_width=True):
                                _pick_sid = _hist_items[_sel_idx][1]
                                try:
                                    _hist_msgs = _chs_get_session_messages(_pick_sid, limit=500) or []
                                    if not _hist_msgs:
                                        st.warning("⚠️ لا توجد رسائل في هذه الجلسة")
                                    else:
                                        st.session_state.nsm_messages = []
                                        for _hm in _hist_msgs:
                                            _hm_role = str(_hm.get("role", "nsm") or "nsm")
                                            _hm_content = str(_hm.get("content", "") or "")
                                            _hm_badge = str(_hm.get("source_badge", "") or "")
                                            _hm_ts = str(_hm.get("created_at", ""))
                                            if _hm_ts and len(_hm_ts) >= 16:
                                                _hm_ts = _hm_ts[11:16]
                                            _ts_slot = _hm_ts if _hm_ts and _hm_ts != "None" else ""
                                            st.session_state.nsm_messages.append(
                                                (_hm_role, _hm_content, "", _hm_badge, _ts_slot)
                                            )
                                        st.session_state.nsm_chat_session_id = _pick_sid
                                        st.session_state.nsm_count = len(st.session_state.nsm_messages)
                                        st.session_state["_nsm_session_loaded_id"] = _pick_sid
                                        st.session_state["_nsm_chat_summary"] = ""
                                        st.session_state["_nsm_chat_display_ceil"] = NSM_CHAT_DISPLAY_LIMIT
                                        st.rerun()
                                except Exception as _load_exc:
                                    st.warning(f"⚠️ تعذّر استعادة الجلسة: {_load_exc}")
                        with _restore_cols[1]:
                            if st.button("🗑 حذف هذه الجلسة", use_container_width=True):
                                _del_sid = _hist_items[_sel_idx][1]
                                # حذف الجلسة المحددة فوريًا: نستخدم نفس جدول
                                # chat_history_store (chat_messages) — API موجود
                                # delete_sessions_older_than يعتمد على الأيام
                                # فلا يُرضي حذفًا فوريًا مستهدفًا؛ لذا SQL مباشر
                                # داخل try محمي كامل.
                                if _del_sid:
                                    try:
                                        _conn = _CHS_MOD._db()
                                        _conn.execute(
                                            "DELETE FROM chat_messages WHERE session_id = ?", (_del_sid,)
                                        )
                                        _conn.commit()
                                        _conn.close()
                                    except Exception:
                                        try:
                                            _conn.close()
                                        except Exception:
                                            pass
                                    st.rerun()
                except Exception as _list_exc:
                    st.warning(f"⚠️ تعذّر قراءة سجل الجلسات: {_list_exc}")
    # ══════════════════════════════════════════════════════════════════════
    # 📊 إحصائيات الجلسة + فلتر الرسائل المرجعية
    # ══════════════════════════════════════════════════════════════════════
    if st.session_state.nsm_messages:
        _stats_cols = st.columns(4)
        _total_msgs = len(st.session_state.nsm_messages)
        _user_msgs = sum(1 for m in st.session_state.nsm_messages if m[0] == "user")
        _nsm_msgs = sum(1 for m in st.session_state.nsm_messages if m[0] == "nsm")
        _total_words = sum(len(m[1].split()) for m in st.session_state.nsm_messages)
        _bookmarks = st.session_state.get("nsm_bookmarks", set())
        
        with _stats_cols[0]:
            st.metric("📨 الرسائل", _total_msgs, delta=f"{_user_msgs} أنت / {_nsm_msgs} NSM")
        with _stats_cols[1]:
            st.metric("📝 الكلمات", _total_words)
        with _stats_cols[2]:
            _avg_len = _total_words // max(1, _nsm_msgs)
            st.metric("📊 متوسط الرد", f"{_avg_len} كلمة")
        with _stats_cols[3]:
            st.metric("⭐ المفضلة", len(_bookmarks))
        
        # فلتر الرسائل المرجعية
        _filter_cols = st.columns([3, 1])
        with _filter_cols[0]:
            _show_bookmarks_only = st.checkbox(
                "⭐ عرض الرسائل المفضلة فقط",
                key="nsm_filter_bookmarks",
                help="اعرض فقط الرسائل التي وضعت عليها نجمة"
            )
        with _filter_cols[1]:
            if _bookmarks and st.button("🗑 مسح المفضلة", key="nsm_clear_bookmarks"):
                st.session_state["nsm_bookmarks"] = set()
                st.rerun()
    else:
        _show_bookmarks_only = False
    
    # عرض المحادثة
    # تهيئة متغيرات العرض قبل الفرع الشرطي حتى يبقى قسم «تحميل المزيد»
    # آمناً عند أول فتح للمحادثة الفارغة، ولا يعتمد على متغيرات عُرّفت
    # فقط داخل فرع وجود رسائل سابقة.
    _hidden_count = 0
    _search_active = False
    _display_ceil = NSM_CHAT_DISPLAY_LIMIT
    html = '<div class="chat-box-wrap"><div class="chat-box" id="nsm-chat-box">'
    if not st.session_state.nsm_messages:
        html += '''<div style="text-align:center;color:var(--text-muted);padding:3rem 1rem;
                    display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%">
            <div style="width:56px;height:56px;border-radius:16px;display:flex;align-items:center;
                        justify-content:center;font-size:1.8rem;background:var(--accent-grad);
                        box-shadow:0 6px 20px var(--gold-soft);margin-bottom:1rem">🧠</div>
            <div style="font-size:1.05rem;font-weight:700;color:var(--text);margin-bottom:0.3rem">
                ابدأ محادثتك مع NSM
            </div>
            <div style="font-size:0.85rem;max-width:360px;line-height:1.85">
                اسأل عن مفهوم إسلامي، آية قرآنية، جذر لغوي، أو أي موضوع آخر.<br>
                النظام يجيب عبر الشبكة المعرفية حتى بدون مفاتيح API.<br>
                <span style="opacity:0.85">جرّب أحد الأسئلة السريعة بالأسفل للبدء فوراً</span>
            </div>
        </div>'''
    else:
        # أداء: نعرض آخر NSM_CHAT_DISPLAY_LIMIT رسالة فقط بدل كل السجل —
        # بدون هذا السقف، HTML المُعاد بناؤه بالكامل في كل rerun (كل رسالة
        # جديدة/تقييم/تفاعل) يتضخم مع طول المحادثة ويُبطئ الواجهة تدريجياً.
        # نحافظ على الفهرس الحقيقي _i (وليس فهرس القائمة المقصوصة) كي تبقى
        # مطابقة _nsm_audio_cache (المُخزَّن بفهرس الرسالة الأصلي) صحيحة.
        _all_msgs_indexed = list(enumerate(st.session_state.nsm_messages))
        # 🆕 حد عرض متدرج (virtual scroll): سقف محفوظ في session_state
        # (`_nsm_chat_display_ceil`) يبدأ من NSM_CHAT_DISPLAY_LIMIT ويرتفع
        # تدريجياً عبر زر «⬆️ تحميل المزيد» — ويعاد ضبطه إلى
        # NSM_CHAT_DISPLAY_LIMIT عند وصول رسالة جديدة. البحث والمفضلة
        # يتجاوزانه تلقائياً لعرض كل النتائج.
        _search_active = bool(_chat_search_query) or st.session_state.get("nsm_filter_bookmarks", False)
        _display_ceil = st.session_state.setdefault(
            "_nsm_chat_display_ceil", NSM_CHAT_DISPLAY_LIMIT
        )
        _active_limit = (
            len(_all_msgs_indexed)
            if _search_active
            else min(_display_ceil, len(_all_msgs_indexed))
        )
        _hidden_count = max(0, len(_all_msgs_indexed) - _active_limit)
        # 🆕 الملخص السياقي التلقائي: عند تجاوز NSM_CHAT_MEMORY_SUMMARY_AT
        # رسالة، يُعرض ملخص مضغوط للجزء الأقدم فوق السجل (السجل الكامل
        # يبقى محفوظاً في nsm_messages وchat_history_store بلا حذف).
        # الملخص مخزّن في session_state فلا يعاد بناؤه في كل rerun.
        if (not _search_active
                and len(st.session_state.nsm_messages) > NSM_CHAT_MEMORY_SUMMARY_AT):
            _memory_summary = st.session_state.get("_nsm_chat_summary", "")
            if not _memory_summary:
                _memory_summary = build_chat_memory_summary(st.session_state.nsm_messages)
                st.session_state["_nsm_chat_summary"] = _memory_summary
            if _memory_summary:
                import html as _html_esc_ms
                _ms_safe = _html_esc_ms.escape(_memory_summary).replace("\n", "<br>")
                html += (
                    f'<details style="margin:0.3rem 0.7rem 0.5rem;padding:0.7rem 0.9rem;'
                    f'border-radius:12px;border:1px solid var(--gold-soft);'
                    f'background:linear-gradient(135deg, var(--gold-tint) 0%, var(--bg-soft) 100%);'
                    f'font-size:0.8rem;line-height:1.7">'
                    f'<summary style="cursor:pointer;font-weight:700;color:var(--text);">'
                    f'📜 ملخص ذاكرة المحادثة الطويلة '
                    f'(أقدم {len(st.session_state.nsm_messages) - NSM_CHAT_MEMORY_SUMMARY_AT} رسالة '
                    f'— الملخص التلقائي، النص الكامل محفوظ)</summary>'
                    f'<div style="color:var(--text-muted);margin-top:0.5rem">{_ms_safe}</div>'
                    f'</details>'
                )
        # 🆕 شريط الذكريات المستحضرة (الذاكرة الطويلة): عند وجود آخر سؤال
        # مستخدم، تستحضر المنصة ذكرياتها ذات الصلة (أسئلة سابقة مشابهة،
        # تصويبات، ودروس جماعية متعلقة) وتعرضها كسياق مساعد فوق المحادثة —
        # لا يُحذف شيء ولا يتضخم التوكنز؛ عرض مساعد فقط.
        _last_user_question = ""
        for _lm in reversed(st.session_state.nsm_messages):
            if _lm[0] == "user" and _lm[1].strip():
                _last_user_question = _lm[1].strip()
                break
        if (not _search_active and _LTM_OK and _last_user_question):
            try:
                _ltm = _get_ltm()
                _ltm_memories = _ltm.recall(_last_user_question)
                if _ltm_memories:
                    import html as _ltm_html
                    _mtype_icons = {"question": "💬", "correction": "✏️", "preference": "⚙️", "lesson": "📖", "fact": "🔖"}
                    _mem_parts = []
                    for _mem in _ltm_memories:
                        _m_icon = _mtype_icons.get(_mem["memory_type"], "🧠")
                        _m_safe = _ltm_html.escape(_mem["topic"] or "").replace("\n", "<br>")
                        _i_safe = _ltm_html.escape((_mem["insight"] or "")[:95]).replace("\n", "<br>")
                        _mem_parts.append(
                            f'<div style="margin:0.2rem 0;padding:0.25rem 0">'
                            f'<span>{_m_icon} <b>{_m_safe}</b></span> '
                            f'<span style="color:var(--text-muted)">{_i_safe}</span></div>'
                        )
                    _mems_joined = "".join(_mem_parts)
                    html += (
                        f'<div style="margin:0.3rem 0.7rem 0.5rem;padding:0.7rem 0.9rem;'
                        f'border-radius:12px;border:1px solid var(--emerald-soft, #34d39940);'
                        f'background:linear-gradient(135deg, var(--gold-tint, #fef3c740) 0%, var(--bg-soft) 100%);'
                        f'font-size:0.8rem;line-height:1.7">'
                        f'<details><summary style="cursor:pointer;font-weight:700;color:var(--emerald, #34d399)">'
                        f'🧠 ذكريات مستحضرة — استرجعت {len(_ltm_memories)} ذاكرة ذات صلة</summary>'
                        f'<div style="color:var(--text);margin-top:0.5rem">{_mems_joined}</div>'
                        f'</details></div>'
                    )
            except Exception:
                pass  # فشل استحضار الذكريات لا يمنع عرض المحادثة إطلاقًا
        if _chat_search_query:
            _q_low = _chat_search_query.lower()
            _all_msgs_indexed = [
                (_i, _m) for _i, _m in _all_msgs_indexed
                if _q_low in (_m[1] or "").lower()
            ]
        if _chat_search_query and not _all_msgs_indexed:
            import html as _html_esc
            html += (
                f'<div style="text-align:center;color:var(--text-muted);padding:2rem 1rem">'
                f'لا توجد رسائل مطابقة لـ«{_html_esc.escape(_chat_search_query)}»'
                f'</div>'
            )
        # فلتر الرسائل المرجعية
        if st.session_state.get("nsm_filter_bookmarks", False):
            _bookmarks = st.session_state.get("nsm_bookmarks", set())
            _all_msgs_indexed = [(_i, _m) for _i, _m in _all_msgs_indexed if _i in _bookmarks]
            if not _all_msgs_indexed and not _chat_search_query:
                html += (
                    '<div style="text-align:center;color:var(--text-muted);padding:2rem 1rem">'
                    'لا توجد رسائل مفضلة بعد — اضغط على ☆ بجانب أي رسالة لإضافتها للمفضلة'
                    '</div>'
                )
        _hidden_count = max(0, len(_all_msgs_indexed) - NSM_CHAT_DISPLAY_LIMIT)
        if _hidden_count:
            _hidden_note = (
                "نتيجة بحث أقدم مخفية" if _chat_search_query else "رسالة أقدم مخفية من العرض، لكنها لا تزال محفوظة"
            )
            html += (
                f'<div style="text-align:center;color:var(--text-muted);'
                f'font-size:0.78rem;padding:0.4rem 0 0.7rem">'
                f'— تُعرض آخر {NSM_CHAT_DISPLAY_LIMIT} رسالة فقط '
                f'({_hidden_count} {_hidden_note}) —'
                f'</div>'
            )
        for _i, msg in _all_msgs_indexed[-NSM_CHAT_DISPLAY_LIMIT:]:
            role, text = msg[0], msg[1]
            ctx_tag    = msg[2] if len(msg) > 2 else ""
            src_badge  = msg[3] if len(msg) > 3 else ""
            ts         = msg[4] if len(msg) > 4 else ""
            ts_html    = f'<div class="bbl-ts">{ts}</div>' if ts else ""
            if role == "user":
                import html as _html
                safe_text = _html.escape(text).replace("\n", "<br>")
                _is_bookmarked = _i in st.session_state.get("nsm_bookmarks", set())
                _bookmark_class = "bookmarked" if _is_bookmarked else ""
                _bookmark_icon = "★" if _is_bookmarked else "☆"
                _bookmark_btn = f'<button class="bookmark-btn {_bookmark_class}" onclick="this.classList.toggle(&quot;bookmarked&quot;); var icon=this.textContent; this.textContent = icon.includes(&quot;★&quot;) ? &quot;☆&quot; : &quot;★&quot;;">{_bookmark_icon}</button>'
                html += f'<div class="chat-user"><div class="bbl">{safe_text}{ts_html}{_bookmark_btn}</div></div>'
            else:
                ctx_html = f'<div class="ctx-tag">📎 {ctx_tag}</div>' if ctx_tag else ""
                src_html = (
                    f'<div class="ctx-tag" style="color:var(--emerald)">{src_badge}</div>'
                    if src_badge else ""
                )
                _audio_html = ""
                _audio_entry = st.session_state.get("_nsm_audio_cache", {}).get(_i)
                if _audio_entry:
                    _a_b64, _a_fmt = _audio_entry
                    _audio_html = (
                        f'<audio controls style="width:100%;margin-top:0.5rem;height:36px" '
                        f'src="data:audio/{_a_fmt};base64,{_a_b64}"></audio>'
                    )
                # ── 🆕 الحزمة 1: شارة أداء last_metadata أسفل كل ردّ ───────
                # لكل ردّ NSM نبحث في nsm_route_log (سجل آخر 100 قرار توجيه) عن
                # آخر قرار نجاح لم يُستهلَك بعد — بهذا تطابق كل فقاعة ردّ
                # بزمن الاستجابة الحقيقي (ms) والمزوّد الذي ردّ فعليًا وعدد
                # محاولات إعادة التوجيه (failover). أي فشل استرجاع = لا شارة
                # إطلاقًا (لا يؤثر على عرض الردّ).
                _perf_html = ""
                _route_log_perf = st.session_state.get("nsm_route_log", [])
                if _route_log_perf:
                    try:
                        _nsm_seen_perf = 0
                        for _k, _mv in enumerate(st.session_state.nsm_messages[: _i + 1]):
                            if _mv[0] != "user":
                                _nsm_seen_perf += 1
                        _succ_entries_perf = [e for e in _route_log_perf if e.get("success")]
                        if len(_succ_entries_perf) >= _nsm_seen_perf:
                            _pe = _succ_entries_perf[_nsm_seen_perf - 1]
                            _pe_node = str(_pe.get("node", "")).replace("nsm:", "")
                            _pe_lat = _pe.get("latency_ms", 0) or 0
                            _pe_fo = bool(_pe.get("failover"))
                            _lat_color = "var(--emerald)" if _pe_lat < 1500 else ("#F59E0B" if _pe_lat > 5000 else "var(--text-muted)")
                            _perf_html = (
                                f'<div class="nsm-perf-meta" title="سجل التوجيه: {_pe_node}">'
                                f'⏱ {int(_pe_lat)}ms'
                                f'<span style="color:{_lat_color}"> · {(_pe_node or "—")[:30]}</span>'
                                + (" · 🔄 failover" if _pe_fo else "") + "</div>"
                            )
                    except Exception:
                        _perf_html = ""
                import html as _html
                if "<" not in text and ">" not in text:
                    safe_reply = _html.escape(text).replace("\n", "<br>")
                else:
                    safe_reply = text
                _is_bookmarked = _i in st.session_state.get("nsm_bookmarks", set())
                _bookmark_class = "bookmarked" if _is_bookmarked else ""
                _bookmark_icon = "★" if _is_bookmarked else "☆"
                html += f'''<div class="chat-nsm">
                    <span style="font-size:1.4rem;margin-top:3px">🧠</span>
                    <div class="bbl">{ctx_html}{src_html}{_perf_html}<div class="bbl-text" id="nsm-bbl-{_i}">{safe_reply}</div>{_audio_html}
                        <div class="bbl-footer">
                            <button class="copy-btn" title="نسخ الرد"
                                onclick="var t=document.getElementById('nsm-bbl-{_i}').innerText;
                                         navigator.clipboard.writeText(t).then(function(){{
                                            var b=event.currentTarget; var old=b.textContent;
                                            b.textContent='✓ تم النسخ';
                                            setTimeout(function(){{b.textContent=old;}}, 1300);
                                         }});">📋 نسخ</button>
                            <button class="bookmark-btn {_bookmark_class}" onclick="this.classList.toggle(&quot;bookmarked&quot;); var icon=this.textContent; this.textContent = icon.includes(&quot;★&quot;) ? &quot;☆&quot; : &quot;★&quot;;">{_bookmark_icon}</button>
                            {ts_html}
                        </div>
                    </div>
                </div>'''
    html += '''</div>
        <button class="scroll-bottom-btn" id="nsm-scroll-bottom" title="النزول لآخر رسالة" aria-label="النزول لآخر رسالة">↓</button>
    </div>'''
    st.markdown(html, unsafe_allow_html=True)
    # 🆕 زر «تحميل المزيد» الحقيقي (virtual scroll): يرفع سقف العرض تدريجيًا
    # (40 → 80 → 120…) بدلًا من رسالة سلبية «رسالة مخفية». السقف محفوظ في
    # session_state (`_nsm_chat_display_ceil`) ويعاد إلى NSM_CHAT_DISPLAY_LIMIT
    # تلقائياً عند وصول رسالة جديدة، أما البحث والمفضلة فيتجاوزانه لعرض كل النتائج.
    if _hidden_count and not _search_active:
        st.caption(f"💾 {_hidden_count} رسالة أقدم محفوظة — السجل الكامل غير محذوف")
        _lm_cols = st.columns([1, 2.5, 1])
        with _lm_cols[1]:
            if st.button(
                "⬆️ تحميل المزيد",
                key="nsm_chat_load_more",
                help=f"عرض {_hidden_count} رسالة أقدم إضافية تدريجيًا (السجل الكامل يبقى محفوظًا)",
            ):
                st.session_state["_nsm_chat_display_ceil"] = min(
                    _display_ceil + NSM_CHAT_DISPLAY_INCREMENT,
                    len(_all_msgs_indexed),
                )
                st.rerun()
    st.components.v1.html("""
    <script>
    (function() {
        function scrollToBottom() {
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('nsm-chat-box');
            if (box) { box.scrollTop = box.scrollHeight; return true; }
            return false;
        }
        // Streamlit يعيد رسم الـ DOM بشكل غير متزامن أحياناً — نحاول عدة مرات
        // بدل الاعتماد على تنفيذ واحد فوري قد يسبق اكتمال العنصر.
        let attempts = 0;
        const tryScroll = () => {
            attempts++;
            if (!scrollToBottom() && attempts < 10) {
                setTimeout(tryScroll, 60);
            }
        };
        tryScroll();

        // ── زر "النزول لآخر رسالة": يظهر فقط عندما يكون المستخدم بعيداً
        // عن أسفل الصندوق (بأكثر من 80px)، ويختفي تلقائياً عند الوصول للأسفل ──
        function bindScrollButton() {
            const doc = window.parent ? window.parent.document : document;
            const box = doc.getElementById('nsm-chat-box');
            const btn = doc.getElementById('nsm-scroll-bottom');
            if (!box || !btn) return false;
            if (btn.dataset.nsmBound) { updateVisibility(); return true; }
            btn.dataset.nsmBound = "1";

            function updateVisibility() {
                const distanceFromBottom = box.scrollHeight - box.scrollTop - box.clientHeight;
                btn.classList.toggle('visible', distanceFromBottom > 80);
            }
            box.addEventListener('scroll', updateVisibility, { passive: true });
            btn.addEventListener('click', function() {
                box.scrollTo({ top: box.scrollHeight, behavior: 'smooth' });
            });
            updateVisibility();
            return true;
        }
        let btnAttempts = 0;
        const tryBind = () => {
            btnAttempts++;
            if (!bindScrollButton() && btnAttempts < 10) { setTimeout(tryBind, 60); }
        };
        tryBind();
    })();
    

    // ── اختصار Ctrl+Enter للإرسال ──
    (function() {
        function setupKeyboardShortcut() {
            const doc = window.parent ? window.parent.document : document;
            const textarea = doc.querySelector('textarea[aria-label="سؤالك"]');
            if (textarea && !textarea.dataset.nsmShortcutBound) {
                textarea.dataset.nsmShortcutBound = "1";
                textarea.addEventListener('keydown', function(e) {
                    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
                        e.preventDefault();
                        const sendBtn = doc.querySelector('button[data-testid*="nsm_send"]') || 
                                         doc.querySelector('.st-key-nsm_send_wrap button');
                        if (sendBtn && !sendBtn.disabled) {
                            sendBtn.click();
                        }
                    }
                });
                return true;
            }
            return false;
        }
        let attempts = 0;
        const trySetup = () => {
            attempts++;
            if (!setupKeyboardShortcut() && attempts < 10) {
                setTimeout(trySetup, 100);
            }
        };
        trySetup();
    })();

</script>
    """, height=0)

    # ── تقييم آخر رد (👍/👎) لتغذية autotune_feedback ──
    # يستخدم مكوّن Streamlit الأصلي st.feedback("thumbs") بدل زرَّين
    # منفصلين مكرَّرَين — نفس السلوك (اختيار مرة واحدة ثم يختفي) بكود
    # أقصر وتناسق بصري أفضل مع بقية عناصر الإدخال في Streamlit.
    # المؤشر المُعاد: 0 = 👎، 1 = 👍 (موثّق رسمياً بمرجع st.feedback).
    if _AUTOTUNE_OK:
        _af_turn = st.session_state.get("_af_last_turn")
        if _af_turn and not _af_turn.get("rated"):
            _af_selected = st.feedback("thumbs", key="_af_feedback_widget")
            if _af_selected is not None:
                _af_is_positive = _af_selected == 1
                _heur = _af_compute_heuristics(_af_turn["response"])
                _af_process_feedback(_AFFeedbackRecord(
                    message_id=str(st.session_state.nsm_count),
                    timestamp=datetime.now().timestamp(),
                    context_type=_af_turn["context_type"],
                    model=_af_turn["model"],
                    persona=_af_turn["persona"],
                    params=_af_turn["params"],
                    rating=1 if _af_is_positive else -1,
                    heuristics=vars(_heur),
                ))
                try:
                    from ai.learning_orchestrator import get_orchestrator
                    get_orchestrator().feedback(_af_turn.get("query", ""), is_positive=_af_is_positive)
                except Exception:
                    pass
                _af_turn["rated"] = True
                st.toast("✅ شكراً — تم تسجيل التقييم")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # ✍️ قسم تأليف الرسالة — يجمع كل أدوات الإدخال (إرفاق + كتابة + صوت)
    # في مكان واحد متتابع بدل تفرّقها بين أعلى وأسفل سجل المحادثة
    # ══════════════════════════════════════════════════════════════════
    # 🆕 فاصل "---" مضاف قبل عنوان القسم — كان غائباً رغم أنه فاصل حقيقي
    # بين منطقة سجل المحادثة/التقييم أعلاه وقسم تأليف الرسالة الجديد هنا،
    # بنفس نمط الفواصل المستخدم لباقي الأقسام الرئيسية بهذه الصفحة (راجع
    # "💬 المحادثة الذكية" أعلى الصفحة) وبقية صفحات التطبيق.
    st.markdown("---")
    st.markdown('<div class="section-header">✍️ رسالة جديدة</div>', unsafe_allow_html=True)

    # ── إرفاق ملف أو صورة (multimodal عبر OpenRouter) ─────────────────────
    _or_key_chat = st.session_state.get("_or_api_key", "").strip()
    _or_model_chat = st.session_state.get("_or_model", "google/gemini-2.5-flash")
    _is_vision_chat = _or_model_chat in VISION_MODELS

    with st.expander("📎 إرفاق ملف أو صورة (يتطلب OpenRouter API Key)",
                      expanded=bool(st.session_state["chat_pending_files"])):
        if not _or_key_chat:
            st.info("🔑 أدخل OpenRouter API Key في الشريط الجانبي لتفعيل رفع الملفات والصور.")
        else:
            col_up, col_info = st.columns([3, 2])
            with col_up:
                # مفتاح ديناميكي — يُعاد ضبط عنصر الرفع بعد كل إرسال/مسح
                # حتى لا تُعاد إضافة نفس الملفات القديمة من الـ widget state
                uploaded = st.file_uploader(
                    "اسحب ملفاً هنا أو انقر للاختيار",
                    type=["png", "jpg", "jpeg", "webp", "gif",
                          "pdf", "txt", "md", "csv", "json",
                          "py", "js", "ts", "html", "yaml", "yml"],
                    accept_multiple_files=True,
                    label_visibility="collapsed",
                    key=f"chat_file_uploader_{st.session_state['chat_uploader_version']}",
                )
                if uploaded:
                    existing_names = {f["name"] for f in st.session_state["chat_pending_files"]}
                    for uf in uploaded:
                        if uf.name not in existing_names:
                            extracted = _extract_file(uf)
                            if extracted:
                                st.session_state["chat_pending_files"].append(extracted)
                                existing_names.add(uf.name)
                            else:
                                st.warning(f"⚠ {uf.name} أكبر من {MAX_FILE_MB} MB")
            with col_info:
                if not _is_vision_chat and any(f["is_image"] for f in st.session_state["chat_pending_files"]):
                    st.warning("⚠ النموذج الحالي لا يدعم الصور. اختر نموذج رؤية في الشريط الجانبي.")
                elif _is_vision_chat:
                    st.markdown('<span class="ctx-tag">👁 رؤية مُفعَّلة</span>', unsafe_allow_html=True)
                st.caption(f"الحد الأقصى: {MAX_FILE_MB} MB للملف الواحد")

        if st.session_state["chat_pending_files"]:
            pf_cols = st.columns(min(len(st.session_state["chat_pending_files"]), 4))
            to_remove = []
            for i, f in enumerate(st.session_state["chat_pending_files"]):
                with pf_cols[i % 4]:
                    if f["is_image"] and f.get("raw_bytes"):
                        st.image(f["raw_bytes"], caption=f["name"], use_container_width=True)
                    else:
                        icon = "📄" if f["text_content"] else "📎"
                        st.caption(f"{icon} {f['name']} ({f['size_kb']} KB)")
                    if st.button("✕", key=f"chat_rm_file_{i}", help="حذف"):
                        to_remove.append(i)
            for idx in sorted(to_remove, reverse=True):
                st.session_state["chat_pending_files"].pop(idx)
            if to_remove:
                st.rerun()
            if st.button("🗑 مسح كل الملفات", key="chat_clear_all_files"):
                st.session_state["chat_pending_files"].clear()
                st.session_state["chat_uploader_version"] += 1
                st.rerun()

    # صندوق الإدخال
    st.markdown("""
    <div class="chat-composer-hint">
        <span>✍️ اكتب سؤالك بالعربية أو أرفق ملفاً</span>
        <span>Enter سطر جديد</span>
        <span>Ctrl+Enter إرسال</span>
    </div>
    """, unsafe_allow_html=True)
    if not st.session_state.get("_nsm_input_css_injected"):
        st.session_state["_nsm_input_css_injected"] = True
        st.markdown("""
    <style>
    div[data-testid="stTextArea"] textarea {
        min-height:96px !important;
        max-height:220px !important;
        font-size:1.05rem !important;
        line-height:1.6 !important;
        direction:rtl;
        text-align:right;
        resize:none !important;
        background:var(--surface2) !important;
        border:1.5px solid var(--border) !important;
        border-radius:18px !important;
        padding:0.9rem 1.1rem !important;
        color:var(--text) !important;
        transition:border-color .15s ease, box-shadow .15s ease;
    }
    div[data-testid="stTextArea"] textarea:focus {
        border-color:var(--gold) !important;
        box-shadow:0 0 0 3px var(--gold-soft) !important;
    }
    div[data-testid="stTextArea"] textarea::placeholder {
        color:var(--text-muted);
    }
    .st-key-nsm_send_wrap button {
        height:96px !important;
        border-radius:18px !important;
        background:linear-gradient(135deg,var(--gold),var(--emerald)) !important;
        color:var(--bg) !important;
        font-size:1.02rem !important;
        font-weight:700 !important;
        border:none !important;
        box-shadow:0 3px 12px var(--shadow) !important;
        transition:transform .12s ease, box-shadow .12s ease;
    }
    .st-key-nsm_send_wrap button:hover {
        transform:translateY(-1px);
        box-shadow:0 5px 16px var(--shadow) !important;
    }
    .st-key-nsm_send_wrap button:active {
        transform:translateY(0);
    }
    @media (max-width: 640px) {
        div[data-testid="stTextArea"] textarea {
            min-height:76px !important;font-size:0.98rem !important;
        }
        .st-key-nsm_send_wrap button { height:52px !important; }
    }
    </style>""", unsafe_allow_html=True)
    c1, c2 = st.columns([5, 1.2], gap="small")
    with c1:
        # مفتاح مُرقَّم (versioned key): لا يمكن تعديل session_state لويدجت
        # بعد إنشائه بنفس تشغيل السكربت (يرمي StreamlitAPIException)، لذلك
        # لا يمكن مسح المربع بتعيين st.session_state["nsm_input"] = "" من
        # _process() لاحقاً بنفس الطريقة. الحل الآمن: تغيير الـkey نفسه بعد
        # كل إرسال (_nsm_input_version += 1) — فيبدأ الويدجت الجديد فارغاً
        # تلقائياً في الـrerun التالي دون لمس قيمة المفتاح القديم إطلاقاً.
        st.session_state.setdefault("_nsm_input_version", 0)
        _input_key = f"nsm_input_v{st.session_state['_nsm_input_version']}"
        user_input = st.text_area(
            label="سؤالك",
            placeholder="اكتب سؤالك هنا… (Enter = سطر جديد)",
            key=_input_key,
            label_visibility="collapsed",
            height=96,
        )
    with c2:
        with st.container(key="nsm_send_wrap"):
            send = st.button("➤\nإرسال", key="nsm_send", use_container_width=True)

    # ── الواجهة الصوتية: تسجيل سؤال بالصوت + قراءة الردود صوتياً ─────────
    voice_col1, voice_col2 = st.columns([3, 2], gap="small")
    _voice_query = None
    with voice_col1:
        if _STT_OK:
            _mic_audio = st.audio_input("🎤 أو سجّل سؤالك صوتياً", key="nsm_mic_input")
            if _mic_audio is not None:
                _mic_bytes = _mic_audio.getvalue()
                _mic_hash = hash(_mic_bytes)
                if st.session_state.get("_nsm_last_mic_hash") != _mic_hash:
                    st.session_state["_nsm_last_mic_hash"] = _mic_hash
                    with st.spinner("⟳ جارٍ تفريغ الصوت..."):
                        _transcribed, _stt_err = _stt_transcribe(_mic_bytes, mime_type="audio/wav")
                    if _stt_err:
                        st.warning(f"⚠️ {_stt_err}")
                    elif _transcribed:
                        _voice_query = _transcribed
        else:
            st.caption("🎤 الإدخال الصوتي غير متاح حالياً")
    with voice_col2:
        _voice_output_on = st.toggle(
            "🔊 قراءة الردود صوتياً", key="_nsm_voice_output",
            value=st.session_state.get("_nsm_voice_output", False),
            disabled=not _TTS_OK,
        )

    # أدوات المحادثة: مسح / إعادة توليد آخر رد / تصدير — بجانب بعضها
    # (لا بعد أدوات المالك)
    st.caption("🛠️ أدوات المحادثة")
    _has_msgs = bool(st.session_state.nsm_messages)
    _last_is_nsm = _has_msgs and st.session_state.nsm_messages[-1][0] == "nsm"
    _tool_col1, _tool_col2, _tool_col3 = st.columns(3)
    with _tool_col1:
        if st.button("🗑 مسح المحادثة", key="nsm_clear", use_container_width=True, disabled=not _has_msgs):
            st.session_state.nsm_messages = []
            st.session_state.nsm_count = 0
            bot.clear_history()
            st.rerun()
    with _tool_col2:
        if st.button("🔄 إعادة توليد آخر رد", key="nsm_regenerate",
                      use_container_width=True, disabled=not _last_is_nsm):
            # ابحث عن آخر رسالة مستخدم قبل آخر رد NSM، احذف الرد القديم
            # (وكاش صوته إن وُجد)، ثم أعد إرسال نفس السؤال بدون تكرار
            # فقاعة المستخدم (add_user_msg=False بمعالج _process بالأسفل).
            _last_user_text = None
            for _m in reversed(st.session_state.nsm_messages[:-1]):
                if _m[0] == "user":
                    _last_user_text = _m[1]
                    break
            if _last_user_text:
                _popped_idx = len(st.session_state.nsm_messages) - 1
                st.session_state.nsm_messages.pop()
                st.session_state.get("_nsm_audio_cache", {}).pop(_popped_idx, None)
                st.session_state["_chat_regenerate_pending"] = _last_user_text
                st.rerun()
    with _tool_col3:
        if _has_msgs:
            _export_lines = []
            for _m in st.session_state.nsm_messages:
                _role_label = "أنت" if _m[0] == "user" else "NSM"
                _export_lines.append(f"[{_m[4]}] {_role_label}: {_m[1]}")
            st.download_button(
                "⬇️ تصدير المحادثة", data="\n\n".join(_export_lines),
                file_name=f"nsm_chat_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                mime="text/plain", key="nsm_export", use_container_width=True,
            )
        else:
            st.button("⬇️ تصدير المحادثة", key="nsm_export_disabled",
                       use_container_width=True, disabled=True)

    # ── أزرار تحليل المشروع (NSM Agent) — للمالك فقط ─────────────
    # الأوامر خلف هذه الأزرار (افحص/عدل/أنشئ/ارفع) تقرأ/تكتب ملفات فعلية
    # على الخادم وتنفّذ git push — عُطِّلت من nsm_chat.py لغير المالك،
    # ونخفي الواجهة نفسها هنا حتى لا تظهر أزرار بلا فائدة للزائر العادي.
    if st.session_state.get("_dev_console_unlocked", False):
        st.markdown("---")
        st.markdown("**🤖 تحليل المشروع (وضع المالك):**")
        agent_cols = st.columns(6)
        agent_btns = [
            ("📋 اقترح (كل)",      "اقترح"),
            ("🗂 غير مستخدم",      "اقترح غير مستخدم"),
            ("⚠️ أخطاء",           "اقترح أخطاء"),
            ("📦 ملفات كبيرة",     "اقترح كبير"),
            ("📁 قائمة الملفات",   "قائمة"),
            ("🔁 مكررة",           "اقترح مكررة"),
        ]
        for i, (label, cmd) in enumerate(agent_btns):
            with agent_cols[i]:
                if st.button(label, key=f"agent_btn_{i}", use_container_width=True):
                    st.session_state._chat_pending = cmd

        # أزرار تحليل ملف محدد
        st.markdown("**🔍 تحليل ملف محدد** — اكتب المسار ثم اختر العملية:")
        file_path_input = st.text_input(
            "مسار الملف", placeholder="مثال: ai/code_agent.py",
            key="agent_file_path", label_visibility="collapsed"
        )
        if file_path_input.strip():
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                if st.button("📄 ملخص", key="btn_summary", use_container_width=True):
                    st.session_state._chat_pending = f"ملخص {file_path_input.strip()}"
            with fc2:
                if st.button("🔧 صحح", key="btn_fix", use_container_width=True):
                    st.session_state._chat_pending = f"صحح {file_path_input.strip()}"
            with fc3:
                if st.button("👁 افحص", key="btn_inspect", use_container_width=True):
                    st.session_state._chat_pending = f"افحص {file_path_input.strip()}"

    # معالجة الإدخال
    def _process(text: str, add_user_msg: bool = True):
        files = list(st.session_state["chat_pending_files"])
        if not text.strip() and not files:
            return

        # فرّغ مربع الإدخال فور الإرسال — قبل أي rerun. نُغيّر رقم إصدار
        # مفتاح الويدجت بدل تعديل session_state["nsm_input"] مباشرة، لأن
        # Streamlit يمنع تعديل قيمة ويدجت بعد إنشائه بنفس تشغيل السكربت
        # (والويدجت أُنشئ فعلاً أعلى هذه الدالة بنفس التشغيل). التغيير هنا
        # يجعل الـrerun التالي يستخدم مفتاحاً جديداً بقيمة افتراضية فارغة.
        st.session_state["_nsm_input_version"] = st.session_state.get("_nsm_input_version", 0) + 1

        st.session_state["chat_pending_files"] = []
        st.session_state["chat_uploader_version"] += 1

        display_text = text.strip()
        if files:
            names = ", ".join(f["name"] for f in files)
            display_text += f"\n\n📎 {names}"

        _ts = datetime.now().strftime("%H:%M")

        # ── أضف رسالة المستخدم فوراً (يُتخطّى عند إعادة توليد رد سابق،
        # لأن رسالة المستخدم موجودة أصلاً بالسجل ولا يجب تكرارها) ──
        if add_user_msg:
            st.session_state.nsm_messages.append(("user", display_text, "", "", _ts))
            _persist_chat_message(st.session_state.nsm_chat_session_id, "user", display_text)
            # 🆕 رسالة جديدة تعني النزول للأسفل — نعيد سقف العرض المتدرج
            # إلى NSM_CHAT_DISPLAY_LIMIT حتى لا تظهر رسالة الأحدث وسط
            # السجل القديم الموسّع (مثل تمرير المحادثات الحديثة).
            st.session_state["_nsm_chat_display_ceil"] = NSM_CHAT_DISPLAY_LIMIT

        # ── فحص أمان أولي (regex محلي، بدون تكلفة API) ──
        _safety_msg = _nsm_safety_gate(text.strip())
        if _safety_msg:
            st.session_state.nsm_messages.append(("nsm", _safety_msg, "", "🛡️ فحص أمان", datetime.now().strftime("%H:%M")))
            _persist_chat_message(st.session_state.nsm_chat_session_id, "nsm", _safety_msg, "🛡️ فحص أمان")
            st.session_state.nsm_count += 1
            st.rerun()
            return

        # ── كاش الردود المتعلَّمة (ConversationLearner عبر LearningOrchestrator) ──
        # يوفّر زمن استجابة وحصة LLM المجانية (Groq/Gemini/Cloudflare) عند
        # تكرار نفس السؤال حرفياً فقط. نتجاهل عمداً المطابقة التقريبية
        # بالكلمات المفتاحية الموجودة داخل recall() الأصلية (source="learned")
        # لأنها قد تُرجع إجابة سؤال مختلف بثقة زائفة — نقبل فقط
        # source="cache" (تطابق كامل لنص السؤال).
        try:
            from ai.learning_orchestrator import get_orchestrator
            _cached = get_orchestrator().recall(text.strip(), min_quality=0.75)
        except Exception:
            _cached = None
        if _cached and _cached.get("source") == "cache" and (_cached.get("answer") or "").strip():
            st.session_state.nsm_messages.append((
                "nsm", _cached["answer"], "", "⚡ كاش متعلَّم", datetime.now().strftime("%H:%M")
            ))
            _persist_chat_message(
                st.session_state.nsm_chat_session_id, "nsm", _cached["answer"], "⚡ كاش متعلَّم"
            )
            st.session_state.nsm_count += 1
            st.rerun()
            return

        # ════════════════════════════════════════════════════════════════════
        # [1] بناء قائمة العقد المتاحة فعلاً
        # ════════════════════════════════════════════════════════════════════
        import time as _time_mod

        _or_key_p = st.session_state.get("_or_api_key", "").strip()
        _available_nodes: list = []
        if _or_key_p:
            _available_nodes.append("nsm:openrouter")

        # فحص NSM Agent مبكراً قبل قرار التوجيه
        _agent = None
        try:
            from ai.nsm_agent_core import NSMAgent as _AgentCls
            _agent = getattr(st.session_state, "_nsm_agent_instance", None)
            if _agent is None:
                _agent = _AgentCls()
                st.session_state._nsm_agent_instance = _agent
            _agent.available = _agent._check_available()
            if _agent.available:
                _available_nodes.append("nsm:agent")
        except Exception:
            _agent = None
        _available_nodes.append("nsm:free_router")   # دائماً متاح

        # ════════════════════════════════════════════════════════════════════
        # [2] التوجيه الدلالي — صنّف الاستعلام وانحَز للعقدة الأنسب
        # ════════════════════════════════════════════════════════════════════
        _sem_category   = "general"
        _sem_confidence = 0.2
        _sem_biased     = list(_available_nodes)
        if _NSM_SEMANTIC_OK and _nsm_semantic:
            try:
                _sem_category, _sem_confidence = _nsm_semantic.classify(text.strip())
                _sem_biased = _nsm_semantic.bias_order(
                    _sem_category, _available_nodes, _sem_confidence
                )
            except Exception:
                pass

        # ════════════════════════════════════════════════════════════════════
        # [3] اختَر العقدة (تاريخي 65% + دلالي 35%)
        # ════════════════════════════════════════════════════════════════════
        if _NSM_BRIDGE_OK and _nsm_bridge:
            _selected_node = _nsm_bridge.select_node_with_semantic(
                text.strip(), _sem_biased, _sem_category, _sem_confidence
            )
        else:
            _selected_node = _sem_biased[0]

        # ════════════════════════════════════════════════════════════════════
        # [4] حلقة تنفيذ مع Failover تلقائي (حتى 2 إعادة توجيه)
        # ════════════════════════════════════════════════════════════════════
        _excluded_nodes: list = []
        _response       = ""
        _ctx_tag        = ""
        _src_badge      = "🤖 NSM"
        _af_params_last = dict(_AF_NEUTRAL_PARAMS) if _AUTOTUNE_OK else {"temperature": 0.7}
        _af_ctx_last    = "conversational"
        _or_model_last  = st.session_state.get("_or_model", "google/gemini-2.5-flash")
        _final_node     = _selected_node
        _total_latency  = 0.0

        for _attempt in range(len(_available_nodes)):
            # Failover: اختر التالية إذا فشلت السابقة
            if _attempt > 0:
                if _NSM_BRIDGE_OK and _nsm_bridge:
                    _selected_node = _nsm_bridge.select_next_node(_available_nodes, _excluded_nodes)
                else:
                    _rem = [n for n in _available_nodes if n not in _excluded_nodes]
                    _selected_node = _rem[0] if _rem else "nsm:free_router"
                _final_node = _selected_node

                # مؤشر إعادة التوجيه للمستخدم
                st.toast(f"🔄 إعادة توجيه تلقائي → {_selected_node.replace('nsm:','')}", icon="⚡")

            _t0_route = _time_mod.time()
            _attempt_success = False

            # ── تنفيذ العقدة المختارة ─────────────────────────────────────
            if _selected_node == "nsm:openrouter" and _or_key_p:
                # ── مسار OpenRouter ──────────────────────────────────────
                _or_model_p = st.session_state.get("_or_model", "google/gemini-2.5-flash")
                _or_model_last = _or_model_p
                can_vision  = _or_model_p in VISION_MODELS
                doc_files   = [f for f in files if not f["is_image"]]
                image_files = [f for f in files if f["is_image"]] if can_vision else []
                user_content = _build_user_content(text.strip(), doc_files, image_files)
                history_msgs = []
                # سقف: نُرسل آخر NSM_CHAT_DISPLAY_LIMIT رسالة فقط كسياق بدل
                # السجل الكامل — بلا هذا، كل رسالة جديدة في محادثة طويلة
                # تُرسل توكنزاً متزايدة بلا حدود لمزوّد OpenRouter (تكلفة
                # متصاعدة + خطر تجاوز نافذة السياق فعلياً في المحادثات
                # الطويلة جداً). السجل الكامل يبقى محفوظاً في nsm_messages.
                for m in st.session_state.nsm_messages[:-1][-NSM_CHAT_DISPLAY_LIMIT:]:
                    role = "user" if m[0] == "user" else "assistant"
                    history_msgs.append({"role": role, "content": m[1]})
                # 🆕 ذاكرة المحادثة الطويلة: عند تجاوز NSM_CHAT_MEMORY_SUMMARY_AT
                # رسالة، يُلحق الملخص السياقي للأقدم كرسالة system إضافية قبل
                # النافذة الأخيرة — يحفظ خيوط الموضوع بلا نمو توكنات بلا حدود.
                # السجل الكامل يبقى محفوظاً في nsm_messages وchat_history_store.
                _ctx_summary = build_chat_memory_summary(st.session_state.nsm_messages[:-1])
                if _ctx_summary:
                    history_msgs.insert(0, {"role": "system", "content": _ctx_summary})
                # 🆕 التفكير متعدد الخطوات: للسؤال المعقد (مقارنة، سببية،
                # عملية، تعداد، تحليل، مركّب...) يُبنى مخطط تفكيك حتمي بلا
                # API ويُلحق كرسالة نظام توجه النموذج للإجابة وفق خطة
                # مرتبة من خطوات — يُحقن قبل النافذة الأخيرة والنموذج هو
                # من ينفذ الإجابة فعليًا. أي فشل أو سؤال بسيط يعيد السلوك
                # الأصلي (سؤال واحد → رد واحد).
                _msr_plan = None
                if _MSR_OK:
                    try:
                        _msr_plan = _plan_system_prompt(text.strip()) if _plan_system_prompt else None
                    except Exception:
                        _msr_plan = None
                if _msr_plan:
                    history_msgs.insert(0, {"role": "system", "content": _msr_plan})
                api_messages = history_msgs + [{"role": "user", "content": user_content}]

                _af_params  = dict(_AF_NEUTRAL_PARAMS) if _AUTOTUNE_OK else {"temperature": 0.7, "top_p": 0.9}
                _af_ctx     = "conversational"
                _af_note    = ""
                if _AUTOTUNE_OK:
                    try:
                        _af_params, _, _af_note = _af_apply_adjustments(_af_params, _af_ctx)
                    except Exception:
                        pass
                _af_params_last = _af_params
                _af_ctx_last    = _af_ctx

                full_response = ""
                with st.chat_message("assistant", avatar="🌐"):
                    placeholder = st.empty()
                    try:
                        for chunk in _or_stream(
                            api_messages, model=_or_model_p, api_key=_or_key_p,
                            temperature=_af_params.get("temperature", 0.7),
                            top_p=_af_params.get("top_p", 0.9),
                        ):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        _attempt_success = bool(full_response.strip())
                    except Exception:
                        placeholder.markdown(full_response or "⚠️ خطأ في OpenRouter — جاري الإعادة...")
                    if _af_note:
                        st.caption(_af_note)

                _response  = full_response
                _ctx_tag   = ""
                _src_badge = f"🌐 OpenRouter · {_or_model_p.split('/')[-1]}"
                if _msr_plan:
                    _src_badge += " · 🧭 مخططة عبر خطوات"

            elif _selected_node == "nsm:agent" and _agent and _agent.available:
                # ── مسار NSM Agent — Streaming ──────────────────────────
                full_response = ""
                with st.chat_message("assistant", avatar="🧠"):
                    placeholder = st.empty()
                    try:
                        for chunk in _agent.run_stream(text.strip()):
                            full_response += chunk
                            placeholder.markdown(full_response + "▌")
                        placeholder.markdown(full_response)
                        _attempt_success = bool(full_response.strip())
                    except Exception:
                        placeholder.markdown(full_response or "⚠️ خطأ في NSM Agent — جاري الإعادة...")
                if hasattr(bot, "_last_source"):
                    bot._last_source = "nsm_agent"
                _response  = full_response.replace("⏳ *أفكر...\n\n", "", 1)
                _ctx_tag   = bot.context_info() if hasattr(bot, "context_info") else ""
                _src_badge = bot.source_badge() if hasattr(bot, "source_badge") else "🧠 NSM Agent"
                # 🆕 شارات التعلم المؤسسي: تلميح خبرة توجيه مثبتة + درس
                # ذاكرة الأخطاء + نتيجة اتساق الفريق — تظهر أسفل الرد مباشرة
                _learn_badges: List[str] = []
                try:
                    _rex_h = st.session_state.pop("_nsm_rex_hint", None)
                    if _rex_h and str(_rex_h).strip() and "لا توجد خبرة" not in str(_rex_h):
                        _learn_badges.append(str(_rex_h)[:260])
                    _rm_h = st.session_state.pop("_nsm_rm_hint", None)
                    if _rm_h and str(_rm_h).strip():
                        _learn_badges.append(str(_rm_h)[:260])
                    _tc_res = st.session_state.pop("_nsm_tc_result", None)
                    if isinstance(_tc_res, dict) and _tc_res.get("warning"):
                        _learn_badges.append("⚖️ " + str(_tc_res["warning"])[:260])
                except Exception:
                    pass
                if _learn_badges:
                    st.caption("\n———\n".join(_learn_badges))

            else:
                # ── مسار free_router (الاحتياطي الأخير) ──────────────────
                with st.chat_message("assistant", avatar="🧠"):
                    _typing_ph = st.empty()
                    _typing_ph.markdown(
                        '''<div class="typing-wrap">
                            <span class="thinking-ring">🧠</span>
                            <span class="typing-dots"><span></span><span></span><span></span></span>
                        </div>''',
                        unsafe_allow_html=True,
                    )
                    try:
                        _resp_raw = bot.chat(text.strip(), system_prompt=NSM_SYSTEM_PROMPT)
                        _attempt_success = bool(_resp_raw and _resp_raw.strip())
                    except Exception:
                        _resp_raw = "⚠️ تعذّر الحصول على رد."
                    _typing_ph.empty()
                _response  = _resp_raw
                _ctx_tag   = bot.context_info() if hasattr(bot, "context_info") else ""
                _src_badge = bot.source_badge() if hasattr(bot, "source_badge") else "⚡ Free Router"

            # ── قياس الزمن + تقييم الجودة + تسجيل النتيجة ────────────────
            _latency_ms = (_time_mod.time() - _t0_route) * 1000
            _total_latency += _latency_ms

            # التقييم الثنائي القديم (فارغ/غير فارغ) يبقى كحد أدنى أولي،
            # ثم نُدقّقه بتقييم الجودة الحقيقي إن كان متاحاً
            _quality: dict = {}
            if _attempt_success and _QUALITY_SCORER_OK:
                try:
                    _quality = _score_response(text.strip(), _response)
                    _attempt_success = bool(_quality.get("is_quality", True))
                except Exception:
                    _quality = {}

            if _NSM_BRIDGE_OK and _nsm_bridge:
                _nsm_bridge.record_result(_selected_node, _attempt_success, _latency_ms)

            # ── سجل التوجيه الحي (آخر 100 قرار) ──────────────────────────
            _sem_icon = ""
            if _NSM_SEMANTIC_OK and _nsm_semantic:
                try:
                    _sem_icon = _nsm_semantic.CATEGORY_LABELS.get(_sem_category, ("💬", ""))[0]
                except Exception:
                    _sem_icon = "💬"
            _route_entry = {
                "ts":         datetime.now().strftime("%H:%M:%S"),
                "query":      text.strip()[:55] + ("…" if len(text.strip()) > 55 else ""),
                "category":   _sem_category,
                "cat_icon":   _sem_icon,
                "confidence": round(_sem_confidence, 2),
                "node":       _selected_node,
                "latency_ms": round(_latency_ms),
                "success":    _attempt_success,
                "attempt":    _attempt + 1,
                "failover":   _attempt > 0,
                "quality_score": _quality.get("score"),
            }
            _rlog = st.session_state.setdefault("nsm_route_log", [])
            _rlog.append(_route_entry)
            if len(_rlog) > 100:
                st.session_state["nsm_route_log"] = _rlog[-100:]
            if _ROUTE_LOG_DB_OK:
                _rlog_append(_route_entry)

            if _attempt_success:
                break   # نجاح — توقّف
            _excluded_nodes.append(_selected_node)

        # ════════════════════════════════════════════════════════════════════
        # [5] حفظ + إظهار الرد النهائي
        # ════════════════════════════════════════════════════════════════════
        _source_key = "chat_openrouter" if "openrouter" in _final_node else "chat_nsm_agent"
        _record_chat_episode(text.strip(), _response, source=_source_key)
        # 🆕 التعلّم الذاتي المستمر: كل رد ناجح يُسجَّل ذكرى في الذاكرة
        # الطويلة (سؤال + جوابه) — عند تكرار السؤال تستحضره المنصة لاحقًا.
        # تصويب/رفض الرد (جودة منخفضة) يسجّل ذكرى weak لتجنّب نفس النمط.
        # + مزامنة نادرة (كل 25 ردًا ناجحًا) لأفضل الدروس الجماعية + تآكل.
        if _LTM_OK and _attempt_success:
            try:
                _ltm_turns = st.session_state.setdefault("_ltm_turns", 0) + 1
                st.session_state["_ltm_turns"] = _ltm_turns
                _ltm_learn = _get_ltm()
                _q_type = "correction" if (_quality.get("score") or 0.0) < 0.4 else "question"
                _ltm_learn.learn(text.strip(), _response[:400], memory_type=_q_type,
                                 quality=0.8 if _q_type == "correction" else 0.5)
                if _ltm_turns % 25 == 0:
                    _ltm_learn.ingest_collective_lessons()
                if _ltm_turns % 10 == 0:
                    _ltm_learn.decay()
            except Exception:
                pass  # فشل التعلّم لا يمنع عرض الرد إطلاقًا
        st.session_state.nsm_messages.append((
            "nsm", _response, _ctx_tag, _src_badge, datetime.now().strftime("%H:%M")
        ))
        _persist_chat_message(
            st.session_state.nsm_chat_session_id, "nsm", _response, _src_badge
        )
        _msg_idx = len(st.session_state.nsm_messages) - 1
        if _TTS_OK and st.session_state.get("_nsm_voice_output") and _response.strip():
            try:
                with st.spinner("⟳ جارٍ تحويل الرد لصوت..."):
                    _tts_result = _TTSEngineCls().synthesize(_response.strip())
                if _tts_result.ok:
                    import base64 as _b64
                    _audio_cache = st.session_state.setdefault("_nsm_audio_cache", {})
                    _audio_cache[_msg_idx] = (
                        _b64.b64encode(_tts_result.audio_bytes).decode("ascii"),
                        _tts_result.format,
                    )
            except Exception:
                pass  # فشل TTS لا يجب أن يُعطّل عرض الرد النصي
        if _AUTOTUNE_OK:
            st.session_state["_af_last_turn"] = {
                "response": _response, "params": _af_params_last,
                "context_type": _af_ctx_last,
                "model": _or_model_last if "openrouter" in _final_node else _src_badge,
                "persona": "nsm", "rated": False,
                "query": text.strip(),
            }
        st.session_state.nsm_count += 1
        st.rerun()

    if send and (user_input or st.session_state["chat_pending_files"]):
        _process(user_input)

    if _voice_query:
        _process(_voice_query)

    if hasattr(st.session_state, "_chat_pending"):
        q = st.session_state._chat_pending
        del st.session_state._chat_pending
        _process(q)

    if hasattr(st.session_state, "_chat_regenerate_pending"):
        q = st.session_state._chat_regenerate_pending
        del st.session_state._chat_regenerate_pending
        _process(q, add_user_msg=False)
