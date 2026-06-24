"""
NSM Chat UI v2 — واجهة Streamlit مع ذاكرة المحادثة
=====================================================
streamlit run nsm_chat_ui.py
"""
import sys
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="NSM Chat",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Noto Naskh Arabic', 'Segoe UI', sans-serif;
    background-color: #0e1117;
    color: #fafafa;
}
.nsm-header {
    text-align: center;
    padding: 1.4rem 0 0.5rem;
    background: linear-gradient(135deg,#1a1a2e,#16213e,#0f3460);
    border-radius: 16px;
    margin-bottom: 1.2rem;
    border: 1px solid #2d3561;
}
.nsm-header h1 { font-size:1.9rem; color:#e0b858; margin:0; }
.nsm-header p  { color:#a0aec0; font-size:0.88rem; margin:0.2rem 0 0; }

.user-bubble { display:flex; justify-content:flex-end; margin:0.5rem 0; }
.user-bubble .bubble {
    background: linear-gradient(135deg,#1a73e8,#0d47a1);
    color:#fff; padding:0.7rem 1.1rem;
    border-radius:18px 18px 4px 18px;
    max-width:74%; font-size:1rem; line-height:1.6;
    text-align:right; direction:rtl;
    box-shadow:0 2px 8px rgba(26,115,232,.3);
}
.nsm-bubble { display:flex; justify-content:flex-start; margin:0.5rem 0; gap:0.5rem; align-items:flex-start; }
.nsm-bubble .avatar { font-size:1.5rem; margin-top:3px; flex-shrink:0; }
.nsm-bubble .bubble {
    background: linear-gradient(135deg,#1e2a3a,#162032);
    color:#e2e8f0; padding:0.7rem 1.1rem;
    border-radius:18px 18px 18px 4px;
    max-width:74%; font-size:1rem; line-height:1.7;
    text-align:right; direction:rtl;
    border:1px solid #2d4a6e;
    box-shadow:0 2px 8px rgba(0,0,0,.3);
}
.ctx-badge {
    display:inline-block; background:#1a2332;
    border:1px solid #2d4a6e; border-radius:20px;
    padding:0.18rem 0.7rem; font-size:0.75rem;
    color:#90cdf4; margin:0.1rem 0 0.3rem 0;
    direction:rtl; text-align:right;
}
.stTextInput > div > div > input {
    background:#1a2332 !important; border:1.5px solid #2d4a6e !important;
    border-radius:12px !important; color:#fafafa !important;
    font-family:'Noto Naskh Arabic',sans-serif !important;
    font-size:1rem !important; padding:0.65rem 1rem !important;
    text-align:right !important; direction:rtl !important;
}
.stTextInput > div > div > input:focus {
    border-color:#1a73e8 !important;
    box-shadow:0 0 0 2px rgba(26,115,232,.25) !important;
}
.stButton > button {
    background:linear-gradient(135deg,#1a73e8,#0d47a1) !important;
    color:#fff !important; border:none !important;
    border-radius:10px !important;
    font-family:'Noto Naskh Arabic',sans-serif !important;
    font-size:0.92rem !important; padding:0.48rem 1rem !important;
    transition:opacity .2s !important;
}
.stButton > button:hover { opacity:.85 !important; }
.stat-box {
    background:#1a2332; border-radius:10px; padding:0.6rem 0.8rem;
    margin:0.3rem 0; border:1px solid #2d4a6e;
    text-align:center; color:#a0aec0; font-size:0.85rem;
}
.stat-box span { color:#e0b858; font-size:1.25rem; font-weight:700; display:block; }
.chat-wrap {
    max-height:520px; overflow-y:auto; padding:0.8rem;
    background:#0a0f1a; border-radius:14px;
    border:1px solid #1e2a3a; margin-bottom:0.8rem;
}
.welcome { text-align:center; color:#2d4a6e; padding:2.5rem 1rem; font-size:1.05rem; }
</style>
""", unsafe_allow_html=True)

# ── تحميل النموذج ─────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

@st.cache_resource(show_spinner="⟳ تحميل NSM Chat...")
def load_model():
    from nsm_chat import NSMChat, _KB
    return NSMChat, len(_KB)

try:
    NSMChat, topics_count = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    model_error = str(e)

# ── تهيئة الجلسة ─────────────────────────────────────────────
if "bot" not in st.session_state:
    if model_ok:
        st.session_state.bot = NSMChat()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

# ── الرأس ────────────────────────────────────────────────────
st.markdown("""
<div class="nsm-header">
    <h1>🧠 NSM Chat</h1>
    <p>مساعد ذكي بذاكرة محادثة — بدون قاعدة بيانات</p>
    <p style="color:#4a9eff;font-size:0.78rem;">
        Neural Service Mesh · Trained Embedding 784×128 · Context Memory · Arabic NLP
    </p>
</div>
""", unsafe_allow_html=True)

# ── الشريط الجانبي ────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NSM Chat")
    st.markdown("---")

    st.markdown("**📊 إحصائيات**")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f'<div class="stat-box"><span>{topics_count if model_ok else "—"}</span>موضوع</div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="stat-box"><span>{st.session_state.msg_count}</span>رسالة</div>', unsafe_allow_html=True)

    st.markdown('<div class="stat-box"><span>784×128</span>Trained Embedding</div>', unsafe_allow_html=True)
    st.markdown('<div class="stat-box"><span>90.5%</span>دقة التمييز</div>', unsafe_allow_html=True)

    # السياق الحالي
    if model_ok and "bot" in st.session_state:
        ctx = st.session_state.bot.context_info()
        if ctx:
            st.markdown("---")
            st.markdown("**📎 السياق الحالي**")
            st.markdown(f'<div class="ctx-badge">{ctx}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**⚡ أسئلة سريعة**")

    quick_questions = [
        "ما هي أركان الإسلام؟",
        "ما هو الذكاء الاصطناعي؟",
        "ما هي سورة الفاتحة؟",
        "ما هو الجبر الخطي؟",
        "من هم الخلفاء الراشدون؟",
        "ما هي لغة Python؟",
        "ما هي اللغة العربية؟",
        "ما هي سورة الكهف؟",
        "ما هو الـ Transformer؟",
        "ما هي التغذية السليمة؟",
    ]
    for q in quick_questions:
        if st.button(q, key=f"qk_{q}", use_container_width=True):
            st.session_state._pending = q

    st.markdown("---")
    if st.button("🗑 مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.msg_count = 0
        if "bot" in st.session_state:
            st.session_state.bot.clear_history()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#2d4a6e;font-size:0.75rem;">
        NSM · Neural Service Mesh<br>
        numpy only · no database · context-aware<br>
        © علي أحمد
    </div>
    """, unsafe_allow_html=True)

# ── عرض المحادثة ─────────────────────────────────────────────
if not model_ok:
    st.error(f"❌ خطأ: {model_error}")
    st.stop()

chat_html = '<div class="chat-wrap">'
if not st.session_state.messages:
    chat_html += """
    <div class="welcome">
        🧠<br><br>
        مرحباً! أنا NSM مساعدك الذكي<br>
        <span style="font-size:0.88rem;color:#1a2d4a;">
            أتذكر سياق محادثتنا — اسألني ما تشاء
        </span>
    </div>"""
else:
    for role, text, ctx in st.session_state.messages:
        if role == "user":
            chat_html += f'<div class="user-bubble"><div class="bubble">{text}</div></div>'
        else:
            ctx_html = f'<div class="ctx-badge">📎 {ctx}</div>' if ctx else ""
            chat_html += f"""
            <div class="nsm-bubble">
                <div class="avatar">🧠</div>
                <div class="bubble">{ctx_html}{text}</div>
            </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# ── صندوق الإدخال ─────────────────────────────────────────────
col_in, col_btn = st.columns([5, 1])
with col_in:
    user_input = st.text_input(
        label="",
        placeholder="اكتب سؤالك... (وكم ركعاتها؟ / وكيف يعمل؟)",
        key="user_input",
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("إرسال ➤", use_container_width=True)

# ── معالجة الرسائل ────────────────────────────────────────────
def process(text: str):
    if not text.strip():
        return
    bot = st.session_state.bot
    response = bot.chat(text.strip())
    ctx = bot.context_info()
    st.session_state.messages.append(("user", text.strip(), ""))
    st.session_state.messages.append(("nsm", response, ctx))
    st.session_state.msg_count += 1
    st.rerun()

if send and user_input:
    process(user_input)

if hasattr(st.session_state, "_pending"):
    q = st.session_state._pending
    del st.session_state._pending
    process(q)

if user_input and user_input != st.session_state.get("_last", ""):
    st.session_state["_last"] = user_input
