"""
NSM Chat UI — واجهة Streamlit للمحادثة الذكية
===============================================
ضعه في جذر مشروع NSM بجانب nsm_chat.py و weights_784x784.csv
شغّله: streamlit run nsm_chat_ui.py
"""
import sys
from pathlib import Path
import streamlit as st

# ── إعداد الصفحة ────────────────────────────────────────────────
st.set_page_config(
    page_title="NSM Chat — المساعد الذكي",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────
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
    padding: 1.5rem 0 0.5rem 0;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 1.5rem;
    border: 1px solid #2d3561;
}
.nsm-header h1 { font-size: 2rem; color: #e0b858; margin: 0; }
.nsm-header p  { color: #a0aec0; font-size: 0.95rem; margin: 0.3rem 0 0 0; }

/* فقاعة المستخدم */
.user-bubble {
    display: flex;
    justify-content: flex-end;
    margin: 0.6rem 0;
}
.user-bubble .bubble {
    background: linear-gradient(135deg, #1a73e8, #0d47a1);
    color: white;
    padding: 0.75rem 1.2rem;
    border-radius: 18px 18px 4px 18px;
    max-width: 72%;
    font-size: 1rem;
    line-height: 1.6;
    text-align: right;
    direction: rtl;
    box-shadow: 0 2px 8px rgba(26,115,232,0.3);
}

/* فقاعة NSM */
.nsm-bubble {
    display: flex;
    justify-content: flex-start;
    margin: 0.6rem 0;
    align-items: flex-start;
    gap: 0.6rem;
}
.nsm-bubble .avatar {
    font-size: 1.6rem;
    margin-top: 2px;
    flex-shrink: 0;
}
.nsm-bubble .bubble {
    background: linear-gradient(135deg, #1e2a3a, #162032);
    color: #e2e8f0;
    padding: 0.75rem 1.2rem;
    border-radius: 18px 18px 18px 4px;
    max-width: 72%;
    font-size: 1rem;
    line-height: 1.7;
    text-align: right;
    direction: rtl;
    border: 1px solid #2d4a6e;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}

/* صندوق الإدخال */
.stTextInput > div > div > input {
    background: #1a2332 !important;
    border: 1.5px solid #2d4a6e !important;
    border-radius: 12px !important;
    color: #fafafa !important;
    font-family: 'Noto Naskh Arabic', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.7rem 1rem !important;
    text-align: right !important;
    direction: rtl !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a73e8 !important;
    box-shadow: 0 0 0 2px rgba(26,115,232,0.25) !important;
}

/* أزرار */
.stButton > button {
    background: linear-gradient(135deg, #1a73e8, #0d47a1) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Noto Naskh Arabic', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.5rem 1.2rem !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* شريط جانبي */
.sidebar-stat {
    background: #1a2332;
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin: 0.4rem 0;
    border: 1px solid #2d4a6e;
    text-align: center;
    color: #a0aec0;
    font-size: 0.9rem;
}
.sidebar-stat span { color: #e0b858; font-size: 1.3rem; font-weight: 700; display: block; }

/* موضوعات */
.topic-chip {
    display: inline-block;
    background: #1e2a3a;
    border: 1px solid #2d4a6e;
    border-radius: 20px;
    padding: 0.25rem 0.75rem;
    margin: 0.2rem;
    font-size: 0.82rem;
    color: #90cdf4;
    cursor: pointer;
}

.chat-container {
    height: 500px;
    overflow-y: auto;
    padding: 1rem;
    background: #0a0f1a;
    border-radius: 14px;
    border: 1px solid #1e2a3a;
    margin-bottom: 1rem;
}

.welcome-msg {
    text-align: center;
    color: #4a5568;
    padding: 3rem 1rem;
    font-size: 1.1rem;
}

div[data-testid="stVerticalBlock"] { gap: 0rem; }
</style>
""", unsafe_allow_html=True)

# ── تحميل النموذج ────────────────────────────────────────────────
_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

@st.cache_resource(show_spinner="⟳ تحميل NSM Chat...")
def load_model():
    from nsm_chat import NSMChat, _KNOWLEDGE_BASE
    return NSMChat(), len(_KNOWLEDGE_BASE)

try:
    bot, topics_count = load_model()
    model_ok = True
except Exception as e:
    model_ok = False
    model_error = str(e)

# ── تهيئة الحالة ────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "msg_count" not in st.session_state:
    st.session_state.msg_count = 0

# ── الرأس ───────────────────────────────────────────────────────
st.markdown("""
<div class="nsm-header">
    <h1>🧠 NSM Chat</h1>
    <p>مساعد ذكي بلا قاعدة بيانات — الذكاء في الأوزان فقط</p>
    <p style="color:#4a9eff;font-size:0.82rem;">Neural Service Mesh · weights_784×784 · Self-Attention · Arabic NLP</p>
</div>
""", unsafe_allow_html=True)

# ── الشريط الجانبي ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 NSM Chat")
    st.markdown("---")

    # إحصائيات
    st.markdown("**📊 إحصائيات النظام**")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""<div class="sidebar-stat"><span>{topics_count if model_ok else '—'}</span>موضوع</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class="sidebar-stat"><span>{st.session_state.msg_count}</span>رسالة</div>""", unsafe_allow_html=True)

    st.markdown(f"""<div class="sidebar-stat"><span>784×784</span>حجم مصفوفة الأوزان</div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="sidebar-stat"><span>4 Heads</span>Self-Attention</div>""", unsafe_allow_html=True)

    st.markdown("---")

    # أسئلة سريعة
    st.markdown("**⚡ أسئلة سريعة**")
    quick = [
        "ما هي أركان الإسلام؟",
        "ما هو الذكاء الاصطناعي؟",
        "ما هي سورة الفاتحة؟",
        "ما هي اللغة العربية؟",
        "ما هو الـ Transformer؟",
        "ما هي سورة الكهف؟",
        "من هو النبي محمد ﷺ؟",
        "ما هي الشبكة العصبية؟",
        "ما هو علم النحو؟",
        "ما هي الزكاة؟",
    ]
    for q in quick:
        if st.button(q, key=f"q_{q}", use_container_width=True):
            st.session_state._quick_q = q

    st.markdown("---")

    # مسح المحادثة
    if st.button("🗑 مسح المحادثة", use_container_width=True):
        st.session_state.messages = []
        st.session_state.msg_count = 0
        if model_ok:
            bot.clear_history()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#4a5568;font-size:0.78rem;">
        NSM · Neural Service Mesh<br>
        numpy only · no database<br>
        © علي أحمد
    </div>
    """, unsafe_allow_html=True)

# ── المحادثة ─────────────────────────────────────────────────────
if not model_ok:
    st.error(f"❌ خطأ في تحميل النموذج: {model_error}")
    st.stop()

# عرض الرسائل
chat_html = '<div class="chat-container" id="chat-box">'

if not st.session_state.messages:
    chat_html += """
    <div class="welcome-msg">
        🧠<br><br>
        مرحباً! أنا NSM مساعدك الذكي<br>
        <span style="font-size:0.9rem;color:#2d4a6e;">اكتب سؤالك أو اختر من الأسئلة السريعة</span>
    </div>
    """
else:
    for role, text in st.session_state.messages:
        if role == "user":
            chat_html += f"""
            <div class="user-bubble">
                <div class="bubble">{text}</div>
            </div>"""
        else:
            chat_html += f"""
            <div class="nsm-bubble">
                <div class="avatar">🧠</div>
                <div class="bubble">{text}</div>
            </div>"""

chat_html += '</div>'
st.markdown(chat_html, unsafe_allow_html=True)

# ── صندوق الإدخال ───────────────────────────────────────────────
col_input, col_btn = st.columns([5, 1])
with col_input:
    user_input = st.text_input(
        label="",
        placeholder="اكتب سؤالك هنا... (عربي أو إنجليزي)",
        key="user_input",
        label_visibility="collapsed",
    )
with col_btn:
    send = st.button("إرسال ➤", use_container_width=True)

# ── معالجة الإدخال ──────────────────────────────────────────────
def process_input(text: str):
    if not text.strip():
        return
    text = text.strip()
    response = bot.chat(text)
    st.session_state.messages.append(("user", text))
    st.session_state.messages.append(("nsm", response))
    st.session_state.msg_count += 1
    st.rerun()

# إرسال بالزر
if send and user_input:
    process_input(user_input)

# إرسال بالسؤال السريع
if hasattr(st.session_state, "_quick_q"):
    q = st.session_state._quick_q
    del st.session_state._quick_q
    process_input(q)

# اضغط Enter
if user_input and user_input != st.session_state.get("_last_input", ""):
    st.session_state["_last_input"] = user_input
