"""
ui_pages/dev_console.py
تم تفكيكه تلقائياً من streamlit_app.py الأصلي (تقسيم الكود لتحسين القابلية للصيانة والأداء).
"""
from __future__ import annotations

from app_core import *  # noqa: F401,F403 — إعادة تصدير كل الاستيرادات والدوال المساعدة المشتركة



# ══════════════════════════════════════════════════════════════════════════
# تبويب 🖥️ لوحة المطوّر — تنفيذ أوامر Bash/Python (محمي بمفتاح المالك)
# ══════════════════════════════════════════════════════════════════════════
def render_dev_console():
    st.markdown('<div class="section-header">🖥️ لوحة المطوّر</div>', unsafe_allow_html=True)
    st.warning(
        "⚠️ هذه الأداة تنفّذ أوامر حقيقية على الخادم. محمية بمفتاح المالك "
        "(`NSM_ADMIN_KEY`) — لا تشاركها مع أحد."
    )

    _admin_key_env = os.environ.get("NSM_ADMIN_KEY", "")
    if not _admin_key_env:
        st.error("❌ لم يتم ضبط NSM_ADMIN_KEY في Secrets — هذه الميزة معطّلة حتى يُضاف المفتاح.")
        return

    if not st.session_state.get("_dev_console_unlocked", False):
        entered = st.text_input("مفتاح المالك", type="password", key="dev_console_key_input")
        if st.button("🔓 فتح لوحة المطوّر", key="dev_console_unlock"):
            if hmac.compare_digest(entered, _admin_key_env):
                st.session_state["_dev_console_unlocked"] = True
                st.rerun()
            else:
                st.error("❌ مفتاح غير صحيح.")
        return

    col_lock, _ = st.columns([1, 4])
    with col_lock:
        if st.button("🔒 قفل", key="dev_console_lock"):
            st.session_state["_dev_console_unlocked"] = False
            st.rerun()

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### تنفيذ أمر")
    cmd_kind = st.radio("النوع", ["Bash", "Python"], horizontal=True, key="dev_console_kind")
    cmd_text = st.text_area("الأمر", height=120, key="dev_console_cmd",
                             placeholder="مثال: ls -la" if cmd_kind == "Bash" else "print(1 + 1)")
    cmd_timeout = st.slider("مهلة التنفيذ (ثوانٍ)", 5, 60, 20, 5, key="dev_console_timeout")
    run_clicked = st.button("▶️ نفّذ", key="dev_console_run", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if run_clicked:
        if not cmd_text.strip():
            st.warning("أدخل أمراً أولاً.")
        else:
            import subprocess as _sp
            _dc_ph = st.empty()
            with _dc_ph.container():
                _skeleton(lines=4)
            try:
                if cmd_kind == "Bash":
                    result = _sp.run(
                        cmd_text, shell=True, capture_output=True, text=True, timeout=cmd_timeout,
                    )
                else:
                    result = _sp.run(
                        ["python3", "-c", cmd_text], capture_output=True, text=True, timeout=cmd_timeout,
                    )
                _dc_ph.empty()
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**رمز الخروج:** `{result.returncode}`")
                if result.stdout:
                    st.markdown("**stdout:**")
                    st.code(result.stdout[-5000:])
                    _copy_button(result.stdout[-5000:], key="dev_console_stdout", label="📋 نسخ stdout")
                if result.stderr:
                    st.markdown("**stderr:**")
                    st.code(result.stderr[-5000:])
                    _copy_button(result.stderr[-5000:], key="dev_console_stderr", label="📋 نسخ stderr")
                if not result.stdout and not result.stderr:
                    st.caption("لا يوجد ناتج.")
                st.markdown("</div>", unsafe_allow_html=True)
                if result.returncode == 0:
                    st.toast("✅ تم تنفيذ الأمر بنجاح", icon="✅")
                else:
                    st.toast(f"⚠️ انتهى الأمر برمز خروج {result.returncode}", icon="⚠️")
            except _sp.TimeoutExpired:
                _dc_ph.empty()
                st.error(f"⏱️ انتهت المهلة ({cmd_timeout}s) قبل اكتمال التنفيذ.")
                st.toast("⏱️ انتهت مهلة التنفيذ", icon="⏱️")
            except Exception as _exec_err:
                _dc_ph.empty()
                st.error(f"❌ خطأ أثناء التنفيذ: {_exec_err}")
                st.toast("❌ فشل تنفيذ الأمر", icon="❌")
