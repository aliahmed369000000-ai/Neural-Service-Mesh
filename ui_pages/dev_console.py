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
        _fails = st.session_state.get("_dev_console_fails", 0)
        _lock_until = st.session_state.get("_dev_console_lock_until", 0)
        _now = time.time()
        if _lock_until and _now < _lock_until:
            st.error(f"⏳ محاولات كثيرة فاشلة — حاول بعد {int(_lock_until - _now)} ثانية.")
            return
        entered = st.text_input("مفتاح المالك", type="password", key="dev_console_key_input")
        if st.button("🔓 فتح لوحة المطوّر", key="dev_console_unlock"):
            if hmac.compare_digest(entered, _admin_key_env):
                st.session_state["_dev_console_unlocked"] = True
                st.session_state["_dev_console_fails"] = 0
                st.rerun()
            else:
                _fails += 1
                st.session_state["_dev_console_fails"] = _fails
                if _fails >= 5:
                    st.session_state["_dev_console_lock_until"] = _now + 30
                    st.session_state["_dev_console_fails"] = 0
                    st.error("❌ مفتاح غير صحيح. محاولات كثيرة — قُفلت المحاولة 30 ثانية.")
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
                _stdout = _redact_secrets(result.stdout)
                _stderr = _redact_secrets(result.stderr)
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(f"**رمز الخروج:** `{result.returncode}`")
                if _stdout:
                    st.markdown("**stdout:**")
                    st.code(_stdout[-5000:])
                    _copy_button(_stdout[-5000:], key="dev_console_stdout", label="📋 نسخ stdout")
                if _stderr:
                    st.markdown("**stderr:**")
                    st.code(_stderr[-5000:])
                    _copy_button(_stderr[-5000:], key="dev_console_stderr", label="📋 نسخ stderr")
                if not _stdout and not _stderr:
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

    # ── تقرير تدقيق المشروع (Phase6Validator) ────────────────────────────
    st.markdown("---")
    st.markdown("#### 🔍 تقرير تدقيق المشروع")
    st.caption(
        "يفحص كل ملفات المستودع فعلياً: أيها مستورد فعلاً من نقاط الدخول "
        "(streamlit_app.py/app_core.py وغيرها) وأيها كود ميت غير مربوط، مع "
        "نسبة تغطية مراحل 1-6 ودرجة جاهزية Phase 7."
    )
    if st.button("📊 شغّل التقرير", key="dc_validator_run"):
        try:
            from ai.validator import Phase6Validator
            with st.spinner("⟳ يفحص كل ملفات المستودع..."):
                import io as _io, contextlib as _cl
                _buf = _io.StringIO()
                with _cl.redirect_stdout(_buf):
                    _report = Phase6Validator(mesh=None, project_root=str(BASE)).generate()
            st.session_state["_dc_validator_report"] = _report
            st.session_state["_dc_validator_text"] = _buf.getvalue()
        except Exception as _val_err:
            st.error(f"❌ تعذّر تشغيل المدقّق: {_val_err}")

    _rep = st.session_state.get("_dc_validator_report")
    if _rep:
        _cv1, _cv2, _cv3, _cv4 = st.columns(4)
        with _cv1:
            metric_card(_rep["files"]["total_py_files"], "ملف بايثون")
        with _cv2:
            metric_card(f"{_rep['dead_code']['dead_pct']}%", "كود ميت")
        with _cv3:
            metric_card(f"{_rep['phase_coverage']['overall_coverage_pct']}%", "تغطية المراحل 1-6")
        with _cv4:
            metric_card(f"{_rep['phase7_readiness']['score']}/100", "جاهزية Phase 7")

        st.info(_rep["phase7_readiness"]["verdict"])

        if _rep["dead_code"]["dead_files"]:
            with st.expander(f"📄 الملفات غير المربوطة ({_rep['dead_code']['dead_count']})"):
                for _layer, _files in sorted(_rep["dead_code"]["dead_by_layer"].items()):
                    st.markdown(f"**{_layer}**")
                    for _df in _files:
                        st.markdown(f"- `{_df}`")

        with st.expander("📋 التقرير الكامل (نص)"):
            st.code(st.session_state.get("_dc_validator_text", ""), language=None)

    # ── سجل محادثات تبويب 💬 المحادثة (ai/chat_history_store.py) ─────────
    st.markdown("---")
    st.markdown("#### 📜 سجل المحادثات المحفوظة")
    st.caption(
        "كل جلسات تبويب 💬 المحادثة المحفوظة بشكل دائم في "
        "memory/chat_history.db (يبقى طالما لم يُعَد نشر التطبيق من الصفر)."
    )
    try:
        from ai.chat_history_store import (
            list_sessions, get_session_messages, get_storage_stats,
            delete_sessions_older_than,
        )

        # ── إحصائيات التخزين (قرص Streamlit Cloud محدود) ──────────────
        _stats = get_storage_stats()
        _sc1, _sc2, _sc3 = st.columns(3)
        with _sc1:
            metric_card(_stats["total_sessions"], "إجمالي الجلسات")
        with _sc2:
            metric_card(_stats["total_messages"], "إجمالي الرسائل")
        with _sc3:
            metric_card(f"{_stats['db_size_bytes'] / 1024:.1f} KB", "حجم قاعدة البيانات")

        # ── تنظيف الجلسات القديمة (عملية تدميرية — تتطلب تأكيداً صريحاً) ──
        with st.expander("🧹 تنظيف الجلسات القديمة"):
            _dc_cleanup_days = st.number_input(
                "احذف الجلسات الأقدم من (يوم)", min_value=1, max_value=3650,
                value=30, step=1, key="dc_chat_cleanup_days",
            )
            _dc_confirm = st.text_input(
                f"اكتب `تأكيد` هنا لحذف كل الجلسات الأقدم من {int(_dc_cleanup_days)} يوماً "
                "(لا يمكن التراجع)",
                key="dc_chat_cleanup_confirm",
            )
            if st.button("🗑️ احذف نهائياً", key="dc_chat_cleanup_run", type="primary"):
                if _dc_confirm.strip() != "تأكيد":
                    st.warning("اكتب كلمة `تأكيد` بالضبط في الحقل أعلاه قبل الحذف.")
                else:
                    _deleted = delete_sessions_older_than(int(_dc_cleanup_days))
                    st.success(f"✅ تم حذف {_deleted} رسالة.")
                    st.session_state.pop("_dc_chat_sessions", None)
                    st.toast(f"🗑️ حُذفت {_deleted} رسالة", icon="🗑️")
                    st.rerun()

        _dc_sessions_limit = st.slider(
            "عدد الجلسات المعروضة", 5, 100, 20, 5, key="dc_chat_sessions_limit"
        )
        if st.button("🔄 تحديث القائمة", key="dc_chat_sessions_refresh"):
            st.session_state["_dc_chat_sessions"] = list_sessions(limit=_dc_sessions_limit)
        if "_dc_chat_sessions" not in st.session_state:
            st.session_state["_dc_chat_sessions"] = list_sessions(limit=_dc_sessions_limit)

        _sessions = st.session_state["_dc_chat_sessions"]
        if not _sessions:
            st.info("لا توجد أي محادثات محفوظة بعد.")
        else:
            metric_card(len(_sessions), "جلسة محفوظة")
            for _sess in _sessions:
                _label = (
                    f"🕒 {_sess['started_at'][:16].replace('T', ' ')} → "
                    f"{_sess['last_at'][:16].replace('T', ' ')} "
                    f"— {_sess['message_count']} رسالة — `{_sess['session_id'][:8]}…`"
                )
                with st.expander(_label):
                    _msgs = get_session_messages(_sess["session_id"])
                    if not _msgs:
                        st.caption("لا رسائل.")
                        continue
                    _first = _msgs[0]
                    st.markdown(
                        f"**أول من ردّ في هذه المحادثة:** "
                        f"{'👤 المستخدم' if _first['role'] == 'user' else '🤖 NSM'} "
                        f"— *{_first['created_at'][:19].replace('T', ' ')}*"
                    )
                    for _m in _msgs:
                        _who = "👤 المستخدم" if _m["role"] == "user" else "🤖 NSM"
                        _badge = f" ({_m['source_badge']})" if _m.get("source_badge") else ""
                        st.markdown(f"**{_who}{_badge}:** {_m['content']}")
    except Exception as _hist_err:
        st.error(f"❌ تعذّر تحميل سجل المحادثات: {_hist_err}")
