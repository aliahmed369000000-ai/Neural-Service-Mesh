"""
NSM Terminal UI — تبويب طرفية احترافي للوكلاء والمالك
=====================================================
يبني على ai/nsm_terminal.py
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
.nsm-term-wrap {
  direction: ltr;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  background: linear-gradient(180deg, #0b1220 0%, #0a0f18 100%);
  border: 1px solid #1e293b;
  border-radius: 14px;
  padding: 0;
  overflow: hidden;
  box-shadow: 0 12px 40px rgba(0,0,0,.35);
}
.nsm-term-titlebar {
  display: flex; align-items: center; gap: 8px;
  padding: 10px 14px;
  background: #111827;
  border-bottom: 1px solid #1f2937;
}
.nsm-term-dot { width: 12px; height: 12px; border-radius: 50%; display: inline-block; }
.nsm-term-dot.r { background: #ff5f56; }
.nsm-term-dot.y { background: #ffbd2e; }
.nsm-term-dot.g { background: #27c93f; }
.nsm-term-title {
  color: #94a3b8; font-size: 0.85rem; margin-left: 8px;
}
.nsm-term-body {
  padding: 14px 16px 18px;
  min-height: 320px;
  max-height: 520px;
  overflow-y: auto;
  color: #e2e8f0;
  font-size: 0.86rem;
  line-height: 1.45;
  white-space: pre-wrap;
  word-break: break-word;
}
.nsm-term-prompt { color: #34d399; }
.nsm-term-cmd { color: #f8fafc; }
.nsm-term-meta { color: #64748b; font-size: 0.75rem; }
.nsm-term-out { color: #cbd5e1; }
.nsm-term-err { color: #fca5a5; }
.nsm-term-ok { color: #6ee7b7; }
.nsm-term-fail { color: #f87171; }
</style>
"""


def _admin_ok() -> bool:
    return bool(st.session_state.get("_dev_console_unlocked", False))


def _ensure_session():
    from ai.nsm_terminal import get_terminal
    term = get_terminal()
    if "nsm_term_session_id" not in st.session_state:
        sess = term.create_session(mode="admin" if _admin_ok() else "safe")
        st.session_state.nsm_term_session_id = sess.id
    return term


def _render_history_html(history: list) -> str:
    parts = []
    for h in history[-40:]:
        cmd = h.get("cmd") or ""
        code = h.get("exit_code", 0)
        badge = "nsm-term-ok" if code == 0 else "nsm-term-fail"
        parts.append(
            f'<div><span class="nsm-term-prompt">nsm@mesh</span>:<span class="nsm-term-meta">{h.get("cwd","")}</span>$ '
            f'<span class="nsm-term-cmd">{_esc(cmd)}</span></div>'
        )
        parts.append(
            f'<div class="nsm-term-meta">exit {code} · {h.get("duration_ms", 0)}ms · {h.get("mode")}</div>'
        )
        if h.get("stdout"):
            parts.append(f'<div class="nsm-term-out">{_esc(h["stdout"])}</div>')
        if h.get("stderr"):
            parts.append(f'<div class="nsm-term-err">{_esc(h["stderr"])}</div>')
        if h.get("error"):
            parts.append(f'<div class="nsm-term-err">[error] {_esc(h["error"])}</div>')
        parts.append(f'<div class="{badge}">{"✔" if code == 0 else "✘"}</div><br/>')
    if not parts:
        parts.append('<div class="nsm-term-meta">NSM Terminal ready. Type a command below.</div>')
    return "\n".join(parts)


def _esc(s: str) -> str:
    return (
        (s or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def render_nsm_terminal():
    """واجهة الطرفيه — مصممة للاستخدام البشري + الوكلاء."""
    st.markdown('<div class="section-header">💻 NSM Terminal</div>', unsafe_allow_html=True)
    st.caption(
        "طرفية حقيقية مربوطة بالوكلاء · جلسات · سجل · أوضاع safe/admin · "
        "الوكلاء يستدعونها عبر action `terminal` أو أمر المحادثة `طرفية <أمر>`"
    )

    if not _admin_ok():
        st.warning("🔒 وضع المالك مطلوب لفتح الطرفيه الكاملة. يمكنك استخدام الأوامر الآمنة من المحادثة.")
        # still allow view of safe status
        try:
            from ai.nsm_terminal import get_terminal
            st.json(get_terminal().get_session().to_dict())
        except Exception as e:
            st.error(str(e))
        return

    st.markdown(_CSS, unsafe_allow_html=True)
    term = _ensure_session()
    sid = st.session_state.nsm_term_session_id
    sess = term.get_session(sid)
    sess.mode = "admin"  # owner tab

    # 🆕 مبدّل الجلسات — يسمح بالتنقل بين كل الجلسات المفتوحة (وليس فقط إنشاء جلسة جديدة)
    all_sessions = term.list_sessions()
    if len(all_sessions) > 1:
        sid_options = [s["id"] for s in all_sessions]
        _labels = {
            s["id"]: f'{s["id"]} · {(s["cwd"].rstrip("/").split("/")[-1] or "/")} · {s["history_len"]} أوامر'
            for s in all_sessions
        }
        chosen = st.selectbox(
            "🗂️ الجلسة النشطة",
            options=sid_options,
            index=sid_options.index(sid) if sid in sid_options else 0,
            format_func=lambda x: _labels.get(x, x),
            key="nsm_term_session_picker",
        )
        if chosen != sid:
            st.session_state.nsm_term_session_id = chosen
            st.rerun()
        sid = chosen
        sess = term.get_session(sid)
        sess.mode = "admin"
        # 🆕 إغلاق الجلسة الحالية (تنظيف يدوي، لا يمكن إغلاق الجلسة الافتراضية)
        if st.button("🗑️ إغلاق هذه الجلسة", key="nsm_term_close"):
            if term.close_session(sid):
                st.session_state.nsm_term_session_id = term._default_id
                st.rerun()
            else:
                st.warning("لا يمكن إغلاق الجلسة الافتراضية.")

    # controls
    c1, c2, c3, c4, c5 = st.columns([2, 1, 1, 1, 1])
    with c1:
        st.markdown(f"**session** `{sid}` · **cwd** `{sess.cwd}` · **mode** `{sess.mode}`")
    with c2:
        if st.button("🧹 Clear", key="nsm_term_clear", use_container_width=True):
            term.run("clear", session_id=sid, mode="admin")
            st.rerun()
    with c3:
        if st.button("📁 New session", key="nsm_term_new", use_container_width=True):
            news = term.create_session(mode="admin")
            st.session_state.nsm_term_session_id = news.id
            st.rerun()
    with c4:
        # 🆕 تنزيل سجل الجلسة الحالية كـ JSON (سجل تدقيق / مشاركة)
        import json as _json_dl
        st.download_button(
            "⬇️ السجل",
            data=_json_dl.dumps(sess.history, ensure_ascii=False, indent=2),
            file_name=f"nsm_terminal_{sid}.json",
            mime="application/json",
            use_container_width=True,
            disabled=not sess.history,
            key="nsm_term_download",
        )
    with c5:
        timeout = st.number_input("timeout s", min_value=5, max_value=300, value=45, key="nsm_term_timeout")

    # quick actions — 🆕 عرض كل الـ presets العشرة المتوفرة فعلياً في المحرك (كانت 6 من 10 فقط)
    st.markdown("**Quick**")
    presets = [
        ("git status", "status"),
        ("git log", "log"),
        ("diff", "diff"),
        ("pytest", "pytest"),
        ("compile ai", "compile_ai"),
        ("ls", "tree"),
        ("disk", "disk"),
        ("python -V", "python"),
        ("branch -vv", "branch"),
        ("lfs files", "lfs"),
    ]
    qcols = st.columns(5)
    for i, (label, key) in enumerate(presets):
        with qcols[i % 5]:
            if st.button(label, key=f"nsm_term_q_{key}", use_container_width=True):
                term.quick(key, session_id=sid, mode="admin")
                st.rerun()

    # 🆕 إعادة تشغيل أمر سابق من السجل بضغطة واحدة (بدل إعادة كتابته يدوياً)
    recent_cmds: list = []
    for h in reversed(sess.history):
        c = (h.get("cmd") or "").strip()
        if c and c not in recent_cmds:
            recent_cmds.append(c)
        if len(recent_cmds) >= 15:
            break
    if recent_cmds:
        rc1, rc2 = st.columns([4, 1])
        with rc1:
            picked = st.selectbox(
                "↺ إعادة تشغيل أمر سابق",
                options=["—"] + recent_cmds,
                key="nsm_term_recall",
                label_visibility="collapsed",
            )
        with rc2:
            if st.button("▶️ تشغيل", key="nsm_term_recall_run", use_container_width=True, disabled=(picked == "—")):
                with st.spinner("running…"):
                    term.run(picked, session_id=sid, mode="admin", timeout=int(timeout))
                st.rerun()

    # 🆕 بحث/تصفية داخل سجل الطرفية (بالأمر أو المخرجات)
    search_q = st.text_input(
        "🔎 بحث في السجل", key="nsm_term_search", placeholder="فلترة حسب الأمر أو المخرجات…"
    )
    display_history = sess.history
    if search_q.strip():
        q = search_q.strip().lower()
        display_history = [
            h for h in sess.history
            if q in (h.get("cmd") or "").lower()
            or q in (h.get("stdout") or "").lower()
            or q in (h.get("stderr") or "").lower()
        ]

    # terminal screen
    html_body = _render_history_html(display_history)
    st.markdown(
        f"""
        <div class="nsm-term-wrap">
          <div class="nsm-term-titlebar">
            <span class="nsm-term-dot r"></span>
            <span class="nsm-term-dot y"></span>
            <span class="nsm-term-dot g"></span>
            <span class="nsm-term-title">nsm — { _esc(sess.cwd) }</span>
          </div>
          <div class="nsm-term-body">{html_body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # input
    with st.form("nsm_term_form", clear_on_submit=True):
        cmd = st.text_input(
            "command",
            placeholder="git status · python -m py_compile ai/nsm_terminal.py · cd ai && ls",
            label_visibility="collapsed",
            key="nsm_term_input",
        )
        fc1, fc2 = st.columns([1, 3])
        with fc1:
            bg = st.checkbox("⏱️ خلفية", key="nsm_term_bg", help="تشغيل بدون حجب — لأوامر التدريب/البناء الطويلة")
        with fc2:
            submitted = st.form_submit_button("Run ↵", use_container_width=True)
        if submitted and cmd.strip():
            if bg:
                job = term.start_background(cmd.strip(), session_id=sid, mode="admin")
                if job.status == "error":
                    st.toast(f"رُفضت: {job.error}", icon="⚠️")
                else:
                    st.toast(f"بدأت مهمة خلفية {job.id}", icon="🚀")
            else:
                with st.spinner("running…"):
                    r = term.run(cmd.strip(), session_id=sid, mode="admin", timeout=int(timeout))
                if r.ok:
                    st.toast(f"exit {r.exit_code} · {r.duration_ms}ms", icon="✅")
                else:
                    st.toast(f"exit {r.exit_code}", icon="⚠️")
            st.rerun()

    # 🆕 لوحة المهام الخلفية — عرض حي + قتل فعلي لأي مهمة قيد التشغيل
    jobs = term.list_jobs(session_id=sid)
    if jobs:
        with st.expander(f"🧵 مهام خلفية ({sum(1 for j in jobs if j['status']=='running')} تعمل الآن)", expanded=any(j["status"] == "running" for j in jobs)):
            for j in jobs:
                jc1, jc2 = st.columns([5, 1])
                with jc1:
                    icon = {"running": "🟡", "done": "🟢", "killed": "🔴", "error": "⚫"}.get(j["status"], "⚪")
                    st.markdown(f"{icon} `{j['id']}` **{j['status']}** — `{_esc(j['cmd'])}`")
                    if j["status"] in ("done", "error") and (j.get("stdout") or j.get("stderr") or j.get("error")):
                        # 🐛 إصلاح خطأ حقيقي: كان هنا st.expander("مخرجات") متداخل
                        # داخل st.expander("🧵 مهام خلفية") الخارجي — وStreamlit
                        # يرفض تعشيش expander داخل expander إطلاقاً (يرمي
                        # StreamlitAPIException وقت التشغيل فعلياً بمجرد وجود
                        # أي مهمة منتهية/فاشلة لها مخرجات، وهو سيناريو شائع لا
                        # حافة نادرة). استُبدل بمفتاح إظهار/إخفاء (checkbox) بلا
                        # تعشيش — نفس الإتاحة الاختيارية لعرض المخرجات بلا خطأ.
                        if st.checkbox("📄 عرض المخرجات", key=f"nsm_term_out_{j['id']}"):
                            if j.get("stdout"):
                                st.code(j["stdout"], language="text")
                            if j.get("stderr"):
                                st.code(j["stderr"], language="text")
                            if j.get("error"):
                                st.caption(f"خطأ: {j['error']}")
                with jc2:
                    if j["status"] == "running":
                        if st.button("⛔ قتل", key=f"nsm_term_kill_{j['id']}", use_container_width=True):
                            term.kill_job(j["id"])
                            st.rerun()

    with st.expander("Agent API / أمثلة ربط الوكلاء"):
        st.code(
            'Action JSON:\n'
            '{"action":"terminal","cmd":"git status --short","mode":"safe"}\n\n'
            "Chat:\n"
            "طرفية git status\n"
            "طرفية !status\n"
            "terminal python3 --version\n\n"
            "export/unset (مستمر داخل الجلسة):\n"
            "export MY_VAR=value\n"
            "unset MY_VAR",
            language="text",
        )
        st.caption("الوضع safe للوكلاء يقيّد الأوامر الخطرة تلقائياً. وضع المالك هنا = admin.")
