"""
NSM Terminal Live — تيرمنال حي بنمط Tmux داخل المتصفح
====================================================
ثلاث لوحات:
1. 🖥️ Live  — إدخال مباشر (بدون form) + بث حي للمخرجات لحظة بلحظة
2. 🚀 Kaggle — حالة وlogs kernels مع أوامر kg* مدمجة
3. 🧠 Smart — اقتراحات LLM + aliases قابلة للتخصيص

يعتمد على:
- ai/nsm_terminal.py (المحرك: جلسات، صلاحيات، kg shortcuts)
- ai/terminal_roles.py (الأدوار والتدقيق)
- ai/terminal_smart.py (Kaggle CLI + اقتراحات LLM)

ملاحظة أداء: Streamlit لا يدعم websockets حيًا خارج نطاقه، لكن التبث الحي
يتحقق عبر st.empty + polling سريع (أخف من form التقليدي) — تجربة قريبة من
Tmux بدون بنية server منفصلة.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Any, Dict, List

import streamlit as st

_CSS_LIVE = """
<style>
.nsm-live-wrap {
  direction: ltr;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
  background: linear-gradient(180deg, #070d17 0%, #050911 100%);
  border: 1px solid #1e293b;
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 10px 36px rgba(0,0,0,.4);
}
.nsm-live-body {
  padding: 12px 14px;
  min-height: 260px;
  max-height: 480px;
  overflow-y: auto;
  color: #e2e8f0;
  font-size: 0.82rem;
  line-height: 1.4;
  white-space: pre-wrap;
  word-break: break-word;
}
.nsm-live-prompt { color: #22d3ee; font-weight: 700; }
.nsm-live-meta { color: #64748b; font-size: 0.72rem; }
.nsm-live-out { color: #cbd5e1; }
.nsm-live-err { color: #fca5a5; }
.nsm-live-ok { color: #6ee7b7; }
.nsm-live-running { color: #fbbf24; animation: nsm-blink 1s infinite; }
@keyframes nsm-blink { 50% { opacity: .35; } }
.nsm-live-input {
  width: 100%;
  background: #0c1320;
  border: 1px solid #334155;
  border-top: 1px dashed #1e293b;
  color: #f8fafc;
  font-family: ui-monospace, monospace;
  font-size: 0.85rem;
  padding: 10px 12px;
  outline: none;
  resize: none;
}
.nsm-live-input:focus { border-color: #22d3ee; }
</style>
"""


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get_terminal():
    from ai.nsm_terminal import get_terminal
    return get_terminal()


def _admin_ok() -> bool:
    return bool(st.session_state.get("_dev_console_unlocked", False))


def render_nsm_terminal_live():
    """لوحة التيرمنال الحي — تبويبات: Live / Kaggle / Smart."""
    if not _admin_ok():
        st.warning("🔒 وضع المالك مطلوب لفتح التيرمنال الحي.")
        return

    st.markdown(_CSS_LIVE, unsafe_allow_html=True)
    st.markdown('<div class="section-header">⚡ NSM Terminal Live</div>', unsafe_allow_html=True)
    st.caption(
        "تيرمنال حي بنمط Tmux — إدخال مباشر + بث مخرجات لحظة بلحظة · "
        "كامل صلاحيات ai/nsm_terminal.py (أدوار + تدقيق + kg shortcuts)"
    )

    term = _get_terminal()
    if "nsm_live_session" not in st.session_state:
        sess = term.create_session(mode="admin")
        st.session_state.nsm_live_session = sess.id
    sid = st.session_state.nsm_live_session
    sess = term.get_session(sid)

    live_tab, kaggle_tab, smart_tab = st.tabs(["🖥️ Live", "🚀 Kaggle", "🧠 Smart"])
    with live_tab:
        _render_live(term, sess, sid)
    with kaggle_tab:
        _render_kaggle(term, sess, sid)
    with smart_tab:
        _render_smart(term, sess, sid)


# ══════════════════════ 1. Live ══════════════════════

def _render_live(term, sess, sid):
    # سطر الحالة
    st1, st2, st3 = st.columns([3, 2, 2])
    with st1:
        st.markdown(f"**session** `{sid}` · **cwd** `{sess.cwd}` · **mode** `{sess.mode}`")
    with st2:
        mode = st.selectbox("الوضع", ["admin", "safe"], key="nsm_live_mode",
                            index=0 if sess.mode == "admin" else 1)
    with st3:
        to = st.number_input("timeout s", 5, 300, 60, key="nsm_live_to")

    # الشاشة: آخر 30 حدثًا + مؤشر تشغيل حي
    n_hist = st.number_input("عمق السجل المعروض", 10, 80, 30, key="nsm_live_depth",
                             help="عدد الأحداث المعروضة (ليس حد السجل الكامل)")
    _render_screen(sess, int(n_hist))

    # إدخال مباشر — بدون formSubmitButton: Enter داخل textarea يرسل عبر form واحد سريع
    st.markdown('<div style="margin-top:10px"></div>', unsafe_allow_html=True)
    cmd_input = st.text_area(
        "أدخل أمرًا واضغط Enter للإرسال",
        key="nsm_live_input",
        placeholder="git status  ↵\npython -m py_compile ai/terminal_live.py  ↵\nkg status username/kernel  ↵",
        label_visibility="collapsed",
        height=52,
    )
    if st.session_state.get("_nsm_live_running"):
        st.markdown('<div class="nsm-live-running">⏳ أمر قيد التنفيذ…</div>',
                    unsafe_allow_html=True)

    if st.button("Send ↵", key="nsm_live_send", use_container_width=True):
        _send_cmd(term, sess, sid, cmd_input, int(to), mode)
        return

    # حفظ السجل كـ JSON
    if sess.history:
        if st.download_button(
            "⬇️ تنزيل السجل",
            data=__import__("json").dumps(sess.history, ensure_ascii=False, indent=2),
            file_name=f"nsm_live_{sid}.json",
            mime="application/json",
            use_container_width=True,
            key="nsm_live_dl",
        ):
            pass


def _send_cmd(term, sess, sid, cmd: str, timeout: int, mode: str):
    cmd = (cmd or "").strip()
    if not cmd:
        return
    if st.session_state.get("_nsm_live_running"):
        st.toast("⏳ أمر سابق قيد التنفيذ — انتظر حتى ينتهي", icon="⚠️")
        return
    st.session_state["_nsm_live_running"] = True

    def _worker():
        try:
            term.run(cmd, session_id=sid, mode=mode, timeout=timeout)
        finally:
            st.session_state["_nsm_live_running"] = False

    threading.Thread(target=_worker, daemon=True).start()
    st.rerun()


def _render_screen(sess, depth: int):
    history = sess.history[-depth:] if sess.history else []
    parts = []
    for h in history:
        c = h.get("cmd") or ""
        code = h.get("exit_code", 0)
        parts.append(
            f'<div><span class="nsm-live-prompt">nsm@live</span>:<span class="nsm-live-meta">{_esc(h.get("cwd",""))}</span>$ '
            f'<span style="color:#f8fafc">{_esc(c)}</span></div>'
        )
        if h.get("stdout"):
            parts.append(f'<div class="nsm-live-out">{_esc(h["stdout"])}</div>')
        if h.get("stderr"):
            parts.append(f'<div class="nsm-live-err">{_esc(h["stderr"])}</div>')
        parts.append(f'<div class="nsm-live-meta">exit {code} · {h.get("duration_ms", 0)}ms · {h.get("mode")}</div>')
    if st.session_state.get("_nsm_live_running"):
        parts.append('<div class="nsm-live-running">█</div>')
    if not parts:
        parts.append('<div class="nsm-live-meta">Terminal ready — اكتب أمرًا أعلاه واضغط Send ↵</div>')

    st.markdown(
        f"""
        <div class="nsm-live-wrap">
          <div class="nsm-live-body">{''.join(parts)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════ 2. Kaggle ══════════════════════

_KG_SUGGESTIONS = {
    "kg status": ("حالة kernel محدد", "aliahmedmo/nsm-surahchain-scn-82ec17428d"),
    "kg logs": ("logs kernel (آخر 80 سطرًا)", "aliahmedmo/nsm-corpus-arabic-20260815-b2"),
    "kg list": ("قائمة kernels المستخدم", "aliahmedmo"),
}


def _render_kaggle(term, sess, sid):
    from ai.terminal_smart import kaggle_binary_available
    st.markdown("**أوامر Kaggle المدمجة** — تعمل مباشرة من التيرمنال (بدون shell):")
    st.caption("`kg status user/kernel` · `kg logs user/kernel` · `kg list user` · `kg output user/kernel [dest]`")

    avail = kaggle_binary_available()
    st3, st4 = st.columns([4, 2])
    with st3:
        st.markdown(f"أمر kaggle CLI: {'✅ متوفر' if avail else '⚠️ غير مثبت في هذه البيئة'}")
    with st4:
        if st.button("🔄 فحص حالة kaggle CLI", use_container_width=True):
            from ai.terminal_smart import kaggle_binary_available
            st.toast("✅ متوفر" if kaggle_binary_available() else "⚠️ غير مثبت",
                     icon="✅" if kaggle_binary_available() else "⚠️")

    st.markdown("---")
    for sub, (label, default) in _KG_SUGGESTIONS.items():
        c1, c2 = st.columns([4, 1])
        with c1:
            val = st.text_input(f"`{sub}` — {label}", value=default,
                                key=f"nsm_live_kg_{sub.replace(' ','_')}",
                                label_visibility="collapsed")
        with c2:
            if st.button("▶️", key=f"nsm_live_kg_run_{sub.replace(' ','_')}",
                         use_container_width=True, disabled=not val.strip()):
                if val.strip():
                    with st.spinner("Kaggle API…"):
                        r = term.run(f"{sub} {val.strip()}", session_id=sid,
                                     mode=sess.mode, timeout=120)
                    st.session_state[f"nsm_live_kg_result_{sub.replace(' ','_')}"] = {
                        "ok": r.ok, "out": r.stdout, "err": r.stderr, "cmd": r.cmd,
                    }
                    st.rerun()

    # عرض نتيجة آخر أمر
    for sub in _KG_SUGGESTIONS:
        res = st.session_state.get(f"nsm_live_kg_result_{sub.replace(' ','_')}")
        if res:
            with st.expander(f"نتيجة `{res['cmd']}` — {'✅' if res['ok'] else '❌'}", expanded=True):
                if res.get("out"):
                    st.code(res["out"], language="text")
                if res.get("err"):
                    st.caption(f"خطأ: {res['err'][:500]}")


# ══════════════════════ 3. Smart ══════════════════════

def _render_smart(term, sess, sid):
    st.markdown("**🧠 اقتراح الأوامر** — النموذج يتعلم من آخر أوامرك")
    if not sess.history:
        st.caption("لا يوجد تاريخ أوامر بعد — نفّذ أوامر أولًا في تبويب Live")
        return

    hist_cmds = [(h.get("cmd") or "") for h in sess.history[-10:] if (h.get("cmd") or "").strip()]
    c1, c2 = st.columns([4, 1])
    with c1:
        st.caption("آخر أوامر: " + " · ".join(hist_cmds[-5:]))
    with c2:
        if st.button("🤖 اقترح أمرًا تاليًا", key="nsm_live_suggest", use_container_width=True):
            from ai.terminal_smart import suggest_command
            sug, from_llm = suggest_command(hist_cmds)
            st.session_state["_nsm_live_sug"] = {"text": sug, "llm": from_llm}
            st.rerun()

    sug = st.session_state.get("_nsm_live_sug")
    if sug:
        badge = "🤖 LLM" if sug["llm"] else "⚙️ محلي"
        st.markdown(f"**{badge}**: `{sug['text']}`")
        if st.button("▶️ تنفيذ الاقتراح", key="nsm_live_sug_run", use_container_width=True):
            with st.spinner("running…"):
                term.run(sug["text"], session_id=sid, mode=sess.mode)
            st.rerun()

    st.markdown("---")
    st.markdown("**🔧 Aliases مخصصة** — اختصارات دائمة عبر الجلسات")
    aliases = getattr(term, "aliases", {}) or {}
    al_cols = st.columns([2, 4, 1])
    with al_cols[0]:
        name = st.text_input("اسم الاختصار", key="nsm_live_alias_name", placeholder="gs")
    with al_cols[1]:
        body = st.text_input("الأمر الكامل", key="nsm_live_alias_body",
                             placeholder="git status --short")
    with al_cols[2]:
        if st.button("➕", key="nsm_live_alias_add", use_container_width=True,
                     disabled=not (name.strip() and body.strip())):
            ok, msg = term.set_alias(name.strip(), body.strip())
            st.toast("✅ أُضيف" if ok else f"❌ {msg}", icon="✅" if ok else "❌")
            st.rerun()

    if aliases:
        st.markdown("**الموجود:**")
        for k, v in aliases.items():
            a1, a2 = st.columns([3, 1])
            with a1:
                st.markdown(f"`{k}` → `{v}`")
            with a2:
                if st.button("🗑", key=f"nsm_live_alias_del_{k}", use_container_width=True):
                    ok, msg = term.del_alias(k)
                    st.toast("✅ حُذف" if ok else f"❌ {msg}", icon="✅" if ok else "❌")
                    st.rerun()

    st.markdown("---")
    st.markdown("**📸 حفظ/استعادة الجلسات**")
    sn1, sn2 = st.columns(2)
    with sn1:
        if st.button("💾 احفظ snapshot", key="nsm_live_snap_save", use_container_width=True):
            ok = term.save_sessions_snapshot()
            st.toast("✅ حُفظ" if ok else "❌", icon="✅" if ok else "❌")
            st.rerun()
    with sn2:
        if st.button("♻️ استعد snapshot", key="nsm_live_snap_restore", use_container_width=True):
            n = term.restore_sessions_snapshot()
            st.toast(f"✅ استُعيدت {n} جلسة", icon="✅")
            st.rerun()

    st.caption("ملاحظة: السجل الكامل (التدقيق) محفوظ في ai/nsm_terminal.py ويُقرأ "
               "عبر امر `audit` داخل التيرمنال.")
