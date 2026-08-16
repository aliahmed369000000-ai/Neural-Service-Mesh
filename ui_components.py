"""مكونات واجهة NSM المخصصة — طبقة عرض صغيرة فوق Streamlit."""
from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st


_DESIGN_CSS = r"""
<style>
:root {
  --nsm-indigo: #6d5dfc;
  --nsm-cyan: #2dd4bf;
  --nsm-amber: #f6c453;
  --nsm-danger: #fb7185;
  --nsm-glass: rgba(255,255,255,.055);
  --nsm-glass-strong: rgba(255,255,255,.09);
  --nsm-line: rgba(148,163,184,.22);
  --nsm-shadow: 0 18px 48px rgba(0,0,0,.16);
}
.nsm-shell-brand { display:flex; align-items:center; gap:.75rem; padding:.55rem .2rem 1.15rem; direction:rtl; }
.nsm-shell-mark { width:42px; height:42px; display:grid; place-items:center; border-radius:14px; color:#fff; font-size:1.35rem; background:linear-gradient(135deg,var(--nsm-indigo),var(--nsm-cyan)); box-shadow:0 10px 24px rgba(45,212,191,.2); }
.nsm-shell-name { font-weight:900; letter-spacing:-.02em; line-height:1.15; color:var(--text); }
.nsm-shell-caption { color:var(--text-muted); font-size:.72rem; margin-top:.18rem; }
.nsm-section-head { display:flex; align-items:center; justify-content:space-between; gap:1rem; padding:.85rem 1rem; margin:.9rem 0 .65rem; border:1px solid var(--nsm-line); border-radius:18px; background:linear-gradient(135deg,var(--nsm-glass-strong),transparent); direction:rtl; }
.nsm-section-title { display:flex; align-items:center; gap:.55rem; font-size:1.05rem; font-weight:850; color:var(--text); }
.nsm-section-kicker { color:var(--text-muted); font-size:.76rem; }
.nsm-section-live { width:8px; height:8px; border-radius:50%; background:var(--nsm-cyan); box-shadow:0 0 0 5px rgba(45,212,191,.12); animation:nsm-live-pulse 2s infinite; }
@keyframes nsm-live-pulse { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.55;transform:scale(.78)} }
.nsm-kpi-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(145px,1fr)); gap:.7rem; margin:.35rem 0 1rem; direction:rtl; }
.nsm-kpi { position:relative; overflow:hidden; padding:1rem 1.05rem .9rem; border:1px solid var(--nsm-line); border-radius:18px; background:linear-gradient(145deg,var(--nsm-glass-strong),var(--nsm-glass)); box-shadow:var(--nsm-shadow); transition:transform .18s ease,border-color .18s ease; }
.nsm-kpi:hover { transform:translateY(-3px); border-color:rgba(109,93,252,.65); }
.nsm-kpi::after { content:""; position:absolute; inset:auto 0 0; height:3px; background:var(--kpi-accent,var(--nsm-indigo)); opacity:.85; }
.nsm-kpi-label { color:var(--text-muted); font-size:.76rem; font-weight:700; }
.nsm-kpi-value { margin-top:.35rem; color:var(--text); font-size:1.55rem; font-weight:900; letter-spacing:-.03em; direction:ltr; text-align:right; }
.nsm-kpi-note { margin-top:.2rem; color:var(--text-muted); font-size:.7rem; }
.nsm-alert-stack { display:grid; gap:.5rem; margin:.3rem 0 1rem; }
.nsm-alert { display:flex; align-items:flex-start; gap:.65rem; padding:.7rem .85rem; border-radius:14px; border:1px solid var(--alert-line); background:var(--alert-bg); direction:rtl; }
.nsm-alert-icon { font-size:1rem; line-height:1.4; }
.nsm-alert-title { color:var(--text); font-weight:850; font-size:.82rem; }
.nsm-alert-detail { color:var(--text-muted); font-size:.76rem; margin-top:.12rem; }
.nsm-agent-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(185px,1fr)); gap:.65rem; margin:.3rem 0 1rem; direction:rtl; }
.nsm-agent-card { padding:.85rem; border:1px solid var(--nsm-line); border-radius:16px; background:var(--nsm-glass); }
.nsm-agent-top { display:flex; align-items:center; justify-content:space-between; gap:.5rem; }
.nsm-agent-name { color:var(--text); font-weight:800; font-size:.84rem; }
.nsm-agent-id { color:var(--text-muted); font-size:.68rem; direction:ltr; }
.nsm-agent-status { display:inline-flex; align-items:center; gap:.35rem; margin-top:.55rem; padding:.25rem .55rem; border-radius:999px; font-size:.7rem; font-weight:800; }
.nsm-agent-status--running { color:var(--nsm-cyan); background:rgba(45,212,191,.12); }
.nsm-agent-status--done { color:#86efac; background:rgba(134,239,172,.12); }
.nsm-agent-status--error { color:var(--nsm-danger); background:rgba(251,113,133,.12); }
.nsm-agent-status--waiting { color:var(--nsm-amber); background:rgba(246,196,83,.12); }
@media (max-width:640px) { .nsm-kpi-grid{grid-template-columns:repeat(2,1fr)} .nsm-section-head{align-items:flex-start;flex-direction:column;gap:.25rem} }

/* شريط الحالة الموحد أسفل العنوان الرئيسي */
.nsm-status-bar {
  display:flex; align-items:center; justify-content:space-between; gap:.65rem;
  flex-wrap:wrap; margin:.25rem 0 1rem; padding:.58rem .75rem;
  border:1px solid var(--nsm-line); border-radius:15px;
  background:linear-gradient(110deg,rgba(109,93,252,.10),rgba(45,212,191,.08));
  box-shadow:0 8px 24px rgba(0,0,0,.08); direction:rtl;
}
.nsm-status-item { display:inline-flex; align-items:center; gap:.42rem; color:var(--text-muted); font-size:.76rem; font-weight:700; white-space:nowrap; }
.nsm-status-item strong { color:var(--text); font-weight:850; }
.nsm-status-dot { width:7px; height:7px; border-radius:50%; background:var(--nsm-cyan); box-shadow:0 0 0 4px rgba(45,212,191,.12); }
.nsm-status-divider { width:1px; height:18px; background:var(--nsm-line); }
.main .block-container { max-width:1440px; padding-top:1.25rem; padding-bottom:2.2rem; }
.stTabs [data-baseweb="tab-list"] { overflow-x:auto !important; scrollbar-width:thin; }
.stTabs [data-baseweb="tab"]:focus-visible, .stButton>button:focus-visible, input:focus-visible, textarea:focus-visible { outline:2px solid var(--nsm-cyan) !important; outline-offset:2px; }
.stButton>button { min-height:2.45rem; font-weight:750 !important; transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease !important; }
.stButton>button:hover { transform:translateY(-1px); box-shadow:0 8px 20px rgba(0,0,0,.12) !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { border-radius:12px !important; }
@media (max-width:640px) {
  .nsm-status-bar { align-items:flex-start; flex-direction:column; gap:.4rem; }
  .nsm-status-divider { display:none; }
  .main .block-container { padding-top:.8rem; }
}
</style>
"""


def inject_design_system() -> None:
    """يحقن CSS مرة واحدة لكل جلسة، مع احترام متغيرات ثيم NSM الحالية."""
    if st.session_state.get("_nsm_design_system_loaded"):
        return
    st.markdown(_DESIGN_CSS, unsafe_allow_html=True)
    st.session_state["_nsm_design_system_loaded"] = True


def render_brand_bar(caption: str = "الذكاء العربي · مراقبة حيّة") -> None:
    st.markdown(
        f'''<div class="nsm-shell-brand"><div class="nsm-shell-mark">✦</div><div><div class="nsm-shell-name">Neural Service Mesh</div><div class="nsm-shell-caption">{escape(caption)}</div></div></div>''',
        unsafe_allow_html=True,
    )


def render_section_header(title: str, kicker: str = "", *, live: bool = False) -> None:
    live_html = '<span class="nsm-section-live"></span>' if live else ""
    kicker_html = f'<span class="nsm-section-kicker">{escape(kicker)}</span>' if kicker else ""
    st.markdown(
        f'<div class="nsm-section-head"><div class="nsm-section-title">{live_html}{escape(title)}</div>{kicker_html}</div>',
        unsafe_allow_html=True,
    )


def render_kpi_cards(cards: Sequence[Mapping[str, Any]]) -> None:
    html = ['<div class="nsm-kpi-grid">']
    for card in cards:
        accent = escape(str(card.get("accent", "var(--nsm-indigo)")))
        label = escape(str(card.get("label", "")))
        value = escape(str(card.get("value", "—")))
        note = escape(str(card.get("note", "")))
        html.append(f'<div class="nsm-kpi" style="--kpi-accent:{accent}"><div class="nsm-kpi-label">{label}</div><div class="nsm-kpi-value">{value}</div><div class="nsm-kpi-note">{note}</div></div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_alert_cards(alerts: Iterable[Mapping[str, Any]], limit: int = 8) -> None:
    palette = {
        "critical": ("🚨", "rgba(251,113,133,.13)", "rgba(251,113,133,.38)"),
        "warning": ("⚠️", "rgba(246,196,83,.13)", "rgba(246,196,83,.38)"),
        "info": ("ℹ️", "rgba(45,212,191,.12)", "rgba(45,212,191,.34)"),
    }
    rows = list(alerts)[-limit:]
    html = ['<div class="nsm-alert-stack">']
    for alert in reversed(rows):
        icon, bg, line = palette.get(str(alert.get("severity")), palette["info"])
        title = escape(str(alert.get("title", "تنبيه")))
        detail = escape(str(alert.get("detail", "")))
        html.append(f'<div class="nsm-alert" style="--alert-bg:{bg};--alert-line:{line}"><div class="nsm-alert-icon">{icon}</div><div><div class="nsm-alert-title">{title}</div><div class="nsm-alert-detail">{detail}</div></div></div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_status_bar(items: Sequence[Mapping[str, Any]]) -> None:
    """يعرض حالة مختصرة للواجهة من دون ربطها بخدمة خارجية أو تغيير منطق الصفحات."""
    html = ['<div class="nsm-status-bar" role="status" aria-label="حالة نظام NSM">']
    for index, item in enumerate(items):
        if index:
            html.append('<span class="nsm-status-divider" aria-hidden="true"></span>')
        label = escape(str(item.get("label", "")))
        value = escape(str(item.get("value", "")))
        html.append(
            f'<span class="nsm-status-item"><span class="nsm-status-dot" aria-hidden="true"></span>'
            f'{label}: <strong>{value}</strong></span>'
        )
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_agent_cards(states: Mapping[str, Mapping[str, Any]]) -> None:
    html = ['<div class="nsm-agent-grid">']
    for agent_id, row in states.items():
        status = str(row.get("status", "waiting"))
        safe_status = status if status in {"running", "done", "error", "waiting"} else "waiting"
        title = escape(str(row.get("title") or agent_id))
        detail = escape(str(row.get("timestamp", "—")))
        html.append(f'<div class="nsm-agent-card"><div class="nsm-agent-top"><div class="nsm-agent-name">{title}</div><div class="nsm-agent-id">{escape(str(agent_id))}</div></div><div class="nsm-agent-status nsm-agent-status--{safe_status}">{escape(status)} · آخر تحديث {detail}</div></div>')
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


__all__ = ["inject_design_system", "render_brand_bar", "render_status_bar", "render_section_header", "render_kpi_cards", "render_alert_cards", "render_agent_cards"]
