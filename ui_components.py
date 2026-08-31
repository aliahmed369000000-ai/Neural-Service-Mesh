"""مكونات واجهة NSM المخصصة — طبقة عرض صغيرة فوق Streamlit."""
from __future__ import annotations

from html import escape
from typing import Any, Iterable, Mapping, Sequence

import streamlit as st


_DESIGN_CSS = r"""
<style>
@font-face {
  font-family: 'NSM Arabic';
  src: url('/app/static/assets/fonts/NotoNaskhArabic-Regular.ttf') format('truetype');
  font-display: swap;
}
:root {
  --nsm-indigo: #6d5dfc;
}
html, body, [data-testid="stAppViewContainer"], [data-testid="stSidebar"], button, input, textarea, select {
  font-family: 'NSM Arabic', 'Noto Naskh Arabic', 'Tahoma', sans-serif !important;
}
:root {
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
.nsm-agent-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(205px,1fr)); gap:.7rem; margin:.3rem 0 1rem; direction:rtl; }
.nsm-agent-card { position:relative; overflow:hidden; padding:.9rem; border:1px solid var(--nsm-line); border-radius:17px; background:linear-gradient(145deg,var(--nsm-glass-strong),var(--nsm-glass)); box-shadow:0 9px 24px rgba(0,0,0,.08); transition:transform .18s ease,border-color .18s ease,box-shadow .18s ease; }
.nsm-agent-card:hover { transform:translateY(-3px); border-color:rgba(45,212,191,.52); box-shadow:0 13px 30px rgba(0,0,0,.13); }
.nsm-agent-card::before { content:""; position:absolute; inset:0 0 auto; height:3px; background:linear-gradient(90deg,var(--nsm-indigo),var(--nsm-cyan)); opacity:.8; }
.nsm-agent-top { display:flex; align-items:flex-start; justify-content:space-between; gap:.5rem; }
.nsm-agent-name { color:var(--text); font-weight:850; font-size:.86rem; line-height:1.45; }
.nsm-agent-id { color:var(--text-muted); font-size:.67rem; direction:ltr; text-align:left; }
.nsm-agent-status { display:inline-flex; align-items:center; gap:.35rem; margin-top:.6rem; padding:.25rem .58rem; border-radius:999px; font-size:.7rem; font-weight:850; }
.nsm-agent-status::before { content:""; width:6px; height:6px; border-radius:50%; background:currentColor; box-shadow:0 0 0 3px color-mix(in srgb,currentColor 16%,transparent); }
.nsm-agent-status--running { color:var(--nsm-cyan); background:rgba(45,212,191,.12); }
.nsm-agent-status--running::before { animation:nsm-live-pulse 1.45s infinite; }
.nsm-agent-status--done { color:#86efac; background:rgba(134,239,172,.12); }
.nsm-agent-status--error { color:var(--nsm-danger); background:rgba(251,113,133,.12); }
.nsm-agent-status--waiting { color:var(--nsm-amber); background:rgba(246,196,83,.12); }
.nsm-agent-meta { display:flex; justify-content:space-between; gap:.45rem; margin-top:.65rem; color:var(--text-muted); font-size:.68rem; }
.nsm-agent-detail { min-height:2.45rem; margin-top:.5rem; color:var(--text-muted); font-size:.73rem; line-height:1.65; }
.nsm-agent-progress { height:4px; margin-top:.65rem; overflow:hidden; border-radius:999px; background:rgba(148,163,184,.16); }
.nsm-agent-progress > span { display:block; height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--nsm-indigo),var(--nsm-cyan)); transition:width .25s ease; }
@media (max-width:640px) { .nsm-kpi-grid{grid-template-columns:repeat(2,1fr)} .nsm-section-head{align-items:flex-start;flex-direction:column;gap:.25rem} .nsm-agent-grid{grid-template-columns:1fr;} }

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
.stButton>button { min-height:2.55rem; border:1px solid rgba(124,92,252,.28) !important; border-radius:12px !important; font-weight:750 !important; background:linear-gradient(135deg,rgba(124,92,252,.16),rgba(45,212,191,.10)) !important; color:var(--text) !important; transition:transform .16s ease,box-shadow .16s ease,border-color .16s ease !important; }
.stButton>button:hover { transform:translateY(-1px); border-color:rgba(45,212,191,.62) !important; box-shadow:0 8px 20px rgba(45,212,191,.12) !important; }
.stButton>button[kind="primary"] { background:linear-gradient(135deg,#7c5cfc,#2dd4bf) !important; color:#071018 !important; border-color:transparent !important; }
[data-testid="stTextInput"] input, [data-testid="stTextArea"] textarea { border-radius:12px !important; border-color:rgba(124,92,252,.28) !important; background:rgba(15,23,42,.56) !important; }
[data-testid="stTextInput"] input:focus, [data-testid="stTextArea"] textarea:focus { border-color:#2dd4bf !important; box-shadow:0 0 0 3px rgba(45,212,191,.12) !important; }
[data-testid="stSidebar"] { border-left:1px solid rgba(148,163,184,.14); background:linear-gradient(180deg,#101827 0%,#0a0e17 100%); }
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] { direction:rtl; }
.stExpander { border:1px solid var(--nsm-line) !important; border-radius:15px !important; background:rgba(255,255,255,.025) !important; }
.stAlert { border-radius:14px !important; }
.stDataFrame, [data-testid="stMetric"] { border-radius:14px; }
@media (max-width:640px) {
  .nsm-status-bar { align-items:flex-start; flex-direction:column; gap:.4rem; }
  .nsm-status-divider { display:none; }
  .main .block-container { padding-top:.8rem; }
}
.nsm-dashboard-hero {
  position:relative; overflow:hidden; direction:rtl; margin:.15rem 0 1.1rem;
  padding:1.25rem 1.35rem; border:1px solid var(--nsm-line); border-radius:20px;
  background:linear-gradient(135deg,rgba(109,93,252,.15),rgba(45,212,191,.08) 58%,rgba(246,196,83,.09));
  box-shadow:0 14px 34px rgba(0,0,0,.10);
}
.nsm-dashboard-hero::after { content:""; position:absolute; width:190px; height:190px; left:-55px; top:-80px; border-radius:50%; background:radial-gradient(circle,rgba(45,212,191,.18),transparent 70%); pointer-events:none; }
.nsm-dashboard-eyebrow { position:relative; color:var(--nsm-cyan); font-size:.76rem; font-weight:850; letter-spacing:.02em; margin-bottom:.3rem; }
.nsm-dashboard-title { position:relative; color:var(--text); font-size:1.42rem; font-weight:900; margin:0 0 .35rem; }
.nsm-dashboard-subtitle { position:relative; max-width:760px; color:var(--text-muted); line-height:1.8; font-size:.91rem; }
.nsm-dashboard-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.65rem; direction:rtl; margin:.8rem 0 1rem; }
.nsm-dashboard-stat { padding:.82rem .9rem; border:1px solid var(--nsm-line); border-radius:15px; background:var(--surface); box-shadow:0 6px 18px rgba(0,0,0,.06); }
.nsm-dashboard-stat-value { color:var(--text); font-size:1.18rem; font-weight:900; line-height:1.25; }
.nsm-dashboard-stat-label { color:var(--text-muted); font-size:.73rem; font-weight:700; margin-top:.28rem; }
.nsm-dashboard-stat-accent { color:var(--nsm-cyan); }
.nsm-dashboard-lower { display:grid; grid-template-columns:minmax(0,1.35fr) minmax(250px,.65fr); gap:.7rem; direction:rtl; margin:.25rem 0 1.1rem; }
.nsm-dashboard-panel { min-height:118px; padding:.9rem 1rem; border:1px solid var(--nsm-line); border-radius:16px; background:color-mix(in srgb,var(--surface) 92%,transparent); }
.nsm-dashboard-panel-title { color:var(--text); font-size:.86rem; font-weight:850; margin-bottom:.4rem; }
.nsm-dashboard-panel-copy { color:var(--text-muted); font-size:.8rem; line-height:1.75; }
.nsm-dashboard-action-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.55rem; direction:rtl; margin:.7rem 0 1.1rem; }
.nsm-dashboard-action-note { color:var(--text-muted); font-size:.76rem; line-height:1.6; margin:.1rem .2rem 0; }
@media (max-width:900px) { .nsm-dashboard-grid{grid-template-columns:repeat(2,minmax(0,1fr));} .nsm-dashboard-lower{grid-template-columns:1fr;} }
@media (max-width:640px) { .nsm-dashboard-hero{padding:1rem;} .nsm-dashboard-title{font-size:1.2rem;} .nsm-dashboard-action-grid{grid-template-columns:1fr;} }

.nsm-agent-monitor-hero { display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; direction:rtl; margin:.35rem 0 1rem; padding:1rem 1.1rem; border:1px solid var(--nsm-line); border-radius:18px; background:linear-gradient(135deg,rgba(109,93,252,.13),rgba(45,212,191,.08)); }
.nsm-agent-monitor-eyebrow { color:var(--nsm-cyan); font-size:.72rem; font-weight:850; }
.nsm-agent-monitor-title { color:var(--text); font-size:1.12rem; font-weight:900; margin-top:.22rem; }
.nsm-agent-monitor-copy { color:var(--text-muted); font-size:.78rem; line-height:1.7; margin-top:.3rem; }
.nsm-agent-monitor-chip { flex:0 0 auto; padding:.35rem .62rem; border:1px solid rgba(45,212,191,.28); border-radius:999px; color:var(--nsm-cyan); background:rgba(45,212,191,.09); font-size:.72rem; font-weight:850; white-space:nowrap; }
/* دليل التنقّل وشريط التبويبات الرئيسي */
.nsm-nav-guide {
  display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:.65rem;
  direction:rtl; margin:.45rem 0 .7rem;
}
.nsm-nav-card {
  display:flex; align-items:flex-start; gap:.6rem; min-height:66px;
  padding:.7rem .8rem; border:1px solid var(--nsm-line); border-radius:15px;
  background:linear-gradient(145deg,var(--nsm-glass-strong),var(--nsm-glass));
  transition:transform .16s ease,border-color .16s ease,background .16s ease;
}
.nsm-nav-card:hover { transform:translateY(-2px); border-color:rgba(45,212,191,.52); }
.nsm-nav-card-icon { flex:0 0 auto; width:32px; height:32px; display:grid; place-items:center; border-radius:10px; background:rgba(109,93,252,.16); font-size:1rem; }
.nsm-nav-card-title { color:var(--text); font-size:.78rem; font-weight:850; }
.nsm-nav-card-copy { margin-top:.18rem; color:var(--text-muted); font-size:.68rem; line-height:1.55; }
.stTabs [data-baseweb="tab-list"] {
  gap:.3rem !important; padding:.3rem !important; border:1px solid var(--nsm-line);
  border-radius:16px; background:linear-gradient(135deg,var(--nsm-glass-strong),var(--nsm-glass));
}
.stTabs [data-baseweb="tab"] {
  min-height:2.5rem; padding:.5rem .82rem !important; border-radius:11px;
  color:var(--text-muted); font-weight:800; transition:background .16s ease,color .16s ease,transform .16s ease;
}
.stTabs [data-baseweb="tab"]:hover { background:rgba(45,212,191,.09); color:var(--text); transform:translateY(-1px); }
.stTabs [data-baseweb="tab"][aria-selected="true"] { color:var(--text); background:linear-gradient(135deg,rgba(109,93,252,.22),rgba(45,212,191,.14)); }
.stTabs [data-baseweb="tab-highlight"] { background:var(--nsm-cyan) !important; height:2px !important; }
@media (max-width:900px) { .nsm-nav-guide{grid-template-columns:1fr;} .nsm-nav-card{min-height:auto;} .nsm-agent-monitor-hero{flex-direction:column;} }
@media (max-width:640px) { .stTabs [data-baseweb="tab"]{padding:.45rem .62rem !important;font-size:.78rem;} }
.nsm-health-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.8rem; margin:.7rem 0 1.15rem; }
.nsm-health-card { position:relative; overflow:hidden; min-height:116px; padding:1rem; border:1px solid var(--nsm-line); border-radius:18px; background:linear-gradient(145deg,var(--nsm-glass-strong),var(--nsm-glass)); box-shadow:var(--nsm-shadow); }
.nsm-health-card::before { content:""; position:absolute; inset:0 auto 0 0; width:4px; background:var(--health-accent,var(--nsm-cyan)); }
.nsm-health-card--ok { --health-accent:#2dd4bf; }
.nsm-health-card--warn { --health-accent:#f6c453; }
.nsm-health-card--bad { --health-accent:#fb7185; }
.nsm-health-card--info { --health-accent:#8b7cff; }
.nsm-health-icon { float:right; font-size:1.35rem; margin-left:.55rem; }
.nsm-health-label { color:var(--text-muted); font-size:.78rem; font-weight:800; }
.nsm-health-value { margin-top:.3rem; color:var(--text); font-size:1.48rem; font-weight:950; letter-spacing:-.02em; }
.nsm-health-note { margin-top:.28rem; color:var(--text-muted); font-size:.72rem; line-height:1.45; }
.nsm-system-columns { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,.85fr); gap:1rem; align-items:start; }
.nsm-system-panel { padding:1rem; border:1px solid var(--nsm-line); border-radius:18px; background:var(--nsm-glass); }
.nsm-system-panel-title { display:flex; justify-content:space-between; gap:.6rem; align-items:center; margin-bottom:.8rem; color:var(--text); font-weight:900; }
.nsm-notice-stack { display:grid; gap:.55rem; }
.nsm-notice-row { display:flex; gap:.7rem; align-items:flex-start; padding:.7rem .75rem; border:1px solid var(--notice-line, var(--nsm-line)); border-radius:13px; background:var(--notice-bg, var(--nsm-glass-strong)); }
.nsm-notice-row--critical { --notice-bg:rgba(251,113,133,.10); --notice-line:rgba(251,113,133,.32); }
.nsm-notice-row--warning { --notice-bg:rgba(246,196,83,.10); --notice-line:rgba(246,196,83,.30); }
.nsm-notice-row--info { --notice-bg:rgba(45,212,191,.08); --notice-line:rgba(45,212,191,.28); }
.nsm-notice-mark { flex:0 0 auto; font-size:1.05rem; }
.nsm-notice-title { color:var(--text); font-size:.82rem; font-weight:900; }
.nsm-notice-detail { margin-top:.15rem; color:var(--text-muted); font-size:.72rem; line-height:1.5; }
.nsm-notice-time { margin-right:auto; color:var(--text-muted); font-size:.67rem; white-space:nowrap; }
.nsm-provider-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:.55rem; }
.nsm-provider { display:flex; gap:.55rem; align-items:center; padding:.6rem .65rem; border:1px solid var(--nsm-line); border-radius:12px; background:var(--nsm-glass-strong); }
.nsm-provider-dot { width:9px; height:9px; flex:0 0 auto; border-radius:50%; background:#94a3b8; box-shadow:0 0 0 4px rgba(148,163,184,.12); }
.nsm-provider--ready .nsm-provider-dot { background:#2dd4bf; box-shadow:0 0 0 4px rgba(45,212,191,.13); }
.nsm-provider--missing .nsm-provider-dot { background:#f6c453; box-shadow:0 0 0 4px rgba(246,196,83,.13); }
.nsm-provider-name { color:var(--text); font-size:.76rem; font-weight:850; }
.nsm-provider-state { margin-right:auto; color:var(--text-muted); font-size:.67rem; }
.nsm-event-list { display:grid; gap:.25rem; }
.nsm-event-row { display:grid; grid-template-columns:auto 1fr auto; gap:.6rem; align-items:center; padding:.55rem .2rem; border-bottom:1px solid var(--nsm-line); }
.nsm-event-row:last-child { border-bottom:0; }
.nsm-event-dot { width:8px; height:8px; border-radius:50%; background:var(--event-dot,#94a3b8); }
.nsm-event-dot--ok { --event-dot:#2dd4bf; }
.nsm-event-dot--bad { --event-dot:#fb7185; }
.nsm-event-dot--run { --event-dot:#f6c453; box-shadow:0 0 0 4px rgba(246,196,83,.12); }
.nsm-event-main { min-width:0; }
.nsm-event-title { overflow:hidden; color:var(--text); font-size:.75rem; font-weight:850; text-overflow:ellipsis; white-space:nowrap; }
.nsm-event-detail { overflow:hidden; color:var(--text-muted); font-size:.68rem; text-overflow:ellipsis; white-space:nowrap; }
.nsm-event-time { color:var(--text-muted); font-size:.66rem; white-space:nowrap; }
.nsm-empty-state { padding:1rem; border:1px dashed var(--nsm-line); border-radius:14px; color:var(--text-muted); text-align:center; font-size:.78rem; }
@media (max-width:900px) { .nsm-health-grid{grid-template-columns:repeat(2,minmax(0,1fr));} .nsm-system-columns{grid-template-columns:1fr;} }
@media (max-width:640px) { .nsm-health-grid{grid-template-columns:1fr 1fr; gap:.55rem;} .nsm-health-card{min-height:102px; padding:.75rem;} .nsm-health-value{font-size:1.22rem;} .nsm-provider-grid{grid-template-columns:1fr;} .nsm-event-row{grid-template-columns:auto 1fr;} .nsm-event-time{grid-column:2;} }

/* تحسينات واجهة موحّدة: مساحة عمل أهدأ، RTL ثابت، وتسلسل بصري أوضح */
[data-testid="stAppViewContainer"] { background: linear-gradient(160deg, #0a0e17 0%, #0f172a 54%, #111827 100%); }
[data-testid="stSidebar"] { width: 17.5rem !important; }
[data-testid="stSidebar"] > div:first-child { padding: 1.15rem .9rem 1.5rem; }
.main .block-container { max-width: 1380px; padding-left: 2rem; padding-right: 2rem; }
.hero-wrap { margin-bottom: 1.1rem !important; }
.hero-split { border: 1px solid rgba(45,212,191,.2) !important; background: linear-gradient(135deg, rgba(124,92,252,.18), rgba(15,23,42,.72) 58%, rgba(45,212,191,.1)) !important; box-shadow: 0 18px 50px rgba(0,0,0,.22) !important; }
.stTabs [data-baseweb="tab-list"] { position: sticky; z-index: 4; }
.stTabs [data-baseweb="tab"] p { font-size: .86rem; }
@media (max-width: 640px) { .main .block-container { padding-left: .8rem; padding-right: .8rem; } [data-testid="stSidebar"] { width: 100% !important; } }

/* الجولة الثانية: تنقّل أوضح وهوية لونية أكثر هدوءاً على كل المقاسات */
[data-testid="stSidebar"] .stButton > button { min-height: 2.35rem; border: 1px solid rgba(148,163,184,.16); border-radius: 11px; background: rgba(255,255,255,.035); color: var(--text); font-size: .78rem; transition: background .16s ease, border-color .16s ease, transform .16s ease; }
[data-testid="stSidebar"] .stButton > button:hover { border-color: rgba(45,212,191,.5); background: rgba(45,212,191,.1); transform: translateY(-1px); }
[data-testid="stSidebar"] .stButton > button:focus-visible { outline: 2px solid var(--nsm-cyan); outline-offset: 2px; }
[data-testid="stSidebar"] [data-testid="stExpander"] { border: 1px solid rgba(148,163,184,.16); border-radius: 14px; background: rgba(255,255,255,.025); }
[data-testid="stSidebar"] hr { border-color: rgba(148,163,184,.14); margin: .85rem 0; }
.stTabs [data-baseweb="tab-list"] { gap: .35rem; padding: .35rem; border: 1px solid rgba(148,163,184,.16); border-radius: 14px; background: rgba(15,23,42,.76); overflow-x: auto; scrollbar-width: thin; }
.stTabs [data-baseweb="tab"] { min-height: 2.5rem; border-radius: 10px; color: var(--text-muted); white-space: nowrap; }
.stTabs [aria-selected="true"] { background: rgba(45,212,191,.12); color: var(--nsm-cyan); }
[data-testid="stMetric"] { padding: .8rem .9rem; border: 1px solid rgba(148,163,184,.15); border-radius: 15px; background: rgba(255,255,255,.035); }
[data-testid="stMetricLabel"] p { color: var(--text-muted) !important; }
@media (max-width: 640px) { .stTabs [data-baseweb="tab-list"] { margin-inline: -.35rem; } [data-testid="stSidebar"] .stButton > button { font-size: .74rem; } }
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
    status_labels = {
        "running": "⏳ يعمل الآن",
        "done": "✅ اكتملت المهمة",
        "error": "❌ تحتاج مراجعة",
        "waiting": "⏸️ في الانتظار",
    }
    progress_values = {"running": 68, "done": 100, "error": 100, "waiting": 22}
    html = ['<div class="nsm-agent-grid">']
    for agent_id, row in states.items():
        status = str(row.get("status", "waiting"))
        safe_status = status if status in status_labels else "waiting"
        title = escape(str(row.get("title") or agent_id))
        safe_id = escape(str(agent_id))
        timestamp = escape(str(row.get("timestamp", "—")))
        event_type = escape(str(row.get("event_type") or "آخر حدث"))
        detail = escape(str(row.get("detail") or "لا توجد تفاصيل إضافية لهذا الوكيل")[:180])
        duration = row.get("duration_ms")
        duration_label = f"{float(duration):.0f} ms" if duration is not None else "—"
        progress = progress_values[safe_status]
        html.append(
            f'<article class="nsm-agent-card">'
            f'<div class="nsm-agent-top"><div class="nsm-agent-name">{title}</div>'
            f'<div class="nsm-agent-id">{safe_id}</div></div>'
            f'<div class="nsm-agent-status nsm-agent-status--{safe_status}">{status_labels[safe_status]}</div>'
            f'<div class="nsm-agent-detail">{detail}</div>'
            f'<div class="nsm-agent-meta"><span>{event_type}</span><span>{timestamp} · {duration_label}</span></div>'
            f'<div class="nsm-agent-progress" role="progressbar" aria-valuenow="{progress}" aria-valuemin="0" aria-valuemax="100"><span style="width:{progress}%"></span></div>'
            f'</article>'
        )
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_health_cards(cards: Sequence[Mapping[str, Any]]) -> None:
    """يعرض بطاقات صحة موجزة؛ لا يتصل بخدمة خارجية ولا يغيّر حالة النظام."""
    html = ['<div class="nsm-health-grid" role="list" aria-label="مؤشرات صحة النظام">']
    for card in cards:
        tone = str(card.get("tone", "info"))
        if tone not in {"ok", "warn", "bad", "info"}:
            tone = "info"
        icon = escape(str(card.get("icon", "•")))
        label = escape(str(card.get("label", "مؤشر")))
        value = escape(str(card.get("value", "—")))
        note = escape(str(card.get("note", "")))
        html.append(
            f'<article class="nsm-health-card nsm-health-card--{tone}" role="listitem">'
            f'<span class="nsm-health-icon" aria-hidden="true">{icon}</span>'
            f'<div class="nsm-health-label">{label}</div>'
            f'<div class="nsm-health-value">{value}</div>'
            f'<div class="nsm-health-note">{note}</div>'
            f'</article>'
        )
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_provider_cards(providers: Sequence[Mapping[str, Any]]) -> None:
    """يعرض حالة مزودي الذكاء الاصطناعي دون إظهار أي مفتاح أو قيمة سرية."""
    if not providers:
        st.markdown('<div class="nsm-empty-state">لا توجد بيانات مزودين متاحة حالياً.</div>', unsafe_allow_html=True)
        return
    html = ['<div class="nsm-provider-grid" role="list" aria-label="حالة مزودي الذكاء الاصطناعي">']
    for provider in providers:
        ready = bool(provider.get("ready"))
        state = "مهيأ" if ready else "غير مهيأ"
        tone = "ready" if ready else "missing"
        name = escape(str(provider.get("name", "مزود")))
        detail = escape(str(provider.get("detail", state)))
        html.append(
            f'<div class="nsm-provider nsm-provider--{tone}" role="listitem">'
            f'<span class="nsm-provider-dot" aria-hidden="true"></span>'
            f'<span><span class="nsm-provider-name">{name}</span>'
            f'<br><span class="nsm-provider-state">{detail} · {state}</span></span>'
            f'</div>'
        )
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


def render_system_events(events: Sequence[Mapping[str, Any]], limit: int = 8) -> None:
    """يعرض أحدث أحداث النظام في جلسة المستخدم فقط."""
    rows = list(events)[-max(1, int(limit)):]
    if not rows:
        st.markdown('<div class="nsm-empty-state">لا توجد أحداث حيّة بعد. ستظهر هنا عند تشغيل الوكلاء أو المهام.</div>', unsafe_allow_html=True)
        return
    html = ['<div class="nsm-event-list" role="list" aria-label="آخر أحداث النظام">']
    for row in reversed(rows):
        status = str(row.get("status", ""))
        if status in {"error", "failed"}:
            tone = "bad"
        elif status in {"running", "pending"}:
            tone = "run"
        else:
            tone = "ok"
        title = escape(str(row.get("title") or row.get("event_type") or "حدث نظام"))
        detail = escape(str(row.get("detail") or row.get("event_type") or "—")[:140])
        timestamp = escape(str(row.get("timestamp", "—")))
        html.append(
            f'<div class="nsm-event-row" role="listitem">'
            f'<span class="nsm-event-dot nsm-event-dot--{tone}" aria-hidden="true"></span>'
            f'<div class="nsm-event-main"><div class="nsm-event-title">{title}</div>'
            f'<div class="nsm-event-detail">{detail}</div></div>'
            f'<span class="nsm-event-time">{timestamp}</span></div>'
        )
    html.append('</div>')
    st.markdown("".join(html), unsafe_allow_html=True)


__all__ = ["inject_design_system", "render_brand_bar", "render_status_bar", "render_section_header", "render_kpi_cards", "render_alert_cards", "render_agent_cards", "render_health_cards", "render_provider_cards", "render_system_events"]
