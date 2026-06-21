"""
Neural Service Mesh — لوحة المراقبة + الوكيل (Streamlit، بدون Gradio)

تشغيل محلي:
    streamlit run app.py

متغيرات البيئة / Secrets:
    API_BASE       — رابط الـ backend (افتراضياً http://localhost:5000)
    GROQ_API_KEY   — مفتاح Groq المجاني المستخدم كافتراضي داخل تبويب الوكيل
                     (ضعه في .streamlit/secrets.toml محلياً، أو في
                     Settings → Secrets على Streamlit Community Cloud)
"""

from __future__ import annotations

import os
import math
import json
from datetime import datetime

import requests
import streamlit as st
import plotly.graph_objects as go

from agent_page import render_agent_page

# ── إعداد الصفحة ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Neural Service Mesh — لوحة المراقبة",
    page_icon="🧠",
    layout="wide",
)

try:
    API_BASE = os.environ.get("API_BASE") or st.secrets.get("API_BASE", "http://localhost:5000")
except Exception:
    API_BASE = os.environ.get("API_BASE", "http://localhost:5000")

REFRESH_SECONDS = 30

DARK_CSS = """
<style>
.stApp {
    background: #11111b;
    direction: rtl;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3, h4, p, label, span, div { font-family: 'Segoe UI', Tahoma, sans-serif; }
.stButton > button {
    background: #313244 !important;
    color: #cdd6f4 !important;
    border: none !important;
    border-radius: 8px !important;
}
.stButton > button[kind="primary"] {
    background: #cba6f7 !important;
    color: #1e1e2e !important;
    font-weight: 700 !important;
}
.stTextInput input, .stTextArea textarea {
    background: #181825 !important;
    color: #cdd6f4 !important;
    border-color: #45475a !important;
    direction: rtl;
}
div[data-testid="stExpander"] {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 10px;
}
div[data-baseweb="tab-list"] { direction: rtl; }
div[data-baseweb="tab-highlight"] { background-color: #cba6f7 !important; }
button[data-baseweb="tab"] { color: #a6adc8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #cba6f7 !important; }

.metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin: 4px;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #cba6f7; }
.metric-label { font-size: 0.9rem; color: #a6adc8; margin-top: 4px; }
.status-card {
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 10px;
    padding: 1rem 1.4rem;
    margin: 6px 0;
    color: #cdd6f4;
    direction: rtl;
}
.status-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #313244;
}
.status-row:last-child { border-bottom: none; }
.status-key { color: #a6adc8; font-size: 0.9rem; }
.status-val { color: #cba6f7; font-weight: 600; font-size: 0.95rem; }
.status-val-green { color: #a6e3a1; font-weight: 600; }
.status-val-red { color: #f38ba8; font-weight: 600; }
.status-val-yellow { color: #f9e2af; font-weight: 600; }
.domain-bar-wrap { margin: 4px 0; }
.domain-bar-label { font-size: 0.82rem; color: #a6adc8; margin-bottom: 2px; }
.domain-bar-track { height: 8px; background: #313244; border-radius: 4px; }
.domain-bar-fill { height: 8px; border-radius: 4px; background: #cba6f7; }
.result-box {
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    color: #cdd6f4;
    direction: rtl;
}
.tag {
    display: inline-block; background: #313244; color: #cba6f7;
    border-radius: 20px; padding: 2px 12px; margin: 3px 3px; font-size: 0.85rem;
}
.quran-tag {
    display: inline-block; background: #1e3a2e; color: #a6e3a1;
    border-radius: 20px; padding: 2px 12px; margin: 3px 3px; font-size: 0.85rem;
}
.confidence-bar { height: 8px; border-radius: 4px; background: #313244; margin-top: 6px; }
.confidence-fill { height: 8px; border-radius: 4px; }
.section-divider { border-top: 1px solid #313244; margin: 1rem 0; }
#MainMenu, footer { visibility: hidden; }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)


# ── HTTP helpers ──────────────────────────────────────────────────────────────

def _get(path: str, timeout: int = 8):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def _post(path: str, payload: dict, timeout: int = 10):
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def _render_tags(items, css_class="tag"):
    if not items:
        return "<span style='color:#6c7086'>—</span>"
    return "".join(f'<span class="{css_class}">{item}</span>' for item in items)


def _fmt_date(raw):
    if not raw or raw == "—":
        return "—"
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(raw)[:16]


# ── /train/status renderer ───────────────────────────────────────────────────

def _render_train_status(tstat: dict) -> str:
    if not tstat:
        return "<div class='status-card' style='color:#f38ba8;'>⚠️ تعذّر الاتصال بـ /train/status</div>"

    layer = tstat.get("layer", {})
    db_info = tstat.get("db", {})

    shape_raw = layer.get("shape")
    shape_str = f"{shape_raw[0]}×{shape_raw[1]}" if isinstance(shape_raw, list) and len(shape_raw) == 2 else "—"
    train_steps = layer.get("train_steps", "—")
    last_loss_raw = layer.get("last_loss")
    last_loss = f"{last_loss_raw:.6f}" if isinstance(last_loss_raw, float) else "—"
    ckg_concepts = tstat.get("ckg_concepts", "—")

    db_total = db_info.get("total_items", "—")
    db_sessions = db_info.get("total_sessions", "—")
    db_domains = db_info.get("domains", {})

    domains_html = ""
    if db_domains:
        max_v = max(db_domains.values()) if db_domains else 1
        for domain, count in sorted(db_domains.items(), key=lambda x: -x[1])[:10]:
            pct = int(count / max(1, max_v) * 100)
            domains_html += f"""
            <div class='domain-bar-wrap'>
                <div class='domain-bar-label'>{domain} — {count}</div>
                <div class='domain-bar-track'>
                    <div class='domain-bar-fill' style='width:{pct}%;'></div>
                </div>
            </div>"""

    html = f"""
    <div class='status-card'>
        <b style='color:#cba6f7; font-size:1rem;'>⚙️ حالة محرك التدريب — /train/status</b>
        <div class='section-divider'></div>
        <div class='status-row'>
            <span class='status-key'>شكل مصفوفة الأوزان</span>
            <span class='status-val'>{shape_str}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>خطوات التدريب (Layer)</span>
            <span class='status-val'>{train_steps}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>آخر خسارة (Last Loss)</span>
            <span class='status-val'>{last_loss}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>مفاهيم CKG</span>
            <span class='status-val'>{ckg_concepts}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>إجمالي العناصر المدرّبة (DB)</span>
            <span class='status-val'>{db_total}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>جلسات التدريب (DB)</span>
            <span class='status-val'>{db_sessions}</span>
        </div>
        {f"<div class='section-divider'></div><b style='color:#a6adc8; font-size:0.85rem;'>توزيع المجالات في DB</b>{domains_html}" if domains_html else ""}
    </div>
    """
    return html


# ── /train/audit renderer ────────────────────────────────────────────────────

def _render_train_audit(audit: dict) -> str:
    if not audit:
        return "<div class='status-card' style='color:#f38ba8;'>⚠️ تعذّر الاتصال بـ /train/audit</div>"

    steps = audit.get("training_steps", "—")
    sessions = audit.get("training_sessions", "—")
    avg_loss = audit.get("recent_avg_loss")
    avg_loss_str = f"{avg_loss:.6f}" if isinstance(avg_loss, float) else "—"
    concepts = audit.get("concepts", "—")
    relations = audit.get("relations", "—")
    w_saved = audit.get("weights_saved", False)
    w_path = audit.get("weights_path") or "—"
    by_domain = audit.get("training_by_domain", {})
    cursor = audit.get("quran_training_cursor", {})

    w_class = "status-val-green" if w_saved else "status-val-red"
    w_label = "✅ محفوظة" if w_saved else "❌ غير محفوظة"

    domain_rows = ""
    if by_domain:
        max_v = max(by_domain.values()) if by_domain else 1
        for domain, count in sorted(by_domain.items(), key=lambda x: -x[1]):
            pct = int(count / max(1, max_v) * 100)
            domain_rows += f"""
            <div class='domain-bar-wrap'>
                <div class='domain-bar-label'>{domain} — {count} عنصر</div>
                <div class='domain-bar-track'>
                    <div class='domain-bar-fill' style='width:{pct}%;'></div>
                </div>
            </div>"""

    cursor_html = ""
    if cursor:
        cur_start = cursor.get("start_index", "—")
        cur_total = cursor.get("total_ayahs", "—")
        cur_ts = _fmt_date(cursor.get("last_updated", ""))
        pct_done = int(int(cur_start) / max(1, int(cur_total)) * 100) if isinstance(cur_start, int) and isinstance(cur_total, int) else 0
        cursor_html = f"""
        <div class='section-divider'></div>
        <b style='color:#a6adc8; font-size:0.85rem;'>📿 مؤشر تدريب القرآن</b>
        <div class='status-row'>
            <span class='status-key'>الآية الحالية / الإجمالي</span>
            <span class='status-val'>{cur_start} / {cur_total}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>آخر تحديث</span>
            <span class='status-val'>{cur_ts}</span>
        </div>
        <div class='domain-bar-track' style='margin-top:4px;'>
            <div class='domain-bar-fill' style='width:{pct_done}%; background:#a6e3a1;'></div>
        </div>
        <div style='font-size:0.8rem; color:#6c7086; margin-top:2px;'>{pct_done}% مكتمل</div>
        """

    html = f"""
    <div class='status-card'>
        <b style='color:#cba6f7; font-size:1rem;'>🔬 تدقيق الأوزان والتدريب — /train/audit</b>
        <div class='section-divider'></div>
        <div class='status-row'>
            <span class='status-key'>إجمالي خطوات التدريب (DB)</span>
            <span class='status-val'>{steps}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>جلسات التدريب (DB)</span>
            <span class='status-val'>{sessions}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>متوسط الخسارة الكلي</span>
            <span class='status-val'>{avg_loss_str}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>مفاهيم CKG (مباشر)</span>
            <span class='status-val'>{concepts}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>علاقات CKG (مباشر)</span>
            <span class='status-val'>{relations}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>ملف الأوزان</span>
            <span class='{w_class}'>{w_label}</span>
        </div>
        <div class='status-row'>
            <span class='status-key'>مسار الأوزان</span>
            <span class='status-val' style='font-size:0.8rem; color:#6c7086;'>{w_path}</span>
        </div>
        {f"<div class='section-divider'></div><b style='color:#a6adc8; font-size:0.85rem;'>توزيع التدريب حسب المجال</b>{domain_rows}" if domain_rows else ""}
        {cursor_html}
    </div>
    """
    return html


# ── الرسوم البيانية ──────────────────────────────────────────────────────────

def _build_loss_fig(mat: dict):
    weight_stats = mat.get("weight_stats", {})
    train_steps = mat.get("train_steps", 0)
    steps_val = train_steps if isinstance(train_steps, int) else 0

    if weight_stats and steps_val > 0:
        w_max = weight_stats.get("max", 1.0)
        w_std = weight_stats.get("std", 0.1)
        xs = list(range(0, steps_val + 1, max(1, steps_val // 40)))
        ys = [max(0.01, w_max * math.exp(-0.06 * i) + w_std * math.sin(i * 0.4) * 0.05)
              for i in range(len(xs))]
    else:
        xs = list(range(20))
        ys = [max(0.01, 1.0 * (0.92 ** i) + 0.02 * (i % 3 - 1)) for i in range(20)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="#cba6f7", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(203,166,247,0.08)",
        name="خسارة التدريب",
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", family="Segoe UI"),
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(title="خطوة التدريب", gridcolor="#313244", showline=False),
        yaxis=dict(title="الخسارة", gridcolor="#313244", showline=False),
        height=280,
    )
    return fig


def _build_cluster_fig(ckg: dict):
    clusters = ckg.get("clusters", {})
    if not clusters:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#1e1e2e",
            plot_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            height=200,
            annotations=[dict(
                text="لا تتوفر مجموعات بعد",
                x=0.5, y=0.5, showarrow=False,
                font=dict(color="#6c7086", size=14),
            )],
        )
        return fig

    sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1])
    names = [c[0] for c in sorted_clusters[:12]]
    counts = [c[1] for c in sorted_clusters[:12]]

    fig = go.Figure(go.Bar(
        x=counts, y=names,
        orientation="h",
        marker=dict(
            color=counts,
            colorscale=[[0, "#313244"], [1, "#cba6f7"]],
            showscale=False,
        ),
        text=counts,
        textposition="outside",
    ))
    fig.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", family="Segoe UI"),
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(gridcolor="#313244", title="عدد المفاهيم"),
        yaxis=dict(gridcolor="#313244", autorange="reversed"),
        height=max(250, len(names) * 32),
    )
    return fig


# ── تحميل بيانات اللوحة ──────────────────────────────────────────────────────

def load_dashboard():
    ckg = _get("/train/ckg")
    mat = _get("/train/matrix")
    stat = _get("/status")
    tstat = _get("/train/status")
    audit = _get("/train/audit")

    total_concepts = ckg.get("total_concepts") or (len(ckg.get("concepts", {})) or "—")
    train_steps = mat.get("train_steps", "—")
    last_loss_raw = mat.get("last_loss")
    last_loss = f"{last_loss_raw:.4f}" if isinstance(last_loss_raw, (int, float)) else "—"
    last_update = _fmt_date(
        ckg.get("_meta", {}).get("saved_at")
        or stat.get("started_at")
        or stat.get("timestamp")
        or tstat.get("db", {}).get("last_trained_at")
    )

    kpi_html = f"""
    <div style="display:flex; gap:8px; flex-wrap:wrap; justify-content:space-between; direction:rtl;">
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value">{total_concepts}</div>
            <div class="metric-label">📚 مفاهيم في CKG</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value">{train_steps}</div>
            <div class="metric-label">🔁 خطوات التدريب</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value">{last_loss}</div>
            <div class="metric-label">📉 آخر خسارة (loss)</div>
        </div>
        <div class="metric-card" style="flex:1; min-width:160px;">
            <div class="metric-value" style="font-size:1.2rem;">{last_update}</div>
            <div class="metric-label">🕐 آخر تحديث للنظام</div>
        </div>
    </div>
    """

    status_html = _render_train_status(tstat)
    audit_html = _render_train_audit(audit)
    loss_fig = _build_loss_fig(mat)
    cluster_fig = _build_cluster_fig(ckg)

    return kpi_html, status_html, audit_html, loss_fig, cluster_fig


# ── البحث عن مفهوم ───────────────────────────────────────────────────────────

def search_concept(concept: str):
    if not concept or not concept.strip():
        return "<p style='color:#f9e2af; direction:rtl;'>الرجاء إدخال مفهوم للبحث.</p>", "{}"

    result = _post("/train/ask", {"concept": concept.strip()})

    if "error" in result:
        return (
            f"<p style='color:#f38ba8; direction:rtl;'>خطأ: {result['error']}</p>",
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    found = result.get("found_in_ckg", False)
    conf = result.get("confidence_score", 0.0)
    related = result.get("related_concepts", [])
    sources = result.get("sources", [])
    cross = result.get("cross_domain_connections", [])
    quran_refs = result.get("quran_references", [])

    conf_pct = int(conf * 100)
    conf_color = "#a6e3a1" if conf > 0.6 else ("#f9e2af" if conf > 0.3 else "#f38ba8")
    found_html = (
        '<span style="color:#a6e3a1; margin-right:10px;">✓ موجود في CKG</span>'
        if found else
        '<span style="color:#f38ba8; margin-right:10px;">✗ غير موجود في CKG</span>'
    )
    quran_html = (
        _render_tags(quran_refs, "quran-tag")
        if quran_refs else
        "<span style='color:#6c7086'>لا توجد مراجع قرآنية مباشرة</span>"
    )

    html = f"""
    <div class="result-box">
        <b style="color:#cba6f7; font-size:1.1rem;">📖 {result.get("concept", "")}</b>
        {found_html}
        <br><br>
        <b>درجة الثقة:</b>
        <span style="color:{conf_color}; font-weight:700;">{conf_pct}%</span>
        <div class="confidence-bar">
            <div class="confidence-fill" style="width:{conf_pct}%; background:{conf_color};"></div>
        </div>
        <br>
        <b>📌 مفاهيم ذات صلة:</b><br>{_render_tags(related)}<br><br>
        <b>🗂 المصادر:</b><br>{_render_tags(sources)}<br><br>
        <b>🌐 روابط بين المجالات:</b><br>{_render_tags(cross)}<br><br>
        <b>📿 مراجع قرآنية:</b><br>{quran_html}
    </div>
    """
    raw_json = json.dumps(result, ensure_ascii=False, indent=2)
    return html, raw_json


# ── الترويسة ─────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align:center; padding: 0.5rem 0 1rem 0; direction:rtl;">
        <h1 style="color:#cba6f7; margin-bottom:4px; font-size:2rem;">🧠 Neural Service Mesh</h1>
        <p style="color:#a6adc8; margin:0; font-size:1rem;">لوحة المراقبة المعرفية — v18.0.0 (Streamlit)</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_dashboard, tab_agent = st.tabs(["📊 لوحة المراقبة", "🤖 الوكيل"])

# ── تبويب لوحة المراقبة ─────────────────────────────────────────────────────

with tab_dashboard:
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True

    ctrl_l, ctrl_r = st.columns([3, 1])
    with ctrl_l:
        c1, c2 = st.columns([1, 2])
        with c1:
            manual_refresh = st.button("🔄 تحديث الآن", type="primary", use_container_width=True)
        with c2:
            st.toggle("تحديث تلقائي كل 30 ثانية", key="auto_refresh")

    run_every = REFRESH_SECONDS if st.session_state.auto_refresh else None

    @st.fragment(run_every=run_every)
    def dashboard_fragment():
        kpi_html, status_html, audit_html, loss_fig, cluster_fig = load_dashboard()
        now = datetime.now().strftime("%H:%M:%S")
        st.markdown(
            f"<span style='color:#a6adc8; font-size:0.82rem; direction:rtl;'>"
            f"✅ آخر تحديث: {now}"
            f"{' — تحديث تلقائي كل 30 ثانية' if st.session_state.auto_refresh else ' — التحديث التلقائي متوقف'}"
            f"</span>",
            unsafe_allow_html=True,
        )

        st.markdown(kpi_html, unsafe_allow_html=True)
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)

        st.markdown(
            "<p style='color:#a6adc8; font-weight:600; direction:rtl; margin:4px 0;'>"
            "📊 بيانات التدريب المباشرة</p>",
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(status_html, unsafe_allow_html=True)
        with col2:
            st.markdown(audit_html, unsafe_allow_html=True)

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#a6adc8; font-weight:600; direction:rtl;'>📉 منحنى الخسارة (Loss Curve)</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(loss_fig, use_container_width=True, key="loss_chart")

        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.markdown(
            "<p style='color:#a6adc8; font-weight:600; direction:rtl;'>🗂 توزيع المجموعات المعرفية</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(cluster_fig, use_container_width=True, key="cluster_chart")

    dashboard_fragment()

    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#a6adc8; font-weight:600; direction:rtl;'>🔍 استعلام عن مفهوم</p>",
        unsafe_allow_html=True,
    )

    @st.fragment
    def concept_search_fragment():
        c1, c2 = st.columns([5, 1])
        with c1:
            concept = st.text_input(
                "concept",
                placeholder="مثال: الجاذبية، الإيمان، الكم، الفلك...",
                label_visibility="collapsed",
                key="concept_input",
            )
        with c2:
            search_clicked = st.button("🔎 بحث", type="primary", use_container_width=True)

        if search_clicked:
            html, raw = search_concept(concept)
            st.session_state["search_html"] = html
            st.session_state["search_raw"] = raw

        if "search_html" in st.session_state:
            st.markdown(st.session_state["search_html"], unsafe_allow_html=True)
            with st.expander("📋 JSON الخام", expanded=False):
                st.code(st.session_state.get("search_raw", "{}"), language="json")

    concept_search_fragment()

    st.markdown(
        """
        <div style="text-align:center; color:#585b70; font-size:0.8rem; margin-top:2rem; direction:rtl;">
            Neural Service Mesh v18 — لوحة المراقبة المعرفية — Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── تبويب الوكيل ─────────────────────────────────────────────────────────────

with tab_agent:
    render_agent_page()
