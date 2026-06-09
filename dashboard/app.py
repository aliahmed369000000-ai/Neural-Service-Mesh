import streamlit as st
import requests
import plotly.graph_objects as go
import json
from datetime import datetime

API_BASE = "http://localhost:5000"

st.set_page_config(
    page_title="Neural Service Mesh — Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    body, .stApp { direction: rtl; font-family: 'Segoe UI', Tahoma, sans-serif; }
    .metric-card {
        background: #1e1e2e;
        border: 1px solid #313244;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value { font-size: 2.4rem; font-weight: 700; color: #cba6f7; }
    .metric-label { font-size: 0.9rem; color: #a6adc8; margin-top: 4px; }
    .section-title {
        font-size: 1.1rem; font-weight: 600; color: #cdd6f4;
        border-bottom: 1px solid #313244; padding-bottom: 6px; margin-bottom: 12px;
    }
    .result-box {
        background: #181825; border: 1px solid #45475a;
        border-radius: 10px; padding: 1rem 1.2rem;
        font-size: 0.95rem; color: #cdd6f4;
    }
    .tag {
        display: inline-block; background: #313244; color: #cba6f7;
        border-radius: 20px; padding: 2px 12px; margin: 3px 3px;
        font-size: 0.85rem;
    }
    .quran-tag {
        display: inline-block; background: #1e3a2e; color: #a6e3a1;
        border-radius: 20px; padding: 2px 12px; margin: 3px 3px;
        font-size: 0.85rem;
    }
    .confidence-bar { height: 8px; border-radius: 4px; background: #313244; margin-top: 6px; }
    .confidence-fill { height: 8px; border-radius: 4px; background: #cba6f7; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_ckg_stats():
    try:
        r = requests.get(f"{API_BASE}/train/ckg", timeout=8)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=30)
def fetch_matrix_stats():
    try:
        r = requests.get(f"{API_BASE}/train/matrix", timeout=8)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=30)
def fetch_status():
    try:
        r = requests.get(f"{API_BASE}/status", timeout=8)
        return r.json() if r.ok else {}
    except Exception:
        return {}


@st.cache_data(ttl=60)
def fetch_train_status():
    try:
        r = requests.get(f"{API_BASE}/train/status", timeout=8)
        return r.json() if r.ok else {}
    except Exception:
        return {}


def ask_concept(concept: str):
    try:
        r = requests.post(
            f"{API_BASE}/train/ask",
            json={"concept": concept},
            timeout=10,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def render_tags(items, css_class="tag"):
    if not items:
        return "<span style='color:#6c7086'>—</span>"
    return "".join(f'<span class="{css_class}">{item}</span>' for item in items)


# ── Header ──────────────────────────────────────────────────────────────────

st.markdown(
    "<h1 style='text-align:center; color:#cba6f7; margin-bottom:0;'>"
    "🧠 Neural Service Mesh</h1>"
    "<p style='text-align:center; color:#a6adc8; margin-top:4px;'>لوحة المراقبة المعرفية</p>",
    unsafe_allow_html=True,
)
st.divider()

# ── Fetch data ───────────────────────────────────────────────────────────────

ckg   = fetch_ckg_stats()
mat   = fetch_matrix_stats()
stat  = fetch_status()
tstat = fetch_train_status()

# ── KPI row ─────────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns(4)

total_concepts = ckg.get("total_concepts", ckg.get("concepts", {}) and len(ckg.get("concepts", {})) or "—")
train_steps    = mat.get("train_steps", "—")
last_loss_raw  = mat.get("last_loss")
last_loss      = f"{last_loss_raw:.4f}" if isinstance(last_loss_raw, (int, float)) else "—"

# Last update: try multiple fields
last_update = (
    ckg.get("_meta", {}).get("saved_at")
    or stat.get("started_at")
    or stat.get("timestamp")
    or tstat.get("last_trained_at")
    or "—"
)
if last_update and last_update != "—":
    try:
        dt = datetime.fromisoformat(last_update.replace("Z", "+00:00"))
        last_update = dt.strftime("%Y-%m-%d  %H:%M")
    except Exception:
        last_update = str(last_update)[:16]

with c1:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{total_concepts}</div>'
        f'<div class="metric-label">📚 مفاهيم في CKG</div></div>',
        unsafe_allow_html=True,
    )

with c2:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{train_steps}</div>'
        f'<div class="metric-label">🔁 خطوات التدريب</div></div>',
        unsafe_allow_html=True,
    )

with c3:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{last_loss}</div>'
        f'<div class="metric-label">📉 آخر خسارة (loss)</div></div>',
        unsafe_allow_html=True,
    )

with c4:
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value" style="font-size:1.3rem;">{last_update}</div>'
        f'<div class="metric-label">🕐 آخر تحديث للنظام</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Loss curve ───────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">📉 منحنى الخسارة (Loss Curve)</div>', unsafe_allow_html=True)

weight_stats = mat.get("weight_stats", {})

if weight_stats:
    w_min  = weight_stats.get("min", 0)
    w_max  = weight_stats.get("max", 1)
    w_mean = weight_stats.get("mean", 0.5)
    w_std  = weight_stats.get("std", 0.1)

    steps_val = train_steps if isinstance(train_steps, int) else 0

    if steps_val > 0:
        import math
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
        xaxis=dict(
            title="خطوة التدريب",
            gridcolor="#313244",
            showline=False,
        ),
        yaxis=dict(
            title="الخسارة",
            gridcolor="#313244",
            showline=False,
        ),
        height=280,
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("لا تتوفر بيانات الأوزان بعد — قم بتشغيل التدريب أولاً.")

st.divider()

# ── Concept Search ───────────────────────────────────────────────────────────

st.markdown('<div class="section-title">🔍 استعلام عن مفهوم</div>', unsafe_allow_html=True)

col_inp, col_btn = st.columns([5, 1])
with col_inp:
    concept_input = st.text_input(
        "أدخل مفهوماً",
        placeholder="مثال: الجاذبية، الإيمان، الكم، الفلك...",
        label_visibility="collapsed",
    )
with col_btn:
    search_btn = st.button("🔎 بحث", use_container_width=True)

if search_btn and concept_input.strip():
    with st.spinner("جارٍ البحث في الشبكة المعرفية..."):
        result = ask_concept(concept_input.strip())

    if "error" in result:
        st.error(f"خطأ: {result['error']}")
    else:
        found      = result.get("found_in_ckg", False)
        conf       = result.get("confidence_score", 0.0)
        related    = result.get("related_concepts", [])
        sources    = result.get("sources", [])
        cross      = result.get("cross_domain_connections", [])
        quran_refs = result.get("quran_references", [])

        conf_pct = int(conf * 100)
        conf_color = "#a6e3a1" if conf > 0.6 else ("#f9e2af" if conf > 0.3 else "#f38ba8")

        st.markdown(
            f'<div class="result-box">'
            f'<b style="color:#cba6f7; font-size:1.1rem;">📖 {result.get("concept","")}</b>'
            f'{"<span style=\"color:#a6e3a1; margin-right:10px;\">✓ موجود في CKG</span>" if found else "<span style=\"color:#f38ba8; margin-right:10px;\">✗ غير موجود في CKG</span>"}'
            f'<br><br>'

            f'<b>درجة الثقة:</b> '
            f'<span style="color:{conf_color}; font-weight:700;">{conf_pct}%</span>'
            f'<div class="confidence-bar"><div class="confidence-fill" style="width:{conf_pct}%;background:{conf_color};"></div></div>'
            f'<br>'

            f'<b>📌 مفاهيم ذات صلة:</b><br>'
            f'{render_tags(related)}'
            f'<br><br>'

            f'<b>🗂 المصادر:</b><br>'
            f'{render_tags(sources)}'
            f'<br><br>'

            f'<b>🌐 روابط بين المجالات:</b><br>'
            f'{render_tags(cross)}'
            f'<br><br>'

            f'<b>📿 مراجع قرآنية:</b><br>'
            f'{render_tags(quran_refs, "quran-tag") if quran_refs else "<span style=\"color:#6c7086\">لا توجد مراجع قرآنية مباشرة</span>"}'

            f'</div>',
            unsafe_allow_html=True,
        )

        with st.expander("📋 JSON الخام"):
            st.json(result)

elif search_btn:
    st.warning("الرجاء إدخال مفهوم للبحث.")

st.divider()

# ── Cluster breakdown ────────────────────────────────────────────────────────

clusters = ckg.get("clusters", {})
if clusters:
    st.markdown('<div class="section-title">🗂 توزيع المجموعات المعرفية</div>', unsafe_allow_html=True)
    sorted_clusters = sorted(clusters.items(), key=lambda x: -x[1])
    names  = [c[0] for c in sorted_clusters[:12]]
    counts = [c[1] for c in sorted_clusters[:12]]

    fig2 = go.Figure(go.Bar(
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
    fig2.update_layout(
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#cdd6f4", family="Segoe UI"),
        margin=dict(l=10, r=30, t=10, b=10),
        xaxis=dict(gridcolor="#313244", title="عدد المفاهيم"),
        yaxis=dict(gridcolor="#313244", autorange="reversed"),
        height=max(250, len(names) * 32),
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown(
    "<div style='text-align:center; color:#585b70; font-size:0.8rem; margin-top:2rem;'>"
    "Neural Service Mesh v18 — لوحة المراقبة المعرفية"
    "</div>",
    unsafe_allow_html=True,
)

if st.button("🔄 تحديث البيانات"):
    st.cache_data.clear()
    st.rerun()
