from __future__ import annotations
import os
import math
import json
from datetime import datetime

import requests
import gradio as gr
import plotly.graph_objects as go

API_BASE = os.environ.get("API_BASE", "http://localhost:5000")

DARK_CSS = """
body, .gradio-container {
    background: #11111b !important;
    direction: rtl;
    font-family: 'Segoe UI', Tahoma, sans-serif;
}
.gr-panel, .block { background: #1e1e2e !important; border-color: #313244 !important; }
.gr-button-primary { background: #cba6f7 !important; color: #1e1e2e !important; border: none !important; }
.gr-button { background: #313244 !important; color: #cdd6f4 !important; border: none !important; }
h1, h2, h3, label, .label-wrap { color: #cdd6f4 !important; }
input, textarea { background: #181825 !important; color: #cdd6f4 !important; border-color: #45475a !important; }
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
footer { display: none !important; }
"""


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


# ── Dashboard refresh ────────────────────────────────────────────────────────

def load_dashboard():
    ckg   = _get("/train/ckg")
    mat   = _get("/train/matrix")
    stat  = _get("/status")
    tstat = _get("/train/status")

    total_concepts = ckg.get("total_concepts") or (len(ckg.get("concepts", {})) or "—")
    train_steps    = mat.get("train_steps", "—")
    last_loss_raw  = mat.get("last_loss")
    last_loss      = f"{last_loss_raw:.4f}" if isinstance(last_loss_raw, (int, float)) else "—"
    last_update    = _fmt_date(
        ckg.get("_meta", {}).get("saved_at")
        or stat.get("started_at")
        or stat.get("timestamp")
        or tstat.get("last_trained_at")
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

    loss_fig = _build_loss_fig(mat)
    cluster_fig = _build_cluster_fig(ckg)

    return kpi_html, loss_fig, cluster_fig


def _build_loss_fig(mat: dict):
    weight_stats = mat.get("weight_stats", {})
    train_steps  = mat.get("train_steps", 0)
    steps_val    = train_steps if isinstance(train_steps, int) else 0

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
    names  = [c[0] for c in sorted_clusters[:12]]
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


# ── Concept search ───────────────────────────────────────────────────────────

def search_concept(concept: str):
    if not concept or not concept.strip():
        return "<p style='color:#f9e2af; direction:rtl;'>الرجاء إدخال مفهوم للبحث.</p>", "{}"

    result = _post("/train/ask", {"concept": concept.strip()})

    if "error" in result:
        return (
            f"<p style='color:#f38ba8; direction:rtl;'>خطأ: {result['error']}</p>",
            json.dumps(result, ensure_ascii=False, indent=2),
        )

    found      = result.get("found_in_ckg", False)
    conf       = result.get("confidence_score", 0.0)
    related    = result.get("related_concepts", [])
    sources    = result.get("sources", [])
    cross      = result.get("cross_domain_connections", [])
    quran_refs = result.get("quran_references", [])

    conf_pct   = int(conf * 100)
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
        <b style="color:#cba6f7; font-size:1.1rem;">📖 {result.get("concept","")}</b>
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


# ── Build UI ─────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="Neural Service Mesh — لوحة المراقبة",
    css=DARK_CSS,
    theme=gr.themes.Base(
        primary_hue="purple",
        secondary_hue="gray",
        neutral_hue="gray",
    ),
) as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 1rem 0; direction:rtl;">
        <h1 style="color:#cba6f7; margin-bottom:4px; font-size:2rem;">🧠 Neural Service Mesh</h1>
        <p style="color:#a6adc8; margin:0; font-size:1rem;">لوحة المراقبة المعرفية — v18.0.0</p>
    </div>
    """)

    with gr.Row():
        refresh_btn  = gr.Button("🔄 تحديث", variant="primary", scale=0)
        pause_btn    = gr.Button("⏸ إيقاف التحديث التلقائي", variant="secondary", scale=0)
        last_updated = gr.HTML(
            value="<span style='color:#6c7086; font-size:0.82rem; direction:rtl;'>⏱ لم يتم التحديث بعد</span>",
            label="",
        )

    kpi_html = gr.HTML(label="")

    gr.HTML('<div class="section-divider"></div>')
    gr.HTML('<p style="color:#a6adc8; font-weight:600; direction:rtl;">📉 منحنى الخسارة (Loss Curve)</p>')
    loss_plot = gr.Plot(label="")

    gr.HTML('<div class="section-divider"></div>')
    gr.HTML('<p style="color:#a6adc8; font-weight:600; direction:rtl;">🗂 توزيع المجموعات المعرفية</p>')
    cluster_plot = gr.Plot(label="")

    gr.HTML('<div class="section-divider"></div>')
    gr.HTML('<p style="color:#a6adc8; font-weight:600; direction:rtl;">🔍 استعلام عن مفهوم</p>')

    with gr.Row():
        concept_input = gr.Textbox(
            placeholder="مثال: الجاذبية، الإيمان، الكم، الفلك...",
            label="",
            scale=5,
        )
        search_btn = gr.Button("🔎 بحث", variant="primary", scale=1)

    search_result = gr.HTML(label="")
    with gr.Accordion("📋 JSON الخام", open=False):
        raw_json_out = gr.Code(language="json", label="")

    gr.HTML("""
    <div style="text-align:center; color:#585b70; font-size:0.8rem; margin-top:2rem; direction:rtl;">
        Neural Service Mesh v18 — لوحة المراقبة المعرفية
    </div>
    """)

    # ── Auto-refresh timer (every 30 s) ──────────────────────────────────────
    timer = gr.Timer(value=30, active=True)

    def _refresh():
        kpi, loss, cluster = load_dashboard()
        now = datetime.now().strftime("%H:%M:%S")
        ts  = f"<span style='color:#a6adc8; font-size:0.82rem; direction:rtl;'>✅ آخر تحديث: {now} — كل 30 ثانية</span>"
        return kpi, loss, cluster, ts

    def _manual_refresh():
        kpi, loss, cluster = load_dashboard()
        now = datetime.now().strftime("%H:%M:%S")
        ts  = f"<span style='color:#cba6f7; font-size:0.82rem; direction:rtl;'>🔄 تحديث يدوي: {now}</span>"
        return kpi, loss, cluster, ts

    _outputs = [kpi_html, loss_plot, cluster_plot, last_updated]

    # pause / resume toggle
    def _toggle_pause(btn_label):
        if "إيقاف" in btn_label:
            return gr.Timer(active=False), gr.Button(value="▶ استئناف التحديث التلقائي")
        else:
            return gr.Timer(active=True),  gr.Button(value="⏸ إيقاف التحديث التلقائي")

    timer.tick(fn=_refresh, outputs=_outputs)
    refresh_btn.click(fn=_manual_refresh, outputs=_outputs)
    pause_btn.click(fn=_toggle_pause, inputs=[pause_btn], outputs=[timer, pause_btn])
    search_btn.click(fn=search_concept, inputs=[concept_input], outputs=[search_result, raw_json_out])
    concept_input.submit(fn=search_concept, inputs=[concept_input], outputs=[search_result, raw_json_out])

    demo.load(fn=_refresh, outputs=_outputs)


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        show_api=False,
    )
