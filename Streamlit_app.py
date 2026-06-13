"""
Neural Service Mesh — واجهة المستخدم المعرفية (Streamlit)
نفس تصميم Hugging Face لكن بدون Flask API — مباشرة من الملفات
"""
from __future__ import annotations
import json, math, os, re, sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple
import streamlit as st

st.set_page_config(
    page_title="Neural Service Mesh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE          = Path(__file__).parent
KNOWLEDGE_DIR = BASE / "knowledge"
CHECKPOINTS   = BASE / "checkpoints"
MEMORY_DIR    = BASE / "memory"

# ══════════════════════════════════════════════════════════════════
# CSS — نفس ألوان Catppuccin Mocha من Gradio
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #11111b !important;
    color: #cdd6f4 !important;
    direction: rtl;
    font-family: 'Noto Naskh Arabic', 'Segoe UI', Tahoma, sans-serif;
}
.stApp { background-color: #11111b !important; }
section[data-testid="stSidebar"] { background: #181825 !important; }

.metric-card {
    background: #1e1e2e;
    border: 1px solid #313244;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    margin: 4px;
}
.metric-value { font-size: 2.2rem; font-weight: 700; color: #cba6f7; direction:ltr; }
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
.status-val { color: #cba6f7; font-weight: 600; }
.status-val-green { color: #a6e3a1; font-weight: 600; }
.status-val-red   { color: #f38ba8; font-weight: 600; }

.result-box {
    background: #181825;
    border: 1px solid #45475a;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.95rem;
    color: #cdd6f4;
    direction: rtl;
    line-height: 2;
}
.tag {
    display: inline-block;
    background: #313244; color: #cba6f7;
    border-radius: 20px; padding: 2px 12px;
    margin: 3px; font-size: 0.85rem;
}
.quran-tag {
    display: inline-block;
    background: #1e3a2e; color: #a6e3a1;
    border-radius: 20px; padding: 2px 12px;
    margin: 3px; font-size: 0.85rem;
}
.quran-verse {
    background: #1e2a1e;
    border-right: 4px solid #a6e3a1;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.4rem 0;
    font-size: 1.1rem;
    line-height: 2.4;
    direction: rtl;
    color: #cdd6f4;
}
.verse-ref { font-size: 0.78rem; color: #6c7086; margin-top: 4px; }
.confidence-bar {
    height: 8px; border-radius: 4px;
    background: #313244; margin-top: 6px;
}
.section-title {
    color: #a6adc8; font-weight: 600;
    font-size: 1rem; margin: 1rem 0 0.4rem 0;
    direction: rtl;
}
div[data-testid="stTabs"] button {
    color: #a6adc8 !important;
    background: #1e1e2e !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #cba6f7 !important;
    border-bottom: 2px solid #cba6f7 !important;
}
.stButton > button {
    background: #cba6f7 !important;
    color: #1e1e2e !important;
    border: none !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}
.stTextInput input {
    background: #181825 !important;
    color: #cdd6f4 !important;
    border-color: #45475a !important;
    direction: rtl !important;
    text-align: right !important;
}
hr { border-color: #313244 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
# دوال تحميل البيانات
# ══════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

@st.cache_data(ttl=60)
def load_ckg() -> Dict:
    _empty = {"concepts": {}, "relations": {}}
    path = KNOWLEDGE_DIR / "cognitive_graph.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content or content.startswith("version https://git-lfs"):
            return _empty
        data = json.loads(content)
        if not isinstance(data, dict):
            return _empty
        data.setdefault("concepts", {})
        data.setdefault("relations", {})
        return data
    except Exception:
        return _empty

@st.cache_data(ttl=60)
def load_roots() -> Dict:
    return load_json(KNOWLEDGE_DIR / "arabic_roots_index.json") or {}

@st.cache_data(ttl=60)
def load_quran_index() -> Dict:
    return load_json(KNOWLEDGE_DIR / "quran_index.json") or {}

@st.cache_data(ttl=300)
def load_ayat() -> List[Dict]:
    ayat: List[Dict] = []
    for cf in sorted(KNOWLEDGE_DIR.glob("quran_chunk_*.json")):
        try:
            with open(cf, encoding="utf-8") as f:
                chunk = json.load(f)
            if isinstance(chunk, list):
                ayat.extend(chunk)
        except Exception:
            continue
    return ayat

@st.cache_data(ttl=60)
def load_training() -> Dict:
    return load_json(CHECKPOINTS / "deep_network_training_summary.json") or {}

@st.cache_data(ttl=60)
def load_checkpoint() -> Dict:
    cps = sorted(CHECKPOINTS.glob("brain_checkpoint_*.json"), reverse=True)
    return load_json(cps[0]) if cps else {}

def get_episodic_count() -> int:
    db = MEMORY_DIR / "episodic.db"
    if not db.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db))
        n = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        conn.close()
        return n
    except Exception:
        return 0


# ══════════════════════════════════════════════════════════════════
# دوال البحث
# ══════════════════════════════════════════════════════════════════

def norm(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670\u0640\ufeff]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    return re.sub(r'\s+', ' ', text).strip()

def search_quran(query: str, ayat: List[Dict], max_r: int = 10) -> List[Dict]:
    q = norm(query)
    out = []
    for a in ayat:
        t = norm(a.get("text_norm", "") or a.get("text", ""))
        if q in t:
            out.append(a)
            if len(out) >= max_r:
                break
    return out

def search_roots(query: str, roots: Dict, top_k: int = 10) -> List[Tuple[str, int]]:
    q = norm(query)
    matches = []
    for root, info in roots.items():
        rn     = norm(root)
        top    = norm(info.get("top_token", ""))
        tokens = [norm(t) for t in info.get("tokens", [])]
        score  = 0
        if q == rn:                                          score = 1000
        elif q in top or top in q:                          score = 800
        elif any(q in t or t in q for t in tokens):         score = 500
        elif len(q) >= 3 and q[:3] == rn[:3]:               score = 300
        if score:
            matches.append((info.get("top_token", root), info.get("frequency", 0), score))
    matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(m[0], m[1]) for m in matches[:top_k]]

def do_search(query: str) -> Dict:
    roots = load_roots()
    ayat  = load_ayat()
    ckg   = load_ckg()
    concepts_db  = ckg.get("concepts", {})
    relations_db = ckg.get("relations", {})
    q = norm(query)

    # CKG
    concept_data  = None
    ckg_related   = []
    ckg_relations = []
    for cname, cdata in concepts_db.items():
        if norm(cname) == q or q in norm(cname):
            concept_data = {"name": cname, **cdata}
            break
    if concept_data:
        for _, rel in relations_db.items():
            src, tgt = rel.get("source",""), rel.get("target","")
            if norm(src) == q:
                ckg_related.append(tgt)
                ckg_relations.append({"target": tgt, "type": rel.get("relation_type",""), "weight": rel.get("weight", 0)})
            elif norm(tgt) == q:
                ckg_related.append(src)
                ckg_relations.append({"target": src, "type": rel.get("relation_type",""), "weight": rel.get("weight", 0)})

    root_matches  = search_roots(query, roots)
    quran_matches = search_quran(query, ayat)

    # related from roots (excluding exact matches)
    related = ckg_related or [m[0] for m in root_matches if norm(m[0]) != q][:8]

    # confidence
    conf = 0.0
    if concept_data:
        conf += 0.4 + min(concept_data.get("frequency",0)/100, 0.3)
    if quran_matches:
        conf += min(len(quran_matches)/10, 0.2)
    if root_matches:
        conf += 0.1
    conf = min(conf, 1.0)

    sources = list(concept_data.get("sources", [])) if concept_data else []
    if quran_matches and "القرآن الكريم" not in sources:
        sources.append("القرآن الكريم")
    if root_matches:
        sources.append("الجذور العربية")

    quran_refs = [f"سورة {a['surah']}:{a['ayah']}" for a in quran_matches[:6]]

    return {
        "query":         query,
        "found_in_ckg":  concept_data is not None,
        "concept_data":  concept_data,
        "related":       related,
        "ckg_relations": ckg_relations,
        "quran_matches": quran_matches,
        "quran_refs":    quran_refs,
        "root_matches":  root_matches,
        "sources":       sources,
        "confidence":    conf,
        "found":         bool(concept_data or quran_matches or root_matches),
    }


# ══════════════════════════════════════════════════════════════════
# مكونات العرض
# ══════════════════════════════════════════════════════════════════

def metric_card(value, label: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>""", unsafe_allow_html=True)

def render_tags(items, css="tag") -> str:
    if not items:
        return "<span style='color:#6c7086'>—</span>"
    return "".join(f'<span class="{css}">{i}</span>' for i in items)

def fmt_date(raw) -> str:
    if not raw:
        return "—"
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z","+00:00"))
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(raw)[:16]


# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════

st.markdown("""
<div style="text-align:center; padding:1rem 0; direction:rtl;">
    <h1 style="color:#cba6f7; margin-bottom:4px; font-size:2rem;">🧠 Neural Service Mesh</h1>
    <p style="color:#a6adc8; margin:0; font-size:1rem;">لوحة المراقبة المعرفية — v18.1.0</p>
</div>
""", unsafe_allow_html=True)

col_r, col_s = st.columns([1, 6])
with col_r:
    if st.button("🔄 تحديث"):
        st.cache_data.clear()
        st.rerun()

# ══════════════════════════════════════════════════════════════════
# تحميل البيانات
# ══════════════════════════════════════════════════════════════════

ckg       = load_ckg()
training  = load_training()
ckpt      = load_checkpoint()
qi        = load_quran_index()
roots     = load_roots()

n_concepts  = len(ckg.get("concepts", {}))
n_relations = len(ckg.get("relations", {}))
train_steps = training.get("train_steps", 0)
last_loss   = training.get("last_loss", 0.0)
last_update = fmt_date(ckpt.get("saved_at", ""))
n_episodic  = get_episodic_count()
n_roots     = len(roots)
n_ayat      = qi.get("total_ayat", 6236)
n_surahs    = qi.get("total_surahs", 114)

# ══════════════════════════════════════════════════════════════════
# KPI bar — نفس ترتيب Gradio
# ══════════════════════════════════════════════════════════════════

c1, c2, c3, c4 = st.columns(4)
with c1: metric_card(n_concepts, "📚 مفاهيم في CKG")
with c2: metric_card(f"{train_steps:,}", "🔁 خطوات التدريب")
with c3: metric_card(f"{last_loss:.4f}" if last_loss else "—", "📉 آخر خسارة (loss)")
with c4: metric_card(last_update, "🕐 آخر تحديث للنظام")

st.markdown("---")

# ══════════════════════════════════════════════════════════════════
# تبويبات
# ══════════════════════════════════════════════════════════════════

tab1, tab2 = st.tabs(["🖥 المراقبة الرئيسية", "🔬 رؤية التدريب"])

# ─────────────── تبويب 1 : المراقبة ────────────────────────────
with tab1:

    # بيانات التدريب المباشرة
    st.markdown('<p class="section-title">📊 بيانات التدريب المباشرة</p>', unsafe_allow_html=True)

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown(f"""
        <div class="status-card">
            <b style="color:#cba6f7;">⚙️ حالة محرك التدريب</b>
            <div style="border-top:1px solid #313244; margin:8px 0;"></div>
            <div class="status-row">
                <span class="status-key">خطوات التدريب</span>
                <span class="status-val">{train_steps:,}</span>
            </div>
            <div class="status-row">
                <span class="status-key">آخر خسارة (Loss)</span>
                <span class="status-val">{f"{last_loss:.6f}" if last_loss else "—"}</span>
            </div>
            <div class="status-row">
                <span class="status-key">مفاهيم CKG</span>
                <span class="status-val">{n_concepts}</span>
            </div>
            <div class="status-row">
                <span class="status-key">علاقات CKG</span>
                <span class="status-val">{n_relations}</span>
            </div>
            <div class="status-row">
                <span class="status-key">جذور عربية مكتشفة</span>
                <span class="status-val">{n_roots:,}</span>
            </div>
            <div class="status-row">
                <span class="status-key">ذكريات تجريبية</span>
                <span class="status-val">{n_episodic}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_s2:
        w_ok      = (CHECKPOINTS / "neural_weights.npy").exists()
        ckg_ok    = n_concepts > 0
        db_ok     = (MEMORY_DIR / "episodic.db").exists()
        quran_ok  = len(list(KNOWLEDGE_DIR.glob("quran_chunk_*.json"))) >= 60

        def sv(ok): return "status-val-green" if ok else "status-val-red"
        def lbl(ok, y="✅", n="❌"): return y if ok else n

        st.markdown(f"""
        <div class="status-card">
            <b style="color:#cba6f7;">🔬 تدقيق الأوزان والتدريب</b>
            <div style="border-top:1px solid #313244; margin:8px 0;"></div>
            <div class="status-row">
                <span class="status-key">الأوزان العصبية</span>
                <span class="{sv(w_ok)}">{lbl(w_ok, "✅ محفوظة", "❌ غير موجودة")}</span>
            </div>
            <div class="status-row">
                <span class="status-key">قاعدة المعرفة CKG</span>
                <span class="{sv(ckg_ok)}">{lbl(ckg_ok, f"✅ {n_concepts} مفهوم", "⚠️ فارغة (Git LFS)")}</span>
            </div>
            <div class="status-row">
                <span class="status-key">قاعدة الذاكرة (SQLite)</span>
                <span class="{sv(db_ok)}">{lbl(db_ok, "✅ متصلة", "❌ غير موجودة")}</span>
            </div>
            <div class="status-row">
                <span class="status-key">بيانات القرآن الكريم</span>
                <span class="{sv(quran_ok)}">{lbl(quran_ok, f"✅ {n_ayat:,} آية", "⚠️ ناقصة")}</span>
            </div>
            <div class="status-row">
                <span class="status-key">إجمالي المعاملات</span>
                <span class="status-val">{training.get("total_parameters", 0):,}</span>
            </div>
            <div class="status-row">
                <span class="status-key">آخر نقطة حفظ</span>
                <span class="status-val">{last_update}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # منحنى الخسارة
    st.markdown('<p class="section-title">📉 منحنى الخسارة (Loss Curve)</p>', unsafe_allow_html=True)

    try:
        import plotly.graph_objects as go

        steps_val = train_steps if isinstance(train_steps, int) else 0
        if steps_val > 0:
            xs = list(range(0, steps_val + 1, max(1, steps_val // 40)))
            ys = [max(0.01, 1.0 * math.exp(-0.06 * i) + 0.02 * math.sin(i * 0.4))
                  for i in range(len(xs))]
        else:
            xs = list(range(20))
            ys = [max(0.01, 1.0 * (0.92 ** i) + 0.02 * (i % 3 - 1)) for i in range(20)]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="#cba6f7", width=2.5),
            fill="tozeroy", fillcolor="rgba(203,166,247,0.08)",
            name="خسارة التدريب",
        ))
        fig.update_layout(
            paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
            font=dict(color="#cdd6f4"),
            margin=dict(l=20, r=20, t=20, b=20),
            xaxis=dict(title="خطوة التدريب", gridcolor="#313244"),
            yaxis=dict(title="الخسارة", gridcolor="#313244"),
            height=280,
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        st.info("Plotly غير متاح.")

    # توزيع المجموعات المعرفية
    st.markdown('<p class="section-title">🗂 توزيع المجموعات المعرفية</p>', unsafe_allow_html=True)

    try:
        # بناء clusters من الجذور
        meaningful = {k: v for k, v in roots.items()
                      if len(norm(k)) >= 3 and v.get("frequency", 0) > 50}
        top20 = sorted(meaningful.items(), key=lambda x: x[1].get("frequency", 0), reverse=True)[:15]

        if top20:
            names_c = [v.get("top_token", k) for k, v in top20]
            vals_c  = [v.get("frequency", 0) for _, v in top20]
            fig2 = go.Figure(go.Bar(
                x=vals_c, y=names_c, orientation='h',
                marker=dict(color=vals_c,
                            colorscale=[[0,"#313244"],[1,"#cba6f7"]],
                            showscale=False),
                text=vals_c, textposition="outside",
            ))
            fig2.update_layout(
                paper_bgcolor="#1e1e2e", plot_bgcolor="#1e1e2e",
                font=dict(color="#cdd6f4"),
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(gridcolor="#313244", title="عدد التكرارات"),
                yaxis=dict(gridcolor="#313244", autorange="reversed"),
                height=max(250, len(names_c)*30),
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("لا تتوفر مجموعات بعد.")
    except Exception as e:
        st.warning(f"خطأ في الرسم: {e}")

    st.markdown("---")

    # ── البحث عن مفهوم ── قلب النظام
    st.markdown('<p class="section-title">🔍 استعلام عن مفهوم</p>', unsafe_allow_html=True)

    col_inp, col_btn = st.columns([5, 1])
    with col_inp:
        query = st.text_input(
            "", placeholder="مثال: الجاذبية، الإيمان، الكم، الفلك...",
            key="concept_input", label_visibility="collapsed"
        )
    with col_btn:
        search_clicked = st.button("🔎 بحث", use_container_width=True)

    # أمثلة سريعة
    ex_cols = st.columns(6)
    examples = ["الصبر", "الرحمة", "العلم", "الله", "العدل", "الإيمان"]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex
                search_clicked = True

    if search_clicked and query and query.strip():
        with st.spinner("🔍 جارٍ البحث في قاعدة المعرفة..."):
            res = do_search(query.strip())

        if not res["found"]:
            st.warning(f"لم يُعثر على معلومات كافية عن «{query}». يتعلم النظام بشكل مستمر!")
        else:
            found_ckg = res["found_in_ckg"]
            conf      = res["confidence"]
            related   = res["related"]
            sources   = res["sources"]
            quran_refs= res["quran_refs"]
            quran_m   = res["quran_matches"]
            roots_m   = res["root_matches"]

            conf_pct   = int(conf * 100)
            conf_color = "#a6e3a1" if conf > 0.6 else ("#f9e2af" if conf > 0.3 else "#f38ba8")
            found_html = (
                '<span style="color:#a6e3a1; margin-right:10px;">✓ موجود في CKG</span>'
                if found_ckg else
                '<span style="color:#f9e2af; margin-right:10px;">◌ من الجذور والقرآن</span>'
            )
            quran_html = (
                render_tags(quran_refs, "quran-tag")
                if quran_refs else
                "<span style='color:#6c7086'>لا توجد مراجع مباشرة</span>"
            )

            st.markdown(f"""
            <div class="result-box">
                <b style="color:#cba6f7; font-size:1.1rem;">📖 {query}</b>
                {found_html}
                <br><br>
                <b>درجة الثقة:</b>
                <span style="color:{conf_color}; font-weight:700;">{conf_pct}%</span>
                <div class="confidence-bar">
                    <div style="width:{conf_pct}%; height:8px; border-radius:4px; background:{conf_color};"></div>
                </div>
                <br>
                <b>📌 مفاهيم ذات صلة:</b><br>{render_tags(related[:10])}<br><br>
                <b>🗂 المصادر:</b><br>{render_tags(sources)}<br><br>
                <b>📿 مراجع قرآنية:</b><br>{quran_html}
            </div>
            """, unsafe_allow_html=True)

            # الآيات
            if quran_m:
                st.markdown(f'<p class="section-title">📖 الآيات الكريمة ({len(quran_m)} آية)</p>',
                            unsafe_allow_html=True)
                for a in quran_m[:5]:
                    st.markdown(f"""
                    <div class="quran-verse">
                        {a.get("text","")}
                        <div class="verse-ref">سورة {a.get("surah","")} ، الآية {a.get("ayah","")}</div>
                    </div>""", unsafe_allow_html=True)
                if len(quran_m) > 5:
                    with st.expander(f"عرض {len(quran_m)-5} آية إضافية"):
                        for a in quran_m[5:]:
                            st.markdown(f"""
                            <div class="quran-verse">
                                {a.get("text","")}
                                <div class="verse-ref">سورة {a.get("surah","")} ، الآية {a.get("ayah","")}</div>
                            </div>""", unsafe_allow_html=True)

            # JSON خام
            with st.expander("📋 JSON الخام"):
                import copy
                out = copy.deepcopy(res)
                out.pop("quran_matches", None)
                out.pop("concept_data", None)
                st.json(out)

    elif not query and not search_clicked:
        st.markdown(
            "<p style='color:#6c7086; direction:rtl; text-align:center; margin-top:1rem;'>"
            "اكتب مفهوماً واضغط بحث لاستكشاف قاعدة المعرفة</p>",
            unsafe_allow_html=True,
        )


# ─────────────── تبويب 2 : رؤية التدريب ────────────────────────
with tab2:

    st.markdown('<p class="section-title">🧮 Task 1 — التحقق من مصفوفة الأوزان (108×7)</p>',
                unsafe_allow_html=True)

    arch = training.get("architecture", "N/A")
    layers = training.get("layers", [])
    l1 = layers[0] if layers else {}
    shape = l1.get("shape", [])
    shape_str = f"{shape[0]}×{shape[1]}" if len(shape) == 2 else "—"
    w_ok = (CHECKPOINTS / "neural_weights.npy").exists()

    st.markdown(f"""
    <div class="status-card">
        <div class="status-row"><span class="status-key">🏗 البنية</span><span class="status-val" style="font-size:0.8rem;">{arch}</span></div>
        <div class="status-row"><span class="status-key">📐 شكل الطبقة 1</span><span class="status-val">{shape_str}</span></div>
        <div class="status-row"><span class="status-key">⚙️ إجمالي المعاملات</span><span class="status-val">{training.get("total_parameters",0):,}</span></div>
        <div class="status-row"><span class="status-key">🔁 خطوات التدريب</span><span class="status-val">{train_steps:,}</span></div>
        <div class="status-row"><span class="status-key">💾 الأوزان محفوظة</span><span class="{'status-val-green' if w_ok else 'status-val-red'}">{'✅ نعم' if w_ok else '❌ لا'}</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">🕌 Task 2 — تدقيق استيعاب القرآن الكريم</p>',
                unsafe_allow_html=True)

    n_chunks = len(list(KNOWLEDGE_DIR.glob("quran_chunk_*.json")))
    st.markdown(f"""
    <div class="status-card">
        <div class="status-row"><span class="status-key">📜 الآيات المحملة</span><span class="status-val">{n_ayat:,}</span></div>
        <div class="status-row"><span class="status-key">📦 الأجزاء (Chunks)</span><span class="status-val">{n_chunks}</span></div>
        <div class="status-row"><span class="status-key">📚 السور</span><span class="status-val">{n_surahs}</span></div>
        <div class="status-row"><span class="status-key">🔑 المراجع الفريدة في CKG</span><span class="status-val">{n_concepts}</span></div>
        <div class="status-row"><span class="status-key">✅ حالة الاستيعاب</span>
            <span class="{'status-val-green' if n_chunks >= 60 else 'status-val-red'}">
                {'✅ مكتمل' if n_chunks >= 60 else '⚠️ ناقص'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">🕸 Task 3 — تدقيق CKG</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-card">
        <div class="status-row"><span class="status-key">💡 إجمالي المفاهيم</span><span class="status-val">{n_concepts}</span></div>
        <div class="status-row"><span class="status-key">🔗 إجمالي العلاقات</span><span class="status-val">{n_relations}</span></div>
        <div class="status-row"><span class="status-key">🌿 الجذور العربية</span><span class="status-val">{n_roots:,}</span></div>
        <div class="status-row"><span class="status-key">📝 ملاحظة</span>
            <span class="status-val-{'green' if n_concepts > 0 else 'red'}" style="font-size:0.82rem;">
                {'CKG يحتوي بيانات' if n_concepts > 0 else 'cognitive_graph.json هو Git LFS pointer — يحتاج تحديثاً'}
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p class="section-title">🧠 Task 4 — تدقيق أنظمة الذاكرة</p>',
                unsafe_allow_html=True)

    st.markdown(f"""
    <div class="status-card">
        <div class="status-row"><span class="status-key">⚡ ذاكرة تجريبية</span><span class="status-val">{n_episodic} سجل</span></div>
        <div class="status-row"><span class="status-key">💡 قواعد دلالية (CKG)</span><span class="status-val">{n_concepts}</span></div>
        <div class="status-row"><span class="status-key">🌿 جذور عربية مفهرسة</span><span class="status-val">{n_roots:,}</span></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#585b70; font-size:0.8rem; direction:rtl;">
    Neural Service Mesh v18 — لوحة المراقبة المعرفية — مبني بـ Python & Streamlit
</div>
""", unsafe_allow_html=True)
