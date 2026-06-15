"""
Neural Service Mesh — واجهة المستخدم المعرفية
================================================
Streamlit front-end لمشروع النظام المعرفي العربي.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

# ── محرك الأسئلة والأجوبة القرآني ────────────────────────────────────────
import sys as _sys
_KNOWLEDGE_MODULE_DIR = str(Path(__file__).parent / "knowledge")
if _KNOWLEDGE_MODULE_DIR not in _sys.path:
    _sys.path.insert(0, _KNOWLEDGE_MODULE_DIR)
from qa_engine import answer_question  # noqa: E402
from episodic_memory import (  # noqa: E402
    store_episode, find_similar_episodes, get_memory_stats,
    consolidate_memory, get_semantic_rules,
)

# ── إعداد الصفحة ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="النظام المعرفي العربي | Neural Service Mesh",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── مسارات الملفات ────────────────────────────────────────────────────────
BASE = Path(__file__).parent
KNOWLEDGE_DIR  = BASE / "knowledge"
CHECKPOINTS_DIR = BASE / "checkpoints"
MEMORY_DIR     = BASE / "memory"

# ── CSS مخصص ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Naskh+Arabic:wght@400;600;700&display=swap');

html, body, [class*="css"] {
    direction: rtl;
    font-family: 'Noto Naskh Arabic', 'Segoe UI', sans-serif;
}

.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1a73e8;
    text-align: center;
    padding: 1rem 0 0.3rem 0;
    direction: rtl;
}

.subtitle {
    text-align: center;
    color: #666;
    font-size: 1rem;
    margin-bottom: 1.5rem;
    direction: rtl;
}

.metric-card {
    background: linear-gradient(135deg, #f8faff 0%, #eef2ff 100%);
    border: 1px solid #c7d2fe;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
    margin-bottom: 0.5rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1a73e8;
    direction: ltr;
}

.metric-label {
    font-size: 0.85rem;
    color: #555;
    margin-top: 0.2rem;
    direction: rtl;
}

.concept-card {
    background: #fff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    direction: rtl;
}

.concept-name {
    font-size: 1.6rem;
    font-weight: 700;
    color: #1e3a5f;
    margin-bottom: 0.5rem;
}

.related-tag {
    display: inline-block;
    background: #dbeafe;
    color: #1e40af;
    border-radius: 20px;
    padding: 0.2rem 0.8rem;
    margin: 0.2rem;
    font-size: 0.9rem;
    cursor: pointer;
}

.quran-verse {
    background: linear-gradient(135deg, #fefce8, #fef3c7);
    border-right: 4px solid #f59e0b;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 1.1rem;
    line-height: 2.2;
    direction: rtl;
    color: #1a1a1a;
}

.verse-ref {
    font-size: 0.8rem;
    color: #92400e;
    font-weight: 600;
    margin-top: 0.3rem;
    direction: rtl;
}

.health-ok {
    color: #16a34a;
    font-weight: 600;
}

.health-err {
    color: #dc2626;
    font-weight: 600;
}

.section-header {
    font-size: 1.3rem;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 2px solid #c7d2fe;
    padding-bottom: 0.4rem;
    margin: 1rem 0 0.8rem 0;
    direction: rtl;
}

.tab-content {
    padding: 1rem 0;
}

.search-box input {
    font-size: 1.2rem !important;
    direction: rtl !important;
    text-align: right !important;
}

.root-item {
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    border-radius: 8px;
    padding: 0.6rem 1rem;
    margin: 0.3rem 0;
    direction: rtl;
}

.badge {
    display: inline-block;
    padding: 0.15rem 0.6rem;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: 600;
}

.badge-blue  { background: #dbeafe; color: #1e40af; }
.badge-green { background: #dcfce7; color: #166534; }
.badge-amber { background: #fef3c7; color: #92400e; }
.badge-purple{ background: #f3e8ff; color: #6b21a8; }

stTabs [data-baseweb="tab"] {
    font-size: 1rem;
    direction: rtl;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# دوال تحميل البيانات
# ═══════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=60)
def load_json(path: Path) -> Any:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


@st.cache_data(ttl=60)
def load_arabic_roots() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "arabic_roots_index.json")
    return data or {}


@st.cache_data(ttl=60)
def load_graph_metrics() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "graph_metrics.json")
    return data or {}


@st.cache_data(ttl=60)
def load_quran_index() -> Dict:
    data = load_json(KNOWLEDGE_DIR / "quran_index.json")
    return data or {}


@st.cache_data(ttl=300)
def load_all_quran_ayat() -> List[Dict]:
    """تحميل كل آيات القرآن من الـ chunks."""
    ayat: List[Dict] = []
    chunk_files = sorted(KNOWLEDGE_DIR.glob("quran_chunk_*.json"))
    for cf in chunk_files:
        try:
            with open(cf, encoding="utf-8") as f:
                chunk = json.load(f)
            if isinstance(chunk, list):
                ayat.extend(chunk)
        except Exception:
            continue
    return ayat


@st.cache_data(ttl=60)
def load_latest_checkpoint() -> Dict:
    """تحميل أحدث brain_checkpoint."""
    checkpoints = sorted(CHECKPOINTS_DIR.glob("brain_checkpoint_*.json"), reverse=True)
    if checkpoints:
        data = load_json(checkpoints[0])
        return data or {}
    return {}


@st.cache_data(ttl=60)
def load_training_summary() -> Dict:
    path = CHECKPOINTS_DIR / "deep_network_training_summary.json"
    data = load_json(path)
    return data or {}


@st.cache_data(ttl=60)
def load_ckg() -> Dict:
    """تحميل الـ CKG — يعود بـ {} إذا كان الملف فارغاً أو Git LFS pointer."""
    _empty = {"concepts": {}, "relations": {}}
    path = KNOWLEDGE_DIR / "cognitive_graph.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        # Git LFS pointer — الملف لم يُنزَّل
        if not content or content.startswith("version https://git-lfs"):
            return _empty
        data = json.loads(content)
        # تأكد من وجود المفاتيح الأساسية
        if not isinstance(data, dict):
            return _empty
        if "concepts" not in data:
            data["concepts"] = {}
        if "relations" not in data:
            data["relations"] = {}
        return data
    except Exception:
        return _empty


@st.cache_data(ttl=60)
def load_entities() -> Dict:
    """تحميل طبقة الكيانات المعرفية (entities.json) — يعود بـ {} إن لم تكن موجودة."""
    path = KNOWLEDGE_DIR / "entities.json"
    try:
        content = path.read_text(encoding="utf-8").strip()
        if not content:
            return {}
        data = json.loads(content)
        return data.get("entities", {}) if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_episodic_stats() -> Dict:
    db_path = MEMORY_DIR / "episodic.db"
    stats = {"working": 0, "semantic": 0, "episodic": 0, "rules": 0}
    if not db_path.exists():
        return stats
    try:
        conn = sqlite3.connect(str(db_path))
        episodes_count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        stats["episodic"] = episodes_count
        conn.close()
    except Exception:
        pass
    return stats


# ── تطبيع النص العربي ────────────────────────────────────────────────────
def normalize_arabic(text: str) -> str:
    text = re.sub(r'[\u064B-\u065F\u0670\u0640]', '', text)
    text = re.sub(r'[أإآٱ]', 'ا', text)
    text = re.sub(r'\ufeff', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# منطق البحث المعرفي
# ═══════════════════════════════════════════════════════════════════════════

def search_quran_for_concept(query: str, ayat: List[Dict], max_results: int = 8) -> List[Dict]:
    """البحث في القرآن عن الآيات التي تحتوي على المفهوم."""
    q_norm = normalize_arabic(query)
    results = []
    for ayah in ayat:
        text_norm = normalize_arabic(ayah.get("text_norm", "") or ayah.get("text", ""))
        if q_norm in text_norm:
            results.append(ayah)
            if len(results) >= max_results:
                break
    return results


def find_related_concepts_from_roots(query: str, roots: Dict, top_k: int = 8) -> List[Tuple[str, int]]:
    """إيجاد المفاهيم المرتبطة بناءً على الجذور العربية."""
    q_norm = normalize_arabic(query)
    matches = []
    for root, info in roots.items():
        root_norm = normalize_arabic(root)
        tokens = [normalize_arabic(t) for t in info.get("tokens", [])]
        top_token = normalize_arabic(info.get("top_token", ""))

        score = 0
        if q_norm == root_norm:
            score = 1000
        elif q_norm in top_token or top_token in q_norm:
            score = 800
        elif any(q_norm in t or t in q_norm for t in tokens):
            score = 500
        elif q_norm[:3] == root_norm[:3] and len(q_norm) >= 3:
            score = 300

        if score > 0:
            matches.append((info.get("top_token", root), info.get("frequency", 0), score))

    matches.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return [(m[0], m[1]) for m in matches[:top_k]]


def search_knowledge(query: str) -> Dict:
    """البحث الشامل في قاعدة المعرفة."""
    roots   = load_arabic_roots()
    ayat    = load_all_quran_ayat()
    ckg     = load_ckg()
    concepts_db = ckg.get("concepts", {})
    relations_db = ckg.get("relations", {})

    q_norm = normalize_arabic(query)

    # ── 1. البحث في CKG ──────────────────────────────────────────────────
    concept_data = None
    ckg_related  = []
    ckg_relations = []

    # بحث مباشر
    for cname, cdata in concepts_db.items():
        if normalize_arabic(cname) == q_norm or q_norm in normalize_arabic(cname):
            concept_data = {"name": cname, **cdata}
            break

    if concept_data:
        cname = concept_data["name"]
        for rel_key, rel_data in relations_db.items():
            src = rel_data.get("source", "")
            tgt = rel_data.get("target", "")
            if normalize_arabic(src) == q_norm:
                ckg_related.append(tgt)
                ckg_relations.append({"target": tgt, "type": rel_data.get("relation_type", ""), "weight": rel_data.get("weight", 0)})
            elif normalize_arabic(tgt) == q_norm:
                ckg_related.append(src)
                ckg_relations.append({"target": src, "type": rel_data.get("relation_type", ""), "weight": rel_data.get("weight", 0)})

    # ── 2. البحث في الجذور العربية ───────────────────────────────────────
    root_matches = find_related_concepts_from_roots(query, roots, top_k=8)

    # ── 3. البحث في القرآن ───────────────────────────────────────────────
    quran_matches = search_quran_for_concept(query, ayat, max_results=10)

    # ── 4. درجة الثقة ────────────────────────────────────────────────────
    confidence = 0.0
    if concept_data:
        confidence += 0.4
        freq = concept_data.get("frequency", 0)
        confidence += min(freq / 100, 0.3)
    if quran_matches:
        confidence += min(len(quran_matches) / 10, 0.2)
    if root_matches:
        confidence += 0.1
    confidence = min(confidence, 1.0)

    # ── 5. مصادر المفهوم ────────────────────────────────────────────────
    sources = []
    if concept_data:
        sources = concept_data.get("sources", [])
    if quran_matches and "القرآن الكريم" not in sources:
        sources.append("القرآن الكريم")

    return {
        "query":         query,
        "concept_data":  concept_data,
        "ckg_related":   ckg_related,
        "ckg_relations": ckg_relations,
        "root_matches":  root_matches,
        "quran_matches": quran_matches,
        "sources":       sources,
        "confidence":    confidence,
        "found":         bool(concept_data or quran_matches or root_matches),
    }


# ═══════════════════════════════════════════════════════════════════════════
# دوال العرض
# ═══════════════════════════════════════════════════════════════════════════

def metric_card(value, label: str):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


def render_home():
    """الصفحة الرئيسية — إحصاءات النظام."""
    roots         = load_arabic_roots()
    ckg           = load_ckg()
    quran_index   = load_quran_index()
    graph_metrics = load_graph_metrics()
    training      = load_training_summary()
    checkpoint    = load_latest_checkpoint()
    episodic      = get_episodic_stats()

    concepts_count  = len(ckg.get("concepts", {}))
    relations_count = len(ckg.get("relations", {}))

    # عدد الجذور ذات المعنى (أكثر من 3 أحرف)
    meaningful_roots = sum(1 for k in roots if len(k) >= 3 and roots[k].get("frequency", 0) > 10)

    train_steps = training.get("train_steps", 0)

    # آخر تحديث
    saved_at = checkpoint.get("saved_at", "")
    if saved_at:
        try:
            dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
            last_update = dt.strftime("%Y-%m-%d %H:%M") + " UTC"
        except Exception:
            last_update = saved_at[:19]
    else:
        last_update = "غير محدد"

    st.markdown('<div class="section-header">📊 إحصاءات النظام المعرفي</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(f"{concepts_count:,}", "مفهوم في CKG")
    with col2: metric_card(f"{relations_count:,}", "علاقة معرفية")
    with col3: metric_card(f"{meaningful_roots:,}", "جذر عربي مكتشف")
    with col4: metric_card(f"{train_steps:,}", "خطوة تدريب")

    st.markdown("")
    col5, col6, col7, col8 = st.columns(4)
    with col5: metric_card(f"{quran_index.get('total_ayat', 6236):,}", "آية قرآنية محملة")
    with col6: metric_card(f"{quran_index.get('total_surahs', 114)}", "سورة كريمة")
    with col7: metric_card(f"{episodic.get('episodic', 0):,}", "ذكرى تجريبية")
    with col8: metric_card(last_update, "آخر تحديث")

    st.markdown("")
    st.markdown('<div class="section-header">🔍 ابحث في المعرفة</div>', unsafe_allow_html=True)
    st.markdown("أدخل مفهوماً للبحث عنه مباشرةً في قلب النظام:")

    col_s, col_b = st.columns([4, 1])
    with col_s:
        quick_q = st.text_input("", placeholder="مثال: الصبر، الجاذبية، الرحمة، العدل...",
                                key="home_search", label_visibility="collapsed")
    with col_b:
        if st.button("🔍 بحث", use_container_width=True, key="home_btn"):
            if quick_q.strip():
                st.session_state["search_query"] = quick_q.strip()
                st.session_state["active_tab"] = 1
                st.rerun()

    if quick_q.strip() and st.session_state.get("home_auto"):
        st.session_state["search_query"] = quick_q.strip()
        st.session_state["active_tab"] = 1
        st.rerun()


def render_search():
    """تبويب البحث المعرفي — قلب النظام."""
    st.markdown('<div class="section-header">🔍 البحث المعرفي</div>', unsafe_allow_html=True)
    st.markdown("ابحث عن أي مفهوم وسيظهر لك ما يعرفه النظام عنه:")

    default_q = st.session_state.get("search_query", "")
    query = st.text_input(
        "",
        value=default_q,
        placeholder="اكتب مفهوماً... مثل: الصبر، الجاذبية، التوبة، العلم",
        key="main_search",
        label_visibility="collapsed",
    )

    # أمثلة سريعة
    st.markdown("**أمثلة:**")
    ex_cols = st.columns(6)
    examples = ["الصبر", "الرحمة", "العلم", "الجاذبية", "العدل", "الإيمان"]
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"ex_{i}", use_container_width=True):
                query = ex
                st.session_state["search_query"] = ex

    st.markdown("---")

    if not query.strip():
        st.info("اكتب مفهوماً في خانة البحث أعلاه لاستكشاف قاعدة المعرفة.")
        return

    # تنفيذ البحث
    with st.spinner("🔍 جارٍ البحث في قاعدة المعرفة..."):
        result = search_knowledge(query.strip())

    if not result["found"]:
        st.warning(f"لم يُعثر على معلومات كافية عن «{query}» حتى الآن. يتعلم النظام بشكل مستمر!")
        return

    # ── عرض النتائج ──────────────────────────────────────────────────────

    # بطاقة المفهوم الرئيسية
    cdata = result["concept_data"]
    st.markdown(f"""
    <div class="concept-card">
        <div class="concept-name">💡 {result['query']}</div>
    """, unsafe_allow_html=True)

    if cdata:
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.markdown(f"**التصنيف:** {cdata.get('cluster', 'غير مصنّف')}")
        with col_b:
            freq = cdata.get("frequency", 0)
            st.markdown(f"**التكرار:** {freq:,} مرة")
        with col_c:
            strength = cdata.get("strength", 0.0)
            st.markdown(f"**قوة المفهوم:** {strength:.2%}")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── المفاهيم المرتبطة ────────────────────────────────────────────────
    related_concepts = []
    if result["ckg_related"]:
        related_concepts = result["ckg_related"]
    elif result["root_matches"]:
        related_concepts = [m[0] for m in result["root_matches"] if m[0] != query]

    if related_concepts:
        st.markdown('<div class="section-header">🔗 المفاهيم المرتبطة</div>', unsafe_allow_html=True)
        tags_html = ""
        for concept in related_concepts[:12]:
            tags_html += f'<span class="related-tag">{concept}</span>'
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── العلاقات من CKG ──────────────────────────────────────────────────
    if result["ckg_relations"]:
        st.markdown('<div class="section-header">↔️ العلاقات المعرفية</div>', unsafe_allow_html=True)
        for rel in result["ckg_relations"][:6]:
            rel_type = rel.get("type", "مرتبط")
            weight   = rel.get("weight", 0)
            target   = rel.get("target", "")
            badge_color = "badge-blue"
            st.markdown(f"""
            <div class="root-item">
                <span class="badge {badge_color}">{rel_type}</span>
                &nbsp;→&nbsp; <strong>{target}</strong>
                &nbsp;&nbsp; <small style="color:#999">قوة: {weight:.2f}</small>
            </div>
            """, unsafe_allow_html=True)

    # ── الإشارات القرآنية ────────────────────────────────────────────────
    quran_matches = result["quran_matches"]
    if quran_matches:
        st.markdown(f'<div class="section-header">📖 الإشارات القرآنية ({len(quran_matches)} آية)</div>', unsafe_allow_html=True)
        for ayah in quran_matches[:6]:
            surah = ayah.get("surah", "")
            verse = ayah.get("ayah", "")
            text  = ayah.get("text", "")
            st.markdown(f"""
            <div class="quran-verse">
                {text}
                <div class="verse-ref">سورة {surah}، الآية {verse}</div>
            </div>
            """, unsafe_allow_html=True)

        if len(quran_matches) > 6:
            with st.expander(f"عرض {len(quran_matches) - 6} آية إضافية"):
                for ayah in quran_matches[6:]:
                    surah = ayah.get("surah", "")
                    verse = ayah.get("ayah", "")
                    text  = ayah.get("text", "")
                    st.markdown(f"""
                    <div class="quran-verse">
                        {text}
                        <div class="verse-ref">سورة {surah}، الآية {verse}</div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-header">📖 الإشارات القرآنية</div>', unsafe_allow_html=True)
        st.info("لم يُعثر على آيات مباشرة لهذا المفهوم بهذه الصياغة. جرّب مرادفاً أو جذر الكلمة.")

    # ── المصادر ودرجة الثقة ──────────────────────────────────────────────
    st.markdown('<div class="section-header">📋 تفاصيل البحث</div>', unsafe_allow_html=True)
    col_src, col_conf = st.columns(2)
    with col_src:
        sources = result["sources"] or ["الجذور العربية"]
        st.markdown(f"**المصادر:** {' ، '.join(sources)}")
    with col_conf:
        conf = result["confidence"]
        bar_color = "#16a34a" if conf > 0.6 else "#f59e0b" if conf > 0.3 else "#dc2626"
        st.markdown(f"**درجة الثقة:** {conf:.0%}")
        st.progress(conf)

    # ── الجذور المرتبطة من الجذور العربية ────────────────────────────────
    root_matches = result["root_matches"]
    if root_matches:
        with st.expander("🌿 الجذور العربية المكتشفة"):
            for token, freq in root_matches[:10]:
                st.markdown(f"""
                <div class="root-item">
                    <strong>{token}</strong>
                    <span class="badge badge-green" style="float:left">تكرار: {freq:,}</span>
                </div>
                """, unsafe_allow_html=True)


def render_quran():
    """تبويب القرآن الكريم."""
    st.markdown('<div class="section-header">📖 القرآن الكريم في النظام</div>', unsafe_allow_html=True)

    quran_index = load_quran_index()
    ayat        = load_all_quran_ayat()
    roots       = load_arabic_roots()

    # إحصاءات
    col1, col2, col3 = st.columns(3)
    with col1: metric_card(f"{quran_index.get('total_ayat', len(ayat)):,}", "آية محملة")
    with col2: metric_card(f"{quran_index.get('total_surahs', 114)}", "سورة")
    with col3: metric_card(f"{len(roots):,}", "مفهوم مستخرج")

    st.markdown("")

    # أكثر المفاهيم تكراراً
    st.markdown('<div class="section-header">🔝 أكثر المفاهيم تكراراً في القرآن</div>', unsafe_allow_html=True)

    # فلترة الجذور ذات المعنى
    filtered = {k: v for k, v in roots.items()
                if len(normalize_arabic(k)) >= 3
                and v.get("frequency", 0) > 50
                and normalize_arabic(k) not in {
                    "من", "في", "على", "إلى", "عن", "مع", "الا", "ومن",
                    "وان", "بهۦ", "بما", "وما", "الذ", "وقا", "وله"
                }}

    top_concepts = sorted(filtered.items(), key=lambda x: x[1].get("frequency", 0), reverse=True)[:20]

    if top_concepts:
        # رسم بياني
        try:
            import plotly.graph_objects as go
            names = [v.get("top_token", k) for k, v in top_concepts[:15]]
            freqs = [v.get("frequency", 0) for _, v in top_concepts[:15]]

            fig = go.Figure(go.Bar(
                x=freqs,
                y=names,
                orientation='h',
                marker_color='#3b82f6',
                text=freqs,
                textposition='outside',
            ))
            fig.update_layout(
                height=450,
                margin=dict(l=20, r=60, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(autorange="reversed"),
                xaxis_title="التكرار",
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            for k, v in top_concepts[:10]:
                token = v.get("top_token", k)
                freq  = v.get("frequency", 0)
                st.markdown(f"**{token}**: {freq:,} مرة")
    else:
        st.info("لم تُكتشف مفاهيم بعد. يحتاج النظام إلى تدريب إضافي.")

    # بحث داخل القرآن
    st.markdown('<div class="section-header">🔍 البحث في آيات القرآن</div>', unsafe_allow_html=True)
    quran_q = st.text_input("", placeholder="ابحث عن كلمة أو مفهوم...", key="quran_search",
                             label_visibility="collapsed")
    if quran_q.strip():
        matches = search_quran_for_concept(quran_q.strip(), ayat, max_results=20)
        if matches:
            st.success(f"وُجد {len(matches)} آية تحتوي على «{quran_q}»")
            for ayah in matches:
                surah = ayah.get("surah", "")
                verse = ayah.get("ayah", "")
                text  = ayah.get("text", "")
                st.markdown(f"""
                <div class="quran-verse">
                    {text}
                    <div class="verse-ref">سورة {surah}، الآية {verse}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning(f"لم يُعثر على «{quran_q}» في الآيات المحملة.")


def render_qa():
    """تبويب الأسئلة والأجوبة القرآني — يعتمد على CKG والآيات فقط."""
    st.markdown('<div class="section-header">❓ الأسئلة والأجوبة القرآني</div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color:#999">اسأل سؤالاً بالعربية، وسيحلل النظام السؤال '
        'ويبحث في 173 مفهوماً و2149 علاقة دلالية و6236 آية للإجابة.</p>',
        unsafe_allow_html=True,
    )

    # ── أمثلة جاهزة ──
    st.markdown("**أمثلة:**")
    examples = [
        "من هو محمد ﷺ؟",
        "ما علاقة الصبر بالإيمان؟",
        "ماذا يقول القرآن عن العدل؟",
        "ما قصة يوسف؟",
    ]
    ex_cols = st.columns(len(examples))
    chosen_example = None
    for i, ex in enumerate(examples):
        with ex_cols[i]:
            if st.button(ex, key=f"qa_example_{i}", use_container_width=True):
                chosen_example = ex

    default_q = chosen_example or st.session_state.get("qa_question", "")
    question = st.text_input(
        "اكتب سؤالك هنا:",
        value=default_q,
        key="qa_input",
        placeholder="مثال: ما علاقة الصبر بالإيمان؟",
    )
    st.session_state["qa_question"] = question

    ask = st.button("🔍 اسأل", type="primary")

    if not (ask or chosen_example) or not question.strip():
        return

    ckg  = load_ckg()
    ayat = load_all_quran_ayat()

    if not ckg.get("concepts"):
        st.error("الذاكرة الدلالية (CKG) فارغة — لا يمكن الإجابة على الأسئلة حالياً.")
        return

    with st.spinner("يتم تحليل السؤال والبحث في قاعدة المعرفة..."):
        entities = load_entities()
        result = answer_question(question, ckg, ayat, entities=entities)

    # ── حفظ الحلقة في الذاكرة التجريبية ──
    db_path = MEMORY_DIR / "episodic.db"
    try:
        store_episode(db_path, question, result)
    except Exception:
        pass

    # ── أسئلة سابقة مشابهة ──
    try:
        similar = find_similar_episodes(db_path, question, threshold=0.4, top_k=3)
    except Exception:
        similar = []

    st.markdown("---")

    if similar:
        st.markdown('<div class="section-header">🕘 أسئلة سابقة مشابهة</div>', unsafe_allow_html=True)
        for s in similar:
            if normalize_arabic(s["question"]) == normalize_arabic(question):
                continue
            st.markdown(f"""
            <div class="root-item">
                <strong>{s['question']}</strong>
                <span class="badge badge-blue">تشابه: {s['similarity']:.0%}</span>
                <span class="badge badge-amber">ثقة: {s['confidence']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("")

    # ── ملخص الإجابة ──
    entity_info = result.get("entity")
    if entity_info:
        st.markdown(
            f'<div class="section-header">📝 ملخص الإجابة '
            f'<span class="badge badge-purple">كيان: {entity_info["name"]} ({entity_info["type"]})</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="section-header">📝 ملخص الإجابة</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="root-item" style="font-size:1.05rem; line-height:1.8">
        {result['summary']}
    </div>
    """, unsafe_allow_html=True)

    # ── درجة الثقة ──
    confidence = result.get("confidence", 0.0)
    st.markdown("")
    st.markdown(f"**درجة الثقة:** {confidence:.0%}")
    st.progress(confidence)

    if not result["primary_concepts"]:
        st.info("لم يتم العثور على مفاهيم مرتبطة بهذا السؤال في قاعدة المعرفة الحالية.")
        return

    # ── المفاهيم الأساسية ──
    st.markdown("")
    st.markdown('<div class="section-header">🧩 المفاهيم المستخرجة من السؤال</div>', unsafe_allow_html=True)
    for c in result["primary_concepts"]:
        if entity_info:
            # في إجابات الكيانات، أرقام "تكرار/تطابق" التقنية لا تضيف
            # قيمة للمستخدم — نعرض فقط الاسم والمجموعة المعرفية
            st.markdown(f"""
            <div class="root-item">
                <strong>{c['name']}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{c['cluster']}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="root-item">
                <strong>{c['name']}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{c['cluster']}</span>
                <span class="badge badge-blue">تكرار في القرآن: {c['frequency']}</span>
                <span class="badge badge-amber">درجة التطابق: {c['match']:.0%}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── المفاهيم المرتبطة (من العلاقات) ──
    related = result.get("related_concepts", [])
    if related:
        st.markdown("")
        st.markdown('<div class="section-header">🔗 مفاهيم مرتبطة (من الذاكرة الدلالية)</div>', unsafe_allow_html=True)
        rel_type_labels = {
            "co_occurrence":     "تزامن في الآية",
            "semantic":          "علاقة دلالية",
            "thematic_cluster":  "تجمّع موضوعي",
            "root_link":         "ربط بجذر",
            "narrative_sequence": "تسلسل سردي",
            "episodic_rule":     "قاعدة من الذاكرة التجريبية",
            "entity_attribute":  "صفة الكيان",
        }
        for r in related[:6]:
            rtype = rel_type_labels.get(r["relation_type"], r["relation_type"])
            st.markdown(f"""
            <div class="root-item">
                <strong>{r['concept']}</strong>
                <span class="badge badge-blue">نوع العلاقة: {rtype}</span>
                <span class="badge badge-amber">وزن العلاقة: {r['weight']:.2f}</span>
            </div>
            """, unsafe_allow_html=True)

    # ── الآيات الداعمة ──
    verses = result.get("verses", [])
    st.markdown("")
    st.markdown(f'<div class="section-header">📖 الآيات الداعمة ({len(verses)})</div>', unsafe_allow_html=True)
    if verses:
        for v in verses:
            st.markdown(f"""
            <div class="quran-verse">
                {v['text']}
                <div class="verse-ref">سورة {v['surah']}، الآية {v['ayah']} — مفهوم: {v['concept']}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("لم يتم العثور على آيات داعمة مباشرة لهذا السؤال.")


def render_training():
    """تبويب التدريب."""
    st.markdown('<div class="section-header">🎓 حالة التدريب</div>', unsafe_allow_html=True)

    training   = load_training_summary()
    checkpoint = load_latest_checkpoint()
    ckg        = load_ckg()

    train_steps = training.get("train_steps", 0)
    last_loss   = training.get("last_loss", 0.0)
    total_params= training.get("total_parameters", 0)
    ckg_size    = len(ckg.get("concepts", {}))

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(f"{train_steps:,}", "خطوات التدريب")
    with col2: metric_card(f"{last_loss:.2e}", "آخر خسارة (Loss)")
    with col3: metric_card(f"{total_params:,}", "معامل في الشبكة")
    with col4: metric_card(f"{ckg_size:,}", "مفهوم في CKG")

    st.markdown("")

    # معلومات الـ Checkpoint
    saved_at = checkpoint.get("saved_at", "")
    if saved_at:
        st.markdown('<div class="section-header">💾 آخر نقطة حفظ</div>', unsafe_allow_html=True)
        try:
            dt = datetime.fromisoformat(saved_at.replace("Z", "+00:00"))
            st.info(f"تم الحفظ في: **{dt.strftime('%Y-%m-%d الساعة %H:%M:%S')} UTC**")
        except Exception:
            st.info(f"تم الحفظ في: {saved_at}")

        state = checkpoint.get("state", {})
        if state:
            st.markdown('<div class="section-header">🧠 محتوى نقطة الحفظ</div>', unsafe_allow_html=True)
            for module_name in state.keys():
                module_labels = {
                    "neural_weights":  "الأوزان العصبية ✅",
                    "deep_network":    "الشبكة العميقة ✅",
                    "dynamic_layer":   "الطبقة الديناميكية ✅",
                    "episodic_memory": "الذاكرة التجريبية ✅",
                    "world_model":     "نموذج العالم ✅",
                    "system_dna":      "الحمض النووي للنظام ✅",
                    "self_awareness":  "الوعي الذاتي ✅",
                    "meta":            "البيانات الوصفية ✅",
                }
                label = module_labels.get(module_name, f"{module_name} ✅")
                st.markdown(f'<span class="badge badge-green">{label}</span>&nbsp;', unsafe_allow_html=True)

    # معلومات التدريب التفصيلية
    if training:
        st.markdown("")
        st.markdown('<div class="section-header">📐 بنية الشبكة العصبية</div>', unsafe_allow_html=True)
        arch = training.get("architecture", "")
        if arch:
            st.code(arch, language=None)

        avg_loss = training.get("avg_recent_loss", 0)
        lr       = training.get("learning_rate", 0)
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"**متوسط الخسارة الأخيرة:** `{avg_loss:.2e}`")
        with col_b:
            st.markdown(f"**معدل التعلم:** `{lr}`")


def render_memory():
    """تبويب الذاكرة."""
    st.markdown('<div class="section-header">🧠 حالة الذاكرة</div>', unsafe_allow_html=True)

    episodic = get_episodic_stats()
    ckg      = load_ckg()
    roots    = load_arabic_roots()

    concepts_count  = len(ckg.get("concepts", {}))
    relations_count = len(ckg.get("relations", {}))

    col1, col2, col3, col4 = st.columns(4)
    with col1: metric_card(episodic.get("episodic", 0), "ذاكرة تجريبية")
    with col2: metric_card(concepts_count, "ذاكرة دلالية (مفاهيم)")
    with col3: metric_card(relations_count, "علاقات مستنتجة")
    with col4: metric_card(len(roots), "جذر عربي مفهرس")

    st.markdown("")
    st.markdown('<div class="section-header">📁 تفاصيل الذاكرة الدلالية (CKG)</div>', unsafe_allow_html=True)

    concepts_db = ckg.get("concepts", {})
    if concepts_db:
        # عرض أقوى المفاهيم
        sorted_concepts = sorted(
            concepts_db.items(),
            key=lambda x: x[1].get("frequency", 0),
            reverse=True
        )[:15]

        for cname, cdata in sorted_concepts:
            freq     = cdata.get("frequency", 0)
            cluster  = cdata.get("cluster", "غير مصنّف")
            strength = cdata.get("strength", 0.0)
            sources  = cdata.get("sources", [])
            st.markdown(f"""
            <div class="root-item">
                <strong>{cname}</strong>
                <span class="badge badge-purple" style="margin-right:8px">{cluster}</span>
                <span class="badge badge-blue">تكرار: {freq}</span>
                <span class="badge badge-amber">قوة: {strength:.2f}</span>
                <br><small style="color:#888">المصادر: {', '.join(sources[:3]) if sources else 'غير محددة'}</small>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("الذاكرة الدلالية (CKG) فارغة حالياً. قم بتشغيل دورة تدريب في Colab لملئها.")

    # ── أنواع العلاقات في CKG ────────────────────────────────────────────
    relations_db = ckg.get("relations", {})
    if relations_db:
        st.markdown("")
        st.markdown('<div class="section-header">🔗 أنواع العلاقات في الذاكرة الدلالية</div>', unsafe_allow_html=True)

        rel_type_counter = Counter(r.get("relation_type", "غير محدد") for r in relations_db.values())
        type_labels = {
            "co_occurrence":    "تزامن في الآية",
            "semantic":         "علاقة دلالية (نفس المجموعة)",
            "thematic_cluster": "تجمّع موضوعي (تشارك سور)",
            "root_link":        "ربط بجذر عربي",
            "narrative_sequence": "تسلسل سردي (قصص الأنبياء)",
            "episodic_rule":    "قاعدة من الذاكرة التجريبية",
        }
        badges = " ".join(
            f'<span class="badge badge-blue" style="margin:3px">{type_labels.get(t, t)}: {n}</span>'
            for t, n in rel_type_counter.most_common()
        )
        st.markdown(badges, unsafe_allow_html=True)

    # ── ملامح السور (Surah Thematic Profiles) ───────────────────────────
    surah_profiles = ckg.get("surah_profiles", {})
    if surah_profiles:
        st.markdown("")
        st.markdown('<div class="section-header">📖 ملامح السور الموضوعية</div>', unsafe_allow_html=True)
        st.markdown(
            f'<p style="color:#999">تم بناء ملامح موضوعية لـ {len(surah_profiles)} سورة '
            f'بناءً على المفاهيم الأكثر ظهوراً في كل سورة.</p>',
            unsafe_allow_html=True,
        )

        surah_options = sorted(surah_profiles.keys(), key=lambda x: int(x))
        chosen_surah = st.selectbox(
            "اختر سورة لعرض ملامحها:",
            options=surah_options,
            format_func=lambda s: f"سورة {s}",
            key="surah_profile_select",
        )
        if chosen_surah:
            profile = surah_profiles.get(chosen_surah, [])
            badges = " ".join(
                f'<span class="badge badge-purple" style="margin:3px">{p["concept"]} ({p["weight"]})</span>'
                for p in profile
            )
            st.markdown(badges, unsafe_allow_html=True)

    # حالة قاعدة البيانات
    st.markdown("")
    st.markdown('<div class="section-header">💾 حالة قواعد البيانات</div>', unsafe_allow_html=True)
    db_path = MEMORY_DIR / "episodic.db"
    if db_path.exists():
        size_kb = db_path.stat().st_size / 1024
        st.markdown(f'<span class="health-ok">✅ قاعدة الذاكرة التجريبية: متصلة ({size_kb:.1f} KB)</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="health-err">❌ قاعدة الذاكرة التجريبية: غير موجودة</span>', unsafe_allow_html=True)

    # ── إحصاءات الذاكرة التجريبية للأسئلة والأجوبة ──────────────────────
    st.markdown("")
    st.markdown('<div class="section-header">📊 إحصاءات ذاكرة الأسئلة والأجوبة</div>', unsafe_allow_html=True)

    try:
        qa_stats = get_memory_stats(db_path)
    except Exception:
        qa_stats = {"total_episodes": 0, "common_concepts": [], "recent_episodes": [], "avg_confidence": 0.0}

    qcol1, qcol2 = st.columns(2)
    with qcol1: metric_card(qa_stats["total_episodes"], "إجمالي الحلقات المخزّنة")
    with qcol2: metric_card(f"{qa_stats['avg_confidence']:.0%}", "متوسط درجة الثقة")

    if qa_stats["total_episodes"] > 0:
        # أكثر المفاهيم تكراراً في الأسئلة
        st.markdown("**أكثر المفاهيم ظهوراً في الأسئلة:**")
        if qa_stats["common_concepts"]:
            badges = " ".join(
                f'<span class="badge badge-blue" style="margin:2px">{c} ({n})</span>'
                for c, n in qa_stats["common_concepts"][:8]
            )
            st.markdown(badges, unsafe_allow_html=True)

        # أحدث الحلقات
        st.markdown("")
        st.markdown("**أحدث الأسئلة:**")
        for ep in qa_stats["recent_episodes"][:5]:
            ts = ep.get("timestamp", "")[:19].replace("T", " ")
            st.markdown(f"""
            <div class="root-item">
                <strong>{ep['question']}</strong>
                <span class="badge badge-amber">ثقة: {ep['confidence']:.0%}</span>
                <br><small style="color:#888">{ts} UTC</small>
            </div>
            """, unsafe_allow_html=True)

        # ── التوحيد (Consolidation) ──
        st.markdown("")
        st.markdown('<div class="section-header">🧬 توحيد الذاكرة (Consolidation)</div>', unsafe_allow_html=True)
        st.markdown(
            '<p style="color:#999">يستخرج هذا الإجراء أزواج المفاهيم المتكررة في الأسئلة السابقة، '
            'ويولّد منها قواعد دلالية، ويضيفها كعلاقات جديدة في الذاكرة الدلالية (CKG) '
            'دون حذف أو تعديل أي علاقة موجودة.</p>',
            unsafe_allow_html=True,
        )

        if st.button("🧬 تشغيل التوحيد الآن", key="consolidate_btn"):
            ckg_path = KNOWLEDGE_DIR / "cognitive_graph.json"
            with st.spinner("يتم تحليل الحلقات واستخراج القواعد الدلالية..."):
                ckg_full = load_json(ckg_path) or {"concepts": {}, "relations": {}}
                cons_result = consolidate_memory(db_path, ckg_full, ckg_path, min_co_occurrence=2)
            st.success(
                f"تم التحليل: {cons_result['pairs_analyzed']} زوج مفاهيم، "
                f"{cons_result['new_rules']} قاعدة جديدة، "
                f"{cons_result['new_relations']} علاقة جديدة في CKG."
            )
            load_json.clear()
            load_ckg.clear()

        rules = get_semantic_rules(db_path, limit=10)
        if rules:
            st.markdown("**القواعد الدلالية المستخرجة:**")
            for r in rules:
                st.markdown(f"""
                <div class="root-item">
                    {r['rule_text']}
                    <span class="badge badge-purple">ثقة: {r['confidence']:.0%}</span>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("لا توجد أسئلة محفوظة بعد. استخدم تبويب «الأسئلة والأجوبة» لبدء بناء الذاكرة التجريبية.")


def render_health():
    """تبويب صحة النظام."""
    st.markdown('<div class="section-header">🏥 صحة النظام</div>', unsafe_allow_html=True)

    checks = []

    # ── 1. الأوزان محفوظة؟
    weights_path = CHECKPOINTS_DIR / "neural_weights.npy"
    if weights_path.exists():
        size_kb = weights_path.stat().st_size / 1024
        checks.append(("✅", "الأوزان العصبية", f"محفوظة ({size_kb:.1f} KB)", True))
    else:
        checks.append(("❌", "الأوزان العصبية", "ملف الأوزان غير موجود", False))

    # ── 2. CKG محفوظ؟
    ckg_path = KNOWLEDGE_DIR / "cognitive_graph.json"
    if ckg_path.exists() and ckg_path.stat().st_size > 10:
        ckg = load_ckg()
        n_concepts = len(ckg.get("concepts", {}))
        checks.append(("✅", "قاعدة المعرفة CKG", f"موجودة ({n_concepts} مفهوم)", True))
    else:
        checks.append(("⚠️", "قاعدة المعرفة CKG", "فارغة أو غير موجودة", False))

    # ── 3. قاعدة البيانات
    db_path = MEMORY_DIR / "episodic.db"
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
            conn.close()
            checks.append(("✅", "قاعدة الذاكرة (SQLite)", f"متصلة ({count} سجل)", True))
        except Exception as e:
            checks.append(("❌", "قاعدة الذاكرة (SQLite)", f"خطأ: {e}", False))
    else:
        checks.append(("❌", "قاعدة الذاكرة (SQLite)", "غير موجودة", False))

    # ── 4. القرآن الكريم
    chunks = list(KNOWLEDGE_DIR.glob("quran_chunk_*.json"))
    if len(chunks) >= 60:
        checks.append(("✅", "بيانات القرآن الكريم", f"{len(chunks)} chunk محمّل (6,236 آية)", True))
    else:
        checks.append(("⚠️", "بيانات القرآن الكريم", f"وُجد {len(chunks)} chunk فقط", False))

    # ── 5. الجذور العربية
    roots = load_arabic_roots()
    if len(roots) > 100:
        checks.append(("✅", "فهرس الجذور العربية", f"{len(roots)} جذر مكتشف", True))
    else:
        checks.append(("⚠️", "فهرس الجذور العربية", f"{len(roots)} جذر فقط", False))

    # ── 6. نقطة حفظ حديثة
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("brain_checkpoint_*.json"), reverse=True)
    if checkpoint_files:
        latest = load_latest_checkpoint()
        saved_at = latest.get("saved_at", "")
        checks.append(("✅", "نقطة الحفظ الأخيرة (Checkpoint)", saved_at[:19] if saved_at else "موجودة", True))
    else:
        checks.append(("❌", "نقطة الحفظ الأخيرة (Checkpoint)", "لا توجد نقطة حفظ", False))

    # ── 7. التدريب
    training = load_training_summary()
    if training.get("train_steps", 0) > 0:
        checks.append(("✅", "حالة التدريب", f"{training['train_steps']:,} خطوة مكتملة", True))
    else:
        checks.append(("⚠️", "حالة التدريب", "لم يكتمل تدريب بعد", False))

    # عرض النتائج
    all_ok = sum(1 for c in checks if c[3])
    total  = len(checks)

    if all_ok == total:
        st.success(f"✅ النظام يعمل بكفاءة كاملة ({all_ok}/{total})")
    elif all_ok >= total * 0.7:
        st.warning(f"⚠️ النظام يعمل جزئياً ({all_ok}/{total})")
    else:
        st.error(f"❌ بعض مكونات النظام تحتاج انتباهاً ({all_ok}/{total})")

    st.markdown("")
    for icon, name, detail, ok in checks:
        css_class = "health-ok" if ok else ("health-err" if icon == "❌" else "")
        st.markdown(f"""
        <div style="padding: 0.6rem 1rem; margin: 0.3rem 0; background: {'#f0fdf4' if ok else '#fef2f2'};
                    border-radius: 8px; border: 1px solid {'#bbf7d0' if ok else '#fecaca'};">
            <span style="font-size:1.2rem">{icon}</span>
            &nbsp;<strong>{name}</strong>
            &nbsp;&nbsp;<small style="color:#666">{detail}</small>
        </div>
        """, unsafe_allow_html=True)

    # أزرار الإجراءات
    st.markdown("")
    st.markdown('<div class="section-header">⚙️ إجراءات</div>', unsafe_allow_html=True)
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        if st.button("🔄 تحديث الإحصاءات", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    with col_r2:
        st.markdown("""
        <div style="background:#f8faff; border:1px solid #c7d2fe; border-radius:8px; padding:0.6rem 1rem; font-size:0.85rem; direction:rtl">
            لتشغيل دورة تدريب، افتح Google Colab وشغّل <code>train_simulate.py</code>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# التطبيق الرئيسي
# ═══════════════════════════════════════════════════════════════════════════

def main():
    # ── العنوان ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="main-title">🧠 النظام المعرفي العربي</div>
    <div class="subtitle">Neural Service Mesh · ذكاء اصطناعي عربي متخصص بالمعرفة الإسلامية</div>
    """, unsafe_allow_html=True)

    # ── التبويبات ─────────────────────────────────────────────────────────
    tabs = st.tabs(["🏠 الرئيسية", "🔍 البحث المعرفي", "📖 القرآن الكريم",
                    "❓ الأسئلة والأجوبة", "🎓 التدريب", "🧠 الذاكرة", "🏥 صحة النظام"])

    with tabs[0]: render_home()
    with tabs[1]: render_search()
    with tabs[2]: render_quran()
    with tabs[3]: render_qa()
    with tabs[4]: render_training()
    with tabs[5]: render_memory()
    with tabs[6]: render_health()

    # ── تذييل الصفحة ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; color:#999; font-size:0.8rem; direction:rtl">
        Neural Service Mesh · نظام معرفي عربي ذاتي التعلم · مبني بـ Python & Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
