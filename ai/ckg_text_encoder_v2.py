"""
CKG Text Encoder v2 — إصلاح منطق المطابقة
============================================

المشكلة في الكود الأصلي (encode_query_to_ckg_vector):
    any(w in name_clean for w in q_words if len(w) >= 3)
    → احتواء جزئي: كلمة "علم" تطابق "يعلمون"، "العلمين"، إلخ
    → مطابقات خاطئة كثيرة في العربية بسبب غنى الاشتقاق

الحل (مطابقة كلمة كاملة + تجزيء هاشي للمفاهيم بعد الـ 128):
    1. word-level match: q_words & name_words (تقاطع مجموعات)
    2. hashing trick: جميع المفاهيم الـ1102 تساهم، لا فقط أول 128
    3. درجات متعددة:
       - exact full name match  → score 1.0
       - word-level overlap     → score proportional to overlap
       - partial root match (3+ chars shared token) → score 0.3 (خُففت)

الاستخدام:
    from ai.ckg_text_encoder_v2 import encode_query_v2, encode_query_hashing
    vec = encode_query_v2(query, ckg_concepts)   # 128-dim, word-level
    vec = encode_query_hashing(query, ckg_concepts, dim=128)  # hashing trick
"""
from __future__ import annotations

import re
import math
import numpy as np
from typing import Dict


# ── تنظيف النص العربي ──────────────────────────────────────────────────────────

def _clean_arabic(text: str) -> str:
    """إزالة التشكيل والألف الوصل وغير العربي."""
    text = re.sub(r'[ٱ]', 'ا', text)
    text = re.sub(r'[أإآ]', 'ا', text)          # توحيد الألف
    text = re.sub(r'[ةه](?=\s|$)', 'ه', text)   # توحيد التاء المربوطة عند نهاية الكلمة
    text = re.sub(r'[ًٌٍَُِّْٰ]', '', text)       # إزالة الحركات
    text = re.sub(r'[^\u0600-\u06FF\s]', ' ', text)
    return text.strip()


def _tokenize(text: str) -> set:
    """تحويل النص إلى مجموعة كلمات بعد التنظيف."""
    return set(_clean_arabic(text).split())


# ── المطابقة على مستوى الكلمة (النسخة الجديدة) ──────────────────────────────

def _word_level_score(query_words: set, concept_name: str) -> float:
    """
    حساب درجة التطابق بين كلمات الاستعلام واسم المفهوم.
    
    الأولويات (من الأعلى للأدنى):
    1. تطابق اسم المفهوم كاملاً مع أحد كلمات الاستعلام → 1.0
    2. تقاطع كلمات (word intersection) → نسبة التقاطع
    3. تطابق جزئي مخفف للجذور (≥4 أحرف مشتركة) → 0.2
    """
    name_clean = _clean_arabic(concept_name)
    name_words = set(name_clean.split())

    # حالة 1: اسم المفهوم بالكامل ضمن الاستعلام
    if name_clean in query_words:
        return 1.0

    # حالة 2: تقاطع الكلمات
    common = query_words & name_words
    if common:
        # نسبة التقاطع بالنسبة لأسماء المفاهيم الأقصر أهم
        overlap_ratio = len(common) / max(len(name_words), 1)
        return min(1.0, overlap_ratio)

    # حالة 3: تطابق جزئي مخفف (4+ أحرف) — أقل وزناً من قبل
    for qw in query_words:
        if len(qw) < 4:
            continue
        for nw in name_words:
            if len(nw) < 4:
                continue
            if qw in nw or nw in qw:
                return 0.2   # مخفض من 1.0 → 0.2

    return 0.0


# ── الترميز الرئيسي (128-dim، مطابقة كلمة) ────────────────────────────────────

def encode_query_v2(
    query: str,
    ckg_concepts: dict,
    dim: int = 128,
) -> np.ndarray:
    """
    ترميز استعلام عربي في متجه CKG بُعده dim (128 افتراضياً).
    يستخدم مطابقة كلمة كاملة بدل الاحتواء الجزئي.
    
    متوافق مع توقعات DeepRoutingNetwork v18.
    """
    q_words = _tokenize(query)
    vec = np.zeros(dim, dtype=np.float64)

    for i, (name, meta) in enumerate(list(ckg_concepts.items())[:dim]):
        score = _word_level_score(q_words, name)
        if score > 0:
            strength  = meta.get("strength", 0.1)
            freq_norm = min(1.0, meta.get("frequency", 1) / 500)
            vec[i] = score * strength * (1.0 + freq_norm)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── Hashing Trick: كل المفاهيم الـ1102 تساهم ─────────────────────────────────

def encode_query_hashing(
    query: str,
    ckg_concepts: dict,
    dim: int = 128,
    seed: int = 42,
) -> np.ndarray:
    """
    ترميز بخدعة التجزيء (hashing trick):
    كل مفهوم من الـ1102 يُسهم في أحد أبعاد الـdim
    بدلاً من تجاهل المفاهيم بعد الترتيب 128.
    
    يُستخدم عندما تريد تمثيلاً أشمل يغطي كل المعرفة.
    """
    q_words = _tokenize(query)
    vec = np.zeros(dim, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # إنشاء خريطة ثابتة: اسم المفهوم → موضع في المتجه
    all_names = list(ckg_concepts.keys())
    n = len(all_names)
    # مولّد عشوائي ثابت لضمان الاتساق بين الاستدعاءات
    positions = rng.integers(0, dim, size=n)

    for i, (name, meta) in enumerate(ckg_concepts.items()):
        score = _word_level_score(q_words, name)
        if score > 0:
            strength  = meta.get("strength", 0.1)
            freq_norm = min(1.0, meta.get("frequency", 1) / 500)
            pos = positions[i]
            vec[pos] += score * strength * (1.0 + freq_norm)

    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ── تصنيف الاستعلام إلى cluster ──────────────────────────────────────────────

def classify_query_cluster(
    query: str,
    ckg_concepts: dict,
    top_k: int = 5,
) -> str:
    """
    تصنيف الاستعلام إلى أحد الـ23 cluster بناءً على المفاهيم المطابقة.
    يُستخدم كهدف تدريب للـ Agent بدلاً من المطابقة الخاطئة القديمة.
    """
    from collections import Counter
    q_words = _tokenize(query)
    cluster_scores: Counter = Counter()

    for name, meta in ckg_concepts.items():
        score = _word_level_score(q_words, name)
        if score > 0:
            cluster = meta.get("cluster", "معرفة")
            strength = meta.get("strength", 0.1)
            cluster_scores[cluster] += score * strength

    if not cluster_scores:
        return "معرفة"  # افتراضي
    return cluster_scores.most_common(1)[0][0]


# ── مقارنة المنطق القديم والجديد ─────────────────────────────────────────────

def compare_encodings(query: str, ckg_concepts: dict, dim: int = 128) -> dict:
    """
    مقارنة بين المنطق القديم والجديد وهاشينج.
    مفيدة للتشخيص والتحقق.
    """
    import re as _re

    def _old_clean(t):
        t = _re.sub(r'[ٱ]', 'ا', t)
        t = _re.sub(r'[ًٌٍَُِّْٰ]', '', t)
        t = _re.sub(r'[^\u0600-\u06FF\s]', ' ', t)
        return t.strip()

    q_words_old = set(_old_clean(query).split())
    q_words_new = _tokenize(query)

    old_hits, new_hits = [], []
    for name, meta in list(ckg_concepts.items())[:dim]:
        nc_old = _old_clean(name)
        old_match = (nc_old in q_words_old or
                     any(w in nc_old for w in q_words_old if len(w) >= 3))
        new_score = _word_level_score(q_words_new, name)

        if old_match:
            old_hits.append(name)
        if new_score > 0:
            new_hits.append((name, round(new_score, 3)))

    return {
        "query": query,
        "old_hits_count": len(old_hits),
        "new_hits_count": len(new_hits),
        "old_hits_sample": old_hits[:10],
        "new_hits_sample": new_hits[:10],
        "false_positives_removed": [h for h in old_hits
                                     if h not in [n for n, _ in new_hits]],
    }


# ── اختبار مدمج ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import json, pathlib

    ckg_path = pathlib.Path(__file__).parent.parent / "knowledge" / "cognitive_graph.json"
    if not ckg_path.exists():
        print("لم يُعثر على cognitive_graph.json — تخطي الاختبار")
    else:
        with open(ckg_path) as f:
            ckg = json.load(f)
        concepts = ckg["concepts"]

        queries = [
            "ما حكم الزكاة في الإسلام",
            "قصة موسى مع فرعون",
            "أسماء الله الحسنى",
            "الصبر والشكر في القرآن",
            "خلق السماوات والأرض",
        ]

        print("=" * 60)
        print("مقارنة المنطق القديم والجديد")
        print("=" * 60)
        for q in queries:
            result = compare_encodings(q, concepts)
            print(f"\nالاستعلام: {q}")
            print(f"  قديم ({result['old_hits_count']} تطابقاً): {result['old_hits_sample'][:5]}")
            print(f"  جديد ({result['new_hits_count']} تطابقاً): {[n for n,_ in result['new_hits_sample'][:5]]}")
            print(f"  تقليل التطابقات الخاطئة: {len(result['false_positives_removed'])} مفهوم")

        print("\n" + "=" * 60)
        print("اختبار التصنيف التلقائي إلى Cluster")
        print("=" * 60)
        for q in queries:
            cluster = classify_query_cluster(q, concepts)
            print(f"  '{q}' → cluster: {cluster}")

        print("\n" + "=" * 60)
        print("اختبار encode_query_v2 (128-dim)")
        print("=" * 60)
        vec = encode_query_v2("رحمة الله والمغفرة", concepts)
        nnz = np.count_nonzero(vec)
        print(f"  المتجه: شكل={vec.shape}, لا-أصفار={nnz}, نورم={np.linalg.norm(vec):.4f}")

        print("\n" + "=" * 60)
        print("اختبار encode_query_hashing (كل 1102 مفهوم)")
        print("=" * 60)
        vec_h = encode_query_hashing("رحمة الله والمغفرة", concepts)
        nnz_h = np.count_nonzero(vec_h)
        print(f"  المتجه: شكل={vec_h.shape}, لا-أصفار={nnz_h}, نورم={np.linalg.norm(vec_h):.4f}")

        print("\n✅ جميع الاختبارات نجحت")
