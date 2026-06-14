"""
qa_engine.py
============
محرك الأسئلة والأجوبة القرآني — Quran Knowledge Q&A Engine
يستخدم فقط:
  - 6236 آية قرآنية
  - 173 مفهوم في الـ CKG
  - 2149 علاقة دلالية
  - 633 جذر عربي مفهرس

لا يضيف طبقات عصبية جديدة ولا مصادر خارجية — يعمل فوق البنية الحالية فقط.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# تطبيع النص العربي (نفس منطق streamlit_app.py)
# ═══════════════════════════════════════════════════════════════════════════
_TASHKEEL  = re.compile(r'[\u064B-\u065F\u0670\u0640]')
_ALEF      = re.compile(r'[أإآٱ]')
_BOM       = re.compile(r'\ufeff')
_SPACES    = re.compile(r'\s+')


def normalize_arabic(text: str) -> str:
    text = _TASHKEEL.sub('', text)
    text = _ALEF.sub('ا', text)
    text = _BOM.sub('', text)
    text = _SPACES.sub(' ', text)
    return text.strip()


# ═══════════════════════════════════════════════════════════════════════════
# كلمات أداة / وقف عربية — تُستثنى من استخراج المفاهيم من السؤال
# ═══════════════════════════════════════════════════════════════════════════
ARABIC_STOPWORDS = {
    "ما", "ماذا", "من", "هل", "كيف", "لماذا", "متى", "اين", "أين",
    "في", "على", "عن", "الى", "إلى", "مع", "هو", "هي", "هم", "نحن",
    "انت", "أنت", "انتم", "أنتم", "كان", "يكون", "قال", "يقول",
    "الذي", "التي", "الذين", "هذا", "هذه", "ذلك", "تلك",
    "لا", "لم", "لن", "قد", "كل", "بل", "أم", "او", "أو", "ثم",
    "اذا", "إذا", "حتى", "كما", "لكن", "وإن", "وان", "بين",
    "علاقة", "علاقه", "يقول", "تقول", "نص", "آية", "ايه", "ايات", "آيات",
    "القران", "القرآن", "الكريم", "حول", "بخصوص", "بشأن", "بشان",
    "موضوع", "معنى", "تفسير", "شرح", "وضح", "اشرح", "بين", "وضّح",
}

# اختصار: كلمات سؤال شائعة + حروف عطف نزيلها من بدايات/نهايات الكلمات
PREFIX_STRIP = ["وال", "بال", "فال", "كال", "لل", "ال", "و", "ف", "ب", "ل", "ك"]

# مفاهيم "إطارية" تُستخدم غالباً في صياغة السؤال نفسه ولا تمثل موضوعه
# (مثل: "ماذا يقول القرآن عن X؟") — تُخفَّض أولويتها إذا وُجد مفهوم آخر معها
META_CONCEPTS = {"قرآن", "كتاب", "وحي", "أنبياء", "رسالة"}


def _strip_prefixes(word: str) -> str:
    """إزالة أل التعريف وحروف الجر/العطف الشائعة من بداية الكلمة."""
    for p in PREFIX_STRIP:
        if word.startswith(p) and len(word) - len(p) >= 2:
            return word[len(p):]
    return word


# ═══════════════════════════════════════════════════════════════════════════
# 1) استخراج المفاهيم من السؤال
# ═══════════════════════════════════════════════════════════════════════════
def extract_concepts_from_question(question: str, concepts_db: Dict[str, Any]) -> List[Tuple[str, float]]:
    """
    يحلل سؤالاً بالعربية ويستخرج المفاهيم الموجودة في الـ CKG التي تطابقه.
    يعيد قائمة (اسم المفهوم، درجة التطابق) مرتبة تنازلياً.
    """
    q_norm = normalize_arabic(question)

    # تقسيم السؤال إلى كلمات وتنظيفها من علامات الترقيم
    raw_words = re.split(r'[\s\u060C\u061F\u061B,.!?؟،؛]+', q_norm)
    words = [w for w in raw_words if w and w not in ARABIC_STOPWORDS]

    matches: Dict[str, float] = {}

    # حد أدنى لطول المفهوم لاعتباره صالحاً لمطابقة "تطابق جزئي" (substring)
    # يمنع مفاهيم قصيرة جداً (مثل "بر") من المطابقة الخاطئة داخل كلمات أطول
    MIN_LEN_FOR_SUBSTRING = 3

    # تجهيز نسخ مطبّعة من كل اسم مفهوم (مع وبدون إزالة السوابق)
    for cname, cdata in concepts_db.items():
        c_norm = normalize_arabic(cname)
        c_len  = len(c_norm)

        score = 0.0

        # (أ) تطابق المفهوم كاملاً كسلسلة فرعية من السؤال (لمفاهيم >= 3 حروف فقط)
        if c_len >= MIN_LEN_FOR_SUBSTRING and c_norm in q_norm:
            score = max(score, 1.0)

        # (ب) تطابق على مستوى الكلمات المفردة
        for w in words:
            w_clean = _strip_prefixes(w)
            if not w_clean:
                continue

            if w == c_norm or w_clean == c_norm:
                score = max(score, 1.0)
            elif c_len >= MIN_LEN_FOR_SUBSTRING and (c_norm in w or c_norm in w_clean):
                score = max(score, 0.85)
            elif (
                c_len >= MIN_LEN_FOR_SUBSTRING
                and (w_clean in c_norm or w in c_norm)
                and len(w_clean) >= MIN_LEN_FOR_SUBSTRING
                and len(w_clean) / c_len >= 0.7  # الكلمة تغطي معظم اسم المفهوم (يمنع تطابق "ايمان" مع "ايمان زواج")
            ):
                score = max(score, 0.7)
            elif c_len >= MIN_LEN_FOR_SUBSTRING and " " in c_norm:
                # (ج) مفاهيم مركّبة (تحتوي مسافة، مثل "خمر ومسكرات"):
                # نطابق على مستوى كل كلمة من كلمات المفهوم على حدة
                concept_words = [cw for cw in c_norm.split(" ") if len(cw) >= 3]
                for cw in concept_words:
                    if w_clean == cw or w == cw:
                        score = max(score, 0.8)
                    elif len(w_clean) >= 3 and cw[:3] == w_clean[:3] and abs(len(cw) - len(w_clean)) <= 2:
                        score = max(score, 0.4)
            else:
                # تشابه جذري بسيط: أول 3 حروف متطابقة (لمفاهيم وكلمات >=3 حروف)
                # بشرط أن يكون طول الكلمة والمفهوم متقاربين (يمنع تطابق كلمة قصيرة
                # مع بداية مفهوم مركّب أطول بكثير، مثل "ايمان" مع "ايمان زواج")
                if (
                    len(w_clean) >= 3 and c_len >= 3
                    and w_clean[:3] == c_norm[:3]
                    and abs(len(w_clean) - c_len) <= 2
                ):
                    score = max(score, 0.4)

        if score > 0:
            matches[cname] = score

    # ── خفض أولوية "المفاهيم الإطارية" إن وُجد مفهوم آخر غير إطاري معها ──
    non_meta = [c for c in matches if c not in META_CONCEPTS]
    if non_meta:
        for c in list(matches.keys()):
            if c in META_CONCEPTS:
                matches[c] *= 0.3

    sorted_matches = sorted(matches.items(), key=lambda x: (-x[1], -concepts_db.get(x[0], {}).get("frequency", 0)))
    return sorted_matches


# ═══════════════════════════════════════════════════════════════════════════
# 2) إيجاد المفاهيم المرتبطة عبر العلاقات في CKG
# ═══════════════════════════════════════════════════════════════════════════
def find_related_concepts(
    primary_concepts: List[str],
    relations_db: Dict[str, Any],
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    يبحث في جدول العلاقات (2149 علاقة) عن المفاهيم المرتبطة
    بالمفاهيم الأساسية المستخرجة من السؤال.
    """
    primary_norm = {normalize_arabic(c) for c in primary_concepts}
    related: Dict[str, Dict[str, Any]] = {}

    for rel_key, rel in relations_db.items():
        src = rel.get("source", "")
        tgt = rel.get("target", "")
        src_n, tgt_n = normalize_arabic(src), normalize_arabic(tgt)

        other = None
        if src_n in primary_norm and tgt_n not in primary_norm:
            other = tgt
        elif tgt_n in primary_norm and src_n not in primary_norm:
            other = src

        if other is None:
            continue

        weight = rel.get("weight", 0.0)
        rtype  = rel.get("relation_type", "")

        # إن وجد المفهوم بأكثر من علاقة، نحتفظ بأعلى وزن
        existing = related.get(other)
        if existing is None or weight > existing["weight"]:
            related[other] = {
                "concept":       other,
                "weight":        weight,
                "relation_type": rtype,
                "evidence":      rel.get("evidence", []),
            }

    ranked = sorted(related.values(), key=lambda x: -x["weight"])
    return ranked[:top_k]


# ═══════════════════════════════════════════════════════════════════════════
# 3) استرجاع الآيات الداعمة
# ═══════════════════════════════════════════════════════════════════════════
def _ref_to_surah_ayah(ref: str) -> Tuple[int, int]:
    """يحوّل مرجعاً مثل 'quran:2:153' إلى (سورة، آية)."""
    try:
        parts = ref.split(":")
        return int(parts[-2]), int(parts[-1])
    except Exception:
        return (0, 0)


def retrieve_supporting_verses(
    concept_matches: List[Tuple[str, float]],
    concepts_db: Dict[str, Any],
    ayat_by_ref: Dict[Tuple[int, int], Dict[str, Any]],
    max_verses: int = 5,
) -> List[Dict[str, Any]]:
    """
    يجمع الآيات الداعمة من حقل sources لكل مفهوم مطابق،
    مع ترتيبها بحسب قوة تطابق المفهوم.
    """
    seen_refs = set()
    verses: List[Dict[str, Any]] = []

    for cname, score in concept_matches:
        cdata = concepts_db.get(cname, {})
        sources = cdata.get("sources", [])
        for ref in sources:
            sa = _ref_to_surah_ayah(ref)
            if sa in seen_refs or sa == (0, 0):
                continue
            ayah_data = ayat_by_ref.get(sa)
            if not ayah_data:
                continue
            seen_refs.add(sa)
            verses.append({
                "surah":   sa[0],
                "ayah":    sa[1],
                "text":    ayah_data.get("text", ""),
                "concept": cname,
                "score":   score,
            })
            if len(verses) >= max_verses:
                return verses
    return verses


# ═══════════════════════════════════════════════════════════════════════════
# 4) درجة الثقة
# ═══════════════════════════════════════════════════════════════════════════
def compute_confidence(
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
) -> float:
    """
    درجة ثقة مبنية على:
      - وجود مفاهيم مباشرة مطابقة (40%)
      - وجود علاقات دلالية مستنتجة (25%)
      - وجود آيات داعمة (35%)
    """
    confidence = 0.0

    if concept_matches:
        best_score = max(s for _, s in concept_matches)
        confidence += 0.40 * best_score

    if related_concepts:
        confidence += 0.25 * min(len(related_concepts) / 5, 1.0)

    if verses:
        confidence += 0.35 * min(len(verses) / 5, 1.0)

    return round(min(confidence, 1.0), 4)


# ═══════════════════════════════════════════════════════════════════════════
# 5) توليد إجابة منظمة
# ═══════════════════════════════════════════════════════════════════════════
def generate_answer(
    question: str,
    concept_matches: List[Tuple[str, float]],
    related_concepts: List[Dict[str, Any]],
    verses: List[Dict[str, Any]],
    concepts_db: Dict[str, Any],
) -> Dict[str, Any]:
    """
    يبني إجابة منظمة (ملخص + مفاهيم مرتبطة + آيات داعمة + درجة ثقة)
    اعتماداً فقط على بيانات الـ CKG والقرآن الموجودة.
    """
    confidence = compute_confidence(concept_matches, related_concepts, verses)

    if not concept_matches:
        return {
            "question":         question,
            "summary":          "لم يتم العثور على مفاهيم مرتبطة بهذا السؤال في قاعدة المعرفة الحالية (173 مفهوم).",
            "primary_concepts": [],
            "related_concepts": [],
            "verses":           [],
            "confidence":       0.0,
        }

    # ── المفاهيم الأساسية المكتشفة ──
    primary_names = [c for c, _ in concept_matches[:3]]

    # استبعاد المفاهيم "الإطارية" (مثل: قرآن، كتاب) من صياغة الملخص
    # إن وُجد معها مفهوم آخر أكثر دلالة على موضوع السؤال
    non_meta_names = [c for c in primary_names if c not in META_CONCEPTS]
    topic_names = non_meta_names if non_meta_names else primary_names

    # ── بناء ملخص الإجابة ──
    summary_parts = []

    if len(topic_names) == 1:
        summary_parts.append(f"يتحدث القرآن الكريم عن مفهوم «{topic_names[0]}» في عدة مواضع.")
    else:
        joined = "، ".join(f"«{c}»" for c in topic_names)
        summary_parts.append(f"يربط القرآن الكريم بين المفاهيم التالية: {joined}.")

    # ── ذكر العلاقات المستنتجة ──
    if related_concepts:
        top_related = related_concepts[:3]
        rel_names = "، ".join(f"«{r['concept']}»" for r in top_related)
        summary_parts.append(
            f"وترتبط هذه المفاهيم دلالياً (عبر {len(related_concepts)} علاقة مستخرجة من الـ CKG) "
            f"بمفاهيم أخرى أبرزها: {rel_names}."
        )

    # ── ذكر الآيات ──
    if verses:
        summary_parts.append(
            f"وقد استُخرجت {len(verses)} آية من القرآن الكريم (من إجمالي 6236 آية) "
            f"تدعم هذه العلاقة المعرفية."
        )
    else:
        summary_parts.append("ولم يتم العثور على آيات مباشرة مرتبطة بهذا المفهوم في الفهرس الحالي.")

    summary = " ".join(summary_parts)

    # ── تفاصيل المفاهيم الأساسية ──
    primary_details = []
    for cname, score in concept_matches[:5]:
        cdata = concepts_db.get(cname, {})
        primary_details.append({
            "name":      cname,
            "cluster":   cdata.get("cluster", "غير مصنّف"),
            "frequency": cdata.get("frequency", 0),
            "match":     score,
        })

    return {
        "question":         question,
        "summary":          summary,
        "primary_concepts": primary_details,
        "related_concepts": related_concepts,
        "verses":           verses,
        "confidence":       confidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# 6) الدالة الرئيسية — تجميع كل المراحل
# ═══════════════════════════════════════════════════════════════════════════
def answer_question(
    question: str,
    ckg: Dict[str, Any],
    ayat: List[Dict[str, Any]],
    max_verses: int = 5,
    max_related: int = 8,
) -> Dict[str, Any]:
    """
    نقطة الدخول الرئيسية لمحرك الأسئلة والأجوبة.

    المراحل:
      1. استخراج المفاهيم من السؤال
      2. البحث في العلاقات (2149 علاقة) عن مفاهيم مرتبطة
      3. استرجاع الآيات الداعمة من sources
      4. توليد إجابة منظمة مع درجة ثقة
    """
    concepts_db  = ckg.get("concepts", {})
    relations_db = ckg.get("relations", {})

    # فهرسة الآيات بحسب (سورة، آية) لتسريع البحث
    ayat_by_ref = {(a.get("surah"), a.get("ayah")): a for a in ayat}

    # 1. استخراج المفاهيم
    concept_matches = extract_concepts_from_question(question, concepts_db)

    # 2. المفاهيم المرتبطة
    primary_names = [c for c, _ in concept_matches[:3]]
    related_concepts = find_related_concepts(primary_names, relations_db, top_k=max_related) if primary_names else []

    # 3. الآيات الداعمة
    verses = retrieve_supporting_verses(concept_matches, concepts_db, ayat_by_ref, max_verses=max_verses)

    # 4. الإجابة المنظمة
    result = generate_answer(question, concept_matches, related_concepts, verses, concepts_db)

    return result


# ═══════════════════════════════════════════════════════════════════════════
# 7) أداة مساعدة: تشابه نصي بين سؤالين (للذاكرة التجريبية لاحقاً)
# ═══════════════════════════════════════════════════════════════════════════
def question_similarity(q1: str, q2: str) -> float:
    """
    تشابه بسيط بين سؤالين بناءً على تقاطع الكلمات (Jaccard)
    بعد التطبيع وإزالة كلمات الوقف.
    """
    def tokenize(q: str) -> set:
        norm = normalize_arabic(q)
        raw = re.split(r'[\s\u060C\u061F\u061B,.!?؟،؛]+', norm)
        return {_strip_prefixes(w) for w in raw if w and w not in ARABIC_STOPWORDS}

    t1, t2 = tokenize(q1), tokenize(q2)
    if not t1 or not t2:
        return 0.0
    inter = len(t1 & t2)
    union = len(t1 | t2)
    return round(inter / union, 4) if union else 0.0
