"""
knowledge/ckg_sentence_builder.py
====================================
توليد جمل تدريب أغنى وأنظف من CKG (cognitive_graph.json) لتدريب
ArabicTransformer، بدل الجيل الأول (ckg_sentences.pkl) الذي كان:

  1. يحتوي نويزاً ضخماً: 6,236 "مفهوم" هو في الحقيقة مجرد مرجع آية
     (cluster='quran') مرتبط بعلاقتين هيكليتين (belongs_to, verse_number)
     لا تحملان أي معنى دلالي — هذا وحده كان يشكّل نصف بيانات التدريب تقريباً.
  2. يستخدم قالب جملة واحد فقط لكل العلاقات الدلالية الحقيقية
     ("X مرتبطة بـ Y في القرآن") بغض النظر عن نوع العلاقة الفعلي
     (co_occurrence, semantic, thematic_cluster, root_link, narrative_sequence)
     — تكرار قالب واحد يُضعف تنوّع الإشارة التدريبية.

هذا الملف:
  - يستبعد عقد مرجع الآيات (cluster == 'quran') والعلاقات الهيكلية
    البحتة (belongs_to, verse_number) من التوليد.
  - يستخدم قالب جملة مخصص لكل relation_type حقيقي (تنوّع لغوي حقيقي).
  - يضيف جملاً من arabic_roots (روابط جذور لغوية) و surah_profiles
    (أهم مفاهيم كل سورة) — مصدرا بيانات موجودان في CKG ولم يُستخدما
    في الجيل الأول إطلاقاً.
  - يضيف جملاً معرفية عن المفاهيم عالية التكرار (frequency) لتثبيت
    معلومات كمية في التمثيل.
  - يزيل التكرار (dedupe) ويُرتّب بترتيب عشوائي ثابت (seed) للتكرارية.

الاستخدام:
    python3 knowledge/ckg_sentence_builder.py
    # يكتب ckg_sentences_v2.pkl + يطبع إحصائيات المقارنة مع الجيل الأول
"""
from __future__ import annotations

import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent
GRAPH_FILE = ROOT / "knowledge" / "cognitive_graph.json"
OUT_FILE = ROOT / "ckg_sentences_v2.pkl"

SEED = 42

# عناقيد/علاقات هيكلية بحتة (مراجع آيات) — تُستبعد من التدريب الدلالي
_NOISE_CLUSTER = "quran"
_NOISE_RELATION_TYPES = {"belongs_to", "verse_number"}

# قوالب متنوعة لكل نوع علاقة دلالي حقيقي
_RELATION_TEMPLATES = {
    "co_occurrence": [
        "{a} و{b} يتكرران معاً في سياق واحد بالقرآن",
        "{a} مرتبطة بـ {b} في القرآن",
    ],
    "semantic": [
        "{a} ترتبط دلالياً بـ {b}",
        "هناك صلة معنوية بين {a} و{b}",
    ],
    "thematic_cluster": [
        "{a} و{b} ينتميان لنفس المحور الموضوعي",
        "{a} تشترك مع {b} في نفس الإطار الفكري",
    ],
    "root_link": [
        "{a} تشترك في الجذر اللغوي مع {b}",
        "{a} و{b} من أصل لغوي واحد",
    ],
    "narrative_sequence": [
        "{a} يسبق {b} في التسلسل القصصي القرآني",
        "{a} يتلوه {b} في سياق القصة",
    ],
}

_CONCEPT_TEMPLATES = [
    "{concept} مفهوم يندرج ضمن محور {cluster}",
    "{concept} من المفاهيم المصنَّفة تحت {cluster}",
]

_ROOT_TEMPLATES = [
    "الجذر {root} أشهر مشتقاته {token} وتكرر {freq} مرة في القرآن",
]

_SURAH_TEMPLATES = [
    "من أبرز مفاهيم السورة رقم {surah} هو {concept}",
]

_HIGH_FREQ_ROOT_THRESHOLD = 20

_HIGH_FREQ_THRESHOLD = 100  # تكرار لا يقل عنه المفهوم ليُعتبر "عالي التكرار"


def _load_graph() -> Dict[str, Any]:
    return json.loads(GRAPH_FILE.read_text(encoding="utf-8"))


def build_sentences() -> List[str]:
    data = _load_graph()
    concepts: Dict[str, dict] = data.get("concepts", {})
    relations: Dict[str, dict] = data.get("relations", {})
    arabic_roots: Dict[str, Any] = data.get("arabic_roots", {}) or {}
    surah_profiles: Dict[str, Any] = data.get("surah_profiles", {}) or {}

    sentences: List[str] = []

    # 1) جمل المفاهيم الدلالية الحقيقية (استبعاد نويز مرجع الآيات)
    real_concepts = {
        name: c for name, c in concepts.items()
        if c.get("cluster") != _NOISE_CLUSTER
    }
    for name, c in real_concepts.items():
        cluster = c.get("cluster", "عام")
        tpl = _CONCEPT_TEMPLATES[hash(name) % len(_CONCEPT_TEMPLATES)]
        sentences.append(tpl.format(concept=name, cluster=cluster))

        freq = c.get("frequency", 0)
        if freq >= _HIGH_FREQ_THRESHOLD:
            sentences.append(f"{name} من أكثر المفاهيم تكراراً في القرآن (تكرر {freq} مرة)")

    # 2) جمل العلاقات الدلالية الحقيقية (قوالب متنوعة حسب النوع)
    for key, r in relations.items():
        rtype = r.get("relation_type", "")
        if rtype in _NOISE_RELATION_TYPES:
            continue
        templates = _RELATION_TEMPLATES.get(rtype)
        if not templates:
            continue
        a, b = r.get("source", ""), r.get("target", "")
        if not a or not b:
            continue
        # تنظيف بادئة "root:" التي تُستخدم داخلياً لتمييز عقدة الجذر
        if rtype == "root_link":
            a = a[5:] if a.startswith("root:") else a
            b = b[5:] if b.startswith("root:") else b
        if a == b:
            continue
        # كلا القالبين لنفس العلاقة يُضيفان تنوّعاً حقيقياً بدل تكرار واحد
        for tpl in templates:
            sentences.append(tpl.format(a=a, b=b))

    # 3) جمل الجذور اللغوية (مصدر لم يُستخدم في الجيل الأول)
    if isinstance(arabic_roots, dict):
        for root, info in arabic_roots.items():
            if not isinstance(info, dict):
                continue
            freq = info.get("frequency", 0)
            token = info.get("top_token", root)
            if freq >= _HIGH_FREQ_ROOT_THRESHOLD:
                sentences.append(_ROOT_TEMPLATES[0].format(root=root, token=token, freq=freq))

    # 4) جمل أهم مفاهيم كل سورة (مصدر لم يُستخدم في الجيل الأول)
    if isinstance(surah_profiles, dict):
        for surah_num, top_concepts in surah_profiles.items():
            if not isinstance(top_concepts, list):
                continue
            for tc in top_concepts[:3]:  # أعلى 3 مفاهيم فقط لكل سورة
                cname = tc.get("concept") if isinstance(tc, dict) else tc
                if cname:
                    sentences.append(_SURAH_TEMPLATES[0].format(surah=surah_num, concept=cname))

    # إزالة التكرار مع الحفاظ على تكرارية الترتيب عبر seed ثابت
    unique_sentences = list(dict.fromkeys(sentences))
    random.Random(SEED).shuffle(unique_sentences)
    return unique_sentences


def main() -> None:
    sentences = build_sentences()

    with open(OUT_FILE, "wb") as f:
        pickle.dump(sentences, f)

    old_count = 0
    old_file = ROOT / "ckg_sentences.pkl"
    if old_file.exists():
        with open(old_file, "rb") as f:
            old_count = len(pickle.load(f))

    print(f"✅ تم توليد {len(sentences)} جملة تدريب نظيفة → {OUT_FILE}")
    print(f"   (الجيل الأول: {old_count} جملة، منها نويز مرجع آيات غير دلالي)")
    print(f"   عيّنة: {sentences[:3]}")


if __name__ == "__main__":
    main()
