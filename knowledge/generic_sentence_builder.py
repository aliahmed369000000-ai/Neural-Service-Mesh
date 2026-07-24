"""
knowledge/generic_sentence_builder.py — إضافي بالكامل، لا يمسّ ckg_sentence_builder.py
=========================================================================================
نسخة محايدة عن الدومين من ckg_sentence_builder.py: تولّد جمل تدريب من
أي cognitive_graph_<domain>.json (مبني عبر generic_ckg_builder.py)،
بقوالب لا تفترض "القرآن" أو أي دومين محدد.

الاستخدام:
    python3 knowledge/generic_sentence_builder.py \\
        --graph knowledge/cognitive_graph_wikipedia_ar.json \\
        --out ckg_sentences_wikipedia_ar.pkl
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).parent.parent
SEED = 42

# نفس أنواع العلاقات اللي يُنتجها RelationInferencer + ingest_from_concept_matches
# (عام تماماً، بدون أي إشارة لدومين محدد)
_RELATION_TEMPLATES = {
    "co_occurrence": [
        "{a} و{b} يتكرران معاً في نفس السياق",
        "{a} مرتبطة بـ {b}",
    ],
    "semantic": [
        "{a} ترتبط دلالياً بـ {b}",
        "هناك صلة معنوية بين {a} و{b}",
    ],
    "cluster_affinity": [
        "{a} و{b} ينتميان لنفس المحور الموضوعي",
        "{a} تشترك مع {b} في نفس الإطار الفكري",
    ],
    "chain_inference": [
        "{a} ترتبط بـ {b} عبر سلسلة من العلاقات المترابطة",
    ],
}

_CONCEPT_TEMPLATES = [
    "{concept} مفهوم يندرج ضمن محور {cluster}",
    "{concept} من المفاهيم المصنَّفة تحت {cluster}",
]

_NOISE_CLUSTER = "هيكل"  # نفس اصطلاح concept_extractor.py للعقد الهيكلية غير الدلالية
_NOISE_RELATION_TYPES: set = set()  # لا يوجد أنواع علاقات هيكلية بحتة في المسار العام


def build_sentences(graph_file: Path) -> List[str]:
    data = json.loads(graph_file.read_text(encoding="utf-8"))
    concepts: Dict[str, dict] = data.get("concepts", {})
    relations: Dict[str, dict] = data.get("relations", {})

    sentences: List[str] = []

    # 1) جمل المفاهيم (استبعاد العقد الهيكلية: مجموعة:/مرجع:)
    real_concepts = {
        name: c for name, c in concepts.items()
        if c.get("cluster") != _NOISE_CLUSTER
    }
    for name, c in real_concepts.items():
        cluster = c.get("cluster", "عام")
        if not cluster or cluster == name or name in cluster.split("_"):
            continue  # تجنّب جمل تكرارية بلا معنى (المفهوم = اسم عنقوده)
        tpl = _CONCEPT_TEMPLATES[hash(name) % len(_CONCEPT_TEMPLATES)]
        sentences.append(tpl.format(concept=name, cluster=cluster))

    # 2) جمل العلاقات (قوالب متنوعة حسب نوع العلاقة)
    for key, r in relations.items():
        rtype = r.get("relation_type", "")
        if rtype in _NOISE_RELATION_TYPES:
            continue
        templates = _RELATION_TEMPLATES.get(rtype)
        if not templates:
            continue
        a, b = r.get("source", ""), r.get("target", "")
        if not a or not b or a == b:
            continue
        for tpl in templates:
            sentences.append(tpl.format(a=a, b=b))

    unique_sentences = list(dict.fromkeys(sentences))
    random.Random(SEED).shuffle(unique_sentences)
    return unique_sentences


def main():
    p = argparse.ArgumentParser(description="توليد جمل تدريب من CKG عام")
    p.add_argument("--graph", required=True, help="مسار cognitive_graph_<domain>.json")
    p.add_argument("--out", required=True, help="مسار ملف .pkl الناتج")
    args = p.parse_args()

    graph_file = Path(args.graph)
    out_file = Path(args.out)

    sentences = build_sentences(graph_file)
    with open(out_file, "wb") as f:
        pickle.dump(sentences, f)

    print(f"✅ تم توليد {len(sentences)} جملة تدريب → {out_file}")
    if sentences:
        print(f"   عيّنة: {sentences[:3]}")


if __name__ == "__main__":
    main()
