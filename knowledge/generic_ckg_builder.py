"""
knowledge/generic_ckg_builder.py — إضافي بالكامل، لا يمسّ ckg_bootstrap.py
============================================================================
يبني Cognitive Knowledge Graph لأي دومين عربي (ويكيبيديا، أدب، تاريخ...)
بإعادة استخدام نفس المحركات العامة الموجودة فعلاً في المشروع:

    GenericConceptExtractor  (knowledge_sources/generic_concept_extractor.py)
    CognitiveKnowledgeGraph  (knowledge/cognitive_graph.py)      ← بدون تعديل
    RelationInferencer       (knowledge/relation_inferencer.py)  ← بدون تعديل

هذان الأخيران عامان تماماً أصلاً (تأكدنا: لا يوجد أي افتراض قرآني
داخل كودهما) — الفرق الوحيد هو استبدال ConceptExtractor (قوائم كلمات
يدوية إسلامية) بـ GenericConceptExtractor (تصنيف تلقائي KMeans).

صيغة ملف الدخل المطلوبة (JSON):
    [
      {"text": "نص المستند...", "reference": "معرّف فريد", "group": "تصنيف اختياري"},
      ...
    ]

الاستخدام:
    python3 knowledge/generic_ckg_builder.py \\
        --input knowledge_sources/data/wikipedia_ar_sample.json \\
        --domain wikipedia_ar \\
        --graph-file knowledge/cognitive_graph_wikipedia_ar.json \\
        --n-clusters 12
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
ROOT = Path(__file__).parent.parent


def load_domain_documents(input_path: Path, max_docs: int = None) -> List[Dict[str, Any]]:
    """يحمّل مستندات الدومين من JSON (قائمة {text, reference, group})."""
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"صيغة غير صحيحة في {input_path}: يجب أن يكون الجذر list")
    docs = [d for d in data if isinstance(d, dict) and d.get("text", "").strip()]
    if max_docs:
        docs = docs[:max_docs]
    return docs


def build_domain_ckg(
    input_path: Path,
    domain: str,
    graph_file: Path = None,
    n_clusters: int = 12,
    max_docs: int = None,
) -> Dict[str, int]:
    """
    يبني ويحفظ CKG كامل لدومين معيّن.
    Returns: إحصائيات البناء (concepts/relations).
    """
    import sys
    sys.path.insert(0, str(ROOT))

    from knowledge_sources.generic_concept_extractor import GenericConceptExtractor
    from knowledge.cognitive_graph import CognitiveKnowledgeGraph
    from knowledge.relation_inferencer import RelationInferencer

    gf = graph_file or (ROOT / "knowledge" / f"cognitive_graph_{domain}.json")

    logger.info(f"[GenericCKG:{domain}] 1/5 تحميل المستندات من {input_path} …")
    docs = load_domain_documents(input_path, max_docs=max_docs)
    if not docs:
        raise ValueError(f"لا يوجد مستندات صالحة في {input_path}")
    texts = [d["text"] for d in docs]
    refs = [d.get("reference", f"{domain}:{i}") for i, d in enumerate(docs)]
    groups = [d.get("group", domain) for d in docs]
    logger.info(f"[GenericCKG:{domain}] مستندات: {len(texts)}")

    logger.info(f"[GenericCKG:{domain}] 2/5 تدريب GenericConceptExtractor (n_clusters={n_clusters}) …")
    extractor = GenericConceptExtractor(n_clusters=n_clusters)
    extractor.fit(texts)

    logger.info(f"[GenericCKG:{domain}] 3/5 استخراج المفاهيم …")
    all_matches = extractor.extract_batch(texts, references=refs, group_names=groups)
    dist = extractor.cluster_distribution(all_matches)
    logger.info(f"[GenericCKG:{domain}] توزيع العناقيد المكتشَفة: {dist}")

    logger.info(f"[GenericCKG:{domain}] 4/5 بناء الجراف المعرفي → {gf}")
    ckg = CognitiveKnowledgeGraph(graph_file=gf)
    result = ckg.ingest_batch(all_matches, refs, auto_save=False)
    logger.info(
        f"[GenericCKG:{domain}] ingest: concepts={result['total_concepts']} "
        f"relations={result['total_relations']}"
    )

    logger.info(f"[GenericCKG:{domain}] 5/5 استنتاج علاقات إضافية (RelationInferencer) …")
    inferencer = RelationInferencer(ckg)
    infer_report = inferencer.run(verbose=False)
    logger.info(f"[GenericCKG:{domain}] استنتاج: {infer_report}")

    ckg.save()
    logger.info(f"[GenericCKG:{domain}] ✅ تم الحفظ → {gf}")

    return {
        "domain": domain,
        "documents": len(texts),
        "concepts": ckg.concept_count(),
        "relations": ckg.relation_count(),
        "graph_file": str(gf),
        "cluster_distribution": dist,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description="بناء CKG عام لأي دومين عربي")
    p.add_argument("--input", required=True, help="مسار JSON للمستندات")
    p.add_argument("--domain", required=True, help="اسم الدومين (مثلاً wikipedia_ar)")
    p.add_argument("--graph-file", default=None, help="مسار حفظ الجراف")
    p.add_argument("--n-clusters", type=int, default=12)
    p.add_argument("--max-docs", type=int, default=None)
    args = p.parse_args()

    stats = build_domain_ckg(
        input_path=Path(args.input),
        domain=args.domain,
        graph_file=Path(args.graph_file) if args.graph_file else None,
        n_clusters=args.n_clusters,
        max_docs=args.max_docs,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
