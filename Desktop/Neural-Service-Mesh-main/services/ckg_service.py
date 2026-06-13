from __future__ import annotations

"""CKG service: expose lightweight functions to return CKG statistics used by both
Flask endpoints and the Gradio dashboard without HTTP calls.
"""
from typing import Dict, Any
from services.backend import get_trainer


def get_ckg_stats() -> Dict[str, Any]:
    trainer = get_trainer()
    ckg = trainer.ckg
    data = getattr(ckg, "_data", {}) or {}
    meta = data.get("_meta", {})
    concepts = data.get("concepts", {})
    relations = data.get("relations", {})

    clusters: dict = {}
    for c in concepts.values():
        cl = c.get("cluster", "unknown")
        clusters[cl] = clusters.get(cl, 0) + 1

    top_concepts = sorted(
        [
            {"name": n, "strength": v.get("strength", 0), "frequency": v.get("frequency", 0)}
            for n, v in concepts.items()
        ],
        key=lambda x: x["strength"],
        reverse=True,
    )[:20]

    return {
        "meta": meta,
        "total_concepts": len(concepts),
        "total_relations": len(relations),
        "clusters": clusters,
        "top_concepts": top_concepts,
    }
