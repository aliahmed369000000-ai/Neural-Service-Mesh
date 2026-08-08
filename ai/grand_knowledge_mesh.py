"""
Grand Knowledge Mesh — ربط معرفي عابر للمجالات فوق CKG
======================================================
  • يقرأ مفاهيم CKG القرآنية ويربطها بمحاور معرفية حديثة (فيزياء، أجنة، أخلاق تقنية…)
  • يكتشف مرشّحات علاقات عابرة (heuristic + تشابه أسماء/جذور)
  • لا يدّعي اكتشافاً علمياً منشوراً — مسارات فرضية للبحث البشري

المخرج: شبكة فرعية artifacts/.../grand_mesh/
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GrandKnowledgeMesh")

ROOT = Path(__file__).resolve().parent.parent
MESH_DIR = ROOT / "artifacts" / "model_training" / "civilization" / "grand_mesh"
MESH_DIR.mkdir(parents=True, exist_ok=True)

# محاور عابرة للمجالات — مفاهيم حديثة للربط الدلالي
CROSS_DOMAIN_SEEDS: Dict[str, List[str]] = {
    "فيزياء_كونية": ["كون", "سماوات", "نور", "ظلمات", "قدر", "ميزان", "فلك", "شمس", "قمر"],
    "علم_الأجنة": ["نطفة", "علقة", "مضغة", "خلق", "جنين", "رحم", "إنسان"],
    "أخلاق_تقنية": ["أمانة", "عدل", "شهادة", "كذب", "فساد", "إصلاح", "علم", "حكمة"],
    "اجتماع_بشري": ["أمة", "شعوب", "قبائل", "تعارف", "حق", "باطل", "شورى"],
    "نفس_ووعي": ["قلب", "نفس", "عقل", "بصيرة", "غفلة", "ذكر", "خشية"],
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()



def _load_ckg_names(limit: int = 5000) -> List[str]:
    names: List[str] = []
    try:
        from knowledge.cognitive_graph import CognitiveKnowledgeGraph
        ckg = CognitiveKnowledgeGraph()
        concepts = getattr(ckg, "_concepts", None) or getattr(ckg, "concepts", None)
        if isinstance(concepts, dict) and concepts:
            names = list(concepts.keys())
    except Exception as e:
        logger.info("CKG class load skip: %s", e)

    def _from_obj(data) -> List[str]:
        if not isinstance(data, dict):
            return []
        concepts = data.get("concepts")
        if isinstance(concepts, dict) and concepts:
            return [str(k) for k in concepts.keys()]
        if isinstance(concepts, list):
            out = []
            for x in concepts:
                if isinstance(x, dict) and x.get("name"):
                    out.append(str(x["name"]))
                elif isinstance(x, str):
                    out.append(x)
            return out
        return []

    if not names:
        for pth in (
            ROOT / "knowledge" / "cognitive_graph_general_ar.json",
            ROOT / "knowledge" / "cognitive_graph.json",
            ROOT / "knowledge" / "entities.json",
        ):
            if not pth.is_file():
                continue
            raw = pth.read_text(encoding="utf-8", errors="ignore")
            if "git-lfs.github.com" in raw or raw.startswith("version https://git-lfs"):
                continue
            try:
                data = json.loads(raw)
            except Exception:
                continue
            names = _from_obj(data)
            if names:
                break
    # إن بقي فارغاً: بذور من المحاور نفسها للتجربة
    if not names:
        for seeds in CROSS_DOMAIN_SEEDS.values():
            names.extend(seeds)
    return names[:limit]


def _score_link(concept: str, seed: str) -> float:
    c, s = concept.strip(), seed.strip()
    if not c or not s:
        return 0.0
    if c == s:
        return 1.0
    if s in c or c in s:
        return 0.75
    # جذر تقريبي: أول 3 حروف
    if len(c) >= 3 and len(s) >= 3 and c[:3] == s[:3]:
        return 0.45
    return 0.0


def build_grand_mesh(min_score: float = 0.45, max_links: int = 200) -> Dict[str, Any]:
    names = _load_ckg_names()
    links: List[Dict[str, Any]] = []
    for domain, seeds in CROSS_DOMAIN_SEEDS.items():
        for seed in seeds:
            scored = []
            for name in names:
                sc = _score_link(name, seed)
                if sc >= min_score:
                    scored.append((name, sc))
            scored.sort(key=lambda x: -x[1])
            for name, sc in scored[:8]:
                links.append(
                    {
                        "ckg_concept": name,
                        "seed": seed,
                        "domain": domain,
                        "score": round(sc, 3),
                        "relation": "cross_domain_hypothesis",
                        "note_ar": "فرضية ربط دلالي — تحتاج مراجعة عالم مختص قبل الاعتماد.",
                    }
                )
                if len(links) >= max_links:
                    break
            if len(links) >= max_links:
                break
        if len(links) >= max_links:
            break

    report = {
        "ok": True,
        "ckg_concepts_scanned": len(names),
        "domains": list(CROSS_DOMAIN_SEEDS.keys()),
        "n_links": len(links),
        "links": links,
        "created_at": _now(),
        "title": "The Grand Knowledge Mesh (hypothesis layer)",
    }
    out = MESH_DIR / f"mesh_{int(datetime.now().timestamp())}.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "## 🕸️ Grand Knowledge Mesh",
        f"- مفاهيم ممسوحة: **{len(names)}**",
        f"- روابط فرضية: **{len(links)}**",
        "",
        "### عيّنة روابط",
    ]
    for L in links[:15]:
        md.append(
            f"- `{L['ckg_concept']}` ↔ **{L['domain']}**/{L['seed']} (score={L['score']})"
        )
    md.append("")
    md.append("_هذه طبقة فرضيات للبحث — ليست حكماً شرعياً أو علمياً نهائياً._")
    (out.with_suffix(".md")).write_text("\n".join(md), encoding="utf-8")
    report["path"] = str(out.relative_to(ROOT))
    return report


def handle_mesh_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(grand\s*mesh|شبك[ةه]\s*معرف|عولم[ةه]\s*ckg|روابط\s*عابره|خريط[ةه]\s*كوني)", text, re.I):
        return None
    r = build_grand_mesh()
    return (
        f"## 🕸️ Grand Knowledge Mesh\n"
        f"- روابط: **{r['n_links']}** · مفاهيم: **{r['ckg_concepts_scanned']}**\n"
        f"- ملف: `{r.get('path')}`\n\n"
        + "```json\n"
        + json.dumps({"domains": r["domains"], "sample": r["links"][:8]}, ensure_ascii=False, indent=2)
        + "\n```"
    )
