#!/usr/bin/env python3
"""
فحص CKG بعد Git LFS + تقرير جودة إجابات حقيقي
==============================================
الاستخدام:
  python3 scripts/ckg_quality_report.py
  python3 scripts/ckg_quality_report.py --questions 10
  python3 scripts/ckg_quality_report.py --json-only

يكتشف مؤشرات LFS، يعدّ المفاهيم إن وُجدت، ويشغّل ReasoningPipeline على أسئلة عربية.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "artifacts" / "model_training" / "ckg_quality"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CKG_CANDIDATES = [
    ROOT / "knowledge" / "cognitive_graph.json",
    ROOT / "knowledge" / "cognitive_graph_general_ar.json",
]

DEFAULT_QUESTIONS = [
    "ما الأمانة؟",
    "ما العدل في القرآن؟",
    "ما التقوى؟",
    "ماذا تعني كلمة نور؟",
    "ما الصبر؟",
    "ما معنى العلم النافع؟",
    "ما الرحمة؟",
    "ما الشكر؟",
    "ما التوبة؟",
    "ما الحكمة؟",
]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_lfs_pointer(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:200]
    except Exception:
        return False
    return "git-lfs.github.com" in head or head.startswith("version https://git-lfs")


def inspect_ckg_file(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    info: Dict[str, Any] = {
        "path": str(path.relative_to(ROOT.resolve())) if path.is_file() and str(path).startswith(str(ROOT.resolve())) else str(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else 0,
        "lfs_pointer": False,
        "n_concepts": 0,
        "n_relations": 0,
        "sample_concepts": [],
        "error": None,
    }
    if not path.is_file():
        info["error"] = "missing"
        return info
    if is_lfs_pointer(path):
        info["lfs_pointer"] = True
        info["error"] = "git_lfs_pointer — شغّل: git lfs pull"
        # try read oid/size from pointer
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.startswith("oid "):
                    info["lfs_oid"] = line.split(" ", 1)[-1]
                if line.startswith("size "):
                    info["lfs_size"] = int(line.split()[-1])
        except Exception:
            pass
        return info
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        concepts = data.get("concepts") if isinstance(data, dict) else None
        if isinstance(concepts, dict):
            info["n_concepts"] = len(concepts)
            info["sample_concepts"] = list(concepts.keys())[:15]
        elif isinstance(concepts, list):
            info["n_concepts"] = len(concepts)
            info["sample_concepts"] = [
                str(c.get("name") if isinstance(c, dict) else c) for c in concepts[:15]
            ]
        rels = data.get("relations") if isinstance(data, dict) else None
        if isinstance(rels, list):
            info["n_relations"] = len(rels)
        elif isinstance(rels, dict):
            info["n_relations"] = len(rels)
    except Exception as e:
        info["error"] = str(e)
    return info


def score_answer(text: str, weights: Dict[str, Any], ranked: list) -> Dict[str, float]:
    t = (text or "").strip()
    length_score = min(1.0, len(t) / 200.0)
    concept_score = min(1.0, len(ranked or []) / 5.0)
    # أوزان موجودة ومتنوعة
    wvals = [float(weights.get(k, 0)) for k in ("W_SEMANTIC", "W_SCORE", "W_MEMORY", "W_TOPOLOGY")]
    weight_ok = 1.0 if sum(wvals) > 0.5 else 0.3
    overall = round(0.45 * length_score + 0.35 * concept_score + 0.20 * weight_ok, 3)
    return {
        "length_score": round(length_score, 3),
        "concept_score": round(concept_score, 3),
        "weight_ok": weight_ok,
        "overall": overall,
    }


def run_pipeline_questions(questions: List[str]) -> List[Dict[str, Any]]:
    results = []
    pipe = None
    init_err = None
    try:
        from ai.reasoning_pipeline import ReasoningPipeline
        pipe = ReasoningPipeline(train_on_query=False, use_deep_routing=True, record_episodes=False)
    except Exception as e:
        init_err = str(e)
    for q in questions:
        row: Dict[str, Any] = {"question": q}
        if pipe is None:
            # مسار بديل خفيف: مطابقة مفاهيم من الملف إن وُجد
            row["ok"] = False
            row["error"] = f"pipeline_init: {init_err}"
            row["scores"] = {"overall": 0.0, "length_score": 0.0, "concept_score": 0.0, "weight_ok": 0.0}
            results.append(row)
            continue
        try:
            r = pipe.answer(q)
            text = getattr(r, "answer_text", "") or ""
            weights = getattr(r, "decision_weights", {}) or {}
            ranked = getattr(r, "ranked_concepts", None) or []
            row["answer_preview"] = text[:240]
            row["answer_len"] = len(text)
            row["weights"] = {k: weights.get(k) for k in ("W_SEMANTIC", "W_SCORE", "W_MEMORY", "W_TOPOLOGY", "_ensemble_routing", "_deep_routing")}
            row["n_ranked"] = len(ranked)
            row["scores"] = score_answer(text, weights, ranked)
            row["ok"] = True
        except Exception as e:
            row["ok"] = False
            row["error"] = str(e)
            row["scores"] = {"overall": 0.0}
        results.append(row)
    return results


def build_report(n_questions: int = 10) -> Dict[str, Any]:
    files = [inspect_ckg_file(p) for p in CKG_CANDIDATES]
    usable = [f for f in files if f.get("n_concepts", 0) > 0 and not f.get("lfs_pointer")]
    lfs_blocked = any(f.get("lfs_pointer") for f in files)
    questions = DEFAULT_QUESTIONS[: max(1, min(20, n_questions))]
    answers = run_pipeline_questions(questions)
    scored = [a for a in answers if a.get("ok") and a.get("scores")]
    avg = None
    if scored:
        avg = round(sum(a["scores"]["overall"] for a in scored) / len(scored), 3)
    report = {
        "at": _now(),
        "ckg_files": files,
        "ckg_ready": bool(usable),
        "lfs_blocked": lfs_blocked,
        "fix_ar": (
            "CKG غير جاهز: الملفات مؤشرات Git LFS. نفّذ: git lfs install && git lfs pull"
            if lfs_blocked and not usable
            else ("CKG جاهز للقياس" if usable else "لم يُعثر على مفاهيم في ملفات CKG")
        ),
        "n_questions": len(questions),
        "avg_answer_quality": avg,
        "answers": answers,
        "next_steps_ar": [
            "git lfs pull" if lfs_blocked else "CKG محلي موجود",
            "أعد تشغيل هذا السكربت بعد السحب",
            "إن بقيت الجودة منخفضة: دورة train_batch_v3.py",
        ],
    }
    return report


def write_outputs(report: Dict[str, Any]) -> Dict[str, str]:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    jp = OUT_DIR / f"report_{ts}.json"
    mp = OUT_DIR / "last_report.md"
    jp.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    lines = [
        f"# تقرير جودة CKG والإجابات — {report['at']}",
        "",
        f"**جاهزية CKG:** {report['ckg_ready']}",
        f"**LFS حاجز:** {report['lfs_blocked']}",
        f"**متوسط جودة الإجابات:** {report.get('avg_answer_quality')}",
        "",
        f"> {report.get('fix_ar')}",
        "",
        "## ملفات CKG",
    ]
    for f in report.get("ckg_files") or []:
        lines.append(
            f"- `{f.get('path')}`: concepts={f.get('n_concepts')} lfs={f.get('lfs_pointer')} size={f.get('size_bytes')}"
        )
        if f.get("sample_concepts"):
            lines.append(f"  - عيّنة: {', '.join(f['sample_concepts'][:8])}")
    lines.append("")
    lines.append("## الإجابات")
    for a in report.get("answers") or []:
        if a.get("ok"):
            sc = (a.get("scores") or {}).get("overall")
            lines.append(f"- **{a.get('question')}** — جودة={sc} — طول={a.get('answer_len')}")
            lines.append(f"  - {(a.get('answer_preview') or '')[:120]}…")
        else:
            lines.append(f"- **{a.get('question')}** — خطأ: {a.get('error')}")
    lines.append("")
    lines.append("## الخطوات التالية")
    for s in report.get("next_steps_ar") or []:
        lines.append(f"1. {s}")
    mp.write_text("\n".join(lines), encoding="utf-8")
    # also stable last json
    (OUT_DIR / "last_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return {"json": str(jp.relative_to(ROOT)), "md": str(mp.relative_to(ROOT))}


def main() -> int:
    ap = argparse.ArgumentParser(description="CKG + answer quality report")
    ap.add_argument("--questions", type=int, default=10)
    ap.add_argument("--json-only", action="store_true")
    ap.add_argument("--ckg-only", action="store_true", help="فحص ملفات CKG فقط بدون Pipeline")
    args = ap.parse_args()
    if args.ckg_only:
        files = [inspect_ckg_file(p) for p in CKG_CANDIDATES]
        lfs = any(f.get("lfs_pointer") for f in files)
        ready = any(f.get("n_concepts", 0) > 0 for f in files)
        report = {
            "at": _now(),
            "ckg_files": files,
            "ckg_ready": ready,
            "lfs_blocked": lfs,
            "diagnosis_ar": (
                "CKG غير جاهز: مؤشرات Git LFS — نفّذ git lfs install && git lfs pull"
                if lfs and not ready else
                ("CKG جاهز (فحص ملفات فقط)" if ready else "لا مفاهيم في ملفات CKG")
            ),
            "answers": [],
            "avg_answer_quality": None,
            "next_steps_ar": [
                "git lfs install && git lfs pull" if lfs else "CKG محلي موجود",
                "python3 scripts/ckg_quality_report.py --questions 10",
            ],
        }
    else:
        report = build_report(n_questions=args.questions)
    paths = write_outputs(report)
    report["outputs"] = paths
    if args.json_only:
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        print(f"CKG ready: {report['ckg_ready']} | LFS blocked: {report['lfs_blocked']}")
        print(f"Avg quality: {report.get('avg_answer_quality')}")
        print(report.get("fix_ar"))
        print(f"Wrote {paths['md']} and {paths['json']}")
    return 0 if report.get("ckg_ready") else 2


if __name__ == "__main__":
    raise SystemExit(main())
