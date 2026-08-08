#!/usr/bin/env python3
"""
System Hub — نبض المشروع ككل
============================
يجمع حالة المكوّنات الحية في NSM في تقرير واحد قابل للاستخدام من:
  الواجهة · الوكيل · CI · المطوّر

لا يغيّر سلوك التشغيل؛ طبقة قراءة وتوجيه فقط.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


def _ok(item: Dict[str, Any]) -> bool:
    return bool(item.get("ok"))


def check_paths() -> List[Dict[str, Any]]:
    critical = [
        ("ai/nsm_agent_core.py", "NSMAgent"),
        ("ai/agent_project_bridge.py", "جسر المشروع"),
        ("ai/agent_growth_loop.py", "دورة نمو الوكيل"),
        ("ai/agent_user_assist.py", "مساعد المستخدم"),
        ("ai/model_training_agent.py", "وكيل التدريب"),
        ("ai/hierarchical_moe.py", "Hierarchical MoE"),
        ("ai/moe_ckg_bridge.py", "جسر MoE↔CKG"),
        ("ai/reasoning_pipeline.py", "مسار الاستدلال"),
        ("ui_pages/moe_agent_studio.py", "واجهة MoE"),
        ("ui_pages/unified_agent.py", "الوكيل الموحّد"),
        ("artifacts/hierarchical_moe/hierarchical_moe.pt", "أوزان MoE"),
        ("data/samples/classification_demo.csv", "بيانات تجريبية"),
    ]
    out = []
    for rel, label in critical:
        p = ROOT / rel
        exists = p.is_file()
        size = p.stat().st_size if exists else 0
        out.append(
            {
                "id": rel,
                "label": label,
                "ok": exists,
                "detail": f"{size/1024:.1f} KB" if exists else "مفقود",
            }
        )
    return out


def check_moe() -> Dict[str, Any]:
    try:
        from ai.moe_ckg_bridge import get_moe_bridge

        br = get_moe_bridge()
        if not br.available:
            return {"ok": False, "detail": str(br._load_error), "experts": 0, "categories": 0}
        m = br.moe
        return {
            "ok": True,
            "detail": f"{m.total_experts()} خبير · {len(m._group_order)} فئة",
            "experts": m.total_experts(),
            "categories": len(m._group_order),
            "best": {
                "temperature": getattr(m, "router_temperature", None),
                "shared_coeff": getattr(m, "shared_coeff", None),
                "residual": getattr(m, "input_residual", None),
            },
        }
    except Exception as e:
        return {"ok": False, "detail": f"{type(e).__name__}: {e}", "experts": 0, "categories": 0}


def check_training() -> Dict[str, Any]:
    try:
        from ai.model_training_agent import _read_training_runs, TRAINING_RUNS_LOG

        runs = _read_training_runs(20)
        completed = sum(1 for r in runs if r.get("status") == "completed")
        failed = sum(1 for r in runs if r.get("status") == "failed")
        return {
            "ok": True,
            "detail": f"{len(runs)} مهمة أخيرة · ✅{completed} · ❌{failed}",
            "runs": len(runs),
            "completed": completed,
            "failed": failed,
            "log_exists": Path(TRAINING_RUNS_LOG).is_file(),
        }
    except Exception as e:
        return {"ok": False, "detail": str(e), "runs": 0}


def check_growth() -> Dict[str, Any]:
    try:
        from ai.agent_growth_loop import MEMORY_PATH, _read_jsonl

        mem = _read_jsonl(MEMORY_PATH, 50)
        succ = sum(1 for r in mem if r.get("success"))
        return {
            "ok": True,
            "detail": f"{len(mem)} خبرة · نجاح {succ}",
            "experiences": len(mem),
            "successes": succ,
        }
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def check_bridge() -> Dict[str, Any]:
    try:
        from ai.agent_project_bridge import agent_integration_status

        st = agent_integration_status()
        comps = st.get("components") or {}
        ok_n = sum(1 for v in comps.values() if v is True)
        bad = [k for k, v in comps.items() if v is not True]
        return {
            "ok": ok_n >= max(1, len(comps) // 2),
            "detail": f"{ok_n}/{len(comps)} مكوّن",
            "components": comps,
            "failing": bad,
        }
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def check_project_stats() -> Dict[str, Any]:
    try:
        from ai.agent_growth_loop import inspect_project

        info = inspect_project()
        return {"ok": True, "detail": f"ai={info['ai_modules']} · tests={info['tests']}", **info}
    except Exception as e:
        return {"ok": False, "detail": str(e)}


def system_snapshot() -> Dict[str, Any]:
    """لقطة كاملة لحالة النظام."""
    paths = check_paths()
    sections = {
        "paths": paths,
        "moe": check_moe(),
        "training": check_training(),
        "growth": check_growth(),
        "bridge": check_bridge(),
        "project": check_project_stats(),
    }
    path_ok = sum(1 for p in paths if p["ok"])
    flags = [
        path_ok == len(paths),
        _ok(sections["moe"]),
        _ok(sections["bridge"]),
        _ok(sections["project"]),
    ]
    score = sum(1 for f in flags if f) / max(1, len(flags))
    # soft bonuses
    if _ok(sections["training"]):
        score = min(1.0, score + 0.05)
    if _ok(sections["growth"]) and sections["growth"].get("experiences", 0) > 0:
        score = min(1.0, score + 0.05)

    return {
        "ts": time.time(),
        "score": round(score, 3),
        "path_ok": path_ok,
        "path_total": len(paths),
        "sections": sections,
    }


def format_system_report(snap: Optional[Dict[str, Any]] = None) -> str:
    snap = snap or system_snapshot()
    sec = snap["sections"]
    score = snap["score"]
    bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
    lines = [
        "## 🌐 تقرير نظام NSM",
        "",
        f"**الصحة الكلية:** {score:.0%} `{bar}`",
        f"- ملفات حرجة: **{snap['path_ok']}/{snap['path_total']}**",
        f"- MoE: {'✅' if _ok(sec['moe']) else '❌'} {sec['moe'].get('detail')}",
        f"- جسر الوكلاء: {'✅' if _ok(sec['bridge']) else '❌'} {sec['bridge'].get('detail')}",
        f"- التدريب: {'✅' if _ok(sec['training']) else '⚠️'} {sec['training'].get('detail')}",
        f"- نمو الوكيل: {'✅' if _ok(sec['growth']) else '⚠️'} {sec['growth'].get('detail')}",
        f"- المشروع: {'✅' if _ok(sec['project']) else '❌'} {sec['project'].get('detail')}",
        "",
        "### ملفات حرجة",
    ]
    for p in sec["paths"]:
        mark = "✅" if p["ok"] else "❌"
        lines.append(f"- {mark} **{p['label']}** (`{p['id']}`) — {p['detail']}")

    failing = (sec.get("bridge") or {}).get("failing") or []
    if failing:
        lines.append("")
        lines.append("### مكوّنات تحتاج انتباهاً")
        for k in failing[:8]:
            lines.append(f"- `{k}`")

    lines.extend(
        [
            "",
            "### 👉 مسارات سريعة",
            "1. `مساعدة` — قدرات الوكيل",
            "2. `صحة moe` / تبويب **MoE والوكيل**",
            "3. `مهمة تدريب data/samples/classification_demo.csv الهدف=label`",
            "4. `نفّذ بأمان: افحص المشروع وشغّل اختبارات`",
            "5. تبويب **صحة النظام** في الواجهة",
        ]
    )
    return "\n".join(lines)


def handle_system_command(user_input: str) -> Optional[str]:
    import re

    text = (user_input or "").strip()
    if not text:
        return None
    if re.search(
        r"(تقرير\s*النظام|نبض\s*المشروع|system\s*report|system\s*hub|"
        r"صحة\s*المشروع|حالة\s*NSM|نظرة\s*شاملة)",
        text,
        re.I,
    ):
        return format_system_report()
    return None
