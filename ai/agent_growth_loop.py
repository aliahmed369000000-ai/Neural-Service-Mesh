#!/usr/bin/env python3
"""
Agent Growth Loop — دورة نمو عملية فوق NSMAgent الموجود
======================================================
يحوّل الطلب إلى: هدف → فحص مشروع → خطة → خطوات آمنة → تحقق → تقييم → ذاكرة.

حدود أمان صارمة:
  • لا shell عشوائي، لا git push، لا نشر، لا كتابة خارج المسارات المسموحة
  • لا تنفيذ تدريب ثقيل إلا بأمر صريح لاحقاً عبر وكيل التدريب
  • كل مهمة تُسجَّل في artifacts/agent_growth/

أوامر مباشرة:
  حالة نمو الوكيل | طوّر الوكيل | خطة: ... | نفّذ بأمان: ... | خبرات الوكيل
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
GROWTH_DIR = ROOT / "artifacts" / "agent_growth"
MEMORY_PATH = GROWTH_DIR / "experiences.jsonl"
REPORTS_PATH = GROWTH_DIR / "mission_reports.jsonl"

# أدوات آمنة فقط — أسماء ثابتة (allowlist)
SAFE_TOOLS = {
    "inspect_project",
    "list_tests",
    "run_safe_tests",
    "training_preview",
    "training_execute",
    "moe_health",
    "read_key_files",
    "integration_status",
}

_DANGEROUS = re.compile(
    r"(rm\s+-rf|git\s+push|force\s+push|deploy|production|"
    r"drop\s+table|format\s+disk|chmod\s+777|/etc/passwd)",
    re.I,
)


def _ensure_dirs() -> None:
    GROWTH_DIR.mkdir(parents=True, exist_ok=True)


def _append_jsonl(path: Path, row: dict) -> None:
    _ensure_dirs()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path, limit: int = 100) -> List[dict]:
    """🆕 أداء: كانت تقرأ الملف بالكامل من البداية (كل سطر مُسجَّل مدى حياة
    المشروع) ثم تأخذ آخر `limit` فقط — تكلفة تكبر بلا حدود مع كل خبرة
    جديدة تُسجَّل، رغم استخدام آخر عدد صغير محدود فقط. هذه تُستدعى عبر
    format_experience_hints في *كل* رسالة محادثة (nsm_agent_core.py).

    الآن: قراءة عكسية من نهاية الملف بمقاطع (byte chunks) تتوسع تدريجياً
    حتى نجمع `limit` سطراً صالحاً على الأقل أو نصل بداية الملف — تكلفة
    تقريبية O(limit) بدل O(حجم الملف الكامل)، مع نفس النتيجة تماماً (آخر
    `limit` سطراً صالحاً بنفس الترتيب).
    """
    if not path.is_file():
        return []

    chunk_size = 8192
    with open(path, "rb") as f:
        f.seek(0, 2)
        file_size = f.tell()
        pos = file_size
        buf = b""
        while True:
            read_size = min(chunk_size, pos)
            pos -= read_size
            f.seek(pos)
            buf = f.read(read_size) + buf

            # عدد الأسطر "المكتملة" داخل buf (نتجاهل أول سطر قد يكون مقطوعاً
            # ما لم نكن قد وصلنا فعلياً لبداية الملف)
            raw_lines = buf.split(b"\n")
            complete = raw_lines if pos == 0 else raw_lines[1:]
            valid_count = 0
            for raw in complete:
                s = raw.strip()
                if not s:
                    continue
                try:
                    json.loads(s)
                    valid_count += 1
                except Exception:
                    pass

            if valid_count >= limit or pos == 0:
                break
            chunk_size *= 2  # لم يكفِ المقطع — وسّع النافذة وأعد المحاولة

    rows: List[dict] = []
    for line in buf.decode("utf-8", errors="replace").split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows[-limit:]


# ── ذاكرة الخبرات ───────────────────────────────────────────────────────────

def record_experience(
    goal: str,
    plan: List[str],
    tools: List[str],
    success: bool,
    summary: str,
    metrics: Optional[dict] = None,
) -> None:
    _append_jsonl(
        MEMORY_PATH,
        {
            "ts": time.time(),
            "goal": (goal or "")[:300],
            "plan": plan[:12],
            "tools": tools[:12],
            "success": bool(success),
            "summary": (summary or "")[:800],
            "metrics": metrics or {},
        },
    )


def similar_experiences(goal: str, k: int = 3) -> List[dict]:
    """استرجاع خبرات مشابهة بتشابه كلمات بسيط (بدون LLM)."""
    rows = _read_jsonl(MEMORY_PATH, limit=200)
    if not rows:
        return []
    tokens = set(re.findall(r"[\w\u0600-\u06FF]{3,}", (goal or "").lower()))
    if not tokens:
        return rows[-k:]

    def score(r: dict) -> float:
        g = set(re.findall(r"[\w\u0600-\u06FF]{3,}", str(r.get("goal", "")).lower()))
        if not g:
            return 0.0
        inter = len(tokens & g)
        return inter / max(1, len(tokens | g)) + (0.15 if r.get("success") else 0.0)

    ranked = sorted(rows, key=score, reverse=True)
    return [r for r in ranked if score(r) > 0][:k] or ranked[-k:]


def format_experience_hints(goal: str) -> str:
    sims = similar_experiences(goal, k=3)
    if not sims:
        return ""
    lines = ["### 📚 خبرات مشابهة", ""]
    for r in sims:
        mark = "✅" if r.get("success") else "❌"
        lines.append(
            f"- {mark} «{(r.get('goal') or '')[:80]}» · أدوات: {', '.join(r.get('tools') or [])}"
        )
    return "\n".join(lines)


# ── فحص المشروع ─────────────────────────────────────────────────────────────

def inspect_project() -> Dict[str, Any]:
    ai_files = list((ROOT / "ai").glob("*.py")) if (ROOT / "ai").is_dir() else []
    tests = list((ROOT / "tests").glob("test_*.py")) if (ROOT / "tests").is_dir() else []
    data_csv = list((ROOT / "data").rglob("*.csv")) if (ROOT / "data").is_dir() else []
    moe = (ROOT / "artifacts" / "hierarchical_moe" / "hierarchical_moe.pt").is_file()
    return {
        "ai_modules": len(ai_files),
        "tests": len(tests),
        "csv_datasets": len(data_csv),
        "moe_weights": moe,
        "has_training_agent": (ROOT / "ai" / "model_training_agent.py").is_file(),
        "has_nsm_agent": (ROOT / "ai" / "nsm_agent_core.py").is_file(),
    }


def inspect_project_report() -> str:
    info = inspect_project()
    return (
        "## 🔍 فحص المشروع\n\n"
        f"- وحدات `ai/`: **{info['ai_modules']}**\n"
        f"- اختبارات: **{info['tests']}**\n"
        f"- CSV في data/: **{info['csv_datasets']}**\n"
        f"- أوزان MoE: {'✅' if info['moe_weights'] else '❌'}\n"
        f"- وكيل تدريب: {'✅' if info['has_training_agent'] else '❌'}\n"
        f"- NSMAgent: {'✅' if info['has_nsm_agent'] else '❌'}\n"
    )


# ── تخطيط الأهداف ───────────────────────────────────────────────────────────

def decompose_goal(goal: str) -> Dict[str, Any]:
    g = (goal or "").strip()
    dangerous = bool(_DANGEROUS.search(g))
    steps: List[str] = []
    tools: List[str] = []

    steps.append("فهم الهدف وصياغته")
    steps.append("فحص هيكل المشروع")
    tools.append("inspect_project")

    low = g.lower()
    if re.search(r"اختبار|test|pytest", g, re.I):
        steps.append("حصر ملفات الاختبار")
        steps.append("تشغيل اختبارات آمنة محدودة")
        tools.extend(["list_tests", "run_safe_tests"])
    if re.search(r"تدريب|train|csv|نموذج", g, re.I):
        if re.search(r"نف[ّ]?ذ|execute|شغ[ّ]?ل\s*التدريب", g, re.I):
            steps.append("تنفيذ مهمة تدريب آمنة على data/samples فقط")
            tools.append("training_execute")
        else:
            steps.append("معاينة مهمة تدريب (بدون تنفيذ ثقيل)")
            tools.append("training_preview")
    if re.search(r"moe|خبراء|تصنيف", g, re.I):
        steps.append("فحص صحة MoE")
        tools.append("moe_health")
    if re.search(r"دمج|bridge|وكيل|agent", g, re.I):
        steps.append("حالة تكامل الوكلاء")
        tools.append("integration_status")
    if not tools:
        tools.append("inspect_project")
        steps.append("تقرير حالة عامة")

    steps.append("تقييم ذاتي وتسجيل الخبرة")
    return {
        "goal": g[:400],
        "dangerous": dangerous,
        "steps": steps,
        "tools": list(dict.fromkeys(tools)),  # unique preserve order
        "confidence": 0.35 if dangerous else 0.7,
    }


def format_plan(plan: Dict[str, Any], experiences_hint: str = "") -> str:
    lines = [
        "## 🗺️ خطة الوكيل",
        "",
        f"**الهدف:** {plan.get('goal')}",
        f"**الثقة:** {plan.get('confidence')}",
        "",
        "### الخطوات",
    ]
    for i, s in enumerate(plan.get("steps") or [], 1):
        lines.append(f"{i}. {s}")
    lines.append("")
    lines.append("### أدوات آمنة")
    for t in plan.get("tools") or []:
        lines.append(f"- `{t}`")
    if plan.get("dangerous"):
        lines.append("")
        lines.append("⚠️ اكتشف نصاً قد يشير لعملية خطرة — **لن تُنفَّذ** خطوات غير آمنة.")
    if experiences_hint:
        lines.append("")
        lines.append(experiences_hint)
    lines.append("")
    lines.append("_للتنفيذ الآمن:_ `نفّذ بأمان: <نفس الهدف>`")
    return "\n".join(lines)


# ── تنفيذ أدوات آمنة ────────────────────────────────────────────────────────

def _tool_list_tests() -> str:
    tests_dir = ROOT / "tests"
    if not tests_dir.is_dir():
        return "لا يوجد مجلد tests/"
    files = sorted(p.name for p in tests_dir.glob("test_*.py"))[:25]
    return "اختبارات: " + ", ".join(f"`{f}`" for f in files) if files else "لا ملفات test_*.py"


def _tool_run_safe_tests() -> Tuple[bool, str]:
    """يشغّل مجموعة صغيرة من الاختبارات غير الخطرة فقط."""
    preferred = [
        "tests/test_model_training_agent.py",
        "tests/test_agent_project_bridge.py",
        "tests/test_ckg_quality_report.py",
    ]
    existing = [t for t in preferred if (ROOT / t).is_file()]
    if not existing:
        return False, "لا توجد اختبارات مفضّلة متاحة"
    cmd = ["python3", "-m", "pytest", "-q", "--tb=line"] + existing
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=90,
        )
        out = (proc.stdout or "")[-1200:] + "\n" + (proc.stderr or "")[-400:]
        ok = proc.returncode == 0
        return ok, f"pytest exit={proc.returncode}\n{out}"
    except subprocess.TimeoutExpired:
        return False, "انتهت مهلة الاختبارات (90s)"
    except Exception as e:
        return False, f"فشل pytest: {type(e).__name__}: {e}"


def _tool_training_preview() -> str:
    try:
        from ai.model_training_agent import run_training_mission
        return run_training_mission(
            "data/samples/classification_demo.csv",
            target_col="label",
            execute=False,
        )
    except Exception as e:
        return f"تعذّرت معاينة التدريب: {e}"


def _tool_training_execute() -> Tuple[bool, str]:
    """تدريب حقيقي محدود على classification_demo فقط (sklearn)."""
    try:
        from ai.model_training_agent import run_training_mission
        out = run_training_mission(
            "data/samples/classification_demo.csv",
            target_col="label",
            prefer="sklearn",
            execute=True,
        )
        ok = "Accuracy" in out or "completed" in out.lower() or "نتائج" in out
        return ok, out
    except Exception as e:
        return False, f"تعذّر تنفيذ التدريب: {e}"


def _tool_moe_health() -> str:
    try:
        from ai.moe_ckg_bridge import get_moe_bridge
        return get_moe_bridge().health_report()
    except Exception as e:
        return f"تعذّر فحص MoE: {e}"


def _tool_integration_status() -> str:
    try:
        from ai.agent_project_bridge import agent_integration_status
        st = agent_integration_status()
        lines = ["## تكامل المكوّنات", ""]
        for k, v in (st.get("components") or {}).items():
            mark = "✅" if v is True else "❌"
            lines.append(f"- {mark} `{k}`" + ("" if v is True else f" — {v}"))
        return "\n".join(lines)
    except Exception as e:
        return f"تعذّر: {e}"


def _tool_read_key_files() -> str:
    keys = [
        "ai/nsm_agent_core.py",
        "ai/agent_project_bridge.py",
        "ai/model_training_agent.py",
    ]
    lines = ["## ملفات مفتاحية (مقتطف)", ""]
    for rel in keys:
        p = ROOT / rel
        if not p.is_file():
            lines.append(f"- `{rel}`: غير موجود")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
            lines.append(f"- `{rel}`: {len(text.splitlines())} سطر · {p.stat().st_size // 1024} KB")
        except Exception as e:
            lines.append(f"- `{rel}`: خطأ {e}")
    return "\n".join(lines)


def execute_tool(name: str) -> Tuple[bool, str]:
    if name not in SAFE_TOOLS:
        return False, f"أداة غير مسموحة: {name}"
    if name == "inspect_project":
        return True, inspect_project_report()
    if name == "list_tests":
        return True, _tool_list_tests()
    if name == "run_safe_tests":
        return _tool_run_safe_tests()
    if name == "training_preview":
        return True, _tool_training_preview()
    if name == "training_execute":
        return _tool_training_execute()
    if name == "moe_health":
        return True, _tool_moe_health()
    if name == "integration_status":
        return True, _tool_integration_status()
    if name == "read_key_files":
        return True, _tool_read_key_files()
    return False, f"غير منفَّذ: {name}"


def evaluate_mission(results: List[Tuple[str, bool, str]]) -> Dict[str, Any]:
    n = len(results) or 1
    ok = sum(1 for _, s, _ in results if s)
    rate = ok / n
    return {
        "success_rate": round(rate, 3),
        "passed": ok,
        "total": len(results),
        "success": rate >= 0.6,
        "note": "نجاح جزئي مقبول" if 0.6 <= rate < 1 else ("نجاح كامل" if rate == 1 else "يحتاج مراجعة"),
    }


def run_safe_mission(goal: str, execute: bool = True) -> str:
    """دورة كاملة: خبرات → خطة → (تنفيذ آمن) → تقييم → ذاكرة."""
    if _DANGEROUS.search(goal or ""):
        record_experience(goal, [], [], False, "رفض: محتوى خطير محتمل")
        return "🚫 رُفض الهدف لأنه قد يتضمن عمليات غير آمنة."

    hints = format_experience_hints(goal)
    plan = decompose_goal(goal)
    report_parts = [format_plan(plan, hints)]

    if not execute:
        return "\n".join(report_parts)

    report_parts.append("\n## ⚙️ تنفيذ آمن\n")
    results: List[Tuple[str, bool, str]] = []
    for tool in plan["tools"]:
        ok, out = execute_tool(tool)
        results.append((tool, ok, out))
        mark = "✅" if ok else "❌"
        report_parts.append(f"### {mark} `{tool}`\n\n{out}\n")

    ev = evaluate_mission(results)
    report_parts.append(
        "\n## 📊 تقييم ذاتي\n\n"
        f"- نجاح: **{ev['passed']}/{ev['total']}** ({ev['success_rate']})\n"
        f"- الحكم: {ev['note']}\n"
    )

    summary = f"{ev['note']} · tools={','.join(plan['tools'])}"
    record_experience(
        goal,
        plan["steps"],
        plan["tools"],
        bool(ev["success"]),
        summary,
        metrics=ev,
    )
    _append_jsonl(
        REPORTS_PATH,
        {
            "ts": time.time(),
            "goal": goal[:300],
            "tools": plan["tools"],
            "evaluation": ev,
        },
    )
    return "\n".join(report_parts)


def growth_status() -> str:
    mem = _read_jsonl(MEMORY_PATH, 50)
    reps = _read_jsonl(REPORTS_PATH, 30)
    succ = sum(1 for r in mem if r.get("success"))
    lines = [
        "## 🌱 حالة نمو الوكيل",
        "",
        f"- خبرات مسجّلة: **{len(mem)}** (نجاح {succ})",
        f"- تقارير مهام: **{len(reps)}**",
        f"- أدوات آمنة: {', '.join(f'`{t}`' for t in sorted(SAFE_TOOLS))}",
        "",
        "### أوامر",
        "- `خطة: <هدف>` — تخطيط فقط",
        "- `نفّذ بأمان: <هدف>` — تنفيذ خطوات آمنة + اختبارات محدودة",
        "- `خبرات الوكيل` — آخر الخبرات",
        "- `طوّر الوكيل` — دورة نمو افتراضية (فحص + اختبارات)",
        "",
        "💡 يمكنك أيضاً: `افحص المشروع` · `شغّل الاختبارات` · `مساعدة`",
    ]
    if mem:
        lines.append("\n### آخر خبرات")
        for r in mem[-5:]:
            mark = "✅" if r.get("success") else "❌"
            lines.append(f"- {mark} {(r.get('goal') or '')[:70]}")
    return "\n".join(lines)


def list_experiences(limit: int = 12) -> str:
    rows = _read_jsonl(MEMORY_PATH, limit)
    if not rows:
        return "## خبرات الوكيل\n\nلا خبرات بعد. نفّذ: `نفّذ بأمان: افحص المشروع وشغّل اختبارات`"
    lines = ["## 🧠 خبرات الوكيل", ""]
    for r in reversed(rows):
        mark = "✅" if r.get("success") else "❌"
        lines.append(
            f"{mark} «{(r.get('goal') or '')[:90]}» · "
            f"{', '.join(r.get('tools') or [])}"
        )
    return "\n".join(lines)


def develop_agent_once() -> str:
    """دورة نمو افتراضية: فحص + تكامل + اختبارات آمنة."""
    return run_safe_mission(
        "طوّر الوكيل: افحص المشروع وحالة التكامل وشغّل اختبارات آمنة",
        execute=True,
    )


def handle_growth_command(user_input: str) -> Optional[str]:
    text = (user_input or "").strip()
    if not text:
        return None

    if re.search(r"(حالة\s*نمو\s*الوكيل|نمو\s*الوكيل|agent\s*growth\s*status)", text, re.I):
        return growth_status()

    if re.search(r"(خبرات\s*الوكيل|experiences\s*agent|ذاكرة\s*الوكيل)", text, re.I):
        return list_experiences()

    if re.search(r"(^| )طو[ّ]?ر\s*الوكيل|develop\s*agent|growth\s*cycle", text, re.I):
        return develop_agent_once()

    m_plan = re.search(r"(?:خطة|plan)\s*[:：]\s*(.+)$", text, re.I | re.S)
    if m_plan:
        goal = m_plan.group(1).strip()
        return run_safe_mission(goal, execute=False)

    m_run = re.search(
        r"(?:نف[ّ]?ذ\s*بأمان|execute\s*safely|safe\s*run)\s*[:：]?\s*(.+)$",
        text,
        re.I | re.S,
    )
    if m_run:
        goal = m_run.group(1).strip()
        return run_safe_mission(goal, execute=True)

    # هدف عام بصيغة «وكيل: ...» أو «مهمة وكيل ...»
    m_goal = re.search(r"(?:وكيل|agent)\s*[:：]\s*(.+)$", text, re.I | re.S)
    if m_goal:
        goal = m_goal.group(1).strip()
        # تخطيط فقط ما لم يُطلب التنفيذ
        execute = bool(re.search(r"نف[ّ]?ذ|execute", text, re.I))
        return run_safe_mission(goal, execute=execute)

    # عبارات طبيعية قصيرة بدون بادئة رسمية
    if re.search(r"^(افحص|فحص)\s*(المشروع|النظام|الكود)?$", text, re.I):
        return run_safe_mission("افحص المشروع", execute=True)
    if re.search(r"(شغ[ّ]?ل|run).{0,8}(اختبار|test)", text, re.I) and len(text) < 40:
        return run_safe_mission("شغّل اختبارات آمنة", execute=True)

    return None
