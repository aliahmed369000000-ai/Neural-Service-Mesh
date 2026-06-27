"""
NSM Planning Engine — ai/nsm_planner.py
=========================================
يحوّل وصفاً نصياً لخطة تنفيذ كاملة ثم يُنفّذها خطوة خطوة.

المراحل:
  1. ANALYZE  — يفهم الفكرة ويحدد نوع التطبيق
  2. PLAN     — يفكك لمهام (tasks) محددة وقابلة للتنفيذ
  3. EXECUTE  — يُنفّذ كل مهمة عبر NSMAgent
  4. VERIFY   — يتحقق من النتيجة النهائية

أنواع التطبيقات المدعومة:
  - streamlit_app   : واجهة Streamlit
  - python_module   : وحدة Python عادية
  - api_endpoint    : نقطة API
  - full_feature    : ميزة كاملة (frontend + backend)
  - data_pipeline   : معالجة بيانات
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════
# هياكل البيانات
# ══════════════════════════════════════════════════════════════════

@dataclass
class PlanTask:
    """مهمة واحدة في الخطة"""
    id: int
    title: str           # عنوان المهمة بالعربية
    description: str     # وصف تفصيلي
    task_type: str       # create_file | edit_file | run_file | install | verify
    files: List[str] = field(default_factory=list)   # الملفات المتأثرة
    depends_on: List[int] = field(default_factory=list)  # تعتمد على مهام أخرى
    status: str = "pending"   # pending | running | done | failed
    result: str = ""

@dataclass
class AppPlan:
    """الخطة الكاملة للتطبيق"""
    idea: str             # الفكرة الأصلية
    app_type: str         # نوع التطبيق
    app_name: str         # اسم التطبيق
    description: str      # وصف مختصر
    tech_stack: List[str] # التقنيات المستخدمة
    tasks: List[PlanTask] # قائمة المهام
    estimated_files: int  # عدد الملفات المتوقعة


# ══════════════════════════════════════════════════════════════════
# كلمات مفتاحية للكشف عن نية المستخدم
# ══════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════
# كلمات تُفعِّل Planning Engine (بناء من الصفر)
# ══════════════════════════════════════════════════════════════════

_PLAN_TRIGGERS = (
    # ── بناء / إنشاء ──
    "أنشئ", "انشئ", "ابنِ", "ابني", "اصنع", "اعمل",
    "أنشأ", "انشأ", "اصنعلي", "اعمللي",

    # ── تطوير / برمجة ──
    "طور", "برمج", "اكتب كود", "اكتب برنامج",
    "اكتب سكريبت", "اكتب script", "طوّر", "برمجلي",

    # ── تطبيق / نظام / موقع / أداة ──
    "تطبيق", "نظام", "موقع", "أداة", "اداة",
    "برنامج", "سكريبت", "script", "بوت", "bot",
    "واجهة", "لوحة", "dashboard", "صفحة",
    "api", "endpoint", "خدمة", "module", "مكتبة",

    # ── أريد / أحتاج ──
    "أريد", "اريد", "أحتاج", "احتاج",
    "أبغى", "ابغى", "أبغي", "ابغي",
    "عايز", "عاوز",

    # ── وصف فكرة ──
    "فكرتي", "فكرة", "مشروع", "project",
    "أريد أن", "اريد ان", "أريد أن أبني",
    "أفكر في", "افكر في", "لدي فكرة", "عندي فكرة",

    # ── طلب مساعدة في البناء ──
    "ساعدني", "ساعدني في", "ساعدني على",
    "هل يمكنك", "هل تستطيع", "هل تقدر",
    "هل ممكن", "ممكن تبني", "ممكن تنشئ",
    "ممكن تعمل", "ممكن تكتب",

    # ── تحسين / إضافة ──
    "أضف", "اضف", "أضف ميزة", "اضف ميزة",
    "حسّن", "حسن", "طوّر", "أضف خاصية",
    "أضف قسم", "اضف قسم", "أضف صفحة",

    # ── إنجليزي مختلط ──
    "build", "create", "make", "develop",
    "generate", "implement", "write",
    "add feature", "new feature",
)

# ── كلمات تُفعِّل وكيل التعديل (Agent عادي بدون تخطيط) ──
_AGENT_ONLY_TRIGGERS = (
    "عدّل", "عدل", "غيّر", "غير", "بدّل", "بدل",
    "صحح", "أصلح", "اصلح", "احذف", "امسح",
    "افحص", "قائمة", "ملخص", "ارفع",
    "هل يحتوي", "هل يستطيع", "هل يمكن", "هل النظام",
    "قيّم", "قيم", "حلل", "حلّل", "قارن",
    "ما رأيك", "اشرح لي", "ما الفرق",
    "كيف يمكن تحسين", "ما نقاط",
    "هل تعتقد", "ما مدى", "قدّم تقريراً",
)

# ── أسماء الملفات أو المسارات تعني تعديل وليس إنشاء ──
import re as _re
_PATH_PATTERN = _re.compile(r"[\w/]+\.(py|json|toml|md|txt|yaml|yml|csv)")


def is_planning_request(text: str) -> bool:
    """
    يكشف إذا كان الطلب يستدعي Planning Engine (بناء من الصفر).

    المنطق:
    1. إذا ذكر مسار ملف محدد → تعديل وليس بناء → False
    2. إذا كان من كلمات التعديل فقط → False
    3. إذا احتوى كلمة من _PLAN_TRIGGERS → True
    """
    t = text.strip()

    # استثناء: إذا ذكر مسار ملف محدد → تعديل
    if _PATH_PATTERN.search(t):
        return False

    # استثناء: إذا بدأ بكلمة تعديل واضحة
    for trigger in _AGENT_ONLY_TRIGGERS:
        if t.startswith(trigger) or t.startswith(trigger + " "):
            return False

    # تحقق من كلمات البناء
    return any(trigger in t for trigger in _PLAN_TRIGGERS)


# ══════════════════════════════════════════════════════════════════
# 1) تحليل الفكرة وبناء الخطة عبر LLM
# ══════════════════════════════════════════════════════════════════

_PLANNER_SYSTEM = """أنت مخطط تطبيقات ذكي. مهمتك: تحليل فكرة المستخدم وتحويلها لخطة تنفيذ دقيقة.

المشروع: Neural Service Mesh — Python/Streamlit، ذكاء اصطناعي عربي، GitHub.

## صيغة الرد — JSON فقط:
{
  "app_name": "اسم قصير للتطبيق بالإنجليزية (snake_case)",
  "app_type": "streamlit_app | python_module | api_endpoint | full_feature | data_pipeline",
  "description": "وصف مختصر بالعربية (جملة واحدة)",
  "tech_stack": ["streamlit", "pandas", ...],
  "tasks": [
    {
      "id": 1,
      "title": "عنوان المهمة",
      "description": "ماذا يجب أن يفعل هذا الملف/الكود بالتفصيل",
      "task_type": "create_file | edit_file | run_file | verify",
      "files": ["المسار/النسبي.py"],
      "depends_on": []
    }
  ]
}

## قواعد التخطيط:
1. أول مهمة دائماً: إنشاء الملف الرئيسي
2. آخر مهمة دائماً: run_file للتحقق
3. المهام تكون صغيرة ومحددة (ملف واحد أو تعديل واحد)
4. المسارات نسبية من جذر المشروع (مثل: ai/new_feature.py)
5. اقترح 3-7 مهام فقط — لا تُعقّد
6. JSON فقط — لا نص خارجه"""

def _build_plan_from_llm(idea: str, call_api_fn) -> Optional[AppPlan]:
    """يستدعي LLM لتحليل الفكرة وبناء الخطة"""
    messages = [
        {"role": "system", "content": _PLANNER_SYSTEM},
        {"role": "user", "content": f"الفكرة: {idea}"},
    ]
    try:
        raw = call_api_fn(messages)
    except Exception as e:
        return None

    # استخراج JSON
    parsed = _extract_json(raw)
    if not parsed:
        return None

    tasks = []
    for i, t in enumerate(parsed.get("tasks", []), 1):
        tasks.append(PlanTask(
            id=t.get("id", i),
            title=t.get("title", f"مهمة {i}"),
            description=t.get("description", ""),
            task_type=t.get("task_type", "create_file"),
            files=t.get("files", []),
            depends_on=t.get("depends_on", []),
        ))

    return AppPlan(
        idea=idea,
        app_type=parsed.get("app_type", "python_module"),
        app_name=parsed.get("app_name", "new_feature"),
        description=parsed.get("description", ""),
        tech_stack=parsed.get("tech_stack", ["python"]),
        tasks=tasks,
        estimated_files=len(tasks),
    )


def _extract_json(raw: str) -> Optional[Dict]:
    """يستخرج JSON من رد LLM بأي شكل"""
    text = raw.strip()
    # مباشر
    try:
        return json.loads(text)
    except Exception:
        pass
    # كتلة ```json
    for m in re.finditer(r"```(?:json)?(.*?)```", text, re.DOTALL):
        try:
            return json.loads(m.group(1).strip())
        except Exception:
            continue
    # أول { ... }
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        try:
            return json.loads(text[s:e+1])
        except Exception:
            # trailing commas
            cleaned = re.sub(r",\s*([}\]])", r"\1", text[s:e+1])
            try:
                return json.loads(cleaned)
            except Exception:
                pass
    return None


# ══════════════════════════════════════════════════════════════════
# 2) بناء prompt تنفيذ كل مهمة
# ══════════════════════════════════════════════════════════════════

def _build_task_prompt(plan: AppPlan, task: PlanTask, completed: List[PlanTask]) -> str:
    """يبني prompt تنفيذ مهمة واحدة مع السياق الكامل"""
    completed_summary = ""
    if completed:
        lines = [f"  ✅ {t.title} → {', '.join(t.files)}" for t in completed]
        completed_summary = "## المهام المنجزة:\n" + "\n".join(lines) + "\n\n"

    files_note = f"الملفات: {', '.join(task.files)}" if task.files else ""

    return f"""أنت تبني تطبيق: **{plan.app_name}**
الوصف: {plan.description}
التقنيات: {', '.join(plan.tech_stack)}

{completed_summary}## المهمة الحالية ({task.id}/{len(plan.tasks)}):
**{task.title}**
{task.description}
{files_note}

## تعليمات:
- اكتب كوداً كاملاً وقابلاً للتشغيل فوراً
- الملفات تكون في المسار الصحيح من جذر المشروع
- أضف docstring واضحة
- رد بـ JSON فقط بصيغة NSM Agent (steps array)"""


# ══════════════════════════════════════════════════════════════════
# 3) Planning Engine الرئيسي
# ══════════════════════════════════════════════════════════════════

class NSMPlanner:
    """
    Planning Engine — يحوّل فكرة نصية لتطبيق كامل.

    الاستخدام:
        planner = NSMPlanner(agent)
        for chunk in planner.build(idea):
            print(chunk, end="", flush=True)
    """

    def __init__(self, agent) -> None:
        """agent: NSMAgent instance"""
        self.agent = agent

    def build(self, idea: str) -> Generator[str, None, None]:
        """
        Generator يبني التطبيق خطوة خطوة مع Streaming.
        """
        yield f"💡 **فهمت الفكرة:** {idea}\n\n"
        yield "📋 **المرحلة 1: تحليل وتخطيط...**\n\n"

        # ── بناء الخطة ──
        plan = _build_plan_from_llm(idea, self.agent._call_api_bound())
        if plan is None:
            yield "❌ لم أتمكن من تحليل الفكرة. حاول وصفها بشكل أوضح.\n"
            return

        # ── عرض الخطة ──
        yield self._format_plan(plan)
        yield "\n---\n\n"
        yield f"🚀 **المرحلة 2: تنفيذ {len(plan.tasks)} مهام...**\n\n"

        # ── تنفيذ المهام ──
        completed: List[PlanTask] = []
        all_files: List[str] = []

        for task in plan.tasks:
            task.status = "running"
            yield f"### 🔧 المهمة {task.id}/{len(plan.tasks)}: {task.title}\n"

            # بناء prompt المهمة
            task_prompt = _build_task_prompt(plan, task, completed)

            # تنفيذ عبر NSMAgent (Streaming)
            task_output = ""
            try:
                for chunk in self.agent.run_stream(task_prompt):
                    task_output += chunk
                    yield chunk
            except Exception as e:
                task.status = "failed"
                task.result = str(e)
                yield f"\n❌ فشلت المهمة: {e}\n\n"
                continue

            task.status = "done"
            task.result = task_output
            all_files.extend(task.files)
            completed.append(task)
            yield "\n"

        # ── ملخص نهائي ──
        yield "\n---\n\n"
        yield self._format_summary(plan, completed, all_files)

    def _call_api_bound(self):
        """يُعيد دالة _call_api من الـ agent لاستخدامها في الـ planner"""
        from ai.nsm_agent_core import _call_api
        return _call_api

    def _format_plan(self, plan: AppPlan) -> str:
        lines = [
            f"## 📐 خطة: {plan.app_name}",
            f"**النوع:** {plan.app_type}",
            f"**الوصف:** {plan.description}",
            f"**التقنيات:** {', '.join(plan.tech_stack)}",
            f"**عدد المهام:** {len(plan.tasks)}",
            "",
            "### المهام:",
        ]
        for t in plan.tasks:
            deps = f" (يعتمد على: {t.depends_on})" if t.depends_on else ""
            files = f" → `{'`, `'.join(t.files)}`" if t.files else ""
            lines.append(f"{t.id}. **{t.title}**{files}{deps}")
            lines.append(f"   {t.description}")
        return "\n".join(lines) + "\n"

    def _format_summary(self, plan: AppPlan, completed: List[PlanTask], files: List[str]) -> str:
        done = [t for t in completed if t.status == "done"]
        failed = [t for t in plan.tasks if t.status == "failed"]

        lines = [
            "## ✅ اكتملت الخطة!" if not failed else "## ⚠️ اكتملت مع أخطاء",
            "",
            f"**المنجز:** {len(done)}/{len(plan.tasks)} مهمة",
        ]

        if files:
            unique_files = list(dict.fromkeys(files))  # إزالة التكرار
            lines.append(f"**الملفات المُنشأة:** `{'`, `'.join(unique_files)}`")

        if failed:
            lines.append("\n**المهام الفاشلة:**")
            for t in failed:
                lines.append(f"  ❌ {t.title}: {t.result[:100]}")

        lines += [
            "",
            "**الخطوة التالية:**",
            f"- افحص الملفات: `افحص {files[0]}`" if files else "",
            "- شغّل التطبيق: `run_file`",
            "- ارفع لـ GitHub: `ارفع`",
        ]
        return "\n".join(l for l in lines if l is not None) + "\n"
