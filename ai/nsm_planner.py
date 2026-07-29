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
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

ROOT = Path(__file__).parent.parent

try:
    from ai.task_manager import (
        create_plan as _tm_create_plan,
        update_task_status as _tm_update_task,
        mark_plan_status as _tm_mark_plan,
        topological_order as _tm_topological_order,
    )
    _TASK_MGR_OK = True
except Exception:
    _TASK_MGR_OK = False

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
_PATH_PATTERN = _re.compile(r"[\w/]+\.(py|json|toml|md|txt|yaml|yml|csv)\b")


def is_planning_request(text: str) -> bool:
    """
    يكشف إذا كان الطلب يستدعي Planning Engine (بناء من الصفر).

    المنطق الصارم:
    1. يجب أن يحتوي كلمة بناء/إنشاء/تطوير واضحة
    2. يجب أن يحتوي كلمة هدف (تطبيق/نظام/أداة/...)
    3. لا يكون سؤالاً (لا يبدأ بـ ما/هل/كيف/من/أين/متى/شرح)
    4. لا يذكر مسار ملف محدد
    5. لا يبدأ بكلمة تعديل
    """
    t = text.strip()

    # ── حد أدنى للطول: الأوامر القصيرة جداً ليست تخطيطاً ──
    if len(t) < 10:
        return False

    # ── استثناء: مسار ملف محدد → تعديل ──
    if _PATH_PATTERN.search(t):
        return False

    # ── استثناء: كلمات أسئلة/شرح/تحليل في البداية → Agent عادي ──
    _QUESTION_STARTS = (
        "ما ", "ما هو", "ما هي", "ما هم", "ما الفرق",
        "هل ", "كيف ", "من ", "أين ", "متى ", "لماذا ",
        "اشرح", "شرح", "وضّح", "وضح", "فسّر", "فسر",
        "قيّم", "قيم", "حلل", "حلّل", "قارن",
        "أريد منك تقييم", "اريد منك تقييم",
        "أخبرني", "اخبرني", "أخبر", "اخبر",
        "ما رأيك", "رأيك", "تقييم",
        "افحص", "قائمة", "ملخص", "ارفع",
        "عدّل", "عدل", "غيّر", "غير", "صحح", "أصلح",
        "اقرأ", "اقرأ ملفات", "المطلوب",
    )
    for start in _QUESTION_STARTS:
        if t.startswith(start):
            return False

    # ── استثناء: نية تحليلية/تدقيقية واضحة في أي مكان بالنص ──
    # (حتى لو لم يبدأ بها النص) → Agent عادي وليس Planner
    _AUDIT_INTENT = (
        "أخبرني", "اخبرني", "ماذا ينقص", "ما الذي ينقص",
        "ماذا ناقص", "وأخبرني", "واخبرني", "أخبرني بالضبط",
    )
    if any(a in t for a in _AUDIT_INTENT):
        return False

    # ── يجب أن يحتوي كلمة بناء صريحة ──
    _BUILD_WORDS = (
        "أنشئ", "انشئ", "ابنِ", "ابني", "اصنع", "اعمل",
        "طور", "طوّر", "برمج", "اكتب كود", "اكتب برنامج",
        "اكتب سكريبت", "برمجلي", "اعمللي", "اصنعلي",
        "بناء تطبيق", "بناء نظام", "بناء موقع", "بناء أداة", "بناء اداة",
        "بناء بوت", "بناء واجهة", "بناء لوحة", "بناء خدمة",
        "إنشاء تطبيق", "إنشاء نظام", "إنشاء موقع", "إنشاء أداة",
        "تطوير تطبيق", "تطوير نظام", "تطوير موقع",
        "build", "create", "make", "develop", "implement",
        "generate", "write a", "add feature", "new feature",
        "أريد تطبيق", "اريد تطبيق",
        "أريد برنامج", "اريد برنامج",
        "أريد نظام", "اريد نظام",
        "أريد أداة", "اريد اداة",
        "أريد بوت", "اريد بوت",
        "أحتاج تطبيق", "احتاج تطبيق",
        "أحتاج نظام", "احتاج نظام",
        "فكرتي", "مشروعي",
        "ساعدني في بناء", "ساعدني على بناء",
        "هل يمكنك بناء", "هل تستطيع بناء",
        "ممكن تبني", "ممكن تنشئ", "ممكن تعمل تطبيق",
        "عايز تطبيق", "عايز نظام", "عايز برنامج",
        "أبغى تطبيق", "ابغى تطبيق",
        "أضف ميزة", "اضف ميزة", "أضف قسم", "اضف قسم",
        "أضف صفحة", "اضف صفحة", "أضف خاصية",
    )

    return any(w in t for w in _BUILD_WORDS)


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

# ══════════════════════════════════════════════════════════════════
# 🆕 المرحلة 3 — تحقّق فعلي من نجاح المهمة (لا الاكتفاء بغياب استثناء)
# ══════════════════════════════════════════════════════════════════
# task.status = "done" سابقاً كان يعني فقط "لم يُرمَ استثناء Python" أثناء
# التنفيذ — حتى لو فشل التحقق الذاتي (المرحلة 1) وانتهى self-healing بلا
# إصلاح، أو انسدّ الإكمال التلقائي (المرحلة 2) عند حدّه الأقصى، أو كان
# الإجراء مرفوضاً بسبب قفل وضع المالك. هذه العلامات تظهر كنص عادي داخل
# stream الوكيل دون رفع استثناء، فكانت "المهمة نجحت" رغم ذلك. نفحص هنا
# نص كل مهمة فعلياً عن علامات فشل حقيقية قبل اعتبارها منجزة.
_TASK_FAILURE_MARKERS = (
    "❌ **فشل الإصلاح",                 # المرحلة 1: استُنفدت محاولات self-healing
    "🔒 هذا الإجراء",                    # لم يُنفَّذ فعلياً (قفل وضع المالك)
    "⚠️ **وصلت لحد الإكمال التلقائي**",  # المرحلة 2: انتهت الجولات بلا اكتمال
    "⚠️ استُنفدت ميزانية الخطوات",
    "⚠️ تعذّر تحليل رد النموذج",
    "⚠️ تعذّر إكمال الجولة التالية تلقائياً",
    "⚠️ لا يمكن الوصول لأي مزوّد LLM",
    "⚠️ تعذّر تحليل رد الإصلاح",
)


def _task_output_has_real_failure(task_output: str) -> bool:
    """يفحص نص المهمة كاملاً عن أي علامة فشل حقيقية لم تُحلّ فعلياً."""
    return any(marker in task_output for marker in _TASK_FAILURE_MARKERS)


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

        # ── تسجيل الخطة في نظام المهام المتعددة (إن توفر) ──────────────
        plan_id = -1
        if _TASK_MGR_OK:
            try:
                plan_id = _tm_create_plan(plan)
            except Exception:
                plan_id = -1

        # ── ترتيب تنفيذ حقيقي يحترم depends_on بدل ترتيب القائمة فقط ──
        if _TASK_MGR_OK:
            try:
                exec_order = _tm_topological_order(plan.tasks)
            except Exception:
                exec_order = plan.tasks
        else:
            exec_order = plan.tasks

        # ── تنفيذ المهام ──
        completed: List[PlanTask] = []
        all_files: List[str] = []

        for task in exec_order:
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
                if _TASK_MGR_OK and plan_id > 0:
                    _tm_update_task(plan_id, task.id, "failed", str(e))
                yield f"\n❌ فشلت المهمة: {e}\n\n"
                continue

            task.status = "done"
            task.result = task_output
            # 🆕 المرحلة 3: تحقّق فعلي — نص المهمة قد يحتوي فشلاً حقيقياً
            # (تحقّق ذاتي فشل / إكمال تلقائي انسدّ / قفل صلاحيات) رغم عدم
            # رمي استثناء Python. لا نعتبرها "منجزة" فعلياً في هذه الحالة.
            if _task_output_has_real_failure(task_output):
                task.status = "failed"
            if _TASK_MGR_OK and plan_id > 0:
                _tm_update_task(plan_id, task.id, task.status, task_output)
            all_files.extend(task.files)
            completed.append(task)
            yield "\n"

        # ── تحديث حالة الخطة النهائية في نظام المهام المتعددة ──────────
        _has_failed = any(t.status == "failed" for t in plan.tasks)

        # ── 🆕 المرحلة 4: معاينة وتحقّق بصري حقيقي ──────────────────────
        # NSM تطبيق Streamlit واحد موحّد (streamlit_app.py) — أي ملف .py
        # عُدِّل أو أُنشئ قد يكسر إقلاع التطبيق كاملاً حتى لو نجح py_compile
        # (المرحلة 1، يفحص syntax فقط ولا يكتشف أخطاء وقت التشغيل مثل
        # استيراد ناقص أو استثناء عند الإقلاع). لذا، إن لم تفشل أي مهمة
        # حتى الآن، نُشغّل التطبيق فعلياً في عملية خلفية مؤقتة ونتحقق أنه
        # يُحمَّل بلا خطأ خادم (500) قبل اعتبار الخطة منجزة فعلاً.
        preview_result = ""
        if not _has_failed and any(f.endswith(".py") for f in all_files):
            yield "\n---\n\n"
            yield "🖥️ **المرحلة 4: تشغيل معاينة حيّة (streamlit run) للتحقق البصري...**\n\n"
            try:
                from ai.preview_check import check_streamlit_boots
                preview_result = check_streamlit_boots("streamlit_app.py")
            except Exception as e:
                preview_result = f"❌ خطأ في المعاينة الحيّة: {e}"
            yield f"{preview_result}\n\n"
            if preview_result.startswith("❌"):
                _has_failed = True

        if _TASK_MGR_OK and plan_id > 0:
            _tm_mark_plan(plan_id, "failed" if _has_failed else "done")

        # ── 🆕 المرحلة 3: تسليم نهاية-لنهاية ──────────────────────────
        # كل المهام done فعلياً (بعد التحقق الحقيقي أعلاه، وليس فقط غياب
        # استثناء) والمعاينة الحيّة (المرحلة 4) نجحت → رفع تلقائي لـ GitHub
        # برسالة عربية واضحة، بدل انتظار طلب "ارفع" منفصل من المستخدم.
        # شرط أمان صارم: أي مهمة فاشلة، أو فشل التحقق الذاتي (المرحلة 1)،
        # أو فشل المعاينة الحيّة (المرحلة 4) يوقف الرفع تماماً.
        pushed = False
        push_result = ""
        if not plan.tasks:
            pass
        elif _has_failed:
            yield "\n---\n\n"
            reason = ("فشل المعاينة الحيّة (المرحلة 4) عند تشغيل التطبيق فعلياً"
                      if preview_result.startswith("❌")
                      else "توجد مهمة واحدة على الأقل فشلت أو لم تجتز التحقق الذاتي بعد الكتابة")
            yield (f"🚫 **لن أرفع تلقائياً لـ GitHub** — {reason}. "
                   f"أصلح الأخطاء أعلاه ثم اطلب الرفع يدوياً (\"ارفع\") بعد التأكد.\n\n")
        else:
            yield "\n---\n\n"
            yield "📦 **المرحلة 3: كل المهام نجحت واجتازت التحقق — رفع تلقائي لـ GitHub...**\n\n"
            commit_msg = (
                f"{plan.app_name}: {plan.description}".strip(": ")
                or f"إضافة {plan.app_name}"
            )[:200]
            try:
                from ai.nsm_agent_core import _run_step
                push_result = _run_step({"action": "git_push", "message": commit_msg})
            except Exception as e:
                push_result = f"❌ خطأ في الرفع التلقائي: {e}"
            yield f"{push_result}\n\n"
            pushed = push_result.startswith("📤")

        # ── ملخص نهائي ──
        yield "\n---\n\n"
        yield self._format_summary(
            plan, completed, all_files,
            pushed=pushed, push_result=push_result, preview_result=preview_result,
        )

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

    def _format_summary(
        self,
        plan: AppPlan,
        completed: List[PlanTask],
        files: List[str],
        pushed: bool = False,
        push_result: str = "",
        preview_result: str = "",
    ) -> str:
        done = [t for t in completed if t.status == "done"]
        failed = [t for t in plan.tasks if t.status == "failed"]
        _overall_failed = bool(failed) or preview_result.startswith("❌")

        lines = [
            "## ✅ اكتملت الخطة!" if not _overall_failed else "## ⚠️ اكتملت مع أخطاء",
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

        # 🆕 المرحلة 4: نتيجة المعاينة الحيّة (إن نُفِّذت)
        if preview_result:
            icon = "✅" if preview_result.startswith("✅") else "❌"
            lines.append(f"\n**المعاينة الحيّة:** {icon} {preview_result[:200]}")

        # 🆕 المرحلة 3: حالة الرفع الفعلية بدل اقتراح "ارفع" الثابت
        lines.append("")
        if pushed:
            lines.append("**الرفع لـ GitHub:** ✅ تم تلقائياً بعد نجاح كل المهام والتحقق والمعاينة الحيّة.")
        elif failed or preview_result.startswith("❌"):
            lines.append("**الرفع لـ GitHub:** 🚫 لم يُنفَّذ — أصلح المشاكل أعلاه أولاً ثم اطلب \"ارفع\".")
        elif push_result:
            lines.append(f"**الرفع لـ GitHub:** ⚠️ حاولت تلقائياً لكن لم ينجح: {push_result[:150]}")
        elif files:
            lines.append("**الرفع لـ GitHub:** ℹ️ لم يُحاول (لا ملفات جديدة نتيجة الخطة).")

        lines += [
            "",
            "**الخطوة التالية:**",
            f"- افحص الملفات: `افحص {files[0]}`" if files else "",
            "- شغّل التطبيق: `run_file`",
        ]
        return "\n".join(l for l in lines if l is not None) + "\n"
