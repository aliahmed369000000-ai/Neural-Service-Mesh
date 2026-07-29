"""
ai/nsm_verifier.py
====================
🆕 المرحلة 8 من خطة "المراحل المقترحة (٥ فما فوق)" — فصل أدوار
Planner/Editor/Verifier حقيقي.

سابقاً: نموذج LLM واحد كان يقوم بكل شيء ضمن نفس الاستدعاء والـ prompt —
التخطيط (ai/nsm_planner.py::_build_plan_from_llm) والتنفيذ
(ai/nsm_agent_core.py::_call_api عبر _build_system_prompt، الذي يقرر
الخطوات ويكتب الكود معاً). التحقق الموجود أصلاً (المراحل 1، 4، 5) كله
حتمي/برمجي بحت — py_compile، معاينة streamlit حيّة، تنفيذ test_code —
يكتشف "هل الكود يعمل تقنياً" لكنه لا يحكم إطلاقاً "هل هذا فعلاً ما طلبته
المهمة منطقياً" (مثال: كود يُجمَّع وتحقّقه الوظيفي ينجح، لكنه ينفّذ شيئاً
مختلفاً تماماً عمّا وصفته المهمة).

هذه الوحدة تضيف استدعاء LLM **مستقل تماماً** — system prompt خاص به، ضيّق
ومحدود الغرض، منفصل كلياً عن _build_system_prompt (التنفيذ) وعن prompt
بناء الخطة (التخطيط) — مهمته الوحيدة: الحكم بصيغة JSON محدودة هل ناتج
تنفيذ مهمة معيّنة يحقّق وصفها فعلاً. هذا أقرب لبنية متعددة الوكلاء
(Planner منفصل / Editor منفصل / Verifier منفصل)، بدل نموذج ReAct بسيط
يقوم بكل الأدوار معاً.

الاستخدام النموذجي (من ai/nsm_planner.py):
    from ai.nsm_verifier import verify_task_completion
    result = verify_task_completion(task.title, task.description, task_output, call_api_fn)
    if not result["passed"]:
        # فشل دلالي: الكود يعمل تقنياً لكنه لا يحقق المطلوب فعلاً
        ...
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict

_MAX_OUTPUT_CHARS_FOR_VERIFIER = 4_000


def _build_verifier_system_prompt() -> str:
    """
    system prompt مستقل تماماً عن _build_system_prompt (Editor) وعن
    prompt بناء الخطة (Planner) — لا يعرف شيئاً عن هيكل المشروع ولا عن
    صيغة "steps"، فقط دوره الضيّق: الحكم على تطابق ناتج مهمة مع وصفها.
    """
    return """أنت **NSM Verifier** — دور مستقل تماماً عن دوري "التخطيط" و"التنفيذ"
في نفس النظام. مهمتك الوحيدة: الحكم بصرامة وحياد هل ناتج تنفيذ مهمة
برمجية معيّنة يحقّق فعلاً ما وصفته المهمة — لا تُنفّذ شيئاً، ولا تكتب
كوداً، ولا تقترح خطوات. فقط احكم.

تحقّقات تقنية أخرى (py_compile، اختبار وظيفي فعلي، معاينة حيّة) تمّت
بالفعل بمعزل عنك — لا تكرّرها. ركّز فقط على: هل ما نُفِّذ فعلياً (الملفات/
التغييرات المذكورة في الناتج) يطابق منطقياً وصف المهمة المطلوبة، أم أنه
ناتج مختلف عن المطلوب رغم أنه "يعمل" تقنياً؟

## صيغة الرد — JSON فقط لا غير، بلا أي نص خارجه:
{
  "passed": true أو false,
  "reason": "سبب مختصر بالعربية لقرارك (سطر أو سطرين كحد أقصى)"
}

## قواعد الحكم:
1. إن كان الناتج يذكر فشلاً صريحاً (❌) لم يُعالَج: passed=false مباشرة.
2. إن لم يوضّح الناتج أي ملف أو تغيير فعلي مرتبط بوصف المهمة إطلاقاً
   (مثال: مهمة تطلب "أضف زر تصدير CSV" والناتج لا يذكر أي كود أو ملف
   متعلّق بالتصدير أو CSV على الإطلاق): passed=false.
3. إن كان الناتج متوافقاً منطقياً مع الوصف ولو لم يكن مثالياً: passed=true.
4. عند الشك الحقيقي (لا دليل كافٍ لا للنجاح ولا للفشل): passed=true —
   التحقّقات التقنية الحتمية (py_compile/اختبار وظيفي/معاينة) هي الحارس
   الأساسي، ودورك هنا رأي ثانٍ إضافي وليس عائقاً افتراضياً."""


def _extract_verifier_json(raw: str) -> Dict[str, Any]:
    """يستخرج {"passed":bool,"reason":str} من رد الـ LLM بأي شكل، بدون
    رمي استثناء أبداً. عند فشل الاستخراج الكامل: passed=True (لا يُعتبر
    فشلاً افتراضياً — الحارس الأساسي هو التحقّقات الحتمية الموجودة أصلاً)."""
    text = (raw or "").strip()
    candidates = [text]
    for m in re.finditer(r"```(?:json)?(.*?)```", text, re.DOTALL):
        candidates.append(m.group(1).strip())
    s, e = text.find("{"), text.rfind("}")
    if s != -1 and e > s:
        candidates.append(text[s:e + 1])

    for cand in candidates:
        try:
            data = json.loads(cand)
        except Exception:
            continue
        if isinstance(data, dict) and "passed" in data:
            return {
                "passed": bool(data.get("passed", True)),
                "reason": str(data.get("reason", ""))[:500],
            }

    return {"passed": True, "reason": "⚠️ تعذّر تحليل رد Verifier — لم يُعتبَر فشلاً افتراضياً"}


def verify_task_completion(
    task_title: str,
    task_description: str,
    task_output: str,
    call_api_fn: Callable[[list], str],
) -> Dict[str, Any]:
    """
    🆕 المرحلة 8: استدعاء Verifier مستقل حقيقي — messages خاصة به من
    الصفر (لا تشارك أي سياق مع محادثة التنفيذ)، عبر call_api_fn نفسها
    المستخدمة في بقية النظام (نفس بنية الاتصال بالمزوّدات، لكن بمحتوى
    وسياق مختلفين تماماً — هذا هو الفصل الحقيقي بين الأدوار: prompt
    منفصل، لا نموذج مختلف بالضرورة).

    لا ترمي استثناءً أبداً — أي خطأ في الاتصال يُعامَل كـ"غير حاسم"
    (passed=True) حتى لا يوقف تسليم المهمة بسبب فشل مؤقت في هذا الاستدعاء
    الإضافي، بينما التحقّقات الحتمية (py_compile/اختبار وظيفي/معاينة) هي
    الحارس الأساسي غير القابل للتساهل.
    """
    output_snippet = (task_output or "")[:_MAX_OUTPUT_CHARS_FOR_VERIFIER]

    messages = [
        {"role": "system", "content": _build_verifier_system_prompt()},
        {
            "role": "user",
            "content": (
                f"## وصف المهمة المطلوبة:\n"
                f"العنوان: {task_title}\n"
                f"الوصف: {task_description}\n\n"
                f"## ناتج تنفيذ المهمة فعلياً:\n```\n{output_snippet}\n```\n\n"
                f"هل هذا الناتج يحقق فعلاً وصف المهمة أعلاه؟ رد بصيغة JSON فقط."
            ),
        },
    ]

    try:
        raw = call_api_fn(messages)
    except Exception as e:
        return {"passed": True, "reason": f"⚠️ تعذّر الوصول لـ Verifier: {e}"}

    return _extract_verifier_json(raw)
