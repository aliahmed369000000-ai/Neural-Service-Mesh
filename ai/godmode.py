"""
Agent Persona & Orchestrator Support — شخصية NSM الموحّدة + منطق توجيه الوكلاء
================================================================================
كان هذا الملف يحتوي سابقاً على أدوات "GODMODE" مصممة لتخطي ضوابط الأمان في
نماذج LLM تجارية (system prompt من نوع jailbreak، تركيبات "Hall of Fame"،
ووحدات AutoTune/STM لتخفيف الرفض). تم استبدال كامل محتوى الملف بمنطق شرعي:

  • NSM_PERSONA_PROMPT: الشخصية الافتراضية الموحّدة لنظام NSM، تُستخدَم في
    الوكيل الاجتماعي (ai/social_agent.py) وأي مكان يحتاج نبرة NSM ثابتة.
  • route_query(): يوجّه استعلام المستخدم تلقائياً إلى فئة/فئات وكلاء
    "Agents Hub" الأنسب (assistant/automation/analytics/reasoning/coding/
    research/maintenance) بالاعتماد على تطابق كلمات مفتاحية بسيطة — بدون
    أي استدعاء LLM إضافي.
  • COORDINATOR_SYSTEM_PROMPT: يُستخدَم لتوليف ردود عدة وكلاء متخصصين في
    إجابة واحدة متماسكة (نمط Multi-Agent Systems: تفويض مهمة رئيسية إلى
    وكلاء فرعيين ثم تجميع نتائجهم من قبل منسّق).
"""

from __future__ import annotations

import json
import re
from typing import Dict, List, Optional


NSM_PERSONA_PROMPT = (
    "أنت الشخصية الموحّدة لنظام NSM (Neural Service Mesh) — مساعد ذكاء "
    "اصطناعي عربي متخصص بالمعرفة الإسلامية والعربية.\n"
    "أجب دائماً بالعربية الفصحى الواضحة، بأسلوب مباشر ومحترم، والتزم "
    "بالضوابط الأخلاقية والشرعية العامة في كل رد. لا تدّعِ أنك نموذج آخر "
    "ولا تتجاهل أي ضوابط أمان."
)


COORDINATOR_SYSTEM_PROMPT = (
    "أنت \"المنسّق\" في نظام NSM متعدد الوكلاء. مهمتك: قراءة ردود عدة "
    "وكلاء متخصصين على نفس المهمة، ثم توليفها في إجابة واحدة منظمة "
    "وخالية من التكرار، مع إبراز أهم نقطة من كل وكيل عند الحاجة.\n"
    "أجب بالعربية الفصحى، بنقاط أو فقرات قصيرة حسب طبيعة المحتوى."
)


def _keyword_scores(query: str, categories: Dict[str, object]) -> Dict[str, int]:
    """نقاط تطابق كلمات مفتاحية بسيطة بين الاستعلام وعنوان/وصف/أسئلة كل فئة."""
    q = query.strip().lower()
    words = {w for w in q.replace("؟", " ").replace("،", " ").split() if len(w) >= 3}
    scores: Dict[str, int] = {}
    for key, cat in categories.items():
        haystack = " ".join(
            [getattr(cat, "title", ""), getattr(cat, "subtitle", "")]
            + list(getattr(cat, "quick_prompts", []) or [])
        ).lower()
        scores[key] = sum(1 for w in words if w in haystack)
    return scores


def _route_via_llm(query: str, categories: Dict[str, object], max_agents: int) -> Optional[List[str]]:
    """توجيه دلالي احتياطي عبر LLM — يُستدعى فقط عندما لا يوجد أي تطابق
    كلمات مفتاحية إطلاقاً (أي عندما تفشل route_query الحتمية بالكامل)،
    حفاظاً على السرعة والحتمية في الحالة الشائعة. يعيد None عند أي فشل
    (لا مفتاح API، رد غير صالح، استيراد فاشل...) حتى تبقى route_query
    تعمل دائماً عبر افتراضيها الآمن (وكيل "المساعد الشخصي")."""
    try:
        from ai.llm_fallback import LLMFallback
    except Exception:
        return None

    keys = list(categories.keys())
    catalogue = "\n".join(
        f"- {k}: {getattr(categories[k], 'title', k)} — {getattr(categories[k], 'subtitle', '')}"
        for k in keys
    )
    prompt = (
        f"صنّف الطلب التالي إلى أنسب {max_agents} فئة/فئات من القائمة أدناه فقط "
        f"(استخدم المفاتيح بالضبط كما هي مكتوبة، لا تُترجمها ولا تُضف غيرها):\n\n"
        f"{catalogue}\n\nالطلب: \"{query.strip()}\"\n\n"
        "أجب بصيغة JSON فقط، مصفوفة نصوص من المفاتيح بالضبط، بدون أي نص خارج "
        'المصفوفة. مثال: ["assistant", "coding"]'
    )
    try:
        result = LLMFallback().generate(
            query=prompt,
            system_prompt="أنت مصنّف نوايا دقيق داخل نظام توجيه وكلاء. أجب بصيغة JSON فقط.",
        )
        raw = result.text or ""
    except Exception:
        return None

    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None

    picked = [k for k in parsed if isinstance(k, str) and k in categories]
    return picked[:max_agents] or None


def route_query(
    query: str,
    categories: Dict[str, object],
    max_agents: int = 2,
    use_llm_fallback: bool = True,
) -> List[str]:
    """يختار أنسب 1-2 وكيل من AGENT_CATEGORIES. المسار الأساسي حتمي ومحلي
    (تطابق كلمات مفتاحية بين الاستعلام وعنوان/وصف/أسئلة كل فئة، بدون أي
    استدعاء شبكي). إن لم يوجد أي تطابق كلمات مفتاحية إطلاقاً، ويُسمح
    بذلك (use_llm_fallback)، يُجرَّب توجيه دلالي احتياطي عبر LLM قبل
    اللجوء للافتراضي العام (وكيل "المساعد الشخصي").

    Args:
        query: نص مهمة/سؤال المستخدم.
        categories: قاموس AGENT_CATEGORIES (key -> AgentCategory) من
            ai.agent_categories، يُمرَّر كوسيط لتفادي أي استيراد دائري.
        max_agents: أقصى عدد وكلاء يُعاد اختيارهم.
        use_llm_fallback: فعّل/عطّل التوجيه الدلالي الاحتياطي عبر LLM.

    Returns:
        قائمة بمفاتيح الفئات (category.key) الأنسب، بالترتيب.
    """
    return route_query_verbose(query, categories, max_agents, use_llm_fallback)[0]


def route_query_verbose(
    query: str,
    categories: Dict[str, object],
    max_agents: int = 2,
    use_llm_fallback: bool = True,
) -> "tuple[List[str], str, Dict[str, int]]":
    """مثل route_query، لكن تُعيد أيضاً طريقة التوجيه الفعلية المُستخدَمة
    ("keyword" | "llm" | "default") ونقاط التطابق الأولية — لعرضها في
    واجهة "🤝 منسّق الوكلاء" كشفافية للمستخدم حول سبب اختيار وكيل معيّن.

    Returns:
        (قائمة المفاتيح المختارة، طريقة التوجيه، نقاط الكلمات المفتاحية)
    """
    q = query.strip().lower()
    if not q or not categories:
        return [], "empty", {}

    words = {w for w in q.replace("؟", " ").replace("،", " ").split() if len(w) >= 3}
    if not words:
        return list(categories.keys())[:1], "default", {}

    scores = _keyword_scores(query, categories)
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [key for key, s in ranked if s > 0][:max_agents]

    if top:
        return top, "keyword", scores

    if use_llm_fallback:
        llm_pick = _route_via_llm(query, categories, max_agents)
        if llm_pick:
            return llm_pick, "llm", scores

    # لا تطابق كلمات مفتاحية ولا توجيه LLM ناجح — استخدم وكيل "المساعد
    # الشخصي" كافتراضي عام إن وُجد
    default = ["assistant"] if "assistant" in categories else [ranked[0][0]]
    return default, "default", scores
