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

from typing import Dict, List


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


def route_query(query: str, categories: Dict[str, object], max_agents: int = 2) -> List[str]:
    """يختار أنسب 1-2 وكيل من AGENT_CATEGORIES بناءً على تطابق كلمات مفتاحية
    بسيطة بين نص الاستعلام وعنوان/وصف/أسئلة كل فئة. توجيه محلي حتمي بدون
    أي استدعاء شبكي أو LLM إضافي.

    Args:
        query: نص مهمة/سؤال المستخدم.
        categories: قاموس AGENT_CATEGORIES (key -> AgentCategory) من
            ai.agent_categories، يُمرَّر كوسيط لتفادي أي استيراد دائري.
        max_agents: أقصى عدد وكلاء يُعاد اختيارهم.

    Returns:
        قائمة بمفاتيح الفئات (category.key) الأنسب، بالترتيب.
    """
    q = query.strip().lower()
    if not q or not categories:
        return []

    words = {w for w in q.replace("؟", " ").replace("،", " ").split() if len(w) >= 3}
    if not words:
        return list(categories.keys())[:1]

    scores: Dict[str, int] = {}
    for key, cat in categories.items():
        haystack = " ".join(
            [getattr(cat, "title", ""), getattr(cat, "subtitle", "")]
            + list(getattr(cat, "quick_prompts", []) or [])
        ).lower()
        scores[key] = sum(1 for w in words if w in haystack)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top = [key for key, s in ranked if s > 0][:max_agents]

    if not top:
        # لا تطابق واضح — استخدم وكيل "المساعد الشخصي" كافتراضي عام إن وُجد
        top = ["assistant"] if "assistant" in categories else [ranked[0][0]]

    return top
