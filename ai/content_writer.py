"""
NSM Content Writer — ai/content_writer.py
=============================================
محرك كتابة مقالات متوافقة مع SEO. اللبنة الثانية من ثلاث لبناء
"وكيل صناعة المحتوى" (اكتشاف ترند → كتابة SEO → نشر تلقائي):

  1) ai/web_search_tool.py  → get_trending_topics() / web_search()
  2) ai/content_writer.py   → هذا الملف (توليد المقال)
  3) ai/social_agent.py     → schedule_post() للنشر (اللبنة القادمة)

يطلب هذا المحرك من النموذج (عبر LLMFallback) إخراج JSON مُهيكل: عنوان H1،
meta description، كلمات مفتاحية، وأقسام H2. لو المزوّد النشط لم يلتزم
بصيغة JSON (يحدث مثلاً مع CKG Synthesis النصي البسيط عند غياب كل مفاتيح
الـ API)، يبني المحرك مقالاً بديلاً من النص الخام نفسه دون اختلاق أي
محتوى إضافي، ويُعلم عن ذلك صراحة عبر SEOArticle.structured=False — لا
يُخفي تدهور الجودة أبداً.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ai.llm_fallback import LLMFallback


_CONTENT_SYSTEM_PROMPT = (
    "أنت كاتب محتوى عربي متخصص في تحسين محركات البحث (SEO).\n"
    "مهمتك: كتابة مقال كامل حول الموضوع المُعطى، متوافق مع معايير SEO.\n"
    "أعد الإجابة بصيغة JSON فقط، بدون أي نص إضافي قبله أو بعده وبدون "
    "أسيجة Markdown (```)، بالمفاتيح التالية بالضبط:\n"
    "{\n"
    '  "title": "عنوان H1 جذاب لا يتجاوز 60 حرفاً ويحتوي الكلمة المفتاحية الرئيسية",\n'
    '  "meta_description": "وصف تعريفي بين 70 و160 حرفاً يحتوي الكلمة المفتاحية",\n'
    '  "keywords": ["كلمة مفتاحية 1", "كلمة مفتاحية 2"],\n'
    '  "slug": "نسخة-مختصرة-من-العنوان-بأحرف-عربية-أو-فارغة",\n'
    '  "sections": [\n'
    '    {"heading": "عنوان فرعي H2", "body": "فقرة أو أكثر من نص المحتوى"}\n'
    "  ]\n"
    "}\n"
    "قواعد:\n"
    "1. العربية الفصحى الواضحة، بدون حشو أو تكرار.\n"
    "2. لا تختلق حقائق أو إحصائيات لا مصدر لها في السياق المُعطى.\n"
    "3. استخدم 3-6 أقسام (H2) بحسب عمق الموضوع.\n"
    "4. إذا سُئلت بجدية عن النموذج الأساسي، أجب بصدق ولا تنفِ ذلك."
)


@dataclass
class SEOArticle:
    topic:             str
    title:             str
    meta_description:  str
    keywords:          List[str]
    slug:              str
    sections:          List[Dict[str, str]]     # [{"heading":..., "body":...}, ...]
    provider:          str
    model:             str
    structured:        bool                      # False = تدهور لنص خام بدل JSON
    word_count:        int = 0
    seo_score:         int = 0
    seo_issues:        List[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """يحوّل المقال إلى Markdown جاهز للنشر (H1 + meta كـ اقتباس + H2 لكل قسم)."""
        lines = [f"# {self.title}", ""]
        if self.meta_description:
            lines += [f"*{self.meta_description}*", ""]
        for sec in self.sections:
            heading = (sec.get("heading") or "").strip()
            body = (sec.get("body") or "").strip()
            if heading:
                lines.append(f"## {heading}")
            if body:
                lines.append(body)
            lines.append("")
        return "\n".join(lines).strip()


def _extract_json(raw: str) -> Optional[dict]:
    """يحاول استخراج أول كائن JSON صالح من نص قد يحتوي أسيجة Markdown أو حشواً حول الـ JSON."""
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        return json.loads(text)
    except Exception:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


def _fallback_from_raw_text(topic: str, raw: str) -> Dict:
    """يبني بنية مقال بسيطة من نص خام لو فشل تحليل JSON — بدون اختلاق أي محتوى جديد، فقط إعادة تهيئة النص الأصلي."""
    text = (raw or "").strip()
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    title = paragraphs[0][:60] if paragraphs else topic
    body = "\n\n".join(paragraphs[1:]) if len(paragraphs) > 1 else text
    meta = text[:157] + "..." if len(text) > 160 else text
    return {
        "title": title or topic,
        "meta_description": meta,
        "keywords": [topic],
        "slug": "",
        "sections": [{"heading": topic, "body": body or text}],
    }


def _word_count(sections: List[Dict[str, str]]) -> int:
    return sum(len((s.get("body") or "").split()) for s in sections)


def _score_seo(data: Dict, topic: str, word_count: int) -> Tuple[int, List[str]]:
    """تقييم SEO بقواعد معروفة وثابتة (طول العنوان/الوصف، عدد الأقسام، عدد الكلمات، وجود الكلمة المفتاحية)."""
    issues: List[str] = []
    score = 100

    title = data.get("title") or ""
    if not (10 <= len(title) <= 60):
        issues.append(f"طول العنوان {len(title)} حرفاً (المثالي: 10-60)")
        score -= 15

    meta = data.get("meta_description") or ""
    if not (70 <= len(meta) <= 160):
        issues.append(f"طول الوصف التعريفي {len(meta)} حرفاً (المثالي: 70-160)")
        score -= 15

    sections = data.get("sections") or []
    if len(sections) < 2:
        issues.append("عدد الأقسام (H2) أقل من 2 — يُفضَّل 3 فأكثر لتغطية أعمق")
        score -= 15

    if word_count < 300:
        issues.append(f"عدد الكلمات {word_count} أقل من الحد الأدنى الموصى به (300)")
        score -= 20

    topic_words = {w for w in re.split(r"\s+", topic.strip()) if len(w) > 2}
    full_text = f"{title} {meta}"
    if topic_words and not any(w in full_text for w in topic_words):
        issues.append("الكلمة المفتاحية الرئيسية غير موجودة صراحة في العنوان أو الوصف")
        score -= 15

    if not data.get("keywords"):
        issues.append("لا توجد قائمة كلمات مفتاحية")
        score -= 10

    return max(0, score), issues


def generate_seo_article(
    topic: str,
    context: Optional[str] = None,
    llm: Optional[LLMFallback] = None,
    target_words: int = 700,
) -> SEOArticle:
    """
    يولّد مقالاً متوافقاً مع SEO حول `topic`.

    topic:   محور المقال (مثلاً عنوان من get_trending_topics).
    context: نص سياق حقيقي اختياري (مقتطف خبر/نتيجة بحث) يستند إليه
             النموذج بدل الاختلاق — لا يُستخدَم كإجابة جاهزة، فقط كمرجع.
    llm:     نسخة LLMFallback جاهزة لإعادة استخدام مزوّد مُهيّأ مسبقاً؛
             لو لم تُمرَّر، تُنشأ نسخة جديدة بحجم استجابة (max_tokens)
             أكبر مناسب لمقال كامل بدل الافتراضي القصير (350).
    """
    topic = (topic or "").strip()
    if not topic:
        raise ValueError("generate_seo_article: مطلوب topic غير فارغ")

    if llm is None:
        llm = LLMFallback(max_tokens=1800, temperature=0.5)

    query = (
        f"اكتب مقالاً محوره: {topic}\n"
        f"الطول المستهدف تقريباً: {target_words} كلمة."
    )
    if context:
        query += (
            "\n\nسياق حقيقي للاستناد إليه (لا تختلق أرقاماً أو حقائق غير "
            f"موجودة فيه):\n{context.strip()[:1500]}"
        )

    result = llm.generate(query, system_prompt=_CONTENT_SYSTEM_PROMPT)

    data = _extract_json(result.text)
    structured = data is not None
    if not structured:
        data = _fallback_from_raw_text(topic, result.text)

    sections = data.get("sections") or []
    word_count = _word_count(sections)
    seo_score, seo_issues = _score_seo(data, topic, word_count)
    if not structured:
        seo_issues.insert(
            0,
            "⚠️ المزوّد النشط لم يلتزم بصيغة JSON المطلوبة — تم بناء المقال "
            "من نص خام بديل (جودة SEO أقل موثوقية، راجع قبل النشر)",
        )

    return SEOArticle(
        topic=topic,
        title=data.get("title", topic) or topic,
        meta_description=data.get("meta_description", "") or "",
        keywords=data.get("keywords", []) or [],
        slug=data.get("slug", "") or "",
        sections=sections,
        provider=result.provider.value,
        model=result.model,
        structured=structured,
        word_count=word_count,
        seo_score=seo_score,
        seo_issues=seo_issues,
    )
