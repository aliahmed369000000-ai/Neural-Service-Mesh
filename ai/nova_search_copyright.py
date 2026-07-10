"""
Nova Search & Copyright System — part3_tone_format.md
======================================================
يُطبّق:
  - قواعد البحث على الويب
  - قواعد حقوق النشر (حد 15 كلمة، اقتباس واحد لكل مصدر)
  - محتوى ضار في نتائج البحث
  - إرشادات الاستجابة
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Tuple


# ══════════════════════════════════════════════════════════════════════════
# قواعد حقوق النشر
# ══════════════════════════════════════════════════════════════════════════

COPYRIGHT_RULES = """
<CRITICAL_COPYRIGHT_COMPLIANCE>
قواعد حقوق النشر — غير قابلة للتفاوض:

LIMIT 1 — طول الاقتباس:
- 15+ كلمة من أي مصدر = انتهاك صارم
- إذا لم تستطع التعبير في أقل من 15 كلمة → أعد الصياغة بالكامل

LIMIT 2 — عدد الاقتباسات لكل مصدر:
- اقتباس واحد فقط لكل مصدر — بعده المصدر "مغلق"
- 2+ اقتباسات من مصدر واحد = انتهاك صارم

LIMIT 3 — الأعمال الكاملة:
- لا تعيد أبداً كلمات أغانٍ (ولو سطراً)
- لا تعيد قصائد (ولو بيتاً)
- لا تعيد هايكو (هي أعمال كاملة)
- لا تعيد فقرات مقالات حرفياً

الصياغة الحقيقية = إعادة كتابة كاملة بصوتك الخاص، ليس إزالة علامات الاقتباس.
لا تعيد هيكل المقال أو تسلسله — قدّم ملخصاً قصيراً 2-3 جمل.

قبل أي نص من نتائج البحث، اسأل نفسك:
- هل الاقتباس 15+ كلمة؟ → انتهاك صارم
- هل اقتبست من هذا المصدر من قبل؟ → المصدر مغلق
- هل هي كلمات أغنية/قصيدة/هايكو؟ → لا تعيدها
- هل تعكس الصياغة الأصلية؟ → أعد الكتابة كلياً
</CRITICAL_COPYRIGHT_COMPLIANCE>
"""

SEARCH_RULES = """
<core_search_behaviors>
قواعد البحث على الويب:

1. ابحث عند الحاجة:
   - المعلومات الراهنة التي قد تغيّرت منذ تاريخ قطع المعرفة
   - الأحداث الجارية، المناصب الحالية، الأسعار، السياسات
   - الكيانات غير المعروفة (كلمات مكتوبة بحرف كبير غير مألوفة)
   - "هل X لا يزال CEO؟" → ابحث دائماً

2. لا تبحث عن:
   - الحقائق الثابتة والتاريخية المعروفة
   - المفاهيم الأساسية والتعريفات
   - الأسئلة الشخصية التي لا تحتاج بيانات خارجية

3. حجّم عدد عمليات البحث:
   - سؤال بسيط: 1 عملية
   - مهمة متوسطة: 3-5 عمليات
   - بحث معمّق: 5-10 عمليات

4. استخدم web_fetch لقراءة المحتوى الكامل بعد البحث
5. اجعل استعلامات البحث موجزة: 1-6 كلمات

محتوى ضار في نتائج البحث:
- لا تشر إلى مصادر تروّج لخطاب الكراهية أو العنصرية أو العنف
- لا تساعد في الوصول إلى محتوى ضار حتى لو ادّعى المستخدم شرعيته
- إذا كان القصد من الاستعلام ضاراً → لا تبحث وفسّر القيود بدلاً من ذلك
</core_search_behaviors>
"""

SEARCH_SYSTEM_PROMPT_SECTION = (
    "عند استخدام أدوات البحث:\n"
    + SEARCH_RULES
    + "\n"
    + COPYRIGHT_RULES
)


# ══════════════════════════════════════════════════════════════════════════
# فحص حقوق النشر البرمجي
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class CopyrightCheckResult:
    is_compliant: bool
    violations: List[str]
    suggestion: str


def check_quote_length(text: str) -> List[str]:
    """يفحص ما إذا كانت هناك اقتباسات تتجاوز 15 كلمة."""
    violations = []
    # يبحث عن نصوص بين علامات اقتباس مزدوجة
    quoted = re.findall(r'"([^"]{50,})"', text)
    for q in quoted:
        words = q.split()
        if len(words) >= 15:
            violations.append(
                f"اقتباس طويل ({len(words)} كلمة): \"{q[:60]}...\""
            )
    return violations


def check_response_copyright(response_text: str) -> CopyrightCheckResult:
    """يفحص الرد للتحقق من الامتثال لقواعد حقوق النشر."""
    violations = []

    # فحص طول الاقتباسات
    long_quotes = check_quote_length(response_text)
    violations.extend(long_quotes)

    is_compliant = len(violations) == 0
    suggestion = (
        "أعد صياغة المحتوى المقتبس بكلماتك الخاصة." if violations
        else "الرد ملتزم بقواعد حقوق النشر."
    )

    return CopyrightCheckResult(
        is_compliant=is_compliant,
        violations=violations,
        suggestion=suggestion,
    )


def is_song_lyrics_request(user_message: str) -> bool:
    """يكشف إذا كان المستخدم يطلب كلمات أغاني أو قصائد."""
    patterns = [
        r'\bكلمات\b.*\bأغنية\b',
        r'\bأغنية\b.*\bكلمات\b',
        r'\bsong\s+lyrics\b',
        r'\blyrics\s+of\b',
        r'\bقصيدة\b.*\bكاملة\b',
        r'\bاكتب\s+لي\b.*\bقصيدة\b',
        r'reproduce.*lyrics',
        r'write.*poem.*full',
    ]
    msg_lower = user_message.lower()
    for pattern in patterns:
        if re.search(pattern, msg_lower):
            return True
    return False


SONG_LYRICS_REFUSAL = (
    "لا أستطيع إعادة إنتاج كلمات الأغاني أو القصائد الكاملة نظراً لحقوق النشر. "
    "يمكنني مناقشة موضوعاتها وأسلوبها وأهميتها بدلاً من ذلك."
)


def get_search_copyright_section() -> str:
    """يُعيد قسم البحث وحقوق النشر لنظام Nova."""
    return SEARCH_SYSTEM_PROMPT_SECTION
