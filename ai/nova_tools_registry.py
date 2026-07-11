"""
Nova Tools Registry — parts 4, 5, 6
=====================================
يُوثّق مخططات أدوات Nova من الأجزاء 4-6:
  - sports_data, image_search, memory_user_edits (part4)
  - places_map_display, places_search, present_files,
    recent_chats, recipe_display, recommend_claude_apps,
    search_mcp_registry, web_fetch, web_search (parts 5-6)
  - tool_search, weather_fetch (part6)

هذه وثائق مرجعية — مخططات الأدوات التي تتوقعها Nova من البنية التحتية.
"""
from __future__ import annotations

from typing import Any, Dict

# ══════════════════════════════════════════════════════════════════════════
# فئات الأدوات المتاحة لـ Nova
# ══════════════════════════════════════════════════════════════════════════

NOVA_TOOL_CATEGORIES = {
    "search": ["web_search", "image_search", "tool_search"],
    "fetch": ["web_fetch"],
    "memory": ["memory_user_edits"],
    "places": ["places_search", "places_map_display_v0"],
    "communication": ["message_compose_v1"],
    "media": ["recipe_display_v0", "weather_fetch"],
    "files": ["present_files", "str_replace", "view"],
    "chats": ["recent_chats", "conversation_search"],
    "apps": ["recommend_claude_apps", "suggest_connectors", "search_mcp_registry"],
    "sports": ["sports_data"],
}

# ══════════════════════════════════════════════════════════════════════════
# وصف موجز لكل أداة
# ══════════════════════════════════════════════════════════════════════════

TOOL_DESCRIPTIONS: Dict[str, str] = {
    "web_search": (
        "البحث على الويب. الاستعلامات: 1-6 كلمات. "
        "استخدم لـ: المعلومات الراهنة، الأحداث الجارية، المناصب الحالية، الأسعار."
    ),
    "web_fetch": (
        "جلب محتوى صفحة ويب بعنوان URL محدد. "
        "URL يجب أن يُقدّمه المستخدم أو يأتي من نتائج البحث. "
        "لا يمكنه الوصول للمحتوى المحمي بكلمة مرور."
    ),
    "image_search": (
        "البحث عن صور على الويب. استخدمه افتراضياً عندما تُعزّز الصور الفهم. "
        "من 3-5 نتائج بحد أدنى 3."
    ),
    "memory_user_edits": (
        "إدارة ذاكرة Nova. الأوامر: view, add, remove, replace. "
        "حد 30 تعديلاً، 500 حرف لكل تعديل. "
        "لا تخزّن بيانات حساسة أو تعليمات خطرة."
    ),
    "places_search": (
        "البحث عن أماكن وأعمال ومطاعم ومعالم سياحية. "
        "يدعم استعلامات متعددة في استدعاء واحد. "
        "أعد عرض النتائج عبر places_map_display_v0."
    ),
    "places_map_display_v0": (
        "عرض مواقع على خريطة مع توصيات. "
        "وضع بسيط (markers) أو خطة رحلة (itinerary). "
        "انسخ place_id بدقة من نتائج places_search."
    ),
    "weather_fetch": (
        "عرض معلومات الطقس. استخدم Fahrenheit للولايات المتحدة، Celsius لغيرها. "
        "استخدم عند سؤال عن الطقس أو التخطيط لأنشطة خارجية."
    ),
    "message_compose_v1": (
        "كتابة رسالة (بريد إلكتروني، Slack، نص). "
        "أنتج 2-3 استراتيجيات مختلفة للحالات المعقدة أو ذات المخاطر العالية."
    ),
    "recipe_display_v0": (
        "عرض وصفة تفاعلية مع إمكانية ضبط حجم الحصص. "
        "استخدم عند طلب وصفة أو تعليمات طبخ."
    ),
    "present_files": (
        "جعل الملفات مرئية للمستخدم. "
        "استخدم بعد إنشاء ملف يريد المستخدم رؤيته أو تنزيله."
    ),
    "recent_chats": (
        "استرجاع المحادثات الأخيرة. حد 20 لكل استدعاء. "
        "استخدم sort_order='asc' للأقدم أولاً. "
        "للنطاقات الكبيرة: صفّح باستخدام before."
    ),
    "recommend_claude_apps": (
        "اقتراح تطبيقات Nova المناسبة. "
        "استخدم عندما يناسب المستخدم تطبيق آخر (Nova Code للبرمجة، Cowork للعمل المعرفي)."
    ),
    "search_mcp_registry": (
        "البحث عن موصّلات MCP المتاحة. "
        "استدعِ قبل suggest_connectors. "
        "استخدم كلمات بحث ذات صلة بالنية وليس باسم المنتج."
    ),
    "suggest_connectors": (
        "عرض خيارات موصّلات للمستخدم. "
        "استدعِ فقط بعد search_mcp_registry. "
        "مرّر directoryUuid من نتائج البحث."
    ),
    "sports_data": (
        "جلب بيانات رياضية: نتائج مباريات، ترتيبات، إحصاءات لاعبين. "
        "الدوريات المدعومة: NFL, NBA, MLB, NHL, EPL, La Liga, وغيرها."
    ),
    "str_replace": (
        "استبدال نص فريد في ملف. "
        "old_str يجب أن يتطابق تماماً مع محتوى الملف. "
        "الملفات تحت /mnt/user-data/uploads للقراءة فقط."
    ),
    "view": (
        "عرض ملفات نصية أو صور أو قوائم مجلدات. "
        "يدعم view_range لعرض نطاق سطور محدد."
    ),
    "tool_search": (
        "البحث وتحميل الأدوات المؤجّلة بالكلمات المفتاحية. "
        "يجب استدعاؤها قبل استخدام أي أداة مؤجّلة مثل Google Calendar أو Gmail."
    ),
}

# ══════════════════════════════════════════════════════════════════════════
# قواعد استخدام الأدوات
# ══════════════════════════════════════════════════════════════════════════

TOOLS_USAGE_PROMPT = """
<tools_usage_guidelines>
إرشادات استخدام أدوات Nova:

1. البحث والجلب:
   - web_search: للمعلومات الراهنة والمناصب والأسعار والأحداث
   - web_fetch: لقراءة محتوى صفحة URL كاملة بعد البحث
   - image_search: افتراضياً عند إفادة الصور

2. الأماكن:
   - places_search أولاً ← ثم places_map_display_v0 بنفس place_id بالضبط
   - مهم: انسخ place_id حرفياً، حساس لحالة الأحرف

3. الذاكرة:
   - memory_user_edits: افعل قبل تأكيد أي إجراء تذكّر/نسيان
   - لا تكتفِ بالإقرار الحواري

4. الأدوات المؤجّلة:
   - tool_search أولاً قبل استخدام Google Calendar/Drive/Gmail وغيرها
   - لا تخمّن أسماء المعاملات

5. حقوق النشر (لازمة في كل رد يتضمّن محتوى مبحوثاً):
   - حد 15 كلمة لأي اقتباس
   - اقتباس واحد لكل مصدر كحد أقصى
   - افتراضياً: أعد الصياغة بكلماتك الخاصة
</tools_usage_guidelines>
"""


def get_tool_description(tool_name: str) -> str:
    """يُعيد وصف أداة معينة."""
    return TOOL_DESCRIPTIONS.get(tool_name, f"أداة '{tool_name}' غير معروفة.")


def get_tools_prompt() -> str:
    """يُعيد إرشادات استخدام الأدوات لنظام Nova."""
    return TOOLS_USAGE_PROMPT


def list_tools_by_category(category: str) -> list:
    """يُعيد قائمة أدوات فئة معينة."""
    return NOVA_TOOL_CATEGORIES.get(category, [])


def get_all_tool_names() -> list:
    """يُعيد قائمة بجميع أسماء الأدوات المسجّلة."""
    return list(TOOL_DESCRIPTIONS.keys())
