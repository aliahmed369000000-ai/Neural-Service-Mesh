"""
اختبارات ai/godmode.py::route_query_verbose — التوجيه الحتمي (بدون LLM)
لاستعلامات "منسّق الوكلاء" إلى أنسب فئة/فئات في AGENT_CATEGORIES.

يغطي تحديداً إصلاح تطبيع الهمزة وإزالة "ال" التعريف في _query_words/
_keyword_scores: قبل الإصلاح كانت كلمة استعلام مثل "اتمتة" (بلا همزة)
لا تطابق عنوان فئة "الأتمتة" (بهمزة + "ال") إطلاقاً رغم التطابق الكامل
بالمعنى، فيسقط الاستعلام إلى الافتراضي العام بدل التوجيه الصحيح.
"""
from ai.godmode import route_query_verbose, _query_words, _normalize_hamza


class _Cat:
    def __init__(self, title, subtitle, quick_prompts=None):
        self.title = title
        self.subtitle = subtitle
        self.quick_prompts = quick_prompts or []


_CATEGORIES = {
    "automation": _Cat("الأتمتة", "أتمتة المهام المتكررة وسير العمل"),
    "assistant": _Cat("المساعد الشخصي", "مساعدة عامة في أي موضوع"),
    "coding": _Cat("الصيانة الذاتية للكود", "فحص وإصلاح الكود ورفعه"),
}


class TestQueryWordsNormalization:
    def test_hamza_variants_normalize_identically(self):
        """'أتمتة' (بهمزة) و'اتمتة' (بدون همزة) يجب أن تُطبَّعا لنفس الشكل."""
        assert _normalize_hamza("أتمتة") == _normalize_hamza("اتمتة") == "اتمتة"

    def test_definite_article_variant_added(self):
        """كلمة معرَّفة بـ'ال' (أطول من 3 أحرف) تُضيف نسخة مجرَّدة أيضاً."""
        words = _query_words("الأتمتة")
        assert "اتمتة" in words  # بعد تطبيع الهمزة وإزالة "ال"

    def test_short_word_not_stripped(self):
        """كلمة بـ3 أحرف بالضبط ('الم') تبقى كما هي فقط — شرط الاقتطاع
        len(w) > 3 لا ينطبق، فلا تُضاف نسخة مجرَّدة (كانت ستصبح 'م' بلا
        معنى)."""
        words = _query_words("الم")
        assert words == ["الم"]


class TestRouteQueryVerbose:
    def test_query_without_al_matches_category_title_with_al_and_hamza(self):
        """السيناريو الأساسي: استعلام بلا 'ال' وبهمزة مختلفة يجب أن يطابق
        فئة عنوانها معرَّف بـ'ال' وبهمزة أخرى — هذا كان يفشل قبل الإصلاح."""
        picked, method, scores = route_query_verbose(
            "ساعدني في اتمتة هذا الإجراء", _CATEGORIES,
            max_agents=2, use_llm_fallback=False,
        )
        assert picked == ["automation"]
        assert method == "keyword"
        assert scores["automation"] > 0

    def test_multi_word_query_still_ranks_correctly(self):
        picked, method, _ = route_query_verbose(
            "راجع خطة إطلاق ميزة جديدة من ناحية الأتمتة والتحليل والمخاطر",
            _CATEGORIES, max_agents=2, use_llm_fallback=False,
        )
        assert picked[0] == "automation"
        assert method == "keyword"

    def test_no_match_falls_back_to_default_assistant(self):
        picked, method, _ = route_query_verbose(
            "طقس اليوم في المدينة", _CATEGORIES,
            max_agents=2, use_llm_fallback=False,
        )
        assert picked == ["assistant"]
        assert method == "default"

    def test_empty_query_returns_empty(self):
        assert route_query_verbose("", _CATEGORIES) == ([], "empty", {})

    def test_empty_categories_returns_empty(self):
        assert route_query_verbose("اتمتة", {}) == ([], "empty", {})
