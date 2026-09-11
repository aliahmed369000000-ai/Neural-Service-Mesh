"""
اختبارات humanize_fallback_error() — يتحقق أن الدالة المسؤولة عن تحويل
FallbackResult.error الخام (يحتوي على رموز HTTP ونصوص استثناءات مكتبات
خام من كل مزوّد جُرِّب) إلى رسالة عربية آمنة لا تسرّب أي تفاصيل تقنية
عند عرضها مباشرة للمستخدم النهائي في صفحات مثل الترجمة والسرد الإبداعي
(ui_pages/translate.py, ui_pages/fable.py) وai/fable_engine.py.
"""
import unittest

from ai.llm_fallback import humanize_fallback_error


class HumanizeFallbackErrorTests(unittest.TestCase):
    def test_empty_or_none_returns_empty_string(self):
        self.assertEqual(humanize_fallback_error(None), "")
        self.assertEqual(humanize_fallback_error(""), "")

    def test_does_not_leak_raw_http_codes_or_exception_text(self):
        raw = (
            "فشلت كل المزوّدين: ["
            "'groq:err(429 Too Many Requests: rate_limit_exceeded)', "
            "'cerebras:err(503 Service Unavailable)', "
            "'anthropic:err(401 invalid x-api-key)'"
            "]"
        )
        friendly = humanize_fallback_error(raw)
        leaked_tokens = (
            "429", "503", "401", "groq:err", "cerebras:err", "anthropic:err",
            "Too Many Requests", "Service Unavailable", "invalid x-api-key",
        )
        for token in leaked_tokens:
            self.assertNotIn(token, friendly, f"تسرّب '{token}' في الرسالة المعروضة للمستخدم")

    def test_returns_non_empty_arabic_message_for_any_error(self):
        friendly = humanize_fallback_error("أي نص خطأ عشوائي 502 Bad Gateway")
        self.assertTrue(friendly.strip())
        self.assertIn("رد احتياطي", friendly)


if __name__ == "__main__":
    unittest.main()
