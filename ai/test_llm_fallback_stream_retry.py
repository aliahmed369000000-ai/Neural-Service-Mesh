"""
اختبارات لإعادة المحاولة على الأخطاء العابرة (429/502/503/504) داخل
LLMFallback.generate_stream() — نفس منطق generate() لكن مُطبَّق فقط قبل
بث أي قطعة فعلية لكل مزوّد (انظر التوثيق أعلى generate_stream في
ai/llm_fallback.py). لا شبكة حقيقية ولا مفاتيح API — كل شيء محاكاة عبر
استبدال _stream_provider و_build_provider_chain.
"""
import time
import unittest

from ai.llm_fallback import LLMFallback, Provider


class _NoSleep:
    """يستبدل time.sleep مؤقتاً لتسريع الاختبار (الإعادة تنتظر 1.5ث/3ث فعلياً)."""

    def __enter__(self):
        self._orig = time.sleep
        time.sleep = lambda s: None
        return self

    def __exit__(self, *a):
        time.sleep = self._orig


def _make_fb():
    fb = LLMFallback.__new__(LLMFallback)
    fb.ckg = None
    fb.max_tokens = 100
    fb.temperature = 0.4
    fb.timeout = 5
    fb._model_key = None
    fb._openrouter_models = []
    fb._provider = Provider.GROQ
    fb._api_key = "fake"
    fb._model = "fake-model"
    fb._failed_until = {}
    return fb


class GenerateStreamRetryTests(unittest.TestCase):
    def _run(self, fb, chain, stream_plan):
        fb._build_provider_chain = lambda: chain
        calls = {}

        def fake_stream_provider(prov, key, mdl, query, history, sp):
            calls[prov] = calls.get(prov, 0) + 1
            for piece in stream_plan(prov, calls[prov]):
                if isinstance(piece, Exception):
                    raise piece
                yield piece

        fb._stream_provider = fake_stream_provider
        fb._call_provider = lambda *a, **kw: self.fail(
            "لا يجب استدعاء _call_provider لمزوّدين قادرين على streaming"
        )
        with _NoSleep():
            got = list(fb.generate_stream("سؤال اختبار", history=[]))
        return got, calls

    def test_retries_once_then_succeeds_on_transient_429_before_any_chunk(self):
        fb = _make_fb()

        def plan(prov, i):
            if prov == Provider.GROQ:
                return [Exception("429 Too Many Requests")] if i == 1 else ["مرحباً"]
            return ["غير متوقع"]

        got, calls = self._run(
            fb, [(Provider.GROQ, "k", "m"), (Provider.CEREBRAS, "k2", "m2")], plan
        )
        self.assertEqual(got, ["مرحباً"])
        self.assertEqual(calls[Provider.GROQ], 2)
        self.assertNotIn(Provider.CEREBRAS, calls)
        self.assertEqual(fb._provider, Provider.GROQ)
        self.assertNotIn(Provider.GROQ, fb._failed_until)

    def test_gives_up_after_two_retries_and_switches_provider(self):
        fb = _make_fb()

        def plan(prov, i):
            if prov == Provider.GROQ:
                return [Exception("503 Service Unavailable")]
            if prov == Provider.CEREBRAS:
                return ["رد احتياطي"]
            return []

        got, calls = self._run(
            fb, [(Provider.GROQ, "k", "m"), (Provider.CEREBRAS, "k2", "m2")], plan
        )
        self.assertEqual(got, ["رد احتياطي"])
        self.assertEqual(calls[Provider.GROQ], 3)  # محاولة أولى + إعادتان
        self.assertEqual(calls[Provider.CEREBRAS], 1)
        self.assertEqual(fb._provider, Provider.CEREBRAS)
        self.assertIn(Provider.GROQ, fb._failed_until)

    def test_no_retry_when_failure_happens_after_partial_chunks(self):
        fb = _make_fb()

        def plan(prov, i):
            if prov == Provider.GROQ:
                return ["جزء1", "جزء2", Exception("502 Bad Gateway")]
            if prov == Provider.CEREBRAS:
                return ["تكملة"]
            return []

        got, calls = self._run(
            fb, [(Provider.GROQ, "k", "m"), (Provider.CEREBRAS, "k2", "m2")], plan
        )
        self.assertEqual(got, ["جزء1", "جزء2", "تكملة"])
        self.assertEqual(calls[Provider.GROQ], 1, "لا إعادة محاولة بعد بث جزئي")
        self.assertEqual(calls[Provider.CEREBRAS], 1)

    def test_no_retry_for_non_transient_error(self):
        fb = _make_fb()

        def plan(prov, i):
            if prov == Provider.GROQ:
                return [Exception("400 Bad Request: invalid api key")]
            if prov == Provider.CEREBRAS:
                return ["رد سليم"]
            return []

        got, calls = self._run(
            fb, [(Provider.GROQ, "k", "m"), (Provider.CEREBRAS, "k2", "m2")], plan
        )
        self.assertEqual(got, ["رد سليم"])
        self.assertEqual(calls[Provider.GROQ], 1)


if __name__ == "__main__":
    unittest.main()
