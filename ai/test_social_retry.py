"""اختبارات وحدة لطبقة الموثوقية retry/backoff في ai/social_platforms/retry.py.

يُشغَّل عبر: python -m unittest ai.test_social_retry -v
لا يحتاج مفاتيح API حقيقية — كل استدعاءات requests مُحاكاة (mocked).
"""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/home/claude/build")

import requests

from ai.social_platforms.retry import with_retry
from ai.social_platforms.base import NotConfiguredError


def _http_error(status: int, retry_after: str | None = None) -> requests.exceptions.HTTPError:
    resp = MagicMock()
    resp.status_code = status
    resp.headers = {"Retry-After": retry_after} if retry_after else {}
    exc = requests.exceptions.HTTPError(f"{status} error")
    exc.response = resp
    return exc


class TestWithRetry(unittest.TestCase):
    def setUp(self):
        # نلغي فعلياً وقت الانتظار الحقيقي حتى تكون الاختبارات سريعة
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.mock_sleep = self.sleep_patcher.start()

    def tearDown(self):
        self.sleep_patcher.stop()

    def test_success_first_try_no_retry(self):
        calls = {"n": 0}

        @with_retry(max_retries=3)
        def fn():
            calls["n"] += 1
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(calls["n"], 1)
        self.mock_sleep.assert_not_called()

    def test_retries_on_429_then_succeeds(self):
        calls = {"n": 0}

        @with_retry(max_retries=3, base_delay=0.01)
        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise _http_error(429)
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(calls["n"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_respects_retry_after_header(self):
        @with_retry(max_retries=1, base_delay=1.0)
        def fn():
            fn.calls = getattr(fn, "calls", 0) + 1
            if fn.calls == 1:
                raise _http_error(429, retry_after="7")
            return "ok"

        self.assertEqual(fn(), "ok")
        # يجب استخدام قيمة Retry-After (7.0) حرفياً بدل backoff الافتراضي
        self.mock_sleep.assert_called_once_with(7.0)

    def test_retries_on_5xx(self):
        calls = {"n": 0}

        @with_retry(max_retries=2, base_delay=0.01)
        def fn():
            calls["n"] += 1
            if calls["n"] < 2:
                raise _http_error(503)
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(calls["n"], 2)

    def test_no_retry_on_permanent_4xx(self):
        calls = {"n": 0}

        @with_retry(max_retries=3, base_delay=0.01)
        def fn():
            calls["n"] += 1
            raise _http_error(403)

        with self.assertRaises(requests.exceptions.HTTPError):
            fn()
        self.assertEqual(calls["n"], 1)  # لا إعادة محاولة على 403
        self.mock_sleep.assert_not_called()

    def test_no_retry_on_not_configured_error(self):
        calls = {"n": 0}

        @with_retry(max_retries=3, base_delay=0.01)
        def fn():
            calls["n"] += 1
            raise NotConfiguredError("بيانات اعتماد ناقصة")

        with self.assertRaises(NotConfiguredError):
            fn()
        self.assertEqual(calls["n"], 1)
        self.mock_sleep.assert_not_called()

    def test_retries_on_connection_error_then_raises_after_exhaustion(self):
        calls = {"n": 0}

        @with_retry(max_retries=2, base_delay=0.01)
        def fn():
            calls["n"] += 1
            raise requests.exceptions.ConnectionError("network down")

        with self.assertRaises(requests.exceptions.ConnectionError):
            fn()
        # محاولة أولى + إعادتان = 3 محاولات، ثم رفع الاستثناء الأخير
        self.assertEqual(calls["n"], 3)
        self.assertEqual(self.mock_sleep.call_count, 2)

    def test_timeout_is_retryable(self):
        calls = {"n": 0}

        @with_retry(max_retries=1, base_delay=0.01)
        def fn():
            calls["n"] += 1
            if calls["n"] == 1:
                raise requests.exceptions.Timeout("slow")
            return "ok"

        self.assertEqual(fn(), "ok")
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
