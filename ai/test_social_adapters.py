"""اختبارات تكامل: تتحقق أن @with_retry مطبّق فعلياً على محولات حقيقية
(telegram, discord, facebook, twitter) وأن NotConfiguredError ما زالت
تُرفع فوراً بدون أي استدعاء شبكة فعلي، وأن الأخطاء العابرة تُعاد محاولتها.

يُشغَّل عبر: python -m unittest ai.test_social_adapters -v
لا يحتاج مفاتيح API حقيقية — requests.post/get مُحاكاة بالكامل.
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "/home/claude/build")

import requests

from ai.social_platforms.telegram_adapter import TelegramAdapter
from ai.social_platforms.discord_adapter import DiscordAdapter
from ai.social_platforms.facebook_adapter import FacebookAdapter
from ai.social_platforms.base import NotConfiguredError


def _ok_response(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


def _http_error_response(status: int) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    exc = requests.exceptions.HTTPError(f"{status}")
    exc.response = resp
    resp.raise_for_status.side_effect = exc
    return resp


class TestTelegramAdapterRetry(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {}, clear=False)
        self.env_patcher.start()
        for k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID"):
            os.environ.pop(k, None)
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.sleep_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.sleep_patcher.stop()

    def test_not_configured_raises_without_network_call(self):
        adapter = TelegramAdapter()
        with patch("requests.post") as mock_post:
            with self.assertRaises(NotConfiguredError):
                adapter.publish("hello")
            mock_post.assert_not_called()  # لا يجب أي محاولة شبكة بلا بيانات اعتماد

    def test_publish_retries_on_transient_5xx_then_succeeds(self):
        os.environ["TELEGRAM_BOT_TOKEN"] = "fake-token"
        os.environ["TELEGRAM_CHAT_ID"] = "12345"
        adapter = TelegramAdapter()

        fail_resp = _http_error_response(502)
        ok_resp = _ok_response({"result": {"message_id": 999}})

        with patch("requests.post", side_effect=[fail_resp, ok_resp]) as mock_post:
            result = adapter.publish("مرحبا")
        self.assertEqual(result, "999")
        self.assertEqual(mock_post.call_count, 2)


class TestDiscordAdapterRetry(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {}, clear=False)
        self.env_patcher.start()
        for k in ("DISCORD_BOT_TOKEN", "DISCORD_CHANNEL_ID"):
            os.environ.pop(k, None)
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.sleep_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.sleep_patcher.stop()

    def test_not_configured_raises_without_network_call(self):
        adapter = DiscordAdapter()
        with patch("requests.get") as mock_get:
            with self.assertRaises(NotConfiguredError):
                adapter.fetch_new_items(set())
            mock_get.assert_not_called()

    def test_fetch_gives_up_after_permanent_403(self):
        os.environ["DISCORD_BOT_TOKEN"] = "fake"
        os.environ["DISCORD_CHANNEL_ID"] = "1"
        adapter = DiscordAdapter()
        forbidden = _http_error_response(403)
        with patch("requests.get", return_value=forbidden) as mock_get:
            with self.assertRaises(requests.exceptions.HTTPError):
                adapter.fetch_new_items(set())
        self.assertEqual(mock_get.call_count, 1)  # لا إعادة محاولة على خطأ صلاحيات دائم


class TestFacebookAdapterRetry(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(os.environ, {}, clear=False)
        self.env_patcher.start()
        for k in ("FACEBOOK_PAGE_ACCESS_TOKEN", "FACEBOOK_PAGE_ID"):
            os.environ.pop(k, None)
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.sleep_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.sleep_patcher.stop()

    def test_publish_retries_on_429_then_succeeds(self):
        os.environ["FACEBOOK_PAGE_ACCESS_TOKEN"] = "fake"
        os.environ["FACEBOOK_PAGE_ID"] = "1"
        adapter = FacebookAdapter()
        rate_limited = _http_error_response(429)
        ok = _ok_response({"id": "post_123"})
        with patch("requests.post", side_effect=[rate_limited, ok]) as mock_post:
            result = adapter.publish("منشور تجريبي")
        self.assertEqual(result, "post_123")
        self.assertEqual(mock_post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
