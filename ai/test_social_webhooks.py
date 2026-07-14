"""اختبارات دعم webhook (المرحلة 2): تحليل تحديثات تيليجرام الموحّد،
set_webhook/delete_webhook/webhook_info، ودمج SocialAgentManager
(ingest_webhook_item/enable_webhook/disable_webhook) — بدون أي مفتاح
API حقيقي أو اتصال شبكة فعلي (requests مُحاكاة بالكامل).

يُشغَّل عبر: python -m unittest ai.test_social_webhooks -v
"""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch, MagicMock

from ai.social_platforms.telegram_adapter import TelegramAdapter
from ai.social_platforms.base import SocialItem


def _ok(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


class TestParseUpdate(unittest.TestCase):
    """_parse_update يجب أن ينتج نفس شكل SocialItem سواء استُدعي من
    fetch_new_items (polling) أو مباشرة من حمولة webhook واردة."""

    def test_parses_message_update(self):
        upd = {
            "update_id": 555,
            "message": {
                "message_id": 10,
                "from": {"username": "ali"},
                "chat": {"id": -100123},
                "text": "مرحبا",
            },
        }
        item = TelegramAdapter._parse_update(upd)
        self.assertIsInstance(item, SocialItem)
        self.assertEqual(item.platform, "telegram")
        self.assertEqual(item.external_id, "555")
        self.assertEqual(item.text, "مرحبا")
        self.assertEqual(item.author, "ali")
        self.assertEqual(item.thread_id, "-100123")

    def test_parses_channel_post(self):
        upd = {"update_id": 1, "channel_post": {"message_id": 2, "chat": {"id": 9}, "text": "خبر"}}
        item = TelegramAdapter._parse_update(upd)
        self.assertEqual(item.text, "خبر")

    def test_ignores_non_text_update(self):
        upd = {"update_id": 2, "message": {"chat": {"id": 1}, "sticker": {}}}
        self.assertIsNone(TelegramAdapter._parse_update(upd))

    def test_ignores_update_without_id(self):
        self.assertIsNone(TelegramAdapter._parse_update({"message": {"text": "x", "chat": {"id": 1}}}))


class TestWebhookManagement(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "t-token", "TELEGRAM_CHAT_ID": "123"},
            clear=False,
        )
        self.env_patcher.start()
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.sleep_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.sleep_patcher.stop()

    def test_set_webhook_sends_secret_token_and_url(self):
        adapter = TelegramAdapter()
        with patch("requests.post", return_value=_ok({"ok": True, "result": True})) as mock_post:
            result = adapter.set_webhook("https://example.com/hook", secret_token="s3cr3t")
        self.assertTrue(result["ok"])
        called_kwargs = mock_post.call_args.kwargs
        self.assertEqual(called_kwargs["json"]["url"], "https://example.com/hook")
        self.assertEqual(called_kwargs["json"]["secret_token"], "s3cr3t")

    def test_set_webhook_raises_on_not_ok(self):
        adapter = TelegramAdapter()
        with patch("requests.post", return_value=_ok({"ok": False, "description": "bad url"})):
            with self.assertRaises(RuntimeError):
                adapter.set_webhook("http://not-https")

    def test_delete_webhook_calls_endpoint(self):
        adapter = TelegramAdapter()
        with patch("requests.post", return_value=_ok({"ok": True})) as mock_post:
            adapter.delete_webhook()
        self.assertIn("deleteWebhook", mock_post.call_args.args[0])

    def test_webhook_info_calls_get(self):
        adapter = TelegramAdapter()
        with patch("requests.get", return_value=_ok({"ok": True, "result": {"url": "x"}})) as mock_get:
            info = adapter.webhook_info()
        self.assertEqual(info["url"], "x")
        self.assertIn("getWebhookInfo", mock_get.call_args.args[0])

    def test_supports_webhook_flag(self):
        self.assertTrue(TelegramAdapter.supports_webhook)


class TestSocialAgentManagerWebhookIntegration(unittest.TestCase):
    """يتحقق أن ingest_webhook_item/enable_webhook/disable_webhook تتكامل
    بشكل صحيح مع إعدادات الوكيل، دون تشغيل أي خيط استطلاع فعلي."""

    def setUp(self):
        # عزل singleton لكل اختبار كي لا تتسرّب الحالة بين الاختبارات
        from ai.social_agent import SocialAgentManager
        SocialAgentManager._instance = None
        self.env_patcher = patch.dict(
            os.environ,
            {"TELEGRAM_BOT_TOKEN": "t-token", "TELEGRAM_CHAT_ID": "123"},
            clear=False,
        )
        self.env_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        from ai.social_agent import SocialAgentManager
        SocialAgentManager._instance = None

    def test_enable_webhook_rejects_unsupported_platform(self):
        from ai.social_agent import get_manager
        mgr = get_manager()
        with self.assertRaises(ValueError):
            mgr.enable_webhook("discord", "https://example.com/hook")

    def test_enable_webhook_registers_platform_and_calls_adapter(self):
        from ai.social_agent import get_manager, get_config
        mgr = get_manager()
        with patch("requests.post", return_value=_ok({"ok": True, "result": True})):
            mgr.enable_webhook("telegram", "https://example.com/hook", secret_token="s")
        self.assertIn("telegram", set(get_config("webhook_enabled_platforms", [])))

    def test_disable_webhook_unregisters_platform(self):
        from ai.social_agent import get_manager, get_config, set_config
        mgr = get_manager()
        set_config("webhook_enabled_platforms", ["telegram"])
        with patch("requests.post", return_value=_ok({"ok": True})):
            mgr.disable_webhook("telegram")
        self.assertNotIn("telegram", set(get_config("webhook_enabled_platforms", [])))

    def test_ingest_webhook_item_ignored_if_platform_not_enabled(self):
        from ai.social_agent import get_manager, set_config, get_recent_events
        mgr = get_manager()
        set_config("enabled_platforms", [])  # تيليجرام غير مفعّلة
        item = SocialItem(platform="telegram", external_id="999", kind="dm",
                           author="u", text="hi", thread_id="1", raw={})
        before = len(get_recent_events(100))
        mgr.ingest_webhook_item("telegram", item)
        after = len(get_recent_events(100))
        self.assertEqual(before, after)  # لا تسجيل لأن المنصة غير مفعّلة أصلاً


if __name__ == "__main__":
    unittest.main()
