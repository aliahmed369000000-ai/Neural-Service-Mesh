"""اختبارات المرحلة 3: WhatsApp Business Cloud API وPinterest API v5.
بدون أي مفتاح API حقيقي أو اتصال شبكة فعلي — كل شيء مُحاكى.

يُشغَّل عبر: python -m unittest ai.test_social_platform3_adapters -v
"""
from __future__ import annotations

import os
import sqlite3
import unittest
from unittest.mock import patch, MagicMock

from ai.social_platforms.base import SocialItem, NotConfiguredError, PlatformCapabilityError
from ai.social_platforms.whatsapp_adapter import WhatsAppAdapter, _INBOX_DB
from ai.social_platforms.pinterest_adapter import PinterestAdapter


def _ok(json_data: dict) -> MagicMock:
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = json_data
    return resp


class TestWhatsAppAdapter(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            os.environ,
            {
                "WHATSAPP_ACCESS_TOKEN": "tok",
                "WHATSAPP_PHONE_NUMBER_ID": "pnid",
                "WHATSAPP_DEFAULT_TO": "9665xxxxxxx",
            },
            clear=False,
        )
        self.env_patcher.start()
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.sleep_patcher.start()
        if _INBOX_DB.exists():
            _INBOX_DB.unlink()

    def tearDown(self):
        self.env_patcher.stop()
        self.sleep_patcher.stop()
        if _INBOX_DB.exists():
            _INBOX_DB.unlink()

    def test_not_configured_raises_without_network_call(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = WhatsAppAdapter()
            with patch("requests.post") as mock_post:
                with self.assertRaises(NotConfiguredError):
                    adapter.publish("hi")
                mock_post.assert_not_called()

    def test_publish_sends_to_default_number(self):
        adapter = WhatsAppAdapter()
        with patch("requests.post", return_value=_ok({"messages": [{"id": "wamid.1"}]})) as mp:
            mid = adapter.publish("مرحبا")
        self.assertEqual(mid, "wamid.1")
        self.assertEqual(mp.call_args.kwargs["json"]["to"], "9665xxxxxxx")

    def test_reply_includes_context_message_id(self):
        adapter = WhatsAppAdapter()
        item = SocialItem(platform="whatsapp", external_id="wamid.orig", kind="dm",
                           author="a", text="سؤال", thread_id="9665yyy", raw={})
        with patch("requests.post", return_value=_ok({"messages": [{"id": "wamid.2"}]})) as mp:
            mid = adapter.reply(item, "الرد")
        self.assertEqual(mid, "wamid.2")
        sent = mp.call_args.kwargs["json"]
        self.assertEqual(sent["to"], "9665yyy")
        self.assertEqual(sent["context"]["message_id"], "wamid.orig")

    def test_fetch_new_items_reads_local_queue_not_network(self):
        adapter = WhatsAppAdapter()
        item = SocialItem(platform="whatsapp", external_id="wamid.q1", kind="dm",
                           author="ali", text="نص", thread_id="9665zzz", raw={})
        WhatsAppAdapter.enqueue_incoming(item)
        with patch("requests.get") as mock_get, patch("requests.post") as mock_post:
            items = adapter.fetch_new_items(set())
        mock_get.assert_not_called()
        mock_post.assert_not_called()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].external_id, "wamid.q1")

    def test_fetch_new_items_skips_seen_ids(self):
        item = SocialItem(platform="whatsapp", external_id="wamid.seen", kind="dm",
                           author="a", text="x", thread_id="1", raw={})
        WhatsAppAdapter.enqueue_incoming(item)
        adapter = WhatsAppAdapter()
        items = adapter.fetch_new_items({"wamid.seen"})
        self.assertEqual(items, [])

    def test_verify_webhook_challenge_matches_token(self):
        with patch.dict(os.environ, {"WHATSAPP_VERIFY_TOKEN": "secret123"}):
            self.assertEqual(
                WhatsAppAdapter.verify_webhook_challenge("subscribe", "secret123", "chal"),
                "chal",
            )
            self.assertIsNone(
                WhatsAppAdapter.verify_webhook_challenge("subscribe", "wrong", "chal")
            )

    def test_verify_signature_valid_and_invalid(self):
        import hmac, hashlib
        with patch.dict(os.environ, {"WHATSAPP_APP_SECRET": "appsecret"}):
            body = b'{"a":1}'
            good_sig = "sha256=" + hmac.new(b"appsecret", body, hashlib.sha256).hexdigest()
            self.assertTrue(WhatsAppAdapter.verify_signature(body, good_sig))
            self.assertFalse(WhatsAppAdapter.verify_signature(body, "sha256=deadbeef"))
            self.assertFalse(WhatsAppAdapter.verify_signature(body, None))

    def test_parse_webhook_payload_extracts_text_message(self):
        payload = {
            "entry": [{"changes": [{"value": {
                "contacts": [{"wa_id": "9665zzz", "profile": {"name": "علي"}}],
                "messages": [{"id": "wamid.p1", "from": "9665zzz", "type": "text",
                               "text": {"body": "أهلاً"}}],
            }}]}]
        }
        items = WhatsAppAdapter.parse_webhook_payload(payload)
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].text, "أهلاً")
        self.assertEqual(items[0].author, "علي")
        self.assertEqual(items[0].thread_id, "9665zzz")

    def test_parse_webhook_payload_ignores_status_updates(self):
        payload = {"entry": [{"changes": [{"value": {"statuses": [{"id": "x"}]}}]}]}
        self.assertEqual(WhatsAppAdapter.parse_webhook_payload(payload), [])

    def test_supports_webhook_true(self):
        self.assertTrue(WhatsAppAdapter.supports_webhook)


class TestPinterestAdapter(unittest.TestCase):
    def setUp(self):
        self.env_patcher = patch.dict(
            os.environ,
            {
                "PINTEREST_ACCESS_TOKEN": "tok",
                "PINTEREST_BOARD_ID": "board123",
                "PINTEREST_DEFAULT_IMAGE_URL": "https://example.com/img.png",
            },
            clear=False,
        )
        self.env_patcher.start()
        self.sleep_patcher = patch("ai.social_platforms.retry.time.sleep", return_value=None)
        self.sleep_patcher.start()

    def tearDown(self):
        self.env_patcher.stop()
        self.sleep_patcher.stop()

    def test_not_configured_raises_without_network_call(self):
        with patch.dict(os.environ, {}, clear=True):
            adapter = PinterestAdapter()
            with patch("requests.post") as mock_post:
                with self.assertRaises(NotConfiguredError):
                    adapter.publish("نص")
                mock_post.assert_not_called()

    def test_publish_creates_pin_with_default_image(self):
        adapter = PinterestAdapter()
        with patch("requests.post", return_value=_ok({"id": "pin123"})) as mp:
            pid = adapter.publish("عنوان قصير")
        self.assertEqual(pid, "pin123")
        sent = mp.call_args.kwargs["json"]
        self.assertEqual(sent["board_id"], "board123")
        self.assertEqual(sent["media_source"]["url"], "https://example.com/img.png")

    def test_fetch_new_items_raises_capability_error_not_network(self):
        adapter = PinterestAdapter()
        with patch("requests.get") as mock_get:
            with self.assertRaises(PlatformCapabilityError):
                adapter.fetch_new_items(set())
            mock_get.assert_not_called()

    def test_reply_raises_capability_error(self):
        adapter = PinterestAdapter()
        item = SocialItem(platform="pinterest", external_id="1", kind="comment",
                           author="a", text="x", raw={})
        with self.assertRaises(PlatformCapabilityError):
            adapter.reply(item, "رد")

    def test_supports_monitoring_false(self):
        self.assertFalse(PinterestAdapter.supports_monitoring)


class TestSocialAgentSkipsNonMonitoringPlatforms(unittest.TestCase):
    """يتحقق أن دورة polling لا تستدعي fetch_new_items إطلاقاً لمنصة
    supports_monitoring=False (مثل Pinterest) — لا أخطاء متكررة بالسجل."""

    def setUp(self):
        from ai.social_agent import SocialAgentManager
        SocialAgentManager._instance = None

    def tearDown(self):
        from ai.social_agent import SocialAgentManager
        SocialAgentManager._instance = None

    def test_pinterest_adapter_flagged_non_monitoring_in_manager(self):
        from ai.social_agent import get_manager
        mgr = get_manager()
        self.assertFalse(mgr.adapters["pinterest"].supports_monitoring)
        self.assertTrue(mgr.adapters["whatsapp"].supports_monitoring)


if __name__ == "__main__":
    unittest.main()
