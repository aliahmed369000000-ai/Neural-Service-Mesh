"""
محول WhatsApp — عبر WhatsApp Business Cloud API الرسمي (مباشرة من Meta،
بدون BSP وسيط). يتطلب: WHATSAPP_ACCESS_TOKEN، WHATSAPP_PHONE_NUMBER_ID،
WHATSAPP_DEFAULT_TO (رقم الوجهة الافتراضي لـpublish — راجع القيد أدناه).

⚠️ قيدان حقيقيان من المنصة نفسها (موثّقان هنا بدل التحايل عليهما):

1. لا يوجد مفهوم "نشر عام" على WhatsApp كما في Twitter/Facebook — هو
   منصة مراسلة بحتة. publish(text) هنا يرسل رسالة نصية لرقم افتراضي واحد
   (WHATSAPP_DEFAULT_TO، مثلاً رقم مجتمع/قناة إشعارات داخلية)، وليس بثاً
   علنياً. من خارج نافذة الـ24 ساعة يتطلب Meta رسائل قوالب (templates)
   معتمدة مسبقاً بدل نص حر — هذا الإصدار يرسل نصاً حراً (يفترض محادثة
   نشطة/نافذة 24 ساعة مفتوحة)؛ رسائل القوالب خارج نطاق هذه المرحلة.

2. **لا يوجد أي REST endpoint من Meta لسرد/سحب الرسائل الواردة** (على
   عكس Telegram getUpdates أو Discord GET messages) — الرسائل الواردة
   تصل *فقط* عبر webhook (POST من Meta). لذلك fetch_new_items هنا لا
   "يستطلع" Meta فعلياً (لا يوجد ما يُستطلع)، بل يُفرّغ طابوراً محلياً
   (sqlite) يُغذّيه api_server.py عند وصول كل webhook — وهو أسلوب حقيقي
   وليس تلفيقاً: البيانات فعلية من Meta، فقط طريقة الوصول لها (دفع محلي
   بدل سحب) مختلفة عن بقية المحولات. supports_webhook=True دائماً؛ لا
   يوجد وضع polling بديل حقيقي لهذه المنصة تحديداً (خلافاً لتيليجرام).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
from pathlib import Path
from typing import List, Optional

import requests

from .base import PlatformAdapter, SocialItem
from .retry import with_retry

API_BASE = "https://graph.facebook.com/v21.0"

ROOT = Path(__file__).resolve().parent.parent.parent
_INBOX_DB = ROOT / "memory" / "whatsapp_inbox.db"
_INBOX_DB.parent.mkdir(exist_ok=True)


def _inbox_db() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_INBOX_DB), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inbox (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT UNIQUE, wa_id TEXT, author TEXT,
            text TEXT, received_at TEXT
        )""")
    conn.commit()
    return conn


class WhatsAppAdapter(PlatformAdapter):
    platform_id = "whatsapp"
    required_env = ["WHATSAPP_ACCESS_TOKEN", "WHATSAPP_PHONE_NUMBER_ID", "WHATSAPP_DEFAULT_TO"]
    supports_webhook = True

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {os.environ['WHATSAPP_ACCESS_TOKEN']}",
            "Content-Type": "application/json",
        }

    def _messages_url(self) -> str:
        return f"{API_BASE}/{os.environ['WHATSAPP_PHONE_NUMBER_ID']}/messages"

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        r = requests.post(
            self._messages_url(), headers=self._headers(),
            json={
                "messaging_product": "whatsapp",
                "to": os.environ["WHATSAPP_DEFAULT_TO"],
                "type": "text", "text": {"body": text},
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["messages"][0]["id"]

    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        """يُفرّغ الطابور المحلي الذي غذّاه webhook — لا استطلاع فعلي لـMeta
        (راجع القيد رقم 2 بأعلى الملف). لا يتطلب بيانات اعتماد لأن القراءة
        محلية بحتة؛ لكن بدون WHATSAPP_ACCESS_TOKEN لن يصل أي webhook أصلاً
        (فشل تفعيله لدى Meta)، فالطابور سيبقى فارغاً بصمت — سلوك متوقّع
        وليس خطأً يستحق NotConfiguredError."""
        items: List[SocialItem] = []
        with _inbox_db() as c:
            rows = c.execute(
                "SELECT message_id, wa_id, author, text FROM inbox ORDER BY id ASC"
            ).fetchall()
        for message_id, wa_id, author, text in rows:
            if message_id in since_ids:
                continue
            items.append(SocialItem(
                platform="whatsapp", external_id=message_id, kind="dm",
                author=author or wa_id, text=text, thread_id=wa_id,
                raw={"message_id": message_id, "wa_id": wa_id},
            ))
        return items

    @with_retry()
    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        r = requests.post(
            self._messages_url(), headers=self._headers(),
            json={
                "messaging_product": "whatsapp",
                "to": item.thread_id,
                "type": "text", "text": {"body": text},
                "context": {"message_id": item.external_id},
            },
            timeout=30,
        )
        r.raise_for_status()
        return r.json()["messages"][0]["id"]

    # ── دعم webhook: تحقق، تحليل، وتغذية الطابور المحلي ───────────────────
    @staticmethod
    def verify_webhook_challenge(mode: Optional[str], token: Optional[str],
                                  challenge: Optional[str]) -> Optional[str]:
        """التحقق الأولي عند ربط الـwebhook بلوحة Meta (GET بـhub.mode/
        hub.verify_token/hub.challenge) — نفس آلية whatsapp_gateway
        الموجودة أصلاً بالمشروع لبوابة Vercel المنفصلة."""
        expected = os.environ.get("WHATSAPP_VERIFY_TOKEN", "").strip()
        if not expected:
            return None
        if mode == "subscribe" and token == expected:
            return challenge
        return None

    @staticmethod
    def verify_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
        """يتحقق من توقيع X-Hub-Signature-256 الذي ترسله Meta مع كل POST
        (HMAC-SHA256 على الجسم الخام باستخدام WHATSAPP_APP_SECRET) — هذا
        هو أسلوب Meta الرسمي للتأكد أن الطلب فعلاً منها، مختلف عن secret
        بالمسار (أسلوب تيليجرام) لأن Meta لا تدعم secret_token مماثلاً."""
        app_secret = os.environ.get("WHATSAPP_APP_SECRET", "").strip()
        if not app_secret or not signature_header or "=" not in signature_header:
            return False
        _, _, sig = signature_header.partition("=")
        expected = hmac.new(app_secret.encode(), raw_body, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, sig)

    @staticmethod
    def parse_webhook_payload(payload: dict) -> List[SocialItem]:
        """يستخرج كل الرسائل النصية الواردة من حمولة webhook (قد تضم أكثر
        من رسالة نظرياً) بنفس بنية extract_incoming_message المستخدمة
        أصلاً بـai/whatsapp/whatsapp_client.py، ويعيدها كـSocialItem موحّد."""
        items: List[SocialItem] = []
        try:
            for entry in payload.get("entry", []):
                for change in entry.get("changes", []):
                    value = change.get("value", {})
                    contacts = {c.get("wa_id"): c.get("profile", {}).get("name")
                                for c in value.get("contacts", [])}
                    for msg in value.get("messages", []) or []:
                        if msg.get("type") != "text":
                            continue
                        wa_id = msg.get("from")
                        text = msg.get("text", {}).get("body")
                        message_id = msg.get("id")
                        if not wa_id or text is None or not message_id:
                            continue
                        items.append(SocialItem(
                            platform="whatsapp", external_id=message_id, kind="dm",
                            author=contacts.get(wa_id, wa_id), text=text,
                            thread_id=wa_id, raw=msg,
                        ))
        except (KeyError, TypeError, AttributeError):
            return items
        return items

    @staticmethod
    def enqueue_incoming(item: SocialItem) -> None:
        """يخزّن العنصر بالطابور المحلي (احتياطي/سجل تدقيق حتى لو عُولج
        العنصر فوراً عبر ingest_webhook_item) — INSERT OR IGNORE لتفادي
        التكرار عبر UNIQUE(message_id)، ثم تقليم الطابور لآخر 1000 عنصر."""
        with _inbox_db() as c:
            c.execute(
                "INSERT OR IGNORE INTO inbox (message_id, wa_id, author, text, received_at) "
                "VALUES (?,?,?,?, datetime('now'))",
                (item.external_id, item.thread_id, item.author, item.text),
            )
            c.execute(
                "DELETE FROM inbox WHERE id NOT IN "
                "(SELECT id FROM inbox ORDER BY id DESC LIMIT 1000)"
            )
