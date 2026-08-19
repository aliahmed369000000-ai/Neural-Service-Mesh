# -*- coding: utf-8 -*-
"""
agent_auto_reply.py — الرد التلقائي الذكي على إيميلات محددة

يتيح للوكلاء:
1. مراقبة صندوق الوارد بشكل دوري
2. تطبيق قواعد مطابقة (من مرسل معين، بموضوع محدد، بكلمات مفتاحية)
3. الرد التلقائي باستخدام AI (Groq/OpenAI/Gemini) أو قالب ثابت
4. علامة الإيميل كمقروء/أرشيف بعد الرد
5. سجل كل الردود التلقائية (audit log)

يعمل بـ Python stdlib + urllib فقط.
"""
import json
import os
import time
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable


class AutoReplyAgent:
    """وكيل الرد التلقائي — يراقب Gmail ويرد تلقائيًا حسب القواعد."""

    def __init__(self, gmail_access_token: str = None,
                 llm_api_key: str = None,
                 log_dir: str = None):
        """
        Args:
            gmail_access_token: Gmail API token (أو GMAIL_TOKEN env)
            llm_api_key: مفتاح Groq/OpenAI لتوليد ردود ذكية
            log_dir: مجلد حفظ سجل الردود
        """
        self.token = gmail_access_token or os.environ.get("GMAIL_TOKEN", "")
        self.llm_key = llm_api_key or os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        self.log_dir = Path(log_dir or os.environ.get("AUTO_REPLY_LOG_DIR", "/tmp/nsm_auto_reply"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "replies.jsonl"
        self.rules = []
        self.stop_event = threading.Event()
        self.replied_ids = set()  # منع الرد مرتين
        self.blacklist = set()  # عناوين مرفوضة — لن يرد عليها أبدًا

    @property
    def available(self) -> bool:
        return bool(self.token)

    def add_blacklist(self, email_or_domain: str) -> None:
        """
        إضافة عنوان أو نطاق إلى القائمة السوداء — لن يرد عليها أبدًا.

        Args:
            email_or_domain: عنوان كامل (user@example.com) أو نطاق (example.com)
        """
        self.blacklist.add(email_or_domain.lower().strip())

    def remove_blacklist(self, email_or_domain: str) -> bool:
        """إزالة عنوان/نطاق من القائمة السوداء."""
        addr = email_or_domain.lower().strip()
        if addr in self.blacklist:
            self.blacklist.discard(addr)
            return True
        return False

    def is_blacklisted(self, email_address: str) -> bool:
        """
        هل العنوان في القائمة السوداء؟ يدعم:
        - تطابق كامل: user@example.com
        - تطابق نطاق: example.com (يطابق user@example.com)
        - تطابق بادئة: *@domain.com
        """
        addr = email_address.lower().strip()

        # استخراج النطاق من العنوان
        if "@" in addr:
            domain = addr.split("@")[1]
            user_part = addr.split("@")[0]
        else:
            domain = addr
            user_part = ""

        # تطابق كامل
        if addr in self.blacklist:
            return True

        # تطابق نطاق
        if domain in self.blacklist:
            return True

        # تطابق *@domain
        if f"*@{domain}" in self.blacklist:
            return True

        # تطابق wildcard user@*
        if f"{user_part}@*" in self.blacklist:
            return True

        return False

    def add_rule(self, from_contains: str = None, subject_contains: str = None,
                 body_contains: str = None, reply_template: str = None,
                 reply_ai: bool = True, sender: str = None,
                 mark_read: bool = True, archive: bool = False) -> dict:
        """
        إضافة قاعدة رد تلقائي.

        Args:
            from_contains: الرد إذا كان المرسل يحتوي هذا النص
            subject_contains: الرد إذا كان الموضوع يحتوي هذا النص
            body_contains: الرد إذا كان المحتوى يحتوي هذا النص
            reply_template: قالب رد ثابت (يدعم {sender}, {subject}, {body})
            reply_ai: استخدام AI لتوليد رد (يحتاج llm_key)
            sender: عنوان المرسل للرد
            mark_read: وضع علامة مقروء بعد الرد
            archive: أرشفة الإيميل بعد الرد
        """
        rule = {
            "from_contains": from_contains,
            "subject_contains": subject_contains,
            "body_contains": body_contains,
            "reply_template": reply_template,
            "reply_ai": reply_ai,
            "sender": sender,
            "mark_read": mark_read,
            "archive": archive,
        }
        self.rules.append(rule)
        return rule

    def _http(self, url: str, method: str = "GET", body: dict = None,
              timeout: int = 30) -> tuple:
        """مساعد HTTP لـ Gmail API."""
        import urllib.request
        headers = {"Authorization": f"Bearer {self.token}"}
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.getcode(), json.loads(resp.read().decode())
        except Exception as e:
            return 0, str(e)

    def _get_gmail(self, path: str) -> tuple:
        """GET request لـ Gmail API."""
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/{path}"
        return self._http(url)

    def _post_gmail(self, path: str, body: dict) -> tuple:
        """POST request لـ Gmail API."""
        url = f"https://gmail.googleapis.com/gmail/v1/users/me/{path}"
        return self._http(url, method="POST", body=body)

    def _get_message(self, msg_id: str) -> dict:
        """جلب محتوى إيميل."""
        status, data = self._get_gmail(f"messages/{msg_id}")
        if status != 200:
            return {"error": str(data)}
        payload = data.get("payload", {})

        def _extract_headers(payload):
            headers = {}
            for h in payload.get("headers", []):
                headers[h["name"].lower()] = h["value"]
            return headers

        def _extract_body(p):
            body_data = p.get("body", {}).get("data", "")
            import base64
            if body_data:
                # Gmail uses URL-safe base64
                body_data = body_data.replace("-", "+").replace("_", "/")
                decoded = base64.urlsafe_b64decode(body_data).decode("utf-8", errors="replace")
                return decoded
            # Multi-part
            for part in p.get("parts", []):
                if part.get("mimeType", "").startswith("text/plain"):
                    return _extract_body(part)
            for part in p.get("parts", []):
                text = _extract_body(part)
                if text:
                    return text
            return ""

        headers = _extract_headers(payload)
        body_text = _extract_body(payload)

        return {
            "id": data.get("id"),
            "thread_id": data.get("threadId"),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "subject": headers.get("subject", ""),
            "date": headers.get("date", ""),
            "body": body_text[:2000],  # أول 2000 حرف
        }

    def _match_rule(self, msg: dict, rule: dict) -> bool:
        """هل الإيميل يطابق القاعدة؟"""
        if rule.get("from_contains"):
            if rule["from_contains"].lower() not in msg.get("from", "").lower():
                return False
        if rule.get("subject_contains"):
            if rule["subject_contains"].lower() not in msg.get("subject", "").lower():
                return False
        if rule.get("body_contains"):
            if rule["body_contains"].lower() not in msg.get("body", "").lower():
                return False
        return True

    def _generate_ai_reply(self, msg: dict, rule: dict) -> str:
        """توليد رد ذكي باستخدام LLM."""
        if not self.llm_key or not rule.get("reply_ai"):
            return self._apply_template(rule, msg)

        # استخدام Groq API (الأرخص والأسرع)
        prompt = f"""أنت مساعد ذكي يرد على إيميلات. رد على هذا الإيميل بشكل احترافي ومختصر.

الإيميل الأصلي:
من: {msg.get('from', '')}
الموضوع: {msg.get('subject', '')}
المحتوى: {msg.get('body', '')[:1000]}

الرد يجب أن يكون:
- احترافي ومختصر
- باللغة العربية ما لم يكن الإيميل بالإنجليزية
- لا تتجاوز 100 كلمة
"""
        try:
            import urllib.request
            data = json.dumps({
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 500,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.groq.com/openai/v1/chat/completions",
                data=data,
                headers={
                    "Authorization": f"Bearer {self.llm_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())
                return result["choices"][0]["message"]["content"].strip()
        except Exception:
            # Fallback: استخدام OpenAI
            try:
                data = json.dumps({
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                }).encode("utf-8")
                req = urllib.request.Request(
                    "https://api.openai.com/v1/chat/completions",
                    data=data,
                    headers={
                        "Authorization": f"Bearer {self.llm_key}",
                        "Content-Type": "application/json",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read().decode())
                    return result["choices"][0]["message"]["content"].strip()
            except Exception:
                return self._apply_template(rule, msg)

    def _apply_template(self, rule: dict, msg: dict) -> str:
        """تطبيق قالب رد ثابت."""
        template = rule.get("reply_template") or (
            f"شكرًا لتواصلك. تم استلام إيميلك بخصوص: {msg.get('subject', '')}.\n"
            "سيتم الرد عليك في أقرب وقت ممكن.\n\nمع تحياتنا،\nNSM Bot"
        )
        return template.format(
            sender=msg.get("from", ""),
            subject=msg.get("subject", ""),
            body=msg.get("body", "")[:200],
        )

    def _send_reply(self, to: str, subject: str, body: str) -> dict:
        """إرسال إيميل رد."""
        import base64

        # بناء MIME message بسيط
        mime = f"From: me\r\nTo: {to}\r\nSubject: {subject}\r\n"
        mime += f"Content-Type: text/plain; charset=utf-8\r\n\r\n{body}"

        encoded = base64.urlsafe_b64encode(mime.encode("utf-8")).decode("utf-8")
        status, result = self._post_gmail("messages/send", {"raw": encoded})
        return {"status": status, "result": result}

    def _log_reply(self, msg_id: str, rule: dict, reply_text: str, sent: bool):
        """تسجيل الرد في audit log."""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "msg_id": msg_id,
            "rule": {k: v for k, v in rule.items() if k != "reply_template" or True},
            "reply": reply_text[:500],
            "sent": sent,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def check_and_reply(self, query: str = "is:unread") -> list:
        """فحص الإيميلات غير المقروءة والرد على المطابقة."""
        if not self.available:
            return [{"error": "لا Gmail token"}]

        status, data = self._get_gmail(f"messages?q={query}&maxResults=20")
        if status != 200:
            return [{"error": f"Gmail API: {data}"}]

        messages = data.get("messages", [])
        results = []

        for msg_info in messages:
            msg_id = msg_info.get("id")

            # منع الرد على نفس الإيميل مرتين
            if msg_id in self.replied_ids:
                continue

            # جلب المحتوى الكامل
            msg = self._get_message(msg_id)
            if "error" in msg:
                continue

            # ── Blacklist Check ──────────────────────────────────
            sender_addr = msg.get("from", "")
            if self.is_blacklisted(sender_addr):
                continue  # تجاهل — لا رد ولا تسجيل

            # مطابقة القواعد
            matched_rule = None
            for rule in self.rules:
                if self._match_rule(msg, rule):
                    matched_rule = rule
                    break

            if matched_rule is None:
                continue

            # توليد الرد
            reply_text = self._generate_ai_reply(msg, matched_rule)

            # تحديد المرسل للرد
            sender = matched_rule.get("sender") or msg.get("from", "")

            # إرسال الرد
            reply_subject = f"Re: {msg.get('subject', '')}"
            send_result = self._send_reply(sender, reply_subject, reply_text)
            sent = send_result.get("status") == 200

            # علامة مقروء + أرشفة
            if sent:
                if matched_rule.get("mark_read"):
                    self._post_gmail(f"messages/{msg_id}/modify", {"addLabelIds": ["UNREAD"]})
                    self._post_gmail(f"messages/{msg_id}/modify", {"removeLabelIds": ["UNREAD"]})
                if matched_rule.get("archive"):
                    self._post_gmail(f"messages/{msg_id}/modify", {"removeLabelIds": ["INBOX"]})

            # تسجيل
            self._log_reply(msg_id, matched_rule, reply_text, sent)
            self.replied_ids.add(msg_id)

            results.append({
                "msg_id": msg_id,
                "from": msg.get("from", ""),
                "subject": msg.get("subject", ""),
                "reply": reply_text[:100],
                "sent": sent,
            })

        return results

    def watch_and_reply(self, interval_seconds: int = 60,
                        query: str = "is:unread",
                        callback: Callable = None) -> None:
        """مراقبة مستمرة — يفحص كل interval ويرد تلقائيًا."""
        while not self.stop_event.is_set():
            try:
                results = self.check_and_reply(query=query)
                if results and callback:
                    callback(results)
            except Exception as e:
                print(f"[AutoReply] خطأ: {e}")
            self.stop_event.wait(interval_seconds)

    def watch_async(self, interval_seconds: int = 60,
                    query: str = "is:unread",
                    callback: Callable = None) -> threading.Thread:
        """بدء المراقبة في thread منفصل."""
        t = threading.Thread(
            target=self.watch_and_reply,
            args=(interval_seconds, query, callback),
            daemon=True,
        )
        t.start()
        return t

    def stop(self):
        """إيقاف المراقبة."""
        self.stop_event.set()

    def get_replies_log(self, limit: int = 50) -> list:
        """قراءة سجل الردود."""
        entries = []
        if self.log_file.exists():
            with open(self.log_file) as f:
                lines = f.readlines()
                for line in lines[-limit:]:
                    try:
                        entries.append(json.loads(line.strip()))
                    except Exception:
                        pass
        return entries
