# -*- coding: utf-8 -*-
"""
agent_external_tools.py — أدوات خارجية للوكلاء

يتيح للوكلاء:
1. الاتصال بـ Gmail (قراءة + بحث + إرسال)
2. الاتصال بـ REST APIs (أي خدمة)
3. الاتصال بـ Webhooks
4. التعامل مع Google Drive (رفع/تنزيل)

كل أداة تعتمد على tokens من environment variables:
- GMAIL_TOKEN / GMAIL_ACCESS_TOKEN
- GDRIVE_TOKEN / GDRIVE_ACCESS_TOKEN
- WEBHOOK_URL

التثبيت: pip install google-auth google-auth-oauthlib google-api-python-client requests
"""
import json
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional


# ── helpers ─────────────────────────────────────────────────────────────────
def _http(url: str, method: str = "GET", headers: dict = None,
          body: dict = None, timeout: int = 30) -> tuple:
    """مساعد HTTP بسيط (بدون requests). يرجع (status_code, dict_or_str)."""
    headers = headers or {}
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers.setdefault("Content-Type", "application/json")
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.getcode()
            raw = resp.read().decode("utf-8", errors="replace")
            try:
                return status, json.loads(raw)
            except Exception:
                return status, raw
    except Exception as e:
        return 0, str(e)


class GmailTool:
    """أداة Gmail — قراءة وبحث وإرسال الإيميلات."""

    BASE = "https://gmail.googleapis.com/gmail/v1"

    def __init__(self, access_token: str = None, max_results: int = 20):
        self.token = access_token or os.environ.get("GMAIL_TOKEN") or os.environ.get("GMAIL_ACCESS_TOKEN") or ""
        self.max_results = max_results

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def list_emails(self, q: str = "in:inbox", max_results: int = None) -> list:
        """جلب قائمة الإيميلات — يرجع [{'id', 'subject', 'from', 'date', 'snippet'}]."""
        if not self.available:
            return []
        max_r = max_results or self.max_results
        params = {"q": q, "maxResults": str(max_r)}
        url = f"{self.BASE}/users/me/messages?{urllib.parse.urlencode(params)}"
        status, data = _http(url, headers=self._headers())
        if status != 200 or not isinstance(data, dict):
            return []
        msgs = data.get("messages", [])
        results = []
        for m in msgs:
            detail = self.get_message(m["id"])
            if detail:
                results.append(detail)
        return results

    def get_message(self, msg_id: str) -> dict:
        """جلب تفاصيل إيميل كامل (header + body)."""
        if not self.available:
            return {}
        url = f"{self.BASE}/users/me/messages/{msg_id}"
        status, data = _http(url, headers=self._headers())
        if status != 200 or not isinstance(data, dict):
            return {}
        headers = {}
        payload = data.get("payload", {})
        for h in payload.get("headers", []):
            headers[h["name"].lower()] = h["value"]
        body = self._extract_body(payload)
        return {
            "id": msg_id,
            "subject": headers.get("subject", ""),
            "from": headers.get("from", ""),
            "to": headers.get("to", ""),
            "date": headers.get("date", ""),
            "snippet": data.get("snippet", ""),
            "body": body,
        }

    def send_email(self, to: str, subject: str, body: str,
                   from_email: str = None) -> bool:
        """إرسال إيميل."""
        if not self.available:
            return False
        import base64
        content = (
            f"From: {from_email or 'me'}\r\n"
            f"To: {to}\r\n"
            f"Subject: {subject}\r\n"
            f"Content-Type: text/plain; charset=utf-8\r\n\r\n"
            f"{body}"
        )
        raw = base64.urlsafe_b64encode(content.encode("utf-8")).decode("utf-8")
        url = f"{self.BASE}/users/me/messages/send"
        status, _ = _http(url, method="POST", headers=self._headers(), body={"raw": raw})
        return status == 200

    def search_emails(self, query: str) -> list:
        """بحث في الإيميلات — يرجع [{'id', 'subject', 'from', 'date', 'snippet'}]."""
        return self.list_emails(q=query)

    def _extract_body(self, payload: dict) -> str:
        """استخراج نص الإيميل من payload."""
        mime = payload.get("mimeType", "")
        if mime == "text/plain" and "data" in payload.get("body", {}):
            import base64
            raw = payload["body"]["data"]
            return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", errors="replace")
        parts = payload.get("parts", [])
        for part in parts:
            pmime = part.get("mimeType", "")
            if pmime == "text/plain" and "data" in part.get("body", {}):
                import base64
                raw = part["body"]["data"]
                return base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", errors="replace")
        return payload.get("body", {}).get("data", "")


class WebhookTool:
    """أداة Webhooks — إرسال إشعارات لأي خدمة (Discord, Telegram, Slack...)."""

    def __init__(self, webhook_url: str = None):
        self.url = webhook_url or os.environ.get("WEBHOOK_URL") or ""

    @property
    def available(self) -> bool:
        return bool(self.url)

    def send(self, message: str, embeds: list = None) -> bool:
        """إرسال رسالة عبر webhook."""
        if not self.available:
            return False
        body = {"content": message}
        if embeds:
            body["embeds"] = embeds
        status, _ = _http(self.url, method="POST", body=body)
        return status in (200, 204)

    def notify_discord(self, message: str, title: str = None) -> bool:
        """إرسال إشعار Discord مع embed."""
        if not self.available:
            return False
        embed = {"title": title or "NSM Agent Alert", "description": message[:2000], "color": 3066993}
        return self.send("", embeds=[embed])

    def notify_telegram(self, chat_id: str, message: str) -> bool:
        """إرسال إشعار Telegram."""
        if not self.available:
            return False
        # افتراض أن webhook_url هو Telegram Bot Token
        url = f"https://api.telegram.org/bot{self.url}/sendMessage"
        status, _ = _http(url, method="POST", body={"chat_id": chat_id, "text": message})
        return status == 200


class GenericAPI:
    """أداة APIs عامة — الاتصال بأي REST API."""

    @staticmethod
    def get(url: str, headers: dict = None, params: dict = None, timeout: int = 30) -> tuple:
        """GET request."""
        if params:
            url = f"{url}?{urllib.parse.urlencode(params)}"
        headers = headers or {"User-Agent": "NSM-Agent/1.0"}
        return _http(url, headers=headers, timeout=timeout)

    @staticmethod
    def post(url: str, body: dict = None, headers: dict = None, timeout: int = 30) -> tuple:
        """POST request."""
        headers = headers or {"User-Agent": "NSM-Agent/1.0"}
        return _http(url, method="POST", headers=headers, body=body, timeout=timeout)

    @staticmethod
    def fetch_json(url: str, headers: dict = None, params: dict = None) -> dict:
        """جلب JSON من URL."""
        status, data = GenericAPI.get(url, headers=headers, params=params)
        if status == 200 and isinstance(data, dict):
            return data
        return {"error": f"status={status}", "raw": str(data)[:500]}


class GoogleDriveTool:
    """أداة Google Drive — رفع/تنزيل الملفات."""

    BASE = "https://www.googleapis.com/upload/drive/v3"
    API = "https://www.googleapis.com/drive/v3"

    def __init__(self, access_token: str = None):
        self.token = access_token or os.environ.get("GDRIVE_TOKEN") or os.environ.get("GDRIVE_ACCESS_TOKEN") or ""

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}"}

    def upload_file(self, file_path: str, folder_id: str = None,
                    mime_type: str = "application/octet-stream") -> dict:
        """رفع ملف إلى Google Drive — يرجع {'id', 'name', 'webViewLink'}."""
        if not self.available:
            return {"error": "لا GDRIVE_TOKEN"}
        path = Path(file_path)
        if not path.is_file():
            return {"error": f"الملف غير موجود: {file_path}"}
        size = path.stat().st_size
        metadata = {"name": path.name}
        if folder_id:
            metadata["parents"] = [folder_id]

        # multipart upload (يدعم حتى 5MB)
        if size <= 5 * 1024 * 1024:
            return self._upload_simple(metadata, path, mime_type)
        else:
            return self._upload_resumable(metadata, path, mime_type)

    def _upload_simple(self, metadata: dict, path: Path, mime_type: str) -> dict:
        """رفع بسيط (<= 5MB)."""
        url = f"{self.BASE}/files?uploadType=multipart"
        import base64
        raw = path.read_bytes()
        b64 = base64.b64encode(raw).decode("utf-8")
        boundary = "foo_bar_baz"
        body = (
            f"--{boundary}\r\n"
            f"Content-Type: application/json; charset=UTF-8\r\n\r\n"
            f"{json.dumps(metadata)}\r\n"
            f"--{boundary}\r\n"
            f"Content-Type: {mime_type}\r\n"
            f"Content-Transfer-Encoding: base64\r\n\r\n"
            f"{b64}\r\n"
            f"--{boundary}--"
        ).encode("utf-8")
        headers = self._headers()
        headers["Content-Type"] = f"multipart/related; boundary={boundary}"
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"error": str(e)}

    def _upload_resumable(self, metadata: dict, path: Path, mime_type: str) -> dict:
        """رفع قابل للاستئناف (> 5MB)."""
        # 1. إنشاء session
        url = f"{self.BASE}/files?uploadType=resumable"
        status, data = _http(url, method="POST",
                             headers={**self._headers(), "Content-Type": "application/json",
                                      "X-Upload-Content-Type": mime_type},
                             body=metadata)
        if status != 200:
            return {"error": f"فشل إنشاء session: {data}"}
        # العثور على Location header
        location = None
        try:
            req = urllib.request.Request(url, data=json.dumps(metadata).encode(),
                                         headers={**self._headers(), "Content-Type": "application/json",
                                                  "X-Upload-Content-Type": mime_type}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as resp:
                location = resp.headers.get("Location")
        except Exception as e:
            return {"error": str(e)}
        if not location:
            return {"error": "لا Location header"}

        # 2. رفع المحتوى
        chunk = 5 * 1024 * 1024  # 5MB chunks
        raw = path.read_bytes()
        total = len(raw)
        start = 0
        result = {"error": "فشل الرفع"}
        while start < total:
            end = min(start + chunk, total)
            slice_data = raw[start:end]
            headers = {
                "Authorization": f"Bearer {self.token}",
                "Content-Length": str(end - start),
                "Content-Range": f"bytes {start}-{end - 1}/{total}",
            }
            req = urllib.request.Request(location, data=slice_data, headers=headers, method="PUT")
            try:
                with urllib.request.urlopen(req, timeout=600) as resp:
                    status = resp.getcode()
                    if status == 200:
                        result = json.loads(resp.read().decode("utf-8"))
                        break
                    elif status == 308:
                        start = end
                    else:
                        result = {"error": f"status={status}"}
                        break
            except Exception as e:
                result = {"error": str(e)}
                break
        return result

    def list_files(self, q: str = None, max_results: int = 50) -> list:
        """جلب قائمة الملفات."""
        if not self.available:
            return []
        params = {"pageSize": str(max_results), "fields": "files(id,name,mimeType,modifiedTime,size)"}
        if q:
            params["q"] = q
        url = f"{self.API}/files?{urllib.parse.urlencode(params)}"
        status, data = _http(url, headers=self._headers())
        if status == 200 and isinstance(data, dict):
            return data.get("files", [])
        return []

    def download_file(self, file_id: str, dest: str) -> bool:
        """تنزيل ملف من Google Drive."""
        if not self.available:
            return False
        url = f"{self.API}/files/{file_id}?alt=media"
        req = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                Path(dest).parent.mkdir(parents=True, exist_ok=True)
                Path(dest).write_bytes(resp.read())
                return True
        except Exception:
            return False


# ── واجهة موحدة للوكيل ─────────────────────────────────────────────────────
class ExternalToolsAgent:
    """وكيل الأدوات الخارجية — يتحكم في كل الأدوات ويوفر واجهة موحدة."""

    def __init__(self):
        self.gmail = GmailTool()
        self.webhook = WebhookTool()
        self.gdrive = GoogleDriveTool()

    def status(self) -> dict:
        """فحص أي الأدوات متاحة."""
        return {
            "gmail": self.gmail.available,
            "webhook": self.webhook.available,
            "gdrive": self.gdrive.available,
        }

    def execute(self, tool_name: str, action: str, **kwargs) -> dict:
        """تنفيذ أمر على أداة خارجية — واجهة موحدة للوكيل."""
        tools = {"gmail": self.gmail, "webhook": self.webhook, "gdrive": self.gdrive}
        tool = tools.get(tool_name)
        if not tool:
            return {"error": f"أداة غير معروفة: {tool_name}"}
        if not hasattr(tool, action):
            return {"error": f"إجراء غير معروف: {action}"}
        fn = getattr(tool, action)
        try:
            result = fn(**kwargs)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
