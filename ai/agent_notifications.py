# -*- coding: utf-8 -*-
"""
agent_notifications.py — متابعة الإشعارات والتنبيهات

يتيح للوكلاء:
1. مراقبة حالة Kaggle kernels (RUNNING, ERROR, COMPLETE)
2. مراقبة GitHub commits و pull requests
3. مراقبة Gmail لرسائل جديدة
4. إرسال تنبيهات فورية (Discord, Telegram, Webhook)
5. جدولة مراقبة دورية (polling)

يعمل بـ Python stdlib فقط.
"""
import json
import os
import time
import threading
import urllib.request
import urllib.parse
from typing import Optional, Callable


# ── helpers ─────────────────────────────────────────────────────────────────
def _http(url: str, method: str = "GET", headers: dict = None,
          body: dict = None, timeout: int = 30) -> tuple:
    """مساعد HTTP."""
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


class KaggleMonitor:
    """مراقبة Kaggle kernels."""

    API = "https://www.kaggle.com/api/i/kernels.KernelsService/KernelGetStatus"

    def __init__(self, username: str = None, kaggle_key: str = None):
        self.username = username or os.environ.get("KAGGLE_USERNAME") or ""
        self.key = kaggle_key or os.environ.get("KAGGLE_KEY") or ""

    @property
    def available(self) -> bool:
        return bool(self.username and self.key)

    def get_kernel_status(self, kernel_slug: str) -> dict:
        """جلب حالة كيرنل Kaggle."""
        if not self.available:
            return {"error": "لا KAGGLE credentials"}
        params = {"kernel_slug": kernel_slug}
        url = f"{self.API}?{urllib.parse.urlencode(params)}"
        headers = {"User-Agent": "NSM-Agent/1.0"}
        status, data = _http(url, headers=headers)
        if status == 200 and isinstance(data, dict):
            return {
                "slug": kernel_slug,
                "status": data.get("status", data),
                "raw": data,
            }
        return {"slug": kernel_slug, "error": f"status={status}", "raw": str(data)[:300]}

    def is_running(self, kernel_slug: str) -> bool:
        """هل الكيرنل يعمل الآن؟"""
        result = self.get_kernel_status(kernel_slug)
        status = result.get("status", {})
        if isinstance(status, dict):
            s = status.get("value", status.get("status", ""))
            return "RUNNING" in str(s).upper()
        return False


class GitHubMonitor:
    """مراقبة GitHub — commits, PRs, issues."""

    API = "https://api.github.com"

    def __init__(self, token: str = None, repo: str = None):
        self.token = token or os.environ.get("GITHUB_TOKEN") or ""
        self.repo = repo or os.environ.get("GITHUB_REPO") or ""

    @property
    def available(self) -> bool:
        return bool(self.token)

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"}

    def get_latest_commit(self, branch: str = "main") -> dict:
        """جلب آخر commit."""
        if not self.repo:
            return {"error": "لا repo"}
        url = f"{self.API}/repos/{self.repo}/commits/{branch}"
        status, data = _http(url, headers=self._headers())
        if status == 200 and isinstance(data, dict):
            return {
                "sha": data.get("sha", "")[:7],
                "message": data.get("commit", {}).get("message", ""),
                "author": data.get("commit", {}).get("author", {}).get("name", ""),
                "date": data.get("commit", {}).get("author", {}).get("date", ""),
            }
        return {"error": f"status={status}"}

    def get_pull_requests(self, state: str = "open") -> list:
        """جلب pull requests."""
        if not self.repo:
            return []
        url = f"{self.API}/repos/{self.repo}/pulls?state={state}"
        status, data = _http(url, headers=self._headers())
        if status == 200 and isinstance(data, list):
            return [{"number": pr["number"], "title": pr["title"], "user": pr["user"]["login"]}
                    for pr in data]
        return []

    def get_issues(self, state: str = "open") -> list:
        """جلب issues."""
        if not self.repo:
            return []
        url = f"{self.API}/repos/{self.repo}/issues?state={state}"
        status, data = _http(url, headers=self._headers())
        if status == 200 and isinstance(data, list):
            return [{"number": i["number"], "title": i["title"], "user": i["user"]["login"]}
                    for i in data if "pull_request" not in i]
        return []

    def check_branch_activity(self, branch: str, minutes: int = 30) -> dict:
        """هل هناك نشاط على الفرع خلال آخر N دقيقة؟"""
        commit = self.get_latest_commit(branch)
        if "error" in commit:
            return {"error": commit["error"]}
        from datetime import datetime, timedelta, timezone
        try:
            commit_time = datetime.fromisoformat(commit["date"].replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            diff = (now - commit_time).total_seconds() / 60
            return {"active": diff < minutes, "minutes_ago": round(diff, 1), "commit": commit}
        except Exception:
            return {"active": False, "commit": commit}


class NotificationAgent:
    """وكيل الإشعارات — يجمع كل المراقبين ويوفر واجهة موحدة."""

    def __init__(self):
        self.kaggle = KaggleMonitor()
        self.github = GitHubMonitor()
        from .agent_external_tools import GmailTool, WebhookTool
        self.gmail = GmailTool()
        self.webhook = WebhookTool()

    def status(self) -> dict:
        """فحص توفر كل المراقبين."""
        return {
            "kaggle": self.kaggle.available,
            "github": self.github.available,
            "gmail": self.gmail.available,
            "webhook": self.webhook.available,
        }

    def check_all(self, kernel_slug: str = None) -> dict:
        """فحص شامل — Kaggle + GitHub + Gmail."""
        result = {}
        if kernel_slug:
            result["kaggle"] = self.kaggle.get_kernel_status(kernel_slug)
        result["github"] = self.github.get_latest_commit()
        result["github_prs"] = self.github.get_pull_requests()
        if self.gmail.available:
            result["gmail"] = self.gmail.list_emails(max_results=5)
        return result

    def send_alert(self, message: str, title: str = "NSM Alert",
                   via: str = "webhook") -> bool:
        """إرسال تنبيه."""
        if via == "webhook" and self.webhook.available:
            return self.webhook.notify_discord(message, title=title)
        elif via == "telegram" and self.webhook.available:
            return self.webhook.notify_telegram(os.environ.get("TELEGRAM_CHAT_ID", ""), message)
        elif via == "gmail" and self.gmail.available:
            to = os.environ.get("ALERT_EMAIL", "")
            if to:
                return self.gmail.send_email(to, title, message)
        return False

    def watch_kernel(self, kernel_slug: str, interval_seconds: int = 300,
                     on_status_change: Callable = None, max_checks: int = None) -> dict:
        """مراقبة كيرنل Kaggle بشكل دوري — يرجع عند تغيير الحالة."""
        last_status = None
        checks = 0
        history = []
        while max_checks is None or checks < max_checks:
            result = self.kaggle.get_kernel_status(kernel_slug)
            current = json.dumps(result.get("status", ""))
            if current != last_status:
                history.append({"check": checks, "status": result})
                if on_status_change:
                    try:
                        on_status_change(result)
                    except Exception:
                        pass
                # إرسال تنبيه عند تغيير الحالة
                if "RUNNING" in str(result.get("status", "")).upper():
                    self.send_alert(f"كيرنل {kernel_slug} بدأ يعمل", title="Kaggle RUNNING")
                elif "COMPLETE" in str(result.get("status", "")).upper():
                    self.send_alert(f"كيرنل {kernel_slug} اكتمل", title="Kaggle COMPLETE")
                elif "ERROR" in str(result.get("status", "")).upper():
                    self.send_alert(f"كيرنل {kernel_slug} فشل!", title="Kaggle ERROR")
                last_status = current
            checks += 1
            time.sleep(interval_seconds)
        return {"checks": checks, "history": history}

    def watch_kernel_async(self, kernel_slug: str, interval_seconds: int = 300,
                           on_status_change: Callable = None) -> threading.Thread:
        """مراقبة كيرنل في thread منفصل."""
        t = threading.Thread(
            target=self.watch_kernel,
            args=(kernel_slug, interval_seconds, on_status_change),
            daemon=True,
        )
        t.start()
        return t
