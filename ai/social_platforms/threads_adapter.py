"""
محول Threads — عبر Threads API الرسمي من Meta (graph.threads.net، منفصل
عن Graph API الخاص بفيسبوك/إنستغرام رغم مشاركة نفس بنية OAuth).
يتطلب: THREADS_ACCESS_TOKEN (توكن طويل الأمد بصلاحيات threads_basic +
threads_content_publish)، THREADS_USER_ID.

النشر نموذج حاوية بخطوتين (نفس نمط إنستغرام):
  1) POST /{user_id}/threads  → ينشئ حاوية مسودة، يعيد creation_id
  2) POST /{user_id}/threads_publish → ينشر الحاوية فعلياً
Threads API مجاني بالكامل من Meta (لا رسوم كـ X API)، لكنه يتطلب مراجعة
تطبيق (App Review) من Meta لتفعيل صلاحيات النشر في الإنتاج.
"""

from __future__ import annotations

import os
import time
from typing import List

import requests

from .base import PlatformAdapter, SocialItem
from .retry import with_retry

API_BASE = "https://graph.threads.net/v1.0"


class ThreadsAdapter(PlatformAdapter):
    platform_id = "threads"
    required_env = ["THREADS_ACCESS_TOKEN", "THREADS_USER_ID"]

    def _token(self) -> str:
        return os.environ["THREADS_ACCESS_TOKEN"]

    def _create_and_publish(self, text: str, reply_to_id: str = "") -> str:
        uid = os.environ["THREADS_USER_ID"]
        params = {"media_type": "TEXT", "text": text, "access_token": self._token()}
        if reply_to_id:
            params["reply_to_id"] = reply_to_id
        create = requests.post(f"{API_BASE}/{uid}/threads", data=params, timeout=30)
        create.raise_for_status()
        creation_id = create.json()["id"]

        time.sleep(2)  # منشورات نصية تُعالَج بسرعة؛ الوسائط تحتاج مهلة أطول (30ث+)

        publish = requests.post(
            f"{API_BASE}/{uid}/threads_publish",
            data={"creation_id": creation_id, "access_token": self._token()},
            timeout=30,
        )
        publish.raise_for_status()
        return publish.json()["id"]

    @with_retry()
    def publish(self, text: str) -> str:
        self._require_configured()
        return self._create_and_publish(text)

    def _latest_own_thread_id(self) -> str:
        uid = os.environ["THREADS_USER_ID"]
        r = requests.get(
            f"{API_BASE}/{uid}/threads",
            params={"fields": "id", "limit": 1, "access_token": self._token()},
            timeout=30,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
        return data[0]["id"] if data else ""

    @with_retry()
    def fetch_new_items(self, since_ids: set) -> List[SocialItem]:
        self._require_configured()
        thread_id = self._latest_own_thread_id()
        if not thread_id:
            return []
        r = requests.get(
            f"{API_BASE}/{thread_id}/replies",
            params={"fields": "id,text,username", "access_token": self._token()},
            timeout=30,
        )
        r.raise_for_status()
        items: List[SocialItem] = []
        for rep in r.json().get("data", []):
            rid = rep.get("id", "")
            if not rid or rid in since_ids:
                continue
            items.append(SocialItem(
                platform="threads", external_id=rid, kind="reply",
                author=rep.get("username", "unknown"), text=rep.get("text", ""),
                thread_id=thread_id, raw=rep,
            ))
        return items

    @with_retry()
    def reply(self, item: SocialItem, text: str) -> str:
        self._require_configured()
        return self._create_and_publish(text, reply_to_id=item.external_id)
