"""
state_store.py — تخزين حالة محادثة واتساب بين الرسائل (phone → state).

⚠️ نسخة طبق الأصل من ai/whatsapp/state_store.py بالمستودع الرئيسي —
راجع الملاحظة بأعلى quran_lookup.py بهذا المجلد لشرح سبب عدم استيراده
مباشرة (whatsapp_gateway مشروع Vercel منفصل بجذر مستقل).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol
from urllib.parse import quote

_HTTP_TIMEOUT = 5
_TTL_SECONDS = 60 * 60 * 6


class StateStore(Protocol):
    def get(self, phone: str) -> Optional[str]: ...
    def set(self, phone: str, state: str) -> None: ...


class InMemoryStateStore:
    """للاختبار المحلي فقط — لا تُستخدم بالإنتاج (الحالة تُفقد بين الاستدعاءات)."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, phone: str) -> Optional[str]:
        return self._data.get(phone)

    def set(self, phone: str, state: str) -> None:
        self._data[phone] = state


class UpstashStateStore:
    def __init__(self, base_url: str, token: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {token}"}

    def get(self, phone: str) -> Optional[str]:
        import requests

        key = _redis_key(phone)
        try:
            resp = requests.get(
                f"{self._base_url}/get/{key}",
                headers=self._headers,
                timeout=_HTTP_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get("result")
        except Exception:
            return None

    def set(self, phone: str, state: str) -> None:
        import requests

        key = _redis_key(phone)
        try:
            requests.post(
                f"{self._base_url}/set/{key}/{quote(state, safe='')}?EX={_TTL_SECONDS}",
                headers=self._headers,
                timeout=_HTTP_TIMEOUT,
            )
        except Exception:
            pass


def _redis_key(phone: str) -> str:
    return "wa_state:" + quote(phone.strip().replace(" ", ""), safe="")


def get_state_store() -> StateStore:
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if url and token:
        return UpstashStateStore(url, token)
    return InMemoryStateStore()
