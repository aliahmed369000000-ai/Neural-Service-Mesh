"""عميل آمن لربط واجهة Streamlit بخدمة NSM API المنشورة."""
from __future__ import annotations

import os
from typing import Any

import requests


class NsmApiError(RuntimeError):
    """خطأ قابل للعرض عند فشل خدمة NSM البعيدة."""


def run_remote_task(task: str, url: str | None = None, timeout: float = 45.0) -> dict[str, Any]:
    """ينفذ مهمة عبر NSM API دون كشف المفتاح للواجهة أو السجل."""
    base = (os.getenv("NSM_API_URL") or "https://world-cup-2026-fun-guide.vercel.app").rstrip("/")
    key = os.getenv("NSM_ADMIN_KEY") or os.getenv("NSM_API_KEY")
    if not key:
        raise NsmApiError("مفتاح NSM غير مضبوط في أسرار Streamlit.")
    payload: dict[str, Any] = {"task": task}
    if url:
        payload["url"] = url
    try:
        response = requests.post(
            f"{base}/api/agent/tasks",
            json=payload,
            headers={"x-api-key": key, "Accept": "application/json"},
            timeout=(8.0, timeout),
        )
    except requests.RequestException as exc:
        raise NsmApiError(f"تعذر الاتصال بخدمة NSM: {exc}") from exc
    if response.status_code == 403:
        raise NsmApiError("رفضت خدمة NSM المفتاح. تحقق من سر Production.")
    if response.status_code >= 400:
        raise NsmApiError(f"خدمة NSM أعادت HTTP {response.status_code}.")
    try:
        data = response.json()
    except ValueError as exc:
        raise NsmApiError("استجابة NSM ليست JSON صالحة.") from exc
    if not isinstance(data, dict):
        raise NsmApiError("استجابة NSM غير متوقعة.")
    return data


def remote_available(timeout: float = 8.0) -> bool:
    """يتحقق من صحة الخدمة دون إرسال مهمة أو كشف أي سر."""
    base = (os.getenv("NSM_API_URL") or "https://world-cup-2026-fun-guide.vercel.app").rstrip("/")
    try:
        return requests.get(f"{base}/health", timeout=timeout).ok
    except requests.RequestException:
        return False
