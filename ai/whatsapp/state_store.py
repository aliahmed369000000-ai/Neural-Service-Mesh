"""
state_store.py — تخزين حالة محادثة واتساب بين الرسائل (phone → state).

مصمَّم كواجهة (StateStore) + تطبيقين:
  - UpstashStateStore: الإنتاج الفعلي، عبر Upstash Redis REST API
    (HTTP بسيط، بدون اتصال TCP دائم — مناسب لـVercel serverless).
  - InMemoryStateStore: للاختبار المحلي فقط، بدون أي شبكة.

يُستدعى من api/webhook.py بهذا الشكل:
    store = get_state_store()
    state = store.get(phone)
    reply, new_state = handle_incoming_message(phone, text, state)
    store.set(phone, new_state)

متغيرات البيئة المطلوبة لـUpstashStateStore (تُضبط بـVercel لاحقاً):
    UPSTASH_REDIS_REST_URL
    UPSTASH_REDIS_REST_TOKEN
لو غابا، get_state_store() يسقط تلقائياً لـInMemoryStateStore مع تحذير
واضح (مفيد للتطوير المحلي، لكن الحالة تُفقد بين كل تشغيل serverless
منفصل بالإنتاج الفعلي — لازم Upstash مضبوط قبل النشر الحقيقي).
"""
from __future__ import annotations

import os
from typing import Optional, Protocol
from urllib.parse import quote

# مهلة قصيرة عمداً: هذا استدعاء داخل معالجة رسالة واتساب، لا يجب أن
# يعلّق الاستجابة لثوانٍ طويلة لو تعطّل Upstash
_HTTP_TIMEOUT = 5
_TTL_SECONDS = 60 * 60 * 6  # 6 ساعات: يكفي لجلسة قائمة نشطة، ويمسح نفسه لو المستخدم اختفى


class StateStore(Protocol):
    def get(self, phone: str) -> Optional[str]: ...
    def set(self, phone: str, state: str) -> None: ...


class InMemoryStateStore:
    """للاختبار المحلي فقط — الحالة تُفقد عند انتهاء العملية. لا تُستخدم بالإنتاج."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    def get(self, phone: str) -> Optional[str]:
        return self._data.get(phone)

    def set(self, phone: str, state: str) -> None:
        self._data[phone] = state


class UpstashStateStore:
    """تطبيق حقيقي عبر Upstash Redis REST API. يستخدم مكتبة requests فقط
    (بدون عميل redis كامل) — طلب HTTP واحد لكل get/set، مناسب لبيئة
    serverless عديمة الحالة بين الاستدعاءات."""

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
            # فشل Upstash لا يجب أن يوقف الرد بالكامل — نتعامل كأنه
            # مستخدم جديد (حالة None) بدل رمي خطأ للمستخدم النهائي
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
            pass  # فشل الحفظ يعني أسوأ حالة: المستخدم يرى القائمة الرئيسية بدل موقعه بالضبط — مقبول، ليس انهياراً


def _redis_key(phone: str) -> str:
    # ترميز آمن للرقم (يحتوي عادة على +) قبل إدراجه بمسار URL — بعض
    # السيرفرات تفسّر + كمسافة خارج سياق query string، فنرمّزه صراحةً
    return "wa_state:" + quote(phone.strip().replace(" ", ""), safe="")


def get_state_store() -> StateStore:
    """نقطة الدخول المستخدمة من webhook.py: تعيد UpstashStateStore لو
    متغيرات البيئة مضبوطة، وإلا InMemoryStateStore (تطوير محلي فقط)."""
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if url and token:
        return UpstashStateStore(url, token)
    return InMemoryStateStore()
