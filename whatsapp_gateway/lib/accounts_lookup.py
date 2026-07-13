"""
accounts_lookup.py — بحث "هل هذا الهاتف مرتبط بحساب NSM؟" من واتساب.

⚠️ هذا **ليس** نسخة من ai/accounts.py — بالعكس، هو القارئ المقابل لما
تكتبه ai/accounts.py._sync_phone_to_upstash() بالمستودع الرئيسي.
السبب: memory/accounts.db (SQLite، مصدر الحقيقة) يعيش على قرص
Streamlit Community Cloud فقط، ودالة Vercel هذي لا تصل له إطلاقاً —
نظاما ملفات منفصلان تماماً (لا توجد وسيلة تقنية لمشاركة ملف بينهما).

لذلك القراءة هنا من Upstash Redis (نفس الحساب المستخدم بـstate_store.py)،
وتعتمد على مزامنة صامتة تحدث من الطرف الآخر (Streamlit) عند إنشاء/ربط
حساب بهاتف. لو المستخدم أنشأ حسابه قبل تفعيل UPSTASH_REDIS_REST_URL/
TOKEN بإعدادات Streamlit، لن يظهر هنا حتى يُعاد ربط هاتفه بعد التفعيل
— هذا قيد معروف ومقبول بالتصميم الحالي، وليس خطأ.
"""
from __future__ import annotations

import json
import os
from typing import Optional, Dict
from urllib.parse import quote

_HTTP_TIMEOUT = 5


def get_user_by_phone(phone_number: str) -> Optional[Dict]:
    """يعيد {"username": str, "created_at": str} لو الهاتف مرتبط بحساب
    مُزامَن، أو None لو غير مرتبط أو تعذّر الاتصال بـUpstash (فشل الشبكة
    يُعامَل كـ'غير موجود' — أأمن من رفع خطأ للمستخدم النهائي على واتساب)."""
    url = os.getenv("UPSTASH_REDIS_REST_URL", "").strip()
    token = os.getenv("UPSTASH_REDIS_REST_TOKEN", "").strip()
    if not url or not token:
        return None

    import requests

    key = "wa_account_phone:" + quote(phone_number.strip(), safe="")
    try:
        resp = requests.get(
            f"{url.rstrip('/')}/get/{key}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=_HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        result = resp.json().get("result")
        if not result:
            return None
        return json.loads(result)
    except Exception:
        return None
