# -*- coding: utf-8 -*-
"""
عرض منتج واحد قابل للبيع — تلخيص متعدد المصادر مع إثبات ونصاب
=============================================================
NSM Verified Summary API

لا يعالج مدفوعات حقيقية هنا؛ المفتاح عبر البيئة فقط:
  NSM_PRODUCT_API_KEYS=key1,key2
  NSM_TRIAL_DAILY_LIMIT=20
"""
from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

PRODUCT = {
    "id": "nsm_verified_summary",
    "name_ar": "تلخيص متعدد المصادر مع إثبات",
    "name_en": "NSM Verified Multi-Source Summary",
    "tagline_ar": "مصادر تُمرَّر من العميل · تلخيص موزّع · source_hash · نصاب عمال · بدون SSRF",
    "endpoint": "POST /v2/product/summarize",
    "what_you_get": [
        "تلخيص عدة مصادر نصية في طلب واحد",
        "بصمة source_hash لكل مصدر",
        "إثبات provenance (عامل + digest + نصاب)",
        "لا جلب عناوين عشوائية من الإنترنت داخل هذا العرض (آمن افتراضياً)",
    ],
    "plans": [
        {
            "id": "trial",
            "name_ar": "تجربة",
            "price_usd": 0,
            "price_note_ar": "مجاناً — حد يومي",
            "daily_limit": int(os.getenv("NSM_TRIAL_DAILY_LIMIT") or 20),
            "max_sources": 5,
            "features": ["حتى 5 مصادر/طلب", "حد يومي للتجربة", "بدون بطاقة"],
        },
        {
            "id": "starter",
            "name_ar": "Starter",
            "price_usd": 29,
            "price_note_ar": "29$/شهر — مفتاح API",
            "daily_limit": 2000,
            "max_sources": 20,
            "features": ["مفتاح API", "حتى 20 مصدراً/طلب", "أولوية دعم عبر GitHub Issues"],
        },
        {
            "id": "growth",
            "name_ar": "Growth",
            "price_usd": 99,
            "price_note_ar": "99$/شهر — استخدام أعلى",
            "daily_limit": 15000,
            "max_sources": 50,
            "features": ["حد أعلى", "مصادر أكثر", "اتفاقية استخدام عند الطلب"],
        },
    ],
    "payment_ar": "الدفع الإلكتروني غير مدمج بعد. للخطة المدفوعة: اطلب مفتاحاً عبر قناة المشروع بعد الاتفاق اليدوي.",
    "contact_ar": "المستودع: Neural-Service-Mesh على GitHub",
}

# عدّاد تجربة بسيط في الذاكرة (لكل عملية؛ ليس موزّعاً)
_trial_hits: Dict[str, List[float]] = defaultdict(list)


def _api_keys() -> List[str]:
    raw = os.getenv("NSM_PRODUCT_API_KEYS") or os.getenv("NSM_API_KEYS") or ""
    return [k.strip() for k in raw.split(",") if k.strip()]


def resolve_plan(api_key: Optional[str]) -> Tuple[str, Dict[str, Any]]:
    """يرجع (plan_id, plan_dict). مفتاح صالح → starter؛ وإلا trial."""
    plans = {p["id"]: p for p in PRODUCT["plans"]}
    keys = _api_keys()
    if api_key and keys and api_key in keys:
        return "starter", plans["starter"]
    return "trial", plans["trial"]


def check_rate_limit(client_id: str, plan: Dict[str, Any]) -> Optional[str]:
    """None إن مسموح، وإلا رسالة خطأ."""
    limit = int(plan.get("daily_limit") or 20)
    now = time.time()
    window = 24 * 3600
    hits = [t for t in _trial_hits[client_id] if now - t < window]
    _trial_hits[client_id] = hits
    if len(hits) >= limit:
        return f"daily_limit_reached:{limit}"
    hits.append(now)
    _trial_hits[client_id] = hits
    return None


def public_catalog() -> Dict[str, Any]:
    return {
        "product": {
            "id": PRODUCT["id"],
            "name_ar": PRODUCT["name_ar"],
            "name_en": PRODUCT["name_en"],
            "tagline_ar": PRODUCT["tagline_ar"],
            "endpoint": PRODUCT["endpoint"],
            "what_you_get": PRODUCT["what_you_get"],
            "plans": PRODUCT["plans"],
            "payment_ar": PRODUCT["payment_ar"],
            "contact_ar": PRODUCT["contact_ar"],
        }
    }
