"""
Stripe Billing — فوترة مفاتيح API (AIaaS)
========================================
يعمل في وضع تجريبي بدون مفاتيح. عند ضبط STRIPE_SECRET_KEY يحاول إنشاء Checkout Session.
"""
from __future__ import annotations

import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
BILL = ROOT / "artifacts" / "aiaas" / "billing"
BILL.mkdir(parents=True, exist_ok=True)

PLAN_PRICES = {
    "pro": {"usd": 49, "env": "STRIPE_PRICE_PRO"},
    "enterprise": {"usd": 299, "env": "STRIPE_PRICE_ENTERPRISE"},
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stripe_configured() -> bool:
    return bool(os.environ.get("STRIPE_SECRET_KEY", "").strip())


def create_checkout_session(
    plan: str = "pro",
    success_url: str = "https://example.com/success",
    cancel_url: str = "https://example.com/cancel",
    customer_email: Optional[str] = None,
) -> Dict[str, Any]:
    plan = (plan or "pro").lower()
    if plan not in PLAN_PRICES:
        plan = "pro"
    if not stripe_configured():
        # وضع تجريبي: يصدر مفتاحاً محلياً فوراً
        demo_key = "nsm_demo_" + secrets.token_hex(12)
        try:
            from ai.aiaas_platform import create_tenant
            ten = create_tenant(name=f"stripe-demo-{plan}", plan=plan, domain="general")
            # store demo mapping
            (BILL / f"demo_{ten['id']}.json").write_text(
                json.dumps({"api_key_hint": demo_key, "tenant": ten, "at": _now()}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return {
                "ok": True,
                "mode": "demo",
                "plan": plan,
                "checkout_url": None,
                "message_ar": "Stripe غير مضبوط — أُنشئ مستأجر تجريبي. اضبط STRIPE_SECRET_KEY للفوترة الحقيقية.",
                "tenant": ten,
            }
        except Exception as e:
            return {"ok": False, "mode": "demo", "error": str(e)}

    try:
        import urllib.parse
        import urllib.request

        secret = os.environ["STRIPE_SECRET_KEY"]
        price = os.environ.get(PLAN_PRICES[plan]["env"], "").strip()
        data = {
            "mode": "subscription" if price else "payment",
            "success_url": success_url,
            "cancel_url": cancel_url,
            "line_items[0][quantity]": "1",
        }
        if price:
            data["line_items[0][price]"] = price
        else:
            data["line_items[0][price_data][currency]"] = "usd"
            data["line_items[0][price_data][unit_amount]"] = str(PLAN_PRICES[plan]["usd"] * 100)
            data["line_items[0][price_data][product_data][name]"] = f"NSM AIaaS {plan}"
        if customer_email:
            data["customer_email"] = customer_email
        body = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in data.items())

        req = urllib.request.Request(
            "https://api.stripe.com/v1/checkout/sessions",
            data=body.encode("utf-8"),
            headers={
                "Authorization": f"Bearer {secret}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        (BILL / f"session_{payload.get('id', 'x')}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return {
            "ok": True,
            "mode": "stripe",
            "plan": plan,
            "checkout_url": payload.get("url"),
            "session_id": payload.get("id"),
        }
    except Exception as e:
        return {"ok": False, "mode": "stripe", "error": str(e)}


def handle_billing_command(user_input: str) -> Optional[str]:
    import re
    text = (user_input or "").strip()
    if not text:
        return None
    if not re.search(r"(stripe|فوتر[ةه]|اشتراك\s*مدفوع|checkout|بواب[ةه]\s*دفع)", text, re.I):
        return None
    plan = "pro"
    if re.search(r"enterprise|مؤسس", text, re.I):
        plan = "enterprise"
    r = create_checkout_session(plan=plan)
    return "## 💳 فوترة / Stripe\n```json\n" + json.dumps(r, ensure_ascii=False, indent=2, default=str)[:3000] + "\n```"
