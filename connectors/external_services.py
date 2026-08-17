# -*- coding: utf-8 -*-
"""External Services Connectors — موصلات خدمات خارجية بنمط ثابت.

يبني هذا الملف نظام موصلات للخدمات الخارجية (بوابات الدفع، الخرائط،
الرسائل النصية SMS) فوق عقد موحّد واحد. الهدفان:
1) تمكين المشروع من التكامل مع خدمات حقيقية مستقبلاً عبر مجرد تغيّر
   «التنفيذ» (swap-in) دون لمس أي كود آخر.
2) توفير تنفيذات محاكاة (mock) موثوقة تعمل الآن بدون مفاتيح API،
   حتى تُختبَر بها مسارات الوكلاء والخدمات المصغرة كاملة.

عقد الاستجابة الموحّد (Unified Envelope) لكل استدعاء:
    {"ok": bool, "action": str, "service": str, "request_id": str,
     "result": Any, "error": str|None, "simulated": bool,
     "latency_ms": float, "cost": str|None, "payload_raw": dict}

- `simulated=True` يعني دائماً أن النتيجة من المحاكي (لا تكلفة حقيقية).
- كل إجراء (action) له وصف و schema مدخلات في `describe()` — وهذا ما
  يجعل الموصلات قابلة للاكتشاف من MCP والخدمات المصغرة.
- لا مفاتيح API حقيقية داخل هذا الملف إطلاقاً.

الاستخدام:
    from connectors.external_services import get_connector, call_connector
    resp = call_connector("payment", "create_payment",
                          {"amount": 100.0, "currency": "USD"})
"""
from __future__ import annotations

import random
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

_lock = threading.Lock()


def _request_id(prefix: str) -> str:
    return f"{prefix}_{int(time.time() * 1000)}_{random.randint(1000, 9999)}"


def _envelope(ok: bool, service: str, action: str,
              result: Any = None, error: Optional[str] = None,
              simulated: bool = True) -> Dict[str, Any]:
    return {"ok": ok, "service": service, "action": action,
            "request_id": _request_id(service),
            "result": result, "error": error, "simulated": simulated,
            "latency_ms": round(random.uniform(5, 40), 2),
            "cost": None if ok else None,
            "payload_raw": None}


class AbstractExternalConnector(ABC):
    """العقد الأساسي لأي موصل خدمة خارجية."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def capabilities(self) -> List[str]:
        """قائمة الإجراءات المدعومة (بالإنجليزية snake_case)."""
        ...

    @abstractmethod
    def describe(self, action: str) -> Dict[str, Any]:
        """وصف إجراء: doc, input_schema (مفاتيح مطلوبة/اختيارية)،
        مثال طلب واستجابة."""
        ...

    @abstractmethod
    def call(self, action: str, payload: Dict[str, Any]
             ) -> Dict[str, Any]:
        """تنفيذ إجراء مع envelope موحّد. خطأ unknown action يعيد
        {"ok": False, ...} ولا يرمي."""
        ...


# ------------------------------------------------------------------ Payment
class PaymentConnector(AbstractExternalConnector):

    @property
    def name(self) -> str:
        return "payment"

    def capabilities(self) -> List[str]:
        return ["create_payment", "invoice", "payment_status", "refund"]

    def describe(self, action: str) -> Dict[str, Any]:
        docs = {
            "create_payment": {
                "doc": "إنشاء عملية دفع محاكاة (تحتاج بوابة حقيقية مستقبلاً)",
                "required": {"amount": "float", "currency": "str"},
                "optional": {"description": "str", "customer_id": "str"},
            },
            "invoice": {
                "doc": "إنشاء فاتورة محاكاة",
                "required": {"customer_id": "str", "items": "list"},
                "optional": {"currency": "str", "due_days": "int"},
            },
            "payment_status": {
                "doc": "الاستعلام عن حالة عملية دفع",
                "required": {"payment_id": "str"}, "optional": {},
            },
            "refund": {
                "doc": "استرداد عملية دفع (محاكاة)",
                "required": {"payment_id": "str"},
                "optional": {"reason": "str"},
            },
        }
        info = docs.get(action)
        if info is None:
            return {"ok": False, "error": f"إجراء غير معروف: {action}"}
        return {"ok": True, "action": action, **info}

    def call(self, action: str, payload: Dict[str, Any]
             ) -> Dict[str, Any]:
        try:
            if action == "create_payment":
                amount = float(payload.get("amount", 0))
                if amount <= 0:
                    return _envelope(False, self.name, action,
                                     error="المبلغ يجب أن يكون أكبر من صفر")
                pid = _request_id("pay")
                return _envelope(True, self.name, action,
                                 result={"payment_id": pid,
                                         "amount": amount,
                                         "currency": str(payload.get(
                                             "currency", "USD")),
                                         "status": "pending",
                                         "description": payload.get(
                                             "description", "")})
            if action == "invoice":
                if not payload.get("customer_id") or not isinstance(
                        payload.get("items"), list):
                    return _envelope(False, self.name, action,
                                     error="customer_id و items مطلوبة")
                iid = _request_id("inv")
                total = sum(float(i.get("price", 0) * i.get("qty", 1))
                            for i in payload["items"])
                return _envelope(True, self.name, action,
                                 result={"invoice_id": iid,
                                         "customer_id": payload[
                                             "customer_id"],
                                         "items": payload["items"],
                                         "total": round(total, 2),
                                         "currency": str(payload.get(
                                             "currency", "USD")),
                                         "status": "issued"})
            if action == "payment_status":
                pid = payload.get("payment_id")
                if not pid:
                    return _envelope(False, self.name, action,
                                     error="payment_id مطلوب")
                return _envelope(True, self.name, action,
                                 result={"payment_id": pid,
                                         "status": random.choice([
                                             "completed", "pending",
                                             "failed"])})
            if action == "refund":
                pid = payload.get("payment_id")
                if not pid:
                    return _envelope(False, self.name, action,
                                     error="payment_id مطلوب")
                return _envelope(True, self.name, action,
                                 result={"payment_id": pid,
                                         "refund_id": _request_id("ref"),
                                         "status": "refunded",
                                         "reason": payload.get("reason", "")})
            return _envelope(False, self.name, action,
                             error=f"إجراء غير معروف: {action}")
        except Exception as e:
            return _envelope(False, self.name, action, error=str(e))


# -------------------------------------------------------------------- Maps
class MapsConnector(AbstractExternalConnector):

    @property
    def name(self) -> str:
        return "maps"

    def capabilities(self) -> List[str]:
        return ["geocode", "reverse_geocode", "route", "distance"]

    def describe(self, action: str) -> Dict[str, Any]:
        docs = {
            "geocode": {
                "doc": "تحويل عنوان إلى إحداثيات (محاكاة)",
                "required": {"address": "str"}, "optional": {},
            },
            "reverse_geocode": {
                "doc": "تحويل إحداثيات إلى عنوان (محاكاة)",
                "required": {"lat": "float", "lng": "float"},
                "optional": {},
            },
            "route": {
                "doc": "حساب مسار بين نقطتين (محاكاة)",
                "required": {"from": "str", "to": "str"},
                "optional": {"mode": "str"},
            },
            "distance": {
                "doc": "حساب مسافة بين نقطتين (محاكاة)",
                "required": {"lat1": "float", "lng1": "float",
                             "lat2": "float", "lng2": "float"},
                "optional": {"unit": "str"},
            },
        }
        info = docs.get(action)
        if info is None:
            return {"ok": False, "error": f"إجراء غير معروف: {action}"}
        return {"ok": True, "action": action, **info}

    def call(self, action: str, payload: Dict[str, Any]
             ) -> Dict[str, Any]:
        try:
            if action == "geocode":
                addr = payload.get("address")
                if not addr:
                    return _envelope(False, self.name, action,
                                     error="address مطلوب")
                h = sum(ord(c) for c in str(addr)) % 1000
                return _envelope(True, self.name, action,
                                 result={"address": addr,
                                         "lat": round(24.0 + h / 100, 4),
                                         "lng": round(45.0 + h / 100, 4)})
            if action == "reverse_geocode":
                lat = payload.get("lat")
                lng = payload.get("lng")
                if lat is None or lng is None:
                    return _envelope(False, self.name, action,
                                     error="lat/lng مطلوبة")
                return _envelope(True, self.name, action,
                                 result={"lat": float(lat), "lng": float(lng),
                                         "address": f"simulated address "
                                                    f"near {lat},{lng}"})
            if action == "route":
                frm, to = payload.get("from"), payload.get("to")
                if not frm or not to:
                    return _envelope(False, self.name, action,
                                     error="from/to مطلوبان")
                km = random.uniform(2, 300)
                return _envelope(True, self.name, action,
                                 result={"from": frm, "to": to,
                                         "mode": str(payload.get(
                                             "mode", "driving")),
                                         "distance_km": round(km, 1),
                                         "duration_min": round(km * 1.2, 1)})
            if action == "distance":
                keys = ("lat1", "lng1", "lat2", "lng2")
                if any(k not in payload for k in keys):
                    return _envelope(False, self.name, action,
                                     error="lat1/lng1/lat2/lng2 مطلوبة")
                lat1, lng1 = float(payload["lat1"]), float(payload["lng1"])
                lat2, lng2 = float(payload["lat2"]), float(payload["lng2"])
                d = ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5 * 111.0
                unit = str(payload.get("unit", "km"))
                value = d if unit == "km" else d * 0.621371
                return _envelope(True, self.name, action,
                                 result={"distance": round(value, 2),
                                         "unit": unit})
            return _envelope(False, self.name, action,
                             error=f"إجراء غير معروف: {action}")
        except Exception as e:
            return _envelope(False, self.name, action, error=str(e))


# --------------------------------------------------------------------- SMS
class SMSConnector(AbstractExternalConnector):

    @property
    def name(self) -> str:
        return "sms"

    def capabilities(self) -> List[str]:
        return ["send_sms", "send_batch", "send_otp", "verify_otp"]

    def describe(self, action: str) -> Dict[str, Any]:
        docs = {
            "send_sms": {
                "doc": "إرسال رسالة نصية محاكاة",
                "required": {"to": "str", "message": "str"},
                "optional": {"sender": "str"},
            },
            "send_batch": {
                "doc": "إرسال رسالة لعدة أرقام (محاكاة)",
                "required": {"recipients": "list[str]", "message": "str"},
                "optional": {"sender": "str"},
            },
            "send_otp": {
                "doc": "توليد رمز تحقق وإرساله (محاكاة — الرمز يُعرض في النتيجة)",
                "required": {"to": "str"}, "optional": {"ttl_seconds": "int"},
            },
            "verify_otp": {
                "doc": "التحقق من رمز سبق توليده",
                "required": {"to": "str", "code": "str"}, "optional": {},
            },
        }
        info = docs.get(action)
        if info is None:
            return {"ok": False, "error": f"إجراء غير معروف: {action}"}
        return {"ok": True, "action": action, **info}

    def call(self, action: str, payload: Dict[str, Any]
             ) -> Dict[str, Any]:
        try:
            if action == "send_sms":
                to, msg = payload.get("to"), payload.get("message")
                if not to or not msg:
                    return _envelope(False, self.name, action,
                                     error="to/message مطلوبان")
                return _envelope(True, self.name, action,
                                 result={"to": to, "message": msg,
                                         "sender": str(payload.get(
                                             "sender", "NSM")),
                                         "status": "simulated_sent"})
            if action == "send_batch":
                recs = payload.get("recipients")
                msg = payload.get("message")
                if not isinstance(recs, list) or not recs or not msg:
                    return _envelope(False, self.name, action,
                                     error="recipients/message مطلوبة")
                return _envelope(True, self.name, action,
                                 result={"sent_to": recs, "message": msg,
                                         "count": len(recs),
                                         "status": "simulated_sent"})
            if action == "send_otp":
                to = payload.get("to")
                if not to:
                    return _envelope(False, self.name, action,
                                     error="to مطلوب")
                code = f"{random.randint(100000, 999999)}"
                with _lock:
                    _OTP_STORE[str(to)] = {
                        "code": code,
                        "expires": time.time() + float(
                            payload.get("ttl_seconds", 300))}
                return _envelope(True, self.name, action,
                                 result={"to": to, "code": code,
                                         "ttl_seconds": float(
                                             payload.get("ttl_seconds", 300)),
                                         "status": "simulated_sent"})
            if action == "verify_otp":
                to, code = payload.get("to"), payload.get("code")
                if not to or not code:
                    return _envelope(False, self.name, action,
                                     error="to/code مطلوبان")
                with _lock:
                    entry = _OTP_STORE.get(str(to))
                if entry is None:
                    return _envelope(False, self.name, action,
                                     error="لا يوجد رمز لهذا الرقم")
                if time.time() > entry["expires"]:
                    with _lock:
                        _OTP_STORE.pop(str(to), None)
                    return _envelope(False, self.name, action,
                                     error="الرمز منتهي الصلاحية")
                match = entry["code"] == str(code)
                if match:
                    with _lock:
                        _OTP_STORE.pop(str(to), None)
                return _envelope(True, self.name, action,
                                 result={"to": to, "valid": match,
                                         "status": "verified" if match
                                         else "invalid"})
            return _envelope(False, self.name, action,
                             error=f"إجراء غير معروف: {action}")
        except Exception as e:
            return _envelope(False, self.name, action, error=str(e))


# ---------------------------------------------------------------- Registry
_OTP_STORE: Dict[str, Dict[str, Any]] = {}

_CONNECTORS: Dict[str, AbstractExternalConnector] = {
    "payment": PaymentConnector(),
    "maps": MapsConnector(),
    "sms": SMSConnector(),
}


def get_connector(service: str) -> Optional[AbstractExternalConnector]:
    return _CONNECTORS.get(str(service).lower())


def list_connectors() -> List[Dict[str, Any]]:
    return [{"name": c.name, "capabilities": c.capabilities()}
            for c in _CONNECTORS.values()]


def call_connector(service: str, action: str,
                   payload: Optional[Dict[str, Any]] = None
                   ) -> Dict[str, Any]:
    """استدعاء إجراء على موصل بالاسم. موصل غير موجود → envelope خطأ."""
    connector = get_connector(service)
    if connector is None:
        return {"ok": False, "service": service, "action": action,
                "request_id": _request_id("ext"), "result": None,
                "error": f"موصل غير مسجل: {service}",
                "simulated": False, "latency_ms": 0.0,
                "cost": None, "payload_raw": None}
    result = connector.call(action, payload or {})
    result["payload_raw"] = payload or {}
    return result


def describe_connector(service: str) -> Dict[str, Any]:
    connector = get_connector(service)
    if connector is None:
        return {"ok": False, "error": f"موصل غير مسجل: {service}"}
    return {"ok": True, "name": connector.name,
            "capabilities": connector.capabilities(),
            "actions": {a: connector.describe(a)
                        for a in connector.capabilities()}}
