"""
mcp_auth.py
============
طبقة مصادقة بسيطة بمفاتيح API + حدود استخدام يومية لخادم mcp_server.py.

كل مفتاح API له خطة (plan) بحد أقصى من الاستدعاءات يومياً. يُخزَّن ملف
المفاتيح في data/api_keys.json (يُنشأ تلقائياً بمفتاح تجريبي عند أول تشغيل
إن لم يوجد).

هذا تصميم أولي (MVP) مناسب لخادم عملية واحدة (single-process):
  - عدّاد الاستخدام في الذاكرة، يُصفَّر تلقائياً كل يوم (بحسب تاريخ UTC)
  - لا يصلح لتوسّع أفقي (عدة نسخ من الخادم) بدون قاعدة بيانات مشتركة —
    يكفي تماماً لخادم واحد على HF Spaces/Render في هذه المرحلة

لإصدار مفتاح جديد يدوياً:
    python mcp_auth.py issue --name "اسم العميل" --plan pro
"""

from __future__ import annotations

import argparse
import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
KEYS_FILE = DATA_DIR / "api_keys.json"

# ── الخطط وحدودها اليومية ────────────────────────────────────────────────
PLANS: Dict[str, dict] = {
    "free": {"daily_limit": 50},
    "pro": {"daily_limit": 2000},
    "unlimited": {"daily_limit": None},
}

_lock = threading.Lock()
# {api_key: {"date": "YYYY-MM-DD", "count": int}}
_usage: Dict[str, dict] = {}


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _load_keys() -> Dict[str, dict]:
    if not KEYS_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        default_key = "nsm_" + secrets.token_urlsafe(24)
        data = {
            default_key: {
                "name": "مفتاح تجريبي افتراضي — استبدله في الإنتاج",
                "plan": "free",
                "created_at": _today(),
                "active": True,
            }
        }
        KEYS_FILE.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return data
    try:
        return json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_keys(data: Dict[str, dict]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


class AuthResult:
    def __init__(self, ok: bool, reason: str = "", status_code: int = 200):
        self.ok = ok
        self.reason = reason
        self.status_code = status_code


def check_api_key(api_key: Optional[str]) -> AuthResult:
    """يتحقق من صلاحية المفتاح والحد اليومي، ويزيد العدّاد عند النجاح."""
    if not api_key:
        return AuthResult(False, "مفتاح API مفقود — أرسله عبر ترويسة X-API-Key", 401)

    keys = _load_keys()
    record = keys.get(api_key)
    if not record or not record.get("active", True):
        return AuthResult(False, "مفتاح API غير صالح", 401)

    plan = PLANS.get(record.get("plan", "free"), PLANS["free"])
    limit = plan["daily_limit"]

    with _lock:
        today = _today()
        entry = _usage.get(api_key)
        if not entry or entry["date"] != today:
            entry = {"date": today, "count": 0}
        if limit is not None and entry["count"] >= limit:
            _usage[api_key] = entry
            return AuthResult(
                False, f"تم تجاوز الحد اليومي ({limit} استدعاء) لهذا المفتاح", 429
            )
        entry["count"] += 1
        _usage[api_key] = entry

    return AuthResult(True)


def issue_key(name: str, plan: str = "free") -> str:
    """يصدر مفتاح API جديداً ويحفظه في data/api_keys.json."""
    if plan not in PLANS:
        raise ValueError(f"خطة غير معروفة: {plan}. الخطط المتاحة: {list(PLANS)}")
    keys = _load_keys()
    new_key = "nsm_" + secrets.token_urlsafe(24)
    keys[new_key] = {
        "name": name,
        "plan": plan,
        "created_at": _today(),
        "active": True,
    }
    _save_keys(keys)
    return new_key


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="إدارة مفاتيح API لخادم NSM MCP")
    sub = parser.add_subparsers(dest="command", required=True)

    issue_p = sub.add_parser("issue", help="إصدار مفتاح جديد")
    issue_p.add_argument("--name", required=True, help="اسم العميل/الاستخدام")
    issue_p.add_argument("--plan", default="free", choices=list(PLANS))

    args = parser.parse_args()
    if args.command == "issue":
        key = issue_key(args.name, args.plan)
        print(f"تم إصدار مفتاح جديد ({args.plan}) لـ '{args.name}':")
        print(key)
