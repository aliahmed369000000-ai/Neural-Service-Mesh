"""
MCP Monetization — بوابة مفتاح مستأجر AIaaS قبل أدوات MCP المدفوعة
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple


def authenticate_mcp_key(api_key: Optional[str]) -> Tuple[bool, Dict[str, Any]]:
    """
    يتحقق من مفتاح مستأجر عبر aiaas_platform.
    إن لم يُمرَّر مفتاح وكان NSM_MCP_OPEN=1 يُسمح بالوصول المجاني المحلي.
    """
    if not api_key:
        if os.environ.get("NSM_MCP_OPEN", "").strip() in ("1", "true", "yes"):
            return True, {"plan": "open_dev", "tenant_id": "local"}
        return False, {"error": "api_key_required"}
    try:
        from ai.aiaas_platform import authenticate_api_key, check_quota
        ten = authenticate_api_key(api_key)
        if not ten:
            return False, {"error": "invalid_api_key"}
        ok, msg = check_quota(ten["id"])
        if not ok:
            return False, {"error": "quota_exceeded", "detail": msg, "tenant_id": ten["id"]}
        return True, {"tenant_id": ten["id"], "plan": ten.get("plan"), "name": ten.get("name")}
    except Exception as e:
        return False, {"error": str(e)}


def meter_usage(tenant_id: str, units: float = 1.0) -> None:
    try:
        from ai.aiaas_platform import record_usage
        record_usage(tenant_id, train_seconds=0.0, models_delta=0, job=True)
    except Exception:
        pass
