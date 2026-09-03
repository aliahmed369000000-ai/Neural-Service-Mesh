"""عقد مجردة لمزوّدي الأنفاق مع تحقق صحي محافظ."""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TunnelCandidate:
    name: str
    public_url: str
    process_alive: Callable[[], bool]


def health_url(public_url: str) -> str:
    return public_url.rstrip("/") + "/health"


def verify_tunnel(candidate: TunnelCandidate, timeout: float = 5.0) -> bool:
    """لا تعتبر النفق ناجحاً إلا إذا بقيت العملية حية وأجاب /health بصحة JSON."""
    if not candidate.public_url or not candidate.process_alive():
        return False
    try:
        request = urllib.request.Request(
            health_url(candidate.public_url),
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return False
            payload = json.loads(response.read().decode("utf-8"))
        return isinstance(payload, dict) and payload.get("status") in {"healthy", "ok", "Running"}
    except (OSError, urllib.error.URLError, ValueError, json.JSONDecodeError):
        return False


__all__ = ["TunnelCandidate", "health_url", "verify_tunnel"]
