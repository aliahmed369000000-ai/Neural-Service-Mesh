"""
Phase 7 – API Sensor
======================
Monitors HTTP endpoints and detects errors, latency spikes, and unavailability.

Config keys:
  endpoints: list of {"url": str, "method": str, "name": str}
  latency_threshold_ms: int  (default 2000)
  timeout_s: float           (default 5.0)
"""
from __future__ import annotations

import logging
import time
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any

from .base_sensor import BaseSensor, SensorEvent

logger = logging.getLogger(__name__)


class APISensor(BaseSensor):
    """
    Monitors a list of HTTP endpoints.

    Emits events for:
      - api_error        : non-2xx response
      - api_timeout      : request timed out
      - api_slow         : response time exceeded threshold
      - api_ok           : healthy probe (info level)
    """

    def __init__(self, name: str = "APISensor", config: Optional[dict] = None):
        super().__init__(name=name, sensor_type="api", config=config or {})
        self.endpoints: List[Dict[str, Any]] = self.config.get("endpoints", [])
        self.latency_threshold_ms: float = float(self.config.get("latency_threshold_ms", 2000))
        self.timeout_s: float = float(self.config.get("timeout_s", 5.0))
        self._history: Dict[str, List[dict]] = {}

    def add_endpoint(self, url: str, name: str = "", method: str = "GET"):
        self.endpoints.append({"url": url, "name": name or url, "method": method})

    def _probe(self, endpoint: dict) -> SensorEvent:
        url = endpoint["url"]
        label = endpoint.get("name", url)
        method = endpoint.get("method", "GET").upper()
        t0 = time.perf_counter()
        status_code = None
        error_msg = None

        try:
            req = urllib.request.Request(url, method=method)
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                status_code = resp.status
                latency_ms = (time.perf_counter() - t0) * 1000

            if status_code and status_code >= 400:
                return SensorEvent(
                    event_type="api_error",
                    severity="error",
                    payload={"endpoint": label, "url": url, "status_code": status_code,
                             "latency_ms": round(latency_ms, 1)},
                )
            if latency_ms > self.latency_threshold_ms:
                return SensorEvent(
                    event_type="api_slow",
                    severity="warning",
                    payload={"endpoint": label, "url": url, "status_code": status_code,
                             "latency_ms": round(latency_ms, 1),
                             "threshold_ms": self.latency_threshold_ms},
                )
            return SensorEvent(
                event_type="api_ok",
                severity="info",
                payload={"endpoint": label, "url": url, "status_code": status_code,
                         "latency_ms": round(latency_ms, 1)},
            )

        except urllib.error.URLError as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            error_msg = str(exc.reason)
            if "timed out" in error_msg.lower():
                return SensorEvent(
                    event_type="api_timeout",
                    severity="error",
                    payload={"endpoint": label, "url": url, "error": error_msg,
                             "latency_ms": round(latency_ms, 1)},
                )
            return SensorEvent(
                event_type="api_error",
                severity="error",
                payload={"endpoint": label, "url": url, "error": error_msg,
                         "latency_ms": round(latency_ms, 1)},
            )
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            return SensorEvent(
                event_type="api_error",
                severity="error",
                payload={"endpoint": label, "url": url, "error": str(exc),
                         "latency_ms": round(latency_ms, 1)},
            )

    def observe(self) -> List[SensorEvent]:
        events = []
        for ep in self.endpoints:
            event = self._probe(ep)
            # Track history
            key = ep.get("name", ep["url"])
            self._history.setdefault(key, []).append({
                "type": event.event_type,
                "severity": event.severity,
                "ts": event.timestamp,
            })
            self._history[key] = self._history[key][-50:]  # keep last 50
            events.append(event)
        return events

    def health_summary(self) -> dict:
        total = sum(len(v) for v in self._history.values())
        errors = sum(
            1 for v in self._history.values()
            for e in v if e["type"] in ("api_error", "api_timeout")
        )
        return {
            "endpoints_monitored": len(self.endpoints),
            "total_probes": total,
            "error_probes": errors,
            "error_rate": round(errors / total, 3) if total else 0.0,
        }
