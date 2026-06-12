"""
Phase 7 – Webhook Sensor
==========================
Receives push events from external systems (GitHub, CI, alerting, etc.).

Events are queued by push_event() and drained during observe().
In production, push_event() is called from a Flask route.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from .base_sensor import BaseSensor, SensorEvent

logger = logging.getLogger(__name__)


class WebhookSensor(BaseSensor):
    """
    Buffer-based sensor for incoming webhooks.

    Usage:
        sensor = WebhookSensor()
        # In Flask route:
        @app.route("/webhook", methods=["POST"])
        def on_webhook():
            sensor.push_event("github_push", payload=request.json)
            return "ok"
    """

    def __init__(self, name: str = "WebhookSensor", config: Optional[dict] = None):
        super().__init__(name=name, sensor_type="webhook", config=config or {})
        self._queue: Deque[SensorEvent] = deque(maxlen=500)
        self._lock = threading.Lock()

    def push_event(
        self,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
        severity: str = "info",
    ):
        """Push a raw event into the sensor queue (called from webhook handlers)."""
        event = SensorEvent(
            event_type=event_type,
            severity=severity,
            payload=payload or {},
        )
        with self._lock:
            self._queue.append(event)
        logger.debug(f"[{self.name}] queued {event_type}")

    def observe(self) -> List[SensorEvent]:
        """Drain the queue and return all pending events."""
        with self._lock:
            events = list(self._queue)
            self._queue.clear()
        return events

    def pending_count(self) -> int:
        with self._lock:
            return len(self._queue)
