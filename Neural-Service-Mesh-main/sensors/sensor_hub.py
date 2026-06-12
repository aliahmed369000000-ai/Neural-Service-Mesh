"""
Phase 7 – Sensor Hub
======================
Central coordinator for the Sensory Layer.

Responsibilities:
  1. Holds all registered sensors
  2. Periodically calls sensor.safe_observe() (background thread)
  3. Routes events to registered event listeners
  4. Feeds events into the EnvironmentModel (world_model)
  5. Provides summary / history API

Usage:
    hub = SensorHub()
    hub.register(APISensor(config={"endpoints": [...]}))
    hub.register(FilesystemSensor())
    hub.register(LogSensor())
    hub.on_event(lambda e: print(e.to_dict()))
    hub.start(interval_s=10)
    ...
    hub.stop()
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable, Deque, Dict, List, Optional

from .base_sensor import BaseSensor, SensorEvent

logger = logging.getLogger(__name__)


class SensorHub:
    """
    Central event bus for Phase 7 sensors.

    Runs a background polling loop; all sensor events flow through here
    before being dispatched to the EnvironmentModel and EvolutionPipeline.
    """

    def __init__(self, interval_s: float = 15.0, max_history: int = 1000):
        self._sensors: Dict[str, BaseSensor] = {}
        self._listeners: List[Callable[[SensorEvent], None]] = []
        self._event_history: Deque[SensorEvent] = deque(maxlen=max_history)
        self._interval_s = interval_s
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._poll_count = 0
        self._total_events = 0
        self._started_at: Optional[str] = None

    # ── Sensor registration ────────────────────────────────────────────────

    def register(self, sensor: BaseSensor):
        self._sensors[sensor.sensor_id] = sensor
        logger.info(f"[SensorHub] registered sensor '{sensor.name}' ({sensor.sensor_type})")

    def unregister(self, sensor_id: str):
        self._sensors.pop(sensor_id, None)

    # ── Event listeners ────────────────────────────────────────────────────

    def on_event(self, callback: Callable[[SensorEvent], None]):
        """Register a global event listener (called for every sensor event)."""
        self._listeners.append(callback)

    # ── Polling loop ───────────────────────────────────────────────────────

    def _dispatch(self, event: SensorEvent):
        self._event_history.append(event)
        self._total_events += 1
        for listener in self._listeners:
            try:
                listener(event)
            except Exception as exc:
                logger.warning(f"[SensorHub] listener error: {exc}")

    def _poll_once(self):
        for sensor in list(self._sensors.values()):
            try:
                events = sensor.safe_observe()
                for event in events:
                    self._dispatch(event)
            except Exception as exc:
                logger.error(f"[SensorHub] poll error ({sensor.name}): {exc}")
        self._poll_count += 1

    def _loop(self):
        logger.info("[SensorHub] polling loop started")
        while self._running:
            self._poll_once()
            time.sleep(self._interval_s)
        logger.info("[SensorHub] polling loop stopped")

    def start(self, interval_s: Optional[float] = None):
        if self._running:
            return
        if interval_s:
            self._interval_s = interval_s
        self._running = True
        self._started_at = datetime.now(timezone.utc).isoformat()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="SensorHub")
        self._thread.start()
        logger.info(f"[SensorHub] started (interval={self._interval_s}s, sensors={len(self._sensors)})")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("[SensorHub] stopped")

    def poll_now(self) -> List[SensorEvent]:
        """Synchronous single poll — useful for tests or manual trigger."""
        events_before = self._total_events
        self._poll_once()
        new_events = list(self._event_history)
        # Return only newly added events
        delta = self._total_events - events_before
        return list(self._event_history)[-delta:] if delta else []

    # ── History & summary ──────────────────────────────────────────────────

    def recent_events(self, limit: int = 50, severity: Optional[str] = None) -> List[dict]:
        events = list(self._event_history)
        if severity:
            events = [e for e in events if e.severity == severity]
        return [e.to_dict() for e in events[-limit:]]

    def summary(self) -> dict:
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for e in self._event_history:
            by_type[e.event_type] = by_type.get(e.event_type, 0) + 1
            by_severity[e.severity] = by_severity.get(e.severity, 0) + 1
        return {
            "sensors_registered": len(self._sensors),
            "running": self._running,
            "poll_count": self._poll_count,
            "total_events": self._total_events,
            "interval_s": self._interval_s,
            "started_at": self._started_at,
            "events_by_type": by_type,
            "events_by_severity": by_severity,
            "sensors": [s.summary() for s in self._sensors.values()],
        }
