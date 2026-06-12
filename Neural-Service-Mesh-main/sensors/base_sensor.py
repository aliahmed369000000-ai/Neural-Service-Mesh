"""
Phase 7 – Base Sensor
======================
Abstract base class for all sensory inputs.

Every sensor:
  - Emits SensorEvents to the SensorHub
  - Tracks its own health and last-fire time
  - Can be paused / resumed
"""
from __future__ import annotations

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SensorEvent:
    """A single observation from a sensor."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    sensor_id: str = ""
    sensor_type: str = ""
    event_type: str = ""          # api_error / file_change / log_pattern / metric_spike / etc.
    severity: str = "info"        # info / warning / error / critical
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processed: bool = False

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "sensor_id": self.sensor_id,
            "sensor_type": self.sensor_type,
            "event_type": self.event_type,
            "severity": self.severity,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "processed": self.processed,
        }


class BaseSensor(ABC):
    """
    Abstract base for all Phase 7 sensors.

    Subclasses implement `observe()` which returns a list of SensorEvents.
    The SensorHub calls observe() periodically and dispatches events.
    """

    def __init__(self, name: str, sensor_type: str, config: Optional[dict] = None):
        self.sensor_id = str(uuid.uuid4())[:12]
        self.name = name
        self.sensor_type = sensor_type
        self.config = config or {}
        self.enabled = True
        self.event_count = 0
        self.error_count = 0
        self.last_observe_at: Optional[str] = None
        self._callbacks: List[Callable[[SensorEvent], None]] = []

    def on_event(self, callback: Callable[[SensorEvent], None]):
        """Register a callback for events from this sensor."""
        self._callbacks.append(callback)

    def emit(self, event: SensorEvent):
        """Stamp sensor info onto event and fire callbacks."""
        event.sensor_id = self.sensor_id
        event.sensor_type = self.sensor_type
        self.event_count += 1
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as exc:
                logger.warning(f"[{self.name}] callback error: {exc}")

    @abstractmethod
    def observe(self) -> List[SensorEvent]:
        """Collect observations and return events. Called periodically by SensorHub."""
        ...

    def safe_observe(self) -> List[SensorEvent]:
        """Wraps observe() with error handling."""
        if not self.enabled:
            return []
        try:
            self.last_observe_at = datetime.now(timezone.utc).isoformat()
            events = self.observe()
            for e in events:
                self.emit(e)
            return events
        except Exception as exc:
            self.error_count += 1
            logger.error(f"[{self.name}] observe error: {exc}")
            return []

    def summary(self) -> dict:
        return {
            "sensor_id": self.sensor_id,
            "name": self.name,
            "sensor_type": self.sensor_type,
            "enabled": self.enabled,
            "event_count": self.event_count,
            "error_count": self.error_count,
            "last_observe_at": self.last_observe_at,
        }
