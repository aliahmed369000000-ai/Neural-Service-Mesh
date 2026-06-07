"""FilesystemSensor — monitors file system changes in watched directories."""
from __future__ import annotations
import os
import hashlib
import logging
from typing import List, Dict, Optional
from .base_sensor import BaseSensor, SensorEvent

logger = logging.getLogger("NeuralServiceMesh.FilesystemSensor")


class FilesystemSensor(BaseSensor):
    """Watches directories for file changes and emits SensorEvents."""

    def __init__(self, name: str = "FilesystemSensor", config: Optional[dict] = None):
        super().__init__(name=name, sensor_type="filesystem", config=config or {})
        self._watch_paths: List[str] = self.config.get("watch_paths", ["./ai", "./services"])
        self._extensions: List[str]  = self.config.get("extensions", [".py"])
        self._snapshots: Dict[str, str] = {}
        self._scan_count = 0
        self._changes_detected = 0

    def _md5(self, path: str) -> str:
        try:
            with open(path, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()
        except Exception:
            return ""

    def observe(self) -> List[SensorEvent]:
        events = []
        current: Dict[str, str] = {}

        for watch_dir in self._watch_paths:
            if not os.path.isdir(watch_dir):
                continue
            for root, _, files in os.walk(watch_dir):
                for fname in files:
                    if any(fname.endswith(ext) for ext in self._extensions):
                        fpath = os.path.join(root, fname)
                        md5   = self._md5(fpath)
                        current[fpath] = md5
                        if fpath not in self._snapshots:
                            events.append(SensorEvent(
                                sensor_name=self.name,
                                event_type="file_created",
                                payload={"path": fpath},
                                severity="info",
                            ))
                        elif self._snapshots[fpath] != md5:
                            events.append(SensorEvent(
                                sensor_name=self.name,
                                event_type="file_modified",
                                payload={"path": fpath},
                                severity="info",
                            ))

        for fpath in list(self._snapshots):
            if fpath not in current:
                events.append(SensorEvent(
                    sensor_name=self.name,
                    event_type="file_deleted",
                    payload={"path": fpath},
                    severity="warning",
                ))

        self._snapshots = current
        self._scan_count += 1
        self._changes_detected += len(events)
        return events

    def summary(self) -> dict:
        return {
            "name":             self.name,
            "watch_paths":      self._watch_paths,
            "tracked_files":    len(self._snapshots),
            "scan_count":       self._scan_count,
            "changes_detected": self._changes_detected,
        }

