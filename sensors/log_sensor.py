"""
Phase 7 – Log Sensor
======================
Tails log files and emits events when patterns are matched.

Config keys:
  log_paths:    list of str   (default: ["./logs"])
  patterns:     list of dict  {"name": str, "pattern": str, "severity": str}
  tail_lines:   int           (default: 100 – lines to scan per observe cycle)
"""
from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

from .base_sensor import BaseSensor, SensorEvent

logger = logging.getLogger(__name__)

_DEFAULT_PATTERNS = [
    {"name": "error",     "pattern": r"\bERROR\b",     "severity": "error"},
    {"name": "critical",  "pattern": r"\bCRITICAL\b",  "severity": "critical"},
    {"name": "exception", "pattern": r"(Exception|Traceback)", "severity": "error"},
    {"name": "warning",   "pattern": r"\bWARNING\b",   "severity": "warning"},
    {"name": "timeout",   "pattern": r"\btimeout\b",   "severity": "warning"},
    {"name": "oom",       "pattern": r"(MemoryError|OOM|out of memory)", "severity": "critical"},
]


class LogSensor(BaseSensor):
    """
    Scans log files for error/warning patterns.

    Emits:
      log_pattern_match – when a configured pattern fires
    """

    def __init__(self, name: str = "LogSensor", config: Optional[dict] = None):
        super().__init__(name=name, sensor_type="log", config=config or {})
        self.log_paths: List[str] = self.config.get("log_paths", ["./logs"])
        self.tail_lines: int = int(self.config.get("tail_lines", 100))
        raw_patterns = self.config.get("patterns", _DEFAULT_PATTERNS)
        self._patterns = [(p["name"], re.compile(p["pattern"], re.IGNORECASE), p.get("severity", "warning"))
                          for p in raw_patterns]
        self._file_positions: Dict[str, int] = {}  # path → last byte offset

    def _find_log_files(self) -> List[str]:
        files = []
        for p in self.log_paths:
            if not os.path.exists(p):
                continue
            if os.path.isfile(p):
                files.append(p)
                continue
            for fn in os.listdir(p):
                if fn.endswith((".log", ".txt", ".jsonl")):
                    files.append(os.path.join(p, fn))
        return files

    def _read_new_lines(self, path: str) -> List[str]:
        try:
            size = os.path.getsize(path)
            pos = self._file_positions.get(path, max(0, size - 8192))
            if pos > size:  # file rotated
                pos = 0
            with open(path, "r", errors="ignore") as f:
                f.seek(pos)
                data = f.read()
                self._file_positions[path] = f.tell()
            return data.splitlines()[-self.tail_lines:]
        except Exception as exc:
            logger.debug(f"[{self.name}] cannot read {path}: {exc}")
            return []

    def observe(self) -> List[SensorEvent]:
        events: List[SensorEvent] = []
        for log_file in self._find_log_files():
            lines = self._read_new_lines(log_file)
            for line in lines:
                for pname, regex, severity in self._patterns:
                    if regex.search(line):
                        events.append(SensorEvent(
                            event_type="log_pattern_match",
                            severity=severity,
                            payload={
                                "file": log_file,
                                "pattern": pname,
                                "line": line[:300],
                            },
                        ))
                        break  # one event per line
        return events

    def add_pattern(self, name: str, pattern: str, severity: str = "warning"):
        self._patterns.append((name, re.compile(pattern, re.IGNORECASE), severity))
