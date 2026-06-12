from __future__ import annotations
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class MeshLogger:
    """
    Phase 2 structured logging: console + rotating file logs under /logs/.
    """

    def __init__(self, log_dir: str = "./logs", level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._setup(level)

    def _setup(self, level: str):
        log_level = getattr(logging, level.upper(), logging.INFO)
        root = logging.getLogger()
        root.setLevel(log_level)

        # Clear existing handlers
        root.handlers.clear()

        fmt = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S"
        )

        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        ch.setLevel(log_level)
        root.addHandler(ch)

        # File handler (daily log file)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        log_file = self.log_dir / f"mesh_{today}.log"
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        fh.setLevel(log_level)
        root.addHandler(fh)

        logging.getLogger(__name__).info(
            f"MeshLogger ready — file={log_file}, level={level}"
        )

    def list_log_files(self):
        return sorted([f.name for f in self.log_dir.glob("*.log")])
