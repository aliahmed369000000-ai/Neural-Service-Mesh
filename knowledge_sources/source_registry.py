"""
Knowledge Sources — Source Registry
=====================================
Central registry that tracks every registered knowledge source.
Persisted to disk as JSON so sources survive restarts.

Each source entry contains the full SourceMetadata descriptor.
The registry is thread-safe and supports hot-reload.
"""
from __future__ import annotations

import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from knowledge_sources.source_metadata import (
    SourceMetadata, SourceStatus, SourceType
)

logger = logging.getLogger(__name__)

_REGISTRY_PATH = Path("./data/source_registry.json")


class SourceRegistry:
    """
    Persistent registry of all knowledge sources.

    Sources are keyed by their `id` (UUID string).
    The registry file is written atomically on every mutation.
    """

    def __init__(self, registry_path: str = str(_REGISTRY_PATH)):
        self._path   = Path(registry_path)
        self._lock   = threading.RLock()
        self._sources: Dict[str, SourceMetadata] = {}
        self._load()
        logger.info(f"[SourceRegistry] loaded {len(self._sources)} sources")

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                with open(self._path) as f:
                    raw = json.load(f)
                for entry in raw.get("sources", []):
                    try:
                        meta = SourceMetadata.from_dict(entry)
                        self._sources[meta.id] = meta
                    except Exception as exc:
                        logger.warning(f"[SourceRegistry] skip bad entry: {exc}")
        except Exception as exc:
            logger.warning(f"[SourceRegistry] load error: {exc}")

    def _persist(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            payload = {
                "_meta": {
                    "schema_version": "1.0.0",
                    "total_sources":  len(self._sources),
                    "updated_at":     datetime.now(timezone.utc).isoformat(),
                },
                "sources": [s.to_dict() for s in self._sources.values()],
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            tmp.replace(self._path)
        except Exception as exc:
            logger.error(f"[SourceRegistry] persist error: {exc}")

    # ── CRUD ───────────────────────────────────────────────────────────────

    def register(self, meta: SourceMetadata) -> SourceMetadata:
        """Register a new source (or update an existing one by id)."""
        with self._lock:
            if meta.id in self._sources:
                logger.info(f"[SourceRegistry] updating existing source: {meta.id}")
            else:
                logger.info(f"[SourceRegistry] registering new source: {meta.name} ({meta.id})")
            meta.status = SourceStatus.ACTIVE
            self._sources[meta.id] = meta
            self._persist()
            return meta

    def unregister(self, source_id: str) -> bool:
        """Remove a source from the registry."""
        with self._lock:
            if source_id in self._sources:
                name = self._sources[source_id].name
                del self._sources[source_id]
                self._persist()
                logger.info(f"[SourceRegistry] unregistered: {name}")
                return True
            return False

    def get(self, source_id: str) -> Optional[SourceMetadata]:
        with self._lock:
            return self._sources.get(source_id)

    def get_by_name(self, name: str) -> Optional[SourceMetadata]:
        with self._lock:
            for s in self._sources.values():
                if s.name.lower() == name.lower():
                    return s
            return None

    def list_all(self) -> List[SourceMetadata]:
        with self._lock:
            return list(self._sources.values())

    def list_by_type(self, source_type: SourceType) -> List[SourceMetadata]:
        with self._lock:
            return [s for s in self._sources.values() if s.source_type == source_type]

    def list_active(self) -> List[SourceMetadata]:
        with self._lock:
            return [s for s in self._sources.values() if s.status == SourceStatus.ACTIVE]

    # ── Status Updates ─────────────────────────────────────────────────────

    def update_status(self, source_id: str, status: SourceStatus) -> bool:
        with self._lock:
            if source_id not in self._sources:
                return False
            self._sources[source_id].status = status
            self._persist()
            return True

    def mark_sync_start(self, source_id: str) -> bool:
        with self._lock:
            s = self._sources.get(source_id)
            if not s:
                return False
            s.mark_sync_start()
            self._persist()
            return True

    def mark_sync_done(
        self,
        source_id: str,
        items_ingested: int = 0,
        items_rejected: int = 0,
        avg_quality: float = 0.0,
    ) -> bool:
        with self._lock:
            s = self._sources.get(source_id)
            if not s:
                return False
            s.mark_sync_done(items_ingested, items_rejected)
            if avg_quality > 0:
                s.avg_quality_score = round(
                    (s.avg_quality_score + avg_quality) / 2, 2
                )
            self._persist()
            return True

    def mark_error(self, source_id: str, error_msg: str) -> bool:
        with self._lock:
            s = self._sources.get(source_id)
            if not s:
                return False
            s.mark_error(error_msg)
            self._persist()
            return True

    # ── Queries ────────────────────────────────────────────────────────────

    def has(self, source_id: str) -> bool:
        return source_id in self._sources

    def count(self) -> int:
        return len(self._sources)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            by_type: Dict[str, int] = {}
            by_status: Dict[str, int] = {}
            for s in self._sources.values():
                by_type[s.source_type.value]   = by_type.get(s.source_type.value, 0) + 1
                by_status[s.status.value]       = by_status.get(s.status.value, 0) + 1

            return {
                "total_sources":  len(self._sources),
                "by_type":        by_type,
                "by_status":      by_status,
                "registry_path":  str(self._path),
            }
