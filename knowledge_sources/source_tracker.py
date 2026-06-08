"""
Knowledge Sources — Source Tracker
=====================================
Tracks every sync operation, item ingestion, and source health event
for the full knowledge pipeline.

Responsibilities:
  - Record every sync result with timestamp + stats
  - Track per-source cumulative stats (total ingested, avg quality, errors)
  - Provide health status per source (healthy / degraded / failing)
  - Expose a summary for the Dashboard and main.py --mode source-status
  - Persist to disk so history survives restarts

Storage: knowledge/source_tracking.json (via KnowledgeStore.write_custom)

Usage:
    from knowledge_sources.source_tracker import SourceTracker
    tracker = SourceTracker()
    tracker.record_sync(result)           # after every sync
    tracker.record_item(item, meta)       # after every ingested item
    print(tracker.health_report())        # full status
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_TRACKING_FILE = Path("./knowledge/source_tracking.json")
_ITEMS_FILE    = Path("./knowledge/source_items_log.json")
_MAX_SYNC_HISTORY = 50   # per source
_MAX_ITEM_LOG     = 500  # global recent items


# ── Health Status ──────────────────────────────────────────────────────────

class SourceHealth:
    HEALTHY  = "healthy"
    DEGRADED = "degraded"
    FAILING  = "failing"
    UNKNOWN  = "unknown"


# ── Source Tracker ─────────────────────────────────────────────────────────

class SourceTracker:
    """
    Persistent tracker for all knowledge source activity.
    Thread-safe.
    """

    def __init__(
        self,
        tracking_file: Path = _TRACKING_FILE,
        items_file:    Path = _ITEMS_FILE,
    ):
        self._tracking_file = tracking_file
        self._items_file    = items_file
        self._lock          = threading.RLock()

        # In-memory state
        self._sources:      Dict[str, Dict[str, Any]] = {}   # source_id → stats
        self._sync_history: Dict[str, List[Dict]]     = {}   # source_id → [results]
        self._recent_items: List[Dict[str, Any]]      = []   # last N ingested items

        self._load()
        logger.info(
            f"[SourceTracker] loaded {len(self._sources)} sources, "
            f"{len(self._recent_items)} recent items"
        )

    # ── Persistence ────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._tracking_file.exists():
                with open(self._tracking_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._sources      = data.get("sources", {})
                self._sync_history = data.get("sync_history", {})
        except Exception as exc:
            logger.warning(f"[SourceTracker] tracking load error: {exc}")

        try:
            if self._items_file.exists():
                with open(self._items_file, encoding="utf-8") as f:
                    data = json.load(f)
                self._recent_items = data.get("recent_items", [])
        except Exception as exc:
            logger.warning(f"[SourceTracker] items log load error: {exc}")

    def _persist(self) -> None:
        try:
            self._tracking_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._tracking_file.with_suffix(".tmp")
            payload = {
                "_meta": {
                    "updated_at":     _now(),
                    "total_sources":  len(self._sources),
                    "schema_version": "1.0",
                },
                "sources":      self._sources,
                "sync_history": self._sync_history,
            }
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
            tmp.replace(self._tracking_file)
        except Exception as exc:
            logger.error(f"[SourceTracker] persist error: {exc}")

    def _persist_items(self) -> None:
        try:
            self._items_file.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._items_file.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(
                    {"recent_items": self._recent_items[-_MAX_ITEM_LOG:]},
                    f, ensure_ascii=False, indent=2, default=str
                )
            tmp.replace(self._items_file)
        except Exception as exc:
            logger.error(f"[SourceTracker] items persist error: {exc}")

    # ── Recording ──────────────────────────────────────────────────────────

    def record_sync(self, result: Any) -> None:
        """
        Record a completed SyncResult from SourceManager.
        Accepts either a SyncResult object or a plain dict.
        """
        with self._lock:
            d = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            sid  = d.get("source_id", "unknown")
            name = d.get("source_name", sid)

            # Init source entry
            if sid not in self._sources:
                self._sources[sid] = {
                    "source_id":         sid,
                    "source_name":       name,
                    "first_sync":        d.get("started_at", _now()),
                    "last_sync":         None,
                    "total_syncs":       0,
                    "successful_syncs":  0,
                    "failed_syncs":      0,
                    "total_fetched":     0,
                    "total_validated":   0,
                    "total_ingested":    0,
                    "total_rejected":    0,
                    "cumulative_quality": 0.0,
                    "avg_quality":       0.0,
                    "last_error":        None,
                    "health":            SourceHealth.UNKNOWN,
                    "consecutive_errors": 0,
                }

            s = self._sources[sid]
            s["last_sync"]       = d.get("finished_at", _now())
            s["total_syncs"]    += 1
            s["total_fetched"]  += d.get("items_fetched",   0)
            s["total_validated"]+= d.get("items_validated", 0)
            s["total_ingested"] += d.get("items_ingested",  0)
            s["total_rejected"] += d.get("items_rejected",  0)
            s["source_name"]     = name  # keep up to date

            # Quality running average
            q = d.get("avg_quality", 0.0)
            if q > 0:
                s["cumulative_quality"] += q
                s["avg_quality"] = round(
                    s["cumulative_quality"] / s["total_syncs"], 2
                )

            # Success / error tracking
            errors = d.get("errors", [])
            if d.get("success", len(errors) == 0):
                s["successful_syncs"]  += 1
                s["consecutive_errors"] = 0
                s["health"] = SourceHealth.HEALTHY
            else:
                s["failed_syncs"]      += 1
                s["consecutive_errors"] = s.get("consecutive_errors", 0) + 1
                s["last_error"] = errors[-1] if errors else "unknown error"
                if s["consecutive_errors"] >= 3:
                    s["health"] = SourceHealth.FAILING
                else:
                    s["health"] = SourceHealth.DEGRADED

            # Store in per-source sync history
            if sid not in self._sync_history:
                self._sync_history[sid] = []
            self._sync_history[sid].append(d)
            # Keep only last N
            if len(self._sync_history[sid]) > _MAX_SYNC_HISTORY:
                self._sync_history[sid] = self._sync_history[sid][-_MAX_SYNC_HISTORY:]

            self._persist()
            logger.info(
                f"[SourceTracker] recorded sync: {name} — "
                f"ingested={d.get('items_ingested',0)} "
                f"quality={d.get('avg_quality',0):.1f} "
                f"health={s['health']}"
            )

    def record_item(self, item: Any, meta: Any) -> None:
        """
        Record a single ingested KnowledgeItem for the recent items log.
        item: KnowledgeItem or dict
        meta: SourceMetadata or dict
        """
        with self._lock:
            item_d = item.to_dict() if hasattr(item, "to_dict") else dict(item)
            meta_d = meta.to_dict() if hasattr(meta, "to_dict") else dict(meta)

            entry = {
                "timestamp":     _now(),
                "source_id":     meta_d.get("id", ""),
                "source_name":   meta_d.get("name", ""),
                "source_type":   str(meta_d.get("source_type", "")),
                "item_id":       item_d.get("item_id", ""),
                "reference":     item_d.get("raw_reference", ""),
                "quality_score": item_d.get("quality_score", 0.0),
                "trust_score":   item_d.get("trust_score", 0.0),
                "tags":          item_d.get("derived_tags", [])[:5],
                "concepts":      item_d.get("derived_concepts", [])[:5],
            }
            self._recent_items.append(entry)
            if len(self._recent_items) > _MAX_ITEM_LOG:
                self._recent_items = self._recent_items[-_MAX_ITEM_LOG:]
            # Persist every 100 items to avoid excessive I/O
            if len(self._recent_items) % 100 == 0:
                self._persist_items()

    def flush_items(self) -> None:
        """Force persist the recent items log."""
        with self._lock:
            self._persist_items()

    # ── Query API ──────────────────────────────────────────────────────────

    def get_source_stats(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Return cumulative stats for one source."""
        return self._sources.get(source_id)

    def all_source_stats(self) -> List[Dict[str, Any]]:
        """Return stats for all tracked sources."""
        return list(self._sources.values())

    def get_sync_history(
        self, source_id: str, last_n: int = 10
    ) -> List[Dict[str, Any]]:
        """Return last N sync results for a source."""
        history = self._sync_history.get(source_id, [])
        return history[-last_n:]

    def get_recent_items(self, n: int = 20) -> List[Dict[str, Any]]:
        """Return the N most recently ingested items."""
        return self._recent_items[-n:]

    def health_report(self) -> Dict[str, Any]:
        """Full health report — used by Dashboard and source-status mode."""
        with self._lock:
            sources_report = []
            for sid, s in self._sources.items():
                history = self._sync_history.get(sid, [])
                last_sync = history[-1] if history else {}
                sources_report.append({
                    "source_id":        sid,
                    "source_name":      s["source_name"],
                    "health":           s["health"],
                    "total_syncs":      s["total_syncs"],
                    "successful_syncs": s["successful_syncs"],
                    "failed_syncs":     s["failed_syncs"],
                    "total_ingested":   s["total_ingested"],
                    "total_rejected":   s["total_rejected"],
                    "avg_quality":      s["avg_quality"],
                    "last_sync":        s["last_sync"],
                    "last_error":       s["last_error"],
                    "consecutive_errors": s.get("consecutive_errors", 0),
                    "last_sync_fetched":    last_sync.get("items_fetched", 0),
                    "last_sync_ingested":   last_sync.get("items_ingested", 0),
                    "last_sync_quality":    last_sync.get("avg_quality", 0.0),
                })

            healthy  = sum(1 for s in self._sources.values() if s["health"] == SourceHealth.HEALTHY)
            degraded = sum(1 for s in self._sources.values() if s["health"] == SourceHealth.DEGRADED)
            failing  = sum(1 for s in self._sources.values() if s["health"] == SourceHealth.FAILING)

            return {
                "generated_at":     _now(),
                "total_sources":    len(self._sources),
                "healthy":          healthy,
                "degraded":         degraded,
                "failing":          failing,
                "total_ingested":   sum(s["total_ingested"] for s in self._sources.values()),
                "total_rejected":   sum(s["total_rejected"] for s in self._sources.values()),
                "recent_items_count": len(self._recent_items),
                "sources":          sources_report,
            }

    def summary(self) -> Dict[str, Any]:
        """Compact summary for NeuralServiceMesh.status()."""
        with self._lock:
            return {
                "tracked_sources":  len(self._sources),
                "total_ingested":   sum(s["total_ingested"] for s in self._sources.values()),
                "healthy_sources":  sum(
                    1 for s in self._sources.values() if s["health"] == SourceHealth.HEALTHY
                ),
            }

    def __repr__(self) -> str:
        s = self.summary()
        return (
            f"<SourceTracker sources={s['tracked_sources']} "
            f"ingested={s['total_ingested']} "
            f"healthy={s['healthy_sources']}>"
        )


# ── Helpers ────────────────────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
