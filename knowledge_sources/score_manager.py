"""
Knowledge Sources — Source Manager
=====================================
Top-level orchestrator for the Knowledge Sources Layer.

Responsibilities:
  - Register / unregister sources
  - Trigger and schedule sync operations
  - Monitor source health and errors
  - Route validated + scored items to the memory engine & knowledge store
  - Expose a clean API to main.py and api/app.py

Pipeline for each item:
  Source Feeder
      ↓
  SourceValidator   (completeness / dedup / corruption / trust)
      ↓
  QualityScorer     (0–100 multi-factor score)
      ↓
  MemoryEngine      (route/node learning)
      ↓
  KnowledgeStore    (JSON persistence)
      ↓
  EnvironmentModel  (world model update)
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

from knowledge_sources.source_metadata import (
    KnowledgeItem, SourceMetadata, SourceStatus, SourceType,
    UpdateFrequency, AccessMode
)
from knowledge_sources.source_registry  import SourceRegistry
from knowledge_sources.source_validator import SourceValidator
from knowledge_sources.quality_scorer   import QualityScorer

if TYPE_CHECKING:
    from ai.memory_engine            import MemoryEngine
    from knowledge.knowledge_store   import KnowledgeStore
    from world_model.environment_model import EnvironmentModel
    from ai.semantic_matcher         import SemanticMatcher

logger = logging.getLogger(__name__)


# ── Sync Result ────────────────────────────────────────────────────────────

class SyncResult:
    def __init__(self, source_id: str, source_name: str):
        self.source_id   = source_id
        self.source_name = source_name
        self.started_at  = datetime.now(timezone.utc).isoformat()
        self.finished_at: Optional[str] = None
        self.items_fetched   = 0
        self.items_validated = 0
        self.items_scored    = 0
        self.items_ingested  = 0
        self.items_rejected  = 0
        self.avg_quality     = 0.0
        self.errors: List[str] = []
        self.success = False

    def finish(self) -> None:
        self.finished_at = datetime.now(timezone.utc).isoformat()
        self.success     = len(self.errors) == 0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__


# ── Source Manager ─────────────────────────────────────────────────────────

class SourceManager:
    """
    Central manager for all knowledge sources.

    Thread-safe. Each sync runs in its own daemon thread.
    """

    def __init__(
        self,
        registry:   Optional[SourceRegistry]  = None,
        validator:  Optional[SourceValidator] = None,
        scorer:     Optional[QualityScorer]   = None,
        min_quality_threshold: float = 30.0,
    ):
        self._registry  = registry  or SourceRegistry()
        self._validator = validator or SourceValidator()
        self._scorer    = scorer    or QualityScorer()
        self._min_quality = min_quality_threshold

        # Injected downstream components
        self._memory_engine:   Optional[Any] = None
        self._knowledge_store: Optional[Any] = None
        self._env_model:       Optional[Any] = None
        self._semantic_matcher: Optional[Any] = None

        # Source feeders: source_id → callable that returns List[KnowledgeItem]
        self._feeders: Dict[str, Callable[[], List[KnowledgeItem]]] = {}

        # Sync history: source_id → last SyncResult
        self._sync_history: Dict[str, SyncResult] = {}

        # Scheduler
        self._scheduler_running = False
        self._scheduler_thread:  Optional[threading.Thread] = None
        self._sync_lock = threading.Lock()

        logger.info("[SourceManager] initialised")

    # ── Dependency Injection ───────────────────────────────────────────────

    def set_memory_engine(self, engine) -> None:
        self._memory_engine = engine
        logger.info("[SourceManager] MemoryEngine connected")

    def set_knowledge_store(self, store) -> None:
        self._knowledge_store = store
        logger.info("[SourceManager] KnowledgeStore connected")

    def set_environment_model(self, model) -> None:
        self._env_model = model
        logger.info("[SourceManager] EnvironmentModel connected")

    def set_semantic_matcher(self, matcher) -> None:
        self._semantic_matcher = matcher
        logger.info("[SourceManager] SemanticMatcher connected")

    # ── Source Registration ────────────────────────────────────────────────

    def register_source(
        self,
        meta: SourceMetadata,
        feeder: Optional[Callable[[], List[KnowledgeItem]]] = None,
    ) -> SourceMetadata:
        """Register a source with optional feeder callable."""
        result = self._registry.register(meta)
        if feeder:
            self._feeders[meta.id] = feeder
            logger.info(f"[SourceManager] feeder attached for: {meta.name}")
        return result

    def attach_feeder(
        self,
        source_id: str,
        feeder: Callable[[], List[KnowledgeItem]],
    ) -> bool:
        if not self._registry.has(source_id):
            return False
        self._feeders[source_id] = feeder
        return True

    def unregister_source(self, source_id: str) -> bool:
        self._feeders.pop(source_id, None)
        return self._registry.unregister(source_id)

    # ── Sync Operations ────────────────────────────────────────────────────

    def sync_source(self, source_id: str) -> SyncResult:
        """
        Synchronous sync of a single source.
        Runs the full pipeline: Feeder → Validate → Score → Ingest.
        """
        meta = self._registry.get(source_id)
        if not meta:
            raise ValueError(f"Source not found: {source_id}")

        result = SyncResult(source_id, meta.name)
        self._registry.mark_sync_start(source_id)

        try:
            # 1. Fetch items from feeder
            feeder = self._feeders.get(source_id)
            if not feeder:
                raise RuntimeError(f"No feeder attached for source: {meta.name}")

            items = feeder()
            result.items_fetched = len(items)
            logger.info(
                f"[SourceManager] {meta.name}: fetched {len(items)} items"
            )

            # 2. Validate
            accepted, val_results = self._validator.validate_batch(items, source=meta)
            result.items_validated = len(accepted)
            result.items_rejected += len(items) - len(accepted)

            # 3. Score
            scores     = self._scorer.score_batch(accepted, source=meta)
            final_items = []
            for item, qs in zip(accepted, scores):
                item.quality_score = qs.total
                item.trust_score   = meta.trust_score
                item.is_validated  = True
                if qs.total >= self._min_quality:
                    final_items.append(item)
                else:
                    result.items_rejected += 1

            result.items_scored   = len(final_items)
            result.avg_quality    = (
                sum(i.quality_score for i in final_items) / max(len(final_items), 1)
            )

            # 4. Ingest into downstream systems
            self._ingest_items(final_items, meta)
            result.items_ingested = len(final_items)

            self._registry.mark_sync_done(
                source_id,
                items_ingested = result.items_ingested,
                items_rejected = result.items_rejected,
                avg_quality    = result.avg_quality,
            )

        except Exception as exc:
            err_msg = str(exc)
            result.errors.append(err_msg)
            self._registry.mark_error(source_id, err_msg)
            logger.error(f"[SourceManager] sync failed for {meta.name}: {exc}")

        result.finish()
        self._sync_history[source_id] = result
        logger.info(
            f"[SourceManager] sync done: {meta.name} — "
            f"{result.items_ingested} ingested, {result.items_rejected} rejected, "
            f"avg_quality={result.avg_quality:.1f}"
        )
        return result

    def sync_source_async(self, source_id: str) -> threading.Thread:
        """Run sync in a background thread."""
        t = threading.Thread(
            target=self.sync_source,
            args=(source_id,),
            daemon=True,
            name=f"ks-sync-{source_id[:8]}",
        )
        t.start()
        return t

    def sync_all(self, async_mode: bool = False) -> List[SyncResult]:
        """Sync all active sources with feeders."""
        results = []
        sources = [
            s for s in self._registry.list_active()
            if s.id in self._feeders
        ]
        logger.info(f"[SourceManager] syncing {len(sources)} sources")
        for s in sources:
            if async_mode:
                self.sync_source_async(s.id)
            else:
                results.append(self.sync_source(s.id))
        return results

    # ── Ingestion Pipeline ─────────────────────────────────────────────────

    def _ingest_items(
        self, items: List[KnowledgeItem], meta: SourceMetadata
    ) -> None:
        """Push validated + scored items to all downstream systems."""
        for item in items:
            # KnowledgeStore
            if self._knowledge_store:
                try:
                    self._ingest_to_knowledge_store(item, meta)
                except Exception as exc:
                    logger.warning(f"[SourceManager] KS ingest error: {exc}")

            # EnvironmentModel — update world model with new knowledge
            if self._env_model:
                try:
                    self._ingest_to_env_model(item, meta)
                except Exception as exc:
                    logger.warning(f"[SourceManager] EnvModel update error: {exc}")

    def _ingest_to_knowledge_store(
        self, item: KnowledgeItem, meta: SourceMetadata
    ) -> None:
        """Persist a knowledge item to the KnowledgeStore."""
        # Store in node profiles as a knowledge node
        profile = {
            "node_id":     f"ks:{item.item_id}",
            "name":        item.raw_reference or item.item_id[:16],
            "description": item.derived_summary or item.raw_content[:200],
            "capability":  f"knowledge:{meta.source_type.value}",
            "tags":        item.derived_tags + [meta.source_type.value, meta.name],
            "version":     "1.0",
            "announced_at": item.ingested_at,
            "is_active":   True,
            "source_id":   meta.id,
            "source_name": meta.name,
            "quality_score": item.quality_score,
            "trust_score":   item.trust_score,
            "raw_reference": item.raw_reference,
            # Raw content is stored only for non-scripture or is read-protected
            "raw_content":   item.raw_content if meta.store_raw_data else "[protected]",
            "derived_concepts": item.derived_concepts,
        }
        if hasattr(self._knowledge_store, "register_node"):
            self._knowledge_store.register_node(profile)

    def _ingest_to_env_model(
        self, item: KnowledgeItem, meta: SourceMetadata
    ) -> None:
        """Update the world model with knowledge from this source."""
        event = {
            "type":        "knowledge_ingested",
            "source_id":   meta.id,
            "source_name": meta.name,
            "source_type": meta.source_type.value,
            "reference":   item.raw_reference,
            "quality":     item.quality_score,
            "tags":        item.derived_tags,
            "timestamp":   datetime.now(timezone.utc).isoformat(),
        }
        if hasattr(self._env_model, "ingest_sensor_event"):
            self._env_model.ingest_sensor_event(event)

    # ── Scheduler ─────────────────────────────────────────────────────────

    def start_scheduler(self, interval_seconds: int = 3600) -> None:
        """Start a background thread that syncs sources on their schedule."""
        if self._scheduler_running:
            return
        self._scheduler_running = True
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            args=(interval_seconds,),
            daemon=True,
            name="ks-scheduler",
        )
        self._scheduler_thread.start()
        logger.info(f"[SourceManager] scheduler started (interval={interval_seconds}s)")

    def stop_scheduler(self) -> None:
        self._scheduler_running = False
        logger.info("[SourceManager] scheduler stopped")

    def _scheduler_loop(self, interval: int) -> None:
        while self._scheduler_running:
            now = datetime.now(timezone.utc)
            for source in self._registry.list_active():
                if source.update_frequency == UpdateFrequency.STATIC:
                    continue
                if source.id not in self._feeders:
                    continue
                # Check if sync is due
                if self._is_sync_due(source, now):
                    logger.info(f"[SourceManager] scheduler triggering sync: {source.name}")
                    self.sync_source_async(source.id)
            time.sleep(interval)

    def _is_sync_due(self, source: SourceMetadata, now: datetime) -> bool:
        if not source.last_sync:
            return True
        try:
            last = datetime.fromisoformat(source.last_sync)
            freq = source.update_frequency
            delta_map = {
                UpdateFrequency.HOURLY:  3600,
                UpdateFrequency.DAILY:   86400,
                UpdateFrequency.WEEKLY:  604800,
                UpdateFrequency.MONTHLY: 2592000,
            }
            delta_secs = delta_map.get(freq, float("inf"))
            elapsed    = (now - last).total_seconds()
            return elapsed >= delta_secs
        except Exception:
            return True

    # ── Control API ────────────────────────────────────────────────────────

    def pause_source(self, source_id: str) -> bool:
        return self._registry.update_status(source_id, SourceStatus.PAUSED)

    def resume_source(self, source_id: str) -> bool:
        return self._registry.update_status(source_id, SourceStatus.ACTIVE)

    def disable_source(self, source_id: str) -> bool:
        return self._registry.update_status(source_id, SourceStatus.DISABLED)

    # ── Status & Reporting ─────────────────────────────────────────────────

    def list_sources(self) -> List[Dict[str, Any]]:
        return [s.to_dict() for s in self._registry.list_all()]

    def source_status(self, source_id: str) -> Optional[Dict[str, Any]]:
        meta = self._registry.get(source_id)
        if not meta:
            return None
        d = meta.to_dict()
        last = self._sync_history.get(source_id)
        if last:
            d["last_sync_result"] = last.to_dict()
        return d

    def all_sync_history(self) -> Dict[str, Any]:
        return {sid: r.to_dict() for sid, r in self._sync_history.items()}

    def summary(self) -> Dict[str, Any]:
        return {
            "registry":  self._registry.summary(),
            "validator": self._validator.summary(),
            "scorer":    self._scorer.summary(),
            "scheduler": {
                "running":   self._scheduler_running,
                "feeders":   len(self._feeders),
            },
            "min_quality_threshold": self._min_quality,
        }
