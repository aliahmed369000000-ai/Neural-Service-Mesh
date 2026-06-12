"""
Knowledge Sources — Base Source
==================================
Abstract base class that every knowledge source must extend.
Provides the standard interface:

    Source Feeder → Sensor → Quality Evaluation → Memory Engine → Knowledge Store → World Model

Future sources (Wikipedia, GitHub, RSS, APIs, Papers, Books) all
extend BaseKnowledgeSource and only need to implement:
  1. build_metadata() → SourceMetadata
  2. fetch_items()    → List[KnowledgeItem]

The pipeline logic (validation, scoring, ingestion) lives in
SourceManager and is identical for all sources.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple

from knowledge_sources.source_metadata import (
    KnowledgeItem, SourceMetadata, SourceType, UpdateFrequency,
    AccessMode
)

logger = logging.getLogger(__name__)


class BaseKnowledgeSource(ABC):
    """
    Abstract base for every knowledge source in the system.

    Subclass this for every new source type:
        - WikipediaSource
        - GitHubSource
        - RSSSource
        - PublicAPISource
        - ScientificPaperSource
        - BookSource
        - DocumentSource
    """

    def __init__(self):
        self._metadata: Optional[SourceMetadata] = None

    @property
    def metadata(self) -> SourceMetadata:
        if self._metadata is None:
            self._metadata = self.build_metadata()
        return self._metadata

    # ── Abstract Interface (subclasses must implement) ─────────────────────

    @abstractmethod
    def build_metadata(self) -> SourceMetadata:
        """
        Return the SourceMetadata descriptor for this source.
        Called once at initialization.

        Example:
            return SourceMetadata(
                id           = "wikipedia-en-v1",
                name         = "Wikipedia English",
                source_type  = SourceType.ENCYCLOPEDIA,
                trust_score  = 0.75,
                update_frequency = UpdateFrequency.DAILY,
                ...
            )
        """
        ...

    @abstractmethod
    def fetch_items(self) -> List[KnowledgeItem]:
        """
        Fetch and return a list of KnowledgeItems from this source.

        This is the FEEDER step in the pipeline:
            Source Data → KnowledgeItems (raw_content protected)

        Rules:
          - raw_content   must contain the ORIGINAL unmodified text
          - raw_reference must be a stable, citable reference
          - derived_*     fields may be pre-populated if the source
                          provides metadata (tags, summaries, etc.)
          - trust_score   should match the source's trust_score

        Returns [] on failure (do not raise — let the manager handle it).
        """
        ...

    # ── Optional Hooks (subclasses may override) ───────────────────────────

    def on_before_sync(self) -> None:
        """Called by SourceManager before fetching items. Override for auth, warmup, etc."""
        pass

    def on_after_sync(self, items_ingested: int, items_rejected: int) -> None:
        """Called by SourceManager after sync completes."""
        pass

    def health_check(self) -> Dict[str, Any]:
        """
        Return a health status dict. Override to add source-specific checks.
        Default: check if fetch_items returns at least 1 item.
        """
        try:
            items = self.fetch_items()
            return {
                "status":  "ok" if items else "empty",
                "items":   len(items),
                "source":  self.metadata.name,
            }
        except Exception as exc:
            return {
                "status":  "error",
                "error":   str(exc),
                "source":  self.metadata.name,
            }

    # ── Convenience ────────────────────────────────────────────────────────

    def as_feeder(self) -> Callable[[], List[KnowledgeItem]]:
        """Return a callable suitable for SourceManager.register_source()."""
        return self.fetch_items

    def register_with(self, source_manager) -> SourceMetadata:
        """
        Register this source with a SourceManager instance.

        Usage:
            src = WikipediaSource()
            src.register_with(source_manager)
            source_manager.sync_source(src.metadata.id)
        """
        return source_manager.register_source(
            self.metadata,
            feeder=self.fetch_items,
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.metadata.id} name={self.metadata.name}>"


# ── Stub Sources (placeholders — to be implemented) ───────────────────────

class WikipediaSource(BaseKnowledgeSource):
    """
    Placeholder for Wikipedia knowledge source.

    Implementation plan:
      - Use https://en.wikipedia.org/w/api.php to fetch articles
      - Extract: title, summary, categories, links
      - Map each article section → one KnowledgeItem
      - trust_score = 0.75 (community-edited, generally reliable)
      - update_frequency = WEEKLY (articles change slowly)
    """

    def build_metadata(self) -> SourceMetadata:
        from knowledge_sources.source_metadata import SourceStatus
        return SourceMetadata(
            id               = "wikipedia-en-v1",
            name             = "Wikipedia (English)",
            description      = "Wikipedia encyclopedia — community-edited knowledge",
            source_type      = SourceType.ENCYCLOPEDIA,
            access_mode      = AccessMode.READ_ONLY,
            trust_score      = 0.75,
            base_trust       = 0.75,
            update_frequency = UpdateFrequency.WEEKLY,
            language         = "en",
            tags             = ["encyclopedia", "wikipedia", "general-knowledge"],
            config           = {
                "api_url":   "https://en.wikipedia.org/w/api.php",
                "topics":    [],   # configure topics to fetch
                "max_articles": 100,
            },
            allow_raw_modification = False,
        )

    def fetch_items(self) -> List[KnowledgeItem]:
        logger.warning("[WikipediaSource] Not yet implemented — returning empty list")
        return []


class GitHubSource(BaseKnowledgeSource):
    """
    Placeholder for GitHub knowledge source.

    Implementation plan:
      - Use GitHub REST API to fetch READMEs, docs, code snippets
      - Extract: repo description, topics, README content
      - trust_score = 0.65 (code quality varies)
      - update_frequency = DAILY
    """

    def build_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            id               = "github-v1",
            name             = "GitHub Repositories",
            description      = "Open-source code and documentation from GitHub",
            source_type      = SourceType.CODE,
            access_mode      = AccessMode.READ_ONLY,
            trust_score      = 0.65,
            base_trust       = 0.65,
            update_frequency = UpdateFrequency.DAILY,
            language         = "en",
            tags             = ["code", "github", "open-source"],
            config           = {
                "api_url":  "https://api.github.com",
                "topics":   [],
                "max_repos": 50,
            },
            allow_raw_modification = False,
        )

    def fetch_items(self) -> List[KnowledgeItem]:
        logger.warning("[GitHubSource] Not yet implemented — returning empty list")
        return []


class RSSSource(BaseKnowledgeSource):
    """
    Placeholder for RSS/Atom feed source.

    Implementation plan:
      - Parse RSS/Atom feeds using feedparser
      - Extract: title, summary, link, published date, categories
      - trust_score varies by feed (configure per-feed)
      - update_frequency = HOURLY
    """

    def __init__(self, feed_url: str = "", trust: float = 0.6):
        super().__init__()
        self._feed_url = feed_url
        self._trust    = trust

    def build_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            id               = f"rss-{hash(self._feed_url) & 0xFFFFFF:06x}",
            name             = f"RSS Feed: {self._feed_url[:50]}",
            description      = "RSS/Atom news feed",
            source_type      = SourceType.FEED,
            access_mode      = AccessMode.READ_ONLY,
            trust_score      = self._trust,
            base_trust       = self._trust,
            update_frequency = UpdateFrequency.HOURLY,
            language         = "en",
            tags             = ["rss", "feed", "news"],
            config           = {"feed_url": self._feed_url},
            allow_raw_modification = False,
        )

    def fetch_items(self) -> List[KnowledgeItem]:
        logger.warning("[RSSSource] Not yet implemented — returning empty list")
        return []


class PublicAPISource(BaseKnowledgeSource):
    """
    Placeholder for generic public REST API source.

    Implementation plan:
      - Configurable endpoint, auth, and response parser
      - trust_score configurable per API
      - update_frequency configurable per API
    """

    def __init__(self, api_name: str = "Custom API", api_url: str = ""):
        super().__init__()
        self._api_name = api_name
        self._api_url  = api_url

    def build_metadata(self) -> SourceMetadata:
        return SourceMetadata(
            id               = f"api-{self._api_name.lower().replace(' ', '-')}-v1",
            name             = self._api_name,
            description      = f"Public API source: {self._api_url}",
            source_type      = SourceType.API,
            access_mode      = AccessMode.READ_ONLY,
            trust_score      = 0.60,
            base_trust       = 0.60,
            update_frequency = UpdateFrequency.ON_DEMAND,
            language         = "en",
            tags             = ["api", "public"],
            config           = {"api_url": self._api_url},
            allow_raw_modification = False,
        )

    def fetch_items(self) -> List[KnowledgeItem]:
        logger.warning(f"[PublicAPISource:{self._api_name}] Not yet implemented — returning empty list")
        return []
