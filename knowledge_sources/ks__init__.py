"""
Knowledge Sources Layer
========================
External knowledge ingestion pipeline for the Neural Service Mesh.

Architecture:
    Knowledge Source
        ↓
    Feeder (source-specific)
        ↓
    SourceValidator (completeness / dedup / corruption / trust)
        ↓
    QualityScorer (0–100 multi-factor score)
        ↓
    MemoryEngine
        ↓
    KnowledgeStore
        ↓
    World Model

Public API:
    from knowledge_sources import SourceManager, SourceRegistry
    from knowledge_sources import SourceMetadata, SourceType, UpdateFrequency
    from knowledge_sources.quran.quran_source import create_quran_source
"""

from knowledge_sources.source_metadata import (
    SourceMetadata, KnowledgeItem,
    SourceType, SourceStatus, UpdateFrequency, AccessMode,
)
from knowledge_sources.source_registry  import SourceRegistry
from knowledge_sources.source_validator import SourceValidator, ValidationResult
from knowledge_sources.quality_scorer   import QualityScorer, QualityScore
from knowledge_sources.source_manager   import SourceManager, SyncResult
from knowledge_sources.base_source      import (
    BaseKnowledgeSource,
    WikipediaSource, GitHubSource, RSSSource, PublicAPISource,
)

__all__ = [
    # Metadata
    "SourceMetadata", "KnowledgeItem",
    "SourceType", "SourceStatus", "UpdateFrequency", "AccessMode",
    # Core Components
    "SourceRegistry",
    "SourceValidator", "ValidationResult",
    "QualityScorer", "QualityScore",
    "SourceManager", "SyncResult",
    # Base & Stubs
    "BaseKnowledgeSource",
    "WikipediaSource", "GitHubSource", "RSSSource", "PublicAPISource",
]
