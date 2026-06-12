"""
Knowledge Sources — Source Metadata
=====================================
Dataclasses and enums that describe every knowledge source registered
in the system. Immutable at the source-data level; derived knowledge
(concepts, relations) is always stored separately.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Enums ──────────────────────────────────────────────────────────────────

class SourceType(str, Enum):
    SCRIPTURE     = "scripture"       # Holy texts — read-only, highest integrity
    ENCYCLOPEDIA  = "encyclopedia"    # Wikipedia, encyclopedic content
    CODE          = "code"            # GitHub, code repositories
    FEED          = "feed"            # RSS / Atom feeds
    API           = "api"             # Public REST / GraphQL APIs
    PAPER         = "paper"           # Scientific papers / journals
    BOOK          = "book"            # Books / long-form documents
    DOCUMENT      = "document"        # Generic documents / reports
    CUSTOM        = "custom"          # User-defined sources


class SourceStatus(str, Enum):
    ACTIVE      = "active"
    PAUSED      = "paused"
    ERROR       = "error"
    SYNCING     = "syncing"
    INITIALIZING = "initializing"
    DISABLED    = "disabled"


class UpdateFrequency(str, Enum):
    STATIC      = "static"       # Never changes (scriptures, classic books)
    DAILY       = "daily"
    HOURLY      = "hourly"
    REALTIME    = "realtime"
    WEEKLY      = "weekly"
    MONTHLY     = "monthly"
    ON_DEMAND   = "on_demand"


class AccessMode(str, Enum):
    READ_ONLY   = "read_only"    # Source data cannot be modified (e.g. Quran)
    READ_WRITE  = "read_write"   # System may annotate/cache
    DERIVED_ONLY = "derived_only" # Only derived knowledge is stored, raw untouched


# ── Metadata Dataclass ─────────────────────────────────────────────────────

@dataclass
class SourceMetadata:
    """
    Full descriptor of a knowledge source.

    Fields that are set once (at registration) are marked FIXED.
    Fields that the system updates during sync are marked MUTABLE.
    """

    # ── Identity (FIXED) ───────────────────────────────────────────────────
    id:               str = field(default_factory=lambda: str(uuid.uuid4()))
    name:             str = ""
    description:      str = ""
    source_type:      SourceType = SourceType.CUSTOM
    access_mode:      AccessMode = AccessMode.READ_ONLY

    # ── Trust & Quality (FIXED baseline, MUTABLE via scoring) ─────────────
    trust_score:      float = 0.5          # 0.0 – 1.0
    base_trust:       float = 0.5          # original trust (never overwritten)
    requires_citation: bool = True

    # ── Sync & Scheduling (MUTABLE) ───────────────────────────────────────
    update_frequency: UpdateFrequency = UpdateFrequency.ON_DEMAND
    last_sync:        Optional[str]   = None
    next_sync:        Optional[str]   = None
    sync_count:       int             = 0
    error_count:      int             = 0
    last_error:       Optional[str]   = None

    # ── Status (MUTABLE) ──────────────────────────────────────────────────
    status:           SourceStatus    = SourceStatus.INITIALIZING
    registered_at:    str             = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Statistics (MUTABLE) ──────────────────────────────────────────────
    total_items_ingested: int  = 0
    total_items_rejected: int  = 0
    avg_quality_score:    float = 0.0

    # ── Configuration (FIXED, source-specific) ────────────────────────────
    config:           Dict[str, Any] = field(default_factory=dict)
    tags:             List[str]      = field(default_factory=list)
    language:         str            = "ar"   # default Arabic (Quran)
    version:          str            = "1.0.0"

    # ── Data Separation Flags ─────────────────────────────────────────────
    allow_raw_modification:   bool = False   # NEVER True for scriptures
    store_raw_data:           bool = True
    store_derived_knowledge:  bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_type"]      = self.source_type.value
        d["access_mode"]      = self.access_mode.value
        d["update_frequency"] = self.update_frequency.value
        d["status"]           = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SourceMetadata":
        d = dict(d)
        d["source_type"]      = SourceType(d.get("source_type", "custom"))
        d["access_mode"]      = AccessMode(d.get("access_mode", "read_only"))
        d["update_frequency"] = UpdateFrequency(d.get("update_frequency", "on_demand"))
        d["status"]           = SourceStatus(d.get("status", "initializing"))
        return cls(**d)

    def mark_sync_start(self) -> None:
        self.status    = SourceStatus.SYNCING
        self.last_sync = datetime.now(timezone.utc).isoformat()

    def mark_sync_done(self, items_ingested: int = 0, items_rejected: int = 0) -> None:
        self.status              = SourceStatus.ACTIVE
        self.sync_count         += 1
        self.total_items_ingested += items_ingested
        self.total_items_rejected += items_rejected

    def mark_error(self, error_msg: str) -> None:
        self.status      = SourceStatus.ERROR
        self.error_count += 1
        self.last_error  = error_msg

    def is_immutable_source(self) -> bool:
        """True if the source raw data must never be modified."""
        return self.access_mode == AccessMode.READ_ONLY or not self.allow_raw_modification


# ── Knowledge Item ─────────────────────────────────────────────────────────

@dataclass
class KnowledgeItem:
    """
    A single unit of knowledge produced by a source feeder.

    Raw content is always kept separate from derived knowledge.
    The system may enrich 'derived' but must never touch 'raw_content'
    for immutable sources.
    """
    item_id:          str = field(default_factory=lambda: str(uuid.uuid4()))
    source_id:        str = ""
    source_type:      SourceType = SourceType.CUSTOM

    # ── Raw (protected) ───────────────────────────────────────────────────
    raw_content:      str = ""             # Original text / data — READ ONLY
    raw_reference:    str = ""             # e.g. "Al-Baqarah:255" or URL
    raw_language:     str = "ar"

    # ── Derived (system may write) ─────────────────────────────────────────
    derived_summary:      str = ""
    derived_concepts:     List[str] = field(default_factory=list)
    derived_relations:    List[Dict[str, str]] = field(default_factory=list)
    derived_embeddings:   Optional[List[float]] = None
    derived_tags:         List[str] = field(default_factory=list)

    # ── Quality ───────────────────────────────────────────────────────────
    quality_score:    float = 0.0          # 0–100
    trust_score:      float = 0.0          # inherited from source
    is_validated:     bool  = False

    # ── Timestamps ────────────────────────────────────────────────────────
    ingested_at:      str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_accessed:    Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["source_type"] = self.source_type.value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "KnowledgeItem":
        d = dict(d)
        d["source_type"] = SourceType(d.get("source_type", "custom"))
        return cls(**d)
