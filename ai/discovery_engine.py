"""
Phase 3 – Discovery Engine
Each node announces itself with a semantic profile.
The engine maintains a live registry and detects new connection opportunities.

Knowledge Layer Integration (Phase 3 completion):
  Writes node profiles and semantic metadata to:
    knowledge/node_profiles.json
  via KnowledgeStore (injected via set_knowledge_store()).
"""
from __future__ import annotations
import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from knowledge.knowledge_store import KnowledgeStore

logger = logging.getLogger(__name__)


class NodeAnnouncement:
    """
    A node's self-announcement containing full semantic metadata.
    This is what the Self-Discovery Layer uses to understand each node's capabilities.
    """

    def __init__(self, node_id: str, name: str, description: str,
                 input_schema: dict, output_schema: dict,
                 tags: List[str] = None, capability: str = "",
                 version: str = "1.0.0"):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.input_schema = input_schema    # {fields: {}, required: []}
        self.output_schema = output_schema
        self.tags = tags or []
        self.capability = capability or description
        self.version = version
        self.announced_at = datetime.utcnow().isoformat()
        self.is_active: bool = True

    @classmethod
    def from_node(cls, node) -> "NodeAnnouncement":
        return cls(
            node_id=node.node_id,
            name=node.name,
            description=node.description,
            input_schema={
                "fields": node.input_schema.fields,
                "required": node.input_schema.required,
                "description": node.input_schema.description,
            },
            output_schema={
                "fields": node.output_schema.fields,
                "required": node.output_schema.required,
                "description": node.output_schema.description,
            },
            tags=node.tags,
            capability=node.description,
            version=getattr(node.metadata, "version", "1.0.0"),
        )

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "tags": self.tags,
            "capability": self.capability,
            "version": self.version,
            "announced_at": self.announced_at,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NodeAnnouncement":
        ann = cls(
            node_id=d["node_id"],
            name=d["name"],
            description=d.get("description", ""),
            input_schema=d.get("input_schema", {}),
            output_schema=d.get("output_schema", {}),
            tags=d.get("tags", []),
            capability=d.get("capability", ""),
            version=d.get("version", "1.0.0"),
        )
        ann.announced_at = d.get("announced_at", ann.announced_at)
        ann.is_active = d.get("is_active", True)
        return ann


class DiscoveryEngine:
    """
    Phase 3 Discovery Engine.

    Responsibilities:
    1. Accept node announcements (self-registration with full metadata).
    2. Persist announcements to SQLite.
    3. Detect new potential connections by comparing input/output schemas.
    4. Feed discovered profiles to SemanticMatcher.
    5. Track node activation/deactivation.
    """

    def __init__(self, db_path: str = "./data/mesh.db",
                 semantic_matcher=None):
        self._db_path = Path(db_path)
        self._announcements: Dict[str, NodeAnnouncement] = {}
        self._semantic_matcher = semantic_matcher
        self._knowledge = None   # KnowledgeStore — injected via set_knowledge_store()
        self._init_schema()
        self._load()
        logger.info("DiscoveryEngine initialised (Phase 3)")

    def set_knowledge_store(self, ks) -> None:
        """Inject the KnowledgeStore so DiscoveryEngine can persist node profiles."""
        self._knowledge = ks
        logger.info("DiscoveryEngine: KnowledgeStore connected")
        # Re-sync all known announcements to knowledge layer
        for ann in self._announcements.values():
            self._write_profile_to_knowledge(ann)

    # ── Schema ─────────────────────────────────────────────────────────────

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self):
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_announcements (
                    node_id         TEXT PRIMARY KEY,
                    announcement_json TEXT NOT NULL,
                    is_active       INTEGER DEFAULT 1,
                    announced_at    TEXT NOT NULL
                )
            """)

    def _load(self):
        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT announcement_json, is_active FROM node_announcements"
                ).fetchall()
            for row in rows:
                ann = NodeAnnouncement.from_dict(json.loads(row["announcement_json"]))
                ann.is_active = bool(row["is_active"])
                self._announcements[ann.node_id] = ann
                if self._semantic_matcher and ann.is_active:
                    self._register_with_matcher(ann)
            logger.info(f"DiscoveryEngine loaded {len(self._announcements)} announcements")
        except Exception as e:
            logger.warning(f"DiscoveryEngine load warning: {e}")

    # ── Announcement API ───────────────────────────────────────────────────

    def announce(self, node) -> NodeAnnouncement:
        """Called when a node registers in the mesh."""
        ann = NodeAnnouncement.from_node(node)
        return self._process_announcement(ann)

    def announce_dict(self, node_dict: dict) -> NodeAnnouncement:
        """Announce from a dict (e.g., from API)."""
        ann = NodeAnnouncement.from_dict(node_dict)
        return self._process_announcement(ann)

    def _process_announcement(self, ann: NodeAnnouncement) -> NodeAnnouncement:
        is_new = ann.node_id not in self._announcements
        self._announcements[ann.node_id] = ann
        self._persist(ann)
        if self._semantic_matcher:
            self._register_with_matcher(ann)
        action = "NEW" if is_new else "UPDATED"
        logger.info(f"DiscoveryEngine [{action}] node '{ann.name}' [{ann.node_id[:8]}]")
        # Write to knowledge layer
        if self._knowledge:
            self._write_profile_to_knowledge(ann)
        return ann

    def deactivate(self, node_id: str):
        if node_id in self._announcements:
            self._announcements[node_id].is_active = False
            self._persist(self._announcements[node_id])
            if self._knowledge:
                self._knowledge.deactivate_node_profile(node_id)

    def _write_profile_to_knowledge(self, ann: NodeAnnouncement) -> None:
        """Persist a node announcement as a knowledge profile."""
        if not self._knowledge:
            return
        try:
            profile = ann.to_dict()
            # Build basic semantic token lists from schema fields + tags
            import re

            def _tokens(text: str):
                return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if t]

            in_fields = list(ann.input_schema.get("fields", {}).keys())
            out_fields = list(ann.output_schema.get("fields", {}).keys())
            cap_words = _tokens(ann.capability or ann.description or "")
            tag_words = [t for tag in ann.tags for t in _tokens(tag)]

            profile["semantic_profile"] = {
                "input_tokens":      [t for f in in_fields for t in _tokens(f)],
                "output_tokens":     [t for f in out_fields for t in _tokens(f)],
                "capability_tokens": list(set(cap_words + tag_words)),
                "updated_at":        ann.announced_at,
            }
            self._knowledge.upsert_node_profile(ann.node_id, profile)
        except Exception as e:
            logger.warning(f"DiscoveryEngine: knowledge profile write failed for [{ann.node_id[:8]}]: {e}")

    def _register_with_matcher(self, ann: NodeAnnouncement):
        """Register semantic profile with the SemanticMatcher."""
        profile_dict = {
            "node_id": ann.node_id,
            "name": ann.name,
            "description": ann.description,
            "input_fields": ann.input_schema.get("fields", {}),
            "output_fields": ann.output_schema.get("fields", {}),
            "tags": ann.tags,
            "capability": ann.capability,
        }
        self._semantic_matcher.register_from_dict(profile_dict)

    def _persist(self, ann: NodeAnnouncement):
        try:
            with self._conn() as conn:
                conn.execute("""
                    INSERT INTO node_announcements (node_id, announcement_json, is_active, announced_at)
                    VALUES (?,?,?,?)
                    ON CONFLICT(node_id) DO UPDATE SET
                        announcement_json=excluded.announcement_json,
                        is_active=excluded.is_active
                """, (
                    ann.node_id,
                    json.dumps(ann.to_dict()),
                    int(ann.is_active),
                    ann.announced_at,
                ))
        except Exception as e:
            logger.error(f"DiscoveryEngine persist error: {e}")

    # ── Discovery queries ──────────────────────────────────────────────────

    def get_announcement(self, node_id: str) -> Optional[NodeAnnouncement]:
        return self._announcements.get(node_id)

    def active_nodes(self) -> List[NodeAnnouncement]:
        return [a for a in self._announcements.values() if a.is_active]

    def find_compatible_pairs(self, threshold: float = 0.15) -> List[dict]:
        """
        Use SemanticMatcher to find node pairs with high output→input compatibility.
        Returns suggestions for new connections.
        """
        if not self._semantic_matcher:
            return []
        existing_edges: List[Tuple[str, str]] = []  # No graph ref here; filter at call site
        return self._semantic_matcher.suggest_new_connections(existing_edges, threshold)

    def find_nodes_for_goal(self, goal: str, top_k: int = 5) -> List[dict]:
        """Find nodes best suited for a goal description."""
        if not self._semantic_matcher:
            return []
        results = self._semantic_matcher.find_nodes_for_goal(goal, top_k)
        output = []
        for node_id, score in results:
            ann = self._announcements.get(node_id)
            output.append({
                "node_id": node_id,
                "name": ann.name if ann else "unknown",
                "capability_score": score,
                "capability": ann.capability if ann else "",
            })
        return output

    def summary(self) -> dict:
        active = self.active_nodes()
        return {
            "total_announced": len(self._announcements),
            "active_nodes": len(active),
            "inactive_nodes": len(self._announcements) - len(active),
        }

    def set_semantic_matcher(self, matcher):
        self._semantic_matcher = matcher
        # Re-register all active nodes
        for ann in self.active_nodes():
            self._register_with_matcher(ann)

    def __repr__(self):
        return f"<DiscoveryEngine announced={len(self._announcements)} active={len(self.active_nodes())}>"
