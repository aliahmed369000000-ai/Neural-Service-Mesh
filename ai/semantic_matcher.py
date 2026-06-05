"""
Phase 3 – Semantic Matcher
Matches node outputs to node inputs using semantic similarity.
Uses keyword-based cosine similarity (no external ML libs required).
Foundation is kept clean for future embedding-based upgrade.
"""
from __future__ import annotations
import re
import math
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Semantic vocabulary ────────────────────────────────────────────────────
# Groups of semantically related terms so that "text" ~ "content" ~ "body"
_SYNONYM_GROUPS: List[List[str]] = [
    ["text", "content", "body", "message", "string", "str", "raw"],
    ["data", "payload", "input", "output", "result", "value"],
    ["analysis", "result", "output", "response", "score", "report"],
    ["sentiment", "emotion", "mood", "tone", "feeling"],
    ["summary", "abstract", "overview", "description", "synopsis"],
    ["number", "count", "int", "integer", "numeric", "quantity"],
    ["user", "customer", "client", "person", "profile"],
    ["error", "exception", "failure", "fault", "issue"],
    ["status", "state", "condition", "health", "flag"],
    ["timestamp", "date", "time", "created_at", "updated_at"],
    ["id", "identifier", "uuid", "key", "ref", "reference"],
    ["list", "array", "items", "collection", "set"],
    ["file", "path", "url", "uri", "link", "source"],
    ["json", "dict", "object", "map", "record", "document"],
    ["log", "event", "trace", "audit", "history"],
    ["model", "prediction", "inference", "classification", "label"],
    ["query", "search", "filter", "request", "prompt"],
    ["config", "settings", "parameters", "options", "args"],
    ["token", "auth", "credential", "secret", "key"],
    ["metric", "measurement", "stat", "kpi", "indicator"],
]

# Build reverse lookup: term -> canonical group index
_TERM_TO_GROUP: Dict[str, int] = {}
for _gi, _group in enumerate(_SYNONYM_GROUPS):
    for _term in _group:
        _TERM_TO_GROUP[_term.lower()] = _gi


def _tokenize(text: str) -> List[str]:
    """Split a schema field name / description into lowercase tokens."""
    text = text.lower()
    # split on non-alphanumeric
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if t]


def _semantic_tokens(tokens: List[str]) -> List[int]:
    """Map tokens to their synonym group index (or unique high IDs if unknown)."""
    groups = []
    unknown_id = len(_SYNONYM_GROUPS)
    for t in tokens:
        gid = _TERM_TO_GROUP.get(t, None)
        if gid is not None:
            groups.append(gid)
        else:
            # Treat unknown token as its own unique group
            groups.append(hash(t) % 10000 + unknown_id)
    return groups


def _bow_vector(tokens: List[str]) -> Dict[int, float]:
    """Build a term-frequency bag-of-words vector from semantic group ids."""
    vec: Dict[int, float] = {}
    gids = _semantic_tokens(tokens)
    for gid in gids:
        vec[gid] = vec.get(gid, 0.0) + 1.0
    # L2 normalise
    norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
    return {k: v / norm for k, v in vec.items()}


def _cosine(a: Dict[int, float], b: Dict[int, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    dot = sum(a[k] * b[k] for k in a if k in b)
    return round(dot, 4)


class NodeSemanticProfile:
    """
    Encapsulates all semantic information for a single node.
    Created once per node at registration / discovery time.
    """

    def __init__(self, node_id: str, name: str, description: str,
                 input_fields: Dict[str, str], output_fields: Dict[str, str],
                 tags: List[str] = None, capability: str = ""):
        self.node_id = node_id
        self.name = name
        self.description = description
        self.input_fields = input_fields    # field_name -> field_type
        self.output_fields = output_fields
        self.tags = tags or []
        self.capability = capability        # free-text capability description

        # Pre-compute vectors
        self._input_vec = self._build_vec(input_fields, description, tags)
        self._output_vec = self._build_vec(output_fields, description, tags)
        self._capability_vec = self._build_vec({}, capability, tags)

    @staticmethod
    def _build_vec(fields: Dict[str, str], text: str, tags: List[str]) -> Dict[int, float]:
        tokens: List[str] = []
        for fname, ftype in fields.items():
            tokens += _tokenize(fname)
            tokens += _tokenize(ftype)
        tokens += _tokenize(text)
        for tag in (tags or []):
            tokens += _tokenize(tag)
        return _bow_vector(tokens)

    def input_similarity(self, other: "NodeSemanticProfile") -> float:
        """How well does `other`'s output match this node's input?"""
        return _cosine(other._output_vec, self._input_vec)

    def capability_similarity(self, goal_text: str) -> float:
        """How well does this node's capability match a goal description?"""
        goal_tokens = _tokenize(goal_text)
        goal_vec = _bow_vector(goal_tokens)
        return _cosine(self._capability_vec, goal_vec)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "name": self.name,
            "description": self.description,
            "input_fields": self.input_fields,
            "output_fields": self.output_fields,
            "tags": self.tags,
            "capability": self.capability,
        }

    @classmethod
    def from_node(cls, node) -> "NodeSemanticProfile":
        """Build a profile directly from a BaseNode instance."""
        return cls(
            node_id=node.node_id,
            name=node.name,
            description=node.description,
            input_fields=node.input_schema.fields,
            output_fields=node.output_schema.fields,
            tags=node.tags,
            capability=node.description,
        )

    @classmethod
    def from_dict(cls, d: dict) -> "NodeSemanticProfile":
        return cls(
            node_id=d["node_id"],
            name=d["name"],
            description=d.get("description", ""),
            input_fields=d.get("input_fields", {}),
            output_fields=d.get("output_fields", {}),
            tags=d.get("tags", []),
            capability=d.get("capability", d.get("description", "")),
        )


class SemanticMatcher:
    """
    Phase 3 Semantic Matcher.
    Maintains a registry of NodeSemanticProfiles and answers
    questions like:
      - Which nodes can accept the output of node X?
      - Which nodes best match a goal description?
      - What is the semantic compatibility score between two nodes?
    """

    def __init__(self):
        self._profiles: Dict[str, NodeSemanticProfile] = {}
        logger.info("SemanticMatcher initialised (Phase 3)")

    # ── Profile management ─────────────────────────────────────────────────

    def register(self, node) -> NodeSemanticProfile:
        """Register (or re-register) a node's semantic profile."""
        profile = NodeSemanticProfile.from_node(node)
        self._profiles[node.node_id] = profile
        logger.debug(f"SemanticMatcher: registered profile for '{node.name}'")
        return profile

    def register_from_dict(self, node_dict: dict) -> NodeSemanticProfile:
        profile = NodeSemanticProfile.from_dict(node_dict)
        self._profiles[node_dict["node_id"]] = profile
        return profile

    def get_profile(self, node_id: str) -> Optional[NodeSemanticProfile]:
        return self._profiles.get(node_id)

    def all_profiles(self) -> List[NodeSemanticProfile]:
        return list(self._profiles.values())

    # ── Matching API ───────────────────────────────────────────────────────

    def find_compatible_consumers(self, producer_id: str,
                                  threshold: float = 0.1) -> List[Tuple[str, float]]:
        """
        Return nodes whose INPUT is semantically compatible with
        `producer_id`'s OUTPUT.
        Returns list of (node_id, score) sorted descending.
        """
        producer = self._profiles.get(producer_id)
        if not producer:
            return []

        results = []
        for nid, profile in self._profiles.items():
            if nid == producer_id:
                continue
            score = profile.input_similarity(producer)
            if score >= threshold:
                results.append((nid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def find_nodes_for_goal(self, goal: str,
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Return nodes best suited to achieve a described goal.
        Returns list of (node_id, score) sorted descending.
        """
        results = []
        for nid, profile in self._profiles.items():
            score = profile.capability_similarity(goal)
            results.append((nid, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def compatibility_score(self, producer_id: str, consumer_id: str) -> float:
        """Direct semantic compatibility from producer → consumer."""
        producer = self._profiles.get(producer_id)
        consumer = self._profiles.get(consumer_id)
        if not producer or not consumer:
            return 0.0
        return consumer.input_similarity(producer)

    def suggest_new_connections(self,
                                existing_edges: List[Tuple[str, str]],
                                threshold: float = 0.15) -> List[dict]:
        """
        Suggest new connections not currently in the graph
        that have high semantic compatibility.
        """
        existing = set((s, t) for s, t in existing_edges)
        suggestions = []

        node_ids = list(self._profiles.keys())
        for i, src_id in enumerate(node_ids):
            for tgt_id in node_ids[i + 1:]:
                if (src_id, tgt_id) in existing or (tgt_id, src_id) in existing:
                    continue
                # Check both directions
                score_fwd = self.compatibility_score(src_id, tgt_id)
                score_rev = self.compatibility_score(tgt_id, src_id)
                best_score = max(score_fwd, score_rev)
                if best_score >= threshold:
                    direction = (src_id, tgt_id) if score_fwd >= score_rev else (tgt_id, src_id)
                    suggestions.append({
                        "source_id": direction[0],
                        "target_id": direction[1],
                        "semantic_score": best_score,
                        "reason": "semantic_compatibility",
                    })

        suggestions.sort(key=lambda x: x["semantic_score"], reverse=True)
        return suggestions

    def profile_count(self) -> int:
        return len(self._profiles)

    def __repr__(self):
        return f"<SemanticMatcher profiles={self.profile_count()}>"
