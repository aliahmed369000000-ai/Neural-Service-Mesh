
from __future__ import annotations
import logging
from collections import deque
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class GraphEdge:
    def __init__(self, source_id: str, target_id: str, weight: float = 1.0, label: str = ""):
        self.source_id = source_id
        self.target_id = target_id
        self.weight = weight
        self.label = label

    def to_dict(self):
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "weight": self.weight,
            "label": self.label,
        }


class ServiceGraph:
    def __init__(self):
        self._adjacency: Dict[str, List[GraphEdge]] = {}
        self._reverse: Dict[str, Set[str]] = {}
        self._node_meta: Dict[str, dict] = {}

    def add_node(self, node_id: str, metadata: dict):
        if node_id in self._adjacency:
            return
        self._adjacency[node_id] = []
        self._reverse[node_id] = set()
        self._node_meta[node_id] = metadata
        logger.info(f"Graph: node added '{metadata.get('name', node_id[:8])}'")

    def remove_node(self, node_id: str) -> bool:
        if node_id not in self._adjacency:
            return False
        del self._adjacency[node_id]
        del self._node_meta[node_id]
        del self._reverse[node_id]
        for sid in self._adjacency:
            self._adjacency[sid] = [e for e in self._adjacency[sid] if e.target_id != node_id]
            self._reverse[sid].discard(node_id)
        return True

    def add_edge(self, source_id: str, target_id: str, weight: float = 1.0, label: str = "") -> GraphEdge:
        self._check(source_id)
        self._check(target_id)
        for e in self._adjacency[source_id]:
            if e.target_id == target_id:
                return e
        edge = GraphEdge(source_id, target_id, weight, label)
        self._adjacency[source_id].append(edge)
        self._reverse[target_id].add(source_id)
        logger.info(f"Graph: edge {source_id[:8]} -> {target_id[:8]}")
        return edge

    def remove_edge(self, source_id: str, target_id: str) -> bool:
        if source_id not in self._adjacency:
            return False
        before = len(self._adjacency[source_id])
        self._adjacency[source_id] = [e for e in self._adjacency[source_id] if e.target_id != target_id]
        if len(self._adjacency[source_id]) < before:
            self._reverse[target_id].discard(source_id)
            return True
        return False

    def get_neighbors(self, node_id: str) -> List[str]:
        self._check(node_id)
        return [e.target_id for e in self._adjacency[node_id]]

    def find_path_bfs(self, start: str, end: str) -> Optional[List[str]]:
        self._check(start)
        self._check(end)
        if start == end:
            return [start]
        visited = {start}
        queue = deque([[start]])
        while queue:
            path = queue.popleft()
            for nb in self.get_neighbors(path[-1]):
                if nb == end:
                    return path + [end]
                if nb not in visited:
                    visited.add(nb)
                    queue.append(path + [nb])
        return None

    def topological_sort(self) -> Optional[List[str]]:
        in_deg = {n: len(self._reverse[n]) for n in self._adjacency}
        q = deque([n for n, d in in_deg.items() if d == 0])
        result = []
        while q:
            n = q.popleft()
            result.append(n)
            for e in self._adjacency[n]:
                in_deg[e.target_id] -= 1
                if in_deg[e.target_id] == 0:
                    q.append(e.target_id)
        return result if len(result) == len(self._adjacency) else None

    def has_node(self, node_id: str) -> bool:
        return node_id in self._adjacency

    def list_nodes(self) -> List[dict]:
        return list(self._node_meta.values())

    def stats(self) -> dict:
        edges = sum(len(v) for v in self._adjacency.values())
        return {
            "node_count": len(self._adjacency),
            "edge_count": edges,
            "has_cycles": self.topological_sort() is None,
        }

    def to_dict(self) -> dict:
        return {
            "nodes": self.list_nodes(),
            "edges": [e.to_dict() for edges in self._adjacency.values() for e in edges],
        }

    def from_dict(self, data: dict):
        for nm in data.get("nodes", []):
            nid = nm["node_id"]
            self._adjacency[nid] = []
            self._reverse[nid] = set()
            self._node_meta[nid] = nm
        for ed in data.get("edges", []):
            self.add_edge(ed["source_id"], ed["target_id"], ed.get("weight", 1.0), ed.get("label", ""))

    def _check(self, node_id: str):
        if node_id not in self._adjacency:
            raise KeyError(f"Node '{node_id}' not in graph")

    def __repr__(self):
        s = self.stats()
        return f"<ServiceGraph nodes={s['node_count']} edges={s['edge_count']}>"
