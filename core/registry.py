from __future__ import annotations
import logging
from typing import Dict, List, Optional
from datetime import datetime

from core.node import BaseNode
from storage.file_storage import FileStorage

logger = logging.getLogger(__name__)
REGISTRY_FILE = "nodes.json"


class NodeRegistry:
    def __init__(self, storage: FileStorage):
        self._storage = storage
        self._nodes: Dict[str, BaseNode] = {}
        self._meta_cache: Dict[str, dict] = {}
        self._load()
        logger.info("NodeRegistry initialized")

    def register(self, node: BaseNode, overwrite: bool = False) -> str:
        if node.node_id in self._nodes and not overwrite:
            raise ValueError(f"Node '{node.node_id}' already registered")
        self._nodes[node.node_id] = node
        self._meta_cache[node.node_id] = node.to_dict()
        self._save()
        logger.info(f"Registered: {node.name} [{node.node_id[:8]}]")
        return node.node_id

    def unregister(self, node_id: str) -> bool:
        if node_id not in self._nodes:
            return False
        del self._nodes[node_id]
        self._meta_cache.pop(node_id, None)
        self._save()
        return True

    def get(self, node_id: str) -> Optional[BaseNode]:
        return self._nodes.get(node_id)

    def get_by_name(self, name: str) -> Optional[BaseNode]:
        return next((n for n in self._nodes.values() if n.name == name), None)

    def get_by_tag(self, tag: str) -> List[BaseNode]:
        return [n for n in self._nodes.values() if tag in n.tags]

    def list_all(self) -> List[BaseNode]:
        return list(self._nodes.values())

    def list_metadata(self) -> List[dict]:
        return list(self._meta_cache.values())

    def count(self) -> int:
        return len(self._nodes)

    def exists(self, node_id: str) -> bool:
        return node_id in self._nodes

    def _save(self):
        self._storage.save(REGISTRY_FILE, {
            "saved_at": datetime.utcnow().isoformat(),
            "count": len(self._meta_cache),
            "nodes": list(self._meta_cache.values()),
        })

    def _load(self):
        data = self._storage.load(REGISTRY_FILE)
        if data:
            for nm in data.get("nodes", []):
                nid = nm.get("node_id")
                if nid:
                    self._meta_cache[nid] = nm
            logger.info(f"Registry loaded {len(self._meta_cache)} records")

    def __repr__(self):
        return f"<NodeRegistry count={self.count()}>"
