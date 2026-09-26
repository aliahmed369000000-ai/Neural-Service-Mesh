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

    def get_meta_by_name(self, name: str) -> Optional[dict]:
        """بحث في meta_cache المحفوظ (وليس العُقد الحيّة فقط) عن آخر سجل
        بنفس الاسم. يُستخدم عند إعادة بناء عقدة بعد إعادة تشغيل العملية
        (get_by_name يرجع None لأن _nodes يبدأ فارغاً كل تشغيل) حتى تقدر
        الطبقة الأعلى تسترجع نفس node_id والتاريخ بدل البدء من الصفر."""
        return next(
            (m for m in self._meta_cache.values() if m.get("name") == name),
            None,
        )

    def get_by_tag(self, tag: str) -> List[BaseNode]:
        return [n for n in self._nodes.values() if tag in n.tags]

    def get_by_state(self, state: str) -> List[BaseNode]:
        """إرجاع كل العُقد الحيّة التي حالتها الحالية تطابق state
        (مثل NodeState.ACTIVE / PAUSED / FAILED / CREATED)."""
        return [n for n in self._nodes.values() if n.state == state]

    def refresh_meta(self, node_id: str, persist: bool = True) -> Optional[dict]:
        """إعادة مزامنة meta_cache من حالة العقدة الحيّة (state، execution_count،
        last_executed...) بعد أي تنفيذ، لأن الكاش كان يُحفظ فقط لحظة register()
        ولا يتحدّث تلقائياً بعد ذلك. تُستدعى من ExecutionEngine بعد كل خطوة."""
        node = self._nodes.get(node_id)
        if not node:
            return None
        snapshot = node.to_dict()
        self._meta_cache[node_id] = snapshot
        if persist:
            self._save()
        return snapshot

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
