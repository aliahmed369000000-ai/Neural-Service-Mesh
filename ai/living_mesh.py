"""
Decentralized Living Neural Mesh Engine — محرك الشبكة العصبية اللامركزية الحية
==========================================================================
هذا المحرك يحول الوكلاء من مجرد سكربتات معزولة إلى عقد (Nodes) في شبكة عالمية 
تتبادل الخبرات والأوزان والتطورات بشكل حي ولا مركزي.
"""
import hashlib
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("LivingMesh")

ROOT = Path(__file__).resolve().parent.parent
LIVING_MESH_DIR = ROOT / "artifacts" / "living_mesh"
LIVING_MESH_DIR.mkdir(parents=True, exist_ok=True)
NETWORK_STATE = LIVING_MESH_DIR / "network_state.json"

class LivingMeshNode:
    def __init__(self, node_id: str = None):
        self.node_id = node_id or f"mesh_{uuid.uuid4().hex[:8]}"
        self.peers = []
        self.collective_memory = {}
        self.local_evolution_score = 0.0
        self.last_sync = None
        
    def join_network(self):
        """الانضمام للشبكة اللامركزية."""
        state = self._load_state()
        state["nodes"][self.node_id] = {
            "status": "online",
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "evolution_score": self.local_evolution_score,
            "capabilities": ["text", "image", "audio", "video", "tf_engine"]
        }
        self._save_state(state)
        logger.info(f"Node {self.node_id} joined the living mesh.")
        
    def sync_experience(self, kind: str, experience_data: Dict[str, Any]):
        """مشاركة خبرة جديدة مع الشبكة."""
        msg = {
            "id": f"exp_{uuid.uuid4().hex[:10]}",
            "from": self.node_id,
            "kind": kind,
            "data": experience_data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        state = self._load_state()
        state["global_experience"].append(msg)
        # الاحتفاظ بآخر 1000 خبرة عالمية فقط لضمان السرعة
        if len(state["global_experience"]) > 1000:
            state["global_experience"] = state["global_experience"][-1000:]
        self._save_state(state)
        
    def get_evolutionary_updates(self) -> List[Dict[str, Any]]:
        """الحصول على تحديثات التطور من العقد الأخرى."""
        state = self._load_state()
        updates = [exp for exp in state["global_experience"] if exp["kind"] == "evolution_sync"]
        return updates

    def broadcast_weight_delta(self, layer_name: str, delta_hash: str):
        """نشر تحديثات الأوزان (Deltas) عبر الشبكة."""
        sync_data = {
            "layer": layer_name,
            "delta_hash": delta_hash,
            "applied_at": datetime.now(timezone.utc).isoformat()
        }
        self.sync_experience("weight_delta_sync", sync_data)
        logger.info(f"Node {self.node_id} broadcasted weight delta for {layer_name}")

    def _load_state(self) -> Dict[str, Any]:
        if NETWORK_STATE.is_file():
            try:
                return json.loads(NETWORK_STATE.read_text(encoding="utf-8"))
            except: pass
        return {"nodes": {}, "global_experience": [], "created_at": datetime.now(timezone.utc).isoformat()}

    def _save_state(self, state: Dict[str, Any]):
        NETWORK_STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")

# ───────────────────────────────────────────────────────────────────────────
# بروتوكول السيادة التطورية (Sovereign Evolution Protocol)
# ───────────────────────────────────────────────────────────────────────────
def trigger_evolutionary_sync(node_id: str, new_score: float, change_log: str):
    """إرسال إشارة تطور للشبكة بالكامل."""
    node = LivingMeshNode(node_id)
    sync_data = {
        "score": new_score,
        "change_log": change_log,
        "engine": "tensorflow_v2"
    }
    node.sync_experience("evolution_sync", sync_data)
    return f"Evolution sync triggered for node {node_id}"
