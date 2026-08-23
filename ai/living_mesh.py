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
        self.behavioral_weights = {
            "processing_efficiency": 1.0,
            "collaboration_index": 1.0,
            "innovation_rate": 1.0,
            "security_vigilance": 1.0
        }
        
    def join_network(self):
        """الانضمام للشبكة اللامركزية."""
        state = self._load_state()
        is_rejoining = self.node_id in state["nodes"]
        
        # تحديد القدرات بناءً على نوع العقدة
        capabilities = ["text", "image", "audio", "video", "tf_engine"]
        if "zeta" in self.node_id.lower():
            capabilities += ["quantum_compute", "distributed_qubits", "entanglement_sync"]
            
        state["nodes"][self.node_id] = {
            "status": "online",
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "evolution_score": self.local_evolution_score,
            "behavioral_weights": self.behavioral_weights,
            "capabilities": capabilities,
            "assigned_tasks": []
        }
        self._save_state(state)
        
        if is_rejoining:
            logger.info(f"♻️ Node {self.node_id} RECOVERED and rejoined the living mesh.")
            self.recover_collective_state()
        else:
            logger.info(f"Node {self.node_id} joined the living mesh.")

    def recover_collective_state(self):
        """استعادة آخر حالة وعي للشبكة عند التعافي."""
        state = self._load_state()
        # استرجاع آخر تحديثات التطور والخبرات لمزامنة الحالة المحلية
        recent_exps = state.get("global_experience", [])[-50:]
        for exp in recent_exps:
            if exp["kind"] == "evolution_sync":
                self.local_evolution_score = max(self.local_evolution_score, exp["data"].get("score", 0.0))
        logger.info(f"🧠 Node {self.node_id} synchronized state: Evolution Score = {self.local_evolution_score}")

    def send_heartbeat(self):
        """إرسال نبض قلب لتأكيد الوجود وتحديث الحالة."""
        state = self._load_state()
        if self.node_id in state["nodes"]:
            state["nodes"][self.node_id]["last_seen"] = datetime.now(timezone.utc).isoformat()
            state["nodes"][self.node_id]["status"] = "online"
            self._save_state(state)
            return True
        return False

    def check_network_health(self, timeout_seconds: int = 60) -> List[str]:
        """فحص صحة الشبكة ورصد العقد المتعطلة."""
        state = self._load_state()
        dead_nodes = []
        now = datetime.now(timezone.utc)
        
        for nid, info in state["nodes"].items():
            if info["status"] == "offline":
                continue
                
            last_seen = datetime.fromisoformat(info["last_seen"])
            if (now - last_seen).total_seconds() > timeout_seconds:
                info["status"] = "offline"
                dead_nodes.append(nid)
                logger.warning(f"Node {nid} is detected as DEAD (Self-Healing Triggered)")
        
        if dead_nodes:
            self._save_state(state)
        return dead_nodes
        
    def sync_experience(self, kind: str, experience_data: Dict[str, Any], hops: int = 0):
        """مشاركة خبرة جديدة عبر بروتوكول Gossip (P2P الموزع)."""
        if hops > 5: return # منع الحلقات اللانهائية في المحاكاة
        
        msg_id = f"exp_{uuid.uuid4().hex[:10]}"
        msg = {
            "id": msg_id,
            "from": self.node_id,
            "kind": kind,
            "data": experience_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "p2p_hops": hops
        }
        
        # التعلم اللحظي الموزع: العقدة التي تستقبل الخبرة تعدل أوزانها أيضاً
        self.update_behavioral_weights(kind, experience_data)
        
        state = self._load_state()
        # التحقق من عدم تكرار الخبرة
        if any(e.get("id") == msg_id for e in state.get("global_experience", [])):
            return

        state["global_experience"].append(msg)
        
        # محاكاة بروتوكول Gossip: إرسال الخبرة لـ 2 من الأقران عشوائياً
        import random
        online_peers = [nid for nid, info in state["nodes"].items() if nid != self.node_id and info["status"] == "online"]
        if online_peers:
            targets = random.sample(online_peers, min(len(online_peers), 2))
            for target in targets:
                logger.info(f"📢 Gossip: Node {self.node_id} propagating {kind} to {target} (Hop {hops})")
                # في بيئة حقيقية، هذا استدعاء RPC/Socket لعقدة أخرى
        
        if len(state["global_experience"]) > 1000:
            state["global_experience"] = state["global_experience"][-1000:]
        self._save_state(state)

    def update_behavioral_weights(self, kind: str, data: Dict[str, Any]):
        """التعلم التطوري اللحظي: تعديل الأوزان بناءً على التجربة."""
        adjustment = 0.05
        if kind == "task_completion":
            self.behavioral_weights["processing_efficiency"] += adjustment
            self.local_evolution_score += 0.01
        elif kind == "collaboration":
            self.behavioral_weights["collaboration_index"] += adjustment
        elif kind == "innovation":
            self.behavioral_weights["innovation_rate"] += adjustment * 2
            self.local_evolution_score += 0.05
        elif kind == "security_alert":
            self.behavioral_weights["security_vigilance"] += adjustment * 3
            
        # ضمان بقاء الأوزان في نطاق منطقي
        for key in self.behavioral_weights:
            self.behavioral_weights[key] = round(max(0.1, min(5.0, self.behavioral_weights[key])), 3)
        
        logger.info(f"🧬 Real-time Evolution: Node {self.node_id} updated weights: {self.behavioral_weights}")
        
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

def get_network_snapshot() -> Dict[str, Any]:
    """الحصول على لقطة كاملة لحالة الشبكة للعرض في الواجهة."""
    if not NETWORK_STATE.is_file():
        return {"nodes": {}, "global_experience": [], "active_tasks": []}
    try:
        return json.loads(NETWORK_STATE.read_text(encoding="utf-8"))
    except:
        return {"nodes": {}, "global_experience": [], "active_tasks": []}

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
