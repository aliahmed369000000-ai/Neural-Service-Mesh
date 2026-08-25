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
import base64
import asyncio
import websockets
import numpy as np
import aiohttp
from datetime import datetime, timezone
from pathlib import Path
from ai.alert_manager import alert_manager
from ai.unified_memory import UnifiedMemoryManager
from ai.git_manager import GitManager
from typing import Any, Dict, List, Optional, Set
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import serialization

logger = logging.getLogger("LivingMesh")

ROOT = Path(__file__).resolve().parent.parent
LIVING_MESH_DIR = ROOT / "artifacts" / "living_mesh"
LIVING_MESH_DIR.mkdir(parents=True, exist_ok=True)
NETWORK_STATE = LIVING_MESH_DIR / "network_state.json"

class LivingMeshNode:
    def __init__(self, node_id: str = None, host: str = "127.0.0.1", port: int = None):
        self.node_id = node_id or f"mesh_{uuid.uuid4().hex[:8]}"
        self.host = host
        self.port = port
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
        self.server = None
        self.active_connections: Set = set()
        
        # تهيئة الذاكرة الموحدة (ANN + Sharding)
        self.memory = UnifiedMemoryManager(base_dir=str(LIVING_MESH_DIR / "memory"))
        
        # إنشاء مفاتيح الهوية السيادية (RSA)
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        
        # مسار تخزين المفتاح العام (للاكتشاف)
        self.keys_dir = LIVING_MESH_DIR / "keys"
        self.keys_dir.mkdir(exist_ok=True)
        self._save_public_key()
        
        # تحميل وعي Surah المسبق
        self.surah_awareness = {"status": "loading"}
        asyncio.create_task(self._load_surah_pretrain())
        
        # تهيئة مدير Git للتطور الذاتي
        self.git_manager = GitManager()

    async def _load_surah_pretrain(self):
        """تحميل أوزان Surah المسبقة من Hugging Face."""
        try:
            from huggingface_hub import hf_hub_download
            repo_id = "AliAhmedMo/surah-chain-d128-pretrain"
            logger.info(f"🧠 Integrating Surah Pretrain Awareness from {repo_id}...")
            
            # تحميل ملف الإعدادات والقاموس
            config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
            vocab_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_vocab_pretrain_d128_s1p0.json")
            
            with open(config_path, 'r') as f:
                self.surah_config = json.load(f)
            with open(vocab_path, 'r') as f:
                self.surah_vocab = json.load(f)
                
            self.surah_awareness = {
                "status": "integrated",
                "model": "surah-chain-d128",
                "vocab_size": len(self.surah_vocab),
                "layers": self.surah_config.get("n_chain_layers", 114)
            }
            logger.info("✅ Surah Awareness Synchronized with the mesh.")
        except Exception as e:
            logger.warning(f"⚠️ Surah Integration Failed: {e}")
            self.surah_awareness = {"status": "failed", "error": str(e)}
        
    def join_network(self, seed_nodes: List[Dict[str, Any]] = None):
        """الانضمام للشبكة اللامركزية واكتشاف الأقران."""
        state = self._load_state()
        is_rejoining = self.node_id in state["nodes"]
        
        capabilities = ["text", "image", "audio", "video", "tf_engine", "self_evolution"]

        self.node_info = {
            "id": self.node_id,
            "status": "online",
            "host": self.host,
            "port": self.port,
            "last_seen": datetime.now(timezone.utc).isoformat(),
            "evolution_score": self.local_evolution_score,
            "behavioral_weights": self.behavioral_weights,
            "capabilities": capabilities
        }
        
        state["nodes"][self.node_id] = self.node_info
        self._save_state(state)
        
        if seed_nodes:
            for seed in seed_nodes:
                if seed["id"] != self.node_id:
                    asyncio.create_task(self.request_peers(seed["host"], seed["port"]))
        
        if is_rejoining:
            logger.info(f"♻️ Node {self.node_id} RECOVERED and rejoined the living mesh.")
            self.recover_collective_state()
        else:
            logger.info(f"Node {self.node_id} joined the living mesh.")

    def recover_collective_state(self):
        """استعادة آخر حالة وعي للشبكة عند التعافي."""
        state = self._load_state()
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

    def check_network_health(self, timeout_seconds: int = 30) -> List[str]:
        """فحص صحة الشبكة ورصد العقد المتعطلة."""
        state = self._load_state()
        dead_nodes = []
        now = datetime.now(timezone.utc)
        
        for nid, info in state["nodes"].items():
            if info["status"] == "offline":
                continue
            
            try:
                last_seen = datetime.fromisoformat(info["last_seen"])
                if (now - last_seen).total_seconds() > timeout_seconds:
                    info["status"] = "offline"
                    dead_nodes.append(nid)
                    logger.warning(f"Node {nid} is detected as DEAD (Self-Healing Triggered)")
                    alert_manager.send_alert("CRITICAL", f"Node {nid} is DEAD", {"host": info.get("host"), "port": info.get("port")})
            except Exception:
                info["status"] = "offline"
                dead_nodes.append(nid)
        
        if dead_nodes:
            self._save_state(state)
        return dead_nodes
        
    def sync_experience(self, kind: str, experience_data: Dict[str, Any], hops: int = 0):
        """مشاركة خبرة جديدة عبر بروتوكول Gossip."""
        if hops > 15: return
        
        state = self._load_state()
        exp_entry = {
            "kind": kind,
            "data": experience_data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "node": self.node_id
        }
        if "global_experience" not in state: state["global_experience"] = []
        state["global_experience"].append(exp_entry)
        self._save_state(state)
        
        active_peers = [info for nid, info in state["nodes"].items() 
                        if info["status"] == "online" and nid != self.node_id and info.get("host")]
        
        if active_peers:
            import random
            sample_size = min(len(active_peers), 3)
            targets = random.sample(active_peers, sample_size)
            for target in targets:
                t_host = target.get("host")
                t_port = target.get("port")
                if t_host:
                    asyncio.create_task(self.send_to_peer(t_host, t_port, kind, experience_data, hops + 1))

    def _load_state(self) -> Dict[str, Any]:
        if not NETWORK_STATE.exists():
            return {"nodes": {}, "global_experience": []}
        try:
            return json.loads(NETWORK_STATE.read_text())
        except:
            return {"nodes": {}, "global_experience": []}

    def _save_state(self, state: Dict[str, Any]):
        NETWORK_STATE.write_text(json.dumps(state, indent=2))

    def _save_public_key(self):
        pub_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        (self.keys_dir / f"{self.node_id}.pub").write_bytes(pub_pem)

    def sign_message(self, message: str) -> str:
        signature = self.private_key.sign(
            message.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return base64.b64encode(signature).decode()

    @staticmethod
    def verify_signature(public_key_pem: bytes, message: str, signature: str) -> bool:
        try:
            public_key = serialization.load_pem_public_key(public_key_pem)
            public_key.verify(
                base64.b64decode(signature),
                message.encode(),
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False

    async def _process_secure_message(self, data, websocket=None):
        try:
            msg = json.loads(data)
            payload = msg.get("payload")
            signature = msg.get("signature")
            if not payload or not signature: return
            
            sender_id = payload.get("from")
            pub_key_path = self.keys_dir / f"{sender_id}.pub"
            
            if not pub_key_path.exists():
                if payload.get("kind") == "peer_discovery_request":
                    pub_pem = payload["data"].get("public_key")
                    if pub_pem: pub_key_path.write_text(pub_pem)
                else: return
            
            pub_key_pem = pub_key_path.read_bytes()
            if not self.verify_signature(pub_key_pem, json.dumps(payload, sort_keys=True), signature):
                return
            
            kind = payload.get("kind")
            exp_data = payload.get("data")
            hops = payload.get("p2p_hops", 0)
            
            if kind == "peer_discovery_request" and websocket:
                peers_list = self._get_active_peers_list()
                response = {
                    "id": f"resp_{uuid.uuid4().hex[:8]}",
                    "kind": "peer_discovery_response",
                    "data": {"peers": peers_list},
                    "from": self.node_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                sig = self.sign_message(json.dumps(response, sort_keys=True))
                resp_msg = json.dumps({"payload": response, "signature": sig})
                if isinstance(websocket, aiohttp.ClientWebSocketResponse) or hasattr(websocket, 'send_str'):
                    await websocket.send_str(resp_msg)
                else:
                    await websocket.send(resp_msg)
            
            elif kind == "peer_discovery_response":
                new_peers = exp_data.get("peers", [])
                for peer in new_peers:
                    peer_id = peer.get("id")
                    if peer_id and peer_id != self.node_id:
                        state = self._load_state()
                        if peer_id not in state["nodes"]:
                            state["nodes"][peer_id] = peer
                            self._save_state(state)
            elif kind == "evolution_task":
                logger.info(f"🧬 Node {self.node_id} received Evolution Task: {exp_data.get('task')}")
                asyncio.create_task(self._execute_evolution(exp_data))
            else:
                self.sync_experience(kind, exp_data, hops + 1)
        except Exception as e:
            logger.error(f"❌ Error processing WS message: {e}")

    async def _execute_evolution(self, task_data: Dict[str, Any]):
        """تنفيذ مهمة التطوير الذاتي برمجياً."""
        try:
            task_desc = task_data.get("task", "General Improvement")
            logger.info(f"🛠️ Starting Self-Evolution for task: {task_desc}")
            
            # تنفيذ التطور عبر GitManager
            self.git_manager.apply_evolution(task_desc)
            
            # تحديث نتيجة التطور محلياً ومشاركتها مع السرب
            self.local_evolution_score += 0.1
            self.sync_experience("evolution_sync", {
                "node": self.node_id,
                "task": task_desc,
                "score": self.local_evolution_score,
                "status": "completed"
            })
            logger.info(f"✅ Self-Evolution Completed. New Score: {self.local_evolution_score}")
        except Exception as e:
            logger.error(f"❌ Evolution Execution Failed: {e}")

    def _get_active_peers_list(self) -> List[Dict[str, Any]]:
        state = self._load_state()
        active_peers = []
        for nid, info in state.get("nodes", {}).items():
            if info.get("status") == "online":
                peer_record = info.copy()
                if "id" not in peer_record: peer_record["id"] = nid
                active_peers.append(peer_record)
        return active_peers

    async def request_peers(self, seed_host: str, seed_port: int):
        # منطق طلب الأقران
        pass

    async def send_to_peer(self, host: str, port: int, kind: str, data: Dict[str, Any], hops: int = 0):
        # منطق إرسال البيانات للأقران
        pass
