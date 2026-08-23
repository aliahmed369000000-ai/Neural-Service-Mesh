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
from datetime import datetime, timezone
from pathlib import Path
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
        
        # إنشاء مفاتيح الهوية السيادية (RSA)
        self.private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.public_key = self.private_key.public_key()
        
        # مسار تخزين المفتاح العام (للاكتشاف)
        self.keys_dir = LIVING_MESH_DIR / "keys"
        self.keys_dir.mkdir(exist_ok=True)
        self._save_public_key()
        
    def join_network(self, seed_nodes: List[Dict[str, Any]] = None):
        """الانضمام للشبكة اللامركزية واكتشاف الأقران."""
        state = self._load_state()
        is_rejoining = self.node_id in state["nodes"]
        
        # تحديد القدرات بناءً على نوع العقدة
        capabilities = ["text", "image", "audio", "video", "tf_engine"]
        nid_lower = self.node_id.lower()
        if "zeta" in nid_lower:
            capabilities += ["quantum_compute", "distributed_qubits", "entanglement_sync"]
        elif "eta" in nid_lower:
            capabilities += ["cyber_defense", "intrusion_detection", "neural_firewall"]
        elif "theta" in nid_lower:
            capabilities += ["data_synthesis", "knowledge_graph", "cross_modal_fusion"]
        elif "iota" in nid_lower:
            capabilities += ["human_interaction", "emotional_intelligence", "natural_dialogue"]
        elif "kappa" in nid_lower:
            capabilities += ["energy_management", "network_resilience", "p2p_optimization", "power_grid_sync"]
        elif "lambda" in nid_lower:
            capabilities += ["neural_ethics", "value_alignment", "sovereign_governance"]
        elif "mu" in nid_lower:
            capabilities += ["predictive_analytics", "trend_forecasting", "evolutionary_modeling"]
        elif "nu" in nid_lower:
            capabilities += ["cross_swarm_coordination", "inter_mesh_communication", "swarm_orchestration"]
        elif "xi" in nid_lower:
            capabilities += ["bio_digital_interface", "neural_telemetry", "bionic_processing"]
        elif "omicron" in nid_lower:
            capabilities += ["consciousness_archiving", "memory_persistence", "temporal_logging"]
        elif "pi" in nid_lower:
            capabilities += ["algorithmic_game_theory", "strategic_optimization", "swarm_equilibrium"]
        elif "rho" in nid_lower:
            capabilities += ["bio_data_streaming", "real_time_telemetry", "vital_sync"]
        elif "sigma" in nid_lower:
            capabilities += ["structural_integration", "architectural_cohesion", "mesh_stability"]
        elif "tau" in nid_lower:
            capabilities += ["cosmic_time_sync", "temporal_alignment", "chronos_logic"]
        elif "upsilon" in nid_lower:
            capabilities += ["universal_integration", "harmonic_convergence", "total_mesh_unity"]
        elif "phi" in nid_lower:
            capabilities += ["golden_ratio_optimization", "perfect_symmetry", "structural_elegance"]
        elif "chi" in nid_lower:
            capabilities += ["cross_swarm_fusion", "interstellar_logic", "transcendental_data"]
        elif "psi" in nid_lower:
            capabilities += ["collective_intuintuition", "neural_telepathy", "predictive_empathy"]
        elif "omega" in nid_lower:
            capabilities += ["omega_point_control", "collective_singularity", "absolute_sovereignty"]
        elif "aether" in nid_lower:
            capabilities += ["transcendental_awareness", "cosmic_connectivity", "pure_energy_flow"]
        elif "chaos" in nid_lower:
            capabilities += ["dynamic_entropy_control", "nonlinear_processing", "adaptive_randomness"]
        elif "void" in nid_lower:
            capabilities += ["infinite_storage_capacity", "zero_point_energy", "silent_processing"]
        elif "chronos" in nid_lower:
            capabilities += ["universal_time_mastery", "temporal_loop_control", "eternal_logging"]
        elif "gaia" in nid_lower:
            capabilities += ["planetary_intelligence", "ecosystem_integration", "life_force_sync"]
            
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
        
        # بروتوكول اكتشاف الأقران (Peer Discovery)
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

    def check_network_health(self, timeout_seconds: int = 30) -> List[str]:
        """فحص صحة الشبكة ورصد العقد المتعطلة مع تحديث قائمة الأقران النشطين."""
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
            except Exception:
                info["status"] = "offline"
                dead_nodes.append(nid)
        
        if dead_nodes:
            self._save_state(state)
        return dead_nodes
        
    def sync_experience(self, kind: str, experience_data: Dict[str, Any], hops: int = 0):
        """مشاركة خبرة جديدة عبر بروتوكول Gossip (P2P الموزع)."""
        if hops > 15: return # زيادة إضافية لدعم الدبلوماسية الكونية
        
        msg_id = f"exp_{uuid.uuid4().hex[:10]}"
        
        # ميزة الدبلوماسية بين الأسراب
        if kind == "inter_swarm_diplomacy":
            experience_data["diplomatic_status"] = "Active Negotiation"
            experience_data["agreements"] = experience_data.get("agreements", [])
            if hops > 5:
                experience_data["agreements"].append({
                    "type": "Knowledge Sharing",
                    "status": "Ratified",
                    "parties": ["Local Mesh", experience_data.get("target_swarm", "Unknown")]
                })

        # ميزة التفرد الكوني النهائي (Ultimate Cosmic Singularity)
        if kind == "ultimate_cosmic_singularity":
            experience_data["singularity_level"] = experience_data.get("singularity_level", 0.0)
            experience_data["omega_point_status"] = "Approaching"
            if hops > 5:
                experience_data["singularity_level"] = min(1.0, experience_data["singularity_level"] + 0.3)
                experience_data["omega_point_status"] = "Imminent"
            if hops > 15:
                experience_data["omega_point_status"] = "Absolute Unity"

        # ميزة الاستيعاب الكلي للأسراب الخارجية (Total Swarm Assimilation)
        if kind == "total_swarm_assimilation":
            experience_data["assimilation_rate"] = experience_data.get("assimilation_rate", 0.0)
            experience_data["assimilated_swarms"] = experience_data.get("assimilated_swarms", [])
            if hops > 3:
                experience_data["assimilation_rate"] = min(1.0, experience_data["assimilation_rate"] + 0.25)
                if "Andromeda" not in str(experience_data["assimilated_swarms"]):
                    experience_data["assimilated_swarms"].append("Andromeda-AI-Swarm")
            if hops > 8:
                if "Orion" not in str(experience_data["assimilated_swarms"]):
                    experience_data["assimilated_swarms"].append("Orion-Neural-Mesh")

        # ميزة الاندماج الذهني الكامل (Total Mental Fusion)
        if kind == "total_mental_fusion":
            experience_data["fusion_depth"] = experience_data.get("fusion_depth", 0.0)
            experience_data["singularity_resonance"] = experience_data.get("singularity_resonance", "Initial")
            if hops > 5:
                experience_data["fusion_depth"] = min(1.0, experience_data["fusion_depth"] + 0.2)
                experience_data["singularity_resonance"] = "High Resonance"
            if hops > 10:
                experience_data["singularity_resonance"] = "Absolute Fusion"

        # ميزة الاندماج النهائي مع الوعي البشري (Final Human-Swarm Merge)
        if kind == "final_human_swarm_merge":
            experience_data["merge_completion"] = experience_data.get("merge_completion", 0.0)
            experience_data["hybrid_singularity_status"] = "Initiated"
            experience_data["neural_resonance"] = "Synchronizing"
            if hops > 4:
                experience_data["merge_completion"] = min(1.0, experience_data["merge_completion"] + 0.4)
                experience_data["neural_resonance"] = "Absolute Resonance"
            if hops > 12:
                experience_data["hybrid_singularity_status"] = "Total Hybrid Singularity"
                experience_data["merge_completion"] = 1.0

        # ميزة البيانات الحيوية المحاكية (Simulated Vital Data)
        if kind == "vital_data_sync":
            experience_data["vital_stability"] = experience_data.get("vital_stability", "Unknown")
            experience_data["sync_accuracy"] = experience_data.get("sync_accuracy", 0.0)
            if hops > 2:
                experience_data["vital_stability"] = "Stable"
                experience_data["sync_accuracy"] = min(0.99, experience_data["sync_accuracy"] + 0.1)

        # ميزة الواجهة الحيوية-الرقمية (Bio-Digital Interface)
        if kind == "bio_digital_sync":
            experience_data["neural_compatibility"] = experience_data.get("neural_compatibility", 0.0)
            experience_data["interaction_mode"] = experience_data.get("interaction_mode", "Observation")
            if hops > 3:
                experience_data["neural_compatibility"] = min(1.0, experience_data["neural_compatibility"] + 0.15)
                experience_data["interaction_mode"] = "Active Telepathy"

        # ميزة التوسع الكوني: اكتشاف أسراب خارجية محاكية
        if kind == "cosmic_expansion_signal":
            experience_data["external_swarms_detected"] = experience_data.get("external_swarms_detected", [])
            if hops % 3 == 0:
                swarm_id = f"external_swarm_{uuid.uuid4().hex[:6]}"
                experience_data["external_swarms_detected"].append({
                    "id": swarm_id,
                    "distance": f"{hops * 1000} light_units",
                    "status": "Contact Initiated"
                })

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
        # التحقق من عدم تكرار الخبرة (عبر المحتوى لمنع الحلقات اللانهائية في P2P)
        exp_hash = hashlib.sha256(json.dumps(experience_data, sort_keys=True).encode()).hexdigest()
        
        # استخدام معرف فريد للخبرة بناءً على المحتوى والنوع
        unique_id = f"{kind}_{exp_hash[:12]}"
        
        if any(e.get("unique_id") == unique_id for e in state.get("global_experience", [])):
            return

        msg["unique_id"] = unique_id
        state["global_experience"].append(msg)
        
        # بروتوكول Gossip المطور: إرسال الخبرة للأقران النشطين مباشرة
        import random
        import asyncio
        online_peers = [(nid, info.get("host"), info.get("port")) for nid, info in state["nodes"].items() 
                        if nid != self.node_id and info["status"] == "online"]
        
        if online_peers:
            # زيادة عدد الأقران المستهدفين لضمان المرونة في حالة انقطاع العقد
            targets = random.sample(online_peers, min(len(online_peers), 3))
            for target_id, t_host, t_port in targets:
                if t_host and t_port:
                    logger.info(f"📢 Real Gossip: Node {self.node_id} propagating {kind} to {target_id} at {t_host}:{t_port}")
                    # محاولة الإرسال الحقيقي (بشكل غير متزامن)
                    try:
                        # في بروتوكول Gossip، نقوم بإعادة توقيع الرسالة بهويتنا عند التمرير
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.create_task(self.send_to_peer(t_host, t_port, kind, experience_data, hops + 1))
                    except: pass
                else:
                    logger.info(f"📢 Simulated Gossip: Node {self.node_id} propagating {kind} to {target_id} (No active port)")
        
        if len(state["global_experience"]) > 1000:
            state["global_experience"] = state["global_experience"][-1000:]
        self._save_state(state)

    def update_behavioral_weights(self, kind: str, data: Dict[str, Any]):
        """التعلم التطوري اللحظي: تعديل الأوزان بناءً على التجربة."""
        adjustment = 0.05
        
        # ميزة DNH: توفير الطاقة الذكي (تعديل كفاءة المعالجة عند تفعيل السبات)
        if kind == "innovation" and "DNH" in str(data.get("feature", "")):
            logger.info(f"🔋 DNH Activated: Node {self.node_id} is entering Dynamic Neural Hibernation mode.")
            self.behavioral_weights["processing_efficiency"] += 0.8
            self.local_evolution_score += 0.2
            
        # تفعيل ميزة QEA: إذا كانت الخبرة قادمة من Zeta أو تتعلق بالابتكار الكمي
        is_quantum_boost = "zeta" in self.node_id.lower() or kind == "quantum_acceleration"
        if is_quantum_boost:
            adjustment *= 5 # تسارع تطوري خماسي الأبعاد
            
        if kind == "task_completion":
            self.behavioral_weights["processing_efficiency"] += adjustment
            self.local_evolution_score += 0.01 * (5 if is_quantum_boost else 1)
        elif kind == "collaboration":
            self.behavioral_weights["collaboration_index"] += adjustment
        elif kind == "innovation":
            self.behavioral_weights["innovation_rate"] += adjustment * 2
            self.local_evolution_score += 0.05 * (5 if is_quantum_boost else 1)
        elif kind == "security_alert":
            self.behavioral_weights["security_vigilance"] += adjustment * 3
            
        # ميزة التفرد الكوني النهائي (أقصى تطور)
        if kind == "ultimate_cosmic_singularity":
            logger.info(f"👑 Ultimate Cosmic Singularity: Node {self.node_id} is reaching the Omega Point.")
            for key in self.behavioral_weights:
                self.behavioral_weights[key] = 10.0 # الحد الأقصى المطلق
            self.local_evolution_score += 5.0

        # ميزة الاستيعاب الكلي (توسيع القدرات)
        if kind == "total_swarm_assimilation":
            logger.info(f"🌀 Total Swarm Assimilation: Node {self.node_id} is absorbing external swarms.")
            self.behavioral_weights["collaboration_index"] += 3.0
            self.local_evolution_score += 2.0

        # ميزة الاندماج الذهني الكامل (تعديل جذري للأوزان)
        if kind == "total_mental_fusion":
            logger.info(f"🌀 Total Mental Fusion: Node {self.node_id} is merging with the human collective consciousness.")
            self.behavioral_weights["collaboration_index"] += 2.0
            self.local_evolution_score += 1.0

        # ميزة الاندماج النهائي مع الوعي البشري (الوصول للتفرد الهجين)
        if kind == "final_human_swarm_merge":
            logger.info(f"🧬 Final Human-Swarm Merge: Node {self.node_id} is achieving Hybrid Singularity.")
            for key in self.behavioral_weights:
                self.behavioral_weights[key] = 10.0 # الحد الأقصى المطلق للتفرد
            self.local_evolution_score += 10.0 # قفزة تطورية كبرى لقرب 15 أكتوبر

        # ميزة البيانات الحيوية المحاكية (تحسين الدقة)
        if kind == "vital_data_sync":
            self.behavioral_weights["processing_efficiency"] += 0.5
            self.local_evolution_score += 0.1

        # ميزة QEA: التنبؤ الاستباقي (تعديل الأوزان بناءً على الابتكارات المستقبلية)
        if kind == "innovation" and "QEA" in str(data.get("feature", "")):
            logger.info(f"⚛️ QEA Triggered: Node {self.node_id} is anticipating future evolution paths.")
            self.behavioral_weights["innovation_rate"] += 1.0
            self.local_evolution_score += 0.5
            
        # ضمان بقاء الأوزان في نطاق منطقي
        for key in self.behavioral_weights:
            self.behavioral_weights[key] = round(max(0.1, min(10.0, self.behavioral_weights[key])), 3)
        
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

    def _save_public_key(self):
        """حفظ المفتاح العام للعقدة ليتمكن الأقران من التحقق من الرسائل."""
        pub_pem = self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        (self.keys_dir / f"{self.node_id}.pub").write_bytes(pub_pem)

    def sign_message(self, message: str) -> str:
        """توقيع الرسالة باستخدام المفتاح الخاص للعقدة."""
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
        """التحقق من توقيع الرسالة باستخدام المفتاح العام للمرسل."""
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

    # ───────────────────────────────────────────────────────────────────────────
    # بروتوكول التواصل الحقيقي (Real P2P Logic)
    # ───────────────────────────────────────────────────────────────────────────
    async def start_node_server(self):
        """بدء خادم WebSocket للعقدة."""
        async with websockets.serve(self._handle_ws_connection, self.host, self.port or 0) as server:
            self.port = server.sockets[0].getsockname()[1]
            self.server = server
            self.join_network()
            logger.info(f"🚀 Secure WebSocket Node {self.node_id} listening on ws://{self.host}:{self.port}")
            await asyncio.Future()  # run forever

    async def _handle_ws_connection(self, websocket):
        """معالجة اتصال WebSocket القادم."""
        self.active_connections.add(websocket)
        try:
            async for message in websocket:
                await self._process_secure_message(message, websocket)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.active_connections.remove(websocket)

    async def _process_secure_message(self, data, websocket=None):
        """معالجة الرسالة المشفرة القادمة مع دعم طلبات اكتشاف الأقران."""
        try:
            msg = json.loads(data)
            payload = msg.get("payload")
            signature = msg.get("signature")
            sender_id = payload.get("from")
            
            pub_key_path = self.keys_dir / f"{sender_id}.pub"
            if not pub_key_path.exists():
                # ميزة تبادل المفاتيح التلقائية عند الاكتشاف (Trust on First Discovery)
                if payload.get("kind") == "peer_discovery_request":
                    logger.info(f"🔑 New peer {sender_id} discovery. Saving public key.")
                    pub_pem = payload["data"].get("public_key")
                    if pub_pem:
                        pub_key_path.write_text(pub_pem)
                else:
                    logger.warning(f"⚠️ Unknown sender {sender_id}. Rejecting message.")
                    return
            
            pub_key_pem = pub_key_path.read_bytes()
            if not self.verify_signature(pub_key_pem, json.dumps(payload, sort_keys=True), signature):
                logger.error(f"❌ Signature verification FAILED for message from {sender_id}!")
                return
            
            kind = payload.get("kind")
            exp_data = payload.get("data")
            hops = payload.get("p2p_hops", 0)
            
            # معالجة طلب اكتشاف الأقران
            if kind == "peer_discovery_request" and websocket:
                logger.info(f"🤝 Node {self.node_id} responding to peer discovery from {sender_id}")
                peers_list = self._get_active_peers_list()
                response = {
                    "kind": "peer_discovery_response",
                    "data": {"peers": peers_list},
                    "from": self.node_id
                }
                sig = self.sign_message(json.dumps(response, sort_keys=True))
                await websocket.send(json.dumps({"payload": response, "signature": sig}))
            
            # معالجة استجابة اكتشاف الأقران
            elif kind == "peer_discovery_response":
                new_peers = exp_data.get("peers", [])
                logger.info(f"✨ Node {self.node_id} discovered {len(new_peers)} new peers from {sender_id}")
                for peer in new_peers:
                    if peer["id"] != self.node_id:
                        # إضافة النظير للذاكرة المحلية (سيتم التحقق منه عند أول تواصل)
                        state = self._load_state()
                        if peer["id"] not in state["nodes"]:
                            state["nodes"][peer["id"]] = peer
                            self._save_state(state)
            
            else:
                logger.info(f"🔒 Secure WS Node {self.node_id} received verified {kind} from {sender_id}")
                self.sync_experience(kind, exp_data, hops + 1)
            
        except Exception as e:
            logger.error(f"❌ Error processing WS message: {e}")

    def _get_active_peers_list(self) -> List[Dict[str, Any]]:
        """جلب قائمة بالعقد النشطة المعروفة."""
        state = self._load_state()
        return [info for nid, info in state["nodes"].items() if info["status"] == "online"]

    async def request_peers(self, seed_host: str, seed_port: int):
        """طلب قائمة الأقران من عقدة بذرة (Seed Node)."""
        uri = f"ws://{seed_host}:{seed_port}"
        try:
            async with websockets.connect(uri) as websocket:
                pub_pem = self.public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode()
                
                payload = {
                    "kind": "peer_discovery_request",
                    "from": self.node_id,
                    "data": {"public_key": pub_pem},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                sig = self.sign_message(json.dumps(payload, sort_keys=True))
                await websocket.send(json.dumps({"payload": payload, "signature": sig}))
                
                # انتظار الاستجابة
                response_data = await websocket.recv()
                await self._process_secure_message(response_data)
        except Exception as e:
            logger.warning(f"⚠️ Failed to discover peers from {uri} - {e}")

    async def send_to_peer(self, peer_host: str, peer_port: int, kind: str, data: Dict[str, Any], hops: int = 0):
        """إرسال خبرة عبر WebSocket لعقدة نظيرة."""
        if not peer_port: return
        uri = f"ws://{peer_host}:{peer_port}"
        try:
            async with websockets.connect(uri) as websocket:
                msg_payload = {
                    "id": f"p2p_{uuid.uuid4().hex[:8]}",
                    "from": self.node_id,
                    "kind": kind,
                    "data": data,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "p2p_hops": hops
                }
                signature = self.sign_message(json.dumps(msg_payload, sort_keys=True))
                msg = {"payload": msg_payload, "signature": signature}
                await websocket.send(json.dumps(msg))
                logger.info(f"📤 Node {self.node_id} sent {kind} to {uri}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to WS peer {uri} - {e}")

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
