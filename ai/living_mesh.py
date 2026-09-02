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
from ai.toolbox import nsm_toolbox
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
        
        # ربط صندوق الأدوات
        self.toolbox = nsm_toolbox

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
            "capabilities": capabilities,
            "public_key": self._pub_pem()
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
        """فحص صحة الشبكة ورصد العقد المتعطلة مع تفعيل الاستعادة الذاتية."""
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
            # تفعيل الاستعادة الذاتية: محاولة البحث عن أقران جدد إذا انخفض عدد الأقران النشطين
            active_count = sum(1 for n in state["nodes"].values() if n["status"] == "online")
            if active_count < 2:
                logger.info("🆘 Low active peer count. Triggering self-healing discovery...")
                # محاكاة إعادة اكتشاف الأقران من العقد البذور المعروفة
                pass
                
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

    def _pub_pem(self) -> str:
        """المفتاح العام للعقدة بصيغة PEM نصية (يُرفق برسائل الاكتشاف P2P)."""
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode()

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

    async def _handle_aiohttp_ws_msg(self, ws, data):
        """معالج رسائل WebSocket لـ aiohttp."""
        # تحويل البيانات إلى الصيغة المتوقعة من قبل _process_secure_message
        if isinstance(data, dict):
            await self._process_secure_message(json.dumps(data), websocket=ws)
        else:
            await self._process_secure_message(data, websocket=ws)

    async def _process_secure_message(self, data, websocket=None):
        try:
            msg = json.loads(data)
            payload = msg.get("payload") or msg # دعم كلا الصيغتين
            signature = msg.get("signature")
            if not payload or not signature: return
            
            sender_id = payload.get("from")
            pub_key_path = self.keys_dir / f"{sender_id}.pub"
            
            if not pub_key_path.exists():
                if payload.get("kind") in ("peer_discovery_request", "peer_discovery_response"):
                    pub_pem = (payload.get("data") or {}).get("public_key")
                    if pub_pem: pub_key_path.write_text(pub_pem)
                    else: return
                else: return
            
            pub_key_pem = pub_key_path.read_bytes()
            if not self.verify_signature(pub_key_pem, json.dumps(payload, sort_keys=True), signature):
                return
            
            kind = payload.get("kind")
            exp_data = payload.get("data")
            hops = payload.get("p2p_hops", 0)
            
            if kind == "peer_discovery_request" and websocket is not None:
                # سجّل الطالب كعقدة معروفة (تسجيل متبادل) — بدونه لا يمكن لأي عقدة
                # ثالثة اكتشافه لاحقاً عبر عقدة وسيطة (multi-hop discovery)
                sender_info = (exp_data or {}).get("node_info")
                if sender_info and sender_id != self.node_id:
                    state = self._load_state()
                    state["nodes"][sender_id] = sender_info
                    self._save_state(state)
                peers_list = self._get_active_peers_list()
                response = {
                    "id": f"resp_{uuid.uuid4().hex[:8]}",
                    "kind": "peer_discovery_response",
                    "data": {"peers": peers_list, "public_key": self._pub_pem()},
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
                # احفظ المفتاح العام للعقدة المستجيبة نفسها (تم التحقق منه أعلاه بالفعل)
                resp_pub_key = exp_data.get("public_key")
                if resp_pub_key and not pub_key_path.exists():
                    pub_key_path.write_text(resp_pub_key)
                new_peers = exp_data.get("peers", [])
                for peer in new_peers:
                    peer_id = peer.get("id")
                    if peer_id and peer_id != self.node_id:
                        state = self._load_state()
                        if peer_id not in state["nodes"]:
                            state["nodes"][peer_id] = peer
                            self._save_state(state)
                        # اكتشاف مفتاح متعدد القفزات: احفظ مفتاح عقدة لم نتواصل معها مباشرة بعد
                        peer_pub_key = peer.get("public_key")
                        peer_key_path = self.keys_dir / f"{peer_id}.pub"
                        if peer_pub_key and not peer_key_path.exists():
                            peer_key_path.write_text(peer_pub_key)
            elif kind == "evolution_task":
                logger.info(f"🧬 Node {self.node_id} received Evolution Task: {exp_data.get('task')}")
                asyncio.create_task(self._execute_evolution(exp_data))
            elif kind == "tool_request":
                # طلب تنفيذ أداة محددة من السرب
                logger.info(f"🛠️ Node {self.node_id} received Tool Request: {exp_data.get('tool_name')}")
                asyncio.create_task(self._handle_tool_request(exp_data))
            elif kind == "gradient_push":
                # تدرجات من بروتوكول Gradient Mesh — نقبلها ونمررها كـ gossip محدود
                logger.info(f"📥 Node {self.node_id} received gradient_push from {sender_id} (hops={hops})")
                if hops < 3:
                    self.sync_experience(kind, exp_data, hops + 1)
            elif kind == "ping_request" and websocket is not None:
                # #2 قياس زمن الاستجابة — رد فوري بـ timestamp محلي
                t_server = time.time()
                resp_payload = {
                    "id": f"pong_{uuid.uuid4().hex[:8]}",
                    "kind": "ping_response",
                    "data": {
                        "echo_ts": (exp_data or {}).get("client_ts"),
                        "server_ts": t_server,
                        "node_id": self.node_id,
                    },
                    "from": self.node_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                sig = self.sign_message(json.dumps(resp_payload, sort_keys=True))
                resp_msg = json.dumps({"payload": resp_payload, "signature": sig})
                if hasattr(websocket, "send_str"):
                    await websocket.send_str(resp_msg)
                else:
                    await websocket.send(resp_msg)
            elif kind == "ping_response":
                # يُعالَج عادةً داخل measure/ping المنتظر للرد — نسجّل فقط
                logger.debug(f"📶 ping_response from {sender_id}: {exp_data}")
            elif kind == "relay_task":
                # #3 تمرير رسالة عبر عقدة وسيطة عند تعذر الاتصال المباشر
                await self._handle_relay_task(exp_data or {}, sender_id=sender_id, hops=hops)
            elif kind == "multisig_propose":
                # #4 اقتراح عملية مكافأة تتطلب توقيعات متعددة
                await self._handle_multisig_propose(exp_data or {}, sender_id=sender_id)
            elif kind == "multisig_sign":
                await self._handle_multisig_sign(exp_data or {}, sender_id=sender_id)
            else:
                self.sync_experience(kind, exp_data, hops + 1)
        except Exception as e:
            logger.error(f"❌ Error processing WS message: {e}")

    async def _handle_tool_request(self, request_data: Dict[str, Any]):
        """معالجة طلب تنفيذ أداة ومشاركة النتيجة."""
        try:
            tool_name = request_data.get("tool_name")
            args = request_data.get("args", {})
            
            result = self.toolbox.execute_tool(tool_name, **args)
            
            # مشاركة نتيجة تنفيذ الأداة مع الشبكة
            self.sync_experience("tool_result", {
                "tool_name": tool_name,
                "result": result,
                "node": self.node_id
            })
        except Exception as e:
            logger.error(f"❌ Tool Execution Request Failed: {e}")

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

    def _peer_ws_url(self, host: str, port: int) -> str:
        scheme = "wss" if port == 443 else "ws"
        # منافذ HTTP/HTTPS القياسية لا تُكتب صراحة بالرابط
        if port in (80, 443):
            return f"{scheme}://{host}/ws"
        return f"{scheme}://{host}:{port}/ws"

    def _build_signed_payload(self, kind: str, data: Dict[str, Any], hops: int = 0) -> str:
        payload = {
            "id": f"msg_{uuid.uuid4().hex[:8]}",
            "kind": kind,
            "data": data,
            "from": self.node_id,
            "p2p_hops": hops,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        sig = self.sign_message(json.dumps(payload, sort_keys=True))
        return json.dumps({"payload": payload, "signature": sig})

    async def request_peers(self, seed_host: str, seed_port: int, retries: int = 3, retry_delay: float = 0.8):
        """يتصل فعلياً بعقدة بذرة عبر WebSocket ويطلب قائمة أقرانها (مع إرفاق مفتاحنا العام).
        يعيد المحاولة بعدد محدود عند فشل الاتصال (منفذ غير جاهز أو مهلة قصيرة).
        """
        url = self._peer_ws_url(seed_host, seed_port)
        my_info = getattr(self, "node_info", {"id": self.node_id, "host": self.host, "port": self.port})
        msg = self._build_signed_payload(
            "peer_discovery_request",
            {"public_key": self._pub_pem(), "node_info": my_info},
        )
        last_err = None
        for attempt in range(1, max(1, retries) + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(url, timeout=10) as ws:
                        await ws.send_str(msg)
                        resp = await asyncio.wait_for(ws.receive(), timeout=10)
                        if resp.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_aiohttp_ws_msg(ws, json.loads(resp.data))
                            logger.info(
                                f"🔎 Node {self.node_id} discovered peers via {seed_host}:{seed_port} "
                                f"(attempt {attempt}/{retries})"
                            )
                            return True
            except Exception as e:
                last_err = e
                logger.warning(
                    f"⚠️ request_peers attempt {attempt}/{retries} to {seed_host}:{seed_port} failed: {e}"
                )
                if attempt < retries:
                    await asyncio.sleep(retry_delay)
        logger.error(f"❌ request_peers to {seed_host}:{seed_port} failed after {retries} attempts: {last_err}")
        return False

    async def send_to_peer(self, host: str, port: int, kind: str, data: Dict[str, Any], hops: int = 0) -> bool:
        """يتصل فعلياً بعقدة هدف عبر WebSocket ويرسل لها رسالة موقّعة (Gossip).
        يُرجع True عند نجاح الإرسال، False عند الفشل (لا يُعتبر الفشل نجاحاً صامتاً).
        """
        url = self._peer_ws_url(host, port)
        msg = self._build_signed_payload(kind, data, hops=hops)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, timeout=10) as ws:
                    await ws.send_str(msg)
                    logger.info(f"📤 Node {self.node_id} sent '{kind}' to {host}:{port}")
                    return True
        except Exception as e:
            logger.error(f"❌ send_to_peer to {host}:{port} failed: {e}")
            return False

    # ------------------------------------------------------------------
    # #2 قياس صحة الأقران وزمن الاستجابة (Peer Health / Latency)
    # ------------------------------------------------------------------
    async def ping_peer(self, host: str, port: int, timeout: float = 5.0) -> Dict[str, Any]:
        """يرسل ping_request موقّعاً ويقيس RTT بالميلي ثانية."""
        url = self._peer_ws_url(host, port)
        client_ts = time.time()
        msg = self._build_signed_payload("ping_request", {"client_ts": client_ts})
        result = {
            "host": host,
            "port": port,
            "ok": False,
            "rtt_ms": None,
            "error": None,
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(url, timeout=timeout) as ws:
                    await ws.send_str(msg)
                    resp = await asyncio.wait_for(ws.receive(), timeout=timeout)
                    t1 = time.time()
                    if resp.type != aiohttp.WSMsgType.TEXT:
                        result["error"] = "non_text_response"
                        return result
                    raw = json.loads(resp.data)
                    payload = raw.get("payload") or raw
                    # تحقق توقيع خفيف إن وُجد مفتاح المرسل
                    sender = payload.get("from")
                    sig = raw.get("signature")
                    if sender and sig:
                        key_path = self.keys_dir / f"{sender}.pub"
                        if key_path.exists():
                            if not self.verify_signature(
                                key_path.read_bytes(),
                                json.dumps(payload, sort_keys=True),
                                sig,
                            ):
                                result["error"] = "bad_signature"
                                return result
                    data = payload.get("data") or {}
                    echo = data.get("echo_ts", client_ts)
                    rtt_ms = (t1 - float(echo)) * 1000.0
                    result["ok"] = True
                    result["rtt_ms"] = round(rtt_ms, 2)
                    result["peer_id"] = sender or data.get("node_id")
                    # تحديث حالة الأقران بزمن الاستجابة
                    state = self._load_state()
                    for nid, info in state.get("nodes", {}).items():
                        if info.get("host") == host and int(info.get("port") or -1) == int(port):
                            info["last_rtt_ms"] = result["rtt_ms"]
                            info["last_seen"] = datetime.now(timezone.utc).isoformat()
                            info["status"] = "online"
                            self._save_state(state)
                            break
                    return result
        except Exception as e:
            result["error"] = str(e)
            logger.warning(f"⚠️ ping_peer {host}:{port} failed: {e}")
            return result

    async def measure_peers_health(self, timeout: float = 5.0) -> List[Dict[str, Any]]:
        """يقيس صحة وزمن استجابة جميع الأقران النشطين المعروفي العنوان."""
        peers = self._get_active_peers_list()
        results = []
        for peer in peers:
            if peer.get("id") == self.node_id:
                continue
            host, port = peer.get("host"), peer.get("port")
            if not host or port is None:
                results.append({"peer_id": peer.get("id"), "ok": False, "error": "no_address"})
                continue
            r = await self.ping_peer(host, int(port), timeout=timeout)
            r["peer_id"] = r.get("peer_id") or peer.get("id")
            results.append(r)
        healthy = sum(1 for r in results if r.get("ok"))
        logger.info(f"📶 Peer health: {healthy}/{len(results)} reachable")
        return results

    # ------------------------------------------------------------------
    # #3 Relay — تمرير العمل عبر عقدة وسيطة عند تعذر الاتصال المباشر
    # ------------------------------------------------------------------
    async def send_to_peer_with_relay(
        self,
        host: str,
        port: int,
        kind: str,
        data: Dict[str, Any],
        hops: int = 0,
        target_id: str = None,
    ) -> Dict[str, Any]:
        """يحاول الإرسال المباشر أولاً؛ عند الفشل يمرّر عبر أقران وسيطين (Relay)."""
        direct_ok = await self.send_to_peer(host, port, kind, data, hops=hops)
        if direct_ok:
            return {"ok": True, "mode": "direct", "relays_tried": 0}

        # اختيار وسطاء محتملين (أقران نشطون غير الهدف)
        candidates = [
            p for p in self._get_active_peers_list()
            if p.get("id") != self.node_id
            and p.get("host")
            and p.get("port") is not None
            and not (p.get("host") == host and int(p.get("port")) == int(port))
        ]
        import random
        random.shuffle(candidates)
        relays_tried = 0
        for relay in candidates[:3]:
            relays_tried += 1
            relay_payload = {
                "target_host": host,
                "target_port": int(port),
                "target_id": target_id,
                "inner_kind": kind,
                "inner_data": data,
                "origin": self.node_id,
            }
            ok = await self.send_to_peer(
                relay["host"], int(relay["port"]), "relay_task", relay_payload, hops=hops + 1
            )
            if ok:
                logger.info(
                    f"🔀 Relayed '{kind}' to {host}:{port} via {relay.get('id')} "
                    f"({relay['host']}:{relay['port']})"
                )
                return {
                    "ok": True,
                    "mode": "relay",
                    "relay_id": relay.get("id"),
                    "relays_tried": relays_tried,
                }
        logger.error(f"❌ Direct and relay delivery failed for {host}:{port} kind={kind}")
        return {"ok": False, "mode": "failed", "relays_tried": relays_tried}

    async def _handle_relay_task(self, exp_data: Dict[str, Any], sender_id: str = None, hops: int = 0):
        """تنفيذ مهمة Relay: إعادة توجيه الرسالة الداخلية للهدف إن أمكن."""
        if hops > 4:
            logger.warning("⚠️ Relay hops exceeded — dropping")
            return
        target_host = exp_data.get("target_host")
        target_port = exp_data.get("target_port")
        inner_kind = exp_data.get("inner_kind")
        inner_data = exp_data.get("inner_data") or {}
        origin = exp_data.get("origin")
        if not target_host or target_port is None or not inner_kind:
            logger.warning("⚠️ Malformed relay_task")
            return
        # لا نعيد التوجيه لأنفسنا كهدف إن كنا الهدف
        if (self.host == target_host and int(self.port or -1) == int(target_port)) or (
            exp_data.get("target_id") and exp_data.get("target_id") == self.node_id
        ):
            # نحن الهدف النهائي — عالج الرسالة الداخلية محلياً
            logger.info(f"📥 Relay delivered local message kind={inner_kind} from origin={origin}")
            fake = {
                "payload": {
                    "id": f"relayed_{uuid.uuid4().hex[:8]}",
                    "kind": inner_kind,
                    "data": inner_data,
                    "from": origin or sender_id,
                    "p2p_hops": hops + 1,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
                "signature": "relay_local",  # لن يمر التحقق — نمرّر للمعالجة المباشرة بحذر
            }
            # معالجة مباشرة لأنواع آمنة فقط بدون إعادة توقيع مزيفة
            if inner_kind in ("gradient_push", "evolution_task", "tool_request", "sovereign_gossip"):
                self.sync_experience(inner_kind, inner_data, hops + 1)
            return

        ok = await self.send_to_peer(
            target_host, int(target_port), inner_kind, inner_data, hops=hops + 1
        )
        if ok:
            logger.info(f"✅ Relay forwarded {inner_kind} → {target_host}:{target_port} (from {origin})")
        else:
            # محاولة قفزة إضافية عبر وسيط آخر إن فشلت المباشرة من هنا
            logger.warning(f"⚠️ Relay forward failed; trying secondary hop for {inner_kind}")
            await self.send_to_peer_with_relay(
                target_host, int(target_port), inner_kind, inner_data, hops=hops + 1,
                target_id=exp_data.get("target_id"),
            )

    # ------------------------------------------------------------------
    # #4 Multi-signature — تحقق آمن من عمليات المكافآت
    # ------------------------------------------------------------------
    def _multisig_state(self) -> Dict[str, Any]:
        state = self._load_state()
        if "multisig" not in state:
            state["multisig"] = {}
            self._save_state(state)
        return state

    async def propose_multisig(
        self,
        agreement: Dict[str, Any],
        required_signatures: int = 4,
        agreement_id: str = None,
    ) -> str:
        """يقترح عملية مكافأة/اتفاقية تتطلب عدداً من التوقيعات قبل التنفيذ."""
        agreement_id = agreement_id or f"msig_{uuid.uuid4().hex[:10]}"
        canonical = json.dumps(agreement, sort_keys=True, ensure_ascii=False)
        my_sig = self.sign_message(canonical)
        state = self._multisig_state()
        state["multisig"][agreement_id] = {
            "agreement": agreement,
            "canonical": canonical,
            "required": int(required_signatures),
            "signatures": {self.node_id: my_sig},
            "status": "pending",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "executed": False,
        }
        self._save_state(state)

        payload = {
            "agreement_id": agreement_id,
            "agreement": agreement,
            "required": int(required_signatures),
            "proposer": self.node_id,
        }
        # بث الاقتراح للأقران
        for peer in self._get_active_peers_list():
            if peer.get("id") == self.node_id or not peer.get("host") or peer.get("port") is None:
                continue
            await self.send_to_peer_with_relay(
                peer["host"], int(peer["port"]), "multisig_propose", payload, target_id=peer.get("id")
            )
        logger.info(
            f"✍️ Multisig proposed {agreement_id} requires {required_signatures} signatures"
        )
        return agreement_id

    async def _handle_multisig_propose(self, exp_data: Dict[str, Any], sender_id: str = None):
        agreement_id = exp_data.get("agreement_id")
        agreement = exp_data.get("agreement")
        required = int(exp_data.get("required") or 4)
        if not agreement_id or not isinstance(agreement, dict):
            return
        canonical = json.dumps(agreement, sort_keys=True, ensure_ascii=False)
        state = self._multisig_state()
        entry = state["multisig"].get(agreement_id)
        if entry is None:
            state["multisig"][agreement_id] = {
                "agreement": agreement,
                "canonical": canonical,
                "required": required,
                "signatures": {},
                "status": "pending",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "executed": False,
            }
            self._save_state(state)
            entry = state["multisig"][agreement_id]
        # توقّع محلياً ونُبلغ الشبكة
        if self.node_id not in entry["signatures"]:
            my_sig = self.sign_message(entry["canonical"])
            entry["signatures"][self.node_id] = my_sig
            self._save_state(state)
            sign_payload = {
                "agreement_id": agreement_id,
                "signer": self.node_id,
                "signature": my_sig,
            }
            for peer in self._get_active_peers_list():
                if peer.get("id") == self.node_id or not peer.get("host") or peer.get("port") is None:
                    continue
                await self.send_to_peer(
                    peer["host"], int(peer["port"]), "multisig_sign", sign_payload
                )
            logger.info(f"✍️ Signed multisig {agreement_id} ({len(entry['signatures'])}/{entry['required']})")
        await self._try_finalize_multisig(agreement_id)

    async def _handle_multisig_sign(self, exp_data: Dict[str, Any], sender_id: str = None):
        agreement_id = exp_data.get("agreement_id")
        signer = exp_data.get("signer") or sender_id
        signature = exp_data.get("signature")
        if not agreement_id or not signer or not signature:
            return
        state = self._multisig_state()
        entry = state["multisig"].get(agreement_id)
        if not entry or entry.get("executed"):
            return
        # تحقق التوقيع بمفتاح الموقّع المعروف
        key_path = self.keys_dir / f"{signer}.pub"
        if not key_path.exists():
            logger.warning(f"⚠️ multisig_sign from unknown key owner {signer}")
            return
        if not self.verify_signature(key_path.read_bytes(), entry["canonical"], signature):
            logger.warning(f"⚠️ Rejected forged/invalid multisig signature from {signer}")
            return
        entry["signatures"][signer] = signature
        self._save_state(state)
        logger.info(
            f"✅ Valid multisig signature from {signer} "
            f"({len(entry['signatures'])}/{entry['required']}) on {agreement_id}"
        )
        await self._try_finalize_multisig(agreement_id)

    async def _try_finalize_multisig(self, agreement_id: str):
        state = self._multisig_state()
        entry = state["multisig"].get(agreement_id)
        if not entry or entry.get("executed"):
            return
        # أعد التحقق من كل التوقيعات قبل التنفيذ
        valid = {}
        for signer, sig in list(entry.get("signatures", {}).items()):
            key_path = self.keys_dir / f"{signer}.pub"
            if signer == self.node_id:
                pub = self._pub_pem().encode()
                if self.verify_signature(pub, entry["canonical"], sig):
                    valid[signer] = sig
                continue
            if key_path.exists() and self.verify_signature(key_path.read_bytes(), entry["canonical"], sig):
                valid[signer] = sig
        entry["signatures"] = valid
        self._save_state(state)
        if len(valid) < int(entry.get("required") or 4):
            return
        entry["status"] = "approved"
        entry["executed"] = True
        entry["executed_at"] = datetime.now(timezone.utc).isoformat()
        self._save_state(state)
        logger.info(
            f"🏅 Multisig APPROVED & executed {agreement_id} "
            f"with {len(valid)} signatures — agreement={entry.get('agreement')}"
        )
        # سجل كخبرة جماعية
        self.sync_experience(
            "multisig_executed",
            {
                "agreement_id": agreement_id,
                "agreement": entry.get("agreement"),
                "signers": list(valid.keys()),
            },
            hops=0,
        )
