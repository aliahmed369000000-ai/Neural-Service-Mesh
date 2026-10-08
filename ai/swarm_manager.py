"""
ai/swarm_manager.py
===================
بروتوكول التوافق (Swarm Consensus) وتنسيق السرب السيادي.
"""
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path
from ai.living_mesh import LivingMeshNode

logger = logging.getLogger("NeuralServiceMesh.SwarmManager")

class SwarmProposal:
    def __init__(self, proposal_id: str, proposer: str, action_type: str, data: Dict[str, Any]):
        self.proposal_id = proposal_id
        self.proposer = proposer
        self.action_type = action_type
        self.data = data
        self.votes = {}  # {agent_id: {"vote": bool, "reason": str, "weight": float}}
        self.status = "pending"  # pending, approved, rejected, expired
        self.created_at = time.time()

class SwarmManager:
    def __init__(self, storage_dir: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        self.storage_dir = Path(storage_dir) if storage_dir else self.root / "artifacts" / "swarm"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.proposals: Dict[str, SwarmProposal] = {}
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.results: List[Dict[str, Any]] = []
        
        # لا تُنشأ عقدة الشبكة ولا تُنفذ join_network أثناء الإنشاء.
        # ستُهيأ عند أول عملية تحتاج حالة الشبكة فقط.
        self._mesh_node: Optional[LivingMeshNode] = None
        
        self.marketplace_tasks: Dict[str, Dict[str, Any]] = {}
        self.competitions: Dict[str, Dict[str, Any]] = {}
        
        self.threshold = 0.66
        self.proposal_timeout = 60
        self.heartbeat_timeout = 20
        
        self.roles_config = {
            "sovereign": {"permissions": ["read", "write", "delete", "spawn", "reflect", "admin"], "trust_min": 0.95},
            "orchestrator": {"permissions": ["read", "write", "delete", "spawn", "reflect"], "trust_min": 0.8},
            "worker": {"permissions": ["read", "write", "spawn"], "trust_min": 0.5},
            "auditor": {"permissions": ["read", "reflect"], "trust_min": 0.7},
            "observer": {"permissions": ["read"], "trust_min": 0.0}
        }

        # حالة النوم (set_agent_sleep/awake) — خفيفة، تُنشأ مباشرة.
        self.sleeping_agents: Dict[str, Dict[str, Any]] = {}
        # IDS / الذاكرة متعددة الوسائط / محرك الوعي العاطفي: تُنشأ عند أول
        # استخدام فقط (انظر الخصائص أدناه) كي يبقى إنشاء SwarmManager بلا
        # آثار جانبية، وهو هدف إعادة الكتابة الخفيفة السابقة.
        self._ids = None
        self._memory = None
        self._emotional_engine = None

    @property
    def mesh_node(self) -> LivingMeshNode:
        """تهيئة LivingMeshNode عند الحاجة فقط لتجنب آثار الشبكة الجانبية."""
        if self._mesh_node is None:
            self._mesh_node = LivingMeshNode()
            self._mesh_node.join_network()
        return self._mesh_node

    @property
    def ids(self):
        """نظام مراقبة واكتشاف التسلل (IDS) — كسول."""
        if self._ids is None:
            from ai.ids_manager import IDSManager
            self._ids = IDSManager(storage_dir=str(self.storage_dir / "ids"))
        return self._ids

    @ids.setter
    def ids(self, value):
        self._ids = value

    @property
    def memory(self):
        """الذاكرة متعددة الوسائط المشتركة — كسولة."""
        if self._memory is None:
            from ai.multimodal_memory import mm_memory
            self._memory = mm_memory
        return self._memory

    @memory.setter
    def memory(self, value):
        self._memory = value

    @property
    def emotional_engine(self):
        """محرك الوعي العاطفي الجماعي — كسول."""
        if self._emotional_engine is None:
            from ai.emotional_awareness import emotional_engine
            self._emotional_engine = emotional_engine
        return self._emotional_engine

    @emotional_engine.setter
    def emotional_engine(self, value):
        self._emotional_engine = value

    def check_permission(self, agent_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        # فحص IDS أولاً (أُعيد: حُذف بإعادة كتابة خفيفة سابقة فصار أي أمر
        # تخريبي مثل "rm -rf /" يمرّ لأي وكيل يملك صلاحية الكتابة، ولا يُحجَر
        # أحد). self.ids كسول فلا يكلّف شيئاً قبل أول فحص.
        if self.ids.is_quarantined(agent_id):
            logger.error(f"🚨 Access Denied: Agent {agent_id} is in QUARANTINE.")
            return False
        ids_res = self.ids.monitor_action(agent_id, action, params or {})
        if ids_res.get("status") == "blocked":
            return False

        if agent_id not in self.workers:
            role = "observer"
            trust_score = 0.0
        else:
            worker = self.workers[agent_id]
            role = worker.get("role", "observer")
            trust_score = worker.get("trust_score", 0.0)

        role_info = self.roles_config.get(role, self.roles_config["observer"])
        if action not in role_info["permissions"]:
            logger.warning(f"🚫 Permission Denied: Agent {agent_id} ({role}) tried to {action}")
            return False
        if trust_score < role_info["trust_min"]:
            logger.warning(f"⚠️ Trust Constraint: Agent {agent_id} trust {trust_score} below required {role_info['trust_min']}")
            return False
        return True

    def register_worker(self, agent_id: str, role: str, trust_score: float = 0.5):
        self.workers[agent_id] = {
            "role": role,
            "status": "active",
            "trust_score": trust_score,
            "last_seen": time.time(),
            "joined_at": time.time()
        }
        logger.info(f"👷 Worker Registered: {agent_id} as {role}")

    def heartbeat(self, agent_id: str):
        if agent_id in self.workers:
            self.workers[agent_id]["last_seen"] = time.time()
            self.workers[agent_id]["status"] = "active"

    def get_active_workers(self) -> List[str]:
        now = time.time()
        active = []
        for aid, info in self.workers.items():
            if now - info["last_seen"] <= self.heartbeat_timeout:
                active.append(aid)
            elif info["status"] == "active":
                # رصد فشل العقدة محلياً وتحديث الحالة
                info["status"] = "offline"
                logger.warning(f"⚠️ Node {aid} timed out and marked as offline.")
                self.trigger_self_healing(aid)
        return active

    def trigger_self_healing(self, dead_agent_id: str):
        """بروتوكول التعافي الذاتي: سحب المهام وإعادة توزيعها."""
        logger.info(f"🚑 Self-Healing Protocol triggered for Agent {dead_agent_id}")
        
        # 1. سحب المهام الموكلة للعقدة المتعطلة
        tasks_to_reassign = []
        for tid, task in self.marketplace_tasks.items():
            if task.get("assigned_to") == dead_agent_id and task["status"] == "assigned":
                tasks_to_reassign.append(tid)
        
        # 2. إعادة طرح المهام في السوق فوراً
        for tid in tasks_to_reassign:
            task = self.marketplace_tasks[tid]
            task["status"] = "open"
            task["assigned_to"] = None
            task["bids"] = [] # تصفير المزايدات القديمة
            logger.info(f"🔄 Task {tid} reassigned to marketplace due to node failure.")
            
        # 3. إخطار الشبكة اللامركزية (Living Mesh)
        self.mesh_node.check_network_health(timeout_seconds=self.heartbeat_timeout)

    def _update_trust(self, agent_id: str, delta: float):
        """تعديل مستوى الثقة لوكيل معين مع ضمان البقاء في النطاق [0.0, 1.0]."""
        if agent_id in self.workers:
            old_score = self.workers[agent_id]["trust_score"]
            new_score = max(0.0, min(1.0, old_score + delta))
            self.workers[agent_id]["trust_score"] = new_score
            if abs(delta) > 0.01:
                logger.info(f"⚖️ Trust Update for {agent_id}: {old_score:.2f} -> {new_score:.2f}")
            
            # التحديث الفوري للمزاج عند حدوث تغيير حاد في الثقة
            if abs(delta) >= 0.1:
                try:
                    self._sync_emotional_state()
                except Exception as e:
                    logger.warning(f"Emotional sync failed after trust update: {e}")


    # ── دوال أُعيدت: حُذفت بإعادة كتابة خفيفة سابقة (كوميت 9f7d6cc) بينما ما
    # ── زال ai/agent_loop.py يستدعيها فعلياً في الإنتاج.

    def set_agent_sleep(self, agent_id: str, snapshot_path: str):
        """تحديث حالة الوكيل إلى 'نائم' وتخزين مسار لقطة الوعي."""
        if agent_id in self.workers:
            self.workers[agent_id]["status"] = "sleeping"
            self.sleeping_agents[agent_id] = {
                "snapshot_path": snapshot_path,
                "sleep_time": time.time()
            }
            logger.info(f"😴 Agent {agent_id} is now in deep sleep.")

    def set_agent_awake(self, agent_id: str):
        """تحديث حالة الوكيل إلى 'نشط' عند الاستيقاظ."""
        if agent_id in self.workers:
            self.workers[agent_id]["status"] = "active"
            self.workers[agent_id]["last_seen"] = time.time()
            if agent_id in self.sleeping_agents:
                del self.sleeping_agents[agent_id]
            logger.info(f"🌅 Agent {agent_id} has awakened.")

    def share_media(self, agent_id: str, file_path: str, media_type: str, description: str, tags: List[str]) -> str:
        """مشاركة أصل وسائط مع السرب مع التحقق من الصلاحيات."""
        if not self.check_permission(agent_id, "write"):
            raise PermissionError(f"الوكيل {agent_id} لا يملك صلاحية الكتابة في الذاكرة الجماعية.")
            
        metadata = {"description": description, "tags": tags}
        asset_id = self.memory.store_asset(agent_id, file_path, media_type, metadata)
        logger.info(f"📸 Media Shared by {agent_id}: {asset_id} ({media_type})")
        return asset_id

    def create_proposal(self, proposer: str, action_type: str, data: Dict[str, Any]) -> str:
        """إنشاء مقترح جديد للمراجعة الجماعية."""
        p_id = f"prop_{uuid.uuid4().hex[:8]}"
        proposal = SwarmProposal(p_id, proposer, action_type, data)
        self.proposals[p_id] = proposal
        self._save_proposal(proposal)
        logger.info(f"🆕 Swarm Proposal Created: {p_id} by {proposer}")
        return p_id

    def cast_vote(self, proposal_id: str, agent_id: str, vote: bool, reason: str, weight: float = 1.0) -> Dict[str, Any]:
        """تسجيل صوت وكيل على مقترح معين."""
        if proposal_id not in self.proposals:
            return {"ok": False, "error": "المقترح غير موجود"}
        
        proposal = self.proposals[proposal_id]
        if proposal.status != "pending":
            return {"ok": False, "error": "المقترح مغلق بالفعل"}
            
        proposal.votes[agent_id] = {
            "vote": vote,
            "reason": reason,
            "weight": weight,
            "ts": time.time()
        }
        
        self._save_proposal(proposal)
        
        # التحقق من الوصول للتوافق بعد كل صوت
        consensus = self.check_consensus(proposal_id)
        return {"ok": True, "consensus": consensus}

    def check_consensus(self, proposal_id: str) -> Dict[str, Any]:
        """تحليل الأصوات الحالية مع مراعاة المهلة والوكلاء النشطين."""
        proposal = self.proposals.get(proposal_id)
        if not proposal: return {"status": "not_found"}
        
        if proposal.status != "pending":
            return {"status": proposal.status, "score": 0}

        now = time.time()
        is_timeout = (now - proposal.created_at) > self.proposal_timeout
        active_agents = self.get_active_workers()
        
        # تصفية الأصوات لتشمل الوكلاء النشطين فقط
        valid_votes = {aid: v for aid, v in proposal.votes.items() if aid in active_agents}
        
        total_weight = sum(v["weight"] for v in valid_votes.values())
        if total_weight == 0:
            if is_timeout:
                proposal.status = "expired"
                self._save_proposal(proposal)
                return {"status": "expired", "score": 0}
            return {"status": "pending", "score": 0}
            
        yes_weight = sum(v["weight"] for v in valid_votes.values() if v["vote"])
        score = yes_weight / total_weight
        
        # التوافق المرن: خفض العتبة قليلاً عند حدوث Timeout لضمان الاستمرارية
        current_threshold = self.threshold if not is_timeout else 0.51
        
        if score >= current_threshold:
            proposal.status = "approved"
            self._save_proposal(proposal)
            return {"status": "approved", "score": score, "adaptive": is_timeout}
        elif is_timeout or (len(valid_votes) >= 3 and score < 0.3):
            proposal.status = "rejected"
            self._save_proposal(proposal)
            return {"status": "rejected", "score": score}
            
        return {"status": "pending", "score": score}

    def _save_proposal(self, proposal: SwarmProposal):
        """حفظ حالة المقترح مشفرة في القرص للمراجعة والتدقيق."""
        from ai.security_manager import security_manager
        
        # تحديث الثقة عند الموافقة أو الرفض
        if proposal.status == "approved":
            self._update_trust(proposal.proposer, 0.1)
        elif proposal.status == "rejected":
            self._update_trust(proposal.proposer, -0.05)

        p_path = self.storage_dir / f"{proposal.proposal_id}.enc"
        data = {
            "id": proposal.proposal_id,
            "proposer": proposal.proposer,
            "type": proposal.action_type,
            "data": proposal.data,
            "votes": proposal.votes,
            "status": proposal.status,
            "created_at": proposal.created_at
        }
        
        # تشفير بيانات المقترح قبل الحفظ
        encrypted_data = security_manager.encrypt(data)
        with open(p_path, "wb") as f:
            f.write(encrypted_data)

    def report_result(self, agent_id: str, task: str, result: str, success: bool = True):
        """تسجيل نتيجة مهمة من وكيل فرعي مع تحديث الثقة تلقائياً."""
        entry = {
            "agent_id": agent_id,
            "task": task,
            "result": result,
            "success": success,
            "ts": time.time()
        }
        self.results.append(entry)
        
        # 🆕 تحديث الثقة: مكافأة أو جزاء
        self._update_trust(agent_id, 0.05 if success else -0.1)
        
        # حفظ النتيجة في ملف سجل السرب
        res_path = self.storage_dir / "swarm_results.jsonl"
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"✅ Result Reported by {agent_id} (Success: {success})")

    def trigger_reflection(self) -> Dict[str, Any]:
        """بدء عملية التلخيص الذاتي بناءً على الأنشطة الأخيرة."""
        from ai.self_reflection import reflection_engine
        # نأخذ آخر 20 نتيجة لمراجعتها
        recent_activity = self.results[-20:]
        result = reflection_engine.reflect_on_activity(recent_activity)
        logger.info(f"🧠 Self-Reflection Triggered: {result.get('summary')}")
        return result

    def _sync_emotional_state(self):
        """مزامنة الحالة العاطفية للمحرك مع بيانات السرب الحالية."""
        swarm_data = {
            "results": self.results,
            "workers": self.workers,
            "alert_level": self.ids.get_threat_level() if hasattr(self.ids, "get_threat_level") else 0.0
        }
        self.emotional_engine.update_mood(swarm_data)
        
        # تطبيق البارامترات المتكيفة
        params = self.emotional_engine.get_adaptive_params()
        self.threshold = params["consensus_threshold"]
        self.heartbeat_timeout = params["heartbeat_interval"]

    # 🛒 Marketplace APIs
    def post_marketplace_task(self, **kwargs):
        """
        طرح مهمة في السوق. يدعم وسائط مرنة.
        """
        task_id = kwargs.get("task_id") or f"task_{uuid.uuid4().hex[:6]}"
        title = kwargs.get("title") or kwargs.get("task_name") or task_id
        description = kwargs.get("description") or kwargs.get("desc") or ""
        reward = kwargs.get("reward") or 0
        
        task = {
            "id": task_id,
            "name": title,
            "description": description,
            "reward": reward,
            "status": "open",
            "bids": [],
            "created_at": time.time()
        }
        self.marketplace_tasks[task_id] = task
        logger.info(f"🛒 New Marketplace Task: {task_id} - {title}")
        return task_id

    def submit_bid(self, task_id: str, agent_id: str, cost: int = 0, time_est: float = 0.0, trust: float = 0.0):
        if task_id not in self.marketplace_tasks: return False
        bid = {
            "agent_id": agent_id,
            "cost": cost,
            "time": time_est,
            "trust_claim": trust,
            "ts": time.time()
        }
        self.marketplace_tasks[task_id]["bids"].append(bid)
        logger.info(f"🙋 Agent {agent_id} bid for task {task_id}")
        return True

    def bid_for_task(self, agent_id: str, task_id: str, proposal: str):
        """توافق مع النسخة السابقة."""
        return self.submit_bid(task_id, agent_id, trust=0.5)

    def assign_task(self, orchestrator: str, task_id: str, agent_id: str):
        if task_id in self.marketplace_tasks:
            self.marketplace_tasks[task_id]["status"] = "assigned"
            self.marketplace_tasks[task_id]["assigned_to"] = agent_id
            return True
        return False

    def award_task(self, task_id: str) -> Optional[str]:
        """تخصيص المهام بناءً على الأوزان التطورية والسيادة."""
        if task_id not in self.marketplace_tasks or not self.marketplace_tasks[task_id]["bids"]:
            return None
            
        task = self.marketplace_tasks[task_id]
        mesh_state = self.mesh_node._load_state()
        
        best_agent = None
        highest_score = -1.0
        
        for bid in task["bids"]:
            aid = bid["agent_id"]
            # جلب الأوزان التطورية للعقدة من الشبكة
            node_info = mesh_state["nodes"].get(aid, {})
            weights = node_info.get("behavioral_weights", {})
            
            # حساب نتيجة الجدارة (Evolutionary Merit Score)
            # تعتمد على الكفاءة، الابتكار، والثقة المعلنة
            efficiency = weights.get("processing_efficiency", 1.0)
            innovation = weights.get("innovation_rate", 1.0)
            trust = bid.get("trust_claim", 0.5)
            
            merit_score = (efficiency * 0.4) + (innovation * 0.3) + (trust * 0.3)
            
            if merit_score > highest_score:
                highest_score = merit_score
                best_agent = aid
        
        if best_agent:
            self.assign_task("system", task_id, best_agent)
            logger.info(f"🏆 Task {task_id} awarded to {best_agent} with Merit Score: {highest_score:.2f}")
            return best_agent
        return None

    # ⚔️ Competition APIs
    def start_competition(self, comp_id: str, task_description: str, competitors: List[str]):
        self.competitions[comp_id] = {
            "task": task_description,
            "competitors": competitors,
            "solutions": {},
            "status": "active",
            "created_at": time.time()
        }
        logger.info(f"⚔️ Competition Started: {comp_id}")
        return True

    def submit_solution(self, comp_id: str, agent_id: str, solution_data: Any):
        if comp_id not in self.competitions: return False
        self.competitions[comp_id]["solutions"][agent_id] = {
            "data": solution_data,
            "ts": time.time()
        }
        return True

    def judge_competition(self, comp_id: str, judge_id: str) -> Optional[str]:
        comp = self.competitions.get(comp_id)
        if not comp or not comp["solutions"]: return None
        # الأسرع هو الفائز حالياً
        winner = min(comp["solutions"].items(), key=lambda x: x[1]["ts"])[0]
        comp["status"] = "finished"
        comp["winner"] = winner
        self._update_trust(winner, 0.1)
        return winner

    def finalize_competition(self, comp_id: str):
        """توافق مع الاختبارات."""
        winner_id = self.judge_competition(comp_id, "system")
        if winner_id:
            comp = self.competitions[comp_id]
            return {"agent_id": winner_id, "score": 1.0, "data": comp["solutions"][winner_id]["data"]}
        return None

    def get_swarm_status(self):
        return {"active": len(self.get_active_workers()), "tasks": len(self.marketplace_tasks)}

swarm_manager = SwarmManager()
