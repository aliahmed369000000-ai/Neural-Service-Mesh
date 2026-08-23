"""
ai/swarm_manager.py
===================
بروتوكول التوافق (Swarm Consensus) وتنسيق السرب السيادي.

يدير هذا الملف عملية اتخاذ القرار الجماعي بين الوكلاء، مما يضمن أن الإجراءات 
الحرجة (مثل إضافة أدوات جديدة) تخضع للمراجعة والتوافق بناءً على أوزان الثقة.
"""
import json
import logging
import time
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("NeuralServiceMesh.SwarmManager")

class SwarmProposal:
    def __init__(self, proposal_id: str, proposer: str, action_type: str, data: Dict[str, Any]):
        self.proposal_id = proposal_id
        self.proposer = proposer
        self.action_type = action_type
        self.data = data
        self.votes = {}  # {agent_id: {"vote": bool, "reason": str, "weight": float}}
        self.status = "pending"  # pending, approved, rejected
        self.created_at = time.time()

class SwarmManager:
    def __init__(self, storage_dir: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        self.storage_dir = Path(storage_dir) if storage_dir else self.root / "artifacts" / "swarm"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.proposals: Dict[str, SwarmProposal] = {}
        self.workers: Dict[str, Dict[str, Any]] = {}
        self.results: List[Dict[str, Any]] = []
        
        # 🆕 سوق المهام والمنافسة
        self.marketplace_tasks: Dict[str, Dict[str, Any]] = {}
        self.bids: Dict[str, List[Dict[str, Any]]] = {}
        self.competitions: Dict[str, Dict[str, Any]] = {}
        
        self.threshold = 0.66  # عتبة التوافق (66%)
        self.proposal_timeout = 60  # مهلة المقترح بالثواني
        self.heartbeat_timeout = 20  # مهلة نبض القلب بالثواني
        
        # تعريف الأدوار والصلاحيات (RBAC)
        self.roles_config = {
            "orchestrator": {"permissions": ["read", "write", "delete", "spawn", "reflect"], "trust_min": 0.8},
            "worker": {"permissions": ["read", "write", "spawn"], "trust_min": 0.5},
            "auditor": {"permissions": ["read", "reflect"], "trust_min": 0.7},
            "observer": {"permissions": ["read"], "trust_min": 0.0}
        }
        
        # ربط الذاكرة متعددة الوسائط
        from ai.multimodal_memory import mm_memory
        self.memory = mm_memory
        
        # 🆕 إدارة حالة النوم
        self.sleeping_agents: Dict[str, Dict[str, Any]] = {}

        # 🆕 نظام مراقبة واكتشاف التسلل (IDS)
        from ai.ids_manager import IDSManager
        self.ids = IDSManager(storage_dir=str(self.storage_dir / "ids"))

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

    def check_permission(self, agent_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
        """التحقق من صلاحيات الوكيل بناءً على الدور ومستوى الثقة مع مراقبة IDS."""
        # 1. التحقق من الحجر الصحي (IDS)
        if self.ids.is_quarantined(agent_id):
            logger.error(f"🚨 Access Denied: Agent {agent_id} is in QUARANTINE.")
            return False

        # 2. مراقبة الفعل عبر IDS
        ids_res = self.ids.monitor_action(agent_id, action, params or {})
        if ids_res["status"] == "blocked":
            return False

        if agent_id not in self.workers:
            role = "observer"
            trust_score = 0.5
        else:
            worker = self.workers[agent_id]
            role = worker.get("role", "observer")
            trust_score = worker.get("trust_score", 0.5)

        role_info = self.roles_config.get(role, self.roles_config["observer"])
        
        if action not in role_info["permissions"]:
            logger.warning(f"🚫 Permission Denied: Agent {agent_id} ({role}) tried to {action}")
            return False
            
        if trust_score < role_info["trust_min"]:
            logger.warning(f"⚠️ Trust Constraint: Agent {agent_id} trust {trust_score} below required {role_info['trust_min']} for {role}")
            return False
            
        return True

    def share_media(self, agent_id: str, file_path: str, media_type: str, description: str, tags: List[str]) -> str:
        """مشاركة أصل وسائط مع السرب مع التحقق من الصلاحيات."""
        if not self.check_permission(agent_id, "write"):
            raise PermissionError(f"الوكيل {agent_id} لا يملك صلاحية الكتابة في الذاكرة الجماعية.")
            
        metadata = {"description": description, "tags": tags}
        asset_id = self.memory.store_asset(agent_id, file_path, media_type, metadata)
        logger.info(f"📸 Media Shared by {agent_id}: {asset_id} ({media_type})")
        return asset_id

    def register_worker(self, agent_id: str, role: str, trust_score: float = 0.5):
        """تسجيل وكيل جديد في السرب مع تهيئة نبض القلب والأذونات."""
        self.workers[agent_id] = {
            "role": role,
            "status": "active",
            "trust_score": trust_score,
            "last_seen": time.time(),
            "joined_at": time.time()
        }
        logger.info(f"👷 Worker Registered: {agent_id} as {role} (Trust: {trust_score})")

    def heartbeat(self, agent_id: str):
        """تحديث نبض القلب للوكيل لضمان أنه لا يزال نشطاً، مع تعافي تدريجي للثقة."""
        if agent_id in self.workers:
            worker = self.workers[agent_id]
            worker["last_seen"] = time.time()
            worker["status"] = "active"
            
            # 🆕 تعافي تدريجي للثقة (Passive Recovery)
            # إذا كانت الثقة منخفضة، تزيد بنسبة ضئيلة جداً مع كل نبضة قلب (دليل استقرار)
            if worker["trust_score"] < 0.5:
                worker["trust_score"] = min(0.5, worker["trust_score"] + 0.001)

    def get_active_workers(self) -> List[str]:
        """الحصول على قائمة الوكلاء الذين أرسلوا نبضات قلب مؤخراً."""
        now = time.time()
        active = []
        for aid, info in self.workers.items():
            if now - info["last_seen"] <= self.heartbeat_timeout:
                active.append(aid)
            else:
                if info["status"] == "active":
                    info["status"] = "offline"
                    logger.warning(f"⚠️ Worker {aid} went offline (Timeout)")
        return active

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

    def get_swarm_status(self) -> Dict[str, Any]:
        """الحصول على نظرة عامة على حالة السرب الحالي."""
        return {
            "active_workers": len(self.workers),
            "completed_tasks": len(self.results),
            "recent_results": self.results[-5:] if self.results else []
        }

    def trigger_reflection(self) -> Dict[str, Any]:
        """بدء عملية التلخيص الذاتي بناءً على الأنشطة الأخيرة."""
        from ai.self_reflection import reflection_engine
        # نأخذ آخر 20 نتيجة لمراجعتها
        recent_activity = self.results[-20:]
        result = reflection_engine.reflect_on_activity(recent_activity)
        logger.info(f"🧠 Self-Reflection Triggered: {result.get('summary')}")
        return result

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

    def _update_trust(self, agent_id: str, delta: float):
        """تعديل مستوى الثقة لوكيل معين مع ضمان البقاء في النطاق [0.0, 1.0]."""
        if agent_id in self.workers:
            old_score = self.workers[agent_id]["trust_score"]
            new_score = max(0.0, min(1.0, old_score + delta))
            self.workers[agent_id]["trust_score"] = new_score
            if abs(delta) > 0.01:
                logger.info(f"⚖️ Trust Update for {agent_id}: {old_score:.2f} -> {new_score:.2f}")

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

    # 🆕 منطق سوق المهام والمزايدة
    def post_marketplace_task(self, task_id: str, description: str, requirements: Dict[str, Any]):
        """طرح مهمة في سوق السرب للمزايدة عليها."""
        self.marketplace_tasks[task_id] = {
            "description": description,
            "requirements": requirements,
            "status": "open",
            "created_at": time.time()
        }
        self.bids[task_id] = []
        logger.info(f"🛒 Task Posted in Marketplace: {task_id}")

    def submit_bid(self, task_id: str, agent_id: str, cost_estimate: int, time_estimate: float, trust_claim: float):
        """تقديم عرض (Bid) من وكيل لمهمة معينة."""
        if task_id not in self.marketplace_tasks or self.marketplace_tasks[task_id]["status"] != "open":
            return False
        
        bid = {
            "agent_id": agent_id,
            "cost": cost_estimate,
            "time": time_estimate,
            "trust": trust_claim,
            "ts": time.time()
        }
        self.bids[task_id].append(bid)
        logger.info(f"💰 Bid Submitted for {task_id} by {agent_id}")
        return True

    def award_task(self, task_id: str) -> Optional[str]:
        """اختيار العرض الأفضل وإرساء المهمة على الوكيل الأنسب."""
        if task_id not in self.bids or not self.bids[task_id]:
            return None
        
        # خوارزمية الاختيار: توازن بين التكلفة، الوقت، وثقة الوكيل الفعلية
        best_bid = None
        best_score = -1.0
        
        for bid in self.bids[task_id]:
            agent_id = bid["agent_id"]
            actual_trust = self.workers.get(agent_id, {}).get("trust_score", 0.5)
            
            # معادلة التقييم (Score): Trust / (Cost * Time)
            score = actual_trust / (max(1, bid["cost"]) * max(0.1, bid["time"]))
            
            if score > best_score:
                best_score = score
                best_bid = bid
        
        if best_bid:
            winner = best_bid["agent_id"]
            self.marketplace_tasks[task_id]["status"] = "awarded"
            self.marketplace_tasks[task_id]["winner"] = winner
            logger.info(f"🏆 Task {task_id} awarded to {winner} (Score: {best_score:.2f})")
            return winner
        return None

    # 🆕 منطق ساحة المنافسة (Competition Arena)
    def start_competition(self, comp_id: str, task_desc: str, competitors: List[str]):
        """بدء منافسة بين عدة وكلاء لحل نفس المهمة بالتوازي."""
        self.competitions[comp_id] = {
            "task": task_desc,
            "competitors": competitors,
            "solutions": {},
            "status": "active",
            "created_at": time.time()
        }
        logger.info(f"⚔️ Competition Started: {comp_id} between {competitors}")

    def submit_solution(self, comp_id: str, agent_id: str, solution_data: Any):
        """تقديم حل لمنافسة جارية."""
        if comp_id not in self.competitions or agent_id not in self.competitions[comp_id]["competitors"]:
            return False
        
        self.competitions[comp_id]["solutions"][agent_id] = {
            "data": solution_data,
            "ts": time.time()
        }
        logger.info(f"📝 Solution submitted for {comp_id} by {agent_id}")
        return True

    def judge_competition(self, comp_id: str, judge_id: str) -> Optional[str]:
        """تقييم الحلول وإعلان الفائز في المنافسة."""
        comp = self.competitions.get(comp_id)
        if not comp or not comp["solutions"]:
            return None
        
        # في النسخة الحالية، القاضي (Auditor) هو من يقرر، أو نستخدم خوارزمية مقارنة
        # سنختار الحل الأسرع كمعيار بسيط حالياً
        winner = None
        earliest_ts = float('inf')
        
        for aid, sol in comp["solutions"].items():
            if sol["ts"] < earliest_ts:
                earliest_ts = sol["ts"]
                winner = aid
        
        if winner:
            comp["status"] = "finished"
            comp["winner"] = winner
            # مكافأة ضخمة للفائز
            self._update_trust(winner, 0.15)
            logger.info(f"🎖️ Agent {winner} won the competition {comp_id}!")
            return winner
        return None

swarm_manager = SwarmManager()
