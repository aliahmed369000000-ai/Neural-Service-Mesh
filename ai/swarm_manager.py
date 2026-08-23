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
        
        self.mesh_node = LivingMeshNode()
        self.mesh_node.join_network()
        
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

    def check_permission(self, agent_id: str, action: str, params: Optional[Dict[str, Any]] = None) -> bool:
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
        return active

    def _update_trust(self, agent_id: str, delta: float):
        if agent_id in self.workers:
            old = self.workers[agent_id]["trust_score"]
            self.workers[agent_id]["trust_score"] = max(0.0, min(1.0, old + delta))
            logger.info(f"⚖️ Trust Update for {agent_id}: {old:.2f} -> {self.workers[agent_id]['trust_score']:.2f}")

    # 🛒 Marketplace APIs
    def post_marketplace_task(self, task_id_or_proposer: str, description: str = "", requirements: Any = None):
        """
        طرح مهمة في السوق.
        ملاحظة: agent_loop يمرر (task_id, desc, reqs).
        بينما الكود القديم قد يمرر (proposer, task_name, desc).
        سندعم كلا التوقيعين عبر التحقق من النوع.
        """
        if isinstance(requirements, float): # التوقيع القديم: (proposer, name, desc, reward)
            proposer = task_id_or_proposer
            task_name = description
            desc = str(requirements)
            task_id = f"task_{uuid.uuid4().hex[:6]}"
        else: # التوقيع الجديد: (task_id, description, requirements)
            task_id = task_id_or_proposer
            desc = description
            task_name = task_id
            
        task = {
            "id": task_id,
            "name": task_name,
            "description": desc,
            "status": "open",
            "bids": [],
            "created_at": time.time()
        }
        self.marketplace_tasks[task_id] = task
        logger.info(f"🛒 New Marketplace Task: {task_id}")
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
        if task_id not in self.marketplace_tasks or not self.marketplace_tasks[task_id]["bids"]:
            return None
        # اختيار صاحب أعلى ثقة معلنة
        winner = max(self.marketplace_tasks[task_id]["bids"], key=lambda x: x["trust_claim"])["agent_id"]
        self.assign_task("system", task_id, winner)
        return winner

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
