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
        self.threshold = 0.66  # عتبة التوافق (66%)
        self.proposal_timeout = 60  # مهلة المقترح بالثواني
        self.heartbeat_timeout = 20  # مهلة نبض القلب بالثواني

    def register_worker(self, agent_id: str, role: str):
        """تسجيل وكيل جديد في السرب مع تهيئة نبض القلب."""
        self.workers[agent_id] = {
            "role": role,
            "status": "active",
            "last_seen": time.time(),
            "joined_at": time.time()
        }
        logger.info(f"👷 Worker Registered: {agent_id} as {role}")

    def heartbeat(self, agent_id: str):
        """تحديث نبض القلب للوكيل لضمان أنه لا يزال نشطاً."""
        if agent_id in self.workers:
            self.workers[agent_id]["last_seen"] = time.time()
            self.workers[agent_id]["status"] = "active"

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

    def report_result(self, agent_id: str, task: str, result: str):
        """تسجيل نتيجة مهمة من وكيل فرعي."""
        entry = {
            "agent_id": agent_id,
            "task": task,
            "result": result,
            "ts": time.time()
        }
        self.results.append(entry)
        # حفظ النتيجة في ملف سجل السرب
        res_path = self.storage_dir / "swarm_results.jsonl"
        with open(res_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.info(f"✅ Result Reported by {agent_id}")

    def get_swarm_status(self) -> Dict[str, Any]:
        """الحصول على نظرة عامة على حالة السرب الحالي."""
        return {
            "active_workers": len(self.workers),
            "completed_tasks": len(self.results),
            "recent_results": self.results[-5:] if self.results else []
        }

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
        """حفظ حالة المقترح في القرص للمراجعة والتدقيق."""
        p_path = self.storage_dir / f"{proposal.proposal_id}.json"
        data = {
            "id": proposal.proposal_id,
            "proposer": proposal.proposer,
            "type": proposal.action_type,
            "data": proposal.data,
            "votes": proposal.votes,
            "status": proposal.status,
            "created_at": proposal.created_at
        }
        with open(p_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

swarm_manager = SwarmManager()
