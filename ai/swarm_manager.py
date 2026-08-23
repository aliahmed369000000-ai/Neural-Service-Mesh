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

    def register_worker(self, agent_id: str, role: str):
        """تسجيل وكيل جديد في السرب."""
        self.workers[agent_id] = {
            "role": role,
            "status": "active",
            "joined_at": time.time()
        }
        logger.info(f"👷 Worker Registered: {agent_id} as {role}")

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
        """تحليل الأصوات الحالية لمعرفة ما إذا تم الوصول للتوافق."""
        proposal = self.proposals.get(proposal_id)
        if not proposal: return {"status": "not_found"}
        
        total_weight = sum(v["weight"] for v in proposal.votes.values())
        if total_weight == 0:
            return {"status": "pending", "score": 0}
            
        yes_weight = sum(v["weight"] for v in proposal.votes.values() if v["vote"])
        score = yes_weight / total_weight
        
        if score >= self.threshold:
            proposal.status = "approved"
            self._save_proposal(proposal)
            return {"status": "approved", "score": score}
        elif len(proposal.votes) >= 3 and score < 0.4: # رفض مبكر إذا كانت الأصوات سلبية جداً
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
