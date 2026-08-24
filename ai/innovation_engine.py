
import logging
import json
from typing import Dict, Any, List

logger = logging.getLogger("NSM.InnovationEngine")

class InnovationEngine:
    """
    محرك الابتكار الخوارزمي (Algorithmic Innovation Engine).
    يسمح للوكلاء باقتراح واختبار خوارزميات جديدة.
    """
    def __init__(self):
        self.proposed_algorithms = []
        self.verified_innovations = []

    def propose_algorithm(self, name: str, description: str, code: str, category: str) -> Dict[str, Any]:
        """اقتراح خوارزمية جديدة."""
        proposal = {
            "id": len(self.proposed_algorithms) + 1,
            "name": name,
            "description": description,
            "code": code,
            "category": category, # [Attention, Optimizer, Loss, Architecture]
            "status": "Proposed"
        }
        self.proposed_algorithms.append(proposal)
        logger.info(f"💡 ابتكار جديد مقترح: {name} ({category})")
        return proposal

    def test_innovation(self, proposal_id: int, success: bool, metrics: Dict[str, float]) -> Dict[str, Any]:
        """تسجيل نتائج اختبار الابتكار."""
        for p in self.proposed_algorithms:
            if p["id"] == proposal_id:
                p["status"] = "Verified" if success else "Failed"
                p["metrics"] = metrics
                if success:
                    self.verified_innovations.append(p)
                    logger.info(f"🚀 تم التحقق من نجاح الابتكار: {p['name']}")
                return p
        return {"error": "Proposal not found"}

    def get_verified_innovations(self) -> List[Dict[str, Any]]:
        return self.verified_innovations

innovation_engine = InnovationEngine()
