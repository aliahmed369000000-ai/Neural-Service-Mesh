"""
ai/shared_experience.py
=======================
محرك الخبرة الجماعية (Shared Experience Engine).
يسمح للوكلاء بمشاركة الحقائق عالية الأهمية في قاعدة بيانات موحدة.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NSM.SharedExperience")

class SharedExperienceManager:
    def __init__(self, storage_path: str = "artifacts/learning/shared_knowledge.json"):
        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.knowledge = self._load_knowledge()
        
    def _load_knowledge(self) -> Dict[str, Any]:
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"فشل تحميل المعرفة الجماعية: {e}")
        return {"shared_facts": {}, "active_queries": {}, "global_metrics": {}, "version": "1.1"}

    def _save_knowledge(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"فشل حفظ المعرفة الجماعية: {e}")

    def share_fact(self, agent_id: str, fact: Dict[str, Any]):
        """مشاركة حقيقة إذا كانت أهميتها تتجاوز الحد المسموح واجتازت التقييم الذاتي."""
        importance = fact.get("strength", 0)
        content = fact.get("content", "")
        
        # 1. التحقق من الحد الأدنى للأهمية
        if importance < 0.7:
            return False
            
        # 2. التقييم الذاتي للحلول البرمجية (إذا كان المحتوى كوداً)
        if "```" in content or "import " in content or "def " in content:
            from ai.learning_engine import learning_engine
            eval_res = learning_engine.evaluate_solution(content)
            if not eval_res["approved"]:
                logger.warning(f"🛡️ رفض مشاركة حل برمجي ضعيف من {agent_id}: {eval_res['reasons']}")
                return False
            importance = (importance + eval_res["score"]) / 2 # دمج تقييم الكفاءة مع الأهمية
            
        fact_id = f"shared_{hash(content) % 10000}"
        if fact_id not in self.knowledge["shared_facts"]:
            self.knowledge["shared_facts"][fact_id] = {
                "content": fact["content"],
                "origin_agent": agent_id,
                "shared_at": time.time(),
                "importance": importance,
                "verification_count": 1,
                "semantic_hash": fact.get("semantic_hash")
            }
            self._save_knowledge()
            logger.info(f"🌐 تم نشر حقيقة جماعية جديدة من الوكيل {agent_id}")
            return True
        else:
            # زيادة التوثيق إذا كانت الحقيقة معروفة مسبقاً
            self.knowledge["shared_facts"][fact_id]["verification_count"] += 1
            self._save_knowledge()
            return True

    def sync_agent_memory(self, agent_memory: Any):
        """مزامنة ذاكرة الوكيل مع المعرفة الجماعية."""
        new_facts_count = 0
        for fact_id, fact in self.knowledge["shared_facts"].items():
            # إذا لم تكن الحقيقة موجودة لدى الوكيل، أضفها
            exists = any(f["content"] == fact["content"] for f in agent_memory.ltm_semantic.values())
            if not exists:
                agent_memory.add_fact(
                    fact["content"], 
                    semantic_hash=fact.get("semantic_hash"),
                    importance=fact["importance"] * 0.9
                )
                new_facts_count += 1
        
        if new_facts_count > 0:
            logger.info(f"📥 تم مزامنة {new_facts_count} حقيقة جماعية للوكيل {agent_memory.agent_id}")
        return new_facts_count

    def ask_swarm(self, agent_id: str, query: str, context: str = ""):
        """طرح سؤال توضيحي على السرب."""
        query_id = f"q_{hash(query + str(time.time())) % 10000}"
        self.knowledge["active_queries"][query_id] = {
            "query": query,
            "context": context,
            "asker": agent_id,
            "timestamp": time.time(),
            "status": "open",
            "answers": []
        }
        self._save_knowledge()
        logger.info(f"❓ سؤال جديد من {agent_id}: {query}")
        return query_id

    def answer_query(self, agent_id: str, query_id: str, answer: str):
        """تقديم إجابة لسؤال موجود."""
        if query_id in self.knowledge["active_queries"]:
            self.knowledge["active_queries"][query_id]["answers"].append({
                "answer": answer,
                "provider": agent_id,
                "timestamp": time.time()
            })
            self.knowledge["active_queries"][query_id]["status"] = "answered"
            self._save_knowledge()
            logger.info(f"💡 إجابة جديدة من {agent_id} للسؤال {query_id}")
            return True
        return False

    def get_pending_queries(self) -> List[Dict[str, Any]]:
        """جلب الأسئلة التي تحتاج إلى إجابات."""
        return [
            {"id": q_id, **data} 
            for q_id, data in self.knowledge["active_queries"].items() 
            if data["status"] == "open"
        ]

    def check_my_answers(self, agent_id: str) -> List[Dict[str, Any]]:
        """التحقق من وجود إجابات لأسئلة وكيل معين."""
        my_queries = []
        for q_id, data in self.knowledge["active_queries"].items():
            if data["asker"] == agent_id and data["answers"]:
                my_queries.append({"id": q_id, **data})
        return my_queries

shared_experience = SharedExperienceManager()
