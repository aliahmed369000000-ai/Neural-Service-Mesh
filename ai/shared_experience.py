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
        return {"shared_facts": {}, "global_metrics": {}, "version": "1.0"}

    def _save_knowledge(self):
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(self.knowledge, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"فشل حفظ المعرفة الجماعية: {e}")

    def share_fact(self, agent_id: str, fact: Dict[str, Any]):
        """مشاركة حقيقة إذا كانت أهميتها تتجاوز الحد المسموح."""
        importance = fact.get("strength", 0)
        if importance < 0.7:  # مشاركة الحقائق المهمة فقط
            return False
            
        fact_id = f"shared_{hash(fact['content']) % 10000}"
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

shared_experience = SharedExperienceManager()
