"""
ai/memory_manager.py
====================
محرك إدارة الذاكرة المستقل للوكلاء.
ينظم تدفق المعلومات بين الذاكرة العاملة (STM) والمخازن الدائمة (LTM).
يدعم التلخيص الهيكلي، استخراج الحقائق، والبحث الدلالي ANN.
"""

import json
import time
import logging
import re
from typing import Any, Dict, List, Optional
from pathlib import Path

logger = logging.getLogger("NSM.MemoryManager")

class MemoryManager:
    """مدير الذاكرة الهرمي (Short-term & Long-term Memory)."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.stm = []  # الذاكرة قصيرة الأجل (Short-term Memory / Working Context)
        self.ltm_episodic = []  # الذاكرة الأحداثية طويلة الأجل (Episodic LTM)
        self.ltm_semantic = {}  # الذاكرة الدلالية طويلة الأجل (Semantic LTM - Facts)
        self.stm_limit = 15  # حد الرسائل في الذاكرة العاملة قبل الترحيل
        
    def add_to_stm(self, message: Dict[str, Any]):
        """إضافة رسالة إلى الذاكرة قصيرة الأجل مع التحقق من الحدود."""
        self.stm.append(message)
        if len(self.stm) > self.stm_limit:
            self.consolidate()

    def consolidate(self, force: bool = False):
        """ترحيل المعلومات من STM إلى LTM (عملية التوحيد)."""
        if not force and len(self.stm) <= 10:
            return

        # استخراج الرسائل القديمة للترحيل (ما عدا النظام وآخر 5 رسائل)
        if len(self.stm) <= 6:
            to_migrate = self.stm[1:]
        else:
            to_migrate = self.stm[1:-5]
            
        if not to_migrate:
            return

        logger.info(f"🧠 بدء توحيد الذاكرة للوكيل {self.agent_id} (ترحيل {len(to_migrate)} رسالة)...")
        
        # 1. استخراج الحقائق الدقيقة (Semantic LTM)
        facts = self._extract_facts(to_migrate)
        self._store_in_semantic(facts)
        
        # 2. إنشاء تلخيص أحداثي (Episodic LTM)
        self._store_in_episodic(to_migrate, facts)
        
        # 3. تنظيف STM (الاحتفاظ برسالة النظام وآخر الرسائل)
        system_msg = self.stm[0]
        recent = self.stm[-5:]
        self.stm = [system_msg] + recent

    def _extract_facts(self, messages: List[Dict[str, Any]]) -> List[str]:
        """استخراج الحقائق والقرارات التقنية من الرسائل."""
        facts = []
        for m in messages:
            content = m.get("content", "")
            # استخراج الأكواد
            codes = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            for c in codes: facts.append(f"كود برمجي: {c[:50]}...")
            
            # استخراج القرارات
            decisions = re.findall(r"(?:تم تنفيذ|تم رفع|القرار هو):\s*([^.\n]+)", content)
            facts.extend([f"قرار: {d.strip()}" for d in decisions])
            
            # استخراج SHA
            shas = re.findall(r"SHA\s*(?:هو|:)?\s*([a-f0-9]{7,40})", content, re.IGNORECASE)
            for s in shas: facts.append(f"SHA التزام: {s}")
            
            # استخراج المقاييس
            metrics = re.findall(r"(\d+(?:\.\d+)?%|\d+\s*ms|\d+\s*KB)", content)
            if metrics: facts.append(f"مقاييس أداء: {', '.join(metrics)}")
            
        return list(set(facts))

    def _store_in_semantic(self, facts: List[str]):
        """حفظ الحقائق في الذاكرة الدلالية مع دعم ANN."""
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        
        for fact in facts:
            embedding = sync_manager._generate_embedding(fact)
            lsh_hash = sync_manager._generate_lsh_hash(embedding)
            
            fact_id = f"fact_{int(time.time())}_{hash(fact)%1000}"
            self.ltm_semantic[fact_id] = {
                "content": fact,
                "timestamp": time.time(),
                "semantic_hash": lsh_hash
            }

    def _store_in_episodic(self, messages: List[Dict[str, Any]], facts: List[str]):
        """حفظ ملخص التجربة في الذاكرة الأحداثية."""
        summary = f"أرشفة {len(messages)} رسالة. تم استخراج {len(facts)} حقيقة."
        if facts: summary += f" أهمها: {facts[0]}"
        
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        embedding = sync_manager._generate_embedding(summary)
        lsh_hash = sync_manager._generate_lsh_hash(embedding)
        
        episode = {
            "timestamp": time.time(),
            "summary": summary,
            "raw_count": len(messages),
            "semantic_hash": lsh_hash,
            "associated_facts": facts[:3]
        }
        self.ltm_episodic.append(episode)

    def search(self, query: str, limit: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """البحث النشط (Active Retrieval) في LTM."""
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        query_vec = sync_manager._generate_embedding(query)
        query_hash = sync_manager._generate_lsh_hash(query_vec)
        
        results = {"episodic": [], "semantic": []}
        
        # البحث في الأحداث
        for ep in self.ltm_episodic:
            e_hash = ep.get("semantic_hash")
            if not e_hash: continue
            dist = sum(c1 != c2 for c1, c2 in zip(query_hash, e_hash))
            if dist <= 2:
                ep["score"] = 1.0 - (dist / len(query_hash))
                results["episodic"].append(ep)
        
        # البحث في الحقائق
        for f_id, fact in self.ltm_semantic.items():
            f_hash = fact.get("semantic_hash")
            if not f_hash: continue
            dist = sum(c1 != c2 for c1, c2 in zip(query_hash, f_hash))
            if dist <= 2:
                fact["score"] = 1.0 - (dist / len(query_hash))
                results["semantic"].append(fact)
                
        # ترتيب النتائج
        results["episodic"] = sorted(results["episodic"], key=lambda x: x.get("score", 0), reverse=True)[:limit]
        results["semantic"] = sorted(results["semantic"], key=lambda x: x.get("score", 0), reverse=True)[:limit]
        
        return results

    def to_dict(self) -> Dict[str, Any]:
        """تصدير الذاكرة كقاموس للحفظ."""
        return {
            "stm": self.stm,
            "ltm_episodic": self.ltm_episodic,
            "ltm_semantic": self.ltm_semantic
        }

    @classmethod
    def from_dict(cls, agent_id: str, data: Dict[str, Any]) -> 'MemoryManager':
        """استعادة الذاكرة من قاموس."""
        manager = cls(agent_id)
        manager.stm = data.get("stm", [])
        manager.ltm_episodic = data.get("ltm_episodic", [])
        manager.ltm_semantic = data.get("ltm_semantic", {})
        return manager
