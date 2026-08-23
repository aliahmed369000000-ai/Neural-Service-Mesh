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
        
        # إعدادات منحنى النسيان (Ebbinghaus Forgetting Curve)
        self.forgetting_enabled = True
        self.decay_rate = 0.0001  # معدل التلاشي (يمكن تعديله حسب الحاجة)
        self.prune_threshold = 0.2  # حد القوة الذي عنده يتم حذف الذكرى
        self.boost_factor = 0.2  # مقدار زيادة القوة عند كل استرجاع ناجح
        
    def add_to_stm(self, message: Dict[str, Any]):
        """إضافة رسالة إلى الذاكرة قصيرة الأجل مع التحقق من الحدود."""
        self.stm.append(message)
        if len(self.stm) > self.stm_limit:
            self.consolidate()

    def consolidate(self, force: bool = False):
        """ترحيل المعلومات من STM إلى LTM (عملية التوحيد) مع التقييم الذاتي والمشاركة الجماعية."""
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
        
        # 1. استخراج الحقائق الدقيقة (Semantic LTM) مع التقييم والمشاركة
        from ai.shared_experience import shared_experience
        facts_data = self._extract_facts(to_migrate)
        for f in facts_data:
            fact_id = self.add_fact(f["content"], importance=f["importance"])
            # مشاركة الحقائق المهمة مع السرب
            if f["importance"] >= 0.7:
                shared_experience.share_fact(self.agent_id, self.ltm_semantic[fact_id])
        
        # مزامنة المعرفة الجماعية الواردة
        shared_experience.sync_agent_memory(self)
        
        # 2. إنشاء تلخيص أحداثي (Episodic LTM)
        fact_contents = [f["content"] for f in facts_data]
        avg_importance = sum(f["importance"] for f in facts_data) / len(facts_data) if facts_data else 0.3
        self._store_in_episodic(to_migrate, fact_contents, importance=avg_importance)
        
        # 3. تنظيف STM (الاحتفاظ برسالة النظام وآخر الرسائل)
        system_msg = self.stm[0]
        recent = self.stm[-5:]
        self.stm = [system_msg] + recent

    def reflect(self, content: str) -> float:
        """
        محرك التقييم الذاتي (Self-Reflection): تقييم أهمية المعلومة (0.0 - 1.0).
        المعلومات التقنية والقرارات تحصل على وزن أعلى.
        """
        importance = 0.3  # القيمة الافتراضية للمعلومات العادية
        
        # 1. البحث عن مؤشرات الأهمية التقنية
        if re.search(r"```python|import |def |class ", content):
            importance += 0.4  # كود برمجي
        if re.search(r"SHA\s*[a-f0-9]{7,40}", content, re.I):
            importance += 0.3  # التزام Git
        if re.search(r"تم (?:تنفيذ|رفع|إنجاز|إصلاح)", content):
            importance += 0.2  # إنجاز مهمة
        if re.search(r"\d+(?:\.\d+)?%|\d+\s*ms|\d+\s*KB", content):
            importance += 0.2  # مقاييس أداء
            
        return min(1.0, importance)

    def _extract_facts(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """استخراج الحقائق والقرارات التقنية مع تقييم أهميتها."""
        extracted = []
        for m in messages:
            content = m.get("content", "")
            facts = []
            
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
            
            for f in set(facts):
                importance = self.reflect(f)
                extracted.append({"content": f, "importance": importance})
                
        return extracted

    def add_fact(self, fact_content: str, semantic_hash: Optional[str] = None, importance: float = 1.0):
        """إضافة حقيقة مفردة إلى الذاكرة الدلالية مع تحديد القوة بناءً على الأهمية."""
        if not semantic_hash:
            from ai.multimodal_sync import MultimodalSyncManager
            sync_manager = MultimodalSyncManager()
            embedding = sync_manager._generate_embedding(fact_content)
            semantic_hash = sync_manager._generate_lsh_hash(embedding)
            
        # ضمان أن fact_content سلسلة نصية لـ hash()
        content_str = str(fact_content)
        fact_id = f"fact_{int(time.time())}_{hash(content_str)%1000}_{len(self.ltm_semantic)}"
        self.ltm_semantic[fact_id] = {
            "content": fact_content,
            "timestamp": time.time(),
            "last_access": time.time(),
            "strength": importance,
            "last_access": time.time(),
            "access_count": 0,
            "semantic_hash": semantic_hash
        }
        return fact_id

    def _store_in_semantic(self, facts: List[str]):
        """حفظ مجموعة حقائق في الذاكرة الدلالية (للتوافق)."""
        for fact in facts:
            self.add_fact(fact)

    def _store_in_episodic(self, messages: List[Dict[str, Any]], facts: List[str], importance: float = 1.0):
        """حفظ ملخص التجربة في الذاكرة الأحداثية مع تحديد القوة بناءً على الأهمية."""
        summary = f"أرشفة {len(messages)} رسالة. تم استخراج {len(facts)} حقيقة."
        if facts: summary += f" أهمها: {facts[0]}"
        
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        embedding = sync_manager._generate_embedding(summary)
        lsh_hash = sync_manager._generate_lsh_hash(embedding)
        
        episode = {
            "timestamp": time.time(),
            "last_access": time.time(),
            "strength": importance,
            "access_count": 0,
            "summary": summary,
            "raw_count": len(messages),
            "semantic_hash": lsh_hash,
            "associated_facts": facts[:3]
        }
        self.ltm_episodic.append(episode)

    def _calculate_current_strength(self, memory: Dict[str, Any]) -> float:
        """حساب القوة الحالية للذكرى بناءً على منحنى النسيان."""
        if not self.forgetting_enabled:
            return memory.get("strength", 1.0)
            
        elapsed = time.time() - memory.get("last_access", time.time())
        # R = e^(-t/S) -> هنا نستخدم تبسيطاً: strength = strength * exp(-decay * elapsed)
        import math
        current_strength = memory.get("strength", 1.0) * math.exp(-self.decay_rate * elapsed)
        return max(0.0, current_strength)

    def search(self, query: str, limit: int = 3) -> Dict[str, List[Dict[str, Any]]]:
        """البحث النشط (Active Retrieval) في LTM مع تطبيق منحنى النسيان."""
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        query_vec = sync_manager._generate_embedding(query)
        query_hash = sync_manager._generate_lsh_hash(query_vec)
        
        results = {"episodic": [], "semantic": []}
        now = time.time()
        
        # البحث في الأحداث
        for ep in self.ltm_episodic:
            e_hash = ep.get("semantic_hash")
            if not e_hash: continue
            dist = sum(c1 != c2 for c1, c2 in zip(query_hash, e_hash))
            if dist <= 2:
                # تحديث القوة عند الاسترجاع
                ep["strength"] = min(1.0, self._calculate_current_strength(ep) + self.boost_factor)
                ep["last_access"] = now
                ep["access_count"] = ep.get("access_count", 0) + 1
                
                ep["score"] = 1.0 - (dist / len(query_hash))
                results["episodic"].append(ep)
        
        # البحث في الحقائق
        for f_id, fact in self.ltm_semantic.items():
            f_hash = fact.get("semantic_hash")
            if not f_hash: continue
            dist = sum(c1 != c2 for c1, c2 in zip(query_hash, f_hash))
            if dist <= 2:
                # تحديث القوة عند الاسترجاع
                fact["strength"] = min(1.0, self._calculate_current_strength(fact) + self.boost_factor)
                fact["last_access"] = now
                fact["access_count"] = fact.get("access_count", 0) + 1
                
                fact["score"] = 1.0 - (dist / len(query_hash))
                results["semantic"].append(fact)
                
        # ترتيب النتائج
        results["episodic"] = sorted(results["episodic"], key=lambda x: x.get("score", 0), reverse=True)[:limit]
        results["semantic"] = sorted(results["semantic"], key=lambda x: x.get("score", 0), reverse=True)[:limit]
        
        # تنظيف الذاكرة دورياً (Pruning)
        self.prune()
        
        return results

    def prune(self):
        """تنظيف الذكريات الضعيفة جداً بناءً على منحنى النسيان."""
        if not self.forgetting_enabled:
            return

        # تنظيف الأحداث
        initial_ep = len(self.ltm_episodic)
        self.ltm_episodic = [ep for ep in self.ltm_episodic if self._calculate_current_strength(ep) > self.prune_threshold]
        
        # تنظيف الحقائق
        initial_sem = len(self.ltm_semantic)
        self.ltm_semantic = {f_id: fact for f_id, fact in self.ltm_semantic.items() 
                             if self._calculate_current_strength(fact) > self.prune_threshold}
        
        diff_ep = initial_ep - len(self.ltm_episodic)
        diff_sem = initial_sem - len(self.ltm_semantic)
        
        if diff_ep > 0 or diff_sem > 0:
            logger.info(f"🧹 تم تنظيف الذاكرة: حذف {diff_ep} حدث و {diff_sem} حقيقة ضعيفة.")

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
