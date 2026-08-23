"""
ai/agent_hibernation.py
=======================
إضافة ميزة "النوم الذكي" (Smart Sleep/Hibernation) للوكلاء.
تسمح هذه الميزة بحفظ حالة الوكيل كاملة (الذاكرة، الخطط، السياق) 
واستعادتها لاحقاً لتوفير الموارد أو تجاوز حدود الجلسات المؤقتة.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional, List

logger = logging.getLogger("NSM.AgentHibernation")

ROOT = Path(__file__).resolve().parent.parent
SLEEP_DIR = ROOT / "artifacts" / "agent_sleep"
SLEEP_DIR.mkdir(parents=True, exist_ok=True)

class AgentState:
    """تمثيل لحالة الوكيل القابلة للحفظ مع دعم الذاكرة الهرمية (Hierarchical Memory)."""
    def __init__(self, agent_id: str, context: List[Dict[str, Any]], plan: Dict[str, Any], 
                 metadata: Dict[str, Any] = None, memory_snapshot: Dict[str, Any] = None,
                 pending_tasks: List[str] = None, memory_shards: Dict[str, str] = None,
                 visual_context: Dict[str, Any] = None, audio_context: Dict[str, Any] = None,
                 multimodal_memory: Dict[str, Any] = None,
                 episodic_memory: List[Dict[str, Any]] = None,
                 semantic_memory: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.context = context  # الذاكرة العاملة (Working Memory)
        self.plan = plan
        self.memory_snapshot = memory_snapshot or {}
        self.pending_tasks = pending_tasks or []
        self.memory_shards = memory_shards or {}
        self.visual_context = visual_context or {}
        self.audio_context = audio_context or {}
        self.multimodal_memory = multimodal_memory or {}
        self.episodic_memory = episodic_memory or []  # ذاكرة التجارب والأحداث (Episodic Memory)
        self.semantic_memory = semantic_memory or {}  # ذاكرة الحقائق والمفاهيم (Semantic Memory)
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def _extract_facts(self, messages: List[Dict[str, Any]]) -> List[str]:
        """استخراج الحقائق والقرارات الهامة من مجموعة رسائل (محاكاة دقيقة)."""
        facts = []
        for m in messages:
            content = m.get("content", "")
            # البحث عن أنماط القرارات والنتائج
            import re
            # استخراج الأكواد البرمجية
            codes = re.findall(r"```python\n(.*?)\n```", content, re.DOTALL)
            for c in codes: facts.append(f"كود برمجي: {c[:50]}...")
            
            # استخراج القرارات الصريحة
            decisions = re.findall(r"(?:تم تنفيذ|تم رفع|القرار هو):\s*([^.\n]+)", content)
            facts.extend([f"قرار: {d.strip()}" for d in decisions])
            
            # استخراج SHA ورموز الالتزام
            shas = re.findall(r"SHA:\s*([a-f0-9]{7,40})", content)
            for s in shas: facts.append(f"SHA التزام: {s}")
            
            # استخراج النتائج الرقمية
            metrics = re.findall(r"(\d+(?:\.\d+)?%|\d+\s*ms|\d+\s*KB)", content)
            if metrics: facts.append(f"مقاييس أداء: {', '.join(metrics)}")
            
        return list(set(facts))

    def _auto_summarize_episodic(self):
        """تحويل الذاكرة العاملة القديمة إلى ذاكرة أحداثية ملخصة مع دعم ANN والكيانات."""
        if len(self.context) <= 15: return
        
        # استخراج الرسائل القديمة (ما عدا النظام وآخر 5 رسائل)
        to_summarize = self.context[1:-5]
        if not to_summarize: return
        
        # 1. استخراج الحقائق الدقيقة قبل التلخيص
        extracted_facts = self._extract_facts(to_summarize)
        for fact in extracted_facts:
            # حفظ الحقائق في الذاكرة الدلالية (Semantic Memory)
            fact_id = f"fact_{int(time.time())}_{hash(fact)%1000}"
            self.semantic_memory[fact_id] = {"content": fact, "timestamp": time.time()}
        
        # 2. محاكاة التلخيص الهيكلي
        summary_text = f"أرشفة {len(to_summarize)} رسالة. تم استخراج {len(extracted_facts)} حقيقة دقيقة."
        if extracted_facts:
            summary_text += f" أهمها: {extracted_facts[0]}"
        
        # توليد تضمين دلالي للتلخيص (ANN Support)
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        embedding = sync_manager._generate_embedding(summary_text)
        lsh_hash = sync_manager._generate_lsh_hash(embedding)
        
        episode = {
            "timestamp": time.time(),
            "summary": summary_text,
            "raw_count": len(to_summarize),
            "importance": 0.7,
            "semantic_hash": lsh_hash
        }
        self.episodic_memory.append(episode)
        
        # تحديث السياق (الاحتفاظ بالنظام وآخر 5 رسائل)
        system_msg = self.context[0]
        recent = self.context[-5:]
        self.context = [system_msg] + recent
        logger.info(f"🧠 تم أرشفة {len(to_summarize)} رسالة في الذاكرة الأحداثية.")

    def compress(self, target_size_kb: int = 100):
        """ضغط الذاكرة لتقليل حجم الحالة المحفوظة."""
        # محاولة التلخيص التلقائي أولاً قبل الضغط الفيزيائي
        self._auto_summarize_episodic()
        
        initial_size = len(json.dumps(self.to_dict())) / 1024
        if initial_size <= target_size_kb: return
        
        logger.info(f"🗜️ بدء ضغط الذاكرة (الحجم الحالي: {initial_size:.1f}KB)...")
        
        # 1. ضغط السياق الإضافي إذا لزم الأمر
        if len(self.context) > 6:
            system_msg = self.context[0]
            recent = self.context[-3:]
            self.context = [system_msg] + recent
            
        # 2. ضغط الذاكرة متعددة الوسائط (الاحتفاظ بنقاط المزامنة ذات الثقة العالية فقط)
        for vid_id in list(self.multimodal_memory.keys()):
            sync_data = self.multimodal_memory[vid_id].get("multimodal_sync", [])
            if len(sync_data) > 20:
                # ترتيب حسب الأهمية (هنا نفترض وجود semantic_index أو نقاط زمنية مفتاحية)
                # للتبسيط: نأخذ عينات منتظمة (Downsampling)
                step = len(sync_data) // 15
                self.multimodal_memory[vid_id]["multimodal_sync"] = sync_data[::step]
                self.multimodal_memory[vid_id]["compressed"] = True

        # 3. إزالة الميتاداتا غير الضرورية
        self.metadata = {k: v for k, v in self.metadata.items() if k in ["lazy_loaded", "agent_type"]}
        
        final_size = len(json.dumps(self.to_dict())) / 1024
        logger.info(f"✅ تم الضغط: {initial_size:.1f}KB -> {final_size:.1f}KB")

    def search_episodic(self, query: str) -> List[Dict[str, Any]]:
        """البحث الدلالي في الذاكرة الأحداثية باستخدام ANN."""
        from ai.multimodal_sync import MultimodalSyncManager
        sync_manager = MultimodalSyncManager()
        query_vec = sync_manager._generate_embedding(query)
        query_hash = sync_manager._generate_lsh_hash(query_vec)
        
        results = []
        for episode in self.episodic_memory:
            e_hash = episode.get("semantic_hash")
            if not e_hash: continue
            
            # حساب مسافة هاملينج (ANN Filter)
            distance = sum(c1 != c2 for c1, c2 in zip(query_hash, e_hash))
            if distance <= 2: # عتبة التشابه
                results.append(episode)
        
        return sorted(results, key=lambda x: x["timestamp"], reverse=True)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "context": self.context,
            "plan": self.plan,
            "memory_snapshot": self.memory_snapshot,
            "pending_tasks": self.pending_tasks,
            "memory_shards": self.memory_shards,
            "visual_context": self.visual_context,
            "audio_context": self.audio_context,
            "multimodal_memory": self.multimodal_memory,
            "episodic_memory": self.episodic_memory,
            "semantic_memory": self.semantic_memory,
            "metadata": self.metadata,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentState':
        state = cls(
            data["agent_id"], 
            data["context"], 
            data["plan"], 
            data.get("metadata", {}),
            data.get("memory_snapshot", {}),
            data.get("pending_tasks", []),
            data.get("memory_shards", {}),
            data.get("visual_context", {}),
            data.get("audio_context", {}),
            data.get("multimodal_memory", {}),
            data.get("episodic_memory", []),
            data.get("semantic_memory", {})
        )
        state.timestamp = data.get("timestamp", time.time())
        return state

def hibernate_agent(agent_id: str, context: List[Dict[str, Any]], plan: Dict[str, Any], 
                    metadata: Dict[str, Any] = None, memory_snapshot: Dict[str, Any] = None,
                    pending_tasks: List[str] = None, memory_shards: Dict[str, str] = None,
                    visual_context: Dict[str, Any] = None, audio_context: Dict[str, Any] = None,
                    multimodal_memory: Dict[str, Any] = None, compress: bool = True) -> bool:
    """حفظ حالة الوكيل في ملف محلي لدخول وضع النوم مع دعم الضغط التلقائي."""
    try:
        state = AgentState(agent_id, context, plan, metadata, memory_snapshot, pending_tasks, memory_shards, visual_context, audio_context, multimodal_memory)
        
        if compress:
            state.compress(target_size_kb=50) # ضغط إذا تجاوز 50KB
            
        file_path = SLEEP_DIR / f"{agent_id}_sleep.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"✅ الوكيل {agent_id} دخل في وضع النوم. تم حفظ الحالة في {file_path}")
        return True
    except Exception as e:
        logger.error(f"❌ فشل دخول الوكيل {agent_id} في وضع النوم: {e}")
        return False

def wake_up_agent(agent_id: str, lazy: bool = False) -> Optional[AgentState]:
    """
    استعادة حالة الوكيل من ملف النوم.
    إذا كان lazy=True، يتم تحميل سياق محدود لتوفير الموارد.
    """
    try:
        file_path = SLEEP_DIR / f"{agent_id}_sleep.json"
        if not file_path.exists():
            logger.info(f"ℹ️ لا توجد حالة نوم مسجلة للوكيل {agent_id}")
            return None
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        state = AgentState.from_dict(data)
        
        if lazy:
            # التحميل الكسول: الاحتفاظ بآخر 5 رسائل فقط + رسالة النظام الأصلية
            system_msg = next((m for m in state.context if m.get("role") == "system"), None)
            recent_context = state.context[-5:]
            if system_msg and system_msg not in recent_context:
                state.context = [system_msg] + recent_context
            else:
                state.context = recent_context
            
            # إضافة وسم metadata يشير إلى أن التحميل كان جزئياً
            state.metadata["lazy_loaded"] = True
            state.metadata["original_context_len"] = len(data["context"])
            
        logger.info(f"🌅 استيقظ الوكيل {agent_id} ({'تحميل كسول' if lazy else 'تحميل كامل'}).")
        return state
    except Exception as e:
        logger.error(f"❌ فشل استيقاظ الوكيل {agent_id}: {e}")
        return None

def extract_pending_tasks(context: List[Dict[str, Any]], plan: Dict[str, Any]) -> List[str]:
    """
    استخراج المهام التي لم تُنجز بعد من الخطة وسياق المحادثة.
    """
    pending = []
    # 1. فحص الخطة الهيكلية إن وجدت
    tasks = plan.get("tasks", [])
    for t in tasks:
        if isinstance(t, dict) and t.get("status") not in ["completed", "done", "finished"]:
            pending.append(str(t.get("title", "مهمة غير محددة")))
        elif isinstance(t, str):
            pending.append(t)
            
    # 2. إذا كانت القائمة فارغة، نحاول الاستنتاج من آخر رسالة مساعد
    if not pending:
        last_assistant = next((m["content"] for m in reversed(context) if m["role"] == "assistant"), "")
        # البحث عن أنماط مثل "الخطوة التالية:" أو "سأقوم بـ..."
        matches = re.findall(r"(?:الخطوة التالية|سأقوم بـ|يجب تنفيذ):\s*(.+)", last_assistant)
        pending.extend([m.strip() for m in matches])
        
    return pending[:5] # نكتفي بأهم 5 مهام لتوفير المساحة

def list_sleeping_agents() -> List[str]:
    """قائمة بالوكلاء الموجودين حالياً في وضع النوم."""
    return [f.stem.replace("_sleep", "") for f in SLEEP_DIR.glob("*_sleep.json")]

def schedule_wake_up(agent_id: str, delay_seconds: int):
    """
    جدولة إيقاظ الوكيل بعد فترة زمنية محددة.
    يتم ذلك عبر تشغيل عملية خلفية بسيطة.
    """
    import subprocess
    import sys
    
    # سكربت بسيط للاستيقاظ بعد فترة
    cmd = [
        sys.executable,
        "-c",
        f"import time; time.sleep({delay_seconds}); from ai.agent_hibernation import wake_up_agent; wake_up_agent('{agent_id}')"
    ]
    
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    logger.info(f"⏰ تم جدولة إيقاظ الوكيل {agent_id} بعد {delay_seconds} ثانية.")
