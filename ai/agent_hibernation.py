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
    """تمثيل لحالة الوكيل القابلة للحفظ."""
    def __init__(self, agent_id: str, context: List[Dict[str, Any]], plan: Dict[str, Any], metadata: Dict[str, Any] = None, memory_snapshot: Dict[str, Any] = None):
        self.agent_id = agent_id
        self.context = context
        self.plan = plan
        self.memory_snapshot = memory_snapshot or {}
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "context": self.context,
            "plan": self.plan,
            "memory_snapshot": self.memory_snapshot,
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
            data.get("memory_snapshot", {})
        )
        state.timestamp = data.get("timestamp", time.time())
        return state

def hibernate_agent(agent_id: str, context: List[Dict[str, Any]], plan: Dict[str, Any], metadata: Dict[str, Any] = None, memory_snapshot: Dict[str, Any] = None) -> bool:
    """حفظ حالة الوكيل في ملف محلي لدخول وضع النوم."""
    try:
        state = AgentState(agent_id, context, plan, metadata, memory_snapshot)
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
