
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("NSM.LearningEngine")

ROOT = Path(__file__).resolve().parent.parent
LEARNING_DIR = ROOT / "artifacts" / "learning"
LEARNING_DIR.mkdir(parents=True, exist_ok=True)

class LearningEngine:
    """محرك التعلم التراكمي للوكلاء."""
    def __init__(self):
        self.knowledge_base_file = LEARNING_DIR / "experience_db.json"
        self.experience_db = self._load_db()

    def _load_db(self) -> List[Dict[str, Any]]:
        if self.knowledge_base_file.exists():
            try:
                with open(self.knowledge_base_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"❌ خطأ تحميل قاعدة الخبرة: {e}")
        return []

    def _save_db(self):
        try:
            with open(self.knowledge_base_file, "w", encoding="utf-8") as f:
                json.dump(self.experience_db, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ خطأ حفظ قاعدة الخبرة: {e}")

    def record_experience(self, task: str, outcome: str, lesson: str, success: bool, agent_id: str = "global"):
        """تسجيل خبرة جديدة مع دعم هوية الوكيل."""
        experience = {
            "agent_id": agent_id,
            "task_type": task,
            "outcome": outcome,
            "lesson": lesson,
            "success": success,
            "timestamp": time.time(),
            "weight": 1.0 if success else 2.0 # الأخطاء لها وزن أكبر للتعلم
        }
        self.experience_db.append(experience)
        # الاحتفاظ بآخر 100 خبرة فقط لضمان الكفاءة
        if len(self.experience_db) > 100:
            self.experience_db.pop(0)
        self._save_db()

    def get_relevant_lessons(self, current_task: str) -> str:
        """جلب الدروس المستفادة ذات الصلة بالمهمة الحالية (جماعياً)."""
        current_keywords = set(current_task.lower().split())
        relevant = []
        for exp in self.experience_db:
            task_type = exp.get("task_type", "")
            exp_keywords = set(task_type.lower().split())
            if current_keywords.intersection(exp_keywords):
                relevant.append(exp)
        
        if not relevant:
            return ""
        
        summary = "\n💡 دروس مستفادة من المعرفة الجماعية (Collective Knowledge):\n"
        for exp in relevant[-5:]: # زيادة عدد الدروس المسترجعة للتنوع
            status = "✅" if exp.get("success") else "❌"
            origin = f" [الوكيل: {exp.get('agent_id', 'global')}]"
            summary += f"- {status}{origin} في مهمة '{exp.get('task_type')}': {exp.get('lesson')}\n"
        return summary

    def import_expert_seeds(self, seeds_file: str):
        """استيراد بذور خبرة خارجية من ملف JSON."""
        try:
            path = Path(seeds_file)
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    seeds = json.load(f)
                    for seed in seeds:
                        seed["agent_id"] = seed.get("agent_id", "expert_seed")
                        seed["timestamp"] = seed.get("timestamp", time.time())
                        self.experience_db.append(seed)
                self._save_db()
                logger.info(f"✅ تم استيراد {len(seeds)} خبرة خبيرة.")
        except Exception as e:
            logger.error(f"❌ خطأ استيراد البذور: {e}")

# نسخة عالمية واحدة
learning_engine = LearningEngine()
