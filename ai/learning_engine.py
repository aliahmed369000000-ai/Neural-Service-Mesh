
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
    """محرك التعلم التراكمي للوكلاء مع حماية المعرفة."""
    def __init__(self):
        self.knowledge_base_file = LEARNING_DIR / "experience_db.json"
        self.trust_scores_file = LEARNING_DIR / "trust_scores.json"
        self.experience_db = self._load_db()
        self.trust_scores = self._load_trust_scores()

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

    def _load_trust_scores(self) -> Dict[str, float]:
        if self.trust_scores_file.exists():
            try:
                with open(self.trust_scores_file, "r") as f:
                    return json.load(f)
            except: pass
        return {}

    def _save_trust_scores(self):
        with open(self.trust_scores_file, "w") as f:
            json.dump(self.trust_scores, f)

    def _auto_review_lesson(self, lesson: str) -> bool:
        """مراجعة تلقائية بسيطة لمنع الكلمات الضارة أو الأوامر الخطيرة في الدروس."""
        blacklist = ["rm -rf", "sudo", "delete all", "drop table", "format"]
        for word in blacklist:
            if word in lesson.lower():
                return False
        return True

    def record_experience(self, task: str, outcome: str, lesson: str, success: bool, agent_id: str = "global"):
        """تسجيل خبرة جديدة مع التحقق من الثقة والمراجعة التلقائية."""
        # 1. مراجعة المحتوى
        if not self._auto_review_lesson(lesson):
            logger.warning(f"🛡️ حظر درس ضار محتمل من الوكيل {agent_id}")
            return

        # 2. تحديث نقاط الثقة
        current_trust = self.trust_scores.get(agent_id, 0.5)
        if success:
            new_trust = min(1.0, current_trust + 0.05)
        else:
            new_trust = max(0.0, current_trust - 0.1)
        self.trust_scores[agent_id] = new_trust
        self._save_trust_scores()

        # 3. حماية: رفض الخبرات من الوكلاء غير الموثوقين
        if new_trust < 0.3 and agent_id != "expert_seed":
            logger.warning(f"⚠️ رفض خبرة من الوكيل {agent_id} بسبب انخفاض الثقة ({new_trust:.2f})")
            return

        experience = {
            "agent_id": agent_id,
            "task_type": task,
            "outcome": outcome,
            "lesson": lesson,
            "success": success,
            "trust_at_record": new_trust,
            "timestamp": time.time(),
            "status": "verified" if new_trust > 0.7 else "pending"
        }
        self.experience_db.append(experience)
        if len(self.experience_db) > 100: self.experience_db.pop(0)
        self._save_db()

    def get_relevant_lessons(self, current_task: str) -> str:
        """جلب الدروس الموثوقة فقط ذات الصلة."""
        current_keywords = set(current_task.lower().split())
        relevant = []
        for exp in self.experience_db:
            # حماية: استرجاع الخبرات الموثقة فقط (Verified) أو بذور الخبراء
            if exp.get("status") != "verified" and exp.get("agent_id") != "expert_seed":
                continue
                
            task_type = exp.get("task_type", "")
            exp_keywords = set(task_type.lower().split())
            if current_keywords.intersection(exp_keywords):
                relevant.append(exp)
        
        if not relevant: return ""
        
        summary = "\n🛡️ دروس مستفادة (موثقة - Verified Knowledge):\n"
        for exp in relevant[-5:]:
            status = "✅" if exp.get("success") else "❌"
            origin = f" [{exp.get('agent_id')}]"
            summary += f"- {status}{origin}: {exp.get('lesson')}\n"
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
