
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

    def _load_trust_scores(self) -> Dict[str, Dict[str, float]]:
        """تحميل نقاط الثقة (العامة والمجالات المتخصصة)."""
        if self.trust_scores_file.exists():
            try:
                with open(self.trust_scores_file, "r") as f:
                    data = json.load(f)
                    # ضمان أن كل مدخل هو قاموس وليس رقماً عائماً قديماً
                    sanitized = {}
                    for k, v in data.items():
                        if isinstance(v, dict):
                            sanitized[k] = v
                        else:
                            sanitized[k] = {"general": float(v)}
                    return sanitized
            except: pass
        return {}

    def get_agent_expertise(self, agent_id: str) -> Dict[str, float]:
        """جلب مجالات خبرة الوكيل ونقاط قوته فيها."""
        return self.trust_scores.get(agent_id, {"general": 0.5})

    def classify_domain(self, text: str) -> str:
        """تصنيف مجال النص (بنية تحتية، خوارزميات، برمجة، إلخ)."""
        text = text.lower()
        domains = {
            "infra": ["oom", "memory", "latency", "scaling", "server", "infra"],
            "algo": ["faiss", "hnsw", "quantization", "algorithm", "complexity", "big o"],
            "dev": ["python", "code", "def ", "class", "fix", "bug", "implement"]
        }
        for domain, keywords in domains.items():
            if any(k in text for k in keywords):
                return domain
        return "general"

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

    def evaluate_solution(self, code: str, context: str = "") -> Dict[str, Any]:
        """
        تقييم جودة وكفاءة الحل البرمجي المقترح.
        يحلل التعقيد الخوارزمي، الأمان، وكفاءة الموارد.
        """
        score = 1.0
        reasons = []
        
        # 1. تحليل التعقيد الخوارزمي (Big O)
        if "for " in code and "for " in code.split("for ", 1)[1]:
            score -= 0.2
            reasons.append("تعقيد O(n^2) محتمل بسبب الحلقات المتداخلة")
        
        if "faiss" in code or "HNSW" in code or "IndexIVFPQ" in code:
            score += 0.2
            reasons.append("استخدام فهرسة ANN عالية الكفاءة (O(log n))")
            
        # 2. تحليل الأمان
        if "eval(" in code or "exec(" in code:
            score -= 0.5
            reasons.append("مخاطر أمنية عالية: استخدام eval/exec")
            
        # 3. كفاءة الموارد
        if "batch_size=1" in code or "accumulation" in code.lower():
            score += 0.1
            reasons.append("تحسين استهلاك الذاكرة (Memory Efficiency)")
            
        if "1,000,000" in context and "quantization" not in code.lower():
            score -= 0.3
            reasons.append("الحل قد لا يتوسع لـ 1 مليون متجه بدون تكميم (Quantization)")
            
        final_score = max(0.0, min(1.0, score))
        return {
            "score": round(final_score, 2),
            "reasons": reasons,
            "approved": final_score >= 0.7
        }

    def record_experience(self, task: str, outcome: str, lesson: str, success: bool, agent_id: str = "global"):
        """تسجيل خبرة جديدة مع التحقق من الثقة والمراجعة التلقائية."""
        # 1. مراجعة المحتوى
        if not self._auto_review_lesson(lesson):
            logger.warning(f"🛡️ حظر درس ضار محتمل من الوكيل {agent_id}")
            return

        # 2. تحديث نقاط الثقة (العامة والمتخصصة)
        domain = self.classify_domain(lesson + " " + task)
        agent_trust = self.trust_scores.get(agent_id, {"general": 0.5})
        
        current_val = agent_trust.get(domain, agent_trust.get("general", 0.5))
        if success:
            new_val = min(1.0, current_val + 0.05)
        else:
            new_val = max(0.0, current_val - 0.1)
            
        agent_trust[domain] = new_val
        agent_trust["general"] = sum(agent_trust.values()) / len(agent_trust) # تحديث المعدل العام
        self.trust_scores[agent_id] = agent_trust
        self._save_trust_scores()
        
        new_trust = agent_trust["general"]

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

    def get_drift_profile(self, video_source: str) -> Optional[Dict[str, Any]]:
        """جلب ملف تعريف الانحراف الزمني لمصدر معين."""
        for exp in reversed(self.experience_db):
            if exp.get("task_type") == "drift_profile" and exp.get("source") == video_source:
                if exp.get("status") == "verified":
                    return exp.get("profile")
        return None

    def save_drift_profile(self, video_source: str, profile: Dict[str, Any], agent_id: str = "global"):
        """حفظ نمط الانحراف المكتشف كمهمة تعلم."""
        lesson = f"تم رصد انحراف بمعدل {profile.get('drift_rate', 0):.4f} في المصدر {video_source}"
        experience = {
            "agent_id": agent_id,
            "task_type": "drift_profile",
            "source": video_source,
            "profile": profile,
            "lesson": lesson,
            "success": True,
            "timestamp": time.time(),
            "status": "verified" # أنماط الانحراف التقنية تعتبر موثقة تلقائياً إذا جاءت من DriftCorrector
        }
        self.experience_db.append(experience)
        if len(self.experience_db) > 100: self.experience_db.pop(0)
        self._save_db()

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
