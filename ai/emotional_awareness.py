
import time
import numpy as np
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger("NeuralServiceMesh.EmotionalAwareness")

class CollectiveMood:
    STABILITY = "Stability"    # حالة استقرار، نجاحات متتالية
    STRESS = "Stress"          # ضغط عالي، فشل مهام، تنبيهات أمنية
    HARMONY = "Harmony"        # تعاون عالي، توافق سريع
    CONFLICT = "Conflict"      # اختلاف في التصويت، انخفاض الثقة
    UNCERTAINTY = "Uncertainty" # نقص البيانات، غياب الوكلاء

class EmotionalAwarenessEngine:
    """
    محرك الوعي العاطفي الجماعي:
    يقوم بتحليل الحالة النفسية والهيكلية للسرب بناءً على الأنشطة والبيانات الحيوية.
    """
    def __init__(self):
        self.current_mood = CollectiveMood.STABILITY
        self.mood_score = 0.8  # [0.0 - 1.0] حيث 1.0 هو قمة الاستقرار
        self.history: List[Dict[str, Any]] = []
        self.last_update = time.time()
        
        # المعايير الحيوية
        self.metrics = {
            "success_rate": 1.0,
            "trust_average": 0.8,
            "alert_level": 0.0,
            "consensus_speed": 1.0
        }

    def update_mood(self, swarm_data: Dict[str, Any]):
        """تحديث الحالة العاطفية بناءً على بيانات السرب الحالية."""
        # 1. تحليل معدل النجاح (آخر 10 نتائج)
        results = swarm_data.get("results", [])[-10:]
        if results:
            successes = sum(1 for r in results if r.get("success", False))
            self.metrics["success_rate"] = successes / len(results)
        
        # 2. تحليل متوسط الثقة
        workers = swarm_data.get("workers", {})
        if workers:
            trust_scores = [w.get("trust_score", 0.5) for w in workers.values()]
            self.metrics["trust_average"] = float(np.mean(trust_scores))
        
        # 3. تحليل مستوى التنبيهات (IDS)
        self.metrics["alert_level"] = swarm_data.get("alert_level", 0.0)
        
        # 4. حساب النتيجة الإجمالية للمزاج
        # معادلة محسنة: (النجاح * 0.5) + (الثقة * 0.5) - (التنبيهات * 0.4)
        new_score = (self.metrics["success_rate"] * 0.5) + \
                    (self.metrics["trust_average"] * 0.5) - \
                    (self.metrics["alert_level"] * 0.4)
        
        self.mood_score = float(np.clip(new_score, 0.0, 1.0))
        
        # 5. تحديد التسمية العاطفية
        old_mood = self.current_mood
        if self.metrics["alert_level"] > 0.6:
            self.current_mood = CollectiveMood.STRESS
        elif self.mood_score > 0.8:
            self.current_mood = CollectiveMood.HARMONY
        elif self.mood_score > 0.5:
            self.current_mood = CollectiveMood.STABILITY
        elif self.mood_score > 0.2:
            self.current_mood = CollectiveMood.UNCERTAINTY
        else:
            self.current_mood = CollectiveMood.CONFLICT
            
        if old_mood != self.current_mood:
            logger.info(f"🎭 Collective Mood Shift: {old_mood} -> {self.current_mood} (Score: {self.mood_score:.2f})")
            
        self.history.append({
            "mood": self.current_mood,
            "score": self.mood_score,
            "ts": time.time()
        })
        self.last_update = time.time()

    def get_adaptive_params(self) -> Dict[str, Any]:
        """الحصول على بارامترات النظام المتكيفة بناءً على المزاج الحالي."""
        params = {
            "consensus_threshold": 0.66,
            "heartbeat_interval": 20,
            "exploration_rate": 0.1
        }
        
        if self.current_mood == CollectiveMood.STRESS:
            params["consensus_threshold"] = 0.8  # تشديد الرقابة عند الضغط
            params["heartbeat_interval"] = 10    # تسريع نبضات القلب للمراقبة
            params["exploration_rate"] = 0.01   # تقليل المخاطرة
        elif self.current_mood == CollectiveMood.HARMONY:
            params["consensus_threshold"] = 0.51 # تسهيل التوافق عند الانسجام
            params["exploration_rate"] = 0.3    # زيادة الابتكار
        elif self.current_mood == CollectiveMood.CONFLICT:
            params["consensus_threshold"] = 0.9  # إغلاق صارم عند النزاع
            
        return params

    def get_mood_report(self) -> str:
        """تقرير وصفي للحالة العاطفية للسرب."""
        descriptions = {
            CollectiveMood.STABILITY: "السرب يعمل بكفاءة وتوازن.",
            CollectiveMood.STRESS: "هناك ضغط خارجي أو تهديدات أمنية، السرب في حالة تأهب.",
            CollectiveMood.HARMONY: "انسجام تام بين الوكلاء، الابتكار في أقصى مستوياته.",
            CollectiveMood.CONFLICT: "هناك تعارض في المصالح أو انخفاض حاد في الثقة.",
            CollectiveMood.UNCERTAINTY: "حالة من الغموض، الوكلاء يحتاجون لمزيد من البيانات."
        }
        return f"الحالة العاطفية: {self.current_mood}. {descriptions.get(self.current_mood)}"

# نسخة عالمية
emotional_engine = EmotionalAwarenessEngine()
