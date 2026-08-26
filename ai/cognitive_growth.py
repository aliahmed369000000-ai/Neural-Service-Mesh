# -*- coding: utf-8 -*-
import json
import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("NeuralServiceMesh.CognitiveGrowth")

class CognitiveGrowthEngine:
    """
    🧠 NSM Cognitive Growth Engine — محرك النمو المعرفي الذاتي (Kaggle Edition).
    يسمح للسرب بتحليل الخبرات، استخلاص الأنماط، واقتراح تطورات هيكلية للنماذج.
    """
    def __init__(self, db_path: str = None):
        # التوافق مع مسارات Kaggle للخبرات
        if db_path is None:
            if os.path.exists("/kaggle/working"):
                self.db_path = "/kaggle/working/experience_db.json"
            else:
                self.db_path = "artifacts/learning/experience_db.json"
        else:
            self.db_path = db_path
            
        self.knowledge_base = []
        self.strategies = {}
        self.evolution_steps = []
        self.last_analysis_time = 0
        self._load_db()
        logger.info(f"🧠 Cognitive Growth Engine Initialized. DB: {self.db_path}")

    def _load_db(self):
        db_dir = os.path.dirname(self.db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load experience DB: {e}")
                self.knowledge_base = []

    def analyze_learning_trend(self, loss_history: List[float]) -> str:
        """تحليل اتجاه التعلم لاكتشاف فرص التطور الهيكلي."""
        if len(loss_history) < 10:
            return "Gathering more data for trend analysis..."
        
        recent_loss = loss_history[-10:]
        trend = np.polyfit(range(len(recent_loss)), recent_loss, 1)[0]
        
        if trend > 0:
            return "⚠️ Learning Stalled: Suggesting structural evolution."
        elif trend > -0.0001:
            return "📉 Slow Convergence: Optimizing attention complexity."
        return "✅ Healthy Growth: Maintaining current trajectory."

    def propose_structural_evolution(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """اقتراح تغييرات هيكلية بناءً على مقاييس الأداء."""
        proposals = []
        if metrics.get("accuracy", 0) < 0.8 and metrics.get("loss_variance", 0) > 0.4:
            proposals.append({
                "type": "add_residual_connection",
                "reason": "Improving gradient stability",
                "impact": "Better convergence for deep layers"
            })
        return proposals

    def apply_evolutionary_patch(self, proposal: Dict[str, Any]):
        """توثيق تطبيق التطور الهيكلي وحفظه محلياً."""
        step = {
            "step_id": len(self.evolution_steps) + 1,
            "type": proposal["type"],
            "reason": proposal["reason"],
            "timestamp": time.time()
        }
        self.evolution_steps.append(step)
        # حفظ التطور في قاعدة البيانات المحلية
        self.knowledge_base.append({"event": "evolution", "details": step})
        with open(self.db_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, ensure_ascii=False, indent=4)
            
        logger.info(f"✨ Evolutionary Step Recorded: {proposal['type']}")
        return step

    def get_growth_report(self) -> str:
        """تقرير شامل عن حالة النمو المعرفي والتطور الذاتي."""
        analysis = {
            "total_experiences": len(self.knowledge_base),
            "evolution_steps": len(self.evolution_steps),
            "intelligence_index": 1.0 + (len(self.evolution_steps) * 0.1)
        }
        
        report = f"--- 🧠 تقرير النمو المعرفي السيادي (Kaggle) ---\n"
        report += f"مؤشر الذكاء الحالي: {analysis['intelligence_index']:.2f}\n"
        report += f"خطوات التطور المنفذة: {analysis['evolution_steps']}\n"
        report += f"إجمالي الخبرات المكتسبة: {analysis['total_experiences']}\n"
        
        if self.evolution_steps:
            report += "أحدث قفزات التطور:\n"
            for step in self.evolution_steps[-3:]:
                report += f"- [{step['type']}]: {step['reason']}\n"
        
        return report

    def push_evolution_to_github(self):
        """رفع تقرير التطور المعرفي إلى المستودع كإنجاز سيادي."""
        from ai.git_manager import GitManager
        git = GitManager()
        
        report = self.get_growth_report()
        repo_path = git.clone("cognitive_evolution_push")
        
        log_path = os.path.join(repo_path, "docs/EVOLUTION_LOG.md")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n## 🧠 Cognitive Update (Kaggle) - {time.ctime()}\n")
            f.write(report)
            
        git.commit_and_push(repo_path, "🧬 NSM Bot: Recording Cognitive Growth Evolution from Kaggle", [log_path])
        return "✅ Evolution recorded and pushed to GitHub."

# نسخة عالمية للنمو
cognitive_engine = CognitiveGrowthEngine()
