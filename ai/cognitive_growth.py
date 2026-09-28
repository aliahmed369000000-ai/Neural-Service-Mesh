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

    def analyze_experiences(self) -> Dict[str, Any]:
        """يحلّل الخبرات المسجّلة (سجلات لها task_type/success) ويستخرج
        معدل النجاح وأكثر أنواع المهام فشلاً والدروس المتكررة. تُتجاهل
        سجلات التطور (event=evolution) وأي سجل بلا task_type."""
        tasks = [
            e for e in self.knowledge_base
            if isinstance(e, dict) and "task_type" in e and "success" in e
        ]
        total = len(tasks)
        successes = sum(1 for e in tasks if e.get("success"))
        top_failures: Dict[str, int] = {}
        failure_lessons: Dict[str, List[str]] = {}
        for e in tasks:
            if not e.get("success"):
                t = str(e.get("task_type"))
                top_failures[t] = top_failures.get(t, 0) + 1
                lesson = e.get("lesson")
                if lesson and lesson not in failure_lessons.setdefault(t, []):
                    failure_lessons[t].append(str(lesson))
        top_failures = dict(
            sorted(top_failures.items(), key=lambda kv: kv[1], reverse=True)
        )
        self.last_analysis_time = time.time()
        return {
            "total_tasks": total,
            "success_rate": (successes / total) if total else 0.0,
            "top_failures": top_failures,
            "failure_lessons": failure_lessons,
        }

    def evolve_strategies(self) -> Dict[str, Any]:
        """يشتق استراتيجيات من تحليل الخبرات: memory_safety (عند تكرار
        أخطاء الذاكرة/OOM في الدروس) وtask_routing (تنبيه لكل نوع مهمة
        يفشل). يُخزَّن الناتج في self.strategies ويُرجَع."""
        analysis = self.analyze_experiences()
        memory_markers = ("oom", "out of memory", "memory", "batch size", "ذاكرة")
        memory_hits = sum(
            1
            for lessons in analysis["failure_lessons"].values()
            for lesson in lessons
            if any(m in lesson.lower() for m in memory_markers)
        )
        strategies: Dict[str, Any] = {
            "memory_safety": {
                "enabled": memory_hits > 0,
                "action": "reduce_batch_size_and_checkpoint" if memory_hits else "none",
                "evidence_count": memory_hits,
            },
            "task_routing": {
                t: {"failures": n, "action": "verify_resources_before_run"}
                for t, n in analysis["top_failures"].items()
            },
        }
        self.strategies = strategies
        return strategies

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
        report += f"إجمالي الخبرات: {analysis['total_experiences']}\n"
        
        if not self.strategies:
            self.evolve_strategies()
        report += "الاستراتيجيات المشتقة: " + ", ".join(self.strategies) + "\n"

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
