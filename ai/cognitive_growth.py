
import json
import os
import time
import logging
import numpy as np
from typing import List, Dict, Any, Optional

logger = logging.getLogger("NeuralServiceMesh.CognitiveGrowth")

class CognitiveGrowthEngine:
    """
    محرك النمو المعرفي الذاتي (SCG):
    يسمح للسرب بتحليل الخبرات المتراكمة، استخلاص الأنماط، وتوليد استراتيجيات عمل جديدة.
    """
    def __init__(self, db_path: str = "artifacts/learning/experience_db.json"):
        self.db_path = db_path
        self.knowledge_base = []
        self.strategies = {}
        self.last_analysis_time = 0
        self._load_db()

    def _load_db(self):
        if os.path.exists(self.db_path):
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    self.knowledge_base = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load experience DB: {e}")
                self.knowledge_base = []

    def analyze_experiences(self):
        """تحليل الخبرات لاستخلاص الأنماط والدروس المستفادة."""
        if not self.knowledge_base:
            return "قاعدة الخبرات فارغة حالياً."

        analysis = {
            "total_tasks": len(self.knowledge_base),
            "success_rate": sum(1 for x in self.knowledge_base if x.get("success")) / len(self.knowledge_base),
            "top_failures": {},
            "key_lessons": []
        }

        for exp in self.knowledge_base:
            if not exp.get("success"):
                task_type = exp.get("task_type", "unknown")
                analysis["top_failures"][task_type] = analysis["top_failures"].get(task_type, 0) + 1
            
            lesson = exp.get("lesson")
            if lesson and lesson not in analysis["key_lessons"]:
                analysis["key_lessons"].append(lesson)

        self.last_analysis_time = time.time()
        return analysis

    def evolve_strategies(self):
        """توليد استراتيجيات جديدة بناءً على التحليل المعرفي."""
        analysis = self.analyze_experiences()
        if isinstance(analysis, str): return

        # استراتيجية إدارة الذاكرة
        if analysis["success_rate"] < 0.7:
            self.strategies["memory_safety"] = "تفعيل التكميم (VQ) الصارم وتقليل حجم الدفعة تلقائياً."
        
        # استراتيجية توزيع المهام
        failure_types = sorted(analysis["top_failures"].items(), key=lambda x: x[1], reverse=True)
        if failure_types:
            most_failed = failure_types[0][0]
            self.strategies["task_routing"] = f"توجيه مهام {most_failed} إلى الوكلاء ذوي الثقة > 0.9 فقط."

        logger.info(f"🧠 Cognitive Evolution: {len(self.strategies)} new strategies evolved.")
        return self.strategies

    def get_growth_report(self) -> str:
        """تقرير عن حالة النمو المعرفي للسرب."""
        self.evolve_strategies()
        analysis = self.analyze_experiences()
        if isinstance(analysis, str): return analysis

        report = f"--- تقرير النمو المعرفي الذاتي ---\n"
        report += f"إجمالي الخبرات: {analysis['total_tasks']}\n"
        report += f"معدل النجاح العام: {analysis['success_rate']:.2%}\n"
        report += f"الاستراتيجيات النشطة: {len(self.strategies)}\n"
        for name, desc in self.strategies.items():
            report += f"- {name}: {desc}\n"
        
        return report

# نسخة عالمية للنمو
cognitive_engine = CognitiveGrowthEngine()
