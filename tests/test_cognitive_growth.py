
import unittest
import os
import json
import sys
from pathlib import Path

# إضافة المجلد الأب إلى sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.cognitive_growth import CognitiveGrowthEngine

class TestCognitiveGrowth(unittest.TestCase):
    def setUp(self):
        self.temp_db = "tests/temp_experience_db.json"
        self.sample_data = [
            {"task_type": "Kaggle Training", "success": False, "lesson": "OOM error, reduce batch size"},
            {"task_type": "Kaggle Training", "success": False, "lesson": "OOM error, reduce batch size"},
            {"task_type": "Data Processing", "success": True, "lesson": "MinHash works well"}
        ]
        with open(self.temp_db, 'w', encoding='utf-8') as f:
            json.dump(self.sample_data, f)
        
        self.engine = CognitiveGrowthEngine(db_path=self.temp_db)

    def tearDown(self):
        if os.path.exists(self.temp_db):
            os.remove(self.temp_db)

    def test_analysis(self):
        analysis = self.engine.analyze_experiences()
        self.assertEqual(analysis["total_tasks"], 3)
        self.assertAlmostEqual(analysis["success_rate"], 1/3)
        self.assertIn("Kaggle Training", analysis["top_failures"])

    def test_evolution(self):
        strategies = self.engine.evolve_strategies()
        self.assertIn("memory_safety", strategies)
        self.assertIn("task_routing", strategies)
        self.assertTrue("Kaggle Training" in strategies["task_routing"])

    def test_report(self):
        report = self.engine.get_growth_report()
        self.assertIn("إجمالي الخبرات: 3", report)
        self.assertIn("memory_safety", report)

if __name__ == "__main__":
    unittest.main()
