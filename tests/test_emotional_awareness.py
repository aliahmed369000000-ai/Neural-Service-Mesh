
import sys
import os
import time
import unittest

# إضافة المسار الحالي للاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.swarm_manager import SwarmManager
from ai.emotional_awareness import CollectiveMood, EmotionalAwarenessEngine

class TestEmotionalAwareness(unittest.TestCase):
    def setUp(self):
        self.manager = SwarmManager()
        self.manager.emotional_engine = EmotionalAwarenessEngine()
        self.manager.register_worker("agent_1", "worker", trust_score=0.8)
        self.manager.register_worker("agent_2", "worker", trust_score=0.8)

    def test_mood_shift_to_harmony(self):
        """اختبار تحول المزاج إلى الانسجام عند توالي النجاحات."""
        for i in range(10):
            self.manager.report_result("agent_1", f"task_{i}", "success", success=True)
        
        self.manager._sync_emotional_state()
        mood = self.manager.emotional_engine.current_mood
        self.assertEqual(mood, CollectiveMood.HARMONY)
        self.assertEqual(self.manager.threshold, 0.51)

    def test_mood_shift_to_conflict(self):
        """اختبار تحول المزاج إلى النزاع عند الفشل المتكرر."""
        for i in range(10):
            self.manager.report_result("agent_1", f"fail_task_{i}", "error", success=False)
        
        self.manager.ids.alert_level = 0.8
        self.manager._sync_emotional_state()
        mood = self.manager.emotional_engine.current_mood
        
        self.assertIn(mood, [CollectiveMood.STRESS, CollectiveMood.CONFLICT])
        self.assertGreaterEqual(self.manager.threshold, 0.8)

    def test_adaptive_consensus_direct(self):
        """اختبار تكيف عتبة التوافق بناءً على تغيير المزاج يدوياً مع تعطيل التحديث التلقائي."""
        # محاكاة حالة استقرار عبر تصفير النتائج
        self.manager.results = []
        self.manager._sync_emotional_state()
        # بما أن النجاح 100% والثقة 0.8، سيصبح HARMONY تلقائياً في الإعدادات الجديدة
        # لذا سنختبر القيم المتكيفة مباشرة
        
        moods_to_test = [
            (CollectiveMood.STABILITY, 0.66),
            (CollectiveMood.HARMONY, 0.51),
            (CollectiveMood.STRESS, 0.8),
            (CollectiveMood.CONFLICT, 0.9)
        ]
        
        for mood, expected_threshold in moods_to_test:
            self.manager.emotional_engine.current_mood = mood
            # بدلاً من _sync_emotional_state التي تعيد حساب المزاج، نطبق البارامترات فقط
            params = self.manager.emotional_engine.get_adaptive_params()
            self.manager.threshold = params["consensus_threshold"]
            self.assertEqual(self.manager.threshold, expected_threshold, f"Failed for mood {mood}")

if __name__ == "__main__":
    unittest.main()
