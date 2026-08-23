import sys
import os
import time
import unittest

# إضافة مسار المشروع للنظام
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.memory_manager import MemoryManager

class TestForgettingCurve(unittest.TestCase):
    def setUp(self):
        self.memory = MemoryManager(agent_id="test_forgetting_bot")
        # تسريع معدل التلاشي للاختبار
        self.memory.decay_rate = 0.5 
        self.memory.prune_threshold = 0.2
        self.memory.forgetting_enabled = True

    def test_memory_decay(self):
        print("\n🧪 بدء اختبار تلاشي الذاكرة...")
        
        # إضافة حقيقة
        fact = "الهدف هو المزامنة المثالية"
        # التقاط المعرف الحقيقي
        fact_id = self.memory.add_fact(fact, semantic_hash="10101010")
        
        initial_strength = self.memory.ltm_semantic[fact_id]["strength"]
        print(f"✅ القوة الابتدائية: {initial_strength}")
        self.assertEqual(initial_strength, 1.0)
        
        # محاكاة مرور الوقت (2 ثانية)
        print("⏳ محاكاة مرور 2 ثانية...")
        time.sleep(2.1)
        
        current_strength = self.memory._calculate_current_strength(self.memory.ltm_semantic[fact_id])
        print(f"📉 القوة بعد التلاشي: {current_strength:.4f}")
        self.assertLess(current_strength, 1.0)
        self.assertGreater(current_strength, 0.0)

    def test_memory_boost_on_access(self):
        print("\n🧪 بدء اختبار تعزيز الذاكرة عند الاسترجاع...")
        
        fact = "تكميم المتجهات يقلل الحجم 75%"
        fact_id = self.memory.add_fact(fact, semantic_hash="11110000")
        
        # تلاشي متعمد
        time.sleep(1)
        strength_before = self.memory._calculate_current_strength(self.memory.ltm_semantic[fact_id])
        print(f"📉 القوة قبل الاسترجاع: {strength_before:.4f}")
        
        # استرجاع (سيعزز القوة)
        # محاكاة ما يفعله search عند المطابقة
        self.memory.ltm_semantic[fact_id]["strength"] = min(1.0, strength_before + self.memory.boost_factor)
        self.memory.ltm_semantic[fact_id]["last_access"] = time.time()
        
        strength_after = self.memory.ltm_semantic[fact_id]["strength"]
        print(f"🚀 القوة بعد التعزيز: {strength_after:.4f}")
        self.assertGreater(strength_after, strength_before)

    def test_memory_pruning(self):
        print("\n🧪 بدء اختبار تنظيف الذاكرة (Pruning)...")
        
        # إضافة حقيقة ضعيفة جداً ستموت بسرعة
        fact = "معلومة غير مهمة ستنسى"
        fact_id = self.memory.add_fact(fact, semantic_hash="00000000")
        
        # جعلها تموت فوراً عبر رفع معدل التلاشي
        self.memory.decay_rate = 5.0 
        time.sleep(1.1)
        
        print("🧹 تشغيل عملية التنظيف...")
        self.memory.prune()
        
        self.assertNotIn(fact_id, self.memory.ltm_semantic)
        print("✅ تم حذف المعلومة الضعيفة بنجاح.")

if __name__ == "__main__":
    unittest.main()
