
import sys
import os
import numpy as np
import unittest

# إضافة المسار الحالي للاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSurahUnification(unittest.TestCase):
    def test_surah_config(self):
        """التحقق من أن Surah 4096 هي الإعداد الافتراضي."""
        from ai.arabic_transformer import ArabicTransformer, D_MODEL, N_HEADS
        
        # التحقق من الثوابت
        self.assertEqual(D_MODEL, 4096, "يجب أن يكون d_model الافتراضي 4096")
        self.assertEqual(N_HEADS, 32, "يجب أن يكون n_heads الافتراضي 32")
        
        # التحقق من الكائن
        model = ArabicTransformer()
        self.assertEqual(model.embedding.d_model, 4096)
        self.assertTrue(model.use_dte, "يجب تفعيل DTE افتراضياً")
        self.assertTrue(hasattr(model, 'dte'), "يجب وجود محرك DTE")
        
    def test_reasoning_pipeline_no_moe(self):
        """التحقق من تعطيل MoE في مسار الإجابة."""
        from ai.reasoning_pipeline import ReasoningPipeline
        
        pipeline = ReasoningPipeline(use_moe=False) # التأكد من التعطيل الصريح أو الافتراضي
        self.assertFalse(pipeline.use_moe, "يجب أن يكون MoE معطلاً في Pipeline")
        self.assertIsNone(pipeline._moe_bridge, "يجب ألا يتم إنشاء جسر MoE")

if __name__ == "__main__":
    unittest.main()
