
import sys
import os
import unittest
import time
import json
import base64

# إضافة المسار الحالي للاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class SecurityStressTest(unittest.TestCase):
    def setUp(self):
        from ai.swarm_manager import SwarmManager
        self.swarm = SwarmManager(storage_dir="/tmp/nsm_stress_swarm")
        self.swarm.register_worker("malicious_agent", role="observer", trust_score=0.1)
        self.swarm.register_worker("sovereign_agent", role="sovereign", trust_score=0.99)

    def test_unauthorized_access_attempt(self):
        """اختبار محاولة وصول غير مصرح بها من وكيل ذو ثقة منخفضة."""
        # محاولة حذف (صلاحية sovereign فقط)
        res = self.swarm.check_permission("malicious_agent", "delete")
        self.assertFalse(res, "يجب رفض محاولة الحذف لوكيل observer")

    def test_ids_pattern_detection(self):
        """اختبار اكتشاف الأنماط المشبوهة عبر IDS."""
        # محاولة تنفيذ أمر مشبوه
        params = {"command": "rm -rf /"}
        res = self.swarm.check_permission("malicious_agent", "write", params=params)
        self.assertFalse(res, "يجب حظر الأوامر التي تحتوي على أنماط تخريبية")
        self.assertTrue(self.swarm.ids.is_quarantined("malicious_agent"), "يجب وضع الوكيل في الحجر الصحي")

    def test_ids_entropy_detection(self):
        """اختبار اكتشاف الأوامر المشوهة (Obfuscated) عبر تحليل العشوائية."""
        self.swarm.register_worker("hacker_agent", role="worker", trust_score=0.6)
        
        # أمر مشوه (Base64) قد يمثل التفافاً
        obfuscated_action = "ZXZhbCh1bmhleCgnNzI2ZDYxMmQ3MjY2MmYyNycpKQ==" 
        params = {"payload": obfuscated_action}
        
        # سنقوم بتمرير النص المشوه كـ action لمحاكاة التحليل
        res = self.swarm.ids.monitor_action("hacker_agent", obfuscated_action, params)
        self.assertIn("High entropy action detected", str(res.get("alerts", [])))

    def tearDown(self):
        import shutil
        if os.path.exists("/tmp/nsm_stress_swarm"):
            shutil.rmtree("/tmp/nsm_stress_swarm")

if __name__ == "__main__":
    unittest.main()
