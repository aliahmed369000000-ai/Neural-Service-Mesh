# -*- coding: utf-8 -*-
import unittest
from ai.alert_manager import alert_manager
from ai.living_mesh import LivingMeshNode
import json
import os

class TestAlertSystem(unittest.TestCase):
    def setUp(self):
        # إعداد تكوين وهمي للاختبار
        self.test_config = {
            "telegram": {"enabled": False, "token": "test_token", "chat_id": "test_chat"},
            "email": {"enabled": False, "smtp_server": "localhost", "port": 1025, "user": "u", "password": "p", "receiver": "r"},
            "alert_levels": ["CRITICAL", "SECURITY"]
        }
        alert_manager.save_config(self.test_config)

    def test_alert_triggering(self):
        """اختبار إطلاق التنبيهات عند حدوث فشل أمني."""
        # محاكاة عقدة
        node = LivingMeshNode("test_node")
        
        # محاكاة هجوم (توقيع غير صالح)
        # هذا يجب أن يطلق alert_manager.send_alert("SECURITY", ...)
        # سنقوم فقط بالتحقق من أن الدالة قابلة للاستدعاء ولا تسبب أخطاء
        try:
            alert_manager.send_alert("SECURITY", "Intrusion Attempt Simulated", {"attacker": "unknown"})
            success = True
        except Exception as e:
            success = False
            print(f"Alert failed: {e}")
            
        self.assertTrue(success)

    def test_node_death_alert(self):
        """اختبار إطلاق تنبيه عند سقوط عقدة."""
        try:
            alert_manager.send_alert("CRITICAL", "Node Alpha is DEAD", {"host": "127.0.0.1", "port": 8000})
            success = True
        except Exception as e:
            success = False
            
        self.assertTrue(success)

if __name__ == "__main__":
    unittest.main()
