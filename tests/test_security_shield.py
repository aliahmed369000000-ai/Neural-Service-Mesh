
import sys
import os
import unittest
import time
import json
from pathlib import Path

# إضافة المسار الحالي للاستيراد
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestSecurityShield(unittest.TestCase):
    def setUp(self):
        from ai.security_manager import SecurityManager
        self.test_dir = Path("/tmp/nsm_test_keys")
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.sm = SecurityManager(master_key="test-secret", storage_dir=str(self.test_dir))

    def test_encryption_decryption(self):
        """اختبار التشفير وفك التشفير الأساسي."""
        secret_msg = "رسالة سرية للغاية من السرب"
        encrypted = self.sm.encrypt(secret_msg)
        decrypted = self.sm.decrypt(encrypted)
        self.assertEqual(secret_msg, decrypted)

    def test_key_rotation(self):
        """اختبار تدوير المفاتيح والقدرة على فك تشفير البيانات القديمة."""
        msg1 = "بيانات قديمة"
        encrypted1 = self.sm.encrypt(msg1)
        
        old_active_id = self.sm.key_vault["active_id"]
        
        # تدوير المفاتيح
        self.sm.rotate_keys()
        new_active_id = self.sm.key_vault["active_id"]
        
        self.assertNotEqual(old_active_id, new_active_id)
        
        # فك تشفير البيانات القديمة بالمفتاح الجديد (عبر MultiFernet)
        decrypted1 = self.sm.decrypt(encrypted1)
        self.assertEqual(msg1, decrypted1)
        
        # تشفير بيانات جديدة وفكها
        msg2 = "بيانات جديدة"
        encrypted2 = self.sm.encrypt(msg2)
        self.assertEqual(msg2, self.sm.decrypt(encrypted2))

    def tearDown(self):
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main()
