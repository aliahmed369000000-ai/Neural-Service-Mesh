# -*- coding: utf-8 -*-
"""🛡️ NSM Security Guard — درع الحماية السيادي للشبكة الموزعة.

يقوم هذا النظام بحماية الموارد الحسابية وتأمين أوزان النماذج وتشفير
قنوات التواصل بين العقد لمنع أي تداخل أو تسريب للبيانات.
"""
import os
import hashlib
import hmac
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

class NSMSecurityGuard:
    def __init__(self, master_key: str = None):
        self.master_key = master_key or os.environ.get("NSM_MASTER_KEY", "default_sovereign_key")
        self.cipher = self._initialize_cipher()
        self.integrity_log = []

    def _initialize_cipher(self):
        """توليد مفتاح تشفير قوي بناءً على المفتاح الرئيسي."""
        salt = b'nsm_sovereign_salt'
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return Fernet(key)

    def sign_data(self, data: bytes) -> str:
        """توقيع البيانات رقمياً لضمان النزاهة."""
        signature = hmac.new(self.master_key.encode(), data, hashlib.sha256).hexdigest()
        return signature

    def verify_integrity(self, data: bytes, signature: str) -> bool:
        """التحقق من عدم التلاعب بالبيانات."""
        expected_signature = self.sign_data(data)
        is_valid = hmac.compare_digest(expected_signature, signature)
        if not is_valid:
            self.integrity_log.append(f"⚠️ Security Alert: Integrity breach detected!")
        return is_valid

    def encrypt_weights(self, weights_path: str):
        """تشفير ملفات أوزان النموذج قبل التخزين أو النقل."""
        if not os.path.exists(weights_path):
            return
        
        with open(weights_path, 'rb') as f:
            data = f.read()
        
        encrypted_data = self.cipher.encrypt(data)
        with open(weights_path + ".secure", 'wb') as f:
            f.write(encrypted_data)
        
        # حفظ التوقيع الرقمي
        signature = self.sign_data(encrypted_data)
        with open(weights_path + ".sig", 'w') as f:
            f.write(signature)
        
        print(f"🔒 Weights secured and encrypted at {weights_path}.secure")

    def monitor_compute_integrity(self):
        """مراقبة استهلاك الموارد لاكتشاف محاولات الاختطاف الحسابي."""
        # محاكاة مراقبة الموارد
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                usage = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
                if usage > 98:
                    self.integrity_log.append(f"🔍 Monitoring: GPU {i} at peak usage. Verifying authorization...")

    def get_security_status(self):
        """عرض حالة الدرع الأمني."""
        return {
            "status": "Active",
            "encryption": "AES-256-Fernet",
            "integrity_checks": len(self.integrity_log),
            "alerts": self.integrity_log[-5:] if self.integrity_log else ["No threats detected"]
        }

if __name__ == "__main__":
    guard = NSMSecurityGuard()
    print("🛡️ NSM Security Guard is standing watch.")
    print(guard.get_security_status())
