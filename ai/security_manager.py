
import os
import base64
import logging
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, Dict, Any
import json

logger = logging.getLogger("NeuralServiceMesh.Security")

class SecurityManager:
    """مدير الأمن: يفرز مفاتيح التشفير ويؤمن البيانات السيادية."""
    
    def __init__(self, master_key: str = "nsm-sovereign-default-secret"):
        self.key = self._generate_key(master_key)
        self.cipher = Fernet(self.key)
        logger.info("🔐 Security Manager initialized with E2EE capabilities.")

    def _generate_key(self, password: str) -> bytes:
        """توليد مفتاح تشفير ثابت من كلمة مرور (Master Key)."""
        salt = b'nsm_sovereign_salt' # في الإنتاج يجب أن يكون فريداً ومخزناً
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt(self, data: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """تشفير البيانات (نص، بايتات، أو قاموس)."""
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.cipher.encrypt(data)

    def decrypt(self, encrypted_data: bytes, as_json: bool = False) -> Union[str, Dict[str, Any]]:
        """فك تشفير البيانات."""
        decrypted = self.cipher.decrypt(encrypted_data)
        if as_json:
            return json.loads(decrypted.decode('utf-8'))
        return decrypted.decode('utf-8')

    def encrypt_file(self, file_path: str):
        """تشفير ملف بالكامل في مكانه."""
        with open(file_path, 'rb') as f:
            data = f.read()
        encrypted = self.encrypt(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted)
        logger.debug(f"🔒 File encrypted: {file_path}")

    def decrypt_file(self, file_path: str) -> bytes:
        """فك تشفير ملف وإرجاع محتواه."""
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        return self.cipher.decrypt(encrypted_data)

# نسخة عالمية للاستخدام في المشروع
security_manager = SecurityManager(os.getenv("NSM_MASTER_KEY", "nsm-sovereign-default-secret"))
