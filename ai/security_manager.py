
import os
import base64
import logging
import time
import json
import secrets
from cryptography.fernet import Fernet, MultiFernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from typing import Union, Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger("NeuralServiceMesh.Security")

class SecurityManager:
    """
    درع الحماية السيادي: يدير التشفير الشامل (E2EE) وتدوير المفاتيح.
    يستخدم MultiFernet لدعم فك تشفير البيانات القديمة بمفاتيح سابقة مع تشفير الجديد بأحدث مفتاح.
    """
    
    def __init__(self, master_key: Optional[str] = None, storage_dir: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        self.storage_dir = Path(storage_dir) if storage_dir else self.root / "artifacts" / "security" / "keys"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.keys_path = self.storage_dir / "key_vault.json"
        
        self.master_key = master_key or os.getenv("NSM_MASTER_KEY", "nsm-sovereign-default-secret")
        self.key_rotation_interval = 86400 * 30  # تدوير كل 30 يوم افتراضياً
        
        self.key_vault: Dict[str, str] = self._load_key_vault()
        self._ensure_active_key()
        self._update_cipher()
        
        logger.info(f"🔐 Sovereign Security Shield active. Keys in vault: {len(self.key_vault)}")

    def _load_key_vault(self) -> Dict[str, str]:
        """تحميل مخزن المفاتيح من الملف."""
        if self.keys_path.exists():
            try:
                with open(self.keys_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to load key vault: {e}")
        return {}

    def _save_key_vault(self):
        """حفظ مخزن المفاتيح بشكل آمن (يجب تشفيره في الإنتاج بمفتاح النظام)."""
        with open(self.keys_path, 'w', encoding='utf-8') as f:
            json.dump(self.key_vault, f, indent=4)

    def _generate_fernet_key(self, salt: bytes) -> str:
        """توليد مفتاح Fernet آمن باستخدام KDF."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key.encode()))
        return key.decode('utf-8')

    def _ensure_active_key(self):
        """التأكد من وجود مفتاح نشط وتدويره إذا لزم الأمر."""
        current_time = int(time.time())
        active_key_id = self.key_vault.get("active_id")
        
        should_rotate = False
        if not active_key_id:
            should_rotate = True
        else:
            last_rotation = self.key_vault.get(f"created_{active_key_id}", 0)
            if current_time - last_rotation > self.key_rotation_interval:
                should_rotate = True
        
        if should_rotate:
            self.rotate_keys()

    def rotate_keys(self):
        """توليد مفتاح جديد وجعله النشط."""
        new_id = secrets.token_hex(8)
        salt = secrets.token_bytes(16)
        new_key = self._generate_fernet_key(salt)
        
        self.key_vault[new_id] = new_key
        self.key_vault[f"salt_{new_id}"] = base64.b64encode(salt).decode('utf-8')
        self.key_vault[f"created_{new_id}"] = int(time.time())
        self.key_vault["active_id"] = new_id
        
        # الاحتفاظ بآخر 5 مفاتيح لفك التشفير
        all_ids = [k for k in self.key_vault.keys() if len(k) == 16 and not k.startswith("active")]
        if len(all_ids) > 5:
            oldest = sorted(all_ids, key=lambda x: self.key_vault.get(f"created_{x}", 0))[0]
            del self.key_vault[oldest]
            del self.key_vault[f"salt_{oldest}"]
            del self.key_vault[f"created_{oldest}"]
            
        self._save_key_vault()
        self._update_cipher()
        logger.info(f"🔄 Keys rotated. New active key ID: {new_id}")

    def _update_cipher(self):
        """تحديث كائن التشفير المتعدد."""
        active_id = self.key_vault.get("active_id")
        if not active_id: return
        
        # ترتيب المفاتيح: النشط أولاً ثم القديمة
        all_keys = [self.key_vault[active_id]]
        other_keys = [v for k, v in self.key_vault.items() if len(k) == 16 and k != active_id]
        all_keys.extend(other_keys)
        
        self.cipher = MultiFernet([Fernet(k.encode()) for k in all_keys])

    def encrypt(self, data: Union[str, bytes, Dict[str, Any]]) -> bytes:
        """تشفير البيانات باستخدام المفتاح النشط."""
        if isinstance(data, dict):
            data = json.dumps(data, ensure_ascii=False).encode('utf-8')
        elif isinstance(data, str):
            data = data.encode('utf-8')
        
        return self.cipher.encrypt(data)

    def decrypt(self, encrypted_data: bytes, as_json: bool = False) -> Union[str, Dict[str, Any]]:
        """فك تشفير البيانات (يحاول مع كل المفاتيح في المخزن)."""
        decrypted = self.cipher.decrypt(encrypted_data)
        if as_json:
            return json.loads(decrypted.decode('utf-8'))
        return decrypted.decode('utf-8')

    def encrypt_file(self, file_path: str):
        """تشفير ملف بالكامل."""
        with open(file_path, 'rb') as f:
            data = f.read()
        encrypted = self.encrypt(data)
        with open(file_path, 'wb') as f:
            f.write(encrypted)

    def decrypt_file(self, file_path: str) -> bytes:
        """فك تشفير ملف."""
        with open(file_path, 'rb') as f:
            encrypted_data = f.read()
        return self.cipher.decrypt(encrypted_data)

# نسخة عالمية للاستخدام في المشروع
security_manager = SecurityManager()
