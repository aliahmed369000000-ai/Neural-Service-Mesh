
import json
import os
import time
import logging
from typing import Any, Dict, Optional
from datetime import datetime

logger = logging.getLogger("NSM.Persistence")

class PersistenceManager:
    """مدير استمرارية الحالة: يقوم بحفظ واستعادة 'وعي' الوكيل بالكامل."""
    
    def __init__(self, base_dir: str = "artifacts/agent_sleep"):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def save_snapshot(self, agent_id: str, state_data: Dict[str, Any]) -> str:
        """حفظ لقطة كاملة لحالة الوكيل مع التشفير السيادي."""
        from ai.security_manager import security_manager
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_id}_{timestamp}.enc"
        filepath = os.path.join(self.base_dir, filename)
        
        snapshot = {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "unix_time": time.time(),
            "state": state_data,
            "version": "1.0",
            "encrypted": True
        }
        
        try:
            # تشفير لقطة الوعي قبل الحفظ
            encrypted_data = security_manager.encrypt(snapshot)
            with open(filepath, 'wb') as f:
                f.write(encrypted_data)
            
            # تحديث رابط 'الأحدث' (Latest)
            latest_path = os.path.join(self.base_dir, f"{agent_id}_latest.enc")
            with open(latest_path, 'wb') as f:
                f.write(encrypted_data)
                
            logger.info(f"💾 تم حفظ لقطة الوعي المشفرة للوكيل {agent_id} في {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ فشل حفظ لقطة الوعي المشفرة: {e}")
            raise

    def load_snapshot(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """استعادة آخر لقطة حالة مشفرة للوكيل."""
        from ai.security_manager import security_manager
        latest_path = os.path.join(self.base_dir, f"{agent_id}_latest.enc")
        
        if not os.path.exists(latest_path):
            logger.warning(f"⚠️ لا توجد لقطات حالة مشفرة محفوظة للوكيل {agent_id}")
            return None
            
        try:
            with open(latest_path, 'rb') as f:
                encrypted_data = f.read()
            
            snapshot = security_manager.decrypt(encrypted_data, as_json=True)
            logger.info(f"🔄 تم استعادة وعي الوكيل {agent_id} من لقطة مشفرة بتاريخ {snapshot['timestamp']}")
            return snapshot["state"]
        except Exception as e:
            logger.error(f"❌ فشل استعادة لقطة الوعي المشفرة: {e}")
            return None

    def list_snapshots(self, agent_id: str) -> list:
        """سرد جميع اللقطات المتاحة لوكيل معين."""
        import glob
        pattern = os.path.join(self.base_dir, f"{agent_id}_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)

# نسخة عالمية للمدير
persistence_manager = PersistenceManager()
