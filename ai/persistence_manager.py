
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
        """حفظ لقطة كاملة لحالة الوكيل."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{agent_id}_{timestamp}.json"
        filepath = os.path.join(self.base_dir, filename)
        
        snapshot = {
            "agent_id": agent_id,
            "timestamp": timestamp,
            "unix_time": time.time(),
            "state": state_data,
            "version": "1.0"
        }
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            
            # تحديث رابط 'الأحدث' (Latest)
            latest_path = os.path.join(self.base_dir, f"{agent_id}_latest.json")
            with open(latest_path, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
                
            logger.info(f"💾 تم حفظ لقطة الوعي للوكيل {agent_id} في {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"❌ فشل حفظ لقطة الوعي: {e}")
            raise

    def load_snapshot(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """استعادة آخر لقطة حالة للوكيل."""
        latest_path = os.path.join(self.base_dir, f"{agent_id}_latest.json")
        
        if not os.path.exists(latest_path):
            logger.warning(f"⚠️ لا توجد لقطات حالة محفوظة للوكيل {agent_id}")
            return None
            
        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                snapshot = json.load(f)
            logger.info(f"🔄 تم استعادة وعي الوكيل {agent_id} من لقطة بتاريخ {snapshot['timestamp']}")
            return snapshot["state"]
        except Exception as e:
            logger.error(f"❌ فشل استعادة لقطة الوعي: {e}")
            return None

    def list_snapshots(self, agent_id: str) -> list:
        """سرد جميع اللقطات المتاحة لوكيل معين."""
        import glob
        pattern = os.path.join(self.base_dir, f"{agent_id}_*.json")
        files = glob.glob(pattern)
        return sorted(files, reverse=True)

# نسخة عالمية للمدير
persistence_manager = PersistenceManager()
