
import json
import hashlib
import time
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("NSM.CacheManager")

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "artifacts" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class CacheManager:
    """مدير التخزين المؤقت الذكي للأدوات والنتائج."""
    def __init__(self, ttl_seconds: int = 3600):
        self.memory_cache = {} # L1 Cache (RAM)
        self.ttl = ttl_seconds

    def _generate_key(self, tool_name: str, params: Dict[str, Any]) -> str:
        """توليد بصمة فريدة للطلب."""
        # إزالة المعاملات الحساسة قبل التوليد
        safe_params = {k: v for k, v in params.items() if "token" not in k.lower() and "key" not in k.lower()}
        param_str = json.dumps(safe_params, sort_keys=True)
        return hashlib.sha256(f"{tool_name}:{param_str}".encode()).hexdigest()

    def get(self, tool_name: str, params: Dict[str, Any]) -> Optional[Any]:
        """محاولة جلب النتيجة من الكاش."""
        key = self._generate_key(tool_name, params)
        
        # 1. البحث في L1
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                logger.info(f"🚀 [L1 Cache Hit] {tool_name}")
                return entry["result"]
        
        # 2. البحث في L2 (القرص)
        cache_file = CACHE_DIR / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    entry = json.load(f)
                if time.time() - entry["timestamp"] < self.ttl:
                    # تحديث L1
                    self.memory_cache[key] = entry
                    logger.info(f"💾 [L2 Cache Hit] {tool_name}")
                    return entry["result"]
                else:
                    cache_file.unlink() # حذف الملف المنتهي الصلاحية
            except Exception as e:
                logger.error(f"❌ خطأ قراءة الكاش: {e}")
        
        return None

    def set(self, tool_name: str, params: Dict[str, Any], result: Any):
        """حفظ النتيجة في الكاش."""
        key = self._generate_key(tool_name, params)
        entry = {
            "tool": tool_name,
            "result": result,
            "timestamp": time.time()
        }
        
        # حفظ في L1
        self.memory_cache[key] = entry
        
        # حفظ في L2
        try:
            with open(CACHE_DIR / f"{key}.json", "w", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"❌ خطأ حفظ الكاش: {e}")

# نسخة عالمية واحدة للمشروع
agent_cache = CacheManager()
