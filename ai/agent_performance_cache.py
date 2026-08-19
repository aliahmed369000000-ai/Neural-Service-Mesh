"""
ai/agent_performance_cache.py
=============================
🆕 تخزين مؤقت عالي الأداء للنتائج المتكررة (Performance Cache).

يقلل استدعاءات LLM المكررة ويُحسّن زمن الاستجابة:
  • LRU Cache في الذاكرة (سريع)
  • Disk Cache (persistent عبر restarts)
  • TTL قابل للضبط
  • Cache stats (hit rate, miss rate)

الاستخدام:
    from ai.agent_performance_cache import AgentCache
    cache = AgentCache(max_size=1000, ttl_seconds=3600)
    result = cache.get_or_compute("key", compute_fn)
    stats = cache.stats()
"""
from __future__ import annotations
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
logger = logging.getLogger("nsm.agent_cache")

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "artifacts" / "agent_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class AgentCache:
    """تخزين مؤقت ثنائي المستوى (memory + disk) للوكلاء."""

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 3600,
        cache_file: Optional[Path] = None,
    ):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache_file = cache_file or (CACHE_DIR / "cache.json")
        self._lock = threading.RLock()

        # LRU memory cache
        self._memory: OrderedDict[str, Dict[str, Any]] = OrderedDict()

        # Stats
        self._hits = 0
        self._misses = 0
        self._writes = 0

        # تحميل من disk
        self._load_from_disk()

    def _make_key(self, *args: Any, **kwargs: Any) -> str:
        """توليد مفتاح cache من args/kwargs."""
        raw = json.dumps({"args": args, "kwargs": kwargs},
                         sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, key: str) -> Optional[Any]:
        """جلب من cache."""
        with self._lock:
            if key not in self._memory:
                self._misses += 1
                return None

            entry = self._memory[key]
            # فحص TTL
            if entry["expires_at"] and time.time() > entry["expires_at"]:
                del self._memory[key]
                self._misses += 1
                return None

            # نقل إلى النهاية (LRU)
            self._memory.move_to_end(key)
            self._hits += 1
            return entry["value"]

    def set(self, key: str, value: Any) -> None:
        """تخزين في cache."""
        expires_at = time.time() + self.ttl_seconds if self.ttl_seconds > 0 else None

        with self._lock:
            self._memory[key] = {
                "value": value,
                "created_at": time.time(),
                "expires_at": expires_at,
            }

            # LRU eviction
            while len(self._memory) > self.max_size:
                self._memory.popitem(last=False)

            self._writes += 1

            # حفظ إلى disk بشكل دوري
            if self._writes % 10 == 0:
                self._save_to_disk()

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """جلب من cache أو حساب جديد.
        
        Args:
            key: مفتاح cache صريح (أو يُولّد من args/kwargs إن لم يُعطَ)
            compute_fn: الدالة التي تُستدعى عند cache miss
            *args, **kwargs: تمرر إلى compute_fn
        """
        if not key:
            key = self._make_key(*args, **kwargs)

        # محاولة من cache
        cached = self.get(key)
        if cached is not None:
            return cached

        # حساب جديد
        result = compute_fn(*args, **kwargs)
        self.set(key, result)
        return result

    def clear(self) -> int:
        """مسح كل الـ cache."""
        with self._lock:
            count = len(self._memory)
            self._memory.clear()
            self._hits = 0
            self._misses = 0
            self._writes = 0
            # مسح disk
            if self._cache_file.exists():
                self._cache_file.unlink()
        return count

    def invalidate(self, key: str) -> bool:
        """حذف entry محددة."""
        with self._lock:
            if key in self._memory:
                del self._memory[key]
                return True
            return False

    def stats(self) -> Dict[str, Any]:
        """إحصائيات الـ cache."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0
        return {
            "size": len(self._memory),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "writes": self._writes,
            "hit_rate": f"{hit_rate:.1f}%",
            "ttl_seconds": self.ttl_seconds,
        }

    def _save_to_disk(self) -> None:
        """حفظ الـ cache إلى disk."""
        try:
            data = {
                "ttl_seconds": self.ttl_seconds,
                "entries": {
                    k: v for k, v in self._memory.items()
                    if v["expires_at"] is None or v["expires_at"] > time.time()
                },
            }
            temp = self._cache_file.with_suffix(".tmp")
            with open(temp, "w") as f:
                json.dump(data, f, default=str)
            os.replace(str(temp), str(self._cache_file))
        except Exception as e:
            logger.warning(f"Cache save error: {e}")

    def _load_from_disk(self) -> None:
        """تحميل الـ cache من disk."""
        if not self._cache_file.exists():
            return
        try:
            with open(self._cache_file) as f:
                data = json.load(f)

            now = time.time()
            for key, entry in data.get("entries", {}).items():
                # فحص هل منتهي الصلاحية
                expires = entry.get("expires_at")
                if expires and now > expires:
                    continue
                self._memory[key] = {
                    "value": entry["value"],
                    "created_at": entry.get("created_at", now),
                    "expires_at": expires,
                }
        except Exception as e:
            logger.warning(f"Cache load error: {e}")
