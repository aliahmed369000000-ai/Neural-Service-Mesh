import hashlib
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

class ShardingEngine:
    """
    محرك التخزين المجزأ (Sharding Engine) لمشروع NSM.
    يقوم بتقسيم البيانات إلى أجزاء (Shards) بناءً على معرف الأصل أو الوسوم
    لضمان قابلية التوسع وتجنب ضخامة ملف الفهرس الواحد.
    """
    def __init__(self, base_dir: Path, num_shards: int = 4):
        self.base_dir = base_dir
        self.num_shards = num_shards
        self.shards_dir = base_dir / "shards"
        self.shards_dir.mkdir(parents=True, exist_ok=True)
        self._init_shards()

    def _init_shards(self):
        """تهيئة ملفات الأجزاء إذا لم تكن موجودة."""
        for i in range(self.num_shards):
            shard_path = self.shards_dir / f"shard_{i}.json"
            if not shard_path.exists():
                with open(shard_path, "w", encoding="utf-8") as f:
                    json.dump({"assets": []}, f, ensure_ascii=False, indent=2)

    def get_shard_index(self, asset_id: str) -> int:
        """تحديد رقم الجزء (Shard Index) بناءً على hash الخاص بمعرف الأصل."""
        hash_val = int(hashlib.md5(asset_id.encode()).hexdigest(), 16)
        return hash_val % self.num_shards

    def add_to_shard(self, entry: Dict[str, Any]):
        """إضافة أصل إلى الجزء المناسب."""
        shard_idx = self.get_shard_index(entry["id"])
        shard_path = self.shards_dir / f"shard_{shard_idx}.json"
        
        with open(shard_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data["assets"].append(entry)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()
        return shard_idx

    def get_all_assets(self) -> List[Dict[str, Any]]:
        """جمع كل الأصول من جميع الأجزاء."""
        all_assets = []
        for i in range(self.num_shards):
            shard_path = self.shards_dir / f"shard_{i}.json"
            with open(shard_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                all_assets.extend(data["assets"])
        return all_assets

    def search_in_shards(self, query: str, media_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """البحث المتوازي (محاكاة) في جميع الأجزاء."""
        results = []
        for i in range(self.num_shards):
            shard_path = self.shards_dir / f"shard_{i}.json"
            with open(shard_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for asset in data["assets"]:
                    if query.lower() in asset["metadata"].get("description", "").lower():
                        if not media_type or asset["type"] == media_type:
                            results.append(asset)
        return results
