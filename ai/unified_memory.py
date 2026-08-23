import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from ai.ann_engine import ANNEngine
from ai.sharding_engine import ShardingEngine
import json
import os

class UnifiedMemoryManager:
    """
    مدير الذاكرة الموحدة (Unified Memory Manager).
    يربط بين ANNEngine للبحث الدلالي السريع و ShardingEngine للتخزين المستدام الموزع.
    """
    def __init__(self, base_dir: str, dimension: int = 1536, num_shards: int = 4):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.ann = ANNEngine(dimension=dimension)
        self.sharding = ShardingEngine(self.base_dir, num_shards=num_shards)
        
        self.index_path = self.base_dir / "unified_ann_index.json"
        self._load_initial_state()

    def _load_initial_state(self):
        """تحميل الفهرس الحالي من ملفات التخزين."""
        if self.index_path.exists():
            self.ann.load_index(str(self.index_path))
        else:
            # إذا لم يوجد فهرس، نقوم ببنائه من الأجزاء (Shards)
            all_assets = self.sharding.get_all_assets()
            for asset in all_assets:
                if "embedding" in asset:
                    vector = np.array(asset["embedding"])
                    self.ann.add_vector(vector, asset)
            if all_assets:
                self.ann.build_index()
                self.ann.save_index(str(self.index_path))

    def store_experience(self, experience: Dict[str, Any], embedding: Optional[List[float]] = None):
        """
        تخزين خبرة جديدة في الذاكرة الموحدة.
        1. حفظ في Sharding للتخزين الدائم.
        2. إضافة للفهرس ANN للبحث السريع.
        """
        if "id" not in experience:
            import uuid
            experience["id"] = str(uuid.uuid4())
            
        # إضافة التضمين للخبرة إذا توفر
        if embedding:
            experience["embedding"] = embedding
            vector = np.array(embedding)
            # التأكد من وجود المعرف الفريد في الميتا بيانات للفهرسة
            meta = experience.copy()
            if "unique_id" not in meta and "id" in meta:
                meta["unique_id"] = meta["id"]
            self.ann.add_vector(vector, meta)
            self.ann.build_index()
            self.ann.save_index(str(self.index_path))
            
        # الحفظ في الأجزاء
        shard_idx = self.sharding.add_to_shard(experience)
        return experience["id"], shard_idx

    def semantic_search(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """البحث الدلالي السريع باستخدام ANN."""
        vector = np.array(query_vector)
        results = self.ann.search(vector, top_k=top_k)
        # إرجاع الميتا بيانات فقط (الخبرات)
        return [res[0] for res in results]

    def keyword_search(self, query: str, media_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """البحث النصي التقليدي عبر الأجزاء."""
        return self.sharding.search_in_shards(query, media_type=media_type)

    def get_memory_stats(self) -> Dict[str, Any]:
        """إحصائيات الذاكرة الموحدة."""
        all_assets = self.sharding.get_all_assets()
        return {
            "total_experiences": len(all_assets),
            "num_shards": self.sharding.num_shards,
            "indexed_vectors": len(self.ann.vectors),
            "dimension": self.ann.dimension
        }
