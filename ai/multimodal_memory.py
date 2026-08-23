"""
ai/multimodal_memory.py
=======================
نظام الذاكرة الجماعية متعددة الوسائط للسرب السيادي.

يدير هذا الملف تخزين، فهرسة، ومشاركة الأصول البصرية والسمعية بين الوكلاء،
مما يسمح للسرب بامتلاك "ذاكرة حسية" مشتركة.
"""
import json
import os
import time
import uuid
import shutil
from typing import Any, Dict, List, Optional
from pathlib import Path
import numpy as np
from ai.quantization_engine import VectorQuantizer
from ai.sharding_engine import ShardingEngine

class MultimodalMemory:
    def __init__(self, storage_dir: Optional[str] = None):
        self.root = Path(__file__).resolve().parent.parent
        if storage_dir:
            self.storage_dir = Path(storage_dir).resolve()
        else:
            self.storage_dir = (self.root / "artifacts" / "memory" / "multimodal").resolve()
            
        self.assets_dir = self.storage_dir / "assets"
        self.index_path = self.storage_dir / "index.json"
        
        # إنشاء المجلدات اللازمة
        self.assets_dir.mkdir(parents=True, exist_ok=True)
        self._init_index()
        self.quantizer = VectorQuantizer()
        self.quantizer.load_codebook()
        self.sharding_engine = ShardingEngine(self.storage_dir)

    def _init_index(self):
        """تهيئة ملف الفهرس إذا لم يكن موجوداً."""
        if not self.index_path.exists():
            with open(self.index_path, "w", encoding="utf-8") as f:
                json.dump({"assets": [], "tags": {}}, f, ensure_ascii=False, indent=2)

    def store_asset(self, agent_id: str, file_path: str, media_type: str, metadata: Dict[str, Any]) -> str:
        """تخزين أصل جديد في الذاكرة الجماعية مع التشفير."""
        from ai.security_manager import security_manager
        asset_id = f"asset_{uuid.uuid4().hex[:8]}"
        src_path = Path(file_path)
        if not src_path.exists():
            raise FileNotFoundError(f"الملف غير موجود: {file_path}")
            
        # تحديد المسار الجديد
        ext = src_path.suffix
        dest_path = self.assets_dir / f"{asset_id}{ext}"
        
        # تشفير الملف أثناء النسخ
        with open(src_path, 'rb') as f_src:
            encrypted_data = security_manager.encrypt(f_src.read())
        
        with open(dest_path, 'wb') as f_dest:
            f_dest.write(encrypted_data)
        
        # تحديث الفهرس (تشفير البيانات الوصفية الحساسة)
        entry = {
            "id": asset_id,
            "owner": agent_id,
            "type": media_type,
            "path": str(dest_path.relative_to(self.root)),
            "metadata": metadata,
            "ts": time.time(),
            "encrypted": True
        }
        
        self._update_index(entry)
        return asset_id

    def _update_index(self, entry: Dict[str, Any]):
        """تحديث ملف الفهرس بالبيانات الجديدة."""
        with open(self.index_path, "r+", encoding="utf-8") as f:
            data = json.load(f)
            
            # تطبيق التكميم إذا وجد متجه دلالي (Embedding)
            if "embedding" in entry.get("metadata", {}):
                vector = entry["metadata"]["embedding"]
                compressed_idx = self.quantizer.quantize(np.array(vector))
                entry["metadata"]["compressed_idx"] = compressed_idx
                # إزالة المتجه الأصلي لتوفير المساحة
                del entry["metadata"]["embedding"]
                entry["quantized"] = True

            data["assets"].append(entry)
            
            # تحديث الوسوم (Tags) للبحث السريع
            for tag in entry.get("metadata", {}).get("tags", []):
                if tag not in data["tags"]:
                    data["tags"][tag] = []
                data["tags"][tag].append(entry["id"])
                
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()
            
            # إضافة إلى التخزين المجزأ أيضاً
            self.sharding_engine.add_to_shard(entry)

    def search_assets(self, query: str, media_type: Optional[str] = None, use_shards: bool = True) -> List[Dict[str, Any]]:
        """البحث عن أصول في الذاكرة بناءً على الوسوم أو النوع، مع دعم التجزئة."""
        if use_shards:
            return self.sharding_engine.search_in_shards(query, media_type)
            
        with open(self.index_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        results = []
        # البحث في الوسوم أولاً
        asset_ids = data["tags"].get(query, [])
        
        for asset in data["assets"]:
            if asset["id"] in asset_ids or query.lower() in asset["metadata"].get("description", "").lower():
                if not media_type or asset["type"] == media_type:
                    results.append(asset)
                    
        return results

mm_memory = MultimodalMemory()
