import json
import time
import os
from typing import List, Dict, Any, Optional

class VideoTemporalIndexer:
    """إدارة الفهارس الزمنية لإطارات الفيديو والبيانات الوصفية المرتبطة بها."""
    
    def __init__(self, storage_dir: str = "artifacts/video_indices"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.active_indices: Dict[str, Dict[str, Any]] = {}

    def create_index(self, video_id: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """إنشاء فهرس جديد لفيديو محدد."""
        index = {
            "video_id": video_id,
            "created_at": time.time(),
            "metadata": metadata or {},
            "keyframes": [],  # قائمة الإطارات المستخلصة
            "events": []      # الأحداث الزمنية المكتشفة
        }
        self.active_indices[video_id] = index
        self._save_index(video_id)
        return index

    def add_keyframe(self, video_id: str, timestamp: float, frame_path: str, description: str, tags: List[str]):
        """إضافة إطار رئيسي للفهرس الزمني."""
        if video_id not in self.active_indices:
            self.load_index(video_id)
            
        keyframe = {
            "timestamp": timestamp,
            "frame_path": frame_path,
            "description": description,
            "tags": tags,
            "hash": hash(frame_path)
        }
        self.active_indices[video_id]["keyframes"].append(keyframe)
        # ترتيب الإطارات زمنياً
        self.active_indices[video_id]["keyframes"].sort(key=lambda x: x["timestamp"])
        self._save_index(video_id)

    def search_by_time(self, video_id: str, start_time: float, end_time: float) -> List[Dict]:
        """البحث عن إطارات ضمن نطاق زمني محدد."""
        index = self.load_index(video_id)
        if not index: return []
        
        return [kf for kf in index["keyframes"] if start_time <= kf["timestamp"] <= end_time]

    def search_by_tag(self, video_id: str, tag: str) -> List[Dict]:
        """البحث عن إطارات تحتوي على وسم معين."""
        index = self.load_index(video_id)
        if not index: return []
        
        return [kf for kf in index["keyframes"] if tag.lower() in [t.lower() for t in kf["tags"]]]

    def load_index(self, video_id: str) -> Optional[Dict]:
        """تحميل فهرس من القرص."""
        if video_id in self.active_indices:
            return self.active_indices[video_id]
            
        path = os.path.join(self.storage_dir, f"{video_id}_index.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                index = json.load(f)
                self.active_indices[video_id] = index
                return index
        return None

    def _save_index(self, video_id: str):
        """حفظ الفهرس على القرص."""
        if video_id in self.active_indices:
            path = os.path.join(self.storage_dir, f"{video_id}_index.json")
            with open(path, "w") as f:
                json.dump(self.active_indices[video_id], f, indent=2)

video_indexer = VideoTemporalIndexer()
