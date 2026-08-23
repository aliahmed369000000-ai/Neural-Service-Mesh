import os
from typing import List, Dict, Any, Optional
from ai.video_indexer import video_indexer
from ai.stt_engine import transcribe_audio

class MultimodalSyncManager:
    """إدارة المزامنة بين المسار الصوتي والإطارات المرئية في الفيديو."""
    
    def sync_video_audio(self, video_id: str, audio_path: str) -> Dict[str, Any]:
        """مزامنة الكلام مع الإطارات المرئية للفيديو."""
        # 1. الحصول على التفريغ الصوتي مع الطوابع الزمنية
        print(f"🎙️ تفريغ الصوت من: {audio_path}...")
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            
        segments, error = transcribe_audio(audio_bytes, with_timestamps=True)
        if error:
            return {"ok": False, "error": error}
            
        # 2. تحميل الفهرس البصري للفيديو
        index = video_indexer.load_index(video_id)
        if not index:
            return {"ok": False, "error": "الفهرس البصري للفيديو غير موجود."}
            
        # 3. المحاذاة (Alignment)
        synced_data = []
        for kf in index.get("keyframes", []):
            ts = kf["timestamp"]
            # البحث عن الكلام الذي قيل في نفس وقت الإطار
            relevant_text = [
                s["text"] for s in segments 
                if s["start"] <= ts <= s["end"]
            ]
            
            synced_item = {
                "timestamp": ts,
                "visual_description": kf["description"],
                "spoken_text": " ".join(relevant_text) if relevant_text else None,
                "frame_path": kf["frame_path"]
            }
            synced_data.append(synced_item)
            
        # 4. حفظ النتائج في الفهرس
        index["multimodal_sync"] = synced_data
        video_indexer._save_index(video_id)
        
        return {
            "ok": True,
            "synced_count": len(synced_data),
            "segments_count": len(segments)
        }

    def query_context(self, video_id: str, keyword: str) -> List[Dict[str, Any]]:
        """البحث عن سياق سمعي بصري باستخدام كلمة مفتاحية."""
        index = video_indexer.load_index(video_id)
        if not index or "multimodal_sync" not in index:
            return []
            
        results = []
        for item in index["multimodal_sync"]:
            text_match = keyword.lower() in (item["spoken_text"] or "").lower()
            visual_match = keyword.lower() in (item["visual_description"] or "").lower()
            
            if text_match or visual_match:
                results.append(item)
        return results

multimodal_sync = MultimodalSyncManager()
