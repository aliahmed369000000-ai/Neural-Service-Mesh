import os
from typing import List, Dict, Any, Optional
from ai.video_indexer import video_indexer
from ai.stt_engine import transcribe_audio
from ai.drift_corrector import DriftCorrector

class MultimodalSyncManager:
    def __init__(self):
        self.drift_corrector = DriftCorrector()
    """إدارة المزامنة بين المسار الصوتي والإطارات المرئية في الفيديو."""
    
    def sync_video_audio(self, video_id: str, audio_path: str, retry_count: int = 0) -> Dict[str, Any]:
        """مزامنة الكلام مع الإطارات المرئية للفيديو مع استراتيجيات التعافي."""
        # 1. الحصول على التفريغ الصوتي مع الطوابع الزمنية
        print(f"🎙️ محاولة المزامنة (محاولة {retry_count + 1}) لـ: {audio_path}...")
        try:
            with open(audio_path, "rb") as f:
                audio_bytes = f.read()
        except Exception as e:
            return {"ok": False, "error": f"فشل قراءة الملف: {e}"}
            
        segments, error = transcribe_audio(audio_bytes, with_timestamps=True)
        
        # استراتيجية التعافي: إعادة المحاولة عند أخطاء الشبكة أو الكوتا
        if error:
            if retry_count < 2 and ("quota" in error.lower() or "timeout" in error.lower()):
                print(f"🛡️ [Auto-Heal]: تم رصد خطأ قابل للإصلاح ({error})، إعادة المحاولة بعد انتظار...")
                import time
                time.sleep(2 * (retry_count + 1))
                return self.sync_video_audio(video_id, audio_path, retry_count + 1)
            return {"ok": False, "error": error}
            
        # 2. تحميل الفهرس البصري للفيديو
        index = video_indexer.load_index(video_id)
        if not index:
            return {"ok": False, "error": "الفهرس البصري للفيديو غير موجود."}
            
        # 3. المحاذاة (Alignment) مع تصحيح الانحراف
        synced_data = []
        for kf in index.get("keyframes", []):
            raw_ts = kf["timestamp"]
            
            # محاكاة قياس الانحراف (في الإنتاج الحقيقي يتم قياسه من المزامنة الفيزيائية)
            # هنا نفترض وجود انحراف بسيط يتزايد مع الزمن
            measured_offset = raw_ts * 0.005  
            
            # تصحيح الطابع الزمني
            correction = self.drift_corrector.correct(raw_ts, measured_offset)
            ts = correction["corrected_timestamp"]
            
            # البحث عن الكلام الذي قيل في نفس وقت الإطار المصحح
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
