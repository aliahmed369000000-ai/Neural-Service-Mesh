import cv2
import os
import time
from typing import List, Dict, Any, Tuple
from ai.video_indexer import video_indexer
from ai.vision_analyzer import vision_analyzer

class SmartVideoSampler:
    """أخذ عينات ذكية من الفيديو بناءً على كشف التغير الجوهري."""
    
    def __init__(self, threshold: float = 30.0, min_interval: float = 1.0):
        self.threshold = threshold  # عتبة التغيير (0-100)
        self.min_interval = min_interval  # الحد الأدنى للوقت بين الإطارات بالثواني

    def process_video(self, video_path: str, video_id: str) -> Dict[str, Any]:
        """معالجة الفيديو واستخراج الإطارات المهمة."""
        if not os.path.exists(video_path):
            return {"status": "error", "message": "ملف الفيديو غير موجود"}

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        video_indexer.create_index(video_id, metadata={
            "path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "duration": total_frames / fps if fps > 0 else 0
        })

        last_frame = None
        last_timestamp = -self.min_interval
        sampled_count = 0
        
        print(f"🚀 بدء معالجة الفيديو: {video_id} ({total_frames} إطار)...")

        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_idx / fps
            
            # التحقق من الحد الأدنى للفاصل الزمني
            if timestamp - last_timestamp >= self.min_interval:
                # تحويل الإطار للرمادي لتبسيط المقارنة
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

                if last_frame is not None:
                    # حساب الفرق المطلق بين الإطارات
                    frame_delta = cv2.absdiff(last_frame, gray_frame)
                    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
                    change_percent = (cv2.countNonZero(thresh) / thresh.size) * 100

                    # إذا تجاوز التغيير العتبة، نحفظ الإطار
                    if change_percent >= self.threshold:
                        self._save_sampled_frame(video_id, timestamp, frame, change_percent)
                        sampled_count += 1
                        last_timestamp = timestamp
                else:
                    # حفظ الإطار الأول دائماً
                    self._save_sampled_frame(video_id, timestamp, frame, 100.0)
                    sampled_count += 1
                    last_timestamp = timestamp
                
                last_frame = gray_frame
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"📊 تمت معالجة {frame_idx}/{total_frames} إطار...")

        cap.release()
        return {
            "status": "success",
            "video_id": video_id,
            "sampled_frames": sampled_count,
            "efficiency": 100 - (sampled_count / total_frames * 100) if total_frames > 0 else 0
        }

    def _save_sampled_frame(self, video_id: str, timestamp: float, frame: Any, change_score: float):
        """حفظ الإطار المستخلص وتحديث الفهرس."""
        save_dir = f"artifacts/sampled_frames/{video_id}"
        os.makedirs(save_dir, exist_ok=True)
        
        frame_name = f"frame_{timestamp:.2f}s.jpg"
        frame_path = os.path.join(save_dir, frame_name)
        cv2.imwrite(frame_path, frame)
        
        # تحليل الإطار تلقائياً باستخدام نموذج الرؤية
        print(f"🧠 تحليل الإطار عند {timestamp:.2f}s...")
        vision_res = vision_analyzer.analyze_image(frame_path)
        description = vision_res.get("description", f"تغير بصرى بنسبة {change_score:.2f}%") if vision_res.get("ok") else f"تغير بصرى بنسبة {change_score:.2f}%"
        
        # إضافة للفهرس الزمني (مع الوصف المستخلص)
        video_indexer.add_keyframe(
            video_id=video_id,
            timestamp=timestamp,
            frame_path=frame_path,
            description=description,
            tags=["motion", "keyframe", "analyzed"]
        )

video_sampler = SmartVideoSampler()
