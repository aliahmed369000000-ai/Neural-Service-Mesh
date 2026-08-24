import sys
import os
import cv2
import numpy as np

# إضافة جذر المشروع للمسار
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.video_sampler import video_sampler

def create_test_video(path: str, duration: int = 5, fps: int = 30):
    """إنشاء فيديو تجريبي يحتوي على تغييرات مشهد."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (640, 480))
    
    for i in range(duration * fps):
        # تغيير اللون كل ثانية لمحاكاة تغيير المشهد
        color = (0, 0, 0) if (i // fps) % 2 == 0 else (255, 255, 255)
        frame = np.full((480, 640, 3), color, dtype=np.uint8)
        # إضافة نص متغير لمنع التطابق التام
        cv2.putText(frame, f"Frame {i}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (127, 127, 127), 2)
        out.write(frame)
    
    out.release()
    print(f"✅ تم إنشاء فيديو تجريبي: {path}")

def test_sampling():
    video_path = "tests/sample_video.mp4"
    create_test_video(video_path)
    
    print("🧪 بدء اختبار أخذ العينات الذكي...")
    result = video_sampler.process_video(video_path, "test_vid_001")
    
    print(f"📝 النتيجة: {result}")
    
    if result["status"] == "success":
        print(f"✅ نجح الاختبار. عدد الإطارات المستخلصة: {result['sampled_frames']}")
        print(f"✅ كفاءة التصفية: {result['efficiency']:.2f}%")
    else:
        print(f"❌ فشل الاختبار: {result['message']}")

if __name__ == "__main__":
    test_sampling()
