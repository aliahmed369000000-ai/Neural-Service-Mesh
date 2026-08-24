import os
import sys

# إضافة مسار المشروع لـ PYTHONPATH
sys.path.append('/home/ubuntu/Neural-Service-Mesh')

from ai.vision_analyzer import vision_analyzer
from ai.video_sampler import video_sampler
from ai.video_indexer import video_indexer

def test_vision_integration():
    print("🧪 بدء اختبار تكامل الرؤية مع الفيديو...")
    
    # 1. اختبار تحليل صورة منفردة
    test_image = "/home/ubuntu/Neural-Service-Mesh/test_image.png"
    if not os.path.exists(test_image):
        print("⚠️ صورة الاختبار غير موجودة، سيتم تخطي هذا الجزء.")
    else:
        print(f"📸 تحليل صورة: {test_image}")
        res = vision_analyzer.analyze_image(test_image)
        if res.get("ok"):
            print(f"✅ وصف الصورة: {res['description'][:100]}...")
        else:
            print(f"❌ فشل تحليل الصورة: {res.get('error')}")

    # 2. اختبار الفهرس الزمني المحدث
    video_id = "test_vision_video"
    video_indexer.create_index(video_id, metadata={"test": True})
    
    # محاكاة إضافة إطار محلل
    video_indexer.add_keyframe(
        video_id=video_id,
        timestamp=10.5,
        frame_path=test_image if os.path.exists(test_image) else "dummy.jpg",
        description="هذا وصف تجريبي تم إنتاجه بواسطة نموذج الرؤية.",
        tags=["vision_test"]
    )
    
    index = video_indexer.load_index(video_id)
    if index and len(index.get("keyframes", [])) > 0:
        kf = index["keyframes"][0]
        print(f"✅ تم العثور على إطار في الفهرس مع الوصف: {kf.get('description')}")
    else:
        print("❌ فشل تحديث الفهرس بالوصف البصري.")

if __name__ == "__main__":
    test_vision_integration()
