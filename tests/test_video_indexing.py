
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.video_indexer import video_indexer

def test_video_indexing():
    print("🎬 بدء اختبار فهرسة الفيديو الزمني (Video Temporal Indexing)...")
    
    video_id = "test_video_001"
    video_indexer.create_index(video_id, metadata={"title": "مقطع تجريبي", "duration": 120})
    
    # محاكاة إضافة إطارات رئيسية
    print("⏳ إضافة إطارات رئيسية...")
    frames = [
        (10.5, "path/to/frame1.jpg", "شخص يتحدث", ["شخص", "حديث"]),
        (45.2, "path/to/frame2.jpg", "رسم بياني يظهر نمواً", ["رسم بياني", "نمو"]),
        (88.0, "path/to/frame3.jpg", "شعار الشركة في النهاية", ["شعار", "نهاية"])
    ]
    
    for ts, path, desc, tags in frames:
        video_indexer.add_keyframe(video_id, ts, path, desc, tags)
    
    # اختبار البحث الزمني
    print("\n🔍 اختبار البحث الزمني (40s - 90s)...")
    results = video_indexer.search_by_time(video_id, 40.0, 90.0)
    for r in results:
        print(f"✅ وجد إطار عند {r['timestamp']}s: {r['description']}")

    # اختبار البحث بالوسوم
    print("\n🔍 اختبار البحث بالوسوم ('رسم بياني')...")
    tag_results = video_indexer.search_by_tag(video_id, "رسم بياني")
    if tag_results:
        print(f"✅ نجاح: وجد {len(tag_results)} إطار بالوسم المطلوب.")
    else:
        print("❌ فشل: لم يجد الإطار بالوسم.")

if __name__ == "__main__":
    test_video_indexing()
