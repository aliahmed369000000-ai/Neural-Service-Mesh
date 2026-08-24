import sys
import os
import json
import shutil

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.video_indexer import VideoTemporalIndexer

def test_sharding_efficiency():
    print("🚀 اختبار ميزة التخزين المجزأ (Sharding)...")
    
    storage_dir = "artifacts/test_shards"
    if os.path.exists(storage_dir): shutil.rmtree(storage_dir)
    
    indexer = VideoTemporalIndexer(storage_dir=storage_dir)
    video_id = "sharded_video_test"
    
    # 1. إنشاء فهرس ضخم (250 إطار > 100 عتبة التجزئة)
    print("📦 إنشاء فهرس يحتوي على 250 إطار...")
    indexer.create_index(video_id)
    for i in range(250):
        indexer.add_keyframe(video_id, float(i), f"frame_{i}.jpg", f"وصف {i}", ["وسم"])
    
    # 2. التحقق من وجود الملفات المجزأة
    main_file = os.path.join(storage_dir, f"{video_id}_index.json")
    shard_dir = os.path.join(storage_dir, f"{video_id}_shards")
    
    print(f"🔍 التحقق من الملف الرئيسي: {os.path.exists(main_file)}")
    print(f"🔍 التحقق من مجلد الأجزاء: {os.path.exists(shard_dir)}")
    
    if os.path.exists(shard_dir):
        shards = os.listdir(shard_dir)
        print(f"📂 عدد الأجزاء المكتشفة: {len(shards)}")
        
        # 3. التحقق من تحميل البيانات بشكل صحيح
        print("🔄 محاولة تحميل الفهرس المجزأ...")
        # إفراغ الذاكرة النشطة لإجبار التحميل من القرص
        indexer.active_indices = {}
        loaded_index = indexer.load_index(video_id)
        
        if loaded_index and len(loaded_index.get("keyframes", [])) == 250:
            print("✅ نجاح: تم تحميل ودمج 250 إطار من الأجزاء بنجاح.")
        else:
            print(f"❌ فشل: عدد الإطارات المحملة غير صحيح ({len(loaded_index.get('keyframes', [])) if loaded_index else 0})")
    else:
        print("❌ فشل: لم يتم إنشاء مجلد الأجزاء.")

if __name__ == "__main__":
    test_sharding_efficiency()
