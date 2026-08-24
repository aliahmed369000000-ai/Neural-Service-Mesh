import sys
import os
import json
import shutil
import time

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.multimodal_sync import MultimodalSyncManager
from ai.video_indexer import VideoTemporalIndexer

def test_ann_sharding_integration():
    print("🚀 اختبار دمج ANN مع التخزين المجزأ (Sharding Integration)...")
    
    storage_dir = "artifacts/test_ann_shards"
    if os.path.exists(storage_dir): shutil.rmtree(storage_dir)
    
    indexer = VideoTemporalIndexer(storage_dir=storage_dir)
    sync_manager = MultimodalSyncManager()
    video_id = "ann_sharding_test"
    
    # 1. إنشاء قاعدة بيانات ضخمة (500 عنصر > 5 أجزاء)
    print("📦 توليد 500 عنصر مزامنة وتجزئتها...")
    indexer.create_index(video_id)
    
    synced_data = []
    for i in range(500):
        text = f"مفهوم تقني {i}"
        desc = f"مشهد مرئي {i}"
        raw_vec = sync_manager._generate_embedding(f"{desc} {text}")
        synced_data.append({
            "timestamp": float(i),
            "spoken_text": text,
            "visual_description": desc,
            "frame_path": f"frame_{i}.jpg",
            "semantic_index": {
                "vector": sync_manager._quantize_vector(raw_vec),
                "lsh_hash": sync_manager._generate_lsh_hash(raw_vec),
                "quantized": True
            }
        })
    
    indexer.active_indices[video_id]["multimodal_sync"] = synced_data
    indexer._save_index(video_id)
    
    # 2. التحقق من التجزئة
    shard_dir = os.path.join(storage_dir, f"{video_id}_shards")
    print(f"📂 عدد الأجزاء المنشأة: {len(os.listdir(shard_dir)) if os.path.exists(shard_dir) else 0}")
    
    # 3. اختبار البحث الدلالي عبر الأجزاء
    print("🔍 إجراء بحث دلالي عبر الفهارس المجزأة باستخدام ANN...")
    # إفراغ الذاكرة النشطة
    indexer.active_indices = {}
    
    # إنشاء مدير مزامنة جديد مع الفهرس المخصص
    sync_manager = MultimodalSyncManager(indexer=indexer)
    
    start_time = time.time()
    # التحميل الكامل للفهرس (يتم دمج الأجزاء تلقائياً في load_index)
    results = sync_manager.query_context(video_id, "مفهوم تقني 250", semantic=True)
    duration = time.time() - start_time
    
    print(f"⏱️ زمن البحث والتحميل: {duration*1000:.2f} ms")
    print(f"✅ عدد النتائج المكتشفة: {len(results)}")
    if len(results) > 0:
        print(f"🔝 النتيجة الأولى (الطابع الزمني): {results[0]['timestamp']}")
        print(f"📊 درجة البحث: {results[0].get('search_score', 0)}")
    
    # البحث الدلالي قد يعيد نتائج مشابهة، نتحقق من وجود النتيجة المطلوبة في القائمة
    found = any(r["timestamp"] == 250.0 for r in results)
    if found:
        print("✅ نجاح: تم العثور على النتيجة المطلوبة عبر الأجزاء بنجاح.")
    else:
        print("❌ فشل: لم يتم العثور على النتيجة المطلوبة.")

if __name__ == "__main__":
    test_ann_sharding_integration()
