import sys
import os
import time

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.multimodal_sync import MultimodalSyncManager
from ai.video_indexer import video_indexer

def test_ann_speedup():
    print("🚀 اختبار أداء الفهرسة المتجهية (ANN Performance)...")
    
    sync_manager = MultimodalSyncManager()
    video_id = "ann_perf_test"
    
    # 1. إنشاء قاعدة بيانات تحتوي على 1000 عنصر
    print("📦 توليد 1000 عنصر مزامنة مع هاشات LSH...")
    synced_data = []
    for i in range(1000):
        text = f"كلمة مفتاحية {i}"
        desc = f"وصف بصري {i}"
        raw_vec = sync_manager._generate_embedding(f"{desc} {text}")
        synced_data.append({
            "timestamp": float(i),
            "spoken_text": text,
            "visual_description": desc,
            "semantic_index": {
                "vector": sync_manager._quantize_vector(raw_vec),
                "lsh_hash": sync_manager._generate_lsh_hash(raw_vec)
            }
        })
    
    # حفظ في الفهرس
    video_indexer.active_indices[video_id] = {"multimodal_sync": synced_data}
    
    # 2. قياس وقت البحث
    print("🔍 إجراء بحث دلالي باستخدام فلترة ANN...")
    start_time = time.time()
    results = sync_manager.query_context(video_id, "كلمة مفتاحية 500", semantic=True)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"⏱️ زمن البحث لـ 1000 عنصر: {duration*1000:.4f} ms")
    print(f"✅ عدد النتائج المكتشفة: {len(results)}")
    
    if duration < 0.01:
        print("🚀 نجاح: البحث سريع جداً بفضل فلترة ANN.")
    else:
        print("⚠️ تنبيه: زمن البحث قد يكون قابلاً للتحسين.")

if __name__ == "__main__":
    test_ann_speedup()
