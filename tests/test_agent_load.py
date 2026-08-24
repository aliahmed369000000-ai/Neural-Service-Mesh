import sys
import os
import time
import json
import random

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.multimodal_sync import MultimodalSyncManager
from ai.video_indexer import video_indexer
from ai.agent_hibernation import AgentState

def run_load_test():
    print("🚀 بدء اختبار الضغط (Load Testing)...")
    
    video_id = "long_complex_video"
    sync_manager = MultimodalSyncManager()
    
    # 1. محاكاة بيانات ضخمة (1000 نقطة زمنية)
    print("📦 توليد بيانات محاكاة ضخمة (1000 نقطة زمنية)...")
    synced_data = []
    for i in range(1000):
        desc = f"وصف بصري رقم {i} لمحتوى معقد"
        spoken = f"نص مسموع في الثانية {i} يحتوي على كلمات إيجابية وسلبية"
        
        item = {
            "timestamp": float(i),
            "visual_description": desc,
            "spoken_text": spoken,
            "frame_path": f"frame_{i}.jpg",
            "sentiment": sync_manager._analyze_sentiment(spoken, desc),
            "semantic_index": {
                "vector": sync_manager._generate_embedding(f"{desc} {spoken}"),
                "tags": ["اختبار", "ضغط", str(i)]
            }
        }
        synced_data.append(item)
    
    # 2. قياس زمن الحفظ والضغط
    print("💾 قياس أداء الحفظ والضغط...")
    start_time = time.time()
    
    index = video_indexer.create_index(video_id)
    index["multimodal_sync"] = synced_data
    video_indexer.active_indices[video_id] = index
    video_indexer._save_index(video_id)
    
    state = AgentState("load_agent", [{"role": "user", "content": "مهمة طويلة"}], {"tasks": []}, multimodal_memory={video_id: index})
    
    initial_size = len(json.dumps(state.to_dict())) / 1024
    state.compress(target_size_kb=200) # ضغط الذاكرة الضخمة
    
    compressed_size = len(json.dumps(state.to_dict())) / 1024
    end_time = time.time()
    
    print(f"⏱️ زمن المعالجة والضغط: {end_time - start_time:.2f} ثانية")
    print(f"📊 الحجم الابتدائي: {initial_size:.2f} KB")
    print(f"📉 الحجم بعد الضغط: {compressed_size:.2f} KB")
    
    # 3. قياس زمن البحث الدلالي تحت الضغط
    print("🔍 قياس سرعة البحث الدلالي في بيانات ضخمة...")
    search_start = time.time()
    results = sync_manager.query_context(video_id, "محتوى معقد", semantic=True)
    search_end = time.time()
    
    print(f"⚡ زمن البحث الدلالي: {(search_end - search_start) * 1000:.2f} ميلي ثانية")
    
    # النتائج النهائية
    success = len(results) > 0 and compressed_size < initial_size
    if success:
        print("✅ نجاح اختبار الضغط: النظام مستقر وفعال تحت الأحمال العالية.")
        return {
            "processing_time": end_time - start_time,
            "initial_size_kb": initial_size,
            "compressed_size_kb": compressed_size,
            "search_latency_ms": (search_end - search_start) * 1000
        }
    else:
        print("❌ فشل اختبار الضغط.")
        return None

if __name__ == "__main__":
    results = run_load_test()
    if results:
        with open("/home/ubuntu/Neural-Service-Mesh/artifacts/load_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
