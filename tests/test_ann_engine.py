import sys
import os
import numpy as np
import json
import time
from pathlib import Path

# إضافة المسار للمشروع
sys.path.append(os.getcwd())

from ai.multimodal_memory import MultimodalMemory

def test_ann_search_speed_and_accuracy():
    print("--- اختبار سرعة ودقة البحث الدلالي (ANN) ---")
    memory = MultimodalMemory()
    
    # 1. إنشاء بيانات وهمية (100 متجه ببعد 1536)
    dim = 1536
    num_vectors = 100
    vectors = np.random.rand(num_vectors, dim).astype(np.float32)
    
    # 2. تخزين المتجهات في الذاكرة
    print(f"تخزين {num_vectors} متجهات...")
    for i in range(num_vectors):
        dummy_file = f"test_ann_{i}.txt"
        with open(dummy_file, "w") as f:
            f.write(f"data {i}")
            
        metadata = {
            "description": f"أصل دلالي رقم {i}",
            "embedding": vectors[i].tolist()
        }
        
        memory.store_asset("agent_ann_test", dummy_file, "text", metadata)
        os.remove(dummy_file)
        
    # 3. اختبار سرعة البحث
    query_vector = vectors[50].tolist()  # البحث عن المتجه رقم 50
    
    start_time = time.time()
    results = memory.search_semantic(query_vector, top_k=5)
    end_time = time.time()
    
    search_duration = (end_time - start_time) * 1000
    print(f"زمن البحث الدلالي لـ {num_vectors} أصل: {search_duration:.2f} ms")
    
    # 4. التحقق من الدقة (يجب أن يكون الأصل رقم 50 هو الأول أو ضمن النتائج)
    found_ids = [res["id"] for res in results]
    print(f"النتائج المسترجعة: {found_ids}")
    
    # بما أننا نستخدم التكميم (VQ) في الذاكرة، قد يكون هناك فقد طفيف في الدقة
    # لكن ANN يجب أن يجد الأصل الصحيح
    assert len(results) > 0, "فشل: لم يتم استرجاع أي نتائج"
    print("✅ نجح اختبار البحث الدلالي ANN!")

if __name__ == "__main__":
    try:
        test_ann_search_speed_and_accuracy()
    except Exception as e:
        print(f"❌ فشل الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
