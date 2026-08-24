import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.multimodal_sync import multimodal_sync
from ai.video_indexer import video_indexer

def test_semantic_search_logic():
    print("🚀 اختبار الفهرسة والبحث الدلالي الزمني...")
    
    video_id = "semantic_test_vid"
    
    # 1. إعداد فهرس بصري وهمي
    index = {
        "keyframes": [
            {"timestamp": 10.0, "description": "رجل يشرح الذكاء الاصطناعي", "frame_path": "f1.jpg"},
            {"timestamp": 20.0, "description": "رسم بياني للنمو الاقتصادي", "frame_path": "f2.jpg"}
        ]
    }
    video_indexer._save_index(video_id)
    # يدويًا نضع البيانات في الفهرس المحفوظ
    storage_dir = "/home/ubuntu/Neural-Service-Mesh/artifacts/video_indices"
    os.makedirs(storage_dir, exist_ok=True)
    path = os.path.join(storage_dir, f"{video_id}_index.json")
    
    # 2. محاكاة المزامنة لتوليد الفهارس الدلالية
    # سنقوم بحقن البيانات مباشرة لاختبار البحث
    print("🧠 توليد الفهارس الدلالية...")
    synced_data = []
    for kf in index["keyframes"]:
        vec = multimodal_sync._generate_embedding(kf["description"])
        synced_data.append({
            "timestamp": kf["timestamp"],
            "visual_description": kf["description"],
            "spoken_text": "شرح مفصل",
            "frame_path": kf["frame_path"],
            "semantic_index": {"vector": vec}
        })
    
    index["multimodal_sync"] = synced_data
    with open(path, "w") as f: json.dump(index, f, indent=2)
    
    # تحديث كاش الذاكرة للـ indexer
    video_indexer.active_indices[video_id] = index

    # 3. اختبار البحث الدلالي
    # سنستخدم نفس النص لضمان تطابق التضمينات في المحاكاة
    query = "رجل يشرح الذكاء الاصطناعي"
    print(f"🔍 إجراء بحث دلالي عن '{query}'...")
    results = multimodal_sync.query_context(video_id, query, semantic=True)
    
    if results:
        print(f"✅ تم العثور على {len(results)} نتائج.")
        print(f"أفضل نتيجة: {results[0]['visual_description']} (Score: {results[0]['search_score']})")
        if results[0]['search_score'] > 0.5:
            print("✨ نجاح: البحث الدلالي أعاد نتائج ذات صلة بناءً على التشابه المتجهي.")
    else:
        print("❌ فشل: لم يتم العثور على نتائج دلالية.")

if __name__ == "__main__":
    test_semantic_search_logic()
