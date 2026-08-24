import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.multimodal_sync import MultimodalSyncManager
from ai.video_indexer import video_indexer
from ai.agent_loop import _tool_video_sentiment

def test_sentiment_analysis():
    print("🚀 اختبار تحليل المشاعر المتزامن...")
    
    video_id = "emotional_video"
    sync_manager = MultimodalSyncManager()
    
    # 1. إعداد بيانات محاكاة للفيديو
    index = video_indexer.create_index(video_id)
    video_indexer.add_keyframe(video_id, 10.0, "f1.jpg", "رجل يبتسم بنجاح", ["فرح", "نجاح"])
    video_indexer.add_keyframe(video_id, 20.0, "f2.jpg", "رسم بياني يظهر فشل ذريع", ["خسارة", "مشكلة"])
    
    # 2. محاكاة المزامنة مع تحليل المشاعر
    # سنقوم بحقن البيانات مباشرة لاختبار الأداة
    print("🎭 تحليل المشاعر المتزامن...")
    synced_data = []
    
    # الحالة الإيجابية
    item1 = {
        "timestamp": 10.0,
        "visual_description": "رجل يبتسم بنجاح",
        "spoken_text": "نحن فخورون بهذا التطور الكبير والنجاح",
        "sentiment": sync_manager._analyze_sentiment("نحن فخورون بهذا التطور الكبير والنجاح", "رجل يبتسم بنجاح")
    }
    
    # الحالة السلبية
    item2 = {
        "timestamp": 20.0,
        "visual_description": "رسم بياني يظهر فشل ذريع",
        "spoken_text": "هناك مشكلة كبيرة أدت إلى انخفاض حاد وفشل",
        "sentiment": sync_manager._analyze_sentiment("هناك مشكلة كبيرة أدت إلى انخفاض حاد وفشل", "رسم بياني يظهر فشل ذريع")
    }
    
    synced_data = [item1, item2]
    
    # تحديث الفهرس
    full_index = video_indexer.load_index(video_id)
    full_index["multimodal_sync"] = synced_data
    video_indexer.active_indices[video_id] = full_index
    video_indexer._save_index(video_id)
    
    # 3. اختبار أداة video_sentiment
    print("🔍 استدعاء أداة video_sentiment...")
    result_json = _tool_video_sentiment({"video_id": video_id})
    result = json.loads(result_json)
    
    print(f"النتيجة: {result_json}")
    
    # التحقق
    if result.get("ok"):
        print(f"✅ نجاح: تم تحليل المشاعر الإجمالية ({result['overall_sentiment']})")
        print(f"متوسط النتيجة: {result['average_score']}")
    else:
        print("❌ فشل: لم يتم الحصول على نتائج صحيحة.")

if __name__ == "__main__":
    test_sentiment_analysis()
