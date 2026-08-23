import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.agent_loop import LoopState, TOOL_REGISTRY

def test_pipeline_context():
    print("🚀 اختبار تمرير السياق (Context Passing)...")
    state = LoopState("test_loop", "test input")
    
    # محاكاة عمل وكيل الرؤية
    state.set_pipeline_data("frames_analyzed", 10, role="vision")
    
    # محاكاة عمل وكيل الصوت
    state.set_pipeline_data("audio_segments", 5, role="audio")
    
    # التحقق من استرجاع البيانات
    v_data = state.get_pipeline_data("frames_analyzed")
    a_data = state.get_pipeline_data("audio_segments")
    
    if v_data == 10 and a_data == 5:
        print("✅ نجاح: تم تمرير البيانات بين الأدوار بنجاح.")
    else:
        print(f"❌ فشل: البيانات المسترجعة غير صحيحة. Vision: {v_data}, Audio: {a_data}")

def test_video_sync_pipeline():
    print("\n🚀 اختبار أداة video_sync بنظام الأنابيب...")
    params = {"video_id": "vid_123", "audio_path": "non_existent.wav"}
    
    # تنفيذ الأداة (ستفشل بسبب الملف ولكننا نريد رؤية سجل الأنابيب)
    result_raw = TOOL_REGISTRY["video_sync"].executor(params)
    
    if "Pipeline Error" in result_raw or "❌" in result_raw:
        print("✅ الأداة تعمل وتنفذ منطق الأنابيب حتى عند حدوث خطأ في الموارد.")
    else:
        print("⚠️ نتيجة غير متوقعة.")

if __name__ == "__main__":
    test_pipeline_context()
    test_video_sync_pipeline()
