import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.agent_loop import TOOL_REGISTRY

def test_tool_registration():
    print("🚀 فحص تسجيل أداة video_sync في Registry...")
    if "video_sync" in TOOL_REGISTRY:
        print("✅ أداة video_sync مسجلة بنجاح.")
        spec = TOOL_REGISTRY["video_sync"]
        print(f"وصف الأداة: {spec.description}")
        print(f"مخطط البارامترات: {spec.params_schema}")
    else:
        print("❌ أداة video_sync غير موجودة في Registry.")
        sys.exit(1)

def test_tool_execution_mock():
    print("\n🚀 اختبار تنفيذ الأداة (Mock)...")
    # محاكاة فشل منطقي بسبب عدم وجود فيديو (لاختبار مسار التنفيذ)
    params = {"video_id": "non_existent", "audio_path": "fake.wav"}
    result = TOOL_REGISTRY["video_sync"].executor(params)
    print(f"نتيجة التنفيذ المتوقعة: {result}")
    
    if "الفهرس البصري للفيديو غير موجود" in result or "❌" in result:
        print("✅ الأداة تعمل وتتعامل مع الأخطاء بشكل صحيح.")
    else:
        print("⚠️ نتيجة غير متوقعة.")

if __name__ == "__main__":
    test_tool_registration()
    test_tool_execution_mock()
