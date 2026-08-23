import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.agent_loop import TOOL_REGISTRY

def test_sync_auto_heal_logic():
    print("🚀 اختبار نظام التعافي التلقائي (Auto-Heal) للمزامنة...")
    
    # محاكاة فشل الملف (خطأ قابل للتشخيص)
    params = {"video_id": "vid_fail", "audio_path": "missing_file.wav"}
    
    print("🎬 استدعاء video_sync مع ملف مفقود...")
    result_raw = TOOL_REGISTRY["video_sync"].executor(params)
    
    print(f"النتيجة النهائية: {result_raw}")
    
    if "Auto-Heal Failed" in result_raw:
        print("✅ نجاح: نظام التعافي حاول الإصلاح وأبلغ عن الفشل النهائي بشكل صحيح.")
    else:
        print("⚠️ النتيجة لم تشر إلى نظام التعافي.")

if __name__ == "__main__":
    test_sync_auto_heal_logic()
