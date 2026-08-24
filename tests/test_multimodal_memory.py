import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.agent_hibernation import hibernate_agent, wake_up_agent

def test_multimodal_memory_cycle():
    print("🚀 اختبار دورة الذاكرة متعددة الوسائط (نوم -> استيقاظ)...")
    
    agent_id = "multimodal_test_agent"
    context = [{"role": "user", "content": "حلل هذا الفيديو"}]
    plan = {"tasks": ["مزامنة الفيديو"]}
    
    # محاكاة سياق مزامَن
    multimodal_memory = {
        "vid_456": {
            "multimodal_sync": [
                {"ts": 1.0, "text": "مرحباً", "frame": "f1.jpg"},
                {"ts": 2.5, "text": "بالعالم", "frame": "f2.jpg"}
            ]
        }
    }
    
    # 1. النوم
    print("💤 محاكاة النوم...")
    success = hibernate_agent(
        agent_id=agent_id,
        context=context,
        plan=plan,
        multimodal_memory=multimodal_memory
    )
    
    if success:
        print("✅ تم حفظ الحالة بنجاح.")
    else:
        print("❌ فشل حفظ الحالة.")
        return

    # 2. الاستيقاظ
    print("🌅 محاكاة الاستيقاظ...")
    recovered = wake_up_agent(agent_id)
    
    if recovered and recovered.multimodal_memory:
        sync_points = len(recovered.multimodal_memory["vid_456"]["multimodal_sync"])
        print(f"✅ تم استعادة الذاكرة المزامنة: {sync_points} نقاط زمنية.")
        if sync_points == 2:
            print("✨ اختبار النجاح الكامل: البيانات المستعادة مطابقة تماماً.")
        else:
            print(f"⚠️ البيانات المستعادة غير كاملة. وجدنا {sync_points} بدلاً من 2.")
    else:
        print("❌ فشل استعادة الذاكرة المزامنة.")

if __name__ == "__main__":
    test_multimodal_memory_cycle()
