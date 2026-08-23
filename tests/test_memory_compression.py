import sys
import os
import json

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.agent_hibernation import AgentState, hibernate_agent

def test_memory_compression():
    print("🚀 اختبار ضغط الذاكرة الديناميكي...")
    
    agent_id = "heavy_agent"
    
    # 1. إنشاء سياق كبير (أكثر من 10 رسائل)
    context = [{"role": "system", "content": "نظام"}] + \
              [{"role": "user", "content": f"رسالة رقم {i}"} for i in range(20)]
              
    # 2. إنشاء ذاكرة مزامنة كبيرة (أكثر من 20 نقطة)
    multimodal_memory = {
        "vid_1": {
            "multimodal_sync": [{"ts": i, "text": "كلام"} for i in range(50)]
        }
    }
    
    state = AgentState(agent_id, context, {"tasks": []}, multimodal_memory=multimodal_memory)
    
    initial_size = len(json.dumps(state.to_dict())) / 1024
    print(f"الحجم الابتدائي: {initial_size:.2f} KB")
    
    # 3. تنفيذ الضغط
    state.compress(target_size_kb=1) # نطلب ضغطاً شديداً جداً لإجبار الضغط على ملف صغير
    
    final_size = len(json.dumps(state.to_dict())) / 1024
    print(f"الحجم بعد الضغط: {final_size:.2f} KB")
    
    # التحقق من النتائج
    if final_size < initial_size:
        print(f"✅ نجاح: تم تقليل الحجم بنسبة {((initial_size - final_size) / initial_size) * 100:.1f}%")
        print(f"عدد الرسائل المتبقية: {len(state.context)}")
        print(f"عدد نقاط المزامنة المتبقية: {len(state.multimodal_memory['vid_1']['multimodal_sync'])}")
    else:
        print("❌ فشل: لم يتم تقليل الحجم.")

if __name__ == "__main__":
    test_memory_compression()
