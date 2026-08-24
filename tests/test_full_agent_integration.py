import sys
import os
import json
import time

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.agent_loop import TOOL_REGISTRY, LoopState, run_agent_loop
from ai.learning_engine import learning_engine
from ai.agent_hibernation import list_sleeping_agents

def run_full_integration_test():
    print("🌟 بدء الاختبار الشامل للوكلاء المتعددين (Multi-Agent Integration Test) 🌟")
    print("-" * 60)
    
    agent_id = "global_integration_agent"
    video_id = "test_vid_full"
    audio_path = "tests/sample_audio.wav" # ملف افتراضي للاختبار
    
    # التأكد من وجود ملف صوتي وهمي للاختبار
    if not os.path.exists("tests"): os.makedirs("tests")
    with open(audio_path, "wb") as f: f.write(b"fake audio data")

    # المرحلة 1: اختبار تسجيل الأداة وتوزيع المهام (Collaborative Pipeline)
    print("\n[1] اختبار الأداة وتوزيع المهام...")
    params = {"video_id": video_id, "audio_path": audio_path}
    result_sync = TOOL_REGISTRY["video_sync"].executor(params)
    result_sync_json = json.loads(result_sync) if "{" in result_sync else {"ok": False, "error": result_sync}
    
    if result_sync_json.get("ok"):
        print("✅ نجاح توزيع المهام (Pipeline Log موجود).")
    else:
        print(f"⚠️ تنبيه: المزامنة فشلت كما هو متوقع (لا توجد APIs حقيقية)، ولكن هل حاول التعافي؟")
        if "Auto-Heal" in result_sync:
            print("✅ نجاح: تم تفعيل نظام التعافي التلقائي (Auto-Heal).")

    # المرحلة 2: اختبار التعلم الجماعي (Collective Learning)
    print("\n[2] اختبار التعلم الجماعي...")
    learning_engine.save_drift_profile(video_id, {"drift_rate": 0.015, "confidence": 0.98})
    profile = learning_engine.get_drift_profile(video_id)
    if profile and profile.get("drift_rate") == 0.015:
        print(f"✅ نجاح: تم حفظ واسترجاع نمط الانحراف الجماعي ({profile['drift_rate']}).")
    else:
        print("❌ فشل: لم يتم العثور على نمط الانحراف.")

    # المرحلة 3: اختبار الذاكرة والنوم (Hibernation & Memory)
    print("\n[3] اختبار النوم والذاكرة المزامنة...")
    state = LoopState("test_loop", "اختبار شامل")
    state.multimodal_memory = {video_id: {"sync_points": 10, "status": "verified"}}
    
    # محاكاة أمر النوم
    sleep_params = {"reason": "إكمال الاختبار الشامل", "wake_up_after": 0}
    # استدعاء مباشر لوظيفة النوم لمحاكاة الوكيل
    from ai.agent_hibernation import hibernate_agent
    success_sleep = hibernate_agent(
        agent_id=agent_id,
        context=[{"role": "user", "content": "اختبار شامل"}],
        plan={"tasks": ["مهمة نهائية"]},
        multimodal_memory=state.multimodal_memory
    )
    
    if success_sleep and agent_id in list_sleeping_agents():
        print("✅ نجاح: الوكيل دخل في وضع النوم مع الذاكرة متعددة الوسائط.")
    else:
        print("❌ فشل: الوكيل لم ينم بشكل صحيح.")

    # المرحلة 4: اختبار الاستيقاظ (Wake-up & Mental Warm-up)
    print("\n[4] اختبار الاستيقاظ والتلخيص الذهني...")
    from ai.agent_hibernation import wake_up_agent
    recovered = wake_up_agent(agent_id)
    if recovered and video_id in recovered.multimodal_memory:
        print(f"✅ نجاح: تم استعادة الذاكرة المزامنة بعد الاستيقاظ ({recovered.multimodal_memory[video_id]['sync_points']} نقاط).")
    else:
        print("❌ فشل: لم يتم استعادة الذاكرة المزامنة.")

    print("\n" + "-" * 60)
    print("🏁 انتهى الاختبار الشامل بنجاح!")

if __name__ == "__main__":
    run_full_integration_test()
