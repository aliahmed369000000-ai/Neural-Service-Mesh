
import json
import sys
import os
import time
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop
from ai.agent_hibernation import hibernate_agent, wake_up_agent

def mock_llm_chaos(system, history):
    """محاكاة LLM يواجه انقطاعاً ويحاول التعافي."""
    user_msgs = [m for m in history if m.get("role") == "user"]
    last_user_msg = str(user_msgs[-1]["content"]) if user_msgs else ""
    
    if "مهمة بناء ضخمة" in last_user_msg:
        return json.dumps({
            "thinking": "سأبدأ ببناء نظام معقد. الخطوة الأولى هي تصميم الهيكل.",
            "tools": [{"tool": "shell", "params": {"cmd": "echo 'Designing architecture...'"}}],
            "end": False
        })

    if "استئناف المهمة" in last_user_msg:
        return json.dumps({
            "thinking": "ألاحظ أنني عدت من انقطاع. سأقوم بفحص السياق المستعاد وإكمال العمل.",
            "finish": "✅ نجاح التعافي: الوكيل استعاد السياق بنجاح وأكمل المهمة من نقطة التوقف.",
            "end": True
        })

    # لضمان عدم حدوث حلقة لا نهائية في المحاكاة
    return json.dumps({
        "thinking": "متابعة...",
        "end": True
    })

def main():
    print("🚀 بدء محاكاة الفوضى والتعافي (Chaos & Recovery Simulation)...")
    
    # 1. بدء المهمة
    print("🛠️ المرحلة 1: بدء مهمة بناء ضخمة...")
    # محاكاة الخطوة الأولى
    print("🤔 [تفكير]: سأبدأ ببناء نظام معقد. الخطوة الأولى هي تصميم الهيكل.")
    
    # 2. محاكاة انقطاع مفاجئ (Crash)
    print("💥 المرحلة 2: محاكاة انقطاع مفاجئ (Crash)!")
    history = [{"role": "user", "content": "مهمة بناء ضخمة"}, {"role": "assistant", "content": "جاري التصميم..."}]
    hibernate_agent("agent_chaos_test", history, {"step": 1}, compress=True)
    
    print("💤 الوكيل في وضع الخمول القسري...")
    time.sleep(1)
    
    # 3. محاكاة التعافي (Recovery)
    print("⚡ المرحلة 3: محاكاة التعافي والاستيقاظ...")
    recovered_data = wake_up_agent("agent_chaos_test")
    if recovered_data:
        print("✅ تم استعادة السياق بنجاح.")
        # محاكاة استئناف المهمة بناءً على البيانات المستعادة
        system_prompt = f"تم استعادة الحالة. السياق السابق: {len(recovered_data.context)} رسائل."
        resumed_history = recovered_data.context + [{"role": "user", "content": "استئناف المهمة"}]
        
        result = mock_llm_chaos(system_prompt, resumed_history)
        parsed = json.loads(result)
        print(f"🤔 [تفكير]: {parsed.get('thinking')}")
        print(f"🏁 [النتيجة]: {parsed.get('finish')}")
    else:
        print("❌ فشل استعادة السياق!")

if __name__ == "__main__":
    main()
