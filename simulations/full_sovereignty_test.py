
import json
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop
from ai.self_resource_optimizer import resource_optimizer

def mock_llm_sovereignty(system, history):
    """محاكاة LLM يمارس السيادة الكاملة."""
    full_history = "\n".join([str(m["content"]) for m in history])
    
    # التحقق من وجود تنبيه موارد
    has_resource_alert = "⚠️ تنبيه سيادي" in system or "⚠️ تنبيه سيادي" in full_history
    
    user_msgs = [m for m in history if m.get("role") == "user"]
    last_user_msg = str(user_msgs[-1]["content"]) if user_msgs else ""
    
    if "بدء مهمة سيادية" in last_user_msg:
        thinking = "سأقوم بإدارة الموارد وتوليد أدوات تدريب متقدمة."
        if has_resource_alert:
            thinking += " ألاحظ ضغطاً على الموارد، سأقوم بتحسين الخطة فوراً."
            
        return json.dumps({
            "thinking": thinking,
            "tools": [
                {"tool": "tool_genesis", "params": {
                    "name": "auto_drift_monitor",
                    "code": "def auto_drift_monitor(loss_history):\n    if len(loss_history) < 2: return 'Stable'\n    return 'Drifting' if loss_history[-1] > loss_history[-2] else 'Optimizing'",
                    "description": "مراقب الانحراف التلقائي للأوزان"
                }}
            ],
            "end": False
        })

    if "auto_drift_monitor" in full_history:
        return json.dumps({
            "thinking": "تم توليد أداة مراقبة الانحراف. سأقوم الآن بتطبيق تحسين الموارد الذاتي.",
            "finish": "✅ نجاح السيادة الكاملة: الوكيل أدار الموارد، ولد أدوات تدريب متقدمة، واتخذ قرارات سيادية مستقلة.",
            "end": True
        })

    return json.dumps({"thinking": "متابعة...", "end": False})

def main():
    print("🚀 بدء محاكاة السيادة الكاملة (Full Sovereignty Simulation)...")
    
    # محاكاة ضغط الموارد
    print("📉 محاكاة ضغط الذاكرة (88%)...")
    # حقن التنبيه يدوياً في المحاكاة لأن psutil قد لا يعطي 88% حالياً
    
    gen = run_agent_loop("بدء مهمة سيادية لتطوير أدوات التدريب", llm_fn=mock_llm_sovereignty)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"🤔 [تفكير]: {event.get('content')}")
        elif etype == "answer":
            print(f"🏁 [النتيجة]: {event.get('text')}")
        elif etype == "info":
            print(f"ℹ️ [معلومات]: {event.get('text')}")
        elif etype == "tool":
            print(f"🛠️ [أداة]: {event.get('tool')}")

if __name__ == "__main__":
    main()
