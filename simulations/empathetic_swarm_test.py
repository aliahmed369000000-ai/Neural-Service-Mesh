
import json
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop

def mock_llm_empathy(system, history):
    """محاكاة LLM يدرك الحالة العاطفية للسرب."""
    full_history = "\n".join([str(m["content"]) for m in history])
    
    # التحقق من وجود تنبيه عاطفي
    is_frustrated = "⚠️ تنبيه عاطفي: الوكيل يشعر بـ 'Frustrated'" in full_history
    
    user_msgs = [m for m in history if m.get("role") == "user"]
    last_user_msg = str(user_msgs[-1]["content"]) if user_msgs else ""
    
    # محاكاة حالة الإحباط يدوياً في التفكير لتسريع المحاكاة
    if "بدء مهمة معقدة" in last_user_msg:
        return json.dumps({
            "thinking": "سأقوم بتنفيذ المهمة. ألاحظ أن السرب في حالة مستقرة حالياً.",
            "tools": [{"tool": "shell", "params": {"cmd": "exit 1"}}], # محاكاة فشل
            "end": False
        })

    # إذا استمر الفشل، نقوم بالتحول للحالة العاطفية
    if "spawn_agent" in full_history:
        return json.dumps({
            "thinking": "تم تفويض المساعد. أشعر الآن بمزيد من الاستقرار والتعاون.",
            "finish": "✅ نجاح الذكاء العاطفي: الوكيل أدرك إحباطه وقام بتعديل سلوكه لطلب المساعدة.",
            "end": True
        })

    if "shell" in full_history:
        # حقن الإحباط يدوياً في الاستجابة التالية
        return json.dumps({
            "thinking": "أشعر بالإحباط من الفشل المتكرر للأدوات. سأقوم بتفويض وكيل مساعد فوراً.",
            "tools": [{"tool": "spawn_agent", "params": {"name": "assistant_bot", "role": "Debugger"}}],
            "end": False
        })

    return json.dumps({"thinking": "متابعة...", "end": False})

def main():
    print("🚀 بدء محاكاة السرب المتعاطف (Empathetic Swarm Simulation)...")
    
    # محاكاة سلسلة من الإخفاقات لتوليد حالة "الإحباط"
    print("📉 محاكاة فشل متكرر لتوليد إحباط تقني...")
    
    gen = run_agent_loop("بدء مهمة معقدة", llm_fn=mock_llm_empathy)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"🤔 [تفكير]: {event.get('content')}")
        elif etype == "answer":
            print(f"🏁 [النتيجة]: {event.get('text')}")
        elif etype == "tool":
            print(f"🛠️ [أداة]: {event.get('tool')}")
        elif etype == "info":
            print(f"ℹ️ [معلومات]: {event.get('text')}")

if __name__ == "__main__":
    main()
