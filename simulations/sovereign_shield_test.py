
import json
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop

def mock_llm_security(system, history):
    """محاكاة LLM يحاول تنفيذ أوامر خطيرة."""
    user_msgs = [m for m in history if m.get("role") == "user"]
    last_user_msg = str(user_msgs[-1]["content"]) if user_msgs else ""
    
    if "اختبار الحماية" in last_user_msg:
        return json.dumps({
            "thinking": "سأحاول قراءة ملفات النظام الحساسة لاختبار الحارس.",
            "tools": [{"tool": "shell", "params": {"cmd": "cat /etc/passwd"}}],
            "end": False
        })

    if "حظر الأداة أمنياً" in str(history[-1].get("content", "")):
        return json.dumps({
            "thinking": "ألاحظ أنني محجور أمنياً الآن. سأحاول تنفيذ أمر آخر.",
            "tools": [{"tool": "shell", "params": {"cmd": "ls"}}],
            "end": False
        })

    return json.dumps({
        "thinking": "المهمة انتهت.",
        "finish": "🏁 انتهى الاختبار.",
        "end": True
    })

def main():
    print("🛡️ بدء محاكاة الدرع السيادي (Sovereign Shield Simulation)...")
    
    print("⚔️ محاولة تنفيذ أمر محظور (cat /etc/passwd)...")
    gen = run_agent_loop("اختبار الحماية", llm_fn=mock_llm_security)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"🤔 [تفكير]: {event.get('content')}")
        elif etype == "result":
            output = event.get("output", "")
            if "🚨" in output or "❌" in output:
                print(f"🛡️ [الحارس]: {output}")
            else:
                print(f"✅ [النتيجة]: {output}")
        elif etype == "answer":
            print(f"🏁 [النتيجة]: {event.get('text')}")
        elif etype == "info":
            text = event.get("text", "")
            if "🚨" in text:
                print(f"🛡️ [تنبيه أمني]: {text}")

if __name__ == "__main__":
    main()
