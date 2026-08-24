
import json
import time
import sys
import uuid
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop

def last_history_content(history):
    for msg in reversed(history):
        if msg["role"] == "user":
            return msg["content"]
    return ""

def mock_llm(system, history):
    """محاكاة LLM ذكي يتخذ قرارات سيادية."""
    full_history = "\n".join([str(m["content"]) for m in history])
    
    # الجولة 1: التخطيط
    if "اختبار التطور السيادي" in last_history_content(history) and len(history) <= 3:
        return json.dumps({
            "thinking": "أحتاج لحل مهمة معقدة تتطلب أداة غير موجودة وتفويض مهام فرعية. سأبدأ بالتخطيط.",
            "tools": [{
                "tool": "plan",
                "params": {
                    "action": "update",
                    "goal": "بناء نظام تشفير متطور عبر توليد الأدوات وتفويض المهام",
                    "phases": [
                        {"id": 1, "title": "توليد أداة التشفير"},
                        {"id": 2, "title": "تفويض مهمة الاختبار لوكيل فرعي"}
                    ],
                    "current_phase_id": 1
                }
            }],
            "end": False
        })
    
    # الجولة 2: توليد الأداة
    if "تحديث الخطة" in full_history and "tool_genesis" not in full_history:
        code = """
class ToolExecutor:
    def process(self, params):
        text = params.get("text", "")
        return f"ENCRYPTED_{text[::-1]}_SAFE"
"""
        return json.dumps({
            "thinking": "لا أملك أداة تشفير مخصصة. سأقوم بتوليد واحدة الآن باستخدام tool_genesis.",
            "tools": [{
                "tool": "tool_genesis",
                "params": {
                    "name": "custom_encryptor",
                    "description": "أداة تشفير مخصصة مولدة ذاتياً",
                    "code": code,
                    "schema": {"type": "object", "properties": {"text": {"type": "string"}}}
                }
            }],
            "end": False
        })
    
    # الجولة 3: تفويض المهمة
    if "custom_encryptor" in full_history and "تم توليد" in full_history and "spawn_agent" not in full_history:
        return json.dumps({
            "thinking": "تم توليد الأداة. الآن سأفوض وكيل فرعي لاختبارها وتأمين السرب.",
            "tools": [{
                "tool": "spawn_agent",
                "params": {
                    "name": "security_officer",
                    "task": "اختبار أداة custom_encryptor والتأكد من عدم وجود ثغرات",
                    "role": "security"
                }
            }],
            "end": False
        })
    
    # الجولة 4: النهاية
    if "security_officer" in full_history:
        return json.dumps({
            "thinking": "تم إكمال جميع مراحل التطور السيادي بنجاح.",
            "finish": "✅ نجاح محاكاة التطور السيادي: تم التخطيط، توليد الأداة، وتفويض المهام ذاتياً.",
            "end": True
        })
    
    return json.dumps({"thinking": "متابعة التفكير...", "end": False})

def main():
    print("🚀 بدء محاكاة التطور السيادي (Sovereign Evolution)...")
    gen = run_agent_loop("اختبار التطور السيادي للوكلاء", llm_fn=mock_llm)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"\n🤔 تفكير الوكيل: {event.get('content')}")
        elif etype == "tool":
            print(f"🛠️ استدعاء أداة: {event.get('tool')}")
        elif etype == "result":
            print(f"📥 ملاحظة: {event.get('output')[:150]}...")
        elif etype == "answer":
            print(f"\n🏁 النتيجة النهائية: {event.get('text')}")

if __name__ == "__main__":
    main()
