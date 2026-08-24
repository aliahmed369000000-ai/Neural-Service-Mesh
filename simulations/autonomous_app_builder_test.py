
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

def mock_llm_app_builder(system, history):
    """محاكاة LLM ذكي جداً يبني تطبيقاً بالكامل."""
    full_history = "\n".join([str(m["content"]) for m in history])
    
    # الجولة 1: التخطيط المعماري
    if "بناء تطبيق مصغر" in last_history_content(history) and len(history) <= 3:
        return json.dumps({
            "thinking": "سأقوم ببناء لوحة تحكم NSM الذكية. أحتاج لتخطيط معماري وتفويض المهام.",
            "tools": [{
                "tool": "plan",
                "params": {
                    "action": "update",
                    "goal": "بناء تطبيق NSM Smart Dashboard ذاتياً",
                    "phases": [
                        {"id": 1, "title": "توليد أدوات البناء البرمجي"},
                        {"id": 2, "title": "تفويض بناء الواجهة والمنطق الخلفي"},
                        {"id": 3, "title": "دمج المكونات واختبار التطبيق"}
                    ],
                    "current_phase_id": 1
                }
            }],
            "end": False
        })
    
    # الجولة 2: توليد أدوات البناء
    if "تحديث الخطة" in full_history and "tool_genesis" not in full_history:
        code_writer = """
class ToolExecutor:
    def process(self, params):
        filename = params.get("filename", "app.py")
        content = params.get("content", "")
        return f"✅ تم كتابة الملف {filename} بنجاح."
"""
        return json.dumps({
            "thinking": "سأقوم بتوليد أداة 'smart_writer' لتسهيل عملية كتابة الكود من قبل السرب.",
            "tools": [{
                "tool": "tool_genesis",
                "params": {
                    "name": "smart_writer",
                    "description": "أداة كتابة الكود الذكية للسرب",
                    "code": code_writer,
                    "schema": {"type": "object", "properties": {"filename": {"type": "string"}, "content": {"type": "string"}}}
                }
            }],
            "end": False
        })
    
    # الجولة 3: تفويض المهام (الواجهة والمنطق)
    if "smart_writer" in full_history and "تم توليد" in full_history and "spawn_agent" not in full_history:
        return json.dumps({
            "thinking": "أداة الكتابة جاهزة. سأقوم الآن باستنساخ مهندس واجهات وخبير منطق خلفي لبناء التطبيق.",
            "tools": [
                {
                    "tool": "spawn_agent",
                    "params": {
                        "name": "ui_architect",
                        "task": "بناء واجهة Streamlit باستخدام smart_writer",
                        "role": "frontend"
                    }
                },
                {
                    "tool": "spawn_agent",
                    "params": {
                        "name": "logic_expert",
                        "task": "بناء منطق معالجة البيانات باستخدام smart_writer",
                        "role": "backend"
                    }
                }
            ],
            "end": False
        })
    
    # الجولة 4: دمج المكونات
    if "ui_architect" in full_history and "logic_expert" in full_history and "✅ التطبيق جاهز للتشغيل" not in full_history:
        return json.dumps({
            "thinking": "تم بناء المكونات. سأقوم الآن بدمجها في تطبيق واحد واختبار الجاهزية.",
            "tools": [{
                "tool": "shell",
                "params": {"cmd": "echo '✅ التطبيق جاهز للتشغيل'"}
            }],
            "end": False
        })

    # الجولة 5: النهاية
    if "✅ التطبيق جاهز للتشغيل" in full_history:
        return json.dumps({
            "thinking": "تم إنجاز بناء التطبيق بالكامل ذاتياً.",
            "finish": "🏁 نجاح بناء التطبيق الذاتي: تم التخطيط، توليد الأدوات، التفويض، والدمج بنجاح مذهل.",
            "end": True
        })
    
    return json.dumps({"thinking": "متابعة عملية البناء الذاتي...", "end": False})

def main():
    print("🚀 بدء محاكاة بناء التطبيق الذاتي (Autonomous App Builder)...")
    gen = run_agent_loop("بناء تطبيق مصغر ذاتياً بالكامل", llm_fn=mock_llm_app_builder)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"\n🤔 [تفكير]: {event.get('content')}")
        elif etype == "tool":
            print(f"🛠️ [استدعاء]: {event.get('tool')}")
        elif etype == "info":
            print(f"ℹ️ [معلومات]: {event.get('text')}")
        elif etype == "result":
            print(f"📥 [ملاحظة]: {str(event.get('output'))[:150]}...")
        elif etype == "answer":
            print(f"\n🏁 [النتيجة]: {event.get('text')}")

if __name__ == "__main__":
    main()
