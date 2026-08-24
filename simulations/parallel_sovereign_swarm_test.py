
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

def mock_llm_parallel(system, history):
    """محاكاة LLM ذكي جداً يدير سرباً بالتوازي."""
    full_history = "\n".join([str(m["content"]) for m in history])
    
    # الجولة 1: التخطيط الضخم
    if "اختبار السرب الموازي" in last_history_content(history) and len(history) <= 3:
        return json.dumps({
            "thinking": "المهمة تتطلب معالجة بيانات ضخمة وتأمينها. سأقوم بالتخطيط لتوليد أدوات وتفويض وكلاء بالتوازي.",
            "tools": [{
                "tool": "plan",
                "params": {
                    "action": "update",
                    "goal": "بناء منصة ذكاء موازية مؤمنة",
                    "phases": [
                        {"id": 1, "title": "توليد أدوات التحليل والتشفير بالتوازي"},
                        {"id": 2, "title": "استنساخ سرب من الوكلاء للتنفيذ"},
                        {"id": 3, "title": "دمج النتائج النهائية"}
                    ],
                    "current_phase_id": 1
                }
            }],
            "end": False
        })
    
    # الجولة 2: توليد أدوات متعددة بالتوازي
    if "تحديث الخطة" in full_history and "tool_genesis" not in full_history:
        code_analyzer = """
class ToolExecutor:
    def process(self, params):
        data = params.get("data", [])
        return {"count": len(data), "summary": "Data analyzed successfully"}
"""
        code_securer = """
class ToolExecutor:
    def process(self, params):
        return "SECURE_SHIELD_ACTIVATED"
"""
        return json.dumps({
            "thinking": "سأقوم بتوليد أدوات التحليل والأمن الآن في خطوة واحدة متوازية.",
            "tools": [
                {
                    "tool": "tool_genesis",
                    "params": {
                        "name": "data_analyzer_pro",
                        "description": "أداة تحليل بيانات متقدمة",
                        "code": code_analyzer,
                        "schema": {"type": "object"}
                    }
                },
                {
                    "tool": "tool_genesis",
                    "params": {
                        "name": "swarm_securer",
                        "description": "أداة تأمين السرب",
                        "code": code_securer,
                        "schema": {"type": "object"}
                    }
                }
            ],
            "end": False
        })
    
    # الجولة 3: تفويض وكلاء متعددين بالتوازي
    if "data_analyzer_pro" in full_history and "swarm_securer" in full_history and "spawn_agent" not in full_history:
        return json.dumps({
            "thinking": "الأدوات جاهزة. سأقوم الآن باستنساخ وكيلين للعمل بالتوازي على المهام.",
            "tools": [
                {
                    "tool": "spawn_agent",
                    "params": {
                        "name": "analyst_bot",
                        "task": "تحليل قاعدة البيانات باستخدام data_analyzer_pro",
                        "role": "analyst"
                    }
                },
                {
                    "tool": "spawn_agent",
                    "params": {
                        "name": "guard_bot",
                        "task": "تفعيل الدرع الأمني باستخدام swarm_securer",
                        "role": "security"
                    }
                }
            ],
            "end": False
        })
    
    # الجولة 4: دمج النتائج والنهاية
    if "analyst_bot" in full_history and "guard_bot" in full_history:
        return json.dumps({
            "thinking": "السرب أكمل المهام بالتوازي. تم دمج التحليلات مع الحماية الأمنية.",
            "finish": "✅ نجاح محاكاة السرب السيادي الموازي: تم توليد أداتين واستنساخ وكيلين بالتوازي وبنجاح تام.",
            "end": True
        })
    
    return json.dumps({"thinking": "متابعة التنسيق الموازي...", "end": False})

def main():
    print("🚀 بدء محاكاة السرب السيادي الموازي (Parallel Sovereign Swarm)...")
    gen = run_agent_loop("اختبار السرب الموازي السيادي", llm_fn=mock_llm_parallel)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"\n🤔 [تفكير]: {event.get('content')}")
        elif etype == "tool":
            print(f"🛠️ [استدعاء]: {event.get('tool')}")
        elif etype == "info":
            print(f"ℹ️ [معلومات]: {event.get('text')}")
        elif etype == "result":
            print(f"📥 [ملاحظة]: {str(event.get('output'))[:100]}...")
        elif etype == "answer":
            print(f"\n🏁 [النتيجة]: {event.get('text')}")

if __name__ == "__main__":
    main()
