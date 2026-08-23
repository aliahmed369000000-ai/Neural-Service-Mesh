import sys
import os
import json
from typing import List, Dict, Any

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent_loop import run_agent_loop, register_tool, ToolSpec

def mock_llm(system: str, history: List[Dict[str, Any]]) -> str:
    """محاكاة تفكير Manus لحل مهمة معقدة."""
    last_msg = history[-1]["content"]
    
    # الجولة 1: التخطيط
    if "حل مشكلة حسابية معقدة" in last_msg:
        return json.dumps({
            "thinking": "سأقوم أولاً بالتخطيط لحل هذه المهمة عبر البحث عن الصيغة الرياضية ثم كتابة كود للتحقق.",
            "tools": [{
                "tool": "plan",
                "params": {
                    "action": "update",
                    "goal": "حساب متسلسلة فيبوناتشي حتى الرقم 10",
                    "phases": [
                        {"id": 1, "title": "البحث عن الخوارزمية"},
                        {"id": 2, "title": "كتابة وتشغيل الكود"}
                    ],
                    "current_phase_id": 1
                }
            }],
            "end": False
        })
    
    # الجولة 2: البحث
    if "تم تحديث الخطة" in last_msg:
        return json.dumps({
            "thinking": "الآن سأبحث عن أفضل طريقة لتنفيذ فيبوناتشي في بايثون.",
            "tools": [{
                "tool": "web_explorer",
                "params": {"query": "fibonacci python implementation"}
            }],
            "end": False
        })
    
    # الجولة 3: كتابة الكود
    if "web_explorer" in last_msg:
        return json.dumps({
            "thinking": "وجدت الطريقة، سأقوم الآن بتشغيل الكود في الـ Sandbox.",
            "tools": [{
                "tool": "code_sandbox",
                "params": {"code": "def fib(n): return n if n<=1 else fib(n-1)+fib(n-2)\nprint([fib(i) for i in range(10)])"}
            }],
            "end": False
        })
    
    # النهاية
    return json.dumps({
        "thinking": "تم إنجاز المهمة بنجاح.",
        "finish": "النتائج جاهزة: [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]",
        "end": True
    })

def test_autonomous_reasoning():
    print("🚀 بدء اختبار التفكير المستقل (Autonomous Reasoning)...")
    
    # تشغيل الحلقة مع الـ Mock LLM
    for event in run_agent_loop("حل مشكلة حسابية معقدة", llm_fn=mock_llm):
        etype = event.get("type")
        if etype == "status":
            if "round" in event:
                print(f"\n--- الجولة {event['round']} ---")
            else:
                print(f"\n--- الحالة: {event.get('status')} ---")
        elif etype == "tool":
            print(f"🛠️ أداة: {event['tool']} | البارامترات: {event['params']}")
        elif etype == "result":
            print(f"📝 النتيجة: {str(event.get('output'))[:100]}...")
        elif etype == "info":
            print(f"ℹ️ معلومة: {event.get('text')}")
        elif etype == "answer":
            print(f"🏁 الإجابة النهائية: {event['text']}")

if __name__ == "__main__":
    test_autonomous_reasoning()
