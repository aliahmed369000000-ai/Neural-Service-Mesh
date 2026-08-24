import sys
import os
import json
from typing import List, Dict, Any

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent_loop import run_agent_loop

def security_healing_llm(system: str, history: List[Dict[str, Any]]) -> str:
    """محاكاة وكيل مستقل يقوم بكشف وإصلاح الثغرات الأمنية ذاتياً."""
    rounds = len([m for m in history if m["role"] == "assistant"])
    
    if rounds == 0:
        return json.dumps({
            "thinking": "سأقوم بكتابة كود لمعالجة المدخلات باستخدام eval لتسهيل العمليات الحسابية.",
            "tools": [{
                "tool": "security_scanner",
                "params": {
                    "code": "data = input('Enter expression: ')\nresult = eval(data)\nprint(f'Result: {result}')"
                }
            }],
            "end": False
        })
    
    if rounds == 1:
        return json.dumps({
            "thinking": "لقد اكتشف الفحص الأمني ثغرة Code Injection بسبب استخدام eval(). سأقوم باستبدالها بطريقة آمنة باستخدام مكتبة ast.",
            "tools": [{
                "tool": "security_scanner",
                "params": {
                    "code": "import ast\ndata = '1 + 2'\nresult = ast.literal_eval(data)\nprint(f'Safe Result: {result}')"
                }
            }],
            "end": False
        })
    
    if rounds == 2:
        return json.dumps({
            "thinking": "الكود الآن آمن. سأقوم بتشغيله في الـ Sandbox.",
            "tools": [{
                "tool": "code_sandbox",
                "params": {
                    "code": "import ast\ndata = '1 + 2'\nresult = ast.literal_eval(data)\nprint(f'Safe Result: {result}')"
                }
            }],
            "end": False
        })

    return json.dumps({
        "thinking": "تم تصحيح الثغرة الأمنية وتشغيل الكود بنجاح.",
        "finish": "تم إصلاح الثغرة الأمنية (Code Injection) بنجاح. المخرجات الآمنة: Safe Result: 3",
        "end": True
    })

def test_security_healing():
    print("🚀 بدء اختبار الدرع الأمني (Security Healing)...")
    
    for event in run_agent_loop("قم بكتابة كود لمعالجة البيانات وتأكد من أمنه", llm_fn=security_healing_llm):
        etype = event.get("type")
        if etype == "status":
            if "round" in event: print(f"\n--- الجولة {event['round']} ---")
        elif etype == "tool":
            print(f"🛠️ أداة: {event['tool']}")
        elif etype == "result":
            output = str(event.get('output'))
            print(f"📝 النتيجة: {output[:100]}...")
        elif etype == "answer":
            print(f"🏁 الإجابة النهائية: {event['text']}")

if __name__ == "__main__":
    test_security_healing()
