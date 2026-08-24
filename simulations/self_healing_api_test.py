import sys
import os
import json
from typing import List, Dict, Any

# إضافة مسار المشروع
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent_loop import run_agent_loop

def self_healing_llm(system: str, history: List[Dict[str, Any]]) -> str:
    """محاكاة وكيل مستقل يقوم بالبناء والتصحيح الذاتي."""
    last_msg = history[-1]["content"]
    
    # الجولة 1: محاولة بناء الـ API مع خطأ متعمد (نسيان import json)
    if "بناء واجهة برمجة تطبيقات" in last_msg:
        return json.dumps({
            "thinking": "سأقوم ببناء واجهة برمجة تطبيقات بسيطة باستخدام FastAPI. سأبدأ بكتابة الكود وتشغيله.",
            "tools": [{
                "tool": "code_sandbox",
                "params": {
                    "code": "from fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef read_root(): return {'status': 'ok'}\n# خطأ متعمد: استخدام json دون استيراده\nprint(json.dumps({'msg': 'Started'}))"
                }
            }],
            "end": False
        })
    
    # الجولة 2: اكتشاف الخطأ وتصحيحه
    if "NameError: name 'json' is not defined" in last_msg or "error" in last_msg.lower():
        return json.dumps({
            "thinking": "لقد حدث خطأ NameError لأنني نسيت استيراد مكتبة json. سأقوم بتصحيح الكود الآن.",
            "tools": [{
                "tool": "code_sandbox",
                "params": {
                    "code": "import json\nfrom fastapi import FastAPI\napp = FastAPI()\n@app.get('/')\ndef read_root(): return {'status': 'ok'}\nprint(json.dumps({'msg': 'Started and Fixed'}))"
                }
            }],
            "end": False
        })
    
    # النهاية
    if "Started and Fixed" in last_msg:
        return json.dumps({
            "thinking": "تم تصحيح الكود وتشغيله بنجاح. الواجهة الآن جاهزة.",
            "finish": "تم بناء وتصحيح الـ API بنجاح. المخرجات: {'msg': 'Started and Fixed'}",
            "end": True
        })

    return json.dumps({"thinking": "أنتظر النتائج...", "end": False})

def test_self_healing_api():
    print("🚀 بدء اختبار البناء والتصحيح الذاتي (Self-Healing API)...")
    
    for event in run_agent_loop("قم ببناء واجهة برمجة تطبيقات وتصحيح أخطائها", llm_fn=self_healing_llm):
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
    test_self_healing_api()
