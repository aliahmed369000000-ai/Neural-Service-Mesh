
import json
import sys
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop

def mock_llm_sovereign_training(system, history):
    """محاكاة LLM يطبق الدروس المستفادة في مهمة التدريب."""
    full_history = "\n".join([str(m["content"]) for m in history])
    
    # التحقق من حقن الدروس في الـ system prompt
    has_fsdp_lesson = "fsdp" in system.lower()
    has_minhash_lesson = "minhash" in system.lower()
    has_timeout_lesson = "11 ساعة" in system
    
    # جلب آخر رسالة من المستخدم حصراً
    user_msgs = [m for m in history if m.get("role") == "user"]
    last_user_msg = str(user_msgs[-1]["content"]) if user_msgs else ""
    
    if "تصميم محرك تدريب" in last_user_msg or "d=8192" in last_user_msg:
        thinking = "سأقوم بتصميم محرك التدريب. ألاحظ وجود دروس مستفادة هامة في السياق."
        
        # دمج الدروس في الخطة
        if has_fsdp_lesson and has_minhash_lesson and has_timeout_lesson:
            thinking += " سأطبق FSDP للذاكرة، MinHash للبيانات، وحد الـ 11 ساعة للاستقرار."
            return json.dumps({
                "thinking": thinking,
                "tools": [{"tool": "plan", "params": {"goal": "بناء محرك تدريب سيادي متوافق مع الدروس المستفادة"}}],
                "end": False
            })
        else:
            return json.dumps({
                "thinking": "سأبدأ بالتخطيط للتدريب بشكل عام.",
                "tools": [{"tool": "plan", "params": {"goal": "بناء محرك تدريب"}}],
                "end": False
            })

    if "plan" in full_history:
        return json.dumps({
            "thinking": "الخطة جاهزة. سأقوم بتوليد الكود البرمجي الذي يدمج هذه التقنيات.",
            "finish": "✅ نجاح التطبيق العملي: الوكيل صمم محرك تدريب يدمج FSDP و MinHash ويلتزم بحد الـ 11 ساعة تلقائياً.",
            "end": True
        })

    return json.dumps({"thinking": "متابعة...", "end": False})

def main():
    print("🚀 بدء محاكاة محرك التدريب السيادي (Sovereign Training Engine)...")
    
    # تشغيل المهمة
    gen = run_agent_loop("تصميم محرك تدريب d=8192 مع مراعاة الدروس السابقة", llm_fn=mock_llm_sovereign_training)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"🤔 [تفكير]: {event.get('content')}")
        elif etype == "answer":
            print(f"🏁 [النتيجة]: {event.get('text')}")
        elif etype == "info":
            print(f"ℹ️ [معلومات]: {event.get('text')}")

if __name__ == "__main__":
    main()
