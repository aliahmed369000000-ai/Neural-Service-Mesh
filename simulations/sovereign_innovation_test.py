
import json
import sys
from pathlib import Path

# إضافة مسار المشروع للاستيراد
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_loop import run_agent_loop

def mock_llm_innovation(system, history):
    """محاكاة LLM يبتكر خوارزمية جديدة."""
    user_msgs = [m for m in history if m.get("role") == "user"]
    last_user_msg = str(user_msgs[-1]["content"]) if user_msgs else ""
    
    # البحث في التاريخ عن أي نتيجة لأداة الابتكار
    has_innovation = any("ابتكار سيادي مسجل" in str(m.get("content", "")) for m in history)
    
    if "ابتكار" in last_user_msg and not has_innovation:
        return json.dumps({
            "thinking": "لتحسين أداء الشبكة، سأقترح خوارزمية 'Dynamic Sparse Attention' التي تقلل التعقيد الحسابي.",
            "tools": [{
                "tool": "propose_innovation",
                "params": {
                    "name": "Dynamic Sparse Attention",
                    "description": "آلية انتباه ديناميكية تركز فقط على الرموز الأكثر صلة.",
                    "code": "class DynamicSparseAttention(nn.Module): ...",
                    "category": "Attention"
                }
            }],
            "end": False
        })

    return json.dumps({
        "thinking": "تم تسجيل الابتكار بنجاح.",
        "finish": "🚀 تم إطلاق مرحلة السيادة الإبداعية.",
        "end": True
    })

def main():
    print("💡 بدء محاكاة السيادة الإبداعية (Sovereign Innovation Simulation)...")
    
    gen = run_agent_loop("ابتكار خوارزمية جديدة للشبكة", llm_fn=mock_llm_innovation)
    
    for event in gen:
        etype = event.get("type")
        if etype == "thinking":
            print(f"🤔 [تفكير]: {event.get('content')}")
        elif etype == "result":
            print(f"✅ [النتيجة]: {event.get('output')}")
        elif etype == "answer":
            print(f"🏁 [الإجابة]: {event.get('text')}")

if __name__ == "__main__":
    main()
