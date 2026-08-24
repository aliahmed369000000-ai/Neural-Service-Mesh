
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للنظام
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_hibernation import AgentState, hibernate_agent, wake_up_agent

def test_hierarchical_memory():
    print("🚀 اختبار نظام الذاكرة الهرمية (Hierarchical Memory Test)...")
    
    agent_id = "test_hierarchical_agent"
    context = [{"role": "system", "content": "أنت وكيل ذكي."}]
    # إضافة 20 رسالة لتفعيل التلخيص التلقائي (>15)
    for i in range(20):
        context.append({"role": "user", "content": f"رسالة اختبار رقم {i}"})
        context.append({"role": "assistant", "content": f"رد اختبار رقم {i}"})
    
    plan = {"tasks": ["مهمة 1", "مهمة 2"]}
    
    # 1. اختبار الحفظ مع التلخيص التلقائي
    print("📦 حفظ الحالة وتفعيل التلخيص التلقائي...")
    success = hibernate_agent(agent_id, context, plan, compress=True)
    if not success:
        print("❌ فشل حفظ الحالة.")
        return

    # 2. اختبار الاستيقاظ واستعادة الذاكرة الأحداثية
    print("🌅 استيقاظ الوكيل والتحقق من الذاكرة الهرمية...")
    state = wake_up_agent(agent_id)
    
    if state:
        print(f"✅ طول الذاكرة العاملة (Working Memory): {len(state.context)}")
        print(f"✅ عدد حلقات الذاكرة الأحداثية (Episodic): {len(state.episodic_memory)}")
        
        if len(state.episodic_memory) > 0:
            print(f"📝 ملخص أول حلقة: {state.episodic_memory[0]['summary']}")
            
            # 3. اختبار البحث الدلالي في الذاكرة الأحداثية
            print("🔍 اختبار البحث الدلالي في الذاكرة الأحداثية...")
            search_results = state.search_episodic("تفاعلات سابقة")
            print(f"✅ نتائج البحث الدلالي: {len(search_results)} حلقة مكتشفة.")
            
            if len(state.context) < 41: # الأصل 1 + 20*2 = 41
                print("✅ نجاح: تم ضغط الذاكرة العاملة وأرشفة القديم.")
            else:
                print("❌ فشل: لم يتم ضغط الذاكرة العاملة.")
        else:
            print("❌ فشل: لم يتم إنشاء ذاكرة أحداثية.")
    else:
        print("❌ فشل استعادة الحالة.")

if __name__ == "__main__":
    test_hierarchical_memory()
