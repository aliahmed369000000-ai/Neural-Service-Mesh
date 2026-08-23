
import sys
import os
import json

# إضافة مسار المشروع للنظام
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.agent_hibernation import AgentState, hibernate_agent, wake_up_agent

def test_memory_semantic_search():
    print("🚀 بدء اختبار البحث الدلالي في الذاكرة...")
    
    agent_id = "test_semantic_agent"
    context = [
        {"role": "user", "content": "الهدف هو تحسين أداء الشبكة العصبية عبر تقليل زمن الاستجابة."},
        {"role": "assistant", "content": "تم تنفيذ ذلك عبر ضغط المتجهات بنسبة 75%."},
        {"role": "user", "content": "ما هو SHA للالتزام الأخير؟"},
        {"role": "assistant", "content": "SHA هو a1b2c3d."}
    ]
    
    # 1. حفظ الحالة مع استخراج الحقائق والهاشات
    print("📝 حفظ الحالة واستخراج الحقائق...")
    # استخدام force_summarize=True لإجبار النظام على أرشفة الرسائل واستخراج الحقائق
    hibernate_agent(agent_id, context, {"steps": []}, force_summarize=True)
    
    # 2. استعادة الحالة
    state = wake_up_agent(agent_id)
    if not state:
        print("❌ فشل استعادة الحالة.")
        return
        
    print(f"✅ تم استعادة {len(state.semantic_memory)} حقائق.")
    
    # 3. البحث الدلالي
    queries = [
        "كيف تم تحسين الأداء؟",
        "أريد رمز SHA",
        "زمن الاستجابة"
    ]
    
    for query in queries:
        print(f"\n🔍 البحث عن: '{query}'")
        results = state.search_semantic(query)
        if results:
            print(f"✨ وجد: {results[0]['content']} (التشابه: {results[0]['score']})")
        else:
            print("❓ لم يتم العثور على نتائج.")

    print("\n✅ انتهى الاختبار بنجاح.")

if __name__ == "__main__":
    test_memory_semantic_search()
