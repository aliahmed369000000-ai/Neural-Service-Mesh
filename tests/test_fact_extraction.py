
import sys
import os
from pathlib import Path

# إضافة مسار المشروع للنظام
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.agent_hibernation import AgentState, hibernate_agent, wake_up_agent

def test_fact_extraction(tmp_path, monkeypatch):
    # عزل: ملفات السبات تُكتب في مجلد مؤقت لا في artifacts/ المتتبَّع
    import ai.agent_hibernation as _hib
    monkeypatch.setattr(_hib, "SLEEP_DIR", tmp_path)
    # الحقائق المهمة (importance>=0.7) تُشارَك فوراً عبر shared_experience
    # (ai/memory_manager.py::_consolidate_to_ltm) — عزل مسار تخزينها أيضاً
    from ai.shared_experience import shared_experience as _se
    monkeypatch.setattr(_se, "storage_path", tmp_path / "shared_knowledge.json")
    print("🚀 اختبار استخراج الحقائق الدقيقة (Fact Extraction Test)...")
    
    agent_id = "test_fact_agent"
    context = [{"role": "system", "content": "أنت وكيل ذكي."}]
    
    # إضافة رسائل تحتوي على معلومات دقيقة
    messages = [
        "القرار هو: استخدام خوارزمية LSH للبحث السريع.",
        "تم رفع الكود بنجاح مع SHA: 9b074aa.",
        "```python\ndef hello(): print('world')\n```",
        "مقاييس الأداء أظهرت سرعة 4.5 ms ودقة 99.9%."
    ]
    
    for i, msg in enumerate(messages):
        context.append({"role": "user", "content": f"مهمة {i}"})
        context.append({"role": "assistant", "content": msg})
    
    # إضافة رسائل حشو لتجاوز حد التلخيص (>15)
    for i in range(10):
        context.append({"role": "user", "content": "رسالة حشو"})
        context.append({"role": "assistant", "content": "رد حشو"})
        
    print(f"📦 حجم السياق قبل الضغط: {len(context)}")
    
    # 1. اختبار الحفظ مع استخراج الحقائق
    hibernate_agent(agent_id, context, {}, compress=True)
    
    # 2. اختبار الاستيقاظ والتحقق من الذاكرة الدلالية
    state = wake_up_agent(agent_id)
    
    assert state, "فشل استعادة الحالة"
    if state:
        print(f"✅ عدد الحقائق المستخرجة في الذاكرة الدلالية: {len(state.memory_manager.ltm_semantic)}")
        
        # التحقق من وجود الكيانات الهامة
        found_facts = [f["content"] for f in state.memory_manager.ltm_semantic.values()]
        
        expected_patterns = ["قرار", "SHA", "كود برمجي", "ms"]
        for pattern in expected_patterns:
            found = any(pattern in f for f in found_facts)
            if found:
                print(f"✔️ تم العثور على حقيقة تحتوي على: {pattern}")
            else:
                print(f"❌ لم يتم العثور على: {pattern}")
            assert found, f"حقيقة مفقودة: {pattern}"
                
        if len(state.memory_manager.ltm_semantic) >= 4:
            print("✅ نجاح: تم استخراج كافة الحقائق الهامة بدقة.")
        else:
            print("❌ فشل: لم يتم استخراج كافة الحقائق.")
    else:
        print("❌ فشل استعادة الحالة.")

if __name__ == "__main__":
    test_fact_extraction()
