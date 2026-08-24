import sys
import os
import json
import time

# إضافة مسار المشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.memory_manager import MemoryManager
from ai.agent_hibernation import AgentState, hibernate_agent, wake_up_agent

def test_stm_ltm_flow():
    print("🚀 بدء اختبار تدفق الذاكرة STM/LTM...")
    
    # 1. إنشاء ذاكرة جديدة وإضافة رسائل (STM)
    mm = MemoryManager(agent_id="test_memory_bot")
    mm.stm.append({"role": "system", "content": "أنت وكيل ذكي."})
    mm.add_to_stm({"role": "user", "content": "أريد تنفيذ عملية commit برمز SHA: a1b2c3d"})
    mm.add_to_stm({"role": "assistant", "content": "تم التنفيذ بنجاح."})
    
    print(f"✅ حجم STM الحالي: {len(mm.stm)}")
    
    # 2. محاكاة التلخيص القسري لنقل البيانات إلى LTM
    mm.consolidate(force=True)
    
    print(f"✅ حجم STM بعد التلخيص: {len(mm.stm)}")
    print(f"✅ عدد الحقائق المستخرجة (LTM-S): {len(mm.ltm_semantic)}")
    print(f"✅ عدد الأحداث المؤرشفة (LTM-E): {len(mm.ltm_episodic)}")
    
    # التحقق من استخراج الحقيقة
    found_sha = False
    for f_id, fact in mm.ltm_semantic.items():
        if "a1b2c3d" in fact["content"]:
            found_sha = True
            break
    assert found_sha, "❌ فشل استخراج رمز SHA في الذاكرة الدائمة"
    print("✅ تم التحقق من استخراج الحقائق التقنية بنجاح.")

    # 3. اختبار الاسترجاع النشط (Active Retrieval)
    query = "ما هو رمز SHA السابق؟"
    results = mm.search(query)
    print(f"🔍 نتائج البحث عن '{query}': {len(results['semantic'])} حقائق")
    assert len(results['semantic']) > 0, "❌ فشل الاسترجاع النشط للحقائق"
    
    # 4. اختبار دورة النوم والاستيقاظ مع MemoryManager
    agent_id = "test_memory_bot"
    history = mm.stm
    plan = {"goal": "test"}
    
    # الحفظ
    success = hibernate_agent(agent_id, history, plan, memory_manager_data=mm.to_dict())
    assert success, "❌ فشل حفظ حالة الذاكرة"
    print("✅ تم حفظ حالة الذاكرة بنجاح.")
    
    # الاستيقاظ
    recovered = wake_up_agent(agent_id)
    assert recovered is not None, "❌ فشل استعادة حالة الذاكرة"
    
    # إعادة بناء MemoryManager من البيانات المستعادة
    state_dict = recovered.to_dict()
    mm_data = state_dict.get("memory_manager_data", {})
    print(f"DEBUG: المستعادة LTM-S count: {len(mm_data.get('ltm_semantic', {}))}")
    mm_new = MemoryManager.from_dict(agent_id, mm_data)
    assert len(mm_new.ltm_semantic) == len(mm.ltm_semantic), f"❌ فشل استعادة الحقائق الدائمة: المتوقع {len(mm.ltm_semantic)}، المستعاد {len(mm_new.ltm_semantic)}"
    print("✅ تم استعادة نظام STM/LTM كاملاً بنجاح.")

if __name__ == "__main__":
    try:
        test_stm_ltm_flow()
        print("\n✨ نجح اختبار التكامل الشامل لنظام STM/LTM!")
    except Exception as e:
        print(f"\n❌ فشل الاختبار: {e}")
        sys.exit(1)
