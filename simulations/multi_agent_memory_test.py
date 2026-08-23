import sys
import os
import time
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.memory_manager import MemoryManager
from ai.shared_experience import shared_experience
from ai.agent_hibernation import hibernate_agent, wake_up_agent

def run_simulation():
    print("🚀 بدء محاكاة التفاعل متعدد الوكلاء لـ NSM...")
    print("-" * 50)
    
    # إعادة تهيئة المعرفة الجماعية لضمان النظافة
    shared_experience.knowledge = {"shared_facts": {}, "global_metrics": {}, "version": "1.0"}

    # 1. إنشاء وكيلين بهويات مختلفة
    dev_agent = MemoryManager("Dev_Agent_01")
    qa_agent = MemoryManager("QA_Agent_02")

    # 2. سيناريو: Dev_Agent يكتشف حل لمشكلة OOM
    print("👨‍💻 [Dev_Agent]: يواجه مشكلة OOM ويجد حلاً...")
    technical_solution = (
        "لحل مشكلة Out of Memory (OOM) في تدريب d_model=8192، "
        "يجب استخدام gradient_accumulation_steps=4 و تقليل batch_size إلى 1."
    )
    
    # إضافة الحقيقة للذاكرة (سيتم تقييمها كأهمية عالية)
    dev_agent.add_fact(technical_solution, importance=0.95)
    
    # محاكاة التوحيد لمشاركة الخبرة
    print("🧠 [Dev_Agent]: يقوم بتوحيد الذاكرة ومشاركة الخبرة مع السرب...")
    # يدوياً هنا للمحاكاة
    fact_id = list(dev_agent.ltm_semantic.keys())[0]
    fact_data = dev_agent.ltm_semantic[fact_id]
    print(f"DEBUG: Fact ID={fact_id}, Hash={fact_data.get('semantic_hash')}")
    shared_experience.share_fact("Dev_Agent_01", fact_data)
    
    # حفظ حالة Dev_Agent (دخول وضع النوم)
    hibernate_agent("Dev_Agent_01", [], {}, memory_manager_data=dev_agent.to_dict())

    print("\n⏳ مرور الوقت... (مزامنة السرب)")
    time.sleep(1)

    # 3. سيناريو: QA_Agent يواجه نفس المشكلة ويبحث عن حل دلالي
    print("🔍 [QA_Agent]: يواجه فشل في التدريب ويبحث في الذاكرة الجماعية...")
    
    # مزامنة QA_Agent مع المعرفة الجماعية
    shared_experience.sync_agent_memory(qa_agent)
    
    # البحث الدلالي باستخدام استعلام مختلف نصياً ولكن مشابه دلالياً
    # اختبار البحث الدلالي باللغة العربية مع كلمات مفتاحية تقنية
    query = "حل مشكلة OOM للتدريب الضخم d_model=8192"
    print(f"❓ استعلام QA_Agent: '{query}'")
    
    from ai.multimodal_sync import MultimodalSyncManager
    sm = MultimodalSyncManager()
    q_hash = sm._generate_lsh_hash(sm._generate_embedding(query))
    print(f"DEBUG: Query Hash={q_hash}")
    
    search_results = qa_agent.search(query, limit=1)
    
    print("\n📊 نتائج البحث الدلالي:")
    if search_results["semantic"]:
        found_fact = search_results["semantic"][0]
        print(f"✅ تم العثور على حل! النتيجة: '{found_fact['content']}'")
        print(f"📈 درجة التشابه الدلالي: {found_fact.get('search_score', 'N/A')}")
        
        if "8192" in found_fact['content'] and "batch_size" in found_fact['content']:
            print("\n🌟 نجاح المحاكاة: تم نقل الخبرة التقنية بنجاح عبر الذاكرة الجماعية!")
        else:
            print("\n⚠️ فشل جزئي: تم العثور على نتيجة ولكنها قد لا تكون دقيقة.")
    else:
        print("❌ فشل المحاكاة: لم يتم العثور على أي حلول مرتبطة.")

    print("-" * 50)
    print("🏁 انتهت المحاكاة.")

if __name__ == "__main__":
    # تنظيف ملفات الاختبار القديمة لضمان دقة المحاكاة
    shared_knowledge_path = Path("artifacts/learning/shared_knowledge.json")
    if shared_knowledge_path.exists():
        shared_knowledge_path.unlink()
        
    run_simulation()
