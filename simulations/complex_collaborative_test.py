import sys
import os
import time
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.memory_manager import MemoryManager
from ai.shared_experience import shared_experience

def run_complex_simulation():
    print("🏢 بدء محاكاة 'أزمة توسع الشبكة العصبية' (Multi-Agent Collaboration)...")
    print("=" * 60)
    
    # إعادة تهيئة المعرفة الجماعية
    shared_experience.knowledge = {"shared_facts": {}, "global_metrics": {}, "version": "1.0"}

    # 1. تعريف الوكلاء المتخصصين
    infra_agent = MemoryManager("Infra_Agent")
    algo_agent = MemoryManager("Algo_Agent")
    dev_agent = MemoryManager("Dev_Agent")

    # --- الخطوة 1: تشخيص Infra_Agent ---
    print("\n🚨 [Infra_Agent]: يكتشف خلل في البنية التحتية...")
    incident_report = (
        "تحذير: انهيار الذاكرة OOM عند معالجة 1,000,000 متجه. "
        "زمن الاستجابة ارتفع إلى 5000ms. النظام غير قادر على التوسع."
    )
    infra_agent.add_fact(incident_report, importance=0.9)
    
    # مشاركة التشخيص مع السرب
    shared_experience.share_fact("Infra_Agent", infra_agent.ltm_semantic[list(infra_agent.ltm_semantic.keys())[0]])
    print("📡 [Infra_Agent]: تم نشر تقرير الحادثة في الذاكرة الجماعية.")

    # --- الخطوة 2: تحليل Algo_Agent ---
    print("\n🧠 [Algo_Agent]: يحلل تقرير الحادثة ويقترح خوارزمية...")
    
    # مزامنة المعرفة
    shared_experience.sync_agent_memory(algo_agent)
    
    # البحث عن المشكلة
    query = "OOM and high latency with 1M vectors"
    problem = algo_agent.search(query, limit=1)["semantic"]
    
    if problem:
        print(f"📥 [Algo_Agent]: استقبلت بلاغ: '{problem[0]['content']}'")
        solution_proposal = (
            "الحل الخوارزمي المقترح: استخدام Vector Quantization (PQ) لتقليل حجم المتجهات بنسبة 4x "
            "مع بناء فهرس HNSW لتقليل زمن البحث من O(n) إلى O(log n)."
        )
        algo_agent.add_fact(solution_proposal, importance=0.95)
        
        # مشاركة الحل
        fact_id = [k for k, v in algo_agent.ltm_semantic.items() if solution_proposal in v['content']][0]
        shared_experience.share_fact("Algo_Agent", algo_agent.ltm_semantic[fact_id])
        print("📡 [Algo_Agent]: تم نشر مقترح الحل الخوارزمي.")

    # --- الخطوة 3: تنفيذ Dev_Agent ---
    print("\n👨‍💻 [Dev_Agent]: يجمع المعلومات ويبني الكود النهائي...")
    
    # مزامنة المعرفة (سيحصل على تقرير Infra وحل Algo)
    shared_experience.sync_agent_memory(dev_agent)
    
    # البحث عن "كيفية حل مشكلة الـ 1M متجه"
    final_query = "how to solve 1M vectors OOM and latency using algorithms"
    knowledge_base = dev_agent.search(final_query, limit=3)["semantic"]
    
    print(f"🔍 [Dev_Agent]: يبحث في الذاكرة الجماعية عن استراتيجية التنفيذ...")
    
    has_infra = any("1,000,000" in k['content'] for k in knowledge_base)
    has_algo = any("HNSW" in k['content'] for k in knowledge_base)
    
    if has_infra and has_algo:
        print("✅ [Dev_Agent]: تم دمج الخبرات! جاري توليد الكود النهائي...")
        final_code = (
            "import faiss\n"
            "d = 128\n"
            "quantizer = faiss.IndexFlatL2(d)\n"
            "index = faiss.IndexIVFPQ(quantizer, d, 100, 8, 8) # Vector Quantization\n"
            "index.train(data)\n"
            "index.add(data) # Scalable solution for 1M vectors"
        )
        dev_agent.add_fact(f"الكود النهائي المعتمد: {final_code}", importance=1.0)
        print("\n🚀 [النتيجة النهائية]:")
        print(final_code)
        print("\n🌟 نجاح التعاون: 3 وكلاء عملوا معاً لحل مشكلة ضخمة عبر الذاكرة الجماعية!")
    else:
        print("⚠️ [Dev_Agent]: لم يتمكن من العثور على سياق كامل للحل.")
        print(f"البيانات المتاحة له: {[k['content'][:50] + '...' for k in knowledge_base]}")

    print("=" * 60)
    print("🏁 انتهت المحاكاة المعقدة.")

if __name__ == "__main__":
    run_complex_simulation()
