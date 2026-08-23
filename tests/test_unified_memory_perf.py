# -*- coding: utf-8 -*-
import sys
import os
import time
import asyncio

# إضافة المسار الرئيسي للمشروع
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ai.living_mesh import LivingMeshNode

async def test_performance():
    print("🚀 بدء اختبار أداء الذاكرة الموحدة (ANN + Sharding)...")
    
    # إنشاء عقدة اختبار
    node = LivingMeshNode(node_id="perf_tester")
    
    # 1. اختبار سرعة الحفظ
    start_time = time.time()
    num_exps = 50
    print(f"📥 حفظ {num_exps} خبرة في الذاكرة الموحدة...")
    for i in range(num_exps):
        exp = {
            "kind": "perf_test",
            "data": {"value": i, "content": f"خبرة تجريبية رقم {i} للتحقق من سرعة التخزين المجزأ"},
            "timestamp": time.time()
        }
        emb = node._generate_simulated_embedding(exp["kind"], exp["data"])
        node.memory.store_experience(exp, embedding=emb)
    
    save_duration = time.time() - start_time
    print(f"✅ تم الحفظ في {save_duration:.4f} ثانية ({save_duration/num_exps:.6f} ثانية لكل خبرة).")
    
    # 2. اختبار سرعة البحث الدلالي (ANN)
    print("🔍 إجراء بحث دلالي (ANN Search)...")
    start_time = time.time()
    results = node.semantic_query("سرعة التخزين المجزأ", top_k=3)
    search_duration = time.time() - start_time
    
    print(f"✅ تم البحث في {search_duration:.4f} ثانية.")
    print(f"📊 عدد النتائج: {len(results)}")
    
    # 3. إحصائيات الذاكرة
    stats = node.memory.get_memory_stats()
    print("\n📊 إحصائيات الذاكرة الموحدة:")
    print(f"- إجمالي الخبرات: {stats['total_experiences']}")
    print(f"- المتجهات المفهرسة: {stats['indexed_vectors']}")
    print(f"- عدد الأجزاء (Shards): {stats['num_shards']}")
    
    assert stats['total_experiences'] >= num_exps
    print("\n✨ انتهى اختبار الأداء بنجاح!")

if __name__ == "__main__":
    asyncio.run(test_performance())
