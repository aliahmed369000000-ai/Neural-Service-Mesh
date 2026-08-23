import sys
import time
import concurrent.futures
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import shared_experience
from ai.agent_loop import _tool_ask_swarm

def send_query(i):
    """محاكاة إرسال سؤال واحد."""
    domains = ["infra", "algo", "dev", "general"]
    domain = domains[i % 4]
    return _tool_ask_swarm({
        "agent_id": f"Agent_{i % 10}",
        "query": f"سؤال معقد رقم {i} حول {domain}...",
        "context": f"سياق تقني مكثف للمجال {domain}."
    })

def run_stress_test(num_queries=200):
    print(f"🔥 بدء اختبار الضغط (Stress Test) لـ {num_queries} سؤال متزامن...")
    print("=" * 60)
    
    # إعادة تهيئة المعرفة الجماعية
    shared_experience.knowledge = {"shared_facts": {}, "active_queries": {}, "global_metrics": {}, "version": "1.1"}
    
    start_time = time.time()
    
    # استخدام ThreadPoolExecutor لمحاكاة التزامن
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(send_query, range(num_queries)))
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # التحقق من النتائج
    success_count = sum(1 for r in results if "✅" in r)
    error_count = num_queries - success_count
    
    print(f"\n📊 نتائج اختبار الضغط:")
    print(f"- إجمالي الأسئلة: {num_queries}")
    print(f"- الوقت المستغرق: {total_time:.2f} ثانية")
    print(f"- معدل المعالجة: {num_queries/total_time:.2f} سؤال/ثانية")
    print(f"- النجاح: {success_count}")
    print(f"- الفشل: {error_count}")
    
    # التحقق من سلامة البيانات في JSON
    shared_queries_count = len(shared_experience.knowledge["active_queries"])
    print(f"- الأسئلة المسجلة فعلياً في الذاكرة: {shared_queries_count}")
    
    if shared_queries_count < num_queries:
        print(f"⚠️ تحذير: تم فقدان {num_queries - shared_queries_count} سؤال بسبب تصادم الـ IDs أو مشاكل التزامن.")
    else:
        print("✅ نجاح: تم تسجيل جميع الأسئلة دون أي فقدان.")

    print("=" * 60)
    print("🏁 انتهى اختبار الضغط.")

if __name__ == "__main__":
    run_stress_test(200)
