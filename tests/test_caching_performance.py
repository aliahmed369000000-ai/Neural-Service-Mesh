
import time
import sys
from pathlib import Path

# إضافة ROOT إلى sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.cache_manager import agent_cache

def mock_heavy_operation(params):
    """محاكاة عملية ثقيلة تستغرق ثانيتين."""
    time.sleep(2)
    return f"Result for {params.get('id')}"

if __name__ == "__main__":
    print("⏳ بدء اختبار فعالية التخزين المؤقت (Caching Benchmarking)...")
    
    tool_name = "heavy_task"
    params = {"id": "test_123"}
    
    # 1. التشغيل الأول (بدون كاش)
    start_1 = time.time()
    print("🔄 تشغيل المهمة للمرة الأولى (تحميل حقيقي)...")
    res_1 = mock_heavy_operation(params)
    agent_cache.set(tool_name, params, res_1)
    duration_1 = time.time() - start_1
    print(f"⏱️ زمن التشغيل الأول: {duration_1:.2f} ثانية")
    
    # 2. التشغيل الثاني (باستخدام الكاش)
    start_2 = time.time()
    print("🚀 تشغيل المهمة للمرة الثانية (من الكاش)...")
    res_2 = agent_cache.get(tool_name, params)
    duration_2 = time.time() - start_2
    print(f"⏱️ زمن التشغيل الثاني: {duration_2:.4f} ثانية")
    
    if res_1 == res_2:
        improvement = ((duration_1 - duration_2) / duration_1) * 100
        print(f"✅ تطابق النتائج!")
        print(f"🚀 نسبة التوفير في الوقت: {improvement:.2f}%")
    else:
        print("❌ خطأ: النتائج غير متطابقة!")
