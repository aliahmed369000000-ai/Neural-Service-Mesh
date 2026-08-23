import sys
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.memory_manager import MemoryManager
from ai.shared_experience import shared_experience
from ai.learning_engine import learning_engine

def run_evaluation_simulation():
    print("🛡️ بدء محاكاة تقييم الحلول البرمجية (Solution Evaluation Test)...")
    print("=" * 60)
    
    # إعادة تهيئة المعرفة الجماعية
    shared_experience.knowledge = {"shared_facts": {}, "global_metrics": {}, "version": "1.0"}
    dev_agent = MemoryManager("Dev_Agent")

    # 1. اختبار حل "ضعيف" (غير كفء وغير آمن)
    print("\n❌ اختبار حل ضعيف (Nested Loops + Security Risk)...")
    bad_code = (
        "def process_data(data):\n"
        "    for i in data:\n"
        "        for j in data: # O(n^2)\n"
        "            eval(i + j) # Security Risk\n"
        "    return True"
    )
    
    fact_bad = {"content": f"حل مقترح:\n```python\n{bad_code}\n```", "strength": 0.95}
    success_bad = shared_experience.share_fact("Dev_Agent", fact_bad)
    
    if not success_bad:
        print("✅ نجاح المحاكاة: تم رفض الحل الضعيف من قبل نظام التقييم الذاتي.")
    else:
        print("❌ فشل المحاكاة: تم قبول حل ضعيف!")

    # 2. اختبار حل "قوي" (كفء وآمن)
    print("\n✅ اختبار حل قوي (Scalable FAISS + Quantization)...")
    good_code = (
        "import faiss\n"
        "def build_index(data):\n"
        "    d = data.shape[1]\n"
        "    index = faiss.IndexIVFPQ(faiss.IndexFlatL2(d), d, 100, 8, 8)\n"
        "    index.train(data)\n"
        "    return index"
    )
    
    fact_good = {"content": f"حل معتمد:\n```python\n{good_code}\n```", "strength": 0.9}
    success_good = shared_experience.share_fact("Dev_Agent", fact_good)
    
    if success_good:
        print("✅ نجاح المحاكاة: تم قبول الحل القوي بنجاح.")
        # التحقق من دمج التقييم في الأهمية
        shared_fact = list(shared_experience.knowledge["shared_facts"].values())[0]
        print(f"📈 الأهمية النهائية المدمجة: {shared_fact['importance']:.2f}")
    else:
        print("❌ فشل المحاكاة: تم رفض حل قوي!")

    print("=" * 60)
    print("🏁 انتهت محاكاة التقييم.")

if __name__ == "__main__":
    run_evaluation_simulation()
