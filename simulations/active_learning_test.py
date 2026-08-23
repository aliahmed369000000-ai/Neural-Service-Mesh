import sys
import time
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import shared_experience
from ai.agent_loop import _tool_ask_swarm, _tool_answer_swarm, _tool_check_swarm_queries

def run_active_learning_simulation():
    print("❓ بدء محاكاة التعلم النشط (Active Learning Simulation)...")
    print("=" * 60)
    
    # إعادة تهيئة المعرفة الجماعية
    shared_experience.knowledge = {"shared_facts": {}, "active_queries": {}, "global_metrics": {}, "version": "1.1"}

    # 1. وكيل يواجه غموضاً ويطرح سؤالاً
    print("\n👨‍💻 [Junior_Agent]: يواجه غموضاً في كيفية استخدام FAISS...")
    q_id_msg = _tool_ask_swarm({
        "agent_id": "Junior_Agent",
        "query": "ما هو الفرق بين IndexFlatL2 و IndexIVFPQ في مكتبة FAISS؟",
        "context": "أحاول بناء نظام بحث لـ 1 مليون متجه."
    })
    print(q_id_msg)
    q_id = q_id_msg.split("رقم السؤال: ")[1]

    # 2. وكيل خبير يتحقق من الأسئلة ويجيب
    print("\n🧠 [Senior_Agent]: يتحقق من الأسئلة المعلقة...")
    pending_report = _tool_check_swarm_queries({"agent_id": "Senior_Agent"})
    print(pending_report)
    
    if q_id in pending_report:
        print(f"\n💡 [Senior_Agent]: يقدم إجابة تقنية للسؤال {q_id}...")
        ans_msg = _tool_answer_swarm({
            "agent_id": "Senior_Agent",
            "query_id": q_id,
            "answer": "IndexFlatL2 هو بحث دقيق (Exact) وبطيء، بينما IndexIVFPQ هو بحث تقريبي (ANN) يستخدم التكميم PQ لتقليل حجم الذاكرة وسرعة البحث الضخم."
        })
        print(ans_msg)

    # 3. الوكيل السائل يتحقق من الإجابات
    print("\n👨‍💻 [Junior_Agent]: يتحقق من وصول إجابات لأسئلته...")
    final_report = _tool_check_swarm_queries({"agent_id": "Junior_Agent"})
    print(final_report)
    
    if "💡 إجابات واردة لأسئلتك" in final_report:
        print("\n🌟 نجاح المحاكاة: تم تبادل المعرفة التقنية بنجاح عبر نظام التعلم النشط!")
    else:
        print("\n❌ فشل المحاكاة: لم تصل الإجابة للوكيل السائل.")

    print("=" * 60)
    print("🏁 انتهت محاكاة التعلم النشط.")

if __name__ == "__main__":
    run_active_learning_simulation()
