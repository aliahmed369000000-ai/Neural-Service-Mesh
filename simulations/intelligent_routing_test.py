import sys
import time
import json
from pathlib import Path

# إضافة مسار المشروع للاستيراد
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ai.shared_experience import shared_experience
from ai.learning_engine import learning_engine
from ai.agent_loop import _tool_ask_swarm, _tool_check_swarm_queries

def run_routing_simulation():
    print("🚀 بدء محاكاة التوجيه الذكي للخبير (Intelligent Routing Simulation)...")
    print("=" * 60)
    
    # 1. إعداد خبير في البنية التحتية (Infra Expert)
    print("\n🧠 إعداد سجل الثقة: جعل Infra_Expert خبيراً في الـ OOM...")
    learning_engine.trust_scores["Infra_Expert"] = {
        "infra": 0.95,
        "general": 0.8
    }
    learning_engine._save_trust_scores()
    
    # إعادة تهيئة المعرفة الجماعية
    shared_experience.knowledge = {"shared_facts": {}, "active_queries": {}, "global_metrics": {}, "version": "1.1"}

    # 2. وكيل يطرح سؤالاً في مجال البنية التحتية
    print("\n👨‍💻 [Junior_Agent]: يطرح سؤالاً حول OOM...")
    q_id_msg = _tool_ask_swarm({
        "agent_id": "Junior_Agent",
        "query": "كيف يمكنني منع وقوع خطأ OOM عند معالجة ملفات ضخمة؟",
        "context": "النظام ينهار باستمرار عند قراءة ملفات CSV."
    })
    print(q_id_msg)
    
    # 3. التحقق من التوجيه التلقائي
    query_id = q_id_msg.split("رقم السؤال: ")[1]
    query_data = shared_experience.knowledge["active_queries"][query_id]
    
    print(f"\n🎯 نتيجة التوجيه الذكي:")
    print(f"- المجال المكتشف: {query_data['domain']}")
    print(f"- الخبير المستهدف: {query_data['target_expert']}")
    
    if query_data['target_expert'] == "Infra_Expert":
        print("✅ نجاح: تم توجيه السؤال للخبير الصحيح تلقائياً.")
    else:
        print("❌ فشل: لم يتم تحديد الخبير المناسب.")

    # 4. الخبير يرى السؤال كأولوية عالية
    print("\n🧐 [Infra_Expert]: يتحقق من مهامه...")
    report = _tool_check_swarm_queries({"agent_id": "Infra_Expert"})
    print(report)
    
    if "HIGH (Direct Expert Request)" in report:
        print("✅ نجاح: السؤال ظهر بوضوح كطلب مباشر للخبير.")
    else:
        print("❌ فشل: لم يتم وسم الأولوية للخبير.")

    print("=" * 60)
    print("🏁 انتهت محاكاة التوجيه الذكي.")

if __name__ == "__main__":
    run_routing_simulation()
