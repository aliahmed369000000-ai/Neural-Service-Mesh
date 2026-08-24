
import sys
from pathlib import Path

# إضافة ROOT إلى sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.learning_engine import learning_engine

def simulate_agent_experience():
    print("🧠 محاكاة خبرة سابقة (فشل في مهمة جمع البيانات)...")
    learning_engine.record_experience(
        task="جمع بيانات Kaggle",
        outcome="فشل بسبب نفاذ الذاكرة (OOM)",
        lesson="يجب استخدام تقنية FSDP أو تقليل حجم الدفعة (Batch Size) عند التعامل مع d=8192.",
        success=False
    )

def simulate_new_task():
    print("🚀 بدء مهمة جديدة مشابهة...")
    task = "أريد جمع بيانات جديدة لمشروع d=8192"
    lessons = learning_engine.get_relevant_lessons(task)
    
    if lessons:
        print("✅ تم استرجاع الدروس المستفادة بنجاح!")
        print(lessons)
    else:
        print("❌ فشل في استرجاع الدروس.")

if __name__ == "__main__":
    simulate_agent_experience()
    print("-" * 30)
    simulate_new_task()
