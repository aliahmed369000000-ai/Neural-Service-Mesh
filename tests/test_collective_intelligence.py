
import sys
from pathlib import Path

# إضافة ROOT إلى sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

from ai.learning_engine import learning_engine

def simulate_collective_learning():
    print("🌟 بدء اختبار المعرفة الجماعية (Collective Intelligence)...")
    
    # 1. استيراد بذور الخبراء
    seeds_file = ROOT / "artifacts" / "learning" / "expert_seeds.json"
    learning_engine.import_expert_seeds(str(seeds_file))
    
    # 2. وكيل (أ) يتعلم درساً جديداً
    print("🤖 الوكيل 'Alpha' يتعلم درساً في البرمجة...")
    learning_engine.record_experience(
        task="إصلاح خطأ استيراد",
        outcome="نجاح",
        lesson="تأكد من إضافة المجلد الأب إلى sys.path عند تشغيل الاختبارات محلياً.",
        success=True,
        agent_id="agent_alpha"
    )
    
    # 3. وكيل (ب) يستفيد من معرفة (أ) ومن الخبراء
    print("🤖 الوكيل 'Beta' يبدأ مهمة جديدة ويستشير الدماغ المركزي...")
    task = "أريد تشغيل اختبار وإصلاح أخطاء الاستيراد وتدريب نموذج d=8192"
    lessons = learning_engine.get_relevant_lessons(task)
    
    if lessons:
        print("✅ تم استرجاع المعرفة الجماعية بنجاح!")
        print(lessons)
    else:
        print("❌ فشل في استرجاع المعرفة.")

if __name__ == "__main__":
    simulate_collective_learning()
