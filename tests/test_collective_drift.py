import sys
import os
import time

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.drift_corrector import DriftCorrector
from ai.learning_engine import learning_engine

def test_collective_learning():
    print("🚀 بدء اختبار التعلم الجماعي لأنماط الانحراف...")
    
    source_id = "zoom_recording_01"
    
    # الجولة الأولى: اكتشاف الانحراف وحفظه
    print("\n--- الجولة 1: اكتشاف الانحراف ---")
    corrector1 = DriftCorrector(source_id=source_id)
    # محاكاة انحراف بمعدل 0.02
    for ts in range(0, 50, 5):
        drift = ts * 0.02
        res = corrector1.correct(float(ts), drift)
        print(f"الوقت: {ts}s | التقدير: {res['estimated_drift']:.4f} | الثقة: {res['confidence']:.2%}")
    
    # الجولة الثانية: استعادة الانحراف من الذاكرة الجماعية
    print("\n--- الجولة 2: استعادة الخبرة الجماعية ---")
    # يجب أن يقرأ DriftCorrector النمط من learning_engine تلقائياً
    corrector2 = DriftCorrector(source_id=source_id)
    
    if abs(corrector2.state[1] - 0.02) < 0.01:
        print(f"✅ نجاح: تم استعادة معدل الانحراف ({corrector2.state[1]:.4f}) من الذاكرة.")
    else:
        print(f"❌ فشل: لم يتم استعادة الخبرة بشكل صحيح. القيمة الحالية: {corrector2.state[1]:.4f}")

if __name__ == "__main__":
    test_collective_learning()
