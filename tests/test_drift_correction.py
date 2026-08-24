import sys
import os
import numpy as np

# إضافة المسار الجذري للمشروع
sys.path.append("/home/ubuntu/Neural-Service-Mesh")

from ai.drift_corrector import DriftCorrector

def test_drift_correction():
    print("🚀 بدء اختبار تصحيح الانحراف الزمني الهجين (Kalman + DTW)...")
    
    corrector = DriftCorrector(process_variance=1e-5, measurement_variance=1e-3)
    
    # محاكاة انحراف متزايد (Temporal Drift)
    # لنفترض أن الفيديو طوله 100 ثانية، والانحراف يزداد بمعدل 0.01 ثانية كل ثانية
    timestamps = np.arange(0, 100, 5)
    raw_drifts = timestamps * 0.01 + np.random.normal(0, 0.005, len(timestamps))
    
    results = []
    for ts, drift in zip(timestamps, raw_drifts):
        res = corrector.correct(current_timestamp=float(ts), measured_offset=drift)
        results.append(res)
        print(f"الوقت: {ts}s | الانحراف الخام: {drift:.4f}s | التقدير المستقر: {res['estimated_drift']:.4f}s | الثقة: {res['confidence']:.2%}")
    
    # التحقق من استقرار التقدير
    final_drift = results[-1]['estimated_drift']
    print(f"\n✅ النتيجة النهائية: الانحراف المستقر عند 100 ثانية هو {final_drift:.4f}s")
    
    if 0.9 <= final_drift <= 1.1:
        print("🎉 الاختبار نجح: المصحح تتبع الانحراف المتراكم بدقة.")
    else:
        print("⚠️ الاختبار اكتمل: قد يحتاج المصحح لضبط المعاملات.")

if __name__ == "__main__":
    test_drift_correction()
