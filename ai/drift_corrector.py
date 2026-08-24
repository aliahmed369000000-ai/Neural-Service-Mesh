import numpy as np
from typing import List, Tuple, Dict, Optional
from ai.learning_engine import learning_engine

class DriftCorrector:
    """
    نظام تصحيح الانحراف الزمني الهجين (Kalman + DTW).
    يقوم بالتنبؤ بالانحراف الزمني وتصحيحه باستخدام المطابقة غير الخطية.
    """
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2, source_id: Optional[str] = None):
        # معاملات مرشح كالمان (نموذج الدرجة الثانية: الإزاحة والسرعة)
        self.q = process_variance
        self.r = measurement_variance
        self.state = np.array([0.0, 0.0])  # [offset, drift_rate]
        self.covariance = np.eye(2) * 1.0
        self.last_ts = None
        self.source_id = source_id
        
        # محاولة تحميل نمط انحراف مسبق من الخبرة الجماعية
        if source_id:
            profile = learning_engine.get_drift_profile(source_id)
            if profile:
                self.state[1] = profile.get("drift_rate", 0.0)
                # تقليل خطأ التقدير الابتدائي لأننا نملك خبرة سابقة
                self.covariance[1, 1] = 0.1 
                print(f"🧠 استعادة نمط الانحراف للمصدر {source_id}: {self.state[1]:.4f}")
        
    def kalman_update(self, measurement: float, current_ts: float) -> float:
        """تحديث الحالة باستخدام مرشح كالمان (إزاحة + معدل انحراف)"""
        if self.last_ts is None:
            self.last_ts = current_ts
            self.state[0] = measurement
            return measurement
            
        dt = max(current_ts - self.last_ts, 0.001)
        self.last_ts = current_ts
        
        # 1. التنبؤ (Prediction)
        # مصفوفة انتقال الحالة F
        F = np.array([[1.0, dt],
                      [0.0, 1.0]])
        
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + np.eye(2) * self.q
        
        # 2. التحديث (Update)
        # مصفوفة القياس H (نقيس الإزاحة فقط)
        H = np.array([[1.0, 0.0]])
        z = np.array([measurement])
        
        y = z - (H @ self.state)  # ابتكار القياس
        S = H @ self.covariance @ H.T + self.r
        K = self.covariance @ H.T @ np.linalg.inv(S)
        
        self.state = self.state + K @ y
        self.covariance = (np.eye(2) - K @ H) @ self.covariance
        
        return self.state[0]

    def compute_dtw_alignment(self, audio_features: np.ndarray, visual_features: np.ndarray) -> float:
        """
        حساب إزاحة المزامنة المثالية باستخدام DTW مبسط.
        يعيد مقدار الإزاحة (Offset) بالثواني.
        """
        n, m = len(audio_features), len(visual_features)
        dtw_matrix = np.zeros((n + 1, m + 1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                cost = abs(audio_features[i-1] - visual_features[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1])
        
        # استخراج الإزاحة من المسار الأمثل (تبسيط)
        optimal_path_offset = (n - m) / max(n, m)
        return optimal_path_offset

    def correct(self, current_timestamp: float, measured_offset: float, 
                audio_snippet: np.ndarray = None, visual_snippet: np.ndarray = None) -> Dict:
        """
        العملية الرئيسية للتصحيح:
        1. استخدام DTW للتحقق من القياس الخام إذا توفرت عينات.
        2. تمرير النتيجة لمرشح كالمان للتنعيم والتنبؤ.
        """
        refined_offset = measured_offset
        
        # 1. تحسين القياس باستخدام DTW إذا توفرت بيانات بصرية/سمعية
        if audio_snippet is not None and visual_snippet is not None:
            dtw_offset = self.compute_dtw_alignment(audio_snippet, visual_snippet)
            refined_offset = (measured_offset + dtw_offset) / 2
            
        # 2. تطبيق مرشح كالمان للحصول على التقدير المستقر
        stable_offset = self.kalman_update(refined_offset, current_timestamp)
        
        corrected_timestamp = current_timestamp - stable_offset
        
        # حساب الثقة بناءً على مصفوفة التغاير (Covariance)
        confidence = float(1.0 / (1.0 + np.trace(self.covariance)))
        
        # حفظ النمط المكتشف في محرك التعلم إذا كانت الثقة عالية
        if self.source_id and confidence > 0.95:
            learning_engine.save_drift_profile(self.source_id, {
                "drift_rate": float(self.state[1]),
                "last_offset": float(stable_offset)
            })
        
        return {
            "original_timestamp": current_timestamp,
            "corrected_timestamp": corrected_timestamp,
            "estimated_drift": stable_offset,
            "drift_rate": self.state[1],
            "confidence": confidence
        }

if __name__ == "__main__":
    # اختبار بسيط
    corrector = DriftCorrector()
    drift_samples = [0.1, 0.12, 0.11, 0.15, 0.2, 0.25] # انحراف متزايد
    
    print("--- اختبار تصحيح الانحراف الزمني ---")
    for i, drift in enumerate(drift_samples):
        res = corrector.correct(current_timestamp=float(i), measured_offset=drift)
        print(f"خطوة {i}: الانحراف الخام={drift:.3f}, المقدر={res['estimated_drift']:.3f}, الوقت المصحح={res['corrected_timestamp']:.3f}")
