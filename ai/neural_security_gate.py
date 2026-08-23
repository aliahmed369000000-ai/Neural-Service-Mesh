
import numpy as np
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger("NeuralServiceMesh.SecurityGate")

class NeuralSecurityGate:
    """
    البوابة العصبية الأمنية (Neural Security Gate):
    تقوم بتحليل النوايا الدلالية للأفعال عبر معالجة مخرجات شبكة Surah 4096.
    """
    def __init__(self, d_model: int = 4096):
        self.d_model = d_model
        # مخزن للمصنفات حسب d_model لضمان المرونة
        self._classifiers = {}
        self._biases = {}

    def _get_classifier(self, dim: int):
        if dim not in self._classifiers:
            # تهيئة أوزان تصنيف النوايا (Safe vs Malicious)
            self._classifiers[dim] = np.random.randn(dim, 2) * 0.01
            self._biases[dim] = np.zeros(2)
        return self._classifiers[dim], self._biases[dim]

    def analyze_intent(self, hidden_state: np.ndarray) -> Tuple[float, str]:
        """
        تحليل الحالة المخفية (Hidden State) لشبكة Surah لتحديد درجة الخطر.
        hidden_state: (d_model,) مخرج الطبقة الأخيرة للشبكة.
        """
        # التأكد من أن المدخل هو متجه واحد
        if hidden_state.ndim > 1:
            hidden_state = hidden_state.mean(axis=0)
            
        dim = hidden_state.shape[-1]
        classifier, bias = self._get_classifier(dim)
        
        # 1. حساب احتمالية الخطر عبر Softmax
        logits = np.dot(hidden_state, classifier) + bias
        probs = self._softmax(logits)
        risk_score = float(probs[1]) # احتمالية الخطر (Malicious)
        
        # 2. تحديد نوع الخطر (دلالياً)
        intent_label = "safe"
        if risk_score > 0.7:
            intent_label = "malicious"
        elif risk_score > 0.4:
            intent_label = "suspicious"
            
        return risk_score, intent_label

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / (e.sum() + 1e-9)

    def enforce_neural_mask(self, attention_weights: np.ndarray, trust_score: float) -> np.ndarray:
        """تطبيق القناع العصبي بناءً على الثقة."""
        if trust_score < 0.5:
            mask_threshold = 0.2
            mask = attention_weights < mask_threshold
            attention_weights[mask] = 0
            logger.warning(f"🛡️ Neural Masking Applied: Trust {trust_score} is low.")
        return attention_weights

# نسخة عالمية
neural_security_gate = NeuralSecurityGate()
