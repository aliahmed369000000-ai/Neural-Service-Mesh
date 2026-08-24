import numpy as np
from typing import List, Tuple, Optional
import json
import os

class VectorQuantizer:
    """
    محرك تكميم المتجهات لضغط ذاكرة الوكلاء بنسبة تصل لـ 75% 
    باستخدام تقنية Product Quantization المبسطة.
    """
    def __init__(self, dimension: int = 1536, num_centroids: int = 256):
        self.dimension = dimension
        self.num_centroids = num_centroids
        self.centroids = None
        self.codebook_path = "artifacts/learning/quantization_codebook.json"

    def _initialize_centroids(self, data: np.ndarray):
        """تهيئة المراكز باستخدام عينة عشوائية من البيانات."""
        indices = np.random.choice(data.shape[0], self.num_centroids, replace=False)
        self.centroids = data[indices]

    def train(self, data: np.ndarray, iterations: int = 10):
        """تدريب المحرك على بيانات المتجهات لبناء كتاب الرموز (Codebook)."""
        if data.shape[0] < self.num_centroids:
            self.centroids = data
            return

        self._initialize_centroids(data)
        
        for _ in range(iterations):
            # حساب المسافات وتعيين النقاط لأقرب مركز
            distances = np.linalg.norm(data[:, np.newaxis] - self.centroids, axis=2)
            labels = np.argmin(distances, axis=1)
            
            # تحديث المراكز
            new_centroids = np.array([data[labels == i].mean(axis=0) if len(data[labels == i]) > 0 
                                     else self.centroids[i] for i in range(self.num_centroids)])
            self.centroids = new_centroids

    def quantize(self, vector: np.ndarray) -> int:
        """تحويل المتجه إلى رمز (Index) مضغوط."""
        if self.centroids is None:
            # إذا لم يتم التدريب، نستخدم تهيئة افتراضية أو نرجع المتجه كما هو
            return vector.tolist()
        
        distances = np.linalg.norm(vector - self.centroids, axis=1)
        return int(np.argmin(distances))

    def dequantize(self, index: int) -> np.ndarray:
        """استعادة المتجه التقريبي من الرمز المضغوط."""
        if self.centroids is None:
            return None
        return self.centroids[index]

    def save_codebook(self):
        """حفظ كتاب الرموز للاستخدام المستقبلي."""
        os.makedirs(os.path.dirname(self.codebook_path), exist_ok=True)
        if self.centroids is not None:
            with open(self.codebook_path, 'w') as f:
                json.dump(self.centroids.tolist(), f)

    def load_codebook(self):
        """تحميل كتاب الرموز المحفوظ."""
        if os.path.exists(self.codebook_path):
            with open(self.codebook_path, 'r') as f:
                self.centroids = np.array(json.load(f))
            return True
        return False

    def compress_batch(self, vectors: List[List[float]]) -> List[int]:
        """ضغط مجموعة من المتجهات دفعة واحدة."""
        np_vectors = np.array(vectors)
        if self.centroids is None:
            self.train(np_vectors)
        
        distances = np.linalg.norm(np_vectors[:, np.newaxis] - self.centroids, axis=2)
        return np.argmin(distances, axis=1).tolist()
