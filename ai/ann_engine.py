import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import json
import os
from pathlib import Path

class ANNEngine:
    """
    محرك البحث التقريبي لأقرب جار (ANN Engine) لمشروع NSM.
    يستخدم فهرسة بسيطة تعتمد على مسافة الجيب التمام (Cosine Similarity)
    مع إمكانية التوسع لتقنيات مثل HNSW أو LSH مستقبلاً.
    """
    def __init__(self, dimension: int = 1536):
        self.dimension = dimension
        self.vectors = []
        self.metadata = []
        self.index_built = False

    def add_vector(self, vector: np.ndarray, meta: Dict[str, Any]):
        """إضافة متجه للفهرس."""
        if vector.shape[0] != self.dimension:
            raise ValueError(f"بعد المتجه {vector.shape[0]} لا يطابق البعد المطلوب {self.dimension}")
        
        # تطبيع المتجه لحساب مسافة الجيب التمام عبر الضرب النقطي
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
        self.vectors.append(vector)
        self.metadata.append(meta)
        self.index_built = False

    def build_index(self):
        """بناء الفهرس للبحث السريع."""
        if not self.vectors:
            return
        self.vectors_np = np.array(self.vectors)
        self.index_built = True

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """البحث عن أقرب المتجهات لمتجه الاستعلام."""
        if not self.index_built:
            self.build_index()
            
        if not self.vectors:
            return []

        # تطبيع متجه الاستعلام
        norm = np.linalg.norm(query_vector)
        if norm > 0:
            query_vector = query_vector / norm

        # حساب التشابه عبر الضرب النقطي (Cosine Similarity للمتجهات المطبعة)
        similarities = np.dot(self.vectors_np, query_vector)
        
        # الحصول على أفضل k نتائج
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.metadata[idx], float(similarities[idx])))
            
        return results

    def save_index(self, path: str):
        """حفظ الفهرس والمتجهات."""
        data = {
            "vectors": [v.tolist() for v in self.vectors],
            "metadata": self.metadata
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    def load_index(self, path: str):
        """تحميل الفهرس من ملف."""
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.vectors = [np.array(v) for v in data["vectors"]]
                self.metadata = data["metadata"]
                self.build_index()
            return True
        return False
