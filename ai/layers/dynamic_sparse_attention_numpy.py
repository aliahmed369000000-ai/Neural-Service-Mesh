import numpy as np
from typing import Optional

def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

class DynamicSparseAttentionNumPy:
    """
    نسخة NumPy من خوارزمية الانتباه المتفرق الديناميكي.
    مصممة للعمل مع ArabicTransformer و MoEArabicTransformer في المشروع.
    """
    def __init__(self, d_model: int, n_heads: int, sparsity_k: float = 0.2):
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.sparsity_k = sparsity_k
        
        # أوزان Q, K, V, O
        self.Wq = np.random.randn(d_model, d_model) * 0.02
        self.Wk = np.random.randn(d_model, d_model) * 0.02
        self.Wv = np.random.randn(d_model, d_model) * 0.02
        self.Wo = np.random.randn(d_model, d_model) * 0.02
        
        self._X = self._attn = self._concat = None

    def forward(self, X: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        self._X = X
        seq_len, _ = X.shape
        
        # Linear projections
        Q = X @ self.Wq.T
        K = X @ self.Wk.T
        V = X @ self.Wv.T
        
        # Reshape to heads
        Qh = Q.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        Kh = K.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        Vh = V.reshape(seq_len, self.n_heads, self.d_head).transpose(1, 0, 2)
        
        # Scaled dot-product attention
        scores = (Qh @ Kh.transpose(0, 2, 1)) / np.sqrt(self.d_head)
        
        if mask is not None:
            scores = np.where(mask[None], -1e9, scores)
            
        # --- Dynamic Sparsity (Vectorized) ---
        k_elements = max(1, int(seq_len * self.sparsity_k))
        
        if k_elements < seq_len:
            # استخدام np.partition بشكل متجه عبر المحور الأخير
            # نأخذ العتبة لكل صف في كل رأس
            thresholds = np.partition(scores, -k_elements, axis=-1)[..., -k_elements, None]
            sparse_scores = np.where(scores >= thresholds, scores, -1e9)
        else:
            sparse_scores = scores
        
        attn = _softmax(sparse_scores)
        self._attn = attn
        
        out = attn @ Vh
        self._concat = out.transpose(1, 0, 2).reshape(seq_len, self.d_model)
        
        return self._concat @ self.Wo.T
