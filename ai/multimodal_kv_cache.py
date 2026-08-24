import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

@dataclass
class MultimodalKVCache:
    """
    مخزن الـ KV Caching لنموذج Surah.
    يحمل حالات الـ Self-Attention لكل طبقة وحالات الـ Cross-Attention للوسائط.
    """
    # Self-Attention Cache: List of (past_k, past_v) per layer
    # Each K, V has shape (n_heads, seq_len, dk)
    layer_past_kv: List[Optional[Tuple[np.ndarray, np.ndarray]]] = field(default_factory=list)
    
    # Multimodal Cross-Attention Cache: (K_mod, V_mod)
    multimodal_kv: Optional[Tuple[np.ndarray, np.ndarray]] = None
    
    # Metadata for validation
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.layer_past_kv:
            # سنقوم بتهيئتها عند أول استخدام بناءً على عدد الطبقات
            pass

    def clear(self):
        """تفريغ الذاكرة المؤقتة بالكامل."""
        self.layer_past_kv = [None] * len(self.layer_past_kv)
        self.multimodal_kv = None
        self.metadata = {}

    def get_seq_len(self) -> int:
        """إرجاع طول التسلسل المخزن حالياً."""
        if not self.layer_past_kv or self.layer_past_kv[0] is None:
            return 0
        return self.layer_past_kv[0][0].shape[1]
