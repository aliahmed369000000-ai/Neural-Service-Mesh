import sys
import os
import numpy as np
import time

# إضافة المسار للمشروع
sys.path.append(os.getcwd())

from ai.layers.dynamic_sparse_attention_numpy import DynamicSparseAttentionNumPy
from ai.arabic_transformer import MultiHeadAttention

def test_sparsity_and_performance():
    print("--- اختبار خوارزمية الانتباه المتفرق الديناميكي (Dynamic Sparse Attention) ---")
    
    d_model = 256
    n_heads = 8
    seq_len = 64
    sparsity_k = 0.25
    
    X = np.random.randn(seq_len, d_model).astype(np.float32)
    
    # 1. اختبار الانتباه الكثيف (Dense Attention) للمقارنة
    dense_attn = MultiHeadAttention(d_model, n_heads)
    start_dense = time.time()
    out_dense = dense_attn.forward(X)
    end_dense = time.time()
    
    # 2. اختبار الانتباه المتفرق (Sparse Attention)
    sparse_attn = DynamicSparseAttentionNumPy(d_model, n_heads, sparsity_k=sparsity_k)
    start_sparse = time.time()
    out_sparse = sparse_attn.forward(X)
    end_sparse = time.time()
    
    print(f"زمن تنفيذ الانتباه الكثيف: {(end_dense - start_dense)*1000:.2f} ms")
    print(f"زمن تنفيذ الانتباه المتفرق: {(end_sparse - start_sparse)*1000:.2f} ms")
    
    # 3. التحقق من التفرقة (Sparsity)
    attn_matrix = sparse_attn._attn
    # عدد العناصر غير الصفرية (أو التي ليست -1e9 قبل softmax)
    # في softmax، القيم المحجوبة تصبح قريبة جداً من الصفر
    non_zero_elements = np.sum(attn_matrix > 1e-6)
    total_elements = attn_matrix.size
    actual_sparsity = 1 - (non_zero_elements / total_elements)
    
    print(f"نسبة التفرقة الفعلية: {actual_sparsity*100:.2f}%")
    print(f"نسبة التفرقة المستهدفة: {(1-sparsity_k)*100:.2f}%")
    
    assert actual_sparsity >= (1 - sparsity_k - 0.05), "فشل: التفرقة الفعلية أقل من المستهدفة"
    assert out_sparse.shape == X.shape, "فشل: شكل المخرجات غير صحيح"
    print("✅ نجح اختبار الانتباه المتفرق الديناميكي!")

if __name__ == "__main__":
    try:
        test_sparsity_and_performance()
    except Exception as e:
        print(f"❌ فشل الاختبار: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
