"""
اختبارات ai/arabic_transformer.py::TokenEmbedding — تحويل float64→float32
واستبدال حلقة for بـ np.subtract.at في backward() (يوليو 2026).
"""
import numpy as np

from ai.arabic_transformer import TokenEmbedding


class TestDtypeMemory:
    def test_weights_are_float32_not_float64(self):
        """float32 يحمل نصف استهلاك الذاكرة — هذا حرج على حد 1GB في Streamlit Cloud."""
        emb = TokenEmbedding(vocab_size=100, d_model=16)
        assert emb.W.dtype == np.float32

    def test_backward_output_stays_float32(self):
        emb = TokenEmbedding(vocab_size=100, d_model=16)
        emb.forward(np.array([1, 2, 3]))
        emb.backward(np.random.randn(3, 16).astype(np.float32), lr=0.01)
        assert emb.W.dtype == np.float32


class TestVectorizedBackwardCorrectness:
    def test_subtract_at_matches_naive_loop_with_duplicate_ids(self):
        """
        الاختبار الأهم: np.subtract.at يجب أن يُنتج نفس نتيجة حلقة for
        القديمة تماماً — خصوصاً عند تكرار نفس الـID داخل نفس الدفعة (كلمة
        متكررة في الجملة)، حيث الفانسي إندكسنغ العادي يفشل بصمت (يفقد
        التحديثات المتكررة بدل تراكمها).
        """
        np.random.seed(0)
        vocab, d = 50, 8
        w_init = (np.random.randn(vocab, d) * 0.02).astype(np.float64)

        # الطريقة القديمة: حلقة for
        w_loop = w_init.copy()
        ids = np.array([3, 7, 3, 3, 12, 7])  # IDs مكرَّرة عمداً
        grad = np.random.randn(len(ids), d)
        lr = 0.1
        for i, idx in enumerate(ids):
            w_loop[idx] -= lr * grad[i]
        np.clip(w_loop, -5.0, 5.0, out=w_loop)

        # الطريقة الجديدة: TokenEmbedding.backward الفعلية
        emb = TokenEmbedding(vocab_size=vocab, d_model=d)
        emb.W = w_init.astype(np.float32).copy()
        emb._last_ids = ids
        emb.backward(grad, lr=lr)

        max_diff = np.abs(w_loop.astype(np.float32) - emb.W).max()
        assert max_diff < 1e-5, (
            f"np.subtract.at لا يطابق الحلقة القديمة رياضياً! أقصى فرق: {max_diff}"
        )

    def test_weights_stay_within_clip_bounds(self):
        emb = TokenEmbedding(vocab_size=20, d_model=4)
        emb.forward(np.array([0, 1, 2]))
        huge_grad = np.full((3, 4), 1000.0, dtype=np.float32)
        emb.backward(huge_grad, lr=1.0)
        assert emb.W.min() >= -5.0 and emb.W.max() <= 5.0
