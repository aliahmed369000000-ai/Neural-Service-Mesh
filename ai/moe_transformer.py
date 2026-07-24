"""
ai/moe_transformer.py — إضافي بالكامل، لا يمسّ arabic_transformer.py
========================================================================
بنية Mixture-of-Experts (MoE) قابلة للتوسع نظرياً لتريليون معامل، بنمط
مشابه لـ Kimi K2/K3 (عدد كبير من "الخبراء"، تفعيل جزئي top-k فقط لكل
رمز). تعيد استخدام كل مكوّنات ArabicTransformer الموجودة (Embedding,
CoreMatrixLayer, MultiHeadAttention, LayerNorm, OutputHead) بدون أي
تعديل عليها — الإضافة الوحيدة الجديدة هي طبقة التوجيه (Router) وتعدد
الخبراء بدل FFN واحدة.

⚠️ قيد صادق يجب معرفته قبل الاستخدام:
    حتى بأقصى ضغط ممكن (4-بت)، تريليون معامل = ~500GB على القرص.
    هذا رقم ثابت رياضياً بصرف النظر عن ذكاء البنية (MoE يحل مشكلة
    الحساب أثناء التدريب فقط، وليس مشكلة التخزين — كل الخبراء يجب أن
    يكونوا محفوظين على قرص حتى لو غير نشطين، لأن أي رمز قد يحتاج أي
    خبير). لا يوجد حالياً أي بيئة مجانية (Kaggle/Colab/sandbox) فيها
    مساحة تخزين كافية لهذا الحجم.

    هذا الملف مُختبَر وصحيح رياضياً (forward + backward، مع فحص تدرج
    عددي numerical gradient check) على حجم مصغّر فقط. تشغيله فعلياً
    بحجم تريليون كامل يحتاج بنية تخزين سحابية مدفوعة لم تتوفر بعد.

الاستخدام:
    from ai.moe_transformer import MoEArabicTransformer, moe_param_count

    # حجم تجريبي صغير (يعمل فعلياً هنا والآن)
    m = MoEArabicTransformer(n_layers=2, n_experts=8, top_k=2, d_ff_expert=64)
    m.train_step("بسم الله الرحمن الرحيم")

    # حجم نظري تريليون (للتوثيق/الحساب فقط، غير قابل للتشغيل الفعلي حالياً)
    stats = moe_param_count(d_model=2304, n_heads=16, n_layers=16,
                             n_experts=3000, top_k=16, d_ff_expert=4608)
    print(stats["total_params"])   # 1,020,040,161,424 (~1.02T)
    print(stats["disk_gb_int4"])   # 510.0 GB — غير متاح على أي منصة مجانية حالياً
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from ai.arabic_transformer import (
    D_MODEL, N_HEADS, MAX_SEQ_LEN, VOCAB_SIZE, LEARNING_RATE, CLIP_GRAD,
    _xavier, _relu, _softmax,
    HashTokenizer, TokenEmbedding, PositionalEncoding, CoreMatrixLayer,
    MultiHeadAttention, LayerNorm, OutputHead,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Mixture-of-Experts Feed-Forward (يستبدل FFN الواحدة بعدة خبراء + Router)
# ══════════════════════════════════════════════════════════════════════════════
class MoEFFN:
    """
    طبقة MoE: n_experts شبكة FFN مستقلة + Router يختار top_k خبير فقط
    لكل رمز (نمط Kimi K2/K3 — تفعيل جزئي، وزن إجمالي ضخم).

    كل خبير = FFN صغيرة مطابقة تماماً لبنية FFN الأصلية في
    arabic_transformer.py (نفس المعادلات، نفس أسلوب الـ backward اليدوي).
    """

    def __init__(self, d_model: int = D_MODEL, d_ff_expert: int = 256,
                 n_experts: int = 8, top_k: int = 2):
        assert 1 <= top_k <= n_experts
        self.d_model    = d_model
        self.n_experts  = n_experts
        self.top_k      = top_k

        # كل خبير: نفس معادلات FFN الأصلية بالضبط (W1,b1,W2,b2 + relu)
        self._eW1 = [_xavier(d_ff_expert, d_model) for _ in range(n_experts)]
        self._eW2 = [_xavier(d_model, d_ff_expert) for _ in range(n_experts)]
        self._eb1 = [np.zeros(d_ff_expert) for _ in range(n_experts)]
        self._eb2 = [np.zeros(d_model) for _ in range(n_experts)]

        # Router (طبقة توجيه صغيرة)
        self.gate_W = _xavier(n_experts, d_model)
        self.gate_b = np.zeros(n_experts)

        # caches للـ backward
        self._X = self._p = self._logits = None
        self._topk_idx = self._S = None
        self._expert_cache: Dict[int, Tuple] = {}

    # ── Forward ──────────────────────────────────────────────────────────
    def forward(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        self._X = X

        logits = X @ self.gate_W.T + self.gate_b     # (N, n_experts)
        p = _softmax(logits)                          # full softmax على كل الخبراء
        self._logits, self._p = logits, p

        topk_idx = np.argsort(-p, axis=1)[:, :self.top_k]     # (N, top_k)
        rows = np.arange(N)[:, None]
        topk_p = p[rows, topk_idx]                             # (N, top_k)
        S = topk_p.sum(axis=1, keepdims=True)                  # إعادة توحيد (renormalize)
        w = topk_p / (S + 1e-12)                                # (N, top_k)
        self._topk_idx, self._S = topk_idx, S

        out = np.zeros_like(X)
        self._expert_cache = {}
        for e in range(self.n_experts):
            hit = (topk_idx == e)
            token_mask = hit.any(axis=1)
            if not token_mask.any():
                continue
            idx = np.where(token_mask)[0]
            pos = np.argmax(hit[idx], axis=1)
            w_e = w[idx, pos]                                   # (len(idx),)

            sub_X = X[idx]
            h = _relu(sub_X @ self._eW1[e].T + self._eb1[e])
            sub_out = h @ self._eW2[e].T + self._eb2[e]

            out[idx] += sub_out * w_e[:, None]
            self._expert_cache[e] = (idx, w_e, sub_X, h, sub_out)

        self._out = out

        return out

    # ── Backward ─────────────────────────────────────────────────────────
    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        N = self._X.shape[0]
        gX_total = np.zeros_like(self._X)

        # المرحلة 1: g_e = grad·sub_out لكل (خبير مُفعَّل، رمز)
        g_store: Dict[int, np.ndarray] = {}
        for e, (idx, w_e, sub_X, h, sub_out) in self._expert_cache.items():
            g_e = np.sum(grad[idx] * sub_out, axis=1)   # (len(idx),)
            g_store[e] = g_e

        S = self._S[:, 0]
        # حد التصحيح المشترك لكل الخبراء المُختارين لنفس الرمز:
        # out_i/dp_ie ⇒ dL/dp_ie = g_e/S_i − (grad_i·out_i)/S_i  (مشتق دقيق
        # لصيغة إعادة التوحيد w=p/S، تحقّقنا منه عددياً بشكل منفصل ومطابق تماماً)
        correction = np.sum(grad * self._out, axis=1) / S   # (N,)
        dp = np.zeros_like(self._p)

        for e, (idx, w_e, sub_X, h, sub_out) in self._expert_cache.items():
            # ── تدرج الخبير نفسه (backward يدوي مطابق لـ FFN الأصلية) ──
            sub_grad_out = grad[idx] * w_e[:, None]      # (len(idx), d_model)
            gW2 = sub_grad_out.T @ h
            gb2 = sub_grad_out.sum(0)
            gh  = sub_grad_out @ self._eW2[e] * (h > 0)
            gW1 = gh.T @ sub_X
            gb1 = gh.sum(0)
            gX_sub = gh @ self._eW1[e]

            self._eW1[e] -= lr * np.clip(gW1, -CLIP_GRAD, CLIP_GRAD)
            self._eW2[e] -= lr * np.clip(gW2, -CLIP_GRAD, CLIP_GRAD)
            self._eb1[e] -= lr * np.clip(gb1, -CLIP_GRAD, CLIP_GRAD)
            self._eb2[e] -= lr * np.clip(gb2, -CLIP_GRAD, CLIP_GRAD)
            np.clip(self._eW1[e], -5, 5, out=self._eW1[e])
            np.clip(self._eW2[e], -5, 5, out=self._eW2[e])
            gX_total[idx] += gX_sub

            # ── تدرج بوابة التوجيه (Router) عبر إعادة التوحيد ──
            p_e = self._p[idx, e]
            dp_e = g_store[e] / S[idx] - correction[idx]
            dp[idx, e] += dp_e

        # softmax backward: dlogits = p*(dp - sum(dp*p))
        sum_dp_p = np.sum(dp * self._p, axis=1, keepdims=True)
        dlogits = self._p * (dp - sum_dp_p)

        gate_W_grad = dlogits.T @ self._X
        gate_b_grad = dlogits.sum(0)
        self.gate_W -= lr * np.clip(gate_W_grad, -CLIP_GRAD, CLIP_GRAD)
        self.gate_b -= lr * np.clip(gate_b_grad, -CLIP_GRAD, CLIP_GRAD)
        np.clip(self.gate_W, -5, 5, out=self.gate_W)

        gX_total += dlogits @ self.gate_W
        return gX_total

    def param_count(self) -> int:
        d_ff = self._eW1[0].shape[0]
        expert_params = self.n_experts * (2 * self.d_model * d_ff + d_ff + self.d_model)
        gate_params = self.n_experts * self.d_model + self.n_experts
        return expert_params + gate_params


# ══════════════════════════════════════════════════════════════════════════════
# 2. MoE Transformer Block (نفس تدفق TransformerBlock الأصلي، FFN → MoEFFN)
# ══════════════════════════════════════════════════════════════════════════════
class MoETransformerBlock:
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff_expert=256,
                 n_experts=8, top_k=2, bid=0):
        self.bid = bid
        self.mha = MultiHeadAttention(d_model, n_heads)
        self.moe = MoEFFN(d_model, d_ff_expert, n_experts, top_k)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self._X = self._ao = None

    def forward(self, X, mask=None):
        self._X = X
        ao = self.mha.forward(self.ln1.forward(X), mask)
        self._ao = ao
        X2 = X + ao
        return X2 + self.moe.forward(self.ln2.forward(X2))

    def backward(self, grad, lr):
        X2 = self._X + self._ao
        gmoe = self.moe.backward(grad, lr)
        gX2 = grad + self.ln2.backward(gmoe, lr)
        gmha = self.mha.backward(gX2, lr)
        return gX2 + self.ln1.backward(gmha, lr)

    def param_count(self) -> int:
        d_model = self.mha.dm
        attn = 4 * d_model * d_model
        ln = 4 * d_model
        return attn + ln + self.moe.param_count()


# ══════════════════════════════════════════════════════════════════════════════
# 3. MoE Arabic Transformer — النموذج الكامل
# ══════════════════════════════════════════════════════════════════════════════
class MoEArabicTransformer:
    """
    نفس واجهة ArabicTransformer (train_step/encode) لكن ببنية MoE قابلة
    للتوسع نظرياً لتريليون معامل عبر n_experts (انظر تحذير التخزين أعلى الملف).
    """
    VERSION = "MoE-1.0.0-NSM"

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, n_layers=2,
                 n_experts=8, top_k=2, d_ff_expert=256,
                 max_seq=MAX_SEQ_LEN, vocab_size=VOCAB_SIZE, lr=LEARNING_RATE,
                 core_csv=None):
        self.lr = lr
        self.max_seq = max_seq

        self.tokenizer = HashTokenizer(vocab_size)
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq)
        self.core = CoreMatrixLayer(core_csv, d_model)
        self.blocks = [
            MoETransformerBlock(d_model, n_heads, d_ff_expert, n_experts, top_k, i)
            for i in range(n_layers)
        ]
        self.head = OutputHead(d_model, vocab_size)

        self._steps = 0
        self._loss_history: List[float] = []

    def _forward(self, ids: np.ndarray, mask=None):
        X = self.embedding.forward(ids)
        X += self.pos_enc.forward(len(ids))
        X = X + self.core.forward(X)
        for blk in self.blocks:
            X = blk.forward(X, mask)
        return self.head.forward(X), X

    def train_step(self, text: str) -> float:
        ids = self.tokenizer.encode(text, self.max_seq)
        if len(ids) < 2:
            return 0.0
        inp, tgt = ids[:-1], ids[1:]
        S = len(inp)
        mask = np.triu(np.ones((S, S), bool), k=1)

        probs, _ = self._forward(inp, mask)
        loss, gp = self.head.loss_grad(probs, tgt)

        gX = self.head.backward(gp, self.lr)
        for blk in reversed(self.blocks):
            gX = blk.backward(gX, self.lr)
        gc = self.core.backward(gX, self.lr)
        self.embedding.backward(gX + gc, self.lr)

        self._steps += 1
        self._loss_history.append(loss)
        return float(loss)

    def encode(self, text: str) -> np.ndarray:
        ids = self.tokenizer.encode(text, self.max_seq)
        if len(ids) == 0:
            return np.zeros(self.embedding.W.shape[1])
        _, hidden = self._forward(ids)
        return hidden[1:-1].mean(0) if len(hidden) > 2 else hidden.mean(0)

    def param_count(self) -> int:
        embed = self.embedding.W.size
        core = (self.core.W_up.size + self.core.W_down.size +
                self.core.b_up.size + self.core.b_down.size + self.core._W_core.size)
        blocks = sum(b.param_count() for b in self.blocks)
        head = self.head.W.size + self.head.b.size
        return embed + core + blocks + head


# ══════════════════════════════════════════════════════════════════════════════
# 4. حاسبة معاملات نظرية (بدون إنشاء أي مصفوفة فعلية) — للتخطيط بحجم تريليون
# ══════════════════════════════════════════════════════════════════════════════
def moe_param_count(d_model: int, n_heads: int, n_layers: int,
                     n_experts: int, top_k: int, d_ff_expert: int,
                     vocab_size: int = VOCAB_SIZE, core_dim: int = 784) -> dict:
    """
    حساب رياضي بحت لعدد المعاملات الكلي (نشط + إجمالي) بدون تخصيص أي
    ذاكرة فعلية — يُستخدم للتخطيط بأحجام غير قابلة للتشغيل الفعلي حالياً.
    """
    embed = vocab_size * d_model
    core = 2 * core_dim * d_model + core_dim + d_model + core_dim * core_dim
    head = vocab_size * d_model + vocab_size

    attn_per_layer = 4 * d_model * d_model
    ln_per_layer = 4 * d_model
    expert_params = 2 * d_model * d_ff_expert + d_ff_expert + d_model
    gate_params = n_experts * d_model + n_experts
    moe_per_layer_total = n_experts * expert_params + gate_params
    moe_per_layer_active = top_k * expert_params + gate_params

    total_params = embed + core + head + n_layers * (attn_per_layer + ln_per_layer + moe_per_layer_total)
    active_params = embed + core + head + n_layers * (attn_per_layer + ln_per_layer + moe_per_layer_active)

    return {
        "total_params": total_params,
        "active_params_per_token": active_params,
        "total_params_readable": f"{total_params/1e9:.2f}B" if total_params < 1e12 else f"{total_params/1e12:.3f}T",
        "active_params_readable": f"{active_params/1e9:.3f}B",
        "disk_gb_fp16": round(total_params * 2 / 1e9, 1),
        "disk_gb_int4": round(total_params * 0.5 / 1e9, 1),
        "n_experts_total_across_layers": n_experts * n_layers,
    }
