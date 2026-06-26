"""
LoRA Adapter — NSM v18.3
========================
Low-Rank Adaptation فوق arabic_transformer.py بدون تعديل أي ملف أصلي.

المبدأ الرياضي:
    W_eff = W_frozen + ΔW
    ΔW    = (B @ A) * scale          shape: (d_out, d_in)
    A     ∈ R^(rank × d_in)          مُهيَّأ عشوائياً (kaiming)
    B     ∈ R^(d_out × rank)         مُهيَّأ بالأصفار → ΔW=0 عند البداية
    scale = alpha / rank

التغطية:
    ✓ MultiHeadAttention  → Wq, Wk, Wv, Wo  (4 adapters / طبقة)
    ✓ FFN                 → W1, W2           (2 adapters / طبقة)
    ✓ N_LAYERS=4 → 24 adapter مجموعاً
    ✓ حفظ / تحميل adapter weights فقط (KB بدلاً من MB)
    ✓ دمج (merge) في الأوزان الأساسية → inference سريع
    ✓ backprop كامل عبر الـ adapters فقط (base frozen)

الاستخدام السريع:
    from ai.lora_adapter import LoRATransformerAdapter
    from ai.arabic_transformer import ArabicTransformer

    base    = ArabicTransformer()
    adapter = LoRATransformerAdapter(base, rank=8, alpha=16.0)

    out = adapter.forward(tokens)          # inference مع LoRA
    adapter.train_step(tokens, targets)    # ضبط الـ adapters فقط
    adapter.save("models/lora/nsm_lora")   # يحفظ KB لا MB
    adapter.merge_into_base()              # يدمج ΔW في الأوزان الأساسية
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("LoRAAdapter")

CLIP_GRAD = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# 1. LoRALayer — اللبنة الأساسية
# ══════════════════════════════════════════════════════════════════════════════

class LoRALayer:
    """
    طبقة LoRA واحدة لمصفوفة وزن واحدة.

    تُضيف ΔW = (B @ A) * scale إلى إخراج المصفوفة الأصلية W_frozen.
    W_frozen لا تُحرَّك أبداً — فقط A و B يتدربان.

    الحجم:
        d_in=256, d_out=256, rank=8  →  256×8 + 8×256 = 4,096 param
        مقارنةً بـ W_full=256×256   = 65,536 param  (توفير 94%)
    """

    def __init__(
        self,
        d_in:  int,
        d_out: int,
        rank:  int   = 8,
        alpha: float = 16.0,
        name:  str   = "",
    ):
        self.d_in  = d_in
        self.d_out = d_out
        self.rank  = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.name  = name

        # A: تهيئة كايمنج (مثل PyTorch LoRA)
        std = 1.0 / math.sqrt(d_in)
        self.A = np.random.uniform(-std, std, (rank, d_in)).astype(np.float64)
        # B: أصفار → ΔW=0 عند البداية (لا تأثير على النموذج الأصلي)
        self.B = np.zeros((d_out, rank), dtype=np.float64)

        # cache للـ backward
        self._x:  Optional[np.ndarray] = None

        # Adam state
        self._mA = np.zeros_like(self.A)
        self._vA = np.zeros_like(self.A)
        self._mB = np.zeros_like(self.B)
        self._vB = np.zeros_like(self.B)
        self._t  = 0

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        x: (seq, d_in)
        returns delta: (seq, d_out)  =  x @ A.T @ B.T * scale
        """
        self._x  = x
        lora_out = (x @ self.A.T) @ self.B.T * self.scale  # (seq, d_out)
        return lora_out

    # ── backward ─────────────────────────────────────────────────────────

    def backward(
        self,
        grad:    np.ndarray,          # dL/d(output) : (seq, d_out)
        lr:      float,
        use_adam: bool = True,
    ) -> np.ndarray:
        """
        يُحدِّث A و B ويُعيد dL/dx_lora للجمع مع gradient الـ base weight.

        المشتقات:
            dL/dB  = grad.T @ (x @ A.T) * scale      (d_out, rank)
            dL/dA  = (B.T @ grad.T @ x) * scale      (rank, d_in)
            dL/dx  = grad @ B @ A * scale             (seq, d_in)
        """
        x = self._x                            # (seq, d_in)
        xA = x @ self.A.T                      # (seq, rank)

        gB = (grad.T @ xA) * self.scale        # (d_out, rank)
        gA = (self.B.T @ grad.T @ x) * self.scale  # (rank, d_in)
        gx = (grad @ self.B) @ self.A * self.scale  # (seq, d_in)

        # clip
        gB = np.clip(gB, -CLIP_GRAD, CLIP_GRAD)
        gA = np.clip(gA, -CLIP_GRAD, CLIP_GRAD)

        if use_adam:
            self._t += 1
            t = self._t
            b1, b2, eps = 0.9, 0.999, 1e-8

            self._mA = b1 * self._mA + (1 - b1) * gA
            self._vA = b2 * self._vA + (1 - b2) * gA ** 2
            mA_h = self._mA / (1 - b1 ** t)
            vA_h = self._vA / (1 - b2 ** t)
            self.A -= lr * mA_h / (np.sqrt(vA_h) + eps)

            self._mB = b1 * self._mB + (1 - b1) * gB
            self._vB = b2 * self._vB + (1 - b2) * gB ** 2
            mB_h = self._mB / (1 - b1 ** t)
            vB_h = self._vB / (1 - b2 ** t)
            self.B -= lr * mB_h / (np.sqrt(vB_h) + eps)
        else:
            self.A -= lr * gA
            self.B -= lr * gB

        np.clip(self.A, -5.0, 5.0, out=self.A)
        np.clip(self.B, -5.0, 5.0, out=self.B)

        return gx

    # ── دمج ΔW في الوزن الأساسي ──────────────────────────────────────────

    def delta_W(self) -> np.ndarray:
        """ΔW = (B @ A) * scale  — shape: (d_out, d_in)"""
        return (self.B @ self.A) * self.scale

    def merge_into(self, W: np.ndarray) -> np.ndarray:
        """أضف ΔW إلى مصفوفة الوزن الأصلية وأعدها."""
        return W + self.delta_W()

    # ── حفظ / تحميل ──────────────────────────────────────────────────────

    def save(self, prefix: str):
        np.save(f"{prefix}_A.npy", self.A)
        np.save(f"{prefix}_B.npy", self.B)

    def load(self, prefix: str):
        pA, pB = f"{prefix}_A.npy", f"{prefix}_B.npy"
        if os.path.exists(pA):
            self.A = np.load(pA).astype(np.float64)
        if os.path.exists(pB):
            self.B = np.load(pB).astype(np.float64)

    # ── إحصاء ────────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return self.A.size + self.B.size

    def info(self) -> Dict:
        dw = self.delta_W()
        return {
            "name":   self.name,
            "rank":   self.rank,
            "alpha":  self.alpha,
            "scale":  round(self.scale, 4),
            "params": self.param_count(),
            "delta_norm": round(float(np.linalg.norm(dw)), 6),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2. LoRAAttentionAdapter — يُغلّف MultiHeadAttention
# ══════════════════════════════════════════════════════════════════════════════

class LoRAAttentionAdapter:
    """
    يُضيف 4 adapters (Wq, Wk, Wv, Wo) على MultiHeadAttention موجودة.
    لا يُعدّل الـ MHA الأصلية — يُضيف فقط ΔW في وقت الـ forward.
    """

    def __init__(self, mha, rank: int = 8, alpha: float = 16.0, layer_id: int = 0):
        self.mha  = mha
        self.rank = rank
        dm        = mha.dm
        prefix    = f"L{layer_id}"

        self.lq = LoRALayer(dm, dm, rank, alpha, f"{prefix}_q")
        self.lk = LoRALayer(dm, dm, rank, alpha, f"{prefix}_k")
        self.lv = LoRALayer(dm, dm, rank, alpha, f"{prefix}_v")
        self.lo = LoRALayer(dm, dm, rank, alpha, f"{prefix}_o")

    # ── forward مُعدَّل (يحقن ΔW بدون لمس الـ MHA) ─────────────────────

    def forward(self, X: np.ndarray, mask=None) -> np.ndarray:
        mha  = self.mha
        S    = len(X)

        # Q, K, V مع LoRA
        Q = X @ mha.Wq.T + self.lq.forward(X)
        K = X @ mha.Wk.T + self.lk.forward(X)
        V = X @ mha.Wv.T + self.lv.forward(X)

        mha._X = X
        mha._Q, mha._K, mha._V = Q, K, V

        Qh = Q.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)
        Kh = K.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)
        Vh = V.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)

        sc = Qh @ Kh.transpose(0, 2, 1) / math.sqrt(mha.dk)
        if mask is not None:
            sc = np.where(mask[None], -1e9, sc)
        at  = _softmax(sc)
        mha._attn = at

        out         = at @ Vh
        mha._concat = out.transpose(1, 0, 2).reshape(S, mha.dm)

        # O projection مع LoRA
        return mha._concat @ mha.Wo.T + self.lo.forward(mha._concat)

    # ── backward: يُحدِّث adapters فقط، base frozen ──────────────────────

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        mha = self.mha
        S   = mha._X.shape[0]

        # backward عبر Wo + LoRA_o
        gWo_lora = self.lo.backward(grad, lr)
        gcat     = grad @ mha.Wo + gWo_lora

        gh  = gcat.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)
        at  = mha._attn
        Vh  = mha._V.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)
        Qh  = mha._Q.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)
        Kh  = mha._K.reshape(S, mha.h, mha.dk).transpose(1, 0, 2)

        gV  = at.transpose(0, 2, 1) @ gh
        gat = gh @ Vh.transpose(0, 2, 1)
        s   = at
        gsc = s * (gat - (gat * s).sum(-1, keepdims=True))
        gsc /= math.sqrt(mha.dk)

        gQ = gsc @ Kh
        gK = gsc.transpose(0, 2, 1) @ Qh

        gQf = gQ.transpose(1, 0, 2).reshape(S, mha.dm)
        gKf = gK.transpose(1, 0, 2).reshape(S, mha.dm)
        gVf = gV.transpose(1, 0, 2).reshape(S, mha.dm)

        # backward عبر Q,K,V adapters فقط (base Wq/Wk/Wv frozen)
        gxQ = self.lq.backward(gQf, lr)
        gxK = self.lk.backward(gKf, lr)
        gxV = self.lv.backward(gVf, lr)

        # تدفق gradient إلى X
        gX = (gQf @ mha.Wq + gxQ
            + gKf @ mha.Wk + gxK
            + gVf @ mha.Wv + gxV)
        return gX

    # ── إحصاء ────────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(l.param_count() for l in [self.lq, self.lk, self.lv, self.lo])

    def merge_into_base(self):
        """يدمج ΔW الـ 4 adapters في أوزان MHA الأصلية."""
        mha      = self.mha
        mha.Wq   = self.lq.merge_into(mha.Wq)
        mha.Wk   = self.lk.merge_into(mha.Wk)
        mha.Wv   = self.lv.merge_into(mha.Wv)
        mha.Wo   = self.lo.merge_into(mha.Wo)
        # أصفر الـ adapters بعد الدمج
        for l in [self.lq, self.lk, self.lv, self.lo]:
            l.B[:] = 0.0
        logger.info(f"[LoRA] دُمجت adapters MHA في الأوزان الأساسية")

    def save(self, prefix: str):
        for tag, l in [("q", self.lq), ("k", self.lk),
                       ("v", self.lv), ("o", self.lo)]:
            l.save(f"{prefix}_{tag}")

    def load(self, prefix: str):
        for tag, l in [("q", self.lq), ("k", self.lk),
                       ("v", self.lv), ("o", self.lo)]:
            l.load(f"{prefix}_{tag}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. LoRAFFNAdapter — يُغلّف FFN
# ══════════════════════════════════════════════════════════════════════════════

class LoRAFFNAdapter:
    """
    يُضيف 2 adapters (W1, W2) على FFN موجودة.
    """

    def __init__(self, ffn, rank: int = 8, alpha: float = 16.0, layer_id: int = 0):
        self.ffn  = ffn
        dm        = ffn.W2.shape[0]
        d_ff      = ffn.W1.shape[0]
        prefix    = f"L{layer_id}"

        self.l1 = LoRALayer(dm,  d_ff, rank, alpha, f"{prefix}_ffn1")
        self.l2 = LoRALayer(d_ff, dm,  rank, alpha, f"{prefix}_ffn2")

    def forward(self, X: np.ndarray) -> np.ndarray:
        ffn       = self.ffn
        ffn._X    = X
        h_base    = _relu(X @ ffn.W1.T + ffn.b1)
        h_lora    = _relu(X @ ffn.W1.T + ffn.b1 + self.l1.forward(X))
        ffn._h    = h_lora
        out_base  = h_lora @ ffn.W2.T + ffn.b2
        out_lora  = self.l2.forward(h_lora)
        return out_base + out_lora

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        ffn = self.ffn

        # backward عبر W2 + LoRA_2
        g_lora2 = self.l2.backward(grad, lr)
        g_h     = (grad + g_lora2) @ ffn.W2 * (ffn._h > 0)

        # backward عبر W1 + LoRA_1
        g_lora1 = self.l1.backward(g_h, lr)
        gX      = g_h @ ffn.W1 + g_lora1

        return gX

    def param_count(self) -> int:
        return self.l1.param_count() + self.l2.param_count()

    def merge_into_base(self):
        ffn    = self.ffn
        ffn.W1 = self.l1.merge_into(ffn.W1)
        ffn.W2 = self.l2.merge_into(ffn.W2)
        for l in [self.l1, self.l2]:
            l.B[:] = 0.0
        logger.info(f"[LoRA] دُمجت adapters FFN في الأوزان الأساسية")

    def save(self, prefix: str):
        self.l1.save(f"{prefix}_1")
        self.l2.save(f"{prefix}_2")

    def load(self, prefix: str):
        self.l1.load(f"{prefix}_1")
        self.l2.load(f"{prefix}_2")


# ══════════════════════════════════════════════════════════════════════════════
# 4. LoRABlockAdapter — يُغلّف TransformerBlock كاملاً
# ══════════════════════════════════════════════════════════════════════════════

class LoRABlockAdapter:
    """
    يُغلّف TransformerBlock ويُضيف LoRA على MHA + FFN.
    LayerNorm تبقى مجمّدة (لا adapters لها — صغيرة ولا قيمة مضافة).
    """

    def __init__(self, block, rank: int = 8, alpha: float = 16.0):
        self.block = block
        self.attn  = LoRAAttentionAdapter(block.mha, rank, alpha, block.bid)
        self.ffn   = LoRAFFNAdapter(block.ffn,       rank, alpha, block.bid)

    def forward(self, X: np.ndarray, mask=None) -> np.ndarray:
        block     = self.block
        block._X  = X

        ao        = self.attn.forward(block.ln1.forward(X), mask)
        block._ao = ao
        X2        = X + ao
        return X2 + self.ffn.forward(block.ln2.forward(X2))

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        block  = self.block
        X2     = block._X + block._ao

        gffn   = self.ffn.backward(grad, lr)
        gX2    = grad + block.ln2.backward(gffn, lr)
        gmha   = self.attn.backward(gX2, lr)
        return gX2 + block.ln1.backward(gmha, lr)

    def param_count(self) -> int:
        return self.attn.param_count() + self.ffn.param_count()

    def merge_into_base(self):
        self.attn.merge_into_base()
        self.ffn.merge_into_base()

    def save(self, prefix: str):
        self.attn.save(f"{prefix}_attn")
        self.ffn.save(f"{prefix}_ffn")

    def load(self, prefix: str):
        self.attn.load(f"{prefix}_attn")
        self.ffn.load(f"{prefix}_ffn")


# ══════════════════════════════════════════════════════════════════════════════
# 5. LoRATransformerAdapter — الواجهة الكاملة
# ══════════════════════════════════════════════════════════════════════════════

class LoRATransformerAdapter:
    """
    يُغلّف ArabicTransformer الكامل بـ LoRA adapters.

    المزايا:
    • ArabicTransformer مُجمَّد تماماً — لا تعديل على ملفه
    • فقط A و B في كل adapter تتدرب (rank × d بدلاً من d × d)
    • حفظ الـ checkpoints في ثوانٍ (KB بدلاً من MB)
    • merge() يُدمج ΔW في الأوزان → inference بدون overhead

    مثال:
        base    = ArabicTransformer()
        lora    = LoRATransformerAdapter(base, rank=8, alpha=16.0)
        out     = lora.forward(tokens)
        loss    = cross_entropy(out, targets)
        lora.backward(d_loss, lr=1e-3)
        lora.save("models/lora/nsm_r8")
        lora.merge_into_base()             # inference سريع بعد الدمج
    """

    def __init__(
        self,
        transformer,
        rank:  int   = 8,
        alpha: float = 16.0,
    ):
        self.transformer = transformer
        self.rank        = rank
        self.alpha       = alpha

        # بناء adapter لكل TransformerBlock
        self.block_adapters: List[LoRABlockAdapter] = [
            LoRABlockAdapter(blk, rank, alpha)
            for blk in transformer.blocks
        ]

        n_adapters = len(self.block_adapters)
        total_lora = sum(b.param_count() for b in self.block_adapters)
        logger.info(
            f"[LoRATransformerAdapter] rank={rank} alpha={alpha} | "
            f"{n_adapters} blocks | {total_lora:,} adapter params"
        )

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, ids: np.ndarray, mask=None) -> np.ndarray:
        """
        ids: (seq,) token IDs
        returns: logits (seq, vocab_size)
        """
        t = self.transformer

        # Embedding + Positional (من base — لا LoRA هنا)
        X  = t.embedding.forward(ids)
        X += t.pos_enc.forward(len(ids))

        # CoreMatrix مع residual (من base — مجمّدة)
        if hasattr(t, 'core') and t.core is not None:
            X = X + t.core.forward(X)

        # Transformer blocks — مع LoRA
        S    = len(ids)
        if mask is None:
            mask = np.triu(np.ones((S, S), bool), k=1)
        for adapter in self.block_adapters:
            X = adapter.forward(X, mask)

        # Output head (من base)
        return t.head.forward(X)

    # ── backward ─────────────────────────────────────────────────────────

    def backward(self, d_logits: np.ndarray, lr: float = 1e-3) -> None:
        """
        d_logits: gradient من دالة الخسارة (seq, vocab_size)
        يُحدِّث فقط adapter weights A و B.
        base ArabicTransformer يبقى مجمَّداً.
        """
        t    = self.transformer
        grad = t.head.backward(d_logits, lr=0.0)   # head frozen أيضاً

        for adapter in reversed(self.block_adapters):
            grad = adapter.backward(grad, lr)

        # لا تحديث على CoreMatrix أو Embedding (frozen)

    # ── دالة الخسارة (cross-entropy للغة العربية) ────────────────────────

    def compute_loss(
        self,
        logits:  np.ndarray,   # (seq, vocab)
        targets: np.ndarray,   # (seq,) IDs
        ignore_id: int = 0,
    ) -> Tuple[float, np.ndarray]:
        """
        Cross-entropy مع تجاهل PAD (ID=0).
        يُعيد (loss_scalar, d_logits).
        """
        seq, V  = logits.shape
        d_logits = np.zeros_like(logits)
        total_loss = 0.0
        n_valid    = 0

        for i in range(seq):
            tid = int(targets[i])
            if tid == ignore_id:
                continue
            # softmax
            z   = logits[i] - logits[i].max()
            exp = np.exp(z)
            p   = exp / (exp.sum() + 1e-9)
            # loss
            total_loss -= math.log(p[tid] + 1e-9)
            # gradient
            d = p.copy()
            d[tid] -= 1.0
            d_logits[i] = d
            n_valid += 1

        loss = total_loss / max(n_valid, 1)
        d_logits /= max(n_valid, 1)
        return loss, d_logits

    # ── خطوة تدريب كاملة ─────────────────────────────────────────────────

    def train_step(
        self,
        ids:    np.ndarray,
        targets: np.ndarray,
        lr:     float = 1e-3,
    ) -> float:
        """
        خطوة تدريب واحدة: forward → loss → backward → update adapters.
        targets: (seq,) token IDs — عادةً ids[1:] + [EOS] لـ language modeling.
        """
        logits    = self.forward(ids)
        loss, dL  = self.compute_loss(logits, targets)
        self.backward(dL, lr)
        return loss

    # ── language modeling data prep ──────────────────────────────────────

    def prepare_lm_sample(
        self, text: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        يُحوّل نصاً عربياً إلى (inputs, targets) لـ language modeling.
        inputs  = tokens[:-1]
        targets = tokens[1:]
        """
        t    = self.transformer
        ids  = t.tokenizer.encode(text, t.max_seq)
        if len(ids) < 2:
            return None, None
        return ids[:-1], ids[1:]

    # ── دمج في الأوزان الأساسية ───────────────────────────────────────────

    def merge_into_base(self):
        """
        يُدمج جميع ΔW في أوزان ArabicTransformer.
        بعد الدمج: inference بدون أي overhead إضافي.
        """
        for adapter in self.block_adapters:
            adapter.merge_into_base()
        logger.info("[LoRA] ✅ تم دمج جميع الـ adapters في النموذج الأصلي")

    # ── حفظ / تحميل ──────────────────────────────────────────────────────

    def save(self, prefix: str):
        """يحفظ adapter weights فقط (KB بدلاً من MB)."""
        Path(prefix).parent.mkdir(parents=True, exist_ok=True)
        for i, adapter in enumerate(self.block_adapters):
            adapter.save(f"{prefix}_block{i}")
        # احفظ meta
        meta = {
            "rank":    self.rank,
            "alpha":   self.alpha,
            "n_blocks": len(self.block_adapters),
            "params":  self.param_count(),
        }
        import json
        with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        logger.info(
            f"[LoRA] ✅ حُفظت في {prefix}_block*.npy | "
            f"حجم تقريبي: {self.param_count()*8/1024:.1f} KB"
        )

    def load(self, prefix: str):
        for i, adapter in enumerate(self.block_adapters):
            adapter.load(f"{prefix}_block{i}")
        logger.info(f"[LoRA] ✅ تم تحميل الـ adapters من {prefix}")

    # ── إحصاءات ──────────────────────────────────────────────────────────

    def param_count(self) -> int:
        return sum(b.param_count() for b in self.block_adapters)

    def base_param_count(self) -> int:
        t  = self.transformer
        n  = 0
        if hasattr(t, 'embed'):  n += t.embed.W.size
        for blk in t.blocks:
            for W in [blk.mha.Wq, blk.mha.Wk, blk.mha.Wv, blk.mha.Wo]:
                n += W.size
            n += blk.ffn.W1.size + blk.ffn.W2.size
        return n

    def summary(self) -> str:
        lora  = self.param_count()
        base  = self.base_param_count()
        ratio = lora / max(base, 1) * 100
        lines = [
            "╔══════════════════════════════════════════════════╗",
            "║          LoRA Adapter — NSM v18.3                ║",
            "╠══════════════════════════════════════════════════╣",
            f"║  rank        : {self.rank:<34}║",
            f"║  alpha       : {self.alpha:<34}║",
            f"║  scale       : {self.alpha/self.rank:<34.4f}║",
            f"║  adapter params  : {lora:>10,}                  ║",
            f"║  base params     : {base:>10,}                  ║",
            f"║  نسبة التدريب   : {ratio:>9.2f}%                  ║",
            f"║  حجم checkpoint  : {lora*8/1024:>9.1f} KB                ║",
            "╠══════════════════════════════════════════════════╣",
            "║  التغطية: Wq + Wk + Wv + Wo + W1_ffn + W2_ffn   ║",
            "║  كل طبقة transformer → 6 LoRA layers             ║",
            "╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
# helpers (نسخ خفيف بدون import من arabic_transformer لتجنب circular import)
# ══════════════════════════════════════════════════════════════════════════════

def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)
