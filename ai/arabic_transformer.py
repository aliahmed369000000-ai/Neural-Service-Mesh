"""
Arabic Transformer — NSM v3.0
==============================
مبدأ التصميم الأساسي:
    النموذج لا يحفظ أي نص أو بيانات تدريب.
    كل ما يتعلمه يُخزَّن حصراً في الأوزان (Weights).
    البيانات تُمتَص → تُعدِّل الأوزان → تُرمى.

ما يُحفَظ على disk:
    ✓ أوزان الشبكة (.npy)        ← الذاكرة الحقيقية للنموذج
    ✓ المصفوفة المدروسة (.csv/.npy) ← بذرة ابتدائية قابلة للتدريب (لم تعد مجمَّدة)، قلب الشبكة
    ✗ لا نصوص، لا كلمات، لا قواعد بيانات

الـ Tokenizer:
    يعمل بـ character n-grams + hash trick.
    لا يحفظ أي كلمة — فقط دالة تحويل رياضية.
    الجدول الوحيد المحفوظ: embedding matrix (أوزان).
"""
from __future__ import annotations

import logging
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════
D_MODEL      = 2304
N_HEADS      = 16
D_FF         = 8384
N_LAYERS     = 16
MAX_SEQ_LEN  = 128
VOCAB_SIZE   = 8192     # حجم ثابت — hash space، لا يتوسع
LEARNING_RATE = 1e-4
CLIP_GRAD    = 1.0
WEIGHTS_DIR  = "models/transformer"

# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════
def _xavier(r, c): 
    l = math.sqrt(6.0 / (r + c))
    return np.random.uniform(-l, l, (r, c)).astype(np.float64)

def _relu(x): return np.maximum(0.0, x)

def _softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / (e.sum(axis=-1, keepdims=True) + 1e-9)

def _layer_norm_fwd(x, g, b, eps=1e-6):
    m = x.mean(-1, keepdims=True)
    v = x.var(-1, keepdims=True)
    return g * (x - m) / np.sqrt(v + eps) + b

# ══════════════════════════════════════════════════════════════════════════════
# 1. Hash Tokenizer — لا يحفظ أي نص
# ══════════════════════════════════════════════════════════════════════════════
class HashTokenizer:
    """
    Tokenizer يعمل بالـ hashing فقط.

    لا يوجد vocab، لا word2id، لا أي نص محفوظ.
    كل كلمة → hash رقمي → ID في النطاق [6, VOCAB_SIZE).
    
    الرموز الخاصة (ثوابت لا تتغير):
        PAD=0, UNK=1, BOS=2, EOS=3, SEP=4, MASK=5
    
    النتيجة: النموذج يتعلم تمثيل الكلمات عبر الأوزان،
    وليس عبر جدول نصي محفوظ.
    """
    PAD, UNK, BOS, EOS, SEP, MASK = 0, 1, 2, 3, 4, 5
    OFFSET = 6  # أول ID متاح للكلمات

    def __init__(self, vocab_size: int = VOCAB_SIZE):
        self.vocab_size = vocab_size

    def _hash_word(self, word: str) -> int:
        """كلمة → ID ثابت بالـ FNV-1a hash."""
        h = 2166136261
        for ch in word.encode("utf-8"):
            h ^= ch
            h = (h * 16777619) & 0xFFFFFFFF
        return self.OFFSET + (h % (self.vocab_size - self.OFFSET))

    def _normalize(self, text: str) -> str:
        import re
        text = re.sub(r'[\u064B-\u065F\u0670]', '', text)   # إزالة التشكيل
        text = text.replace('أ','ا').replace('إ','ا').replace('آ','ا')
        text = text.replace('ى','ي').replace('ة','ه')
        return text

    def encode(self, text: str, max_len: int = MAX_SEQ_LEN) -> np.ndarray:
        """نص → array of IDs. لا يُخزَّن شيء."""
        import re
        words = re.findall(r'[\u0600-\u06FF]+|\d+', self._normalize(text))
        ids   = [self.BOS] + [self._hash_word(w) for w in words] + [self.EOS]
        ids   = ids[:max_len]
        return np.array(ids, dtype=np.int64)

    def vocab_id(self) -> int:
        return self.vocab_size

    # لا توجد دوال save/load — لا شيء للحفظ


# ══════════════════════════════════════════════════════════════════════════════
# 2. Token Embedding (الأوزان فقط)
# ══════════════════════════════════════════════════════════════════════════════
class TokenEmbedding:
    """
    جدول embeddings — هذا هو المكان الوحيد الذي يُعبِّر فيه النموذج
    عن "معنى" الكلمات. لا نصوص، فقط أوزان رقمية.
    """
    def __init__(self, vocab_size: int, d_model: int):
        self.W = np.random.randn(vocab_size, d_model).astype(np.float64) * 0.02
        self._last_ids = None

    def forward(self, ids: np.ndarray) -> np.ndarray:
        self._last_ids = np.clip(ids, 0, self.W.shape[0] - 1)
        return self.W[self._last_ids]

    def backward(self, grad: np.ndarray, lr: float):
        for i, idx in enumerate(self._last_ids):
            self.W[idx] -= lr * grad[i]
        np.clip(self.W, -5.0, 5.0, out=self.W)

    def save(self, path: str): np.save(path, self.W)
    def load(self, path: str):
        self.W = np.load(path).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 3. Positional Encoding (ثابت رياضي، لا أوزان)
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding:
    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        pe  = np.zeros((max_len, d_model))
        pos = np.arange(max_len).reshape(-1, 1)
        div = np.power(10000.0, np.arange(0, d_model, 2) / d_model)
        pe[:, 0::2] = np.sin(pos / div)
        pe[:, 1::2] = np.cos(pos / div)
        self._table = pe.astype(np.float64)

    def forward(self, seq_len: int) -> np.ndarray:
        return self._table[:seq_len]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Core Matrix Layer — قلب الشبكة (784×784 ثابتة)
# ══════════════════════════════════════════════════════════════════════════════
class CoreMatrixLayer:
    """
    المصفوفة المدروسة 784×784 — قابلة للتدريب بالكامل (لم تعد مجمَّدة).

    كانت في السابق ثابتة تماماً (frozen) لا تتأثر بالتدريب. الآن تُحدَّث
    بالـ gradient الحقيقي في backward() مثل بقية الأوزان، لكن بمعدل تعلم
    أبطأ افتراضياً (core_lr_scale) حفاظاً على استقرارها كـ "anchor" دلالي
    تتعلم فيه باقي الطبقات دون أن تتذبذب بعنف من أول خطوة تدريب.

    • trainable_core=True (افتراضي)  → تتدرب فعلياً بكل خطوة backward.
    • core_lr_scale (افتراضي 0.1)    → نسبة معدل تعلمها إلى معدل باقي الطبقات؛
      اجعلها 1.0 لتدريبها بنفس السرعة، أو 0.0 لتجميدها يدوياً إن احتجت ذلك صراحة.

    المسار:
        X(seq,256) → W_up(256→784) → W_core(784→784)[قابلة للتدريب]
                   → sign_flip+relu → W_down(784→256) → out(seq,256)
    """
    def __init__(
        self,
        csv_path: Optional[str] = None,
        d_model: int = D_MODEL,
        trainable_core: bool = True,
        core_lr_scale: float = 0.1,
    ):
        self.d_model        = d_model
        self.core_dim        = 784
        self.trainable_core  = trainable_core
        self.core_lr_scale   = core_lr_scale
        self._W_core: Optional[np.ndarray] = None
        self._loaded  = False

        self.W_up   = _xavier(self.core_dim, d_model)
        self.W_down = _xavier(d_model, self.core_dim)
        self.b_up   = np.zeros(self.core_dim)
        self.b_down = np.zeros(d_model)

        self._cx = self._cup = self._cout = None  # cache

        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)

        if self._W_core is None:
            # لا مصفوفة محمَّلة من CSV/NPY — نبدأ ببذرة Xavier قابلة للتدريب
            # بدل مصفوفة الهوية الثابتة القديمة (كانت تمنع أي تعلّم فعلي).
            self._W_core = _xavier(self.core_dim, self.core_dim)

    def _load_csv(self, path: str) -> bool:
        try:
            W = np.genfromtxt(path, delimiter=',')
            if W.shape == (784, 784):
                self._W_core = W.astype(np.float64)
                self._loaded = True
                logger.info(f"[CoreMatrix] ✓ 784×784 محملة (بذرة قابلة للتدريب) | min={W.min():.3f} max={W.max():.3f}")
                return True
        except Exception as e:
            logger.error(f"[CoreMatrix] {e}")
        return False

    def load_array(self, W: np.ndarray):
        assert W.shape == (784, 784)
        self._W_core = W.astype(np.float64)
        self._loaded = True

    def _core(self) -> np.ndarray:
        return self._W_core

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._cx  = X
        up        = X @ self.W_up.T + self.b_up          # (seq,784)
        self._cup = up
        out       = up @ self._core().T                  # (seq,784)
        # sign-flip activation (من NSM)
        act       = _relu(out)
        mask      = np.abs(out) > 0.15
        act[mask] *= -0.5
        self._cout = act
        return act @ self.W_down.T + self.b_down          # (seq,256)

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        gWd   = grad.T @ self._cout
        gbd   = grad.sum(0)
        g_act = grad @ self.W_down                        # (seq,784)
        # relu grad (تقريبي عبر sign-flip)
        g_out = g_act * (self._cout > 0).astype(float)
        g_up  = g_out @ self._core()                      # (seq,784)
        gWu   = g_up.T @ self._cx
        gbu   = g_up.sum(0)
        gX    = g_up @ self.W_up

        if self.trainable_core and self.core_lr_scale > 0.0:
            # out = up @ core.T  ⇒  dL/dcore = g_out.T @ up
            gCore = g_out.T @ self._cup                   # (784,784)
            self._W_core -= (lr * self.core_lr_scale) * np.clip(gCore, -CLIP_GRAD, CLIP_GRAD)
            np.clip(self._W_core, -5.0, 5.0, out=self._W_core)

        for W, g in [(self.W_down, gWd), (self.W_up, gWu)]:
            W -= lr * np.clip(g, -CLIP_GRAD, CLIP_GRAD)
            np.clip(W, -5.0, 5.0, out=W)
        self.b_down -= lr * np.clip(gbd, -CLIP_GRAD, CLIP_GRAD)
        self.b_up   -= lr * np.clip(gbu, -CLIP_GRAD, CLIP_GRAD)
        return gX

    def info(self) -> dict:
        W = self._W_core
        return {
            "loaded": self._loaded,
            "shape":  [784, 784],
            "frozen": not self.trainable_core,
            "core_lr_scale": self.core_lr_scale,
            "stats":  {"min": round(float(W.min()),4),
                       "max": round(float(W.max()),4),
                       "mean":round(float(W.mean()),4)},
        }

    def save(self, prefix: str):
        np.save(f"{prefix}_Wu.npy",   self.W_up)
        np.save(f"{prefix}_Wd.npy",   self.W_down)
        np.save(f"{prefix}_bu.npy",   self.b_up)
        np.save(f"{prefix}_bd.npy",   self.b_down)
        np.save(f"{prefix}_core.npy", self._W_core)  # تُحفظ الآن لأنها تتغيّر بالتدريب

    def load(self, prefix: str):
        for attr, fname in [("W_up","Wu"),("W_down","Wd"),
                             ("b_up","bu"),("b_down","bd")]:
            p = f"{prefix}_{fname}.npy"
            if os.path.exists(p):
                setattr(self, attr, np.load(p).astype(np.float64))
        core_p = f"{prefix}_core.npy"
        if os.path.exists(core_p):
            self._W_core = np.load(core_p).astype(np.float64)
            self._loaded = True


# ══════════════════════════════════════════════════════════════════════════════
# 5. Multi-Head Self-Attention
# ══════════════════════════════════════════════════════════════════════════════
class MultiHeadAttention:
    def __init__(self, d_model: int = D_MODEL, n_heads: int = N_HEADS):
        assert d_model % n_heads == 0
        self.h  = n_heads
        self.dk = d_model // n_heads
        self.dm = d_model
        # Q, K, V, O — كلها أوزان
        self.Wq = _xavier(d_model, d_model)
        self.Wk = _xavier(d_model, d_model)
        self.Wv = _xavier(d_model, d_model)
        self.Wo = _xavier(d_model, d_model)
        self._X = self._Q = self._K = self._V = None
        self._attn = self._concat = None

    def forward(self, X: np.ndarray, mask=None) -> np.ndarray:
        self._X = X
        S = len(X)
        Q = X @ self.Wq.T                                # (S, dm)
        K = X @ self.Wk.T
        V = X @ self.Wv.T
        self._Q, self._K, self._V = Q, K, V

        # reshape → (h, S, dk)
        Qh = Q.reshape(S, self.h, self.dk).transpose(1,0,2)
        Kh = K.reshape(S, self.h, self.dk).transpose(1,0,2)
        Vh = V.reshape(S, self.h, self.dk).transpose(1,0,2)

        sc = Qh @ Kh.transpose(0,2,1) / math.sqrt(self.dk)  # (h,S,S)
        if mask is not None:
            sc = np.where(mask[None], -1e9, sc)
        at = _softmax(sc)                                 # (h,S,S)
        self._attn = at
        out = at @ Vh                                     # (h,S,dk)
        self._concat = out.transpose(1,0,2).reshape(S, self.dm)
        return self._concat @ self.Wo.T                  # (S, dm)

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        S = self._X.shape[0]
        gWo  = grad.T @ self._concat
        gcat = grad @ self.Wo
        self.Wo -= lr * np.clip(gWo, -CLIP_GRAD, CLIP_GRAD)
        np.clip(self.Wo, -5, 5, out=self.Wo)

        gh = gcat.reshape(S, self.h, self.dk).transpose(1,0,2)  # (h,S,dk)
        at = self._attn
        Vh = self._V.reshape(S,self.h,self.dk).transpose(1,0,2)
        Qh = self._Q.reshape(S,self.h,self.dk).transpose(1,0,2)
        Kh = self._K.reshape(S,self.h,self.dk).transpose(1,0,2)

        gV  = at.transpose(0,2,1) @ gh                  # (h,S,dk)
        gat = gh @ Vh.transpose(0,2,1)
        # softmax backward
        s   = at
        gsc = s * (gat - (gat * s).sum(-1, keepdims=True))
        gsc /= math.sqrt(self.dk)

        gQ = gsc @ Kh
        gK = gsc.transpose(0,2,1) @ Qh

        gQf = gQ.transpose(1,0,2).reshape(S, self.dm)
        gKf = gK.transpose(1,0,2).reshape(S, self.dm)
        gVf = gV.transpose(1,0,2).reshape(S, self.dm)

        for W, g in [(self.Wq, gQf.T @ self._X),
                     (self.Wk, gKf.T @ self._X),
                     (self.Wv, gVf.T @ self._X)]:
            W -= lr * np.clip(g, -CLIP_GRAD, CLIP_GRAD)
            np.clip(W, -5, 5, out=W)

        return (gQf @ self.Wq + gKf @ self.Wk + gVf @ self.Wv)

    def save(self, p):
        for n, W in [("q",self.Wq),("k",self.Wk),("v",self.Wv),("o",self.Wo)]:
            np.save(f"{p}_W{n}.npy", W)

    def load(self, p):
        for n, attr in [("q","Wq"),("k","Wk"),("v","Wv"),("o","Wo")]:
            f = f"{p}_W{n}.npy"
            if os.path.exists(f):
                setattr(self, attr, np.load(f).astype(np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# 6. Feed-Forward Network
# ══════════════════════════════════════════════════════════════════════════════
class FFN:
    def __init__(self, d_model: int = D_MODEL, d_ff: int = D_FF):
        self.W1 = _xavier(d_ff,    d_model)
        self.W2 = _xavier(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)
        self._X = self._h = None

    def forward(self, X: np.ndarray) -> np.ndarray:
        self._X = X
        self._h = _relu(X @ self.W1.T + self.b1)
        return self._h @ self.W2.T + self.b2

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        gW2 = grad.T @ self._h
        gb2 = grad.sum(0)
        gh  = grad @ self.W2 * (self._h > 0)
        gW1 = gh.T @ self._X
        gb1 = gh.sum(0)
        gX  = gh @ self.W1
        for W, g in [(self.W1,gW1),(self.W2,gW2)]:
            W -= lr * np.clip(g, -CLIP_GRAD, CLIP_GRAD)
            np.clip(W, -5, 5, out=W)
        self.b1 -= lr * np.clip(gb1, -CLIP_GRAD, CLIP_GRAD)
        self.b2 -= lr * np.clip(gb2, -CLIP_GRAD, CLIP_GRAD)
        return gX

    def save(self, p):
        for n, a in [("W1",self.W1),("W2",self.W2),("b1",self.b1),("b2",self.b2)]:
            np.save(f"{p}_{n}.npy", a)

    def load(self, p):
        for n, attr in [("W1","W1"),("W2","W2"),("b1","b1"),("b2","b2")]:
            f = f"{p}_{n}.npy"
            if os.path.exists(f): setattr(self, attr, np.load(f).astype(np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# 7. Layer Norm
# ══════════════════════════════════════════════════════════════════════════════
class LayerNorm:
    def __init__(self, d: int):
        self.g = np.ones(d); self.b = np.zeros(d)
        self._xn = self._std = None

    def forward(self, X):
        m = X.mean(-1, keepdims=True); v = X.var(-1, keepdims=True)
        self._std = np.sqrt(v + 1e-6); self._xn = (X - m) / self._std
        return self.g * self._xn + self.b

    def backward(self, grad, lr):
        dg = (grad * self._xn).sum(0)
        db = grad.sum(0)
        N  = self.g.shape[0]
        dxn = grad * self.g
        dX  = (dxn / self._std
               + 2 * ((dxn * self._xn / self._std**2).sum(-1, keepdims=True)) * self._xn / N
               + (-dxn / self._std).sum(-1, keepdims=True) / N)
        self.g -= lr * np.clip(dg, -CLIP_GRAD, CLIP_GRAD)
        self.b -= lr * np.clip(db, -CLIP_GRAD, CLIP_GRAD)
        return dX

    def save(self, p): np.save(f"{p}_g.npy", self.g); np.save(f"{p}_b.npy", self.b)
    def load(self, p):
        for a, n in [("g","g"),("b","b")]:
            f = f"{p}_{n}.npy"
            if os.path.exists(f): setattr(self, a, np.load(f).astype(np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# 8. Transformer Block
# ══════════════════════════════════════════════════════════════════════════════
class TransformerBlock:
    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF, bid=0):
        self.bid  = bid
        self.mha  = MultiHeadAttention(d_model, n_heads)
        self.ffn  = FFN(d_model, d_ff)
        self.ln1  = LayerNorm(d_model)
        self.ln2  = LayerNorm(d_model)
        self._X   = self._ao = None

    def forward(self, X, mask=None):
        self._X  = X
        ao       = self.mha.forward(self.ln1.forward(X), mask)
        self._ao = ao
        X2       = X + ao
        return X2 + self.ffn.forward(self.ln2.forward(X2))

    def backward(self, grad, lr):
        X2     = self._X + self._ao
        gffn   = self.ffn.backward(grad, lr)
        gX2    = grad + self.ln2.backward(gffn, lr)
        gmha   = self.mha.backward(gX2, lr)
        return gX2 + self.ln1.backward(gmha, lr)

    def save(self, p):
        self.mha.save(f"{p}_mha"); self.ffn.save(f"{p}_ffn")
        self.ln1.save(f"{p}_ln1"); self.ln2.save(f"{p}_ln2")

    def load(self, p):
        self.mha.load(f"{p}_mha"); self.ffn.load(f"{p}_ffn")
        self.ln1.load(f"{p}_ln1"); self.ln2.load(f"{p}_ln2")


# ══════════════════════════════════════════════════════════════════════════════
# 9. Output Head
# ══════════════════════════════════════════════════════════════════════════════
class OutputHead:
    def __init__(self, d_model: int, vocab_size: int):
        self.W = _xavier(vocab_size, d_model)
        self.b = np.zeros(vocab_size)
        self._X = self._p = None

    def forward(self, X):
        self._X = X
        self._p = _softmax(X @ self.W.T + self.b)
        return self._p

    def loss_grad(self, probs, targets):
        n    = len(targets)
        loss = -np.log(np.clip(probs[np.arange(n), targets], 1e-10, 1)).mean()
        g    = probs.copy(); g[np.arange(n), targets] -= 1; g /= n
        return loss, g

    def backward(self, grad, lr):
        gW = grad.T @ self._X; gb = grad.sum(0); gX = grad @ self.W
        self.W -= lr * np.clip(gW, -CLIP_GRAD, CLIP_GRAD); np.clip(self.W,-5,5,out=self.W)
        self.b -= lr * np.clip(gb, -CLIP_GRAD, CLIP_GRAD)
        return gX

    def save(self, p): np.save(f"{p}_W.npy", self.W); np.save(f"{p}_b.npy", self.b)
    def load(self, p):
        for a, n in [("W","W"),("b","b")]:
            f = f"{p}_{n}.npy"
            if os.path.exists(f): setattr(self, a, np.load(f).astype(np.float64))


# ══════════════════════════════════════════════════════════════════════════════
# 10. Arabic Transformer — النموذج الكامل
# ══════════════════════════════════════════════════════════════════════════════
class ArabicTransformer:
    """
    ما يتعلمه النموذج يُخزَّن في الأوزان فقط:
        embedding.npy        — تمثيل الكلمات
        core_matrix_*.npy    — projection حول المصفوفة المدروسة
        block_N_*.npy        — طبقات الانتباه والـ FFN
        output_head_*.npy    — التنبؤ بالكلمة التالية

    لا يوجد أي ملف نصي يحتوي على بيانات تدريب.
    """
    VERSION = "3.0.0-NSM"

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS, d_ff=D_FF,
                 n_layers=N_LAYERS, max_seq=MAX_SEQ_LEN,
                 vocab_size=VOCAB_SIZE, lr=LEARNING_RATE,
                 weights_dir=WEIGHTS_DIR, core_csv=None):

        self.lr          = lr
        self.max_seq     = max_seq
        self.weights_dir = weights_dir

        self.tokenizer   = HashTokenizer(vocab_size)
        self.embedding   = TokenEmbedding(vocab_size, d_model)
        self.pos_enc     = PositionalEncoding(d_model, max_seq)
        self.core        = CoreMatrixLayer(core_csv, d_model)
        self.blocks      = [TransformerBlock(d_model, n_heads, d_ff, i)
                            for i in range(n_layers)]
        self.head        = OutputHead(d_model, vocab_size)

        self._steps = 0
        self._loss_history: List[float] = []

    # ── forward ───────────────────────────────────────────────────────────────
    def _forward(self, ids: np.ndarray, mask=None):
        X  = self.embedding.forward(ids)
        X += self.pos_enc.forward(len(ids))
        # قلب الشبكة: المصفوفة المدروسة
        X  = X + self.core.forward(X)          # Residual
        for blk in self.blocks:
            X = blk.forward(X, mask)
        return self.head.forward(X), X

    # ── train (يمتص الأنماط → يُعدِّل الأوزان → يرمي النص) ──────────────────
    def train_step(self, text: str) -> float:
        """
        يأخذ النص → يُعدِّل الأوزان → لا يحفظ النص.
        البيانات تُستهلَك وترمى، الأوزان وحدها تبقى.
        """
        ids = self.tokenizer.encode(text, self.max_seq)
        if len(ids) < 2:
            return 0.0

        inp = ids[:-1]; tgt = ids[1:]
        S   = len(inp)
        mask = np.triu(np.ones((S, S), bool), k=1)

        probs, _ = self._forward(inp, mask)
        loss, gp = self.head.loss_grad(probs, tgt)

        # backward — يُعدِّل الأوزان
        gX = self.head.backward(gp, self.lr)
        for blk in reversed(self.blocks):
            gX = blk.backward(gX, self.lr)
        # Residual: grad → core + embedding
        gc = self.core.backward(gX, self.lr)
        self.embedding.backward(gX + gc, self.lr)

        self._steps += 1
        self._loss_history.append(loss)
        if len(self._loss_history) > 500:
            self._loss_history = self._loss_history[-250:]
        return float(loss)

    def train_batch(self, texts: List[str]) -> float:
        losses = [self.train_step(t) for t in texts if t.strip()]
        return float(np.mean(losses)) if losses else 0.0

    # ── inference ─────────────────────────────────────────────────────────────
    def encode(self, text: str) -> np.ndarray:
        """نص → متجه 256-dim (mean pooling). للاستخدام مع NSM routing."""
        ids = self.tokenizer.encode(text, self.max_seq)
        if len(ids) == 0:
            return np.zeros(self.embedding.W.shape[1])
        _, hidden = self._forward(ids)
        return hidden[1:-1].mean(0) if len(hidden) > 2 else hidden.mean(0)

    def predict_next(self, text: str, top_k=5, temp=1.0) -> List[Tuple[int, float]]:
        """يُعيد top_k من أزواج (hash_id, prob) — لا توجد كلمات محفوظة."""
        ids = self.tokenizer.encode(text, self.max_seq - 1)
        if not len(ids): return []
        S    = len(ids)
        mask = np.triu(np.ones((S,S), bool), k=1)
        p, _ = self._forward(ids, mask)
        lp   = p[-1]
        if temp != 1.0:
            lp = _softmax((np.log(np.clip(lp,1e-10,1)) / temp).reshape(1,-1)).flatten()
        top  = np.argsort(lp)[::-1][:top_k]
        return [(int(i), float(lp[i])) for i in top]

    def generate(self, text: str, max_new=20, temp=0.8) -> np.ndarray:
        """يُولِّد تسلسل IDs (hash). لا كلمات محفوظة."""
        ids = list(self.tokenizer.encode(text, self.max_seq - max_new))
        for _ in range(max_new):
            if len(ids) >= self.max_seq: break
            arr  = np.array(ids[-self.max_seq:], np.int64)
            S    = len(arr)
            mask = np.triu(np.ones((S,S), bool), k=1)
            p, _ = self._forward(arr, mask)
            lp   = p[-1]
            if temp != 1.0:
                lp = _softmax((np.log(np.clip(lp,1e-10,1))/temp).reshape(1,-1)).flatten()
            lp = np.clip(lp, 0, None); lp /= lp.sum()
            nxt = int(np.random.choice(len(lp), p=lp))
            if nxt == HashTokenizer.EOS: break
            ids.append(nxt)
        return np.array(ids, np.int64)

    # ── stats ─────────────────────────────────────────────────────────────────
    def stats(self) -> dict:
        avg = np.mean(self._loss_history) if self._loss_history else 0
        rec = np.mean(self._loss_history[-100:]) if len(self._loss_history) >= 100 else avg
        return {
            "version":      self.VERSION,
            "train_steps":  self._steps,
            "avg_loss":     round(float(avg), 5),
            "recent_loss":  round(float(rec), 5),
            "core_matrix":  self.core.info(),
            "storage":      "weights_only — no text stored",
        }

    # ── save / load (أوزان فقط) ───────────────────────────────────────────────
    def save(self, directory: Optional[str] = None) -> None:
        """يحفظ الأوزان فقط. لا نصوص، لا بيانات."""
        d = Path(directory or self.weights_dir)
        d.mkdir(parents=True, exist_ok=True)

        self.embedding.save(str(d / "embedding.npy"))
        self.core.save(str(d / "core_matrix"))
        self.head.save(str(d / "output_head"))
        for i, blk in enumerate(self.blocks):
            blk.save(str(d / f"block_{i}"))

        # meta: معلومات فنية فقط، لا بيانات تدريب
        import json
        meta = {
            "version": self.VERSION, "train_steps": self._steps,
            "storage_policy": "weights_only",
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        (d / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(f"[Transformer] ✓ حُفِظت الأوزان → {d}")

    def load(self, directory: Optional[str] = None) -> "ArabicTransformer":
        d = Path(directory or self.weights_dir)
        if not d.exists(): return self

        emb = d / "embedding.npy"
        if emb.exists(): self.embedding.load(str(emb))

        self.core.load(str(d / "core_matrix"))
        self.head.load(str(d / "output_head"))

        for i, blk in enumerate(self.blocks):
            blk.load(str(d / f"block_{i}"))

        meta_p = d / "meta.json"
        if meta_p.exists():
            import json
            self._steps = json.loads(meta_p.read_text())  .get("train_steps", 0)

        logger.info(f"[Transformer] ✓ الأوزان محملة ← {d}")
        return self


# ══════════════════════════════════════════════════════════════════════════════
# 11. NSM Bridge
# ══════════════════════════════════════════════════════════════════════════════
class NSMTransformerBridge:
    """
    جسر NSM ↔ ArabicTransformer.
    نص → متجه 256-dim → deep_routing_network.
    """
    def __init__(self, weights_dir=WEIGHTS_DIR, core_csv=None):
        if core_csv is None:
            c = os.path.join(weights_dir, "weights_784x784.csv")
            core_csv = c if os.path.exists(c) else None
        self.model = ArabicTransformer(
            weights_dir=weights_dir, core_csv=core_csv
        )

    def absorb(self, texts: List[str], epochs=1, log_every=500) -> dict:
        """
        يمتص النصوص ويُعدِّل الأوزان.
        النصوص لا تُحفَظ — فقط تأثيرها على الأوزان يبقى.
        """
        for ep in range(epochs):
            np.random.shuffle(texts)
            losses = []
            for i, t in enumerate(texts):
                if t.strip():
                    losses.append(self.model.train_step(t))
                if (i+1) % log_every == 0:
                    logger.info(f"[Bridge] ep={ep+1} step={i+1} loss={np.mean(losses[-log_every:]):.4f}")
        return self.model.stats()

    def text_to_nsm_vector(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def save(self): self.model.save()
    def load(self): self.model.load(); return self


# ══════════════════════════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════════════════════════
def get_transformer(weights_dir=WEIGHTS_DIR, core_csv=None,
                    load_if_exists=True) -> ArabicTransformer:
    m = ArabicTransformer(weights_dir=weights_dir, core_csv=core_csv)
    if load_if_exists and (Path(weights_dir) / "meta.json").exists():
        m.load(weights_dir)
    return m

def get_nsm_bridge(weights_dir=WEIGHTS_DIR, core_csv=None,
                   load_if_exists=True) -> NSMTransformerBridge:
    b = NSMTransformerBridge(weights_dir=weights_dir, core_csv=core_csv)
    if load_if_exists and (Path(weights_dir) / "meta.json").exists():
        b.load()
    return b


# ══════════════════════════════════════════════════════════════════════════════
# 12. PyTorch Decoder-Only Path — Yemeni Generative LLM
# ══════════════════════════════════════════════════════════════════════════════
"""
هذا القسم يُضيف مسار توليدي كامل (decoder-only) فوق النظام الحالي بدون
تعديل أي شيء في الـ NumPy path.

التصميم:
  • Grouped-Query Attention (GQA) — n_kv_heads أقل من n_heads لتوفير ذاكرة KV
  • RMSNorm بدل LayerNorm (أسرع على CPU)
  • قناع سببي ديناميكي يُبنى تلقائياً من طول التسلسل
  • float32 بدل float64 (أسرع على CPU بمرتين عملياً)
  • التوافق الكامل مع YemeniTokenizer (vocab_size مرن)

الفصل المعماري:
  NumPy path  (ArabicTransformer)     — قائم / مدرَّب / يُستخدم للـ routing
  PyTorch path (YemeniDecoder)        — جديد / للتوليد النصي الحر
"""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False
    logger.warning(
        "[YemeniDecoder] PyTorch غير متاح — المسار التوليدي معطّل. "
        "ثبّت الحزمة بـ: pip install torch"
    )


def _require_torch(fn):
    """ديكوراتور: يرفع ImportError واضحة إن استُدعيت دالة تحتاج torch."""
    def wrapper(*args, **kwargs):
        if not _TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch مطلوب لهذه الوظيفة. ثبّته بـ: pip install torch"
            )
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


# ────────────────────────────────────────────────────────────────────────────
# 12-A. RMSNorm
# ────────────────────────────────────────────────────────────────────────────

class YemeniRMSNorm(nn.Module if _TORCH_AVAILABLE else object):
    """
    Root Mean Square Layer Normalisation.
    أسرع من LayerNorm (لا طرح للمتوسط) — نفس الاستقرار.
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):          # x: (..., d_model)
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        return self.weight * (x / rms)


# ────────────────────────────────────────────────────────────────────────────
# 12-B. Grouped-Query Attention (GQA)
# ────────────────────────────────────────────────────────────────────────────

class YemeniGQAAttention(nn.Module if _TORCH_AVAILABLE else object):
    """
    Grouped-Query Attention — يقلّل عدد رؤوس K/V مع الحفاظ على رؤوس Q الكاملة.

    مثال: n_heads=16, n_kv_heads=4
      → 4 مجموعات، كل مجموعة تشارك نفس K/V عبر 4 رؤوس Q.
      → يوفّر 75% من ذاكرة KV cache مقارنةً بـ MHA كاملة.

    يدعم:
      • قناع سببي ديناميكي (يُبنى تلقائياً من seq_len)
      • قناع PAD خارجي (key_padding_mask: bool tensor, True = PAD)
      • float32 لكفاءة CPU
    """

    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int,
        dropout:    float = 0.0,
    ):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        assert d_model % n_heads == 0, \
            f"d_model({d_model}) يجب أن يقبل القسمة على n_heads({n_heads})"
        assert n_heads % n_kv_heads == 0, \
            f"n_heads({n_heads}) يجب أن يقبل القسمة على n_kv_heads({n_kv_heads})"

        self.n_heads    = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups   = n_heads // n_kv_heads   # عدد رؤوس Q لكل مجموعة KV
        self.d_head     = d_model // n_heads
        self.d_kv       = self.d_head * n_kv_heads
        self.scale      = self.d_head ** -0.5

        self.Wq  = nn.Linear(d_model, d_model,      bias=False)
        self.Wk  = nn.Linear(d_model, self.d_kv,    bias=False)
        self.Wv  = nn.Linear(d_model, self.d_kv,    bias=False)
        self.Wo  = nn.Linear(d_model, d_model,      bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # تهيئة Xavier
        for layer in [self.Wq, self.Wk, self.Wv, self.Wo]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(
        self,
        x:                  "torch.Tensor",   # (B, S, d_model)
        causal_mask:        Optional["torch.Tensor"] = None,  # (S, S) bool
        key_padding_mask:   Optional["torch.Tensor"] = None,  # (B, S) bool — True=PAD
    ) -> "torch.Tensor":
        B, S, _ = x.shape

        # ── Projections ──────────────────────────────────────────────────
        Q = self.Wq(x)                        # (B, S, d_model)
        K = self.Wk(x)                        # (B, S, d_kv)
        V = self.Wv(x)                        # (B, S, d_kv)

        # ── Reshape → heads ──────────────────────────────────────────────
        Q = Q.view(B, S, self.n_heads,    self.d_head).transpose(1, 2)   # (B, H, S, dh)
        K = K.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)   # (B, Hkv, S, dh)
        V = V.view(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)   # (B, Hkv, S, dh)

        # ── GQA: توسيع K/V ليطابق n_heads ──────────────────────────────
        # كل مجموعة KV تُكرَّر n_groups مرات لتطابق رؤوس Q
        K = K.repeat_interleave(self.n_groups, dim=1)   # (B, H, S, dh)
        V = V.repeat_interleave(self.n_groups, dim=1)   # (B, H, S, dh)

        # ── Scaled Dot-Product Attention ─────────────────────────────────
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (B, H, S, S)

        # قناع سببي
        if causal_mask is not None:
            # causal_mask: (S, S) bool — True = يُحجَب
            scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        # قناع PAD
        if key_padding_mask is not None:
            # key_padding_mask: (B, S) bool — True = PAD
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn   = F.softmax(scores, dim=-1)       # (B, H, S, S)
        attn   = self.drop(attn)
        out    = torch.matmul(attn, V)            # (B, H, S, dh)

        # ── Merge heads ──────────────────────────────────────────────────
        out = out.transpose(1, 2).contiguous().view(B, S, -1)   # (B, S, d_model)
        return self.Wo(out)


# ────────────────────────────────────────────────────────────────────────────
# 12-C. Feed-Forward Network (SwiGLU variant — موثوق على CPU)
# ────────────────────────────────────────────────────────────────────────────

class YemeniFFN(nn.Module if _TORCH_AVAILABLE else object):
    """
    FFN بنمط SwiGLU:
        FFN(x) = SiLU(xW1) ⊙ (xW2)·W3
    توازن جيد بين الأداء والسرعة على CPU.
    d_ff الافتراضي = 4/3 × d_model × 2 (تقريب LLaMA: 8/3 × d_model).
    """
    def __init__(self, d_model: int, d_ff: Optional[int] = None):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        # 8/3 × d_model مقرّباً لأقرب 256
        d_ff = d_ff or int(((d_model * 8 // 3) + 255) // 256 * 256)
        self.W1 = nn.Linear(d_model, d_ff, bias=False)
        self.W2 = nn.Linear(d_model, d_ff, bias=False)
        self.W3 = nn.Linear(d_ff,   d_model, bias=False)
        for layer in [self.W1, self.W2, self.W3]:
            nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        return self.W3(F.silu(self.W1(x)) * self.W2(x))


# ────────────────────────────────────────────────────────────────────────────
# 12-D. Decoder Block
# ────────────────────────────────────────────────────────────────────────────

class YemeniDecoderBlock(nn.Module if _TORCH_AVAILABLE else object):
    """
    طبقة decoder-only واحدة:
      Pre-Norm → GQA → Residual → Pre-Norm → FFN → Residual
    """
    def __init__(
        self,
        d_model:    int,
        n_heads:    int,
        n_kv_heads: int,
        d_ff:       Optional[int] = None,
        dropout:    float = 0.0,
    ):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.norm1 = YemeniRMSNorm(d_model)
        self.attn  = YemeniGQAAttention(d_model, n_heads, n_kv_heads, dropout)
        self.norm2 = YemeniRMSNorm(d_model)
        self.ffn   = YemeniFFN(d_model, d_ff)
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(
        self,
        x:                "torch.Tensor",
        causal_mask:      Optional["torch.Tensor"] = None,
        key_padding_mask: Optional["torch.Tensor"] = None,
    ) -> "torch.Tensor":
        # Self-attention + residual
        x = x + self.drop(self.attn(self.norm1(x), causal_mask, key_padding_mask))
        # FFN + residual
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ────────────────────────────────────────────────────────────────────────────
# 12-E. YemeniDecoder — Full Decoder-Only Model
# ────────────────────────────────────────────────────────────────────────────

class YemeniDecoder(nn.Module if _TORCH_AVAILABLE else object):
    """
    نموذج توليدي decoder-only مُحسَّن لـ CPU بالعربية اليمنية.

    البنية الافتراضية (مطابقة لأبعاد ArabicTransformer الموجودة):
      d_model    = 2304
      n_heads    = 16
      n_kv_heads = 4      ← GQA: 4 KV groups لـ 16 query heads (توفير 75% ذاكرة KV)
      n_layers   = 16
      vocab_size = 32000   ← حجم YemeniTokenizer بعد نموه

    المسار:
      tokens → Embedding → (RMSNorm + GQA + FFN) × N → RMSNorm → LM Head → logits

    التكامل مع ReasoningPipeline:
      يُستخدم عند generation_mode=True فقط.
      يستقبل grounding_context (نص من CKG/SQLite) كـ prefix tokens.
    """

    VERSION = "1.0.0-Yemeni"

    def __init__(
        self,
        vocab_size:  int   = 32000,
        d_model:     int   = D_MODEL,        # 2304 — مطابق لـ ArabicTransformer
        n_heads:     int   = N_HEADS,         # 16
        n_kv_heads:  int   = 4,               # GQA: 4 مجموعات KV
        n_layers:    int   = N_LAYERS,        # 16
        d_ff:        Optional[int] = None,    # None → 8/3 × d_model
        max_seq_len: int   = MAX_SEQ_LEN,     # 128
        dropout:     float = 0.0,
        pad_id:      int   = 0,
        weights_dir: str   = "models/yemeni_decoder",
    ):
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch مطلوب. ثبّته بـ: pip install torch")
        super().__init__()

        self.vocab_size  = vocab_size
        self.d_model     = d_model
        self.n_heads     = n_heads
        self.n_kv_heads  = n_kv_heads
        self.n_layers    = n_layers
        self.max_seq_len = max_seq_len
        self.pad_id      = pad_id
        self.weights_dir = weights_dir

        # ── Embedding ────────────────────────────────────────────────────
        self.embed     = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.embed_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ── Sinusoidal positional encoding (ثابت، غير متعلَّم) ──────────
        self.register_buffer(
            "pos_enc",
            self._build_sinusoidal(max_seq_len, d_model),
            persistent=False,
        )

        # ── Decoder blocks ───────────────────────────────────────────────
        self.layers = nn.ModuleList([
            YemeniDecoderBlock(d_model, n_heads, n_kv_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # ── Output ───────────────────────────────────────────────────────
        self.norm_out = YemeniRMSNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: embedding ↔ lm_head (يقلّل الأوزان ويُحسّن التقارب)
        self.lm_head.weight = self.embed.weight

        # ── Training state ───────────────────────────────────────────────
        self._steps: int = 0
        self._loss_history: List[float] = []

        n_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[YemeniDecoder] ✓ جاهز | "
            f"vocab={vocab_size} d_model={d_model} "
            f"n_heads={n_heads} n_kv={n_kv_heads} layers={n_layers} | "
            f"params={n_params:,}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _build_sinusoidal(max_len: int, d_model: int) -> "torch.Tensor":
        """Sinusoidal positional encoding — نفس المعادلة الأصلية (Vaswani 2017)."""
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe  # (max_len, d_model)

    def _causal_mask(self, seq_len: int, device: "torch.device") -> "torch.Tensor":
        """
        قناع سببي ديناميكي — يُبنى من طول التسلسل الفعلي.
        True = يُحجَب (المستقبل).
        """
        return torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )

    def _pad_mask(
        self, ids: "torch.Tensor"
    ) -> Optional["torch.Tensor"]:
        """قناع PAD — True حيث يوجد PAD token."""
        mask = ids == self.pad_id
        return mask if mask.any() else None

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(
        self,
        ids:              "torch.Tensor",              # (B, S) int64
        key_padding_mask: Optional["torch.Tensor"] = None,  # (B, S) bool
    ) -> "torch.Tensor":                               # (B, S, vocab_size)
        """
        Forward pass كامل.

        Parameters
        ----------
        ids              : (B, S) token IDs — int64.
        key_padding_mask : (B, S) bool — True = PAD (يُحسب تلقائياً إن لم يُمرَّر).

        Returns
        -------
        logits: (B, S, vocab_size) — float32.
        """
        B, S = ids.shape
        device = ids.device

        # Embedding + Positional
        x = self.embed(ids)                             # (B, S, d_model)
        x = self.embed_drop(x + self.pos_enc[:S])       # (B, S, d_model)

        # Causal mask (ديناميكي)
        causal = self._causal_mask(S, device)           # (S, S)

        # PAD mask (تلقائي إن لم يُمرَّر)
        if key_padding_mask is None:
            key_padding_mask = self._pad_mask(ids)

        # Decoder blocks
        for layer in self.layers:
            x = layer(x, causal_mask=causal, key_padding_mask=key_padding_mask)

        # Output
        x = self.norm_out(x)                            # (B, S, d_model)
        return self.lm_head(x)                          # (B, S, vocab_size)

    # ── Training step ─────────────────────────────────────────────────────

    def train_step(
        self,
        ids:    "torch.Tensor",   # (B, S) int64
        lr:     float = 1e-4,
        optimizer: Optional[object] = None,
    ) -> float:
        """
        خطوة تدريب language-model واحدة (next-token prediction).

        إن لم يُمرَّر optimizer، يُستخدم AdamW داخلي مؤقت (مفيد للاختبار
        السريع؛ للتدريب الجاد استخدم optimizer خارجي).

        Parameters
        ----------
        ids       : (B, S) — النموذج يتنبأ بـ ids[1:] من ids[:-1].
        lr        : معدل التعلم (يُستخدم فقط إن لم يُمرَّر optimizer).
        optimizer : torch.optim.Optimizer اختياري.

        Returns
        -------
        float — قيمة الخسارة.
        """
        self.train()
        inp = ids[:, :-1]     # (B, S-1)
        tgt = ids[:, 1:]      # (B, S-1)

        logits = self.forward(inp)               # (B, S-1, vocab)
        loss   = F.cross_entropy(
            logits.reshape(-1, self.vocab_size),
            tgt.reshape(-1),
            ignore_index=self.pad_id,
        )

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()
        else:
            # AdamW مؤقت للاختبار السريع
            _opt = torch.optim.AdamW(self.parameters(), lr=lr)
            _opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            _opt.step()

        loss_val = float(loss.item())
        self._steps += 1
        self._loss_history.append(loss_val)
        if len(self._loss_history) > 500:
            self._loss_history = self._loss_history[-250:]
        return loss_val

    # ── Generation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids:         "torch.Tensor",    # (1, S_prompt) int64
        max_new_tokens:     int   = 50,
        temperature:        float = 0.8,
        top_k:              int   = 50,
        top_p:              float = 0.95,
        eos_id:             int   = 3,
        grounding_ids:      Optional["torch.Tensor"] = None,  # (1, S_ctx) prefix
    ) -> "torch.Tensor":
        """
        توليد autoregressive مع Top-K/Top-P sampling.

        Parameters
        ----------
        prompt_ids      : (1, S) token IDs — بداية التوليد.
        max_new_tokens  : أقصى عدد من الرموز الجديدة.
        temperature     : حرارة التوزيع (أعلى = أكثر إبداعاً).
        top_k           : يُبقي أعلى K احتمالاً فقط.
        top_p           : Nucleus sampling — أبقِ الرموز التي مجموعها ≤ p.
        eos_id          : ID رمز النهاية (EOS).
        grounding_ids   : سياق تأسيسي (من CKG/SQLite) يُضاف كـ prefix قبل prompt.

        Returns
        -------
        torch.Tensor int64 (1, S_total) — التسلسل الكامل (prompt + مولَّد).
        """
        self.eval()
        device = prompt_ids.device

        # دمج grounding context مع prompt إن وُجد
        if grounding_ids is not None:
            ids = torch.cat([grounding_ids, prompt_ids], dim=1)
        else:
            ids = prompt_ids.clone()

        for _ in range(max_new_tokens):
            # اقتطع إن تجاوز max_seq_len
            ids_in = ids[:, -self.max_seq_len:]
            logits = self.forward(ids_in)[:, -1, :]   # (1, vocab)

            # Temperature
            if temperature != 1.0:
                logits = logits / temperature

            # Top-K
            if top_k > 0:
                top_vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < top_vals[:, -1:], float("-inf"))

            # Top-P (Nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                remove    = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs  = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)   # (1, 1)
            ids = torch.cat([ids, next_id], dim=1)

            if int(next_id.item()) == eos_id:
                break

        return ids

    # ── Stats ─────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        avg = float(np.mean(self._loss_history)) if self._loss_history else 0.0
        rec = float(np.mean(self._loss_history[-100:])) if len(self._loss_history) >= 100 else avg
        n   = sum(p.numel() for p in self.parameters())
        return {
            "version":      self.VERSION,
            "train_steps":  self._steps,
            "avg_loss":     round(avg, 5),
            "recent_loss":  round(rec, 5),
            "total_params": n,
            "architecture": {
                "vocab_size":  self.vocab_size,
                "d_model":     self.d_model,
                "n_heads":     self.n_heads,
                "n_kv_heads":  self.n_kv_heads,
                "gqa_groups":  self.n_heads // self.n_kv_heads,
                "n_layers":    self.n_layers,
                "max_seq_len": self.max_seq_len,
            },
        }

    # ── Save / Load ───────────────────────────────────────────────────────

    def save(self, directory: Optional[str] = None) -> str:
        """يحفظ الأوزان + meta إلى directory."""
        if not _TORCH_AVAILABLE:
            raise ImportError("PyTorch مطلوب للحفظ.")
        d = Path(directory or self.weights_dir)
        d.mkdir(parents=True, exist_ok=True)
        ckpt = d / "yemeni_decoder.pt"
        torch.save({
            "model_state_dict": self.state_dict(),
            "train_steps":      self._steps,
            "loss_history":     self._loss_history[-100:],
            "version":          self.VERSION,
            "arch": {
                "vocab_size":  self.vocab_size,
                "d_model":     self.d_model,
                "n_heads":     self.n_heads,
                "n_kv_heads":  self.n_kv_heads,
                "n_layers":    self.n_layers,
                "max_seq_len": self.max_seq_len,
                "pad_id":      self.pad_id,
            },
        }, ckpt)
        logger.info(f"[YemeniDecoder] ✓ حُفِظ → {ckpt}")
        return str(ckpt)

    def load(self, directory: Optional[str] = None) -> "YemeniDecoder":
        """يُحمِّل الأوزان من directory (يُتجاهل إن لم يوجد الملف)."""
        if not _TORCH_AVAILABLE:
            return self
        d    = Path(directory or self.weights_dir)
        ckpt = d / "yemeni_decoder.pt"
        if not ckpt.exists():
            logger.info(f"[YemeniDecoder] لا يوجد checkpoint في {d} — يعمل بأوزان عشوائية")
            return self
        data = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.load_state_dict(data["model_state_dict"])
        self._steps        = data.get("train_steps", 0)
        self._loss_history = data.get("loss_history", [])
        logger.info(f"[YemeniDecoder] ✓ محمَّل ← {ckpt} (steps={self._steps})")
        return self


# ══════════════════════════════════════════════════════════════════════════════
# Factory — YemeniDecoder
# ══════════════════════════════════════════════════════════════════════════════

@_require_torch
def get_yemeni_decoder(
    vocab_size:   int  = 32000,
    d_model:      int  = D_MODEL,
    n_heads:      int  = N_HEADS,
    n_kv_heads:   int  = 4,
    n_layers:     int  = N_LAYERS,
    weights_dir:  str  = "models/yemeni_decoder",
    load_if_exists: bool = True,
) -> "YemeniDecoder":
    """
    يُعيد مثيل YemeniDecoder:
    • يُحمِّل الأوزان المحفوظة إن وُجدت.
    • يبدأ بأوزان عشوائية إن لم توجد.
    """
    model = YemeniDecoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        n_layers=n_layers,
        weights_dir=weights_dir,
    )
    if load_if_exists:
        model.load(weights_dir)
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Quick Test
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("=" * 60)
    print("Arabic Transformer v3 — Weights-Only Storage")
    print("=" * 60)

    CSV = "/mnt/user-data/uploads/weights_784x784.csv"
    if not os.path.exists(CSV):
        CSV = None
        print("⚠ المصفوفة غير موجودة، تعمل بـ Identity")

    model = ArabicTransformer(
        n_layers=2, d_ff=256, vocab_size=VOCAB_SIZE, core_csv=CSV
    )

    print(f"\n✓ CoreMatrix: {model.core.info()}")
    print(f"✓ لا يوجد vocab محفوظ — HashTokenizer فقط")

    verses = [
        "بسم الله الرحمن الرحيم",
        "الحمد لله رب العالمين",
        "الرحمن الرحيم",
        "مالك يوم الدين",
        "اياك نعبد واياك نستعين",
        "اهدنا الصراط المستقيم",
        "قل هو الله احد",
        "الله الصمد",
        "لم يلد ولم يولد",
        "ولم يكن له كفوا احد",
    ]

    print("\n── تدريب 5 epochs (النصوص تُمتَص فقط) ──")
    for ep in range(5):
        losses = [model.train_step(v) for v in verses]
        print(f"  epoch {ep+1}: loss={np.mean(losses):.4f}")

    vec = model.encode("بسم الله الرحمن الرحيم")
    print(f"\n✓ encode → shape={vec.shape}, norm={np.linalg.norm(vec):.3f}")

    s = model.stats()
    print(f"\n✓ Stats:")
    print(f"  steps   = {s['train_steps']}")
    print(f"  loss    = {s['recent_loss']}")
    print(f"  storage = {s['storage']}")

    # تأكد: لا يوجد أي نص محفوظ
    assert not hasattr(model.tokenizer, 'word2id'), "خطأ: word2id موجود!"
    assert not hasattr(model.tokenizer, '_freq'),   "خطأ: _freq موجود!"
    print("\n✓ تأكيد: لا يوجد نص أو vocab محفوظ في الذاكرة")
    print("✓ الأوزان وحدها تحمل المعرفة")
