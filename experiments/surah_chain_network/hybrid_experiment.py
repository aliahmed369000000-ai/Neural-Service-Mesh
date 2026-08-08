"""
Surah-Chain Network — معمارية نموذج لغوي (LLM-style) مع وسط سوري

البنية الكاملة:
  [طبقة إدخال بأسلوب LLM]
      Token Embedding (d_model) + ترميز موضع + LayerNorm
      → Adapter W_in: d_model → 7
  [114 طبقة وسطى من شبكهه_114-1.xlsx]  SurahChainNetwork
      أبعاد كل طبقة = آيات سورة → السورة التالية
  [طبقة إخراج بأسلوب LLM]
      Adapter W_out: 7 → d_model
      → LM Head مربوط (weight tying): logits = h @ E.T

هذا يستبدل W_down/W_up العشوائية الصغيرة السابقة بطرفَي نموذج لغوي
حقيقيين (embedding + tied head)، مع الإبقاء على سلسلة السور في الوسط.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# جذر المستودع: experiments/surah_chain_network/ → ../..
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai.arabic_transformer import HashTokenizer

_DIMS_PATH = _HERE / "surah_layer_dims.json"
if not _DIMS_PATH.exists():
    raise FileNotFoundError(
        f"missing {_DIMS_PATH} — generate from شبكهه_114-1.xlsx first"
    )
LAYER_DIMS: List[List[int]] = json.loads(_DIMS_PATH.read_text())
CHAIN_WIDTH = int(LAYER_DIMS[0][0])  # 7
VOCAB_SIZE = 8192
DEFAULT_D_MODEL = 512


class LayerNorm1D:
    """LayerNorm بسيط لتثبيت مقياس الإشارة."""

    def __init__(self, dim: int, eps: float = 1e-5):
        self.g = np.ones(dim, dtype=np.float64)
        self.b = np.zeros(dim, dtype=np.float64)
        self.eps = eps
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        self._cache = (xhat,)
        return xhat * self.g + self.b

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        (xhat,) = self._cache
        self.g -= lr * (grad * xhat).sum(axis=0)
        self.b -= lr * grad.sum(axis=0)
        return grad * self.g


class SurahChainLayer:
    """طبقة FC واحدة + LayerNorm + Residual (مع إسقاط عند اختلاف الأبعاد)."""

    def __init__(self, d_in: int, d_out: int, seed: Optional[int] = None):
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (d_in + d_out))
        self.W = rng.uniform(-limit, limit, (d_out, d_in)).astype(np.float64)
        self.b = np.zeros(d_out, dtype=np.float64)
        self.ln = LayerNorm1D(d_out)
        self._x = None
        self._pre_res = None
        self.has_shortcut_proj = d_in != d_out
        if self.has_shortcut_proj:
            lim_s = np.sqrt(6.0 / (d_in + d_out))
            self.W_shortcut = rng.uniform(-lim_s, lim_s, (d_out, d_in)).astype(np.float64)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        pre = x @ self.W.T + self.b
        act = np.tanh(pre)
        normed = self.ln.forward(act)
        shortcut = (x @ self.W_shortcut.T) if self.has_shortcut_proj else x
        self._pre_res = (pre, act)
        return normed + shortcut

    def backward(self, grad_out: np.ndarray, lr: float) -> np.ndarray:
        pre, act = self._pre_res
        if self.has_shortcut_proj:
            g_shortcut_x = grad_out @ self.W_shortcut
            gW_s = grad_out.T @ self._x
            np.clip(gW_s, -5, 5, out=gW_s)
            self.W_shortcut -= lr * gW_s
        else:
            g_shortcut_x = grad_out
        g_normed = self.ln.backward(grad_out, lr)
        d_pre = g_normed * (1.0 - act ** 2)
        gW = d_pre.T @ self._x
        gb = d_pre.sum(axis=0)
        gx_main = d_pre @ self.W
        np.clip(gW, -5, 5, out=gW)
        np.clip(gb, -5, 5, out=gb)
        self.W -= lr * gW
        self.b -= lr * gb
        return gx_main + g_shortcut_x


class SurahChainNetwork:
    """114 طبقة وسطى مخفية — أبعادها من ملف السور فقط."""

    def __init__(self, layer_dims: Optional[List[List[int]]] = None):
        dims = layer_dims or LAYER_DIMS
        self.layers = [
            SurahChainLayer(int(d_in), int(d_out), seed=i)
            for i, (d_in, d_out) in enumerate(dims)
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad, lr)
        return grad

    def param_count(self) -> int:
        n = 0
        for L in self.layers:
            n += L.W.size + L.b.size + L.ln.g.size + L.ln.b.size
            if L.has_shortcut_proj:
                n += L.W_shortcut.size
        return n


def _sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    pos = np.zeros((seq_len, d_model), dtype=np.float64)
    for i in range(seq_len):
        for k in range(0, d_model, 2):
            ang = i / (10000.0 ** (k / d_model))
            pos[i, k] = np.sin(ang)
            if k + 1 < d_model:
                pos[i, k + 1] = np.cos(ang)
    return pos


class HybridExperimentModel:
    """
    نموذج لغوي كامل: LLM Input → SurahChain → LLM Output (tied head).

    المعاملات:
      d_model   — بُعد التضمين/الرأس (افتراضي 512، بأسلوب LLM صغير)
      vocab_size
      lr
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        project_d_model: Optional[int] = None,  # توافق خلفي مع train_no_decay
        vocab_size: int = VOCAB_SIZE,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        if project_d_model is not None:
            d_model = project_d_model
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.lr = float(lr)
        self.tokenizer = HashTokenizer(vocab_size)
        self.chain = SurahChainNetwork(LAYER_DIMS)

        rng = np.random.default_rng(seed)
        # —— طبقة الإدخال (LLM-style) ——
        self.E = rng.normal(0.0, 0.02, (vocab_size, self.d_model)).astype(np.float64)
        self.ln_in = LayerNorm1D(self.d_model)
        lim = np.sqrt(6.0 / (self.d_model + CHAIN_WIDTH))
        self.W_in = rng.uniform(-lim, lim, (CHAIN_WIDTH, self.d_model)).astype(np.float64)

        # —— طبقة الإخراج (LLM-style, tied with E) ——
        self.W_out = rng.uniform(-lim, lim, (self.d_model, CHAIN_WIDTH)).astype(np.float64)
        self.b_out = np.zeros(self.d_model, dtype=np.float64)

        self._cache: dict = {}

    # ---- forward pieces ----
    def _encode_input(self, ids: np.ndarray) -> np.ndarray:
        """LLM input: embed + position + LN."""
        tok = self.E[ids]
        pos = _sinusoidal_positions(len(ids), self.d_model)
        x = self.ln_in.forward(tok + pos)
        self._cache["ids"] = ids
        self._cache["x_in"] = x
        return x

    def _to_chain(self, x: np.ndarray) -> np.ndarray:
        self._cache["x_for_adapter"] = x
        return x @ self.W_in.T

    def _from_chain(self, h7: np.ndarray) -> np.ndarray:
        self._cache["h7"] = h7
        return h7 @ self.W_out.T + self.b_out

    def _lm_logits(self, h: np.ndarray) -> np.ndarray:
        """Tied output head: logits = h @ E.T"""
        self._cache["h"] = h
        return h @ self.E.T

    def forward_logits(self, ids: np.ndarray) -> np.ndarray:
        x = self._encode_input(ids)
        h7 = self._to_chain(x)
        h7o = self.chain.forward(h7)
        h = self._from_chain(h7o)
        return self._lm_logits(h)

    def train_step(self, text: str, max_len: int = 16) -> Optional[float]:
        ids = self.tokenizer.encode(text, max_len)
        if len(ids) < 2:
            return None
        inp = np.asarray(ids[:-1], dtype=np.int64)
        tgt = np.asarray(ids[1:], dtype=np.int64)

        logits = self.forward_logits(inp)
        z = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=-1, keepdims=True)

        n = len(tgt)
        loss = -np.log(np.clip(probs[np.arange(n), tgt], 1e-10, 1.0)).mean()

        g = probs.copy()
        g[np.arange(n), tgt] -= 1.0
        g /= n

        # backprop LM head (tied E)
        h = self._cache["h"]
        g_h = g @ self.E
        g_E = g.T @ h
        np.clip(g_E, -1.0, 1.0, out=g_E)
        self.E -= self.lr * g_E

        # adapter out
        h7 = self._cache["h7"]
        g_W_out = g_h.T @ h7
        g_b = g_h.sum(axis=0)
        np.clip(g_W_out, -5.0, 5.0, out=g_W_out)
        self.W_out -= self.lr * g_W_out
        self.b_out -= self.lr * g_b
        g7o = g_h @ self.W_out

        # surah chain
        g7 = self.chain.backward(g7o, self.lr)

        # adapter in
        x = self._cache["x_for_adapter"]
        g_W_in = g7.T @ x
        np.clip(g_W_in, -5.0, 5.0, out=g_W_in)
        self.W_in -= self.lr * g_W_in
        g_x = g7 @ self.W_in

        # LN + token embed
        g_x = self.ln_in.backward(g_x, self.lr)
        for i, tid in enumerate(self._cache["ids"]):
            self.E[tid] -= self.lr * np.clip(g_x[i], -1.0, 1.0)

        return float(loss)

    def param_count(self) -> dict:
        return {
            "chain": self.chain.param_count(),
            "embedding_E": int(self.E.size),
            "adapters": int(self.W_in.size + self.W_out.size + self.b_out.size),
            "d_model": self.d_model,
            "chain_width": CHAIN_WIDTH,
            "n_chain_layers": len(self.chain.layers),
        }


# توافق الاسم القديم
SurahChainLM = HybridExperimentModel


if __name__ == "__main__":
    from hybrid_data import SENTENCES

    print("SurahChain LM — فحص سريع")
    m = HybridExperimentModel(d_model=256, lr=2e-3)
    print("params:", m.param_count())
    losses = []
    t0 = time.time()
    for t in SENTENCES[:20]:
        loss = m.train_step(t)
        if loss is not None:
            losses.append(loss)
    print(f"{len(losses)} خطوات في {time.time()-t0:.1f}s")
    if losses:
        print("أول/آخر loss:", round(losses[0], 3), "→", round(losses[-1], 3))
