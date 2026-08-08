"""
Surah-Chain LM — نموذج لغوي بوسط سوري + انتباه ذاتي قوي

البنية:
  Embedding + موضع + LN
  → TransformerBlock × N_PRE  (Multi-Head Causal Attention + FFN)
  → Adapter → SurahChain×114 → Adapter
  → Residual bypass (W_skip)
  → TransformerBlock × N_POST
  → Tied LM Head

قدرات: WordTokenizer, train_batch, cosine LR, generate.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from ai.arabic_transformer import HashTokenizer, WordTokenizer

_DIMS_PATH = _HERE / "surah_layer_dims.json"
if not _DIMS_PATH.exists():
    raise FileNotFoundError(f"missing {_DIMS_PATH}")
LAYER_DIMS: List[List[int]] = json.loads(_DIMS_PATH.read_text())
CHAIN_WIDTH = int(LAYER_DIMS[0][0])
VOCAB_SIZE = 8192
DEFAULT_D_MODEL = 512
DEFAULT_N_HEADS = 8
DEFAULT_N_PRE = 2
DEFAULT_N_POST = 2
DEFAULT_D_FF_MULT = 4


class LayerNorm1D:
    def __init__(self, dim: int, eps: float = 1e-5):
        self.g = np.ones(dim, dtype=np.float64)
        self.b = np.zeros(dim, dtype=np.float64)
        self.eps = eps
        self._cache = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        xhat = (x - mu) / np.sqrt(var + self.eps)
        self._cache = (xhat, mu, var)
        return xhat * self.g + self.b

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        xhat, mu, var = self._cache
        self.g -= lr * (grad * xhat).sum(axis=0)
        self.b -= lr * grad.sum(axis=0)
        # تقريب شائع: تمرير التدرّج عبر الغيت فقط
        return grad * self.g


class SurahChainLayer:
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
    def __init__(self, layer_dims: Optional[List[List[int]]] = None):
        dims = layer_dims or LAYER_DIMS
        self.layers = [
            SurahChainLayer(int(a), int(b), seed=i) for i, (a, b) in enumerate(dims)
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


class MultiHeadCausalAttention:
    """انتباه ذاتي متعدد الرؤوس مع قناع سببي (Causal)."""

    def __init__(self, d_model: int, n_heads: int = 8, seed: int = 0):
        assert d_model % n_heads == 0, "d_model يجب أن يقبل القسمة على n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        rng = np.random.default_rng(seed)
        scale = 0.02
        self.W_q = rng.normal(0, scale, (d_model, d_model)).astype(np.float64)
        self.W_k = rng.normal(0, scale, (d_model, d_model)).astype(np.float64)
        self.W_v = rng.normal(0, scale, (d_model, d_model)).astype(np.float64)
        self.W_o = rng.normal(0, scale, (d_model, d_model)).astype(np.float64)
        self._cache = {}

    def _split(self, x: np.ndarray) -> np.ndarray:
        # (S, d) -> (n_heads, S, d_head)
        S = x.shape[0]
        return x.reshape(S, self.n_heads, self.d_head).transpose(1, 0, 2)

    def _merge(self, x: np.ndarray) -> np.ndarray:
        # (n_heads, S, d_head) -> (S, d)
        return x.transpose(1, 0, 2).reshape(x.shape[1], self.d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        S = x.shape[0]
        Q = self._split(x @ self.W_q)
        K = self._split(x @ self.W_k)
        V = self._split(x @ self.W_v)
        scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / math.sqrt(self.d_head)
        # causal mask
        mask = np.triu(np.ones((S, S), dtype=np.float64), k=1) * (-1e9)
        scores = scores + mask[None, :, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        weights = np.exp(scores)
        weights = weights / (weights.sum(axis=-1, keepdims=True) + 1e-12)
        attn = np.matmul(weights, V)  # (H, S, dh)
        out = self._merge(attn) @ self.W_o
        self._cache = {"x": x, "Q": Q, "K": K, "V": V, "weights": weights, "attn": attn}
        return out

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        x = self._cache["x"]
        Q, K, V = self._cache["Q"], self._cache["K"], self._cache["V"]
        weights = self._cache["weights"]
        attn = self._cache["attn"]
        S = x.shape[0]

        # out = merge(attn) @ W_o
        merged = self._merge(attn)
        g_W_o = merged.T @ grad
        g_merged = grad @ self.W_o.T
        np.clip(g_W_o, -1, 1, out=g_W_o)
        self.W_o -= lr * g_W_o

        g_attn = self._split(g_merged)  # (H,S,dh)

        # attn = weights @ V
        g_weights = np.matmul(g_attn, np.transpose(V, (0, 2, 1)))  # (H,S,S)
        g_V = np.matmul(np.transpose(weights, (0, 2, 1)), g_attn)

        # softmax backward (approx per row)
        # dL/ds_i = w_i * (g_i - sum_j w_j g_j)
        sum_gw = (g_weights * weights).sum(axis=-1, keepdims=True)
        g_scores = weights * (g_weights - sum_gw)

        scale = 1.0 / math.sqrt(self.d_head)
        g_Q = np.matmul(g_scores, K) * scale
        g_K = np.matmul(np.transpose(g_scores, (0, 2, 1)), Q) * scale

        g_q = self._merge(g_Q)
        g_k = self._merge(g_K)
        g_v = self._merge(g_V)

        g_W_q = x.T @ g_q
        g_W_k = x.T @ g_k
        g_W_v = x.T @ g_v
        for gW, W in ((g_W_q, "W_q"), (g_W_k, "W_k"), (g_W_v, "W_v")):
            np.clip(gW, -1, 1, out=gW)
        self.W_q -= lr * g_W_q
        self.W_k -= lr * g_W_k
        self.W_v -= lr * g_W_v

        g_x = g_q @ self.W_q.T + g_k @ self.W_k.T + g_v @ self.W_v.T
        return g_x


class FeedForward:
    def __init__(self, d_model: int, d_ff: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.02, (d_ff, d_model)).astype(np.float64)
        self.b1 = np.zeros(d_ff, dtype=np.float64)
        self.W2 = rng.normal(0, 0.02, (d_model, d_ff)).astype(np.float64)
        self.b2 = np.zeros(d_model, dtype=np.float64)
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.W1.T + self.b1
        a = np.maximum(h, 0)  # ReLU
        out = a @ self.W2.T + self.b2
        self._cache = {"x": x, "h": h, "a": a}
        return out

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        x, h, a = self._cache["x"], self._cache["h"], self._cache["a"]
        g_W2 = grad.T @ a
        g_b2 = grad.sum(axis=0)
        g_a = grad @ self.W2
        g_h = g_a * (h > 0)
        g_W1 = g_h.T @ x
        g_b1 = g_h.sum(axis=0)
        g_x = g_h @ self.W1
        for arr in (g_W1, g_W2, g_b1, g_b2):
            np.clip(arr, -1, 1, out=arr)
        self.W1 -= lr * g_W1
        self.b1 -= lr * g_b1
        self.W2 -= lr * g_W2
        self.b2 -= lr * g_b2
        return g_x


class TransformerBlock:
    """Pre-LN: x + Attn(LN(x)) ثم x + FFN(LN(x))."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, seed: int = 0):
        self.ln1 = LayerNorm1D(d_model)
        self.attn = MultiHeadCausalAttention(d_model, n_heads, seed=seed)
        self.ln2 = LayerNorm1D(d_model)
        self.ffn = FeedForward(d_model, d_ff, seed=seed + 1)
        self._cache = {}

    def forward(self, x: np.ndarray) -> np.ndarray:
        n1 = self.ln1.forward(x)
        a = self.attn.forward(n1)
        x = x + a
        n2 = self.ln2.forward(x)
        f = self.ffn.forward(n2)
        out = x + f
        self._cache = {"x_in": self._cache.get("x_in"), "after_attn": x}
        self._cache["x0"] = None
        self._x0 = None
        self._after_attn = x - f  # x before ffn residual was x after attn residual
        # store properly
        self._res1_in = None
        return out

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        n1 = self.ln1.forward(x)
        a = self.attn.forward(n1)
        x2 = x + a
        self._x2 = x2
        n2 = self.ln2.forward(x2)
        f = self.ffn.forward(n2)
        return x2 + f

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        # out = x2 + ffn(ln2(x2))
        g_x2 = grad
        g_f = grad
        g_n2 = self.ffn.backward(g_f, lr)
        g_x2 = g_x2 + self.ln2.backward(g_n2, lr)
        # x2 = x + attn(ln1(x))
        g_x = g_x2
        g_a = g_x2
        g_n1 = self.attn.backward(g_a, lr)
        g_x = g_x + self.ln1.backward(g_n1, lr)
        return g_x

    def param_count(self) -> int:
        n = 0
        for W in (self.attn.W_q, self.attn.W_k, self.attn.W_v, self.attn.W_o):
            n += W.size
        n += self.ffn.W1.size + self.ffn.W2.size + self.ffn.b1.size + self.ffn.b2.size
        n += self.ln1.g.size * 2 + self.ln2.g.size * 2
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


def cosine_lr(step, total_steps, base_lr, warmup_steps=0, min_lr_ratio=0.1):
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, t / T))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


class HybridExperimentModel:
    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        project_d_model: Optional[int] = None,
        vocab_size: int = VOCAB_SIZE,
        lr: float = 1e-3,
        seed: int = 42,
        tokenizer: str = "word",
        vocab_path: Optional[str] = None,
        n_heads: int = DEFAULT_N_HEADS,
        n_pre: int = DEFAULT_N_PRE,
        n_post: int = DEFAULT_N_POST,
        d_ff_mult: int = DEFAULT_D_FF_MULT,
    ):
        if project_d_model is not None:
            d_model = project_d_model
        # d_model divisible by n_heads
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.base_lr = float(lr)
        self.lr = float(lr)
        self.seed = seed
        self.n_heads = n_heads
        self.n_pre = n_pre
        self.n_post = n_post

        if tokenizer == "hash":
            self.tokenizer: Union[WordTokenizer, HashTokenizer] = HashTokenizer(vocab_size)
        else:
            vp = vocab_path
            if vp is None:
                cand = _HERE / "tokenizer_vocab.json"
                vp = str(cand) if cand.exists() else None
            self.tokenizer = WordTokenizer(vocab_size, vocab_path=vp)

        self.chain = SurahChainNetwork(LAYER_DIMS)
        rng = np.random.default_rng(seed)
        self.E = rng.normal(0.0, 0.02, (self.vocab_size, self.d_model)).astype(np.float64)
        self.ln_in = LayerNorm1D(self.d_model)

        d_ff = self.d_model * d_ff_mult
        self.pre_blocks = [
            TransformerBlock(self.d_model, n_heads, d_ff, seed=seed + 10 + i)
            for i in range(n_pre)
        ]
        self.post_blocks = [
            TransformerBlock(self.d_model, n_heads, d_ff, seed=seed + 50 + i)
            for i in range(n_post)
        ]

        lim = np.sqrt(6.0 / (self.d_model + CHAIN_WIDTH))
        self.W_in = rng.uniform(-lim, lim, (CHAIN_WIDTH, self.d_model)).astype(np.float64)
        self.W_out = rng.uniform(-lim, lim, (self.d_model, CHAIN_WIDTH)).astype(np.float64)
        self.b_out = np.zeros(self.d_model, dtype=np.float64)
        lim_s = np.sqrt(6.0 / (self.d_model + self.d_model))
        self.W_skip = rng.uniform(-lim_s, lim_s, (self.d_model, self.d_model)).astype(np.float64)
        self.use_residual_bypass = True
        self._cache: dict = {}

    def build_tokenizer_from_texts(self, texts: Sequence[str], max_vocab: Optional[int] = None) -> int:
        if not isinstance(self.tokenizer, WordTokenizer):
            self.tokenizer = WordTokenizer(self.vocab_size)
        n = self.tokenizer.build_from_texts(list(texts), max_vocab=max_vocab or self.vocab_size)
        self.vocab_size = max(self.vocab_size, n)
        if self.E.shape[0] < self.vocab_size:
            rng = np.random.default_rng(self.seed)
            extra = rng.normal(0.0, 0.02, (self.vocab_size - self.E.shape[0], self.d_model)).astype(np.float64)
            self.E = np.vstack([self.E, extra])
        return n

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)

    def _encode_input(self, ids: np.ndarray) -> np.ndarray:
        ids = np.clip(ids.astype(np.int64), 0, self.E.shape[0] - 1)
        tok = self.E[ids]
        pos = _sinusoidal_positions(len(ids), self.d_model)
        x = self.ln_in.forward(tok + pos)
        self._cache["ids"] = ids
        return x

    def forward_logits(self, ids: np.ndarray) -> np.ndarray:
        x = self._encode_input(ids)
        for blk in self.pre_blocks:
            x = blk.forward(x)
        self._cache["x_pre"] = x

        h7 = x @ self.W_in.T
        self._cache["x_for_adapter"] = x
        h7o = self.chain.forward(h7)
        self._cache["h7"] = h7o
        h_chain = h7o @ self.W_out.T + self.b_out
        if self.use_residual_bypass:
            h = h_chain + x @ self.W_skip.T
            self._cache["x_skip"] = x
        else:
            h = h_chain

        for blk in self.post_blocks:
            h = blk.forward(h)
        self._cache["h"] = h
        return h @ self.E.T

    def train_step(self, text: str, max_len: int = 64) -> Optional[float]:
        ids = self.tokenizer.encode(text, max_len)
        if len(ids) < 2:
            return None
        inp = np.asarray(ids[:-1], dtype=np.int64)
        tgt = np.clip(np.asarray(ids[1:], dtype=np.int64), 0, self.E.shape[0] - 1)

        logits = self.forward_logits(inp)
        z = logits - logits.max(axis=-1, keepdims=True)
        exp = np.exp(z)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        n = len(tgt)
        loss = -np.log(np.clip(probs[np.arange(n), tgt], 1e-10, 1.0)).mean()

        g = probs.copy()
        g[np.arange(n), tgt] -= 1.0
        g /= n

        h = self._cache["h"]
        g_h = g @ self.E
        g_E = g.T @ h
        np.clip(g_E, -1.0, 1.0, out=g_E)
        self.E -= self.lr * g_E

        for blk in reversed(self.post_blocks):
            g_h = blk.backward(g_h, self.lr)

        g_x_skip = None
        g_chain_side = g_h
        if self.use_residual_bypass and "x_skip" in self._cache:
            x_skip = self._cache["x_skip"]
            g_W_skip = g_h.T @ x_skip
            np.clip(g_W_skip, -5.0, 5.0, out=g_W_skip)
            self.W_skip -= self.lr * g_W_skip
            g_x_skip = g_h @ self.W_skip

        h7 = self._cache["h7"]
        g_W_out = g_chain_side.T @ h7
        g_b = g_chain_side.sum(axis=0)
        np.clip(g_W_out, -5.0, 5.0, out=g_W_out)
        self.W_out -= self.lr * g_W_out
        self.b_out -= self.lr * g_b
        g7o = g_chain_side @ self.W_out
        g7 = self.chain.backward(g7o, self.lr)

        x = self._cache["x_for_adapter"]
        g_W_in = g7.T @ x
        np.clip(g_W_in, -5.0, 5.0, out=g_W_in)
        self.W_in -= self.lr * g_W_in
        g_x = g7 @ self.W_in
        if g_x_skip is not None:
            g_x = g_x + g_x_skip

        for blk in reversed(self.pre_blocks):
            g_x = blk.backward(g_x, self.lr)

        g_x = self.ln_in.backward(g_x, self.lr)
        for i, tid in enumerate(self._cache["ids"]):
            self.E[tid] -= self.lr * np.clip(g_x[i], -1.0, 1.0)
        return float(loss)

    def train_batch(self, texts, max_len=64, step=None, total_steps=None, warmup_steps=0):
        if step is not None and total_steps is not None:
            self.set_lr(cosine_lr(step, total_steps, self.base_lr, warmup_steps=warmup_steps))
        losses = []
        for t in texts:
            loss = self.train_step(t, max_len=max_len)
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else float("nan")

    def generate(self, prompt, max_new_tokens=32, temperature=0.9, top_k=40, max_ctx=96):
        ids = list(self.tokenizer.encode(prompt, max_ctx))
        eos = getattr(self.tokenizer, "EOS", 3)
        bos = getattr(self.tokenizer, "BOS", 2)
        pad = getattr(self.tokenizer, "PAD", 0)
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        if not ids:
            ids = [bos]
        for _ in range(max_new_tokens):
            ctx = np.asarray(ids[-max_ctx:], dtype=np.int64)
            logits = self.forward_logits(ctx)
            next_logits = logits[-1].astype(np.float64)
            next_logits[pad] = -1e9
            next_logits[bos] = -1e9
            if top_k and top_k > 0:
                k = min(top_k, next_logits.size)
                thresh = np.partition(next_logits, -k)[-k]
                next_logits = np.where(next_logits < thresh, -1e9, next_logits)
            temp = max(temperature, 1e-6)
            next_logits = next_logits / temp
            next_logits -= next_logits.max()
            probs = np.exp(next_logits)
            probs = probs / probs.sum()
            nid = int(np.random.choice(len(probs), p=probs))
            ids.append(nid)
            if nid == eos:
                break
        return self.tokenizer.decode(ids, skip_special=True)

    def param_count(self) -> dict:
        attn_params = sum(b.param_count() for b in self.pre_blocks + self.post_blocks)
        return {
            "chain": self.chain.param_count(),
            "attention_blocks": attn_params,
            "embedding_E": int(self.E.size),
            "adapters": int(self.W_in.size + self.W_out.size + self.b_out.size + self.W_skip.size),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_pre": self.n_pre,
            "n_post": self.n_post,
            "vocab_size": self.vocab_size,
            "tokenizer": type(self.tokenizer).__name__,
        }


SurahChainLM = HybridExperimentModel


if __name__ == "__main__":
    from hybrid_data import SENTENCES

    print("SurahChain LM + Multi-Head Attention — فحص")
    m = HybridExperimentModel(d_model=128, n_heads=4, n_pre=1, n_post=1, lr=1e-3)
    m.build_tokenizer_from_texts(SENTENCES)
    print(m.param_count())
    losses = [m.train_step(s) for s in SENTENCES[:12]]
    losses = [x for x in losses if x is not None]
    print("losses", [round(x, 3) for x in losses[:6]], "...", round(losses[-1], 3) if losses else None)
    print("gen:", m.generate("الصبر", max_new_tokens=10, temperature=0.8))
