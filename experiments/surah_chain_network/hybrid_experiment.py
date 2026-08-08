"""
Surah-Chain Network — نموذج لغوي (LLM-style) بوسط سوري

البنية:
  [Input LLM] Embedding + موضع + LayerNorm + Adapter→7
  [SurahChain ×114] من شبكهه_114-1.xlsx
  [Output LLM] Adapter 7→d + LM Head مربوط (weight tying)

قدرات LM:
  - WordTokenizer (decode حقيقي) مع بناء قاموس من النصوص
  - train_step / train_batch
  - generate() مع temperature و top-k
  - جدول LR: warmup + cosine decay
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
CHAIN_WIDTH = int(LAYER_DIMS[0][0])  # 7
VOCAB_SIZE = 8192
DEFAULT_D_MODEL = 512


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
        self._cache = (xhat,)
        return xhat * self.g + self.b

    def backward(self, grad: np.ndarray, lr: float) -> np.ndarray:
        (xhat,) = self._cache
        self.g -= lr * (grad * xhat).sum(axis=0)
        self.b -= lr * grad.sum(axis=0)
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


def _sinusoidal_positions(seq_len: int, d_model: int) -> np.ndarray:
    pos = np.zeros((seq_len, d_model), dtype=np.float64)
    for i in range(seq_len):
        for k in range(0, d_model, 2):
            ang = i / (10000.0 ** (k / d_model))
            pos[i, k] = np.sin(ang)
            if k + 1 < d_model:
                pos[i, k + 1] = np.cos(ang)
    return pos


def cosine_lr(
    step: int,
    total_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
    min_lr_ratio: float = 0.1,
) -> float:
    """Warmup خطي ثم cosine decay حتى min_lr_ratio * base_lr."""
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
    """
    نموذج لغوي: WordTokenizer → LLM Input → SurahChain → Tied LM Head.
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        project_d_model: Optional[int] = None,
        vocab_size: int = VOCAB_SIZE,
        lr: float = 1e-3,
        seed: int = 42,
        tokenizer: str = "word",  # "word" | "hash"
        vocab_path: Optional[str] = None,
    ):
        if project_d_model is not None:
            d_model = project_d_model
        self.d_model = int(d_model)
        self.vocab_size = int(vocab_size)
        self.base_lr = float(lr)
        self.lr = float(lr)
        self.seed = seed

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
        lim = np.sqrt(6.0 / (self.d_model + CHAIN_WIDTH))
        self.W_in = rng.uniform(-lim, lim, (CHAIN_WIDTH, self.d_model)).astype(np.float64)
        self.W_out = rng.uniform(-lim, lim, (self.d_model, CHAIN_WIDTH)).astype(np.float64)
        self.b_out = np.zeros(self.d_model, dtype=np.float64)
        self._cache: dict = {}

    def build_tokenizer_from_texts(self, texts: Sequence[str], max_vocab: Optional[int] = None) -> int:
        """يبني قاموس WordTokenizer من نصوص التدريب (يجب قبل تدريب جدّي)."""
        if not isinstance(self.tokenizer, WordTokenizer):
            self.tokenizer = WordTokenizer(self.vocab_size)
        n = self.tokenizer.build_from_texts(list(texts), max_vocab=max_vocab or self.vocab_size)
        self.vocab_size = max(self.vocab_size, n)
        # توسيع E إن لزم
        if self.E.shape[0] < self.vocab_size:
            rng = np.random.default_rng(self.seed)
            extra = rng.normal(
                0.0, 0.02, (self.vocab_size - self.E.shape[0], self.d_model)
            ).astype(np.float64)
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

    def _to_chain(self, x: np.ndarray) -> np.ndarray:
        self._cache["x_for_adapter"] = x
        return x @ self.W_in.T

    def _from_chain(self, h7: np.ndarray) -> np.ndarray:
        self._cache["h7"] = h7
        return h7 @ self.W_out.T + self.b_out

    def _lm_logits(self, h: np.ndarray) -> np.ndarray:
        self._cache["h"] = h
        return h @ self.E.T

    def forward_logits(self, ids: np.ndarray) -> np.ndarray:
        x = self._encode_input(ids)
        h7 = self._to_chain(x)
        h7o = self.chain.forward(h7)
        h = self._from_chain(h7o)
        return self._lm_logits(h)

    def train_step(self, text: str, max_len: int = 32) -> Optional[float]:
        ids = self.tokenizer.encode(text, max_len)
        if len(ids) < 2:
            return None
        inp = np.asarray(ids[:-1], dtype=np.int64)
        tgt = np.asarray(ids[1:], dtype=np.int64)
        tgt = np.clip(tgt, 0, self.E.shape[0] - 1)

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

        h7 = self._cache["h7"]
        g_W_out = g_h.T @ h7
        g_b = g_h.sum(axis=0)
        np.clip(g_W_out, -5.0, 5.0, out=g_W_out)
        self.W_out -= self.lr * g_W_out
        self.b_out -= self.lr * g_b
        g7o = g_h @ self.W_out

        g7 = self.chain.backward(g7o, self.lr)

        x = self._cache["x_for_adapter"]
        g_W_in = g7.T @ x
        np.clip(g_W_in, -5.0, 5.0, out=g_W_in)
        self.W_in -= self.lr * g_W_in
        g_x = g7 @ self.W_in

        g_x = self.ln_in.backward(g_x, self.lr)
        for i, tid in enumerate(self._cache["ids"]):
            self.E[tid] -= self.lr * np.clip(g_x[i], -1.0, 1.0)

        return float(loss)

    def train_batch(
        self,
        texts: Sequence[str],
        max_len: int = 32,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        warmup_steps: int = 0,
    ) -> float:
        """
        دفعة نصوص: يحدّث LR حسب cosine+warmup إن مُرّر step/total_steps،
        ثم يدرّب كل جملة في الدفعة ويُرجع متوسط الخسارة.
        """
        if step is not None and total_steps is not None:
            self.set_lr(
                cosine_lr(step, total_steps, self.base_lr, warmup_steps=warmup_steps)
            )
        losses = []
        for t in texts:
            loss = self.train_step(t, max_len=max_len)
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else float("nan")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 24,
        temperature: float = 0.9,
        top_k: int = 40,
        max_ctx: int = 48,
    ) -> str:
        """
        توليد سببي (autoregressive): يضيف رمزاً واحداً كل مرة حتى EOS أو الحد.
        """
        ids = list(self.tokenizer.encode(prompt, max_ctx))
        # أزل EOS الختامي من الـprompt إن وُجد ليُكمِل التوليد
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

            # امنع PAD/BOS في العيّنة
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
        return {
            "chain": self.chain.param_count(),
            "embedding_E": int(self.E.size),
            "adapters": int(self.W_in.size + self.W_out.size + self.b_out.size),
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "tokenizer": type(self.tokenizer).__name__,
            "chain_width": CHAIN_WIDTH,
            "n_chain_layers": len(self.chain.layers),
        }


SurahChainLM = HybridExperimentModel


if __name__ == "__main__":
    from hybrid_data import SENTENCES

    print("SurahChain LM — فحص generate + batch")
    m = HybridExperimentModel(d_model=128, lr=2e-3, tokenizer="word")
    m.build_tokenizer_from_texts(SENTENCES)
    print("params:", m.param_count())
    print("vocab words:", len(getattr(m.tokenizer, "word_to_id", {})))

    losses = []
    for step, batch_start in enumerate(range(0, len(SENTENCES), 8)):
        batch = SENTENCES[batch_start : batch_start + 8]
        loss = m.train_batch(batch, step=step, total_steps=20, warmup_steps=2)
        losses.append(loss)
        if step >= 5:
            break
    print("batch losses:", [round(x, 3) for x in losses])

    out = m.generate("الصبر", max_new_tokens=12, temperature=0.8, top_k=20)
    print("generate('الصبر') →", out)
