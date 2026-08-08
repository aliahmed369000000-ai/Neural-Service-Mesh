"""
Surah-Chain LM — نسخة PyTorch

نفس المعمارية:
  StrongTokenizer
  → Embed + Pos + LN
  → TransformerBlock × N_PRE  (Causal MHA + GELU FFN)
  → Adapter → SurahChain×114 → Adapter + residual skip
  → TransformerBlock × N_POST
  → Tied LM Head

استخدام GPU تلقائياً إن وُجد: device = cuda | cpu
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from strong_tokenizer import StrongTokenizer

_HERE = Path(__file__).resolve().parent
_DIMS_PATH = _HERE / "surah_layer_dims.json"
LAYER_DIMS: List[List[int]] = json.loads(_DIMS_PATH.read_text())
CHAIN_WIDTH = int(LAYER_DIMS[0][0])  # 7

DEFAULT_D_MODEL = 256
DEFAULT_N_HEADS = 8
DEFAULT_N_PRE = 2
DEFAULT_N_POST = 2
DEFAULT_MAX_CTX = 256
DEFAULT_MAX_LEN = 128


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cosine_lr(step, total_steps, base_lr, warmup_steps=0, min_lr_ratio=0.1):
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, t / T))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


class SurahChainLayer(nn.Module):
    """طبقة FC + GELU + LN + residual (مع إسقاط إن اختلف البعد)."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.shortcut = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(F.gelu(self.fc(x)))
        sc = self.shortcut(x) if self.shortcut is not None else x
        return h + sc


class SurahChainNetwork(nn.Module):
    def __init__(self, layer_dims: Optional[List[List[int]]] = None):
        super().__init__()
        dims = layer_dims or LAYER_DIMS
        self.layers = nn.ModuleList(
            [SurahChainLayer(int(a), int(b)) for a, b in dims]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # causal mask
        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        att = att.masked_fill(mask, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.drop(att)
        y = att @ v  # (B, H, S, dh)
        y = y.transpose(1, 2).contiguous().reshape(B, S, D)
        return self.drop(self.proj(y))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class SurahChainLM(nn.Module):
    """نموذج لغوي كامل على PyTorch."""

    def __init__(
        self,
        vocab_size: int = 8192,
        d_model: int = DEFAULT_D_MODEL,
        n_heads: int = DEFAULT_N_HEADS,
        n_pre: int = DEFAULT_N_PRE,
        n_post: int = DEFAULT_N_POST,
        dropout: float = 0.1,
        max_seq: int = DEFAULT_MAX_CTX,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_seq = max_seq

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln_in = nn.LayerNorm(d_model)

        d_ff = d_model * 4
        self.pre_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_pre)]
        )
        self.chain = SurahChainNetwork(LAYER_DIMS)
        self.W_in = nn.Linear(d_model, CHAIN_WIDTH)
        self.W_out = nn.Linear(CHAIN_WIDTH, d_model)
        self.W_skip = nn.Linear(d_model, d_model, bias=False)
        self.post_blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_post)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        # tied head: use tok_emb.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        idx: (B, S) long
        returns logits: (B, S, vocab)
        """
        B, S = idx.shape
        if S > self.max_seq:
            idx = idx[:, -self.max_seq :]
            S = idx.shape[1]
        pos = torch.arange(S, device=idx.device).unsqueeze(0)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.ln_in(x)

        for blk in self.pre_blocks:
            x = blk(x)

        h7 = self.W_in(x)
        # SurahChain يتوقع آخر بُعد = عرض السلسلة؛ طبّق على كل موضع
        flat = h7.reshape(-1, CHAIN_WIDTH)
        flat = self.chain(flat)
        h_chain = self.W_out(flat).reshape(B, S, self.d_model)
        x = h_chain + self.W_skip(x)

        for blk in self.post_blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = F.linear(x, self.tok_emb.weight)  # weight tying
        return logits

    def param_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        chain = sum(p.numel() for p in self.chain.parameters())
        return {
            "total": total,
            "chain": chain,
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
        }


class HybridExperimentModelTorch:
    """
    غلاف يوفّر نفس أسلوب الاستخدام تقريباً:
      build_tokenizer_from_texts, train_batch, generate, set_lr
    """

    def __init__(
        self,
        d_model: int = DEFAULT_D_MODEL,
        vocab_size: int = 8192,
        lr: float = 1e-3,
        n_heads: int = DEFAULT_N_HEADS,
        n_pre: int = DEFAULT_N_PRE,
        n_post: int = DEFAULT_N_POST,
        device: Optional[Union[str, torch.device]] = None,
        max_seq: int = DEFAULT_MAX_CTX,
    ):
        self.device = torch.device(device) if device else get_device()
        self.base_lr = lr
        self.lr = lr
        self.tokenizer = StrongTokenizer(vocab_size)
        self.model = SurahChainLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_pre=n_pre,
            n_post=n_post,
            max_seq=max_seq,
        ).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        self.vocab_size = vocab_size

    def build_tokenizer_from_texts(self, texts: Sequence[str], max_vocab: Optional[int] = None) -> int:
        n = self.tokenizer.build_from_texts(list(texts), max_vocab=max_vocab or self.vocab_size)
        self.vocab_size = max(self.vocab_size, n)
        # إعادة بناء embedding إن كبر القاموس
        if self.model.tok_emb.num_embeddings < self.vocab_size:
            old = self.model
            self.model = SurahChainLM(
                vocab_size=self.vocab_size,
                d_model=old.d_model,
                n_heads=old.pre_blocks[0].attn.n_heads if old.pre_blocks else DEFAULT_N_HEADS,
                n_pre=len(old.pre_blocks),
                n_post=len(old.post_blocks),
                max_seq=old.max_seq,
            ).to(self.device)
            self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        return int(n)

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def train_step(self, text: str, max_len: int = DEFAULT_MAX_LEN) -> Optional[float]:
        ids = self.tokenizer.encode(text, max_len)
        if len(ids) < 2:
            return None
        x = torch.tensor(ids[:-1], dtype=torch.long, device=self.device).unsqueeze(0)
        y = torch.tensor(ids[1:], dtype=torch.long, device=self.device).unsqueeze(0)
        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    def train_batch(
        self,
        texts: Sequence[str],
        max_len: int = DEFAULT_MAX_LEN,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        warmup_steps: int = 0,
    ) -> float:
        if step is not None and total_steps is not None:
            self.set_lr(cosine_lr(step, total_steps, self.base_lr, warmup_steps))
        losses = []
        for t in texts:
            loss = self.train_step(t, max_len=max_len)
            if loss is not None:
                losses.append(loss)
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 48,
        temperature: float = 0.85,
        top_k: int = 50,
        top_p: float = 0.92,
        repetition_penalty: float = 1.15,
        max_ctx: int = DEFAULT_MAX_CTX,
        min_new_tokens: int = 1,
    ) -> str:
        self.model.eval()
        ids = list(self.tokenizer.encode(prompt, max_ctx))
        eos, bos, pad = self.tokenizer.EOS, self.tokenizer.BOS, self.tokenizer.PAD
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        if not ids:
            ids = [bos]

        for step_i in range(max_new_tokens):
            ctx = torch.tensor(ids[-max_ctx:], dtype=torch.long, device=self.device).unsqueeze(0)
            logits = self.model(ctx)[0, -1].float()
            logits[pad] = -1e9
            logits[bos] = -1e9
            if step_i < min_new_tokens:
                logits[eos] = -1e9
            if repetition_penalty and repetition_penalty != 1.0:
                for prev in set(ids):
                    if logits[prev] > 0:
                        logits[prev] /= repetition_penalty
                    else:
                        logits[prev] *= repetition_penalty
            if top_k and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(0)))
                logits[logits < v[-1]] = -float("inf")
            logits = logits / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)
            if top_p is not None and 0 < top_p < 1:
                sorted_p, sorted_i = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_p, dim=0)
                mask = cum > top_p
                mask[0] = False
                sorted_p[mask] = 0
                sorted_p = sorted_p / sorted_p.sum()
                choice = torch.multinomial(sorted_p, 1).item()
                nid = int(sorted_i[choice].item())
            else:
                nid = int(torch.multinomial(probs, 1).item())
            ids.append(nid)
            if nid == eos and step_i + 1 >= min_new_tokens:
                break
        return self.tokenizer.decode(ids, skip_special=True)

    def param_count(self) -> dict:
        d = self.model.param_count()
        d["device"] = str(self.device)
        d["tokenizer"] = "StrongTokenizer"
        return d

    def save(self, path: str) -> None:
        path = str(path)
        torch.save(
            {
                "model": self.model.state_dict(),
                "vocab_size": self.vocab_size,
                "d_model": self.model.d_model,
                "tokenizer": {
                    "word_to_id": self.tokenizer.word_to_id,
                    "merges": self.tokenizer.merges,
                    "vocab_size": self.tokenizer.vocab_size,
                },
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.vocab_size = int(ckpt.get("vocab_size", self.vocab_size))
        tok = ckpt.get("tokenizer")
        if tok:
            self.tokenizer.word_to_id = {str(k): int(v) for k, v in tok["word_to_id"].items()}
            self.tokenizer.id_to_word = {int(v): str(k) for k, v in self.tokenizer.word_to_id.items()}
            self.tokenizer.merges = [tuple(x) for x in tok.get("merges", [])]
            self.tokenizer.vocab_size = int(tok.get("vocab_size", self.vocab_size))
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)


# توافق الاسم
HybridExperimentModel = HybridExperimentModelTorch


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(_HERE))
    from hybrid_data import SENTENCES

    print("device:", get_device())
    m = HybridExperimentModelTorch(d_model=128, n_heads=4, n_pre=1, n_post=1, lr=1e-3)
    m.build_tokenizer_from_texts(SENTENCES)
    print(m.param_count())
    losses = [m.train_step(s) for s in SENTENCES[:20]]
    losses = [x for x in losses if x is not None]
    print("loss", round(losses[0], 3), "->", round(losses[-1], 3))
    print("gen:", m.generate("الصبر", max_new_tokens=12))
