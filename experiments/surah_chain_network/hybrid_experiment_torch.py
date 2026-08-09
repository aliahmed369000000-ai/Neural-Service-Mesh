"""
Surah-Chain LM — PyTorch فقط (بدون NumPy)

تحسينات الأداء:
  - دفعات حقيقية (padding + ignore_index)
  - scaled_dot_product_attention (سببي / Flash عند الإمكان)
  - AdamW + clip_grad + cosine warmup
  - torch.compile اختياري
  - GPU تلقائي
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from strong_tokenizer import StrongTokenizer

_HERE = Path(__file__).resolve().parent
LAYER_DIMS: List[List[int]] = json.loads((_HERE / "surah_layer_dims.json").read_text())
CHAIN_WIDTH = int(LAYER_DIMS[0][0])

DEFAULT_D_MODEL = 256
DEFAULT_N_HEADS = 8
DEFAULT_N_PRE = 2
DEFAULT_N_POST = 2
DEFAULT_MAX_CTX = 256
DEFAULT_MAX_LEN = 128


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cosine_lr(step, total_steps, base_lr, warmup_steps=0, min_lr_ratio=0.1):
    if total_steps <= 0:
        return base_lr
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, t / float(T)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


class SurahChainLayer(nn.Module):
    """طبقة سلسلة السور مع Highway gate + LayerScale + دعم التوسيع الذاتي."""

    def __init__(self, d_in: int, d_out: int, layer_scale_init: float = 1e-2):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.fc = nn.Linear(d_in, d_out)
        self.ln = nn.LayerNorm(d_out)
        self.shortcut = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else None
        self.gate = nn.Linear(d_in, d_out)
        self.layer_scale = nn.Parameter(torch.ones(d_out) * layer_scale_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln(F.gelu(self.fc(x)))
        sc = self.shortcut(x) if self.shortcut is not None else x
        g = torch.sigmoid(self.gate(x))
        h = self.layer_scale * h
        return g * h + (1.0 - g) * sc

    def expand_out(self, delta: int = 1, noise: float = 1e-4) -> None:
        """يوسّع بُعد المخرج +delta (صف جديد)."""
        if delta <= 0:
            return
        old_out = self.d_out
        new_out = old_out + delta
        device, dtype = self.fc.weight.device, self.fc.weight.dtype

        def _expand_linear_out(lin: nn.Linear, has_bias: bool) -> nn.Linear:
            new = nn.Linear(lin.in_features, new_out, bias=has_bias).to(device=device, dtype=dtype)
            with torch.no_grad():
                new.weight[:old_out] = lin.weight
                new.weight[old_out:].normal_(0.0, noise)
                if has_bias and lin.bias is not None:
                    new.bias[:old_out] = lin.bias
                    new.bias[old_out:].zero_()
            return new

        self.fc = _expand_linear_out(self.fc, True)
        self.gate = _expand_linear_out(self.gate, True)
        if self.shortcut is not None:
            self.shortcut = _expand_linear_out(self.shortcut, False)
        else:
            self.shortcut = nn.Linear(self.d_in, new_out, bias=False).to(device=device, dtype=dtype)
            with torch.no_grad():
                self.shortcut.weight.zero_()
                eye = min(self.d_in, old_out)
                self.shortcut.weight[:eye, :eye] = torch.eye(eye, device=device, dtype=dtype)

        new_ln = nn.LayerNorm(new_out).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_ln.weight[:old_out] = self.ln.weight
            new_ln.bias[:old_out] = self.ln.bias
            new_ln.weight[old_out:].fill_(1.0)
            new_ln.bias[old_out:].zero_()
        self.ln = new_ln

        new_ls = nn.Parameter(torch.ones(new_out, device=device, dtype=dtype) * 1e-2)
        with torch.no_grad():
            new_ls[:old_out] = self.layer_scale
        self.layer_scale = new_ls
        self.d_out = new_out

    def expand_in(self, delta: int = 1, noise: float = 1e-4) -> None:
        """يوسّع بُعد المدخل +delta (عمود جديد)."""
        if delta <= 0:
            return
        old_in = self.d_in
        new_in = old_in + delta
        device, dtype = self.fc.weight.device, self.fc.weight.dtype

        def _expand_linear_in(lin: nn.Linear) -> nn.Linear:
            new = nn.Linear(new_in, lin.out_features, bias=lin.bias is not None).to(
                device=device, dtype=dtype
            )
            with torch.no_grad():
                new.weight[:, :old_in] = lin.weight
                new.weight[:, old_in:].normal_(0.0, noise)
                if lin.bias is not None:
                    new.bias.copy_(lin.bias)
            return new

        self.fc = _expand_linear_in(self.fc)
        self.gate = _expand_linear_in(self.gate)
        if self.shortcut is not None:
            self.shortcut = _expand_linear_in(self.shortcut)
        else:
            self.shortcut = nn.Linear(new_in, self.d_out, bias=False).to(device=device, dtype=dtype)
            with torch.no_grad():
                self.shortcut.weight.zero_()
                eye = min(old_in, self.d_out)
                self.shortcut.weight[:eye, :eye] = torch.eye(eye, device=device, dtype=dtype)
        self.d_in = new_in


class SurahChainNetwork(nn.Module):
    def __init__(self, layer_dims: Optional[List[List[int]]] = None):
        super().__init__()
        dims = layer_dims or LAYER_DIMS
        self.layers = nn.ModuleList([SurahChainLayer(int(a), int(b)) for a, b in dims])
        self.layer_dims = [[int(a), int(b)] for a, b in dims]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def expand_narrowest(self, delta: int = 1) -> Optional[dict]:
        """
        عند توقف الـloss: يوسّع الأضيق تلقائياً (عمود + صف).
        يختار الحافة الداخلية ذات أصغر سعة، يوسّع d_out للطبقة i
        و d_in للطبقة i+1. لا يلمس مدخل/مخرج السلسلة (عرض 7) حفاظاً على W_in/W_out.
        """
        n = len(self.layers)
        if n < 3:
            return None
        candidates = []
        for i in range(1, n - 1):
            d_in = self.layers[i].d_in
            d_out = self.layers[i].d_out
            capacity = min(d_in, d_out) * 1_000_000 + (d_in * d_out)
            candidates.append((capacity, i, d_in, d_out))
        if not candidates:
            return None
        candidates.sort()
        _, idx, old_in, old_out = candidates[0]
        self.layers[idx].expand_out(delta)
        self.layers[idx + 1].expand_in(delta)
        self.layer_dims[idx][1] += delta
        self.layer_dims[idx + 1][0] += delta
        return {
            "layer_idx": idx,
            "old": (old_in, old_out),
            "new_out": self.layers[idx].d_out,
            "next_new_in": self.layers[idx + 1].d_in,
            "delta": delta,
        }


class CausalSelfAttention(nn.Module):
    """انتباه سببي عبر scaled_dot_product_attention (أسرع)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, S, D)  key_padding_mask: (B, S) True = PAD
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, dh)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # SDPA: is_causal=True أسرع من mask يدوي
        drop_p = self.dropout if self.training else 0.0
        # attn_mask for padding: True means "ignore" in SDPA float mask convention differs;
        # use key padding by setting k/v positions - we use additive mask
        if key_padding_mask is not None:
            # (B, S) True=pad -> (B, 1, 1, S)
            pad = key_padding_mask[:, None, None, :]
            # float mask: -inf where pad
            attn_bias = torch.zeros(B, 1, 1, S, device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(pad, float("-inf"))
            # combine with causal via is_causal only when no pad is tricky;
            # build full causal+pad mask
            causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            # (1, 1, S, S)
            full = causal[None, None, :, :]
            bias = torch.zeros(B, 1, S, S, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(full, float("-inf"))
            bias = bias.masked_fill(pad.expand(B, 1, S, S), float("-inf"))
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias, dropout_p=drop_p)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop_p)

        y = y.transpose(1, 2).contiguous().reshape(B, S, D)
        return self.resid_drop(self.proj(y))


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

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x


class SurahChainLM(nn.Module):
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
        self.n_heads = n_heads
        self.n_pre = n_pre
        self.n_post = n_post

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
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
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if getattr(m, "padding_idx", None) is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(
        self,
        idx: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        idx: (B, S)
        key_padding_mask: (B, S) True حيث PAD
        logits: (B, S, vocab)
        """
        B, S = idx.shape
        if S > self.max_seq:
            idx = idx[:, -self.max_seq :]
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, -self.max_seq :]
            S = idx.shape[1]

        pos = torch.arange(S, device=idx.device).unsqueeze(0).expand(B, -1)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        x = self.ln_in(x)

        for blk in self.pre_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        h7 = self.W_in(x)
        flat = h7.reshape(B * S, CHAIN_WIDTH)
        flat = self.chain(flat)
        h_chain = self.W_out(flat).reshape(B, S, self.d_model)
        x = h_chain + self.W_skip(x)

        for blk in self.post_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        return F.linear(x, self.tok_emb.weight)

    def param_count(self) -> dict:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "chain": sum(p.numel() for p in self.chain.parameters()),
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
        }


class HybridExperimentModelTorch:
    """واجهة تدريب/توليد — PyTorch فقط، دفعات حقيقية."""

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
        compile_model: bool = False,
    ):
        self.device = torch.device(device) if device else get_device()
        self.base_lr = lr
        self.lr = lr
        self.vocab_size = vocab_size
        self.tokenizer = StrongTokenizer(vocab_size)
        self.model = SurahChainLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_pre=n_pre,
            n_post=n_post,
            max_seq=max_seq,
        ).to(self.device)

        if compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                self._compiled = True
            except Exception:
                self._compiled = False
        else:
            self._compiled = False

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95)
        )

    def build_tokenizer_from_texts(self, texts: Sequence[str], max_vocab: Optional[int] = None) -> int:
        n = self.tokenizer.build_from_texts(list(texts), max_vocab=max_vocab or self.vocab_size)
        self.vocab_size = max(self.vocab_size, int(n))
        need = self.vocab_size
        emb = self.model.tok_emb if not self._compiled else self.model
        # إن كبر القاموس أعد إنشاء النموذج
        current_vs = self.model.vocab_size if hasattr(self.model, "vocab_size") else need
        # access underlying module if compiled
        core = getattr(self.model, "_orig_mod", self.model)
        if core.tok_emb.num_embeddings < need:
            cfg = dict(
                vocab_size=need,
                d_model=core.d_model,
                n_heads=core.n_heads,
                n_pre=core.n_pre,
                n_post=core.n_post,
                max_seq=core.max_seq,
            )
            self.model = SurahChainLM(**cfg).to(self.device)
            self._compiled = False
            self.opt = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
        return int(n)

    def set_lr(self, lr: float) -> None:
        self.lr = float(lr)
        for g in self.opt.param_groups:
            g["lr"] = self.lr

    def _encode_batch(self, texts: Sequence[str], max_len: int):
        """يحضّر tensors: input_ids, labels, pad_mask (True=PAD)."""
        seqs = []
        for t in texts:
            ids = self.tokenizer.encode(t, max_len)
            if len(ids) < 2:
                continue
            seqs.append(ids.tolist() if hasattr(ids, "tolist") else list(ids))
        if not seqs:
            return None
        # x = all but last, y = all but first
        xs = [s[:-1] for s in seqs]
        ys = [s[1:] for s in seqs]
        max_s = max(len(x) for x in xs)
        B = len(xs)
        x_pad = torch.zeros(B, max_s, dtype=torch.long)
        y_pad = torch.full((B, max_s), -100, dtype=torch.long)  # ignore_index
        pad_mask = torch.ones(B, max_s, dtype=torch.bool)  # True = pad
        for i, (x, y) in enumerate(zip(xs, ys)):
            L = len(x)
            x_pad[i, :L] = torch.tensor(x, dtype=torch.long)
            y_pad[i, :L] = torch.tensor(y, dtype=torch.long)
            pad_mask[i, :L] = False
        return (
            x_pad.to(self.device),
            y_pad.to(self.device),
            pad_mask.to(self.device),
        )

    def train_batch(
        self,
        texts: Sequence[str],
        max_len: int = DEFAULT_MAX_LEN,
        step: Optional[int] = None,
        total_steps: Optional[int] = None,
        warmup_steps: int = 0,
    ) -> float:
        """دفعة واحدة حقيقية (مصفوفة B×S) — أسرع بكثير من جملة بجملة."""
        if step is not None and total_steps is not None:
            self.set_lr(cosine_lr(step, total_steps, self.base_lr, warmup_steps))

        packed = self._encode_batch(texts, max_len)
        if packed is None:
            return float("nan")
        x, y, pad_mask = packed

        self.model.train()
        self.opt.zero_grad(set_to_none=True)
        logits = self.model(x, key_padding_mask=pad_mask)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1),
            ignore_index=-100,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        return float(loss.item())

    def train_step(self, text: str, max_len: int = DEFAULT_MAX_LEN) -> Optional[float]:
        return self.train_batch([text], max_len=max_len)

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
            ctx = torch.tensor([ids[-max_ctx:]], dtype=torch.long, device=self.device)
            logits = self.model(ctx)[0, -1].float()
            logits[pad] = -1e9
            logits[bos] = -1e9
            if step_i < min_new_tokens:
                logits[eos] = -1e9
            if repetition_penalty != 1.0:
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
                sorted_p = sorted_p.masked_fill(mask, 0.0)
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
        core = getattr(self.model, "_orig_mod", self.model)
        d = core.param_count()
        d["device"] = str(self.device)
        d["compiled"] = self._compiled
        d["tokenizer"] = "StrongTokenizer"
        return d

    def expand_narrowest(self, delta: int = 1) -> Optional[dict]:
        """
        توسيع ذاتي: إذا توقف الـloss يُستدعى هذا لتوسيع الأضيق
        في سلسلة السور (عمود + صف). يعيد بناء الـoptimizer لأن
        شكل بعض المعاملات تغيّر.
        """
        core = getattr(self.model, "_orig_mod", self.model)
        info = core.chain.expand_narrowest(delta=delta)
        if info is None:
            return None
        # إعادة إنشاء الـoptimizer بعد تغيّر أشكال الأوزان
        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
        )
        return info

    def save(self, path: str) -> None:
        core = getattr(self.model, "_orig_mod", self.model)
        torch.save(
            {
                "model": core.state_dict(),
                "vocab_size": self.vocab_size,
                "d_model": core.d_model,
                "n_heads": core.n_heads,
                "n_pre": core.n_pre,
                "n_post": core.n_post,
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
        core = getattr(self.model, "_orig_mod", self.model)
        core.load_state_dict(ckpt["model"])
        core.to(self.device)


HybridExperimentModel = HybridExperimentModelTorch


if __name__ == "__main__":
    import sys
    import time

    sys.path.insert(0, str(_HERE))
    from hybrid_data import SENTENCES

    print("device:", get_device())
    m = HybridExperimentModelTorch(d_model=128, n_heads=4, n_pre=1, n_post=1, lr=1e-3)
    m.build_tokenizer_from_texts(SENTENCES)
    print(m.param_count())
    t0 = time.time()
    for _ in range(5):
        loss = m.train_batch(SENTENCES[:16], max_len=48)
    print(f"5 real batches in {time.time()-t0:.2f}s  last_loss={loss:.3f}")
    print("gen:", m.generate("الصبر", max_new_tokens=12))
