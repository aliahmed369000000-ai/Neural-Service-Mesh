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

import contextlib
import json
import math
import os
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

DEFAULT_D_MODEL = 128
DEFAULT_N_HEADS = 8
DEFAULT_N_PRE = 2
DEFAULT_N_POST = 2
DEFAULT_MAX_CTX = 256
DEFAULT_MAX_LEN = 128


def scale_surah_dims(dims: List[List[int]], scale: float) -> List[List[int]]:
    """
    يوسّع أبعاد سلسلة السور بنسبة scale مع الحفاظ على نسب السور وتواصل الطبقات.
    scale=1 → الأبعاد الأصلية | scale=2 → سعة أعلى مع نفس البنية.
    """
    if scale is None or scale <= 1.0:
        return [[int(a), int(b)] for a, b in dims]
    widths = [int(dims[0][0])] + [int(b) for _, b in dims]
    widths = [max(1, int(round(w * float(scale)))) for w in widths]
    return [[widths[i], widths[i + 1]] for i in range(len(widths) - 1)]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


_XLA_DEVICE_CACHE: Optional[torch.device] = None
_XLA_DEVICE_TRIED: bool = False


def _xla_available() -> bool:
    """هل PyTorch/XLA (TPU) متاح في بيئة التشغيل؟"""
    try:
        import torch_xla  # noqa: F401
        return True
    except Exception:
        return False


def get_xla_device() -> Optional[torch.device]:
    """يُرجع جهاز XLA/TPU إن كان متاحًا ومطلوبًا عبر SCN_TPU=1، وإلا None.

    على Kaggle TPU v5e-8: صور TPUVM تأتي مع torch_xla مثبتًا. تفعيل
    SCN_TPU=1 يحرك التدريب إلى xm.xla_device() مع mark_step() بعد كل خطوة.

    إصلاح انهيار SIGABRT ("InitializeComputationClient() can only be called
    once"): جهاز XLA يُهيَّأ مرة واحدة فقط ويُخزَّن في cache داخل الوحدة،
    حتى لا تتكرر تهيئة XLAGuardImpl/GetComputationClient من استيرادات
    xm متعددة داخل الدوال أو من torch.autograd.Engine عند النقل إلى xla.
    """
    global _XLA_DEVICE_CACHE, _XLA_DEVICE_TRIED
    if os.environ.get("SCN_TPU", "0") != "1":
        return None
    if _XLA_DEVICE_TRIED:
        return _XLA_DEVICE_CACHE
    _XLA_DEVICE_TRIED = True
    try:
        import torch_xla  # noqa: F401  (تهيئة واحدة عبر import الجذر)
    except Exception:
        return None
    if not _xla_available():
        return None
    try:
        import torch_xla.core.xla_model as xm
        dev = xm.xla_device()
        _XLA_DEVICE_CACHE = dev
        return dev
    except Exception:
        return None


def xla_mark_step(device: Optional[torch.device]) -> None:
    """مزامنة XLA/TPU بعد خطوات المحسّن — إلزامية على TPU."""
    if device is None or getattr(device, "type", "") != "xla":
        return
    try:
        import torch_xla.core.xla_model as xm
        xm.mark_step(device)
    except Exception:
        pass


def xla_dtype() -> torch.dtype:
    """دقة الافتراض على TPU: bfloat16 (الأفضل أداءً واستقرارًا على v5e)."""
    return torch.bfloat16 if os.environ.get("SCN_TPU_DTYPE", "bf16") == "bf16" else torch.float32


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


class RMSNorm(nn.Module):
    """RMSNorm (بدون bias) — الاستقرار الموصى به للنماذج الكبيرة بدل LayerNorm العادي."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class SurahChainLayer(nn.Module):
    """طبقة سلسلة السور مع Highway gate + LayerScale + دعم التوسيع الذاتي.

    ملاحظة تهيئة (fix): مسار الـshortcut غير مُطبَّع (بدون LayerNorm)، فتكرار
    ضربه بمصفوفة عشوائية عبر 114 طبقة يُنتج انكماشاً أُسّياً للإشارة (تحقّق
    مُختبر: RMS المخرج يهبط لـ~1e-107 بعد 113 طبقة عند التهيئة العشوائية
    الافتراضية). لذلك يُهيَّأ الـshortcut هنا identity-like (نفس نمط eye()
    المستخدم أصلاً في expand_out/expand_in) بدل عشوائي بالكامل، مع رفع
    layer_scale_init الافتراضي من 1e-2 إلى 1.0 — تحقّق numpy مستقل أظهر أن
    هذا يبقي RMS المخرج مستقراً (~1 إلى ~50) عبر كل الطبقات الـ114 بدل
    الانهيار لصفر عملياً."""

    def __init__(self, d_in: int, d_out: int, layer_scale_init: float = 1.0):
        super().__init__()
        self.d_in = int(d_in)
        self.d_out = int(d_out)
        self.fc = nn.Linear(d_in, d_out)
        # السلسلة الوسطى RMSNorm أيضًا (تحقيقًا لطلب استقرار التدريب
        # عبر كامل الشبكة — السلسلة نفسها لا تتغير بنيتها 114 طبقة)
        self.ln = RMSNorm(d_out)
        self.shortcut = nn.Linear(d_in, d_out, bias=False) if d_in != d_out else None
        if self.shortcut is not None:
            self._init_shortcut_identity()
        self.gate = nn.Linear(d_in, d_out)
        self.layer_scale = nn.Parameter(torch.ones(d_out) * layer_scale_init)

    def _init_shortcut_identity(self) -> None:
        """يهيّئ self.shortcut.weight إلى مصفوفة شبه-وحدة (identity-like) بدل
        عشوائية بالكامل، للحفاظ على مقياس الإشارة عبر السلسلة الطويلة."""
        with torch.no_grad():
            self.shortcut.weight.zero_()
            eye = min(self.d_in, self.d_out)
            self.shortcut.weight[:eye, :eye] = torch.eye(
                eye, device=self.shortcut.weight.device, dtype=self.shortcut.weight.dtype
            )

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

        new_ln = RMSNorm(new_out).to(device=device, dtype=dtype)
        with torch.no_grad():
            new_ln.weight[:old_out] = self.ln.weight
            new_ln.weight[old_out:].fill_(1.0)
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

    def reinit_shortcuts_identity(self) -> None:
        """يعيد تهيئة كل مصفوفات shortcut للسلسلة identity-like. ضروري لأن
        SurahChainLM.__init__ يستدعي self.apply(self._init_weights) على كامل
        النموذج بعد بناء السلسلة، وهو يُعيد كتابة shortcut.weight بتهيئة
        عشوائية normal(0, 0.02) فتُلغي تهيئة identity الأصلية. يُستدعى بعد
        self.apply() تماماً كما تُعاد تهيئة بوابات الانتباه هناك."""
        for layer in self.layers:
            if layer.shortcut is not None:
                layer._init_shortcut_identity()

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
    """
    انتباه سببي عبر SDPA مع تحسينات اختيارية:
      - QK-Norm: تطبيع Q و K لكل رأس (استقرار) — RMSNorm محليّ
      - Gated Attention (NeurIPS 2025): بوابة sigmoid بعد SDPA لكل رأس
      - GQA (Grouped-Query Attention): n_kv_heads رؤوس KV مشتركة بين
        مجموعات رؤوس الاستعلام — 8 KV heads × 128/رأس مع 64 رأس استعلام
        يوفر ضغط KV كبيرًا دون فقدان الجودة.
    لا تمس سلسلة السور — فقط مسار الانتباه.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        n_kv_heads: Optional[int] = None,
        d_head: Optional[int] = None,
    ):
        super().__init__()
        # GQA: d_head صريح (مثلاً 128) ثم d_model = n_heads × d_head.
        self.n_heads = n_heads
        self.d_head = int(d_head) if d_head else d_model // n_heads
        assert d_model == n_heads * self.d_head, (
            f"d_model ({d_model}) != n_heads×d_head ({n_heads}×{self.d_head})"
        )
        # رؤوس KV: افتراضي = n_heads (MHA)؛ للنماذج الكبيرة GQA
        # (مثلاً n_heads=64 و n_kv_heads=8).
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads else n_heads
        assert n_heads % self.n_kv_heads == 0
        self.gqa = self.n_kv_heads != n_heads
        self.n_groups = n_heads // self.n_kv_heads
        self.use_qk_norm = use_qk_norm
        self.use_gated_attn = use_gated_attn
        self.q_proj = nn.Linear(d_model, n_heads * self.d_head, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.d_head, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.resid_drop = nn.Dropout(dropout)

        if use_qk_norm:
            self.q_norm = RMSNorm(self.d_head)
            self.k_norm = RMSNorm(self.d_head)

        if use_gated_attn:
            # bias=+2 ⇒ sigmoid≈0.88 عند البداية (قريب من السلوك القديم)
            self.attn_gate = nn.Linear(d_model, n_heads, bias=True)
            nn.init.zeros_(self.attn_gate.weight)
            nn.init.constant_(self.attn_gate.bias, 2.0)

    def _repeat_kv(self, t: torch.Tensor) -> torch.Tensor:
        """يكرر KV heads: (B, K, S, dh) → (B, n_heads, S, dh) عبر البث."""
        if not self.gqa:
            return t
        return t.repeat_interleave(self.n_groups, dim=1)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S, D = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        # TPU/bf16: RMSNorm قد يُرجع float32 بينما v يبقى bfloat16 — SDPA يرفض ذلك
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        k_r, v_r = self._repeat_kv(k), self._repeat_kv(v)

        drop_p = self.dropout if self.training else 0.0
        if key_padding_mask is not None:
            pad = key_padding_mask[:, None, None, :]
            causal = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            full = causal[None, None, :, :]
            bias = torch.zeros(B, 1, S, S, device=x.device, dtype=q.dtype)
            bias = bias.masked_fill(full, float("-inf"))
            bias = bias.masked_fill(pad.expand(B, 1, S, S), float("-inf"))
            y = F.scaled_dot_product_attention(q, k_r, v_r, attn_mask=bias, dropout_p=drop_p)
        else:
            y = F.scaled_dot_product_attention(q, k_r, v_r, is_causal=True, dropout_p=drop_p)

        # y: (B, n_heads, S, dh)
        if self.use_gated_attn:
            g = torch.sigmoid(self.attn_gate(x))  # (B, S, H)
            g = g.permute(0, 2, 1).unsqueeze(-1)  # (B, H, S, 1)
            y = y * g

        y = y.transpose(1, 2).contiguous().reshape(B, S, D)
        return self.resid_drop(self.proj(y))

    def forward_incremental(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        مسار توليد تدريجي (KV-cache) — يُستخدم فقط أثناء الاستدلال (eval,
        بلا dropout). x يحمل فقط الرمز/الرموز الجديدة (S عادة 1)؛ past_kv
        يحمل مفاتيح/قيم n_kv_heads كل الرموز السابقة فتُلحَق بها الجديدة
        بدل إعادة حسابها (في GQA يخزن KV بأبعاد n_kv_heads فقط).
        يُرجع (المخرج، (k الكاملة، v الكاملة)) ليُمرَّر past_kv للخطوة التالية.
        """
        B, S, D = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).reshape(B, S, self.n_kv_heads, self.d_head).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        if q.dtype != v.dtype or k.dtype != v.dtype:
            q = q.to(dtype=v.dtype)
            k = k.to(dtype=v.dtype)

        if past_kv is not None:
            past_k, past_v = past_kv
            k_full = torch.cat([past_k, k], dim=2)
            v_full = torch.cat([past_v, v], dim=2)
        else:
            k_full, v_full = k, v

        k_r, v_r = self._repeat_kv(k_full), self._repeat_kv(v_full)

        T_past = k_full.shape[2] - S
        if T_past > 0:
            # الاستعلامات الجديدة تنتبه لكل الماضي بحرّية + سببياً فيما بينها
            causal_new = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
            allow_past = torch.zeros(S, T_past, device=x.device, dtype=torch.bool)
            full_mask = torch.cat([allow_past, causal_new], dim=1)
            attn_bias = torch.zeros(S, T_past + S, device=x.device, dtype=q.dtype)
            attn_bias = attn_bias.masked_fill(full_mask, float("-inf"))
            y = F.scaled_dot_product_attention(q, k_r, v_r, attn_mask=attn_bias, dropout_p=0.0)
        else:
            y = F.scaled_dot_product_attention(q, k_r, v_r, is_causal=True, dropout_p=0.0)

        if self.use_gated_attn:
            g = torch.sigmoid(self.attn_gate(x))
            g = g.permute(0, 2, 1).unsqueeze(-1)
            y = y * g

        y = y.transpose(1, 2).contiguous().reshape(B, S, D)
        out = self.resid_drop(self.proj(y))
        return out, (k_full, v_full)


class SwiGLUFFN(nn.Module):
    """Feed-Forward بنظام SwiGLU (مثل نماذج Llama/Mistral):
    FFN(x) = Drop(SwiGLU(W_up·x) ⊙ W_gate·x)·W_down.
    w_up/w_gate: (d_ff × d_model)، w_down: (d_model × d_ff).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.w_down(F.silu(self.w_up(x)) * self.w_gate(x)))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        n_kv_heads: Optional[int] = None,
        d_head: Optional[int] = None,
    ):
        super().__init__()
        # Pre-Norm: RMSNorm بدل LayerNorm وتحديثه قبل كل طبقة
        self.ln1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model,
            n_heads,
            dropout,
            use_qk_norm=use_qk_norm,
            use_gated_attn=use_gated_attn,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
        )
        self.ln2 = RMSNorm(d_model)
        # SwiGLU بدل GELU (استقرار أكبر وممارسات النماذج الكبيرة)
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout=dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.ln2(x))
        return x

    def forward_incremental(
        self,
        x: torch.Tensor,
        past_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv = self.attn.forward_incremental(self.ln1(x), past_kv=past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ln2(x))
        return x, kv


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
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        chain_scale: float = 1.0,
        n_kv_heads: Optional[int] = None,
        d_head: Optional[int] = None,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        # النوازل المعمارية الجديدة (نمط النماذج الكبيرة):
        # - d_head صريح (مثلاً 128) بحيث d_model = n_heads × d_head
        # - n_kv_heads رؤوس KV مشتركة (GQA: 8 KV لكل 64 رأس استعلام)
        # - d_ff صريح (28672 = 3.5 × d_model)؛ الافتراضي 4×d_model
        # - RMSNorm Pre-Norm عبر كامل الشبكة
        # - سلسلة السور الـ114 تبقى كما هي (التوسع أثناء التدريب)
        if d_head:
            d_head = int(d_head)
            # d_model مشتق من n_heads × d_head إن لم يتوافق الدخل
            d_model = n_heads * d_head
        if d_model % n_heads != 0:
            d_model = (d_model // n_heads) * n_heads
        self.d_model = d_model
        self.d_head = d_head if d_head else d_model // n_heads
        self.n_heads = n_heads
        self.n_kv_heads = int(n_kv_heads) if n_kv_heads else self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        self.d_ff = int(d_ff) if d_ff else self.d_model * 4
        self.vocab_size = vocab_size
        self.max_seq = max_seq
        self.n_pre = n_pre
        self.n_post = n_post
        self.use_qk_norm = use_qk_norm
        self.use_gated_attn = use_gated_attn
        self.chain_scale = float(chain_scale) if chain_scale else 1.0

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_seq, d_model)
        self.drop = nn.Dropout(dropout)
        self.ln_in = RMSNorm(d_model)

        self.pre_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    self.d_ff,
                    dropout,
                    use_qk_norm=use_qk_norm,
                    use_gated_attn=use_gated_attn,
                    n_kv_heads=self.n_kv_heads,
                    d_head=self.d_head,
                )
                for _ in range(n_pre)
            ]
        )
        scaled_dims = scale_surah_dims(LAYER_DIMS, self.chain_scale)
        self.chain_width = int(scaled_dims[0][0])
        self.chain = SurahChainNetwork(scaled_dims)
        self.W_in = nn.Linear(d_model, self.chain_width)
        self.W_out = nn.Linear(self.chain_width, d_model)
        self.W_skip = nn.Linear(d_model, d_model, bias=False)
        self.post_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    self.d_ff,
                    dropout,
                    use_qk_norm=use_qk_norm,
                    use_gated_attn=use_gated_attn,
                    n_kv_heads=self.n_kv_heads,
                    d_head=self.d_head,
                )
                for _ in range(n_post)
            ]
        )
        self.ln_f = RMSNorm(d_model)
        self.apply(self._init_weights)
        # أعد تهيئة بوابات الانتباه بعد apply (حتى لا يصفّر _init_weights الـbias)
        if use_gated_attn:
            for blk in list(self.pre_blocks) + list(self.post_blocks):
                gate = getattr(blk.attn, "attn_gate", None)
                if gate is not None:
                    nn.init.zeros_(gate.weight)
                    nn.init.constant_(gate.bias, 2.0)
        # تهيئة SwiGLU القياسية للنماذج الكبيرة: w_up/w_gate ~N(0,√(2/5d_model))
        # w_down يبدأ بأصفار لتقليل اضطراب التدرج في بداية التدريب
        for blk in list(self.pre_blocks) + list(self.post_blocks):
            if hasattr(blk.ffn, "w_down"):
                nn.init.zeros_(blk.ffn.w_down.weight)
        # أعد تهيئة shortcut سلسلة السور identity-like بعد apply (نفس السبب:
        # self.apply(self._init_weights) يستبدل تهيئة identity الأصلية
        # بتهيئة عشوائية normal(0, 0.02) — راجع SurahChainLayer._init_shortcut_identity
        # وتوثيق المشكلة/الإصلاح في SurahChainLayer).
        self.chain.reinit_shortcuts_identity()

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

        h_c = self.W_in(x)
        flat = h_c.reshape(B * S, self.chain_width)
        flat = self.chain(flat)
        h_chain = self.W_out(flat).reshape(B, S, self.d_model)
        x = h_chain + self.W_skip(x)

        for blk in self.post_blocks:
            x = blk(x, key_padding_mask=key_padding_mask)
        x = self.ln_f(x)
        return F.linear(x, self.tok_emb.weight)

    def forward_step(
        self,
        idx_new: torch.Tensor,
        pos_start: int,
        cache: Optional[dict] = None,
    ):
        """
        خطوة توليد تدريجية (KV-cache) — تُعالج فقط الرموز الجديدة idx_new
        (S_new=1 عادة أثناء فك الترميز، أو طول الـprompt عند أول استدعاء)
        بدل إعادة تشغيل التسلسل الكامل. سلسلة السور الـ114 حساب مستقل لكل
        موضع (لا تفاعل بين المواضع)، لذا لا حاجة إطلاقاً لإعادة حسابها
        للرموز القديمة — فقط الانتباه (pre/post_blocks) يحتاج KV-cache
        لأنه الجزء الوحيد الذي "يرى" مواضع أخرى.
        idx_new: (B, S_new). pos_start: الموضع المطلق للرمز الأول في idx_new.
        cache: None عند أول استدعاء، أو dict مُرجَع من نداء سابق.
        يُرجع (logits (B, S_new, vocab), cache جديد).
        """
        B, S_new = idx_new.shape
        pos = torch.arange(pos_start, pos_start + S_new, device=idx_new.device).unsqueeze(0).expand(B, -1)
        x = self.drop(self.tok_emb(idx_new) + self.pos_emb(pos))
        x = self.ln_in(x)

        pre_kv_in = (cache or {}).get("pre_kv") or [None] * len(self.pre_blocks)
        new_pre_kv = []
        for blk, past in zip(self.pre_blocks, pre_kv_in):
            x, kv = blk.forward_incremental(x, past_kv=past)
            new_pre_kv.append(kv)

        h_c = self.W_in(x)
        flat = h_c.reshape(B * S_new, self.chain_width)
        flat = self.chain(flat)
        h_chain = self.W_out(flat).reshape(B, S_new, self.d_model)
        x = h_chain + self.W_skip(x)

        post_kv_in = (cache or {}).get("post_kv") or [None] * len(self.post_blocks)
        new_post_kv = []
        for blk, past in zip(self.post_blocks, post_kv_in):
            x, kv = blk.forward_incremental(x, past_kv=past)
            new_post_kv.append(kv)

        x = self.ln_f(x)
        logits = F.linear(x, self.tok_emb.weight)
        return logits, {"pre_kv": new_pre_kv, "post_kv": new_post_kv}

    def param_count(self) -> dict:
        return {
            "total": sum(p.numel() for p in self.parameters()),
            "chain": sum(p.numel() for p in self.chain.parameters()),
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "chain_width": getattr(self, "chain_width", CHAIN_WIDTH),
            "chain_scale": getattr(self, "chain_scale", 1.0),
            "use_qk_norm": getattr(self, "use_qk_norm", False),
            "use_gated_attn": getattr(self, "use_gated_attn", False),
            "n_kv_heads": getattr(self, "n_kv_heads", self.n_heads),
            "d_head": getattr(self, "d_head", self.d_model // self.n_heads),
            "d_ff": getattr(self, "d_ff", self.d_model * 4),
        }



class ExpandPlateauController:
    """توسيع الأضيق فقط بعد هضبة خسارة طويلة (عدة عصور)، وليس بعد عصر أو اثنين.

    الافتراضات صارمة عمداً:
      patience >= 20 عصراً بلا تحسّن حقيقي
      min_epoch قبل أي توسيع
      cooldown بين توسيعين
      نافذة loss شبه مستوية (flat) + عدم اتجاه هبوط واضح
    """

    def __init__(
        self,
        patience: Optional[int] = None,
        max_expands: Optional[int] = None,
        min_epoch: Optional[int] = None,
        cooldown: Optional[int] = None,
        warmup_run: Optional[int] = None,
        flat_rel: Optional[float] = None,
        improve_eps: float = 1e-5,
    ) -> None:
        self.patience = int(patience if patience is not None else os.environ.get("SCN_EXPAND_PATIENCE", "20"))
        self.max_expands = int(max_expands if max_expands is not None else os.environ.get("SCN_MAX_EXPANDS", "5"))
        self.min_epoch = int(min_epoch if min_epoch is not None else os.environ.get("SCN_EXPAND_MIN_EPOCH", "25"))
        self.cooldown = int(cooldown if cooldown is not None else os.environ.get("SCN_EXPAND_COOLDOWN", "15"))
        self.warmup_run = int(warmup_run if warmup_run is not None else os.environ.get("SCN_EXPAND_WARMUP_RUN", "8"))
        self.flat_rel = float(flat_rel if flat_rel is not None else os.environ.get("SCN_EXPAND_FLAT_REL", "0.01"))
        self.improve_eps = float(improve_eps)
        self.best = float("inf")
        self.no_improve = 0
        self.n_expands = 0
        self.expand_log: List[dict] = []
        self.history: List[float] = []
        self.start_epoch = 0

    def state_dict(self) -> dict:
        return {
            "best": self.best,
            "no_improve": self.no_improve,
            "n_expands": self.n_expands,
            "expand_log": list(self.expand_log),
            "history": list(self.history[-200:]),
            "start_epoch": self.start_epoch,
        }

    def load_state_dict(self, d: dict) -> None:
        if not d:
            return
        self.best = float(d.get("best", self.best))
        self.no_improve = int(d.get("no_improve", 0))
        self.n_expands = int(d.get("n_expands", 0))
        self.expand_log = list(d.get("expand_log") or [])
        self.history = list(d.get("history") or [])
        self.start_epoch = int(d.get("start_epoch", 0))

    def on_epoch_end(self, model: "HybridExperimentModelTorch", epoch: int, epoch_loss: float) -> Optional[dict]:
        """يُستدعى في نهاية كل عصر. يعيد info التوسيع إن حصل، وإلا None."""
        if epoch_loss != epoch_loss:  # NaN
            return None
        self.history.append(float(epoch_loss))
        improved = epoch_loss < self.best - self.improve_eps
        if improved:
            self.best = float(epoch_loss)
            self.no_improve = 0
            return None
        self.no_improve += 1

        epochs_into_run = epoch - self.start_epoch
        last_expand_ep = self.expand_log[-1]["epoch"] if self.expand_log else -10**9
        since_expand = epoch - last_expand_ep
        window = self.history[-self.patience :]
        flat = False
        if len(window) >= max(5, self.patience // 2):
            w_mean = sum(window) / len(window)
            w_span = max(window) - min(window)
            flat = w_mean > 0 and (w_span / w_mean) <= self.flat_rel
        weak_trend = False
        if len(window) >= 5:
            # لم ينخفض بوضوح من أول النافذة لآخرها
            weak_trend = window[-1] > window[0] - 1e-3

        can = (
            self.n_expands < self.max_expands
            and self.no_improve >= self.patience
            and epoch >= self.min_epoch
            and epochs_into_run >= self.warmup_run
            and since_expand >= self.cooldown
            and flat
            and weak_trend
        )
        if not can:
            if self.no_improve >= self.patience and self.n_expands < self.max_expands:
                reasons = []
                if epoch < self.min_epoch:
                    reasons.append(f"min_epoch({self.min_epoch})")
                if epochs_into_run < self.warmup_run:
                    reasons.append(f"run_warmup({self.warmup_run})")
                if since_expand < self.cooldown:
                    reasons.append(f"cooldown({self.cooldown})")
                if not flat:
                    reasons.append("not_flat")
                if not weak_trend:
                    reasons.append("still_trending_down")
                if reasons and (self.no_improve == self.patience or self.no_improve % 10 == 0):
                    print(f"  · لا توسيع بعد: {', '.join(reasons)}", flush=True)
            return None

        info = model.expand_narrowest(delta=1)
        if info is None:
            return None
        self.n_expands += 1
        self.no_improve = 0
        rec = {"epoch": epoch, **info}
        self.expand_log.append(rec)
        print(
            f"  → توسيع مدروس #{self.n_expands}: طبقة {info['layer_idx']} "
            f"{info['old']} → out={info['new_out']} next_in={info['next_new_in']} "
            f"(patience={self.patience}, flat_ok)",
            flush=True,
        )
        return rec


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
        use_qk_norm: bool = True,
        use_gated_attn: bool = True,
        chain_scale: float = 1.0,
        n_kv_heads: Optional[int] = None,
        d_head: Optional[int] = None,
        d_ff: Optional[int] = None,
    ):
        # دعم TPU عبر SCN_TPU=1 (torch_xla) — يُفضَّل على CUDA/CPU إن وُجد
        if device is None and get_xla_device() is not None:
            device = get_xla_device()
        self.device = torch.device(device) if device else get_device()
        self.base_lr = lr
        self.lr = lr
        self.vocab_size = vocab_size
        self.use_qk_norm = use_qk_norm
        self.use_gated_attn = use_gated_attn
        self.chain_scale = float(chain_scale) if chain_scale else 1.0
        self.n_kv_heads = n_kv_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.tokenizer = StrongTokenizer(vocab_size)
        self.model = SurahChainLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_heads=n_heads,
            n_pre=n_pre,
            n_post=n_post,
            max_seq=max_seq,
            use_qk_norm=use_qk_norm,
            use_gated_attn=use_gated_attn,
            chain_scale=self.chain_scale,
            n_kv_heads=n_kv_heads,
            d_head=d_head,
            d_ff=d_ff,
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
                use_qk_norm=getattr(core, "use_qk_norm", self.use_qk_norm),
                use_gated_attn=getattr(core, "use_gated_attn", self.use_gated_attn),
                chain_scale=getattr(core, "chain_scale", self.chain_scale),
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

    def _encode_batch(self, texts, max_len: int):
        """يحضّر tensors: input_ids, labels, pad_mask (True=PAD).

        تدعم وضعين متوافقين:
        - Sequence[str]: تشفير حي (الوضع القديم)
        - Sequence[Sequence[int]]: تسلسلات مشفرة مسبقًا (pre-tokenized):
          تشفير كامل البيانات مرة واحدة في البداية ثم إعادة استخدام tensors
          جاهزة في كل خطوة — تسريع 2-4× على CPU (إزالة tokenize الحي).
        """
        seqs = []
        for t in texts:
            if isinstance(t, str):
                ids = self.tokenizer.encode(t, max_len)
            else:
                # تسلسل مشفّر مسبقًا — قصّه إلى max_len فقط
                ids = list(t)[:max_len]
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
        accum: bool = False,
    ) -> float:
        """دفعة واحدة حقيقية (مصفوفة B×S) — أسرع بكثير من جملة بجملة.

        accum=True: يجمّع التدرجات بدل خطوة محسّن فورية (gradient
        accumulation) — تُستدعى accum_step() بعد عدد GRAD_ACCUM استدعاءات.
        """
        if step is not None and total_steps is not None:
            self.set_lr(cosine_lr(step, total_steps, self.base_lr, warmup_steps))
        packed = self._encode_batch(texts, max_len)
        if packed is None:
            return float("nan")
        x, y, pad_mask = packed
        self.model.train()
        if not accum:
            self.opt.zero_grad(set_to_none=True)
        _is_xla = getattr(self.device, "type", "") == "xla"
        if _is_xla:
            # TPU: bfloat16 افتراضيًا — أسرع وأخف على الذاكرة (v5e أبطأ في FP32)
            autocast = torch.autocast(device_type="xla", dtype=xla_dtype(), enabled=True)
        else:
            autocast = contextlib.nullcontext()
        with autocast:
            logits = self.model(x, key_padding_mask=pad_mask)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                y.reshape(-1),
                ignore_index=-100,
            )
        # مقسوم على عدد micro-batches المتراكمة لإبقاء مقياس التدرج ثابتًا
        scale = max(1, getattr(self, "_accum_scale", 1))
        (loss / scale).backward()
        if not accum:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            xla_mark_step(self.device)
        return float(loss.item())

    def accum_step(self) -> None:
        """يُنهي تجميع التدرجات: clip → optimizer step → zero_grad → mark_step (TPU)."""
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        xla_mark_step(self.device)

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

    @torch.no_grad()
    def generate_fast(
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
        """
        نفس generate() منطقياً (نفس التوزيع الاحتمالي عند كل خطوة) لكن عبر
        KV-cache تدريجي بدل إعادة تشغيل التسلسل الكامل عبر السلسلة الـ114
        + الانتباه في كل خطوة. أسرع بشكل كبير خصوصاً مع نصوص مولَّدة طويلة
        (زمن كل خطوة يبقى شبه ثابت بدل التزايد مع طول السياق).
        """
        self.model.eval()
        core = getattr(self.model, "_orig_mod", self.model)
        max_ctx = min(max_ctx, core.max_seq)
        ids = list(self.tokenizer.encode(prompt, max_ctx))
        eos, bos, pad = self.tokenizer.EOS, self.tokenizer.BOS, self.tokenizer.PAD
        if ids and ids[-1] == eos:
            ids = ids[:-1]
        if not ids:
            ids = [bos]
        ids = ids[-max_ctx:]

        # تعبئة أولى (prefill) بكامل الـprompt دفعة واحدة
        prefill = torch.tensor([ids], dtype=torch.long, device=self.device)
        logits, cache = core.forward_step(prefill, pos_start=0, cache=None)
        pos = len(ids)
        next_logits = logits[0, -1].float()

        for step_i in range(max_new_tokens):
            logits_i = next_logits.clone()
            logits_i[pad] = -1e9
            logits_i[bos] = -1e9
            if step_i < min_new_tokens:
                logits_i[eos] = -1e9
            if repetition_penalty != 1.0:
                for prev in set(ids):
                    if logits_i[prev] > 0:
                        logits_i[prev] /= repetition_penalty
                    else:
                        logits_i[prev] *= repetition_penalty
            if top_k and top_k > 0:
                v, _ = torch.topk(logits_i, min(top_k, logits_i.size(0)))
                logits_i[logits_i < v[-1]] = -float("inf")
            logits_i = logits_i / max(temperature, 1e-6)
            probs = F.softmax(logits_i, dim=-1)
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
            if pos >= max_ctx - 1:
                break
            step_in = torch.tensor([[nid]], dtype=torch.long, device=self.device)
            logits, cache = core.forward_step(step_in, pos_start=pos, cache=cache)
            pos += 1
            next_logits = logits[0, -1].float()
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

    def save(self, path: str, train_meta: Optional[dict] = None) -> None:
        """حفظ النموذج + الـoptimizer + بيانات الاستكمال (resume)."""
        core = getattr(self.model, "_orig_mod", self.model)
        payload = {
            "model": core.state_dict(),
            "optimizer": self.opt.state_dict(),
            "vocab_size": self.vocab_size,
            "d_model": core.d_model,
            "n_heads": core.n_heads,
            "n_pre": core.n_pre,
            "n_post": core.n_post,
            "max_seq": getattr(core, "max_seq", DEFAULT_MAX_CTX),
            "use_qk_norm": bool(getattr(core, "use_qk_norm", True)),
            "use_gated_attn": bool(getattr(core, "use_gated_attn", True)),
            "chain_scale": float(getattr(core, "chain_scale", 1.0)),
            "n_kv_heads": int(getattr(core, "n_kv_heads", core.n_heads)),
            "d_head": int(getattr(core, "d_head", core.d_model // core.n_heads)),
            "d_ff": int(getattr(core, "d_ff", core.d_model * 4)),
            "lr": self.lr,
            "tokenizer": {
                "word_to_id": self.tokenizer.word_to_id,
                "merges": self.tokenizer.merges,
                "vocab_size": self.tokenizer.vocab_size,
            },
        }
        if train_meta:
            payload["train_meta"] = train_meta
        torch.save(payload, path)

    def load(self, path: str, load_optimizer: bool = True) -> dict:
        """
        تحميل checkpoint. يعيد train_meta إن وُجدت.
        إن اختلفت أبعاد النموذج في الـcheckpoint عن الحالي، يُعاد بناء النموذج.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.vocab_size = int(ckpt.get("vocab_size", self.vocab_size))
        tok = ckpt.get("tokenizer")
        if tok:
            self.tokenizer.word_to_id = {str(k): int(v) for k, v in tok["word_to_id"].items()}
            self.tokenizer.id_to_word = {int(v): str(k) for k, v in self.tokenizer.word_to_id.items()}
            self.tokenizer.merges = [tuple(x) for x in tok.get("merges", [])]
            self.tokenizer.vocab_size = int(tok.get("vocab_size", self.vocab_size))

        want_d = int(ckpt.get("d_model", self.model.d_model if hasattr(self.model, "d_model") else DEFAULT_D_MODEL))
        want_h = int(ckpt.get("n_heads", DEFAULT_N_HEADS))
        want_pre = int(ckpt.get("n_pre", DEFAULT_N_PRE))
        want_post = int(ckpt.get("n_post", DEFAULT_N_POST))
        want_max = int(ckpt.get("max_seq", DEFAULT_MAX_CTX))
        # GQA: ckpt قديم (MHA/FFN GELU) لا يحمل n_kv_heads/d_head/d_ff —
        # نشتقها: n_kv_heads=n_heads، d_head=d_model//n_heads، d_ff=4×d_model
        # (التوافق العكسي يبقي checkpoints القديمة MHA كما هي)
        if "n_kv_heads" in ckpt:
            want_kvh = int(ckpt["n_kv_heads"])
            want_dh = int(ckpt.get("d_head", want_d // want_h))
            want_ff = int(ckpt.get("d_ff", want_d * 4))
        else:
            want_kvh = want_h
            want_dh = want_d // want_h
            want_ff = want_d * 4
        sd = ckpt.get("model") or {}
        # اكتشاف الميزات من الـcheckpoint (توافق مع أوزان قديمة بدون QK-Norm/Gated)
        if "use_qk_norm" in ckpt:
            want_qk = bool(ckpt["use_qk_norm"])
        else:
            want_qk = any("q_norm" in k for k in sd)
        if "use_gated_attn" in ckpt:
            want_gate = bool(ckpt["use_gated_attn"])
        else:
            want_gate = any("attn_gate" in k for k in sd)
        # إن طُلبت الميزات الجديدة عبر الكائن الحالي رغم غيابها في ckpt → ترقية جزئية
        if self.use_qk_norm and not want_qk:
            want_qk = True
        if self.use_gated_attn and not want_gate:
            want_gate = True
        want_scale = float(ckpt.get("chain_scale", getattr(self, "chain_scale", 1.0)))
        self.use_qk_norm = want_qk
        self.use_gated_attn = want_gate
        self.chain_scale = want_scale
        self.n_kv_heads = want_kvh
        self.d_head = want_dh
        self.d_ff = want_ff

        core = getattr(self.model, "_orig_mod", self.model)
        need_rebuild = (
            getattr(core, "d_model", None) != want_d
            or getattr(core, "n_heads", None) != want_h
            or getattr(core, "n_pre", None) != want_pre
            or getattr(core, "n_post", None) != want_post
            or core.tok_emb.num_embeddings < self.vocab_size
            or bool(getattr(core, "use_qk_norm", False)) != want_qk
            or bool(getattr(core, "use_gated_attn", False)) != want_gate
            or abs(float(getattr(core, "chain_scale", 1.0)) - want_scale) > 1e-6
            or int(getattr(core, "n_kv_heads", core.n_heads)) != want_kvh
            or int(getattr(core, "d_head", core.d_model // core.n_heads)) != want_dh
            or int(getattr(core, "d_ff", core.d_model * 4)) != want_ff
        )
        if need_rebuild:
            self.model = SurahChainLM(
                vocab_size=max(self.vocab_size, int(ckpt.get("vocab_size", 8192))),
                d_model=want_d,
                n_heads=want_h,
                n_pre=want_pre,
                n_post=want_post,
                max_seq=want_max,
                use_qk_norm=want_qk,
                use_gated_attn=want_gate,
                chain_scale=want_scale,
                n_kv_heads=want_kvh,
                d_head=want_dh,
                d_ff=want_ff,
            ).to(self.device)
            self._compiled = False
            self.opt = torch.optim.AdamW(
                self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
            )
            core = self.model

        # تحميل الأوزان (جزئي مقبول عند ترقية QK-Norm/Gated أو توسيع السلسلة)
        try:
            core.load_state_dict(ckpt["model"], strict=True)
        except RuntimeError:
            missing, unexpected = core.load_state_dict(ckpt["model"], strict=False)
            print(
                f"تحذير: تحميل جزئي (missing={len(missing)}, unexpected={len(unexpected)}) "
                f"— طبيعي عند تفعيل QK-Norm/Gated Attention لأول مرة"
            )
        core.to(self.device)

        if load_optimizer and "optimizer" in ckpt:
            try:
                self.opt.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                print(f"تحذير: تعذّر تحميل حالة الـoptimizer ({e}) — يُتابع بأوزان فقط")
                self.opt = torch.optim.AdamW(
                    self.model.parameters(), lr=self.lr, weight_decay=0.01, betas=(0.9, 0.95)
                )
        if "lr" in ckpt:
            self.set_lr(float(ckpt["lr"]))
        return ckpt.get("train_meta") or {}


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
