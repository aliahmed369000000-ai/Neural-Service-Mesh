"""
train_yemeni.py
================
تدريب LoRA (Parameter-Efficient Fine-Tuning) لـ YemeniDecoder على مجموعة
تعليمات يمنية (data/yemeni_instructions.json).

⚠️ ملاحظات هندسية مهمة (اقرأها قبل التشغيل):
  1. YemeniDecoder شبكة ~1 مليار معامل (D_MODEL=2304×16 طبقة×16 رأس).
     بصيغة fp32 هذا ≈ 4GB فقط للأوزان الأساسية، قبل أي gradients/activations.
     يتجاوز حد ذاكرة Streamlit Community Cloud (~1GB) بوضوح — هذا السكربت
     مصمم للتشغيل على بيئة بذاكرة كافية (Replit بخطة مدفوعة، أو جهاز محلي)،
     وليس على Streamlit Cloud نفسها.
  2. LoRA هنا PyTorch أصلي (LoRALinear بالأسفل) — غير متوافق مع
     ai/lora_adapter.py (ذاك NumPy بحت، مصمم لـ ArabicTransformer القديم).
  3. لا يوجد أي checkpoint أساسي مُدرَّب مسبقاً لـ YemeniDecoder. هذا السكربت
     يبدأ من أوزان عشوائية + LoRA فوقها + 30 مثال فقط → النتيجة تثبت أن خط
     الأنابيب (tokenize → forward → loss → backward → save) يعمل صحيحاً
     تقنياً، وليست نموذجاً جاهزاً لإجابات يمنية طليقة. ذلك يتطلب لاحقاً
     تدريباً أساسياً (pretraining) على كمّية بيانات أكبر بكثير.

الاستخدام:
    python train_yemeni.py
    # متغيرات بيئة اختيارية:
    YEMENI_EPOCHS=5 YEMENI_LR=1e-3 YEMENI_LORA_RANK=8 python train_yemeni.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_yemeni")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    logger.error(
        "PyTorch غير مثبَّت. ثبّته أولاً: pip install torch --break-system-packages "
        "(أو بدون --break-system-packages حسب بيئتك). لن يعمل هذا السكربت بدونه."
    )
    sys.exit(1)

from ai.arabic_transformer import get_yemeni_decoder, YemeniGQAAttention
from ai.yemeni_tokenizer import get_yemeni_tokenizer

# ═══════════════════════════════════════════════════════════════════════════
# إعدادات قابلة للتخصيص عبر متغيرات بيئة
# ═══════════════════════════════════════════════════════════════════════════
DATA_PATH     = os.environ.get("YEMENI_DATA_PATH", "data/yemeni_instructions.json")
WEIGHTS_DIR   = os.environ.get("YEMENI_WEIGHTS_DIR", "models/yemeni_decoder")
LOSS_LOG_PATH = os.environ.get("YEMENI_LOSS_LOG", "memory/training_loss.json")
EPOCHS        = int(os.environ.get("YEMENI_EPOCHS", "4"))
LR            = float(os.environ.get("YEMENI_LR", "1e-3"))
LORA_RANK     = int(os.environ.get("YEMENI_LORA_RANK", "8"))
LORA_ALPHA    = float(os.environ.get("YEMENI_LORA_ALPHA", "16.0"))
MAX_LEN       = int(os.environ.get("YEMENI_MAX_LEN", "96"))
BATCH_SIZE    = int(os.environ.get("YEMENI_BATCH_SIZE", "4"))
PAD_ID        = 0


# ═══════════════════════════════════════════════════════════════════════════
# 1) LoRA أصلي بـ PyTorch — يستهدف Wq/Wv فقط داخل كل YemeniGQAAttention
# ═══════════════════════════════════════════════════════════════════════════
class LoRALinear(nn.Module):
    """
    يُغلّف nn.Linear مجمَّدة بمصفوفتين منخفضتَي الرتبة (A, B) قابلتَين
    للتدريب فقط. الإخراج = base(x) + (alpha/rank) * B(A(x)).
    عدد المعاملات القابلة للتدريب لكل طبقة = rank × (in+out)، صغير جداً
    مقارنة بـ in×out الكاملة.
    """

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        in_f, out_f = base.in_features, base.out_features
        self.rank  = rank
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, in_f) * (1.0 / rank ** 0.5))
        self.B = nn.Parameter(torch.zeros(out_f, rank))  # يبدأ من صفر → دلتا أولية = صفر

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        base_out = self.base(x)
        delta = F.linear(F.linear(x, self.A), self.B) * self.scale
        return base_out + delta

    def lora_state_dict(self) -> dict:
        return {"A": self.A.detach().cpu(), "B": self.B.detach().cpu()}


def attach_lora(decoder, rank: int = 8, alpha: float = 16.0) -> List[LoRALinear]:
    """
    يجمّد كل أوزان YemeniDecoder، ثم يستبدل Wq/Wv داخل كل طبقة انتباه
    (YemeniGQAAttention) بـ LoRALinear. يُرجع قائمة الوحدات المُضافة
    (نحتاجها للحفظ لاحقاً).
    """
    for p in decoder.parameters():
        p.requires_grad = False

    lora_modules: List[LoRALinear] = []
    for block in decoder.layers:
        attn = block.attn
        if not isinstance(attn, YemeniGQAAttention):
            continue
        attn.Wq = LoRALinear(attn.Wq, rank=rank, alpha=alpha)
        attn.Wv = LoRALinear(attn.Wv, rank=rank, alpha=alpha)
        lora_modules.append(attn.Wq)
        lora_modules.append(attn.Wv)

    logger.info(f"تم ربط LoRA (rank={rank}, alpha={alpha}) بـ {len(lora_modules)} "
                f"طبقة (Wq+Wv × {len(lora_modules)//2} بلوك)")
    return lora_modules


def save_lora_weights(lora_modules: List[LoRALinear], weights_dir: str):
    """يحفظ فقط أوزان LoRA (A/B) — ليس النموذج الأساسي كاملاً (~1B معامل)."""
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    state = {f"layer_{i}": m.lora_state_dict() for i, m in enumerate(lora_modules)}
    torch.save(state, os.path.join(weights_dir, "lora_adapter.pt"))
    meta = {
        "rank": LORA_RANK, "alpha": LORA_ALPHA,
        "n_lora_layers": len(lora_modules),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(os.path.join(weights_dir, "lora_adapter_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ أوزان LoRA محفوظة في {weights_dir}/lora_adapter.pt")


# ═══════════════════════════════════════════════════════════════════════════
# 2) بناء الدفعات (batches) — ترميز + padding + attention mask
# ═══════════════════════════════════════════════════════════════════════════
def load_dataset(path: str) -> List[dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"تحميل {len(data)} مثال من {path}")
    return data


def build_batches(
    data: List[dict], tokenizer, batch_size: int, max_len: int,
) -> List[Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
    """
    لكل مثال: ندمج instruction + output بنص واحد (سياق تعليمي بسيط)،
    نرمّزه، نبني padding + causal mask، ونعيد دفعات (input_ids, pad_mask, labels).
    labels = input_ids نفسها مُزاحة بخطوة واحدة (next-token prediction قياسي)،
    مع تجاهل مواضع الـPAD في حساب الخسارة (ignore_index=-100).
    """
    sequences: List = []
    for ex in data:
        text = f"{ex['instruction']} </س> {ex['output']}"
        ids = tokenizer.encode(text, max_len=max_len, add_bos=True, add_eos=True)
        sequences.append(ids)

    batches = []
    for i in range(0, len(sequences), batch_size):
        chunk = sequences[i:i + batch_size]
        padded = tokenizer.pad_sequence(chunk, max_len=max_len, pad_id=PAD_ID)  # (B, max_len)
        pad_mask = tokenizer.make_pad_mask(padded, pad_id=PAD_ID)               # (B, max_len)

        input_ids = torch.tensor(padded[:, :-1], dtype=torch.int64)
        labels    = torch.tensor(padded[:, 1:], dtype=torch.int64)
        labels[labels == PAD_ID] = -100  # تجاهل الـPAD بحساب cross_entropy

        attn_mask = torch.tensor(pad_mask[:, :-1], dtype=torch.bool)
        batches.append((input_ids, attn_mask, labels))

    return batches


# ═══════════════════════════════════════════════════════════════════════════
# 3) حلقة التدريب
# ═══════════════════════════════════════════════════════════════════════════
def train():
    tokenizer = get_yemeni_tokenizer()
    decoder = get_yemeni_decoder(weights_dir=WEIGHTS_DIR, load_if_exists=True)
    decoder.train()

    lora_modules = attach_lora(decoder, rank=LORA_RANK, alpha=LORA_ALPHA)
    trainable = [p for m in lora_modules for p in (m.A, m.B)]
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in decoder.parameters())
    logger.info(f"معاملات قابلة للتدريب (LoRA فقط): {n_trainable:,} "
                f"من إجمالي {n_total:,} ({100 * n_trainable / n_total:.3f}%)")

    optimizer = torch.optim.AdamW(trainable, lr=LR)

    data = load_dataset(DATA_PATH)
    batches = build_batches(data, tokenizer, BATCH_SIZE, MAX_LEN)
    logger.info(f"عدد الدفعات لكل epoch: {len(batches)} (batch_size={BATCH_SIZE})")

    loss_history = []
    for epoch in range(1, EPOCHS + 1):
        epoch_losses = []
        t0 = time.time()
        for input_ids, attn_mask, labels in batches:
            optimizer.zero_grad()
            logits = decoder(input_ids, key_padding_mask=attn_mask)  # (B, S, vocab)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        elapsed = time.time() - t0
        loss_history.append({"epoch": epoch, "avg_loss": round(avg_loss, 4), "elapsed_sec": round(elapsed, 1)})
        logger.info(f"[epoch {epoch}/{EPOCHS}] avg_loss={avg_loss:.4f} elapsed={elapsed:.1f}s")

    # حفظ سجل الخسارة لعرضه بلوحة Streamlit
    Path(os.path.dirname(LOSS_LOG_PATH)).mkdir(parents=True, exist_ok=True)
    with open(LOSS_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "model": "YemeniDecoder-LoRA",
            "lora_rank": LORA_RANK,
            "epochs": EPOCHS,
            "history": loss_history,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"✓ سجل الخسارة محفوظ في {LOSS_LOG_PATH}")

    save_lora_weights(lora_modules, WEIGHTS_DIR)
    logger.info("DONE_ALL — التدريب اكتمل.")


if __name__ == "__main__":
    train()
