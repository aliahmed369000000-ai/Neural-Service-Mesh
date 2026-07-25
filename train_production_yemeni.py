"""
train_production_yemeni.py  [production-7b-llm]
=================================================
تدريب LoRA صناعي على Qwen2.5-7B-Instruct (أو أي قاعدة 7B/8B بديلة) على
مجموعة تعليمات يمنية، عبر Hugging Face TRL (SFTTrainer) + PEFT (LoRA) +
bitsandbytes (كمّية 4-bit).

⚠️ متطلبات إلزامية — هذا السكربت لن يعمل على Streamlit Community Cloud أو
أي بيئة CPU فقط:
  - GPU واحد على الأقل بذاكرة ≥24GB (A100 40/80GB أو H100 موصى به لـ7B/8B).
  - pip install torch transformers accelerate peft bitsandbytes trl datasets

منفصل تماماً عن train_yemeni.py (YemeniDecoder ~1B عشوائي القديم) —
لا يستدعيه ولا يعدّله.

الاستخدام (على جهاز GPU سحابي):
    python train_production_yemeni.py \
        --dataset data/yemeni_production_instructions.jsonl \
        --output-dir models/yemeni_qwen7b_lora \
        --epochs 3

متغيرات بيئة اختيارية:
    NSM_PRODUCTION_BASE_MODEL   (افتراضي: Qwen/Qwen2.5-7B-Instruct)
    NSM_PRODUCTION_LOSS_LOG     (افتراضي: memory/production_loss.json)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("train_production_yemeni")

# ══════════════════════════════════════════════════════════════════════════
# فحص مبكر وواضح للاعتماديات — فشل فوري ومفهوم بدل تتبع أخطاء غامض لاحقاً
# ══════════════════════════════════════════════════════════════════════════
_MISSING = []
try:
    import torch
except ImportError:
    _MISSING.append("torch")
try:
    from transformers import TrainingArguments, TrainerCallback
except ImportError:
    _MISSING.append("transformers")
try:
    from peft import LoraConfig
except ImportError:
    _MISSING.append("peft")
try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    _MISSING.append("trl")
try:
    import bitsandbytes  # noqa: F401
except ImportError:
    _MISSING.append("bitsandbytes")

if _MISSING:
    logger.error(
        "مكتبات ناقصة: %s\nثبّتها أولاً: pip install %s",
        ", ".join(_MISSING), " ".join(_MISSING),
    )
    sys.exit(1)

if not torch.cuda.is_available():
    logger.error(
        "لا يوجد GPU (CUDA) متاح. هذا السكربت مصمَّم للتدريب على GPU سحابي "
        "(A100/H100) فقط — لن يعمل على CPU/Streamlit Community Cloud."
    )
    sys.exit(1)

from ai.arabic_transformer import get_production_base_model, QWEN_BASE_MODEL_ID
from data.dataset_loader import load_production_dataset, format_for_sft, DEFAULT_DATASET_PATH


# ══════════════════════════════════════════════════════════════════════════
# إعدادات قابلة للتخصيص
# ══════════════════════════════════════════════════════════════════════════
LOSS_LOG_PATH = os.environ.get("NSM_PRODUCTION_LOSS_LOG", "memory/production_loss.json")

DEFAULT_TARGET_MODULES = [
    "q_proj", "v_proj", "k_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


# ══════════════════════════════════════════════════════════════════════════
# Callback: يسجّل Cross-Entropy loss مباشرة إلى ملف JSON لواجهة Streamlit
# ══════════════════════════════════════════════════════════════════════════
class ProductionLossLoggerCallback(TrainerCallback):
    """
    يكتب سجلّ خسارة تراكمي إلى memory/production_loss.json بعد كل خطوة
    logging. صيغة الملف متوافقة مع نمط memory/training_loss.json الحالي
    (قائمة نقاط {step, loss, epoch, timestamp}) لتسهيل قراءتها من
    streamlit_app.py دون تعديل واجهة العرض.
    """

    def __init__(self, log_path: str = LOSS_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.history: List[Dict[str, Any]] = []
        if self.log_path.exists():
            try:
                self.history = json.loads(self.log_path.read_text(encoding="utf-8"))
            except Exception:
                self.history = []

    def on_log(self, args, state, control, logs=None, **kwargs):  # noqa: D401
        if not logs or "loss" not in logs:
            return
        entry = {
            "step": state.global_step,
            "epoch": round(logs.get("epoch", 0.0), 4),
            "loss": logs["loss"],
            "learning_rate": logs.get("learning_rate"),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.history.append(entry)
        try:
            self.log_path.write_text(
                json.dumps(self.history, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"[loss-logger] فشل كتابة {self.log_path}: {e}")


# ══════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="تدريب LoRA صناعي على قاعدة 7B/8B للهجة اليمنية")
    p.add_argument("--dataset", default=DEFAULT_DATASET_PATH, help="مسار JSON/JSONL")
    p.add_argument("--base-model", default=QWEN_BASE_MODEL_ID)
    p.add_argument("--output-dir", default="models/yemeni_qwen7b_lora")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-seq-len", type=int, default=1024)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--logging-steps", type=int, default=5)
    p.add_argument("--save-steps", type=int, default=100)
    p.add_argument("--min-rows-warning", type=int, default=50_000)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 70)
    logger.info("تدريب صناعي — Yemeni LLM (LoRA على %s)", args.base_model)
    logger.info("=" * 70)

    # 1) تحميل النموذج الأساس بكمّية مخفَّضة
    model, tokenizer = get_production_base_model(
        model_id=args.base_model, quantization=args.quantization
    )

    # 2) تحميل وتنسيق البيانات
    dataset = load_production_dataset(args.dataset, min_rows_warning=args.min_rows_warning)
    dataset = dataset.map(format_for_sft, remove_columns=dataset.column_names)

    # 3) تهيئة LoRA — target modules تغطي attention + MLP الكاملة (نمط 7B/8B قياسي)
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=DEFAULT_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 4) إعدادات التدريب — مهيّأة لـ A100/H100 (bf16, gradient checkpointing)
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        max_seq_length=args.max_seq_len,
        dataset_text_field="text",
        report_to=[],  # لا نعتمد على wandb/hub — التتبع محلي عبر الـcallback
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    loss_callback = ProductionLossLoggerCallback()

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
        callbacks=[loss_callback],
    )

    logger.info("عدد المعاملات القابلة للتدريب (LoRA فقط):")
    trainer.model.print_trainable_parameters()

    trainer.train()

    # 5) حفظ الـLoRA adapters فقط (ليس النموذج الأساس كاملاً — بضعة MB فقط)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"✓ اكتمل التدريب. adapters محفوظة في: {args.output_dir}")
    logger.info(f"✓ سجل الخسارة: {LOSS_LOG_PATH}")


if __name__ == "__main__":
    main()
