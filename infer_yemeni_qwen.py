"""
infer_yemeni_qwen.py  [production-7b-llm]
=================================================
استدلال (تشغيل) لنموذج Qwen2.5-7B-Instruct + LoRA adapter مدرَّب عبر
train_production_yemeni.py — يحمّل القاعدة بنفس الإعدادات بالضبط
(get_production_base_model، بما فيها إصلاح bf16→fp16 التلقائي)، يركّب
الـadapter فوقها عبر PEFT، ويبني نفس صيغة ChatML المستخدمة أثناء التدريب
حرفياً (data.dataset_loader.SYSTEM_PROMPT_TEMPLATE + format_for_sft) —
عدم تطابق الصيغة بين التدريب والاستدلال يُفسد جودة الاستجابة بصمت.

⚠️ نفس متطلبات train_production_yemeni.py: GPU ≥16GB (T4 فما فوق)، ولا
يعمل على Streamlit Community Cloud أو أي بيئة CPU فقط.
    pip install torch transformers accelerate peft bitsandbytes

الاستخدام:
    # سؤال واحد
    python infer_yemeni_qwen.py --adapter-dir models/smoke_test_lora \\
        --instruction "وش يعني لما احد يقول لك (سدا) في صنعاء؟"

    # جلسة تفاعلية (سؤال بعد سؤال حتى Ctrl+C)
    python infer_yemeni_qwen.py --adapter-dir models/yemeni_qwen7b_lora --interactive
"""
from __future__ import annotations

import argparse
import logging
import sys

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("infer_yemeni_qwen")

# ══════════════════════════════════════════════════════════════════════════
# فحص مبكر وواضح للاعتماديات — نفس نمط train_production_yemeni.py
# ══════════════════════════════════════════════════════════════════════════
_MISSING = []
try:
    import torch
except ImportError:
    _MISSING.append("torch")
try:
    from transformers import AutoTokenizer
except ImportError:
    _MISSING.append("transformers")
try:
    from peft import PeftModel
except ImportError:
    _MISSING.append("peft")
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
        "لا يوجد GPU (CUDA) متاح. هذا السكربت يحتاج GPU حقيقي (T4 16GB فما "
        "فوق) — لن يعمل على CPU/Streamlit Community Cloud."
    )
    sys.exit(1)

from ai.arabic_transformer import get_production_base_model, QWEN_BASE_MODEL_ID
from data.dataset_loader import SYSTEM_PROMPT_TEMPLATE


def build_prompt(instruction: str, context_ckg: str = "") -> str:
    """يبني نفس صيغة ChatML حرفياً مثل data.dataset_loader.format_for_sft،
    لكن بدون قسم <|im_start|>assistant المكتمل (نتركه مفتوحاً للنموذج
    يكمّله هو). أي انحراف عن هذه الصيغة (حتى مسافة/سطر إضافي) عن صيغة
    التدريب يُضعف جودة الاستجابة بصمت دون أي خطأ ظاهر — لذلك هذه الدالة
    تُبقي النمط مطابقاً حرفياً لـformat_for_sft عمداً بدل إعادة صياغته."""
    system = SYSTEM_PROMPT_TEMPLATE
    context_ckg = (context_ckg or "").strip()
    if context_ckg:
        system = f"{system}\n\nسياق معرفي موثوق (لا تقتبسه حرفياً، استخدمه فقط للدقة):\n{context_ckg}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{instruction}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def load_model_and_tokenizer(adapter_dir: str, base_model: str, quantization: str):
    logger.info(f"[1/2] تحميل القاعدة ({base_model}, quantization={quantization})...")
    model, _base_tokenizer = get_production_base_model(
        model_id=base_model, quantization=quantization,
    )

    logger.info(f"[2/2] تركيب LoRA adapter من {adapter_dir}...")
    model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    # التوكنايزر المحفوظ مع الـadapter (tokenizer.save_pretrained(output_dir)
    # في train_production_yemeni.py) هو المطابق فعلياً لما تدرّب عليه النموذج
    # — نفضّله على توكنايزر القاعدة الخام لو موجود.
    try:
        tokenizer = AutoTokenizer.from_pretrained(adapter_dir)
        logger.info("توكنايزر من adapter_dir")
    except Exception:
        tokenizer = _base_tokenizer
        logger.warning("لم يُعثر على توكنايزر بـadapter_dir — استخدام توكنايزر القاعدة")

    return model, tokenizer


def generate(model, tokenizer, instruction: str, context_ckg: str,
             max_new_tokens: int, temperature: float) -> str:
    prompt = build_prompt(instruction, context_ckg)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=False)
    # القطع عند أول <|im_end|> — توليد ما بعده (لو حصل) ليس جزءاً من
    # إجابة هذا الدور، وskip_special_tokens=True كان سيحذفه بصمت بدل
    # القطع عنده، فيُبقي نصاً زائداً من دور تالٍ لم يُطلَب.
    text = text.split("<|im_end|>")[0].strip()
    return text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="استدلال لنموذج Qwen2.5-7B + LoRA يمني")
    p.add_argument("--adapter-dir", required=True, help="مجلد الـadapter المدرَّب (output-dir من التدريب)")
    p.add_argument("--base-model", default=QWEN_BASE_MODEL_ID)
    p.add_argument("--quantization", default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--instruction", default=None, help="سؤال واحد (بلا --interactive)")
    p.add_argument("--context-ckg", default="", help="سياق CKG اختياري لنفس السؤال")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--interactive", action="store_true", help="جلسة أسئلة متتالية حتى Ctrl+C")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.instruction and not args.interactive:
        logger.error("مرّر --instruction \"سؤالك\" أو --interactive")
        sys.exit(1)

    model, tokenizer = load_model_and_tokenizer(
        args.adapter_dir, args.base_model, args.quantization,
    )
    logger.info("✅ النموذج جاهز.")

    if args.interactive:
        print("جلسة تفاعلية — Ctrl+C للخروج.\n")
        try:
            while True:
                instruction = input("سؤالك: ").strip()
                if not instruction:
                    continue
                response = generate(
                    model, tokenizer, instruction, args.context_ckg,
                    args.max_new_tokens, args.temperature,
                )
                print(f"\n{response}\n")
        except KeyboardInterrupt:
            print("\nإلى اللقاء.")
    else:
        response = generate(
            model, tokenizer, args.instruction, args.context_ckg,
            args.max_new_tokens, args.temperature,
        )
        print(response)


if __name__ == "__main__":
    main()
