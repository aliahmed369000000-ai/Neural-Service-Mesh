"""
data/dataset_loader.py  [production-7b-llm]
============================================
خط أنابيب تحميل بيانات قابل للتوسّع لمجموعة التعليمات اليمنية الصناعية
(هدف: 50,000+ صف محادثة). مصمَّم للعمل مع Hugging Face `datasets` على
GPU سحابي — منفصل تماماً عن train_yemeni.py القديم (YemeniDecoder 1B
العشوائي) ولا يعدّله.

بنية الصف الواحد (JSON schema):
{
  "id": "yem-000123",
  "category": "idiom" | "geography" | "socioeconomic" | "history" | "religious_ckg" | "conversational",
  "instruction": "السؤال/الطلب باللهجة اليمنية",
  "context_ckg": "سياق اختياري من CKG/الرسم المعرفي الإسلامي (نص جاهز يُحقن في البرومبت)",
  "output": "الإجابة المتوقعة باللهجة اليمنية",
  "dialect_region": "صنعاني" | "تعزي" | "حضرمي" | "تهامي" | "عام",
  "difficulty": "basic" | "intermediate" | "advanced"
}

الاستخدام:
    from data.dataset_loader import load_production_dataset
    ds = load_production_dataset("data/yemeni_production_instructions.jsonl")
    formatted = ds.map(format_for_sft)
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════
# مسارات افتراضية
# ══════════════════════════════════════════════════════════════════════════
DEFAULT_DATASET_PATH = os.environ.get(
    "NSM_PRODUCTION_DATASET", "data/yemeni_production_instructions.jsonl"
)

# البرومبت النظامي الأساسي — يُستخدم في التنسيق الهجين (CKG + سؤال)
SYSTEM_PROMPT_TEMPLATE = (
    "أنت مساعد ذكاء اصطناعي يتحدث اللهجة اليمنية بطلاقة، متخصص بالمعرفة "
    "الإسلامية والسياق الثقافي والجغرافي اليمني. أجب دائماً باللهجة اليمنية "
    "الطبيعية، واستخدم السياق المرفق (إن وُجد) كمصدر حقائق موثوق دون ذكره "
    "حرفياً كنص خام."
)


def _iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"[dataset_loader] سطر {line_no} تالف في {path}: {e}")


def _load_raw_rows(path: str) -> List[Dict[str, Any]]:
    """
    يدعم كلا الصيغتين: JSON (قائمة كائنات) و JSONL (سطر لكل كائن) — يختار
    تلقائياً حسب امتداد الملف. يرمي FileNotFoundError واضح لو الملف غير
    موجود بدل فشل صامت.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"لم يُعثر على ملف البيانات: {path}. "
            f"استخدم القالب في data/yemeni_production_instructions.template.json "
            f"كبداية لبناء مجموعة الـ50K+ صف."
        )
    if p.suffix == ".jsonl":
        rows = list(_iter_jsonl(p))
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data if isinstance(data, list) else data.get("rows", [])
    logger.info(f"[dataset_loader] تحميل {len(rows)} صف من {path}")
    return rows


def _validate_row(row: Dict[str, Any], idx: int) -> Optional[str]:
    """يتحقق من الحقول الإلزامية. يُرجع رسالة خطأ أو None إن كان الصف سليماً."""
    for field in ("instruction", "output"):
        if not row.get(field):
            return f"صف #{idx}: حقل '{field}' مفقود أو فارغ"
    return None


def load_production_dataset(
    path: str = DEFAULT_DATASET_PATH,
    min_rows_warning: int = 50_000,
    strict: bool = False,
):
    """
    يحمّل مجموعة بيانات التعليمات اليمنية الصناعية عبر مكتبة `datasets` من
    Hugging Face (streaming-friendly، قابل للتوسّع لعشرات آلاف الصفوف بدون
    استهلاك ذاكرة زائد عبر .map مع batched=True لاحقاً في التدريب).

    Args:
        path: مسار JSON أو JSONL.
        min_rows_warning: يُصدر تحذيراً (لا خطأ) لو عدد الصفوف أقل من هذا الهدف.
        strict: لو True، أي صف غير صالح يوقف التحميل بالكامل (فشل واضح).
                لو False (افتراضي)، الصفوف غير الصالحة تُستبعَد ويُسجَّل عددها.

    Returns:
        datasets.Dataset جاهز لـ SFTTrainer.

    Raises:
        RuntimeError: لو مكتبة `datasets` غير مثبَّتة.
        FileNotFoundError: لو الملف غير موجود.
        ValueError: لو strict=True وهناك صفوف غير صالحة.
    """
    try:
        from datasets import Dataset
    except ImportError as e:
        raise RuntimeError(
            "load_production_dataset يتطلب: pip install datasets"
        ) from e

    raw_rows = _load_raw_rows(path)
    valid_rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    for i, row in enumerate(raw_rows):
        err = _validate_row(row, i)
        if err:
            errors.append(err)
            if strict:
                raise ValueError(err)
            continue
        valid_rows.append(row)

    if errors:
        logger.warning(
            f"[dataset_loader] استُبعِد {len(errors)} صف غير صالح "
            f"(من أصل {len(raw_rows)}) — أول 5: {errors[:5]}"
        )

    if len(valid_rows) < min_rows_warning:
        logger.warning(
            f"[dataset_loader] عدد الصفوف الحالي ({len(valid_rows)}) أقل من "
            f"الهدف الصناعي ({min_rows_warning}). النتائج على هذا الحجم أقرب "
            f"لإثبات مفهوم (proof of concept) وليست جاهزة لإنتاج فعلي."
        )

    logger.info(f"[dataset_loader] {len(valid_rows)} صف صالح جاهز للتدريب")
    return Dataset.from_list(valid_rows)


def format_for_sft(row: Dict[str, Any]) -> Dict[str, str]:
    """
    ينسّق صفاً واحداً إلى نص محادثة كامل بصيغة ChatML (متوافقة مع Qwen2.5)
    عبر حقن سياق CKG كسطر نظام إضافي — هذا هو "حقن CKG داخل موتّر البرومبت"
    المطلوب: نص جاهز يُحوَّل لاحقاً إلى tokens عبر tokenizer الأساس نفسه،
    وليس تمثيلاً منفصلاً.

    يُستخدم كـ: dataset.map(format_for_sft) قبل تمريره لـ SFTTrainer
    (formatting_func أو حقل "text" مباشرة).
    """
    system = SYSTEM_PROMPT_TEMPLATE
    ckg_context = row.get("context_ckg", "").strip()
    if ckg_context:
        system = f"{system}\n\nسياق معرفي موثوق (لا تقتبسه حرفياً، استخدمه فقط للدقة):\n{ckg_context}"

    text = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{row['instruction']}<|im_end|>\n"
        f"<|im_start|>assistant\n{row['output']}<|im_end|>"
    )
    return {"text": text}


# ══════════════════════════════════════════════════════════════════════════
# قالب إنتاجي: أمثلة تُظهر التغطية المطلوبة (لهجة/جغرافيا/اجتماعي/تاريخ/CKG)
# هذا ليس المجموعة الكاملة (50K+) بل بذرة بنيوية يُبنى عليها بالتوسّع.
# ══════════════════════════════════════════════════════════════════════════
PRODUCTION_TEMPLATE_SAMPLE: List[Dict[str, Any]] = [
    {
        "id": "yem-idiom-0001",
        "category": "idiom",
        "instruction": "وش يعني لما احد يقول لك (سدا) في صنعاء؟",
        "context_ckg": "",
        "output": "(سدا) كلمة صنعانية دارجة معناها (فعلاً/صحيح/بجد) — تُستخدم للتأكيد على كلام قيل، زي لما تقول 'سدا كذا؟' يعني 'هل هذا صحيح فعلاً؟'.",
        "dialect_region": "صنعاني",
        "difficulty": "basic",
    },
    {
        "id": "yem-idiom-0002",
        "category": "idiom",
        "instruction": "ايش الفرق بين (أبشر) و(صنف) في اللهجة اليمنية؟",
        "context_ckg": "",
        "output": "(أبشر) رد إيجابي معناه 'حاضر/تم/على عيني' لما حد يطلب منك شي، أما (صنف) فتُقال للشخص المتمكن أو الشي الممتاز، زي 'فلان صنف في شغله' يعني ماهر جداً.",
        "dialect_region": "عام",
        "difficulty": "basic",
    },
    {
        "id": "yem-geo-0001",
        "category": "geography",
        "instruction": "كيف اروح من ميدان التحرير في صنعاء الى سوق الملح؟",
        "context_ckg": "",
        "output": "من ميدان التحرير امشي جنوب باتجاه باب اليمن، سوق الملح داخل مدينة صنعاء القديمة قريب من باب اليمن مباشرة، المسافة تقريباً 10-15 دقيقة مشي.",
        "dialect_region": "صنعاني",
        "difficulty": "intermediate",
    },
    {
        "id": "yem-socio-0001",
        "category": "socioeconomic",
        "instruction": "ايش وضع سعر صرف الريال اليمني وتأثيره على الجهال في المدارس؟",
        "context_ckg": "",
        "output": "تذبذب سعر الصرف أثر على تكلفة الأدوات المدرسية والمواصلات، وكثير من الجهال (الأطفال) صار يصعب على أهاليهم توفير احتياجاتهم الدراسية بسبب غلاء الأسعار المستمر.",
        "dialect_region": "عام",
        "difficulty": "advanced",
    },
    {
        "id": "yem-religious-ckg-0001",
        "category": "religious_ckg",
        "instruction": "وش تفسير آية الرحمن الرحيم بأسلوب سهل؟",
        "context_ckg": "سورة الفاتحة آية 3: الرحمن الرحيم. المفاهيم ذات الصلة: الرحمة، أسماء الله الحسنى.",
        "output": "(الرحمن الرحيم) يعني ان الله سبحانه رحمته وسعت كل شي في الدنيا لجميع خلقه، و(الرحيم) رحمته خاصة بالمؤمنين يوم القيامة — اسمين من أسماء الله الحسنى يدلّان على صفة الرحمة.",
        "dialect_region": "عام",
        "difficulty": "intermediate",
    },
]


def write_template_file(
    out_path: str = "data/yemeni_production_instructions.template.json",
) -> str:
    """يكتب ملف قالب صغير (5 أمثلة) كنقطة انطلاق لبناء مجموعة الـ50K+."""
    Path(out_path).write_text(
        json.dumps(PRODUCTION_TEMPLATE_SAMPLE, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"[dataset_loader] قالب مكتوب في {out_path}")
    return out_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    p = write_template_file()
    print(f"✓ قالب البيانات الإنتاجية مكتوب في: {p}")
    print(f"✓ عدد أمثلة القالب: {len(PRODUCTION_TEMPLATE_SAMPLE)}")
    print("✓ الفئات المغطاة:", sorted({r['category'] for r in PRODUCTION_TEMPLATE_SAMPLE}))
