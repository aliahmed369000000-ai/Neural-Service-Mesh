"""
data/fetch_hf_yemeni_dataset.py  [production-7b-llm]
======================================================
تحميل مجموعة/مجموعات بيانات من Hugging Face Hub، وتحويلها آلياً إلى
صيغة NSM الموحّدة (نفس schema المستخدم في data/dataset_loader.py):

{
  "id": str, "category": str, "instruction": str, "context_ckg": str,
  "output": str, "dialect_region": str, "difficulty": str
}

ثم كتابتها إلى JSONL (data/yemeni_production_instructions.jsonl) جاهزة
لـ load_production_dataset()، بالإضافة لملف ChatML منسّق مسبقاً
(data/yemeni_production_instructions.chatml.jsonl) عبر format_for_sft
من dataset_loader.py — بدون تكرار منطق التنسيق.

⚠️ يتطلب اتصال شبكة فعلي بـ huggingface.co وحزمة `datasets` —
لن يعمل من بيئة sandbox معزولة عن الشبكة. صُمِّم للتشغيل على سيرفر GPU
الإنتاج ضمن run_production_pipeline.sh.

الاستخدام:
    python data/fetch_hf_yemeni_dataset.py \
        --hf-dataset some-org/yemeni-dialect-instructions \
        --split train \
        --output data/yemeni_production_instructions.jsonl

    # عدة مجموعات بيانات مدموجة:
    python data/fetch_hf_yemeni_dataset.py \
        --hf-dataset org1/ds1 --hf-dataset org2/ds2 \
        --output data/yemeni_production_instructions.jsonl

متغيرات بيئة:
    HF_TOKEN                    (اختياري — لمجموعات بيانات خاصة/gated)
    NSM_FETCH_CACHE_DIR          (افتراضي: .hf_cache — يُحذف لاحقاً بواسطة
                                   run_production_pipeline.sh، وليس هذا السكربت)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("fetch_hf_yemeni_dataset")

DEFAULT_OUTPUT = "data/yemeni_production_instructions.jsonl"
DEFAULT_CHATML_OUTPUT = "data/yemeni_production_instructions.chatml.jsonl"
VALID_CATEGORIES = {
    "idiom", "geography", "socioeconomic", "history", "religious_ckg", "conversational",
}
VALID_REGIONS = {"صنعاني", "تعزي", "حضرمي", "تهامي", "عام"}
VALID_DIFFICULTY = {"basic", "intermediate", "advanced"}


# ══════════════════════════════════════════════════════════════════════════
# تحويل صف خام من HF (أعمدة غير معروفة مسبقاً) إلى schema NSM
# ══════════════════════════════════════════════════════════════════════════
def _first_present(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return default


def _row_id(source_tag: str, row: Dict[str, Any], idx: int) -> str:
    raw = row.get("id") or f"{source_tag}-{idx}"
    digest = hashlib.sha1(str(raw).encode("utf-8")).hexdigest()[:8]
    return f"yem-hf-{digest}"


def normalize_row(row: Dict[str, Any], source_tag: str, idx: int) -> Optional[Dict[str, str]]:
    """
    يحاول التعرّف على الأعمدة الشائعة في مجموعات بيانات التعليمات
    (instruction/prompt/question ← output/response/answer) ويطبّع القيم
    التصنيفية على القوائم المسموحة. يُرجع None لو الصف غير قابل للاستخدام
    (بلا instruction أو output فعليين) بدل رمي استثناء يوقف المسار كله.
    """
    instruction = _first_present(row, ["instruction", "prompt", "question", "input"])
    output = _first_present(row, ["output", "response", "answer", "completion"])
    if not instruction or not output:
        return None

    category = str(row.get("category", "")).strip().lower()
    if category not in VALID_CATEGORIES:
        category = "conversational"

    region = str(row.get("dialect_region", "")).strip()
    if region not in VALID_REGIONS:
        region = "عام"

    difficulty = str(row.get("difficulty", "")).strip().lower()
    if difficulty not in VALID_DIFFICULTY:
        difficulty = "basic"

    return {
        "id": _row_id(source_tag, row, idx),
        "category": category,
        "instruction": instruction,
        "context_ckg": _first_present(row, ["context_ckg", "context", "system"]),
        "output": output,
        "dialect_region": region,
        "difficulty": difficulty,
    }


# ══════════════════════════════════════════════════════════════════════════
# تحميل من Hugging Face (مع فحص اعتمادية مبكر وواضح)
# ══════════════════════════════════════════════════════════════════════════
def load_hf_rows(dataset_name: str, split: str, cache_dir: str) -> Iterable[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            "حزمة `datasets` غير مثبتة. ثبّتها عبر: pip install datasets"
        ) from e

    token = os.environ.get("HF_TOKEN")
    logger.info(f"تحميل {dataset_name} (split={split}) من Hugging Face Hub...")
    ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir, token=token)
    logger.info(f"✓ تم تحميل {len(ds)} صف من {dataset_name}")
    return ds


def dedupe(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    seen = set()
    unique: List[Dict[str, str]] = []
    for r in rows:
        key = hashlib.sha1((r["instruction"] + "||" + r["output"]).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    dropped = len(rows) - len(unique)
    if dropped:
        logger.info(f"إزالة {dropped} صف مكرر")
    return unique


def write_jsonl(rows: List[Dict[str, str]], path: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    logger.info(f"✓ كُتب {len(rows)} صف إلى {path}")


def write_chatml(rows: List[Dict[str, str]], path: str) -> None:
    from data.dataset_loader import format_for_sft

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(format_for_sft(r), ensure_ascii=False) + "\n")
    logger.info(f"✓ كُتب تنسيق ChatML إلى {path}")


# ══════════════════════════════════════════════════════════════════════════
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf-dataset", action="append", dest="hf_datasets", required=True,
        help="اسم مجموعة بيانات على HF Hub (يمكن تكرار الخيار لدمج عدة مجموعات)",
    )
    p.add_argument("--split", default="train")
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    p.add_argument("--chatml-output", default=DEFAULT_CHATML_OUTPUT)
    p.add_argument("--cache-dir", default=os.environ.get("NSM_FETCH_CACHE_DIR", ".hf_cache"))
    p.add_argument("--min-rows-warning", type=int, default=50_000)
    p.add_argument(
        "--skip-chatml", action="store_true",
        help="لا تكتب ملف ChatML، فقط JSONL الخام بصيغة NSM",
    )
    return p


def run(args: argparse.Namespace) -> int:
    all_rows: List[Dict[str, str]] = []
    for name in args.hf_datasets:
        source_tag = name.replace("/", "_")
        raw = load_hf_rows(name, args.split, args.cache_dir)
        kept = 0
        for idx, row in enumerate(raw):
            normalized = normalize_row(dict(row), source_tag, idx)
            if normalized is not None:
                all_rows.append(normalized)
                kept += 1
        logger.info(f"  → {kept}/{len(raw)} صف صالح من {name}")

    if not all_rows:
        logger.error("لا توجد صفوف صالحة بعد التطبيع — تحقق من أسماء مجموعات البيانات.")
        return 1

    all_rows = dedupe(all_rows)

    if len(all_rows) < args.min_rows_warning:
        logger.warning(
            f"العدد النهائي ({len(all_rows)}) أقل من الحد الموصى به "
            f"({args.min_rows_warning}) لتدريب إنتاجي جاد — تابع على مسؤوليتك."
        )

    write_jsonl(all_rows, args.output)
    if not args.skip_chatml:
        write_chatml(all_rows, args.chatml_output)

    logger.info(f"✓ اكتمل. إجمالي الصفوف النهائية: {len(all_rows)}")
    return 0


def main() -> None:
    args = build_arg_parser().parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
