"""
data/convert_lisan_yemeni.py  [production-7b-llm]
====================================================
يحوّل مدوّنة Lisan اليمنية (SinaLab / جامعة بيرزيت، مرخّصة CC BY 4.0،
~1.05M توكن مُعلَّم صرفياً من تويتر) إلى مصدرين متكاملين:

  1) knowledge/yemeni_graph.py (YKG) — كل كلمة لهجية فريدة تُضاف كعقدة
     Concept، وتُربط بمقابلها الفصيح (MSA lemma) عبر علاقة
     "dialect_to_msa". هذا جراف معرفي مستقل تماماً عن CKG (الحصري
     للقرآن) — راجع knowledge/yemeni_graph.py للتفاصيل.

  2) data/yemeni_lisan_instructions.jsonl — أزواج تعليمات/إجابة
     مُولَّدة بالقالب (templated) من صفوف المعجم، بنفس schema
     data/dataset_loader.py (id/category/instruction/context_ckg/
     output/dialect_region/difficulty)، فئة "idiom" افتراضياً.

⚠️ الملف الخام غير موجود في المستودع ولا يُنزَّل تلقائياً: مدوّنة
   Lisan اليمنية محجوبة خلف نموذج طلب Google Forms (ليست تنزيلاً
   مباشراً)، راجع:
   https://sina.birzeit.edu/currasat/about-en.html
   رابط الطلب المباشر لكورپس اليمني:
   https://docs.google.com/forms/d/e/1FAIpQLSfIhp5wJx2Ku9rIy0XQPkp6aU0s0mLRKvSXpTOnmSXb-J5D1Q/viewform
   بعد الحصول على الملف (CSV/TSV)، ضعه محلياً ومرّر مساره عبر --input.

بنية الأعمدة المتوقّعة (موثّقة من SinaLab/ADAT — قابلة للتخصيص عبر
--col-* لو اختلفت أسماء الأعمدة الفعلية في الملف المُستلَم):
   word        : الكلمة اللهجية كما وردت (dialect surface form)
   msa_lemma   : المقابل الفصيح / اللemma
   gloss_en    : شرح إنجليزي مختصر
   pos         : جزء الكلام (Part-of-Speech)

الاستخدام:
    python data/convert_lisan_yemeni.py --input /path/to/lisan_yemeni.csv

    # أعمدة بأسماء مختلفة عن الافتراضي:
    python data/convert_lisan_yemeni.py --input file.tsv --delimiter '\\t' \\
        --col-word Word --col-lemma MSA_Lemma --col-gloss Gloss --col-pos POS
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("convert_lisan_yemeni")

from knowledge.yemeni_graph import get_ykg  # noqa: E402

DEFAULT_INSTRUCTIONS_OUTPUT = "data/yemeni_lisan_instructions.jsonl"
SOURCE_TAG_PREFIX = "lisan-yemeni"

# قوالب تعليمات متعددة لتفادي التكرار الحرفي بين كل الصفوف
_TEMPLATES = [
    ("شو معنى كلمة \"{word}\" باللهجة اليمنية؟",
     "\"{word}\" تعني \"{msa}\" بالفصحى."),
    ("وش يقصدون أهل اليمن لما يقولون \"{word}\"؟",
     "يقصدون بها \"{msa}\"."),
    ("ترجم لي كلمة \"{word}\" اليمنية للفصحى.",
     "\"{word}\" بالفصحى معناها \"{msa}\"."),
]

_TEMPLATES_WITH_GLOSS = [
    ("شو معنى كلمة \"{word}\" باللهجة اليمنية، وشرحها بالإنجليزي؟",
     "\"{word}\" تعني \"{msa}\" بالفصحى ({gloss} بالإنجليزي)."),
]


def read_rows(path: str, delimiter: str) -> Iterator[Dict[str, str]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"الملف غير موجود: {path}\n"
            "الكورپس اليمني من Lisan محجوب خلف نموذج طلب (Google Form)، راجع "
            "docstring هذا الملف للرابط — نزّله يدوياً أولاً ثم مرّر مساره هنا."
        )
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        for row in reader:
            yield row


def _row_source_tag(row: Dict[str, str], idx: int) -> str:
    raw = f"{row.get('word', '')}-{idx}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
    return f"{SOURCE_TAG_PREFIX}:{digest}"


def normalize_row(
    row: Dict[str, str],
    col_word: str,
    col_lemma: str,
    col_gloss: str,
    col_pos: str,
) -> Optional[Dict[str, str]]:
    """يطبّع صفاً خاماً من ملف Lisan إلى (word, msa_lemma, gloss_en, pos).
    يُرجع None لو الصف بلا كلمة لهجية فعلية (بدل رمي استثناء يوقف كل التحويل)."""
    word = (row.get(col_word) or "").strip()
    if not word:
        return None
    return {
        "word": word,
        "msa_lemma": (row.get(col_lemma) or "").strip(),
        "gloss_en": (row.get(col_gloss) or "").strip(),
        "pos": (row.get(col_pos) or "").strip(),
    }


def build_instruction_pairs(word: str, msa: str, gloss: str, cluster: str) -> List[Dict[str, str]]:
    """يبني زوج/أزواج تعليمات من صف معجمي واحد، بصيغة data/dataset_loader.py."""
    if not msa:
        return []
    pairs: List[Dict[str, str]] = []
    templates = _TEMPLATES_WITH_GLOSS if gloss else _TEMPLATES
    for q_tpl, a_tpl in templates:
        instruction = q_tpl.format(word=word)
        output = a_tpl.format(word=word, msa=msa, gloss=gloss)
        row_id = "yem-lisan-" + hashlib.sha1((word + "|" + q_tpl).encode("utf-8")).hexdigest()[:10]
        pairs.append({
            "id": row_id,
            "category": "idiom",
            "instruction": instruction,
            "context_ckg": "",
            "output": output,
            "dialect_region": "عام",  # Lisan اليمني غير مقسّم لمناطق فرعية في مصدره
            "difficulty": "basic",
        })
        break  # قالب واحد فقط لكل كلمة لتفادي تضخّم اصطناعي — يمكن رفعه لاحقاً
    return pairs


def run(args: argparse.Namespace) -> int:
    ykg = get_ykg(Path(args.ykg_file)) if args.ykg_file else get_ykg()

    all_pairs: List[Dict[str, str]] = []
    seen_words = 0
    added_concepts = 0
    skipped_no_word = 0
    skipped_no_lemma = 0

    for idx, raw in enumerate(read_rows(args.input, args.delimiter)):
        normalized = normalize_row(raw, args.col_word, args.col_lemma, args.col_gloss, args.col_pos)
        if normalized is None:
            skipped_no_word += 1
            continue
        seen_words += 1
        source_tag = _row_source_tag(raw, idx)

        ykg.add_concept(
            normalized["word"],
            dialect_region="عام",
            cluster=normalized["pos"] or "غير مصنّف",
            source=source_tag,
            msa_equivalent=normalized["msa_lemma"],
            gloss_en=normalized["gloss_en"],
        )
        added_concepts += 1

        if normalized["msa_lemma"]:
            ykg.add_relation(
                normalized["word"], normalized["msa_lemma"],
                evidence=source_tag, relation_type="dialect_to_msa",
            )
        else:
            skipped_no_lemma += 1

        all_pairs.extend(build_instruction_pairs(
            normalized["word"], normalized["msa_lemma"],
            normalized["gloss_en"], normalized["pos"],
        ))

    ykg.save()

    out = Path(args.instructions_output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(__import__("json").dumps(p, ensure_ascii=False) + "\n")

    logger.info(
        f"✓ اكتمل: {seen_words} كلمة معالَجة، {added_concepts} عقدة YKG، "
        f"{skipped_no_word} صف بلا كلمة، {skipped_no_lemma} كلمة بلا مقابل فصيح، "
        f"{len(all_pairs)} زوج تعليمات → {out}"
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True, help="مسار ملف Lisan اليمني (CSV/TSV) بعد تنزيله يدوياً")
    p.add_argument("--delimiter", default=",", help="فاصل الأعمدة (افتراضي ',', استخدم '\\t' لملفات TSV)")
    p.add_argument("--col-word", default="word")
    p.add_argument("--col-lemma", default="msa_lemma")
    p.add_argument("--col-gloss", default="gloss_en")
    p.add_argument("--col-pos", default="pos")
    p.add_argument("--instructions-output", default=DEFAULT_INSTRUCTIONS_OUTPUT)
    p.add_argument("--ykg-file", default="", help="مسار مخصص لملف YKG (افتراضي: knowledge/yemeni_graph.json)")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
