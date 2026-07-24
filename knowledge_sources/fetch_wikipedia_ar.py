"""
knowledge_sources/fetch_wikipedia_ar.py
==========================================
⚠️ هذا السكربت مُصمَّم للتشغيل على Kaggle / Google Colab — حيث الإنترنت
مفتوح بالكامل. لا يعمل داخل بيئة sandbox المستخدمة لتطوير NSM (شبكتها
مقيّدة لنطاقات معيّنة فقط ولا تصل إلى huggingface.co أو wikipedia.org).

الوظيفة: يجلب مقالات ويكيبيديا العربية، يقسّمها لفقرات، ويحوّلها لنفس
صيغة JSON اللي يقرأها knowledge/generic_ckg_builder.py:
    [{"text": ..., "reference": ..., "group": ...}, ...]

طريقة الاستخدام على Kaggle:
    1. أنشئ Notebook جديد، فعّل Internet من إعدادات الـ Notebook (يمين الشاشة)
    2. !pip install datasets -q
    3. ارفع هذا الملف كـ Kaggle Dataset أو الصقه مباشرة في خلية
    4. شغّل:
         python3 fetch_wikipedia_ar.py --max-articles 5000 \\
             --out wikipedia_ar_sample.json

    5. حمّل الملف الناتج (wikipedia_ar_sample.json) وارفعه لمستودع GitHub
       تحت knowledge_sources/data/، أو استخدمه مباشرة في نفس بيئة Kaggle
       مع generic_ckg_builder.py.

⚠️ اسم/إصدار مجموعة بيانات ويكيبيديا العربية على HuggingFace قد يتغيّر
مع الوقت. تحقق وقت التشغيل من:
    https://huggingface.co/datasets/wikimedia/wikipedia
واختر أحدث تفريغ متاح لمكوّن "ar" (مثال: "20231101.ar" أو أحدث).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

# فئات تصنيف تقريبية حسب كلمات مفتاحية في عنوان/بداية المقالة
# (تصنيف خفيف اختياري فقط لعمود "group" — التصنيف الحقيقي يتم لاحقاً
#  تلقائياً عبر GenericConceptExtractor نفسه، هذا فقط لتوزيع أولي مفيد)
_ROUGH_CATEGORIES: Dict[str, List[str]] = {
    "علوم": ["فيزياء", "كيمياء", "أحياء", "رياضيات", "طب", "فلك", "تقنية", "حاسوب"],
    "تاريخ": ["تاريخ", "حرب", "معركة", "دولة", "خلافة", "عصر", "حضارة"],
    "جغرافيا": ["مدينة", "دولة", "نهر", "جبل", "قارة", "محافظة", "إقليم"],
    "أدب": ["رواية", "شعر", "شاعر", "أديب", "قصة", "مسرحية"],
    "أخرى": [],
}


def _rough_group(title: str, text: str) -> str:
    sample = f"{title} {text[:200]}"
    for group, keywords in _ROUGH_CATEGORIES.items():
        if any(kw in sample for kw in keywords):
            return group
    return "عام"


def _split_paragraphs(text: str, min_len: int = 40, max_len: int = 600) -> List[str]:
    """يقسّم مقالة طويلة إلى فقرات بحجم مناسب للتدريب (مو طويلة جداً ولا قصيرة جداً)."""
    raw_paras = re.split(r"\n{1,}", text)
    out = []
    for p in raw_paras:
        p = p.strip()
        if len(p) < min_len:
            continue
        # قصّ الفقرات الطويلة جداً لعدة قطع بدل حذفها
        while len(p) > max_len:
            cut = p[:max_len].rfind("۔") or p[:max_len].rfind(".")
            cut = cut if cut > min_len else max_len
            out.append(p[:cut].strip())
            p = p[cut:].strip()
        if len(p) >= min_len:
            out.append(p)
    return out


def fetch_and_convert(max_articles: int = 5000, max_paragraphs_per_article: int = 5) -> List[Dict]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "مكتبة datasets غير مثبتة. شغّل: pip install datasets -q\n"
            "(هذا السكربت مخصص للتشغيل على Kaggle/Colab بإنترنت مفتوح)"
        )

    print("⏳ تحميل مجموعة بيانات ويكيبيديا العربية من HuggingFace …")
    print("   (تحقق من اسم/إصدار المجموعة الأحدث على: "
          "https://huggingface.co/datasets/wikimedia/wikipedia)")
    ds = load_dataset("wikimedia/wikipedia", "20231101.ar", split="train", streaming=True)

    docs: List[Dict] = []
    for i, article in enumerate(ds):
        if len(docs) >= max_articles * max_paragraphs_per_article:
            break
        title = article.get("title", f"مقالة_{i}")
        text = article.get("text", "")
        if not text:
            continue
        paragraphs = _split_paragraphs(text)[:max_paragraphs_per_article]
        group = _rough_group(title, text)
        for j, para in enumerate(paragraphs):
            docs.append({
                "text": para,
                "reference": f"wikipedia_ar:{title}:{j}",
                "group": group,
            })
        if i > 0 and i % 500 == 0:
            print(f"   … {i} مقالة معالَجة، {len(docs)} فقرة مُستخرجة")

    print(f"✅ انتهى الجلب: {len(docs)} فقرة من ويكيبيديا العربية")
    return docs


def main():
    p = argparse.ArgumentParser(description="جلب وتحويل ويكيبيديا العربية لصيغة generic_ckg_builder")
    p.add_argument("--max-articles", type=int, default=5000)
    p.add_argument("--max-paragraphs-per-article", type=int, default=5)
    p.add_argument("--out", default="wikipedia_ar_sample.json")
    args = p.parse_args()

    docs = fetch_and_convert(
        max_articles=args.max_articles,
        max_paragraphs_per_article=args.max_paragraphs_per_article,
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"💾 حُفظ → {out_path} ({len(docs)} مستند)")


if __name__ == "__main__":
    main()
