#!/usr/bin/env python3
"""
خط أنابيب تحسين دقة اللهجة اليمنية (بدون تدريب 1B كامل).

الخطوات:
  1) استخراج Lisan-Yemeni من Yemeni.zip
  2) توسيع YemeniTokenizer بالمعجم اللهجي
  3) تدريب Modern-BBPE اختياري على جمل يمنية (تغطية بايتات اللهجة)
  4) تقرير كشف لهجة على عيّنات

  python3 scripts/boost_yemeni_accuracy.py
  python3 scripts/boost_yemeni_accuracy.py --with-modern-bbpe
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default="data/yemeni/Yemeni.zip")
    ap.add_argument("--with-modern-bbpe", action="store_true")
    ap.add_argument("--lisan-max", type=int, default=5000)
    args = ap.parse_args()

    print("=== 1) تجهيز مدونة Lisan-Yemeni ===")
    rc = subprocess.call(
        [sys.executable, "scripts/prepare_yemeni_lisan.py", "--zip", args.zip],
    )
    if rc != 0:
        print("تحذير: فشل الاستخراج — تأكد من وجود Yemeni.zip")
    else:
        stats_path = "data/yemeni/corpus_stats.json"
        if os.path.exists(stats_path):
            print(json.dumps(json.load(open(stats_path, encoding="utf-8")), ensure_ascii=False, indent=2))

    print("\n=== 2) توسيع YemeniTokenizer ===")
    from ai.yemeni_tokenizer import get_yemeni_tokenizer
    tok = get_yemeni_tokenizer(expand_lisan=True, lisan_max_words=args.lisan_max)
    print("vocab_size:", tok.vocab_size)
    print("info:", tok.info())

    samples = [
        "كيفك ياخوي؟ ايش اخبارك",
        "ابشر سدا والأمور طيبة",
        "وين القات اليوم في صنعاء",
        "الجيش الوطني جديد علا الساحة",
        "What is patience in Islam?",  # غير يمني
    ]
    print("\n=== 3) عيّنات encode/decode + كشف لهجة ===")
    from ai.yemeni_dialect import detect_yemeni_score, normalize_yemeni
    for s in samples:
        ids = tok.encode(s)
        dec = tok.decode(ids)
        score = detect_yemeni_score(s)
        print(f"  score={score:.2f} | {s!r}")
        print(f"           → {dec!r} | ids={len(ids)}")

    if args.with_modern_bbpe:
        print("\n=== 4) Modern BBPE على جمل يمنية ===")
        from ai.modern_bbpe_tokenizer import ModernBBPETokenizer
        from ai.yemeni_dialect import load_yemeni_sentences
        sents = load_yemeni_sentences(limit=20000)
        if not sents:
            print("لا جمل — تخطّي")
        else:
            mtok = ModernBBPETokenizer(vocab_size=16000)
            n = mtok.train(sents)
            out = "models/yemeni_modern_bbpe.json"
            os.makedirs("models", exist_ok=True)
            mtok.save(out)
            print(f"✓ Yemeni Modern-BBPE vocab={n} → {out}")

    print("\n✓ انتهى تحسين مسار اللهجة اليمنية")
    print("الخطوات التالية للتدريب الثقيل:")
    print("  NSM_TOKENIZER=modern_bbpe python3 train_batch_v3.py  # إن دُمجت الجمل")
    print("  python3 train_yemeni.py  # LoRA على جهاز بذاكرة كافية")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
