#!/usr/bin/env python3
"""
تدريب BPE Tokenizer على مصادر NSM وحفظه.

  python3 scripts/train_bpe_tokenizer.py
  python3 scripts/train_bpe_tokenizer.py --vocab-size 8192 --out models/transformer_ckg_v3/bpe_tokenizer.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect_texts() -> list:
    texts = []
    for path in (
        "ckg_sentences_v3.pkl",
        "ckg_sentences_v2.pkl",
        "ckg_sentences.pkl",
        "ckg_sentences_general_ar.pkl",
    ):
        if not os.path.exists(path):
            continue
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                texts.extend(str(x) for x in data if x)
                print(f"  + {path}: {len(data)}")
        except Exception as e:
            print(f"  ! {path}: {e}")

    ckg = "knowledge/cognitive_graph.json"
    if os.path.exists(ckg):
        try:
            with open(ckg, encoding="utf-8") as f:
                data = json.load(f)
            concepts = data.get("concepts") or {}
            if isinstance(concepts, dict):
                texts.extend(str(k) for k in concepts.keys())
                print(f"  + {ckg}: {len(concepts)} مفهوم")
        except Exception as e:
            print(f"  ! {ckg}: {e}")

    quran = "knowledge_sources/quran/data/quran.json"
    if os.path.exists(quran):
        try:
            with open(quran, encoding="utf-8") as f:
                data = json.load(f)
            n = 0
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and item.get("text"):
                        texts.append(str(item["text"]))
                        n += 1
            print(f"  + {quran}: {n}")
        except Exception as e:
            print(f"  ! {quran}: {e}")
    return texts


def main() -> int:
    ap = argparse.ArgumentParser(description="تدريب BPE tokenizer لـ NSM")
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--num-merges", type=int, default=None, help="عدد عمليات الدمج (افتراضي: تلقائي)")
    ap.add_argument(
        "--out",
        default="models/transformer_ckg_v3/bpe_tokenizer.json",
    )
    args = ap.parse_args()

    from ai.bpe_tokenizer import BPETokenizer

    print("جمع النصوص…")
    texts = collect_texts()
    if not texts:
        print("ABORT: لا توجد نصوص")
        return 1
    print(f"إجمالي النصوص: {len(texts)}")

    tok = BPETokenizer(vocab_size=args.vocab_size)
    n = tok.train(texts, num_merges=args.num_merges)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"✓ BPE vocab={n} merges={len(tok.merges)} → {args.out}")

    sample = "الصبر مفتاح الفرج"
    ids = tok.encode(sample)
    print(f"عينة: {sample!r}")
    print(f"  ids={ids.tolist()}")
    print(f"  decode={tok.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
