#!/usr/bin/env python3
"""تدريب Modern BBPE (أسلوب GPT-4/tiktoken) — التقنية الموصى بها."""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def collect() -> list:
    texts = []
    for path in ("ckg_sentences_v3.pkl", "ckg_sentences_v2.pkl", "ckg_sentences.pkl"):
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)
            if isinstance(data, list):
                texts.extend(str(x) for x in data if x)
                print(f"  + {path}: {len(data)}")
    ckg = "knowledge/cognitive_graph.json"
    if os.path.exists(ckg):
        with open(ckg, encoding="utf-8") as f:
            data = json.load(f)
        concepts = data.get("concepts") or {}
        if isinstance(concepts, dict):
            texts.extend(map(str, concepts.keys()))
            print(f"  + ckg concepts: {len(concepts)}")
    # بذرة متعددة اللغات
    texts += [
        "Hello world", "It's a test.", "Bonjour le monde",
        "こんにちは", "Привет", "مرحبا بالعالم",
    ]
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--out", default="models/transformer_ckg_v3/modern_bbpe_tokenizer.json")
    args = ap.parse_args()
    from ai.modern_bbpe_tokenizer import ModernBBPETokenizer
    texts = collect()
    if not texts:
        print("ABORT")
        return 1
    tok = ModernBBPETokenizer(vocab_size=args.vocab_size)
    n = tok.train(texts)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"✓ Modern BBPE vocab={n} merges={len(tok.merges)} → {args.out}")
    for s in ("الصبر مفتاح الفرج", "Hello, it's NSM!", "يمني + English"):
        print(f"  {s!r} → {tok.decode(tok.encode(s))!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
