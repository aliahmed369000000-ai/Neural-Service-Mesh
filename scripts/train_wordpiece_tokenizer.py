#!/usr/bin/env python3
"""تدريب WordPiece tokenizer على مصادر NSM."""
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
                texts.extend(map(str, concepts.keys()))
                print(f"  + {ckg}: {len(concepts)}")
        except Exception as e:
            print(f"  ! {ckg}: {e}")
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--num-merges", type=int, default=None)
    ap.add_argument("--out", default="models/transformer_ckg_v3/wordpiece_tokenizer.json")
    args = ap.parse_args()

    from ai.wordpiece_tokenizer import WordPieceTokenizer

    texts = collect_texts()
    if not texts:
        print("ABORT: لا نصوص")
        return 1
    tok = WordPieceTokenizer(vocab_size=args.vocab_size)
    n = tok.train(texts, num_merges=args.num_merges)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"✓ WordPiece vocab={n} → {args.out}")
    s = "الصبر مفتاح الفرج"
    ids = tok.encode(s)
    print(f"عينة decode: {tok.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
