#!/usr/bin/env python3
"""
بناء قاموس WordTokenizer من مصادر المشروع المتاحة.

المصادر (بالترتيب، أي ملف موجود يُستخدم):
  - ckg_sentences_v3.pkl / ckg_sentences_v2.pkl / ckg_sentences.pkl
  - knowledge/cognitive_graph.json (أسماء المفاهيم)
  - knowledge_sources/quran/data/quran.json (نصوص الآيات إن وُجد)

الاستخدام:
  python3 scripts/build_tokenizer_vocab.py
  python3 scripts/build_tokenizer_vocab.py --max-vocab 8192 --out models/transformer_ckg_v3/tokenizer_vocab.json
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_sentence_files() -> list:
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
                print(f"  + {path}: {len(data)} جملة")
        except Exception as e:
            print(f"  ! {path}: {e}")
    return texts


def _load_ckg_concepts() -> list:
    path = "knowledge/cognitive_graph.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            ckg = json.load(f)
        concepts = ckg.get("concepts") or {}
        if isinstance(concepts, dict):
            names = [str(k) for k in concepts.keys()]
        elif isinstance(concepts, list):
            names = [str(c.get("name", c)) if isinstance(c, dict) else str(c) for c in concepts]
        else:
            names = []
        print(f"  + {path}: {len(names)} مفهوم")
        return names
    except Exception as e:
        print(f"  ! {path}: {e}")
        return []


def _load_quran_texts() -> list:
    path = "knowledge_sources/quran/data/quran.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        texts = []
        # أشكال شائعة: list of ayat, or dict surah->ayat
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    t = item.get("text") or item.get("ayah") or item.get("content")
                    if t:
                        texts.append(str(t))
                elif isinstance(item, str):
                    texts.append(item)
        elif isinstance(data, dict):
            for v in data.values():
                if isinstance(v, str):
                    texts.append(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict) and item.get("text"):
                            texts.append(str(item["text"]))
                        elif isinstance(item, str):
                            texts.append(item)
        print(f"  + {path}: {len(texts)} نص")
        return texts
    except Exception as e:
        print(f"  ! {path}: {e}")
        return []


def main() -> int:
    ap = argparse.ArgumentParser(description="بناء قاموس WordTokenizer")
    ap.add_argument("--max-vocab", type=int, default=8192)
    ap.add_argument(
        "--out",
        default="models/transformer_ckg_v3/tokenizer_vocab.json",
        help="مسار حفظ القاموس",
    )
    args = ap.parse_args()

    from ai.arabic_transformer import WordTokenizer

    print("جمع النصوص من مصادر المشروع…")
    texts: list = []
    texts.extend(_load_sentence_files())
    texts.extend(_load_ckg_concepts())
    texts.extend(_load_quran_texts())

    if not texts:
        print("ABORT: لم يُعثر على أي نصوص لبناء القاموس.")
        return 1

    print(f"إجمالي النصوص: {len(texts)}")
    tok = WordTokenizer(vocab_size=args.max_vocab, vocab_path=None)
    n = tok.build_from_texts(texts, max_vocab=args.max_vocab)
    out = args.out
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    tok.save(out)
    print(f"✓ القاموس: {n} رمز → {out}")
    # عينة تحقق
    sample = "الصبر مفتاح الفرج"
    ids = tok.encode(sample)
    print(f"عينة: {sample!r} → {ids.tolist()} → {tok.decode(ids)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
