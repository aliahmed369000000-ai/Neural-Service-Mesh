#!/usr/bin/env python3
"""
تدريب Byte-level BPE متعدد اللغات (محايد السكربت).

المصادر الاختيارية (ما وُجد يُستخدم):
  - جمل CKG العربية (ckg_sentences_*.pkl)
  - مفاهيم cognitive_graph.json
  - أرشيف Lisan-Yemeni: data/yemeni/Yemeni.zip
  - ملفات نصية إضافية: --extra path1.txt path2.txt
  - نص مباشر عبر stdin إن مُرِّر --stdin

  python3 scripts/train_byte_bpe_multilingual.py
  python3 scripts/train_byte_bpe_multilingual.py --vocab-size 16000 --out models/byte_bpe_multi.json
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pickle
import sys
import zipfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def from_pkl() -> list:
    out = []
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
                out.extend(str(x) for x in data if x)
                print(f"  + {path}: {len(data)}")
        except Exception as e:
            print(f"  ! {path}: {e}")
    return out


def from_ckg() -> list:
    path = "knowledge/cognitive_graph.json"
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        concepts = data.get("concepts") or {}
        if isinstance(concepts, dict):
            names = list(map(str, concepts.keys()))
            print(f"  + {path}: {len(names)}")
            return names
    except Exception as e:
        print(f"  ! {path}: {e}")
    return []


def from_yemeni_zip(max_rows: int = 50000) -> list:
    """يستخرج عمود النص من Lisan-Yemeni داخل Yemeni.zip إن وُجد."""
    candidates = [
        "data/yemeni/Yemeni.zip",
        "Yemeni.zip",
        "data/Yemeni.zip",
    ]
    zpath = next((p for p in candidates if os.path.exists(p)), None)
    if not zpath:
        print("  · Yemeni.zip غير موجود (تخطّي)")
        return []
    texts = []
    try:
        with zipfile.ZipFile(zpath) as zf:
            # جمل صفّية إن وُجدت
            sent_name = None
            for n in zf.namelist():
                if "RowText_sentences" in n and n.endswith(".csv"):
                    sent_name = n
                    break
            if sent_name:
                raw = zf.read(sent_name)
                # محاولة utf-8 ثم cp1256
                for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
                    try:
                        text = raw.decode(enc)
                        break
                    except Exception:
                        text = None
                if text:
                    reader = csv.reader(io.StringIO(text))
                    rows = 0
                    for row in reader:
                        if not row:
                            continue
                        # خذ أطول خلية كنص
                        cell = max(row, key=lambda c: len(c or ""))
                        if cell and len(cell.strip()) > 1:
                            texts.append(cell.strip())
                            rows += 1
                        if rows >= max_rows:
                            break
                    print(f"  + {zpath}::{sent_name}: {rows} سطر")
            # عيّنة من dataset الكامل
            ds_name = next((n for n in zf.namelist() if n.endswith("Lisan-Yemeni-dataset.csv")), None)
            if ds_name and len(texts) < max_rows:
                raw = zf.read(ds_name)
                for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
                    try:
                        text = raw.decode(enc)
                        break
                    except Exception:
                        text = None
                if text:
                    reader = csv.reader(io.StringIO(text))
                    header = next(reader, None)
                    # ابحث عن عمود نصي
                    text_idx = 0
                    if header:
                        for i, h in enumerate(header):
                            hl = (h or "").lower()
                            if any(k in hl for k in ("text", "sentence", "content", "نص", "جملة")):
                                text_idx = i
                                break
                    added = 0
                    for row in reader:
                        if len(row) > text_idx and row[text_idx].strip():
                            texts.append(row[text_idx].strip())
                            added += 1
                        if len(texts) >= max_rows:
                            break
                    print(f"  + {zpath}::{ds_name}: +{added} (إجمالي {len(texts)})")
    except Exception as e:
        print(f"  ! Yemeni.zip: {e}")
    return texts


def from_extra(paths: list) -> list:
    out = []
    for path in paths or []:
        if not os.path.exists(path):
            print(f"  ! missing {path}")
            continue
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            out.extend(lines)
            print(f"  + {path}: {len(lines)}")
        except Exception as e:
            print(f"  ! {path}: {e}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="تدريب Byte-BPE متعدد اللغات")
    ap.add_argument("--vocab-size", type=int, default=16000)
    ap.add_argument("--num-merges", type=int, default=None)
    ap.add_argument("--yemeni-rows", type=int, default=30000)
    ap.add_argument("--extra", nargs="*", default=[])
    ap.add_argument("--stdin", action="store_true")
    ap.add_argument("--out", default="models/byte_bpe_multilingual.json")
    ap.add_argument("--demo", action="store_true", help="اختبار decode بعد التدريب")
    args = ap.parse_args()

    from ai.byte_bpe_tokenizer import ByteBPETokenizer

    print("جمع نصوص متعددة المصادر/اللغات…")
    texts: list = []
    texts.extend(from_pkl())
    texts.extend(from_ckg())
    texts.extend(from_yemeni_zip(max_rows=args.yemeni_rows))
    texts.extend(from_extra(args.extra))
    if args.stdin:
        stdin_lines = [ln.strip() for ln in sys.stdin if ln.strip()]
        texts.extend(stdin_lines)
        print(f"  + stdin: {len(stdin_lines)}")

    # عيّنة متعددة اللغات مضمّنة لضمان تغطية بايتات غير عربية حتى بدون ملفات
    seed_multi = [
        "Hello world",
        "Bonjour le monde",
        "Hola mundo",
        "Guten Tag",
        "Ciao mondo",
        "こんにちは世界",
        "안녕하세요",
        "Привет мир",
        "שלום עולם",
        "مرحبا بالعالم",
        "Python 3.12 — UTF-8 bytes",
    ]
    texts.extend(seed_multi)
    print(f"  + seed multilingual: {len(seed_multi)}")

    if not texts:
        print("ABORT: لا نصوص")
        return 1

    print(f"إجمالي النصوص: {len(texts)}")
    tok = ByteBPETokenizer(vocab_size=args.vocab_size)
    n = tok.train(texts, num_merges=args.num_merges)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tok.save(args.out)
    print(f"✓ Byte-BPE multilingual vocab={n} merges={len(tok.merges)} → {args.out}")

    if args.demo:
        samples = [
            "الصبر مفتاح الفرج",
            "Hello from NSM",
            "اللهجة اليمنية غنية",
            "こんにちは",
            "mixed عربي English 123",
        ]
        print("عرض decode:")
        for s in samples:
            print(f"  {s!r} → {tok.decode(tok.encode(s))!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
