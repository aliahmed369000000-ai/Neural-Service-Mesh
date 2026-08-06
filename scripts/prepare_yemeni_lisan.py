#!/usr/bin/env python3
"""
استخراج وتصميم موارد دقة اللهجة اليمنية من data/yemeni/Yemeni.zip (Lisan-Yemeni).

المخرجات:
  data/yemeni/sentences.jsonl          — جمل لهجية
  data/yemeni/dialect_lexicon.json     — مفردات + تكرار + مقابل فصيح
  data/yemeni/msa_dialect_pairs.jsonl  — أزواج MSA ↔ Dialect
  data/yemeni/corpus_stats.json        — إحصاءات

الاستخدام:
  python3 scripts/prepare_yemeni_lisan.py
  python3 scripts/prepare_yemeni_lisan.py --zip data/yemeni/Yemeni.zip --max-sentences 80000
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import zipfile
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

OUT_DIR = "data/yemeni"
DEFAULT_ZIP = "data/yemeni/Yemeni.zip"

_TASHKEEL = re.compile(r"[\u064B-\u065F\u0670\u0640]")


def norm(t: str) -> str:
    t = _TASHKEEL.sub("", t or "")
    t = re.sub(r"[أإآٱ]", "ا", t)
    t = re.sub(r"[ىئ]", "ي", t)
    t = t.replace("ة", "ه")
    return t.strip()


def find_zip(path: str) -> Optional[str]:
    for p in (path, "data/yemeni/Yemeni.zip", "Yemeni.zip", "data/Yemeni.zip"):
        if os.path.exists(p):
            return p
    return None


def extract_sentences(zf: zipfile.ZipFile, max_n: int) -> List[str]:
    name = next((n for n in zf.namelist() if "RowText_sentences" in n and n.endswith(".csv")), None)
    if not name:
        return []
    raw = zf.read(name)
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if not text:
        return []
    reader = csv.DictReader(io.StringIO(text))
    out = []
    for row in reader:
        s = (row.get("sentence") or row.get("Sentence") or "").strip()
        # بعض الصفوف تبدأ بـ ": "
        if s.startswith(":"):
            s = s[1:].strip()
        if len(s) < 4:
            continue
        out.append(s)
        if len(out) >= max_n:
            break
    return out


def extract_lexicon(zf: zipfile.ZipFile, max_rows: int) -> Tuple[Dict, List[dict]]:
    name = next((n for n in zf.namelist() if n.endswith("Lisan-Yemeni-dataset.csv")), None)
    if not name:
        return {}, []
    # قراءة متدفقة للملف الكبير
    raw = zf.read(name)
    text = None
    for enc in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except Exception:
            pass
    if not text:
        return {}, []

    reader = csv.DictReader(io.StringIO(text))
    freq: Counter = Counter()
    # token -> {msa, gloss, pos, count}
    meta: Dict[str, dict] = {}
    pairs = []
    rows = 0
    for row in reader:
        rows += 1
        if rows > max_rows:
            break
        raw_tok = (row.get("rawToken") or row.get("Token") or "").strip()
        token = (row.get("Token") or raw_tok).strip()
        msa = (row.get("MSALemma") or "").strip()
        gloss = (row.get("Gloss") or "").strip()
        pos = (row.get("POS") or "").strip()
        stem = (row.get("Stem") or "").strip()
        if not raw_tok and not token:
            continue
        key = norm(raw_tok or token)
        if not key or len(key) < 2:
            continue
        freq[key] += 1
        if key not in meta:
            meta[key] = {
                "raw": raw_tok or token,
                "token": token,
                "msa": msa,
                "stem": stem,
                "pos": pos,
                "gloss": gloss,
                "count": 0,
            }
        meta[key]["count"] = freq[key]
        if msa and norm(msa) != key:
            pairs.append({
                "dialect": raw_tok or token,
                "dialect_norm": key,
                "msa": msa,
                "msa_norm": norm(msa),
                "gloss": gloss,
            })

    # lexicon: top by frequency
    lexicon = {
        "version": "lisan-yemeni-v1",
        "entries": [],
    }
    for w, c in freq.most_common(20000):
        e = meta.get(w, {"raw": w, "token": w})
        e = dict(e)
        e["count"] = c
        e["norm"] = w
        lexicon["entries"].append(e)
    return lexicon, pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--max-sentences", type=int, default=100000)
    ap.add_argument("--max-token-rows", type=int, default=500000)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()

    zpath = find_zip(args.zip)
    if not zpath:
        print(f"ABORT: لم يُعثر على Yemeni.zip (جرّب --zip)")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"قراءة {zpath}…")
    with zipfile.ZipFile(zpath) as zf:
        sentences = extract_sentences(zf, args.max_sentences)
        print(f"  جمل: {len(sentences)}")
        lexicon, pairs = extract_lexicon(zf, args.max_token_rows)
        print(f"  مداخل معجم: {len(lexicon.get('entries', []))}")
        print(f"  أزواج MSA↔لهجة: {len(pairs)}")

    sent_path = os.path.join(args.out_dir, "sentences.jsonl")
    with open(sent_path, "w", encoding="utf-8") as f:
        for i, s in enumerate(sentences):
            f.write(json.dumps({"id": i, "text": s}, ensure_ascii=False) + "\n")

    lex_path = os.path.join(args.out_dir, "dialect_lexicon.json")
    with open(lex_path, "w", encoding="utf-8") as f:
        json.dump(lexicon, f, ensure_ascii=False)

    pairs_path = os.path.join(args.out_dir, "msa_dialect_pairs.jsonl")
    # إزالة تكرار تقريبي
    seen = set()
    uniq_pairs = []
    for p in pairs:
        k = (p["dialect_norm"], p["msa_norm"])
        if k in seen:
            continue
        seen.add(k)
        uniq_pairs.append(p)
    with open(pairs_path, "w", encoding="utf-8") as f:
        for p in uniq_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    stats = {
        "zip": zpath,
        "sentences": len(sentences),
        "lexicon_entries": len(lexicon.get("entries", [])),
        "msa_dialect_pairs": len(uniq_pairs),
        "outputs": [sent_path, lex_path, pairs_path],
    }
    with open(os.path.join(args.out_dir, "corpus_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("✓ جاهز:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
