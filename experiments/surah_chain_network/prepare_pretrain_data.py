"""
تحضير بيانات Pre-training من الإنترنت (بدون CKG) — نسخة مقاومة لانقطاع الشبكة.
"""
from __future__ import annotations

import os
import pickle
import random
import re
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
CACHE_DIR = _HERE / "data"
CACHE_FILE = CACHE_DIR / "pretrain_sentences.pkl"
HF_DATASET = os.environ.get("SCN_HF_DATASET", "Jr23xd23/ArabicText-Large")
N = int(os.environ.get("SCN_N", "8000"))
MIN_CHARS = int(os.environ.get("SCN_MIN_CHARS", "40"))
MAX_CHARS = int(os.environ.get("SCN_MAX_CHARS", "512"))
SEED = int(os.environ.get("SCN_SEED", "42"))


def _split_to_segments(text: str) -> list[str]:
    if not text or not isinstance(text, str):
        return []
    text = text.strip()
    if len(text) < MIN_CHARS:
        return []
    parts = re.split(r"(?<=[.!?؟。\n])\s+", text)
    out = []
    buf = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(buf) + len(p) + 1 <= MAX_CHARS:
            buf = (buf + " " + p).strip() if buf else p
        else:
            if len(buf) >= MIN_CHARS:
                out.append(buf)
            buf = p if len(p) <= MAX_CHARS else p[:MAX_CHARS]
    if len(buf) >= MIN_CHARS:
        out.append(buf)
    return out


def _extract_text_field(example: dict) -> str:
    for key in ("text", "content", "article", "body", "raw_text", "paragraph"):
        v = example.get(key)
        if isinstance(v, str) and len(v.strip()) >= MIN_CHARS:
            return v.strip()
    title = example.get("title") or example.get("name") or ""
    body = example.get("text") or example.get("content") or ""
    combined = f"{title}\n{body}".strip()
    return combined if len(combined) >= MIN_CHARS else ""


def _save_partial(sentences: list[str], note: str = ""):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"[حفظ تدريجي{(' - ' + note) if note else ''}] {len(sentences)} مقطع → {CACHE_FILE}")


def load_and_prepare(max_n: int = N) -> list[str]:
    try:
        from datasets import load_dataset
    except ImportError:
        print("خطأ: مكتبة datasets غير مثبتة.\nثبّتها بـ: pip install datasets")
        sys.exit(1)

    print(f"جاري سحب البيانات من Hugging Face: {HF_DATASET}")
    print(f"الهدف: حتى {max_n} مقطع نصي (MIN_CHARS={MIN_CHARS})")

    try:
        ds = load_dataset(HF_DATASET, split="train", streaming=True)
    except Exception as e:
        print(f"فشل التحميل بـ streaming، محاولة عادية: {e}")
        ds = load_dataset(HF_DATASET, split="train")

    sentences: list[str] = []
    seen: set[str] = set()
    rng = random.Random(SEED)

    try:
        for i, ex in enumerate(ds):
            if len(sentences) >= max_n:
                break
            raw = _extract_text_field(ex if isinstance(ex, dict) else {})
            if not raw:
                continue
            for seg in _split_to_segments(raw):
                seg = re.sub(r"\s+", " ", seg).strip()
                if len(seg) < MIN_CHARS or seg in seen:
                    continue
                seen.add(seg)
                sentences.append(seg)
                if len(sentences) >= max_n:
                    break
            if (i + 1) % 500 == 0:
                print(f"  ... مرّ على {i+1} مقال → جُمع {len(sentences)} مقطع")
                _save_partial(sentences, note=f"بعد {i+1} مقال")
    except KeyboardInterrupt:
        print("\nتم الإيقاف يدوياً — يُحفظ الجزئي المتوفر.")
    except Exception as e:
        print(f"\nانقطع الاتصال أثناء السحب ({type(e).__name__}: {e})")
        print("يُحفظ الجزئي المتوفر بدل خسارة كل التقدم.")

    rng.shuffle(sentences)
    sentences = sentences[:max_n]
    print(f"النتيجة النهائية: {len(sentences)} مقطع")
    return sentences


def main():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if CACHE_FILE.exists() and os.environ.get("SCN_FORCE_REBUILD", "0") != "1":
        with open(CACHE_FILE, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, list) and len(cached) >= min(N, 100):
            print(f"كاش موجود مسبقاً: {CACHE_FILE} ({len(cached)} مقطع)")
            print("لحذف الكاش وإعادة البناء: SCN_FORCE_REBUILD=1")
            return

    sentences = load_and_prepare(N)
    if not sentences:
        print("لم يُستخرج أي نص — تحقق من اسم المجموعة أو الاتصال.")
        sys.exit(1)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(sentences, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"حُفظ الكاش النهائي: {CACHE_FILE} ({len(sentences)} مقطع)")


if __name__ == "__main__":
    main()
