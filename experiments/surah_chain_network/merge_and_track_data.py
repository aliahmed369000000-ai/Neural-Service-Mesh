#!/usr/bin/env python3
"""
دمج شاردات التجميع + سجل تتبّع لمنع التكرار في التدريب
=====================================================
- يدمج كل ملفات shard_*.pkl / *_final.pkl / *_partial.pkl
- يزيل التكرار ببصمة محتوى (hash)
- يحفظ:
    data/pretrain_sentences.pkl     ← البيانات الموحدة للتدريب
    data/data_registry.json         ← سجل: كم جملة، بصمات، مصدر كل دفعة
    data/training_cursor.json       ← أين وصل التدريب (جمل مُستخدمة)

الاستخدام:
  # دمج من مجلد محلي فيه الـpkls
  python experiments/surah_chain_network/merge_and_track_data.py --input-dir /path/to/pkls

  # دمج + تحديث سجل فقط
  python experiments/surah_chain_network/merge_and_track_data.py --input-dir ./collected

  # تعليم التدريب أنه استخدم N جملة (بعد جولة تدريب)
  python experiments/surah_chain_network/merge_and_track_data.py --mark-used 50000

  # عرض الحالة
  python experiments/surah_chain_network/merge_and_track_data.py --status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set

_HERE = Path(__file__).resolve().parent
DATA_DIR = _HERE / "data"
SENTENCES_FILE = DATA_DIR / "pretrain_sentences.pkl"
REGISTRY_FILE = DATA_DIR / "data_registry.json"
CURSOR_FILE = DATA_DIR / "training_cursor.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_hash(text: str) -> str:
    """بصمة قصيرة حتمية للجملة — لمنع التكرار عبر الشاردات والجولات."""
    return hashlib.sha1(text.strip().encode("utf-8", errors="ignore")).hexdigest()[:16]


def load_pkl(path: Path) -> List[str]:
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, list):
            return [s for s in data if isinstance(s, str) and len(s.strip()) >= 40]
        return []
    except Exception as e:
        print(f"  [تحذير] فشل قراءة {path.name}: {e}")
        return []


def load_registry() -> Dict[str, Any]:
    if REGISTRY_FILE.exists():
        return json.loads(REGISTRY_FILE.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "total_unique": 0,
        "batches": [],          # كل دفعة دمج: {at, source_files, added, total_after}
        "hashes": [],           # قائمة بصمات (قد تكون كبيرة — نحتفظ بآخر ملخص)
        "updated_at": None,
    }


def load_cursor() -> Dict[str, Any]:
    if CURSOR_FILE.exists():
        return json.loads(CURSOR_FILE.read_text(encoding="utf-8"))
    return {
        "version": 1,
        "used_count": 0,        # عدد الجمل التي دُرّب عليها (من بداية الملف الموحّد)
        "used_hashes_sample": [],  # عينة للتحقق
        "last_train_at": None,
        "runs": [],             # سجل جولات التدريب
    }


def save_registry(reg: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    reg["updated_at"] = _now()
    REGISTRY_FILE.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def save_cursor(cur: Dict[str, Any]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CURSOR_FILE.write_text(json.dumps(cur, ensure_ascii=False, indent=2), encoding="utf-8")


def merge_from_dir(input_dir: Path, min_chars: int = 40) -> Dict[str, Any]:
    """يدمج كل pkl في المجلد مع الموجود مسبقاً بدون تكرار."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # حمّل الموجود
    existing: List[str] = []
    if SENTENCES_FILE.exists():
        existing = load_pkl(SENTENCES_FILE)
        print(f"موجود مسبقاً: {len(existing)} جملة في {SENTENCES_FILE.name}")

    known: Set[str] = {content_hash(s) for s in existing}
    added = 0
    source_files = []

    pkl_files = sorted(input_dir.rglob("*.pkl"))
    if not pkl_files:
        print(f"لا ملفات .pkl في {input_dir}")
        return {"added": 0, "total": len(existing)}

    for path in pkl_files:
        rows = load_pkl(path)
        new_from_file = 0
        for s in rows:
            if len(s.strip()) < min_chars:
                continue
            h = content_hash(s)
            if h in known:
                continue
            known.add(h)
            existing.append(s.strip())
            added += 1
            new_from_file += 1
        source_files.append({"file": str(path.name), "rows": len(rows), "new": new_from_file})
        print(f"  + {path.name}: {len(rows)} صف → {new_from_file} جديدة")

    # حفظ الموحد
    with open(SENTENCES_FILE, "wb") as f:
        pickle.dump(existing, f, protocol=pickle.HIGHEST_PROTOCOL)

    reg = load_registry()
    reg["total_unique"] = len(existing)
    reg["batches"].append({
        "at": _now(),
        "source_files": source_files,
        "added": added,
        "total_after": len(existing),
    })
    # لا نخزن كل الـhashes في JSON (قد يصبح ضخماً) — فقط العدد + آخر دفعة
    reg["last_hash_count"] = len(known)
    save_registry(reg)

    size_mb = SENTENCES_FILE.stat().st_size / 1e6
    print(f"\n✓ الدمج: +{added} جديدة | الإجمالي الفريد: {len(existing)} | {size_mb:.1f} MB")
    print(f"  → {SENTENCES_FILE}")
    print(f"  → {REGISTRY_FILE}")
    return {"added": added, "total": len(existing)}


def mark_used(n: int, note: str = "") -> None:
    """يسجّل أن التدريب استخدم n جملة إضافية من بداية الملف الموحّد."""
    cur = load_cursor()
    prev = cur.get("used_count", 0)
    cur["used_count"] = prev + max(0, int(n))
    cur["last_train_at"] = _now()
    cur.setdefault("runs", []).append({
        "at": _now(),
        "used_this_run": int(n),
        "used_total": cur["used_count"],
        "note": note,
    })
    save_cursor(cur)
    print(f"✓ training_cursor: كان {prev} → أصبح {cur['used_count']} (هذه الجولة +{n})")


def show_status() -> None:
    reg = load_registry()
    cur = load_cursor()
    n_sent = 0
    if SENTENCES_FILE.exists():
        n_sent = len(load_pkl(SENTENCES_FILE))
    used = cur.get("used_count", 0)
    remaining = max(0, n_sent - used)
    print("=== حالة بيانات SurahChain Pretrain ===")
    print(f"  جمل فريدة مدموجة : {n_sent}")
    print(f"  استُخدمت في تدريب: {used}")
    print(f"  متبقية (غير مكررة): {remaining}")
    print(f"  دفعات دمج         : {len(reg.get('batches', []))}")
    if reg.get("updated_at"):
        print(f"  آخر دمج           : {reg['updated_at']}")
    if cur.get("last_train_at"):
        print(f"  آخر تدريب         : {cur['last_train_at']}")
    print(f"  ملف البيانات      : {SENTENCES_FILE}")
    print(f"  سجل الدمج         : {REGISTRY_FILE}")
    print(f"  مؤشر التدريب      : {CURSOR_FILE}")


def main():
    ap = argparse.ArgumentParser(description="دمج شاردات + تتبّع تدريب بدون تكرار")
    ap.add_argument("--input-dir", type=str, default="", help="مجلد فيه ملفات .pkl للدمج")
    ap.add_argument("--mark-used", type=int, default=0, help="سجّل أن التدريب استخدم N جملة")
    ap.add_argument("--note", type=str, default="", help="ملاحظة مع --mark-used")
    ap.add_argument("--status", action="store_true", help="عرض الحالة فقط")
    ap.add_argument("--min-chars", type=int, default=40)
    args = ap.parse_args()

    if args.status or (not args.input_dir and not args.mark_used):
        show_status()
        return 0

    if args.input_dir:
        d = Path(args.input_dir)
        if not d.is_dir():
            print(f"المجلد غير موجود: {d}")
            return 1
        merge_from_dir(d, min_chars=args.min_chars)

    if args.mark_used > 0:
        mark_used(args.mark_used, note=args.note)

    show_status()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
