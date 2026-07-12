"""
quran_lookup.py — بحث آية مباشر بالرقم (سورة:آية)، خفيف بما يكفي لدالة
serverless (Vercel) — بدون أي اعتماد على cognitive_graph.json (49 م.ب)
أو محرك answer_question الدلالي الكامل.

يستخدم فقط:
  - knowledge/quran_index.json  (8 ك.ب) — فهرس سورة → رقم أول chunk
  - knowledge/quran_chunk_NNNN.json (ملف واحد فقط يُحمَّل عند الطلب،
    ~40-220 ك.ب) — النص الفعلي للآيات

هذا يكفي لنطاق واتساب المحدود (بحث نص آية بالرقم)؛ لا يوفّر تفسيراً
ولا بحثاً دلالياً مفتوحاً — تلك خارج النطاق المخطط له عمداً.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict

ROOT = Path(__file__).resolve().parent.parent.parent
KNOWLEDGE_DIR = ROOT / "knowledge"

_index_cache: Optional[Dict] = None
_chunk_cache: Dict[int, list] = {}  # chunk_number -> list[dict] (ذاكرة تخزين مؤقت داخل نفس التشغيل الدافئ)


class AyahNotFound(ValueError):
    """السورة/الآية المطلوبة غير موجودة أو رقمها خارج النطاق الصحيح."""


def _load_index() -> Dict:
    global _index_cache
    if _index_cache is None:
        with open(KNOWLEDGE_DIR / "quran_index.json", encoding="utf-8") as f:
            _index_cache = json.load(f)
    return _index_cache


def _load_chunk(chunk_number: int) -> list:
    if chunk_number not in _chunk_cache:
        path = KNOWLEDGE_DIR / f"quran_chunk_{chunk_number:04d}.json"
        with open(path, encoding="utf-8") as f:
            _chunk_cache[chunk_number] = json.load(f)
    return _chunk_cache[chunk_number]


def get_ayah(surah: int, ayah: int) -> Dict:
    """يعيد {"surah": int, "ayah": int, "text": str} لآية محددة.
    يرفع AyahNotFound برسالة عربية واضحة لو رقم السورة/الآية غير صالح."""
    index = _load_index()
    surah_info = index.get("surah_index", {}).get(str(surah))
    if surah_info is None:
        raise AyahNotFound(f"لا توجد سورة برقم {surah} (السور من 1 إلى 114)")
    if not (1 <= ayah <= surah_info["ayah_count"]):
        raise AyahNotFound(
            f"سورة {surah} تحتوي {surah_info['ayah_count']} آية فقط — "
            f"لا توجد آية رقم {ayah}"
        )

    # الآيات مرتّبة تسلسلياً عبر السور بحجم chunk ثابت (chunk_size)؛ قد
    # تمتد سورة واحدة عبر أكثر من chunk، فنبحث بدءاً من first_chunk حتى نجدها
    chunk_number = surah_info["first_chunk"]
    while True:
        chunk = _load_chunk(chunk_number)
        for item in chunk:
            if item.get("surah") == surah and item.get("ayah") == ayah:
                return {"surah": surah, "ayah": ayah, "text": item["text"]}
        # لو انتهى هذا الـchunk بآيات من نفس السورة أو أقل، جرّب التالي
        if chunk and chunk[-1].get("surah", 0) <= surah:
            chunk_number += 1
            if chunk_number >= index.get("total_chunks", 0):
                break
            continue
        break

    raise AyahNotFound(f"تعذّر إيجاد الآية {surah}:{ayah} (خطأ بيانات غير متوقع)")


def parse_ayah_reference(text: str) -> Optional[tuple[int, int]]:
    """يحاول استخراج (سورة, آية) من نص حر بصيغة 'سورة:آية' فقط (مثال: '2:255').
    يعيد None لو النص لا يطابق الصيغة — لا يخمّن أسماء سور نصية بعد."""
    text = text.strip()
    if ":" not in text:
        return None
    parts = text.split(":")
    if len(parts) != 2:
        return None
    try:
        surah, ayah = int(parts[0].strip()), int(parts[1].strip())
        return surah, ayah
    except ValueError:
        return None
