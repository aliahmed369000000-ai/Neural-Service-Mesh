"""
quran_lookup.py — بحث آية مباشر بالرقم (سورة:آية)، خفيف بما يكفي لدالة
serverless (Vercel).

⚠️ هذا الملف نسخة طبق الأصل من ai/whatsapp/quran_lookup.py بالمستودع
الرئيسي، منقولة هنا عمداً (وليس مستوردة عبر ai.*) لأن whatsapp_gateway/
مشروع Vercel منفصل بجذر مستقل (راجع README.md بهذا المجلد) — لا رؤية
له لأي شي خارج هذا المجلد على الإطلاق. أي تعديل منطقي هنا يجب تكراره
يدوياً بالنسخة الأصلية بـai/whatsapp/quran_lookup.py والعكس.

يستخدم فقط:
  - knowledge/quran_index.json  (8 ك.ب)
  - knowledge/quran_chunk_NNNN.json (ملف واحد فقط عند الطلب)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict

KNOWLEDGE_DIR = Path(__file__).resolve().parent.parent / "knowledge"

_index_cache: Optional[Dict] = None
_chunk_cache: Dict[int, list] = {}


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
    index = _load_index()
    surah_info = index.get("surah_index", {}).get(str(surah))
    if surah_info is None:
        raise AyahNotFound(f"لا توجد سورة برقم {surah} (السور من 1 إلى 114)")
    if not (1 <= ayah <= surah_info["ayah_count"]):
        raise AyahNotFound(
            f"سورة {surah} تحتوي {surah_info['ayah_count']} آية فقط — "
            f"لا توجد آية رقم {ayah}"
        )

    chunk_number = surah_info["first_chunk"]
    while True:
        chunk = _load_chunk(chunk_number)
        for item in chunk:
            if item.get("surah") == surah and item.get("ayah") == ayah:
                return {"surah": surah, "ayah": ayah, "text": item["text"]}
        if chunk and chunk[-1].get("surah", 0) <= surah:
            chunk_number += 1
            if chunk_number >= index.get("total_chunks", 0):
                break
            continue
        break

    raise AyahNotFound(f"تعذّر إيجاد الآية {surah}:{ayah} (خطأ بيانات غير متوقع)")


def parse_ayah_reference(text: str) -> Optional[tuple[int, int]]:
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
