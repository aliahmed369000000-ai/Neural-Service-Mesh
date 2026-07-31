"""
check_whatsapp_gateway_sync.py — يتأكد إن نسخة بيانات القرآن داخل
whatsapp_gateway/knowledge/ لسه متطابقة بايت لبايت مع النسخة الأصلية
بـknowledge/ بجذر المستودع.

السياق: whatsapp_gateway/ مشروع Vercel منفصل بجذر مستقل (راجع
whatsapp_gateway/README.md) — لا رؤية له لأي شي خارج مجلده، فبيانات
القرآن (quran_index.json + quran_chunk_NNNN.json) منسوخة هناك عمداً
وليست مستوردة. هذا التكرار مقصود ومبرر معمارياً، لكنه بلا أي حماية آلية
ضد الانحراف (drift): لو حدّث أحد ملفات القرآن بـknowledge/ الرئيسي ونسي
ينسخها لـwhatsapp_gateway/knowledge/، بوت واتساب يرجع يعرض آيات قديمة
أو ناقصة بصمت تام بدون أي خطأ ظاهر.

هذا السكربت يسد تلك الفجوة: يقارن كل ملف موجود بالنسختين ويفشل (exit 1)
فوراً لو لقى أي اختلاف، بدل الاعتماد فقط على تذكير نصي بالتعليقات.

الاستخدام: python3 scripts/check_whatsapp_gateway_sync.py
"""
from __future__ import annotations

import filecmp
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_KNOWLEDGE = ROOT / "knowledge"
GATEWAY_KNOWLEDGE = ROOT / "whatsapp_gateway" / "knowledge"

# الملفات اللي يفترض تكون منسوخة حرفياً (بيانات، وليست كود فيه فروقات
# تعليقات متوقعة). مطابق تماماً لما هو موثّق بداخل docstring الخاص
# بـwhatsapp_gateway/lib/quran_lookup.py.
DATA_FILE_PATTERNS = ("quran_index.json", "quran_chunk_*.json")


def find_synced_files() -> list[str]:
    names: set[str] = set()
    for pattern in DATA_FILE_PATTERNS:
        names.update(p.name for p in GATEWAY_KNOWLEDGE.glob(pattern))
    return sorted(names)


def main() -> int:
    if not GATEWAY_KNOWLEDGE.exists():
        print(f"⚠️  {GATEWAY_KNOWLEDGE} غير موجود — تخطّي الفحص.")
        return 0

    filenames = find_synced_files()
    if not filenames:
        print("⚠️  لا توجد ملفات بيانات قرآن بـwhatsapp_gateway/knowledge/ للمقارنة.")
        return 0

    missing: list[str] = []
    mismatched: list[str] = []

    for name in filenames:
        main_path = MAIN_KNOWLEDGE / name
        gw_path = GATEWAY_KNOWLEDGE / name
        if not main_path.exists():
            missing.append(name)
            continue
        if not filecmp.cmp(main_path, gw_path, shallow=False):
            mismatched.append(name)

    if missing:
        print("❌ ملفات موجودة بـwhatsapp_gateway/knowledge/ لكن غائبة عن knowledge/ الرئيسي:")
        for name in missing:
            print(f"   - {name}")

    if mismatched:
        print("❌ ملفات غير متطابقة بين knowledge/ وwhatsapp_gateway/knowledge/ (انحراف/drift):")
        for name in mismatched:
            print(f"   - {name}")
        print(
            "\nالحل: انسخ النسخة المحدّثة من knowledge/ إلى whatsapp_gateway/knowledge/ "
            "يدوياً، ثم ادفع التعديلين معاً بنفس الـcommit."
        )

    if missing or mismatched:
        return 1

    print(f"✅ {len(filenames)} ملف بيانات قرآن متطابق تماماً بين knowledge/ وwhatsapp_gateway/knowledge/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
