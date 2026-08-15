"""test_fable_ui_import.py — يستورد ui_pages/fable.py ويتحقق أن الدوال والمكوّنات
الجديدة قابلة للاستيراد بدون أخطاء syntax/import (لا يحتاج مفاتيح API)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = FAIL = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"✅ [{PASS}] {name}")
    else:
        FAIL += 1
        print(f"❌ [{FAIL}] {name} {detail}")


def main() -> int:
    global PASS, FAIL

    # 1) استيراد المحرك
    from ai import fable_engine
    check("fable_engine يستورد", hasattr(fable_engine, "FableEngine"))
    check("rebuild_short_segments موجودة",
          hasattr(fable_engine.FableEngine, "rebuild_short_segments"))
    check("generate_short_social_description موجودة",
          hasattr(fable_engine.FableEngine, "generate_short_social_description"))

    # 2) استيراد صفحة الواجهة (streamlit dummy)
    try:
        from ui_pages import fable as fable_ui
        check("ui_pages/fable يستورد", True)
    except Exception as e:
        check("ui_pages/fable يستورد", False, str(e))
        print(f"\n{'='*50}\nالنتيجة: {PASS} ناجحة / {FAIL} فاشلة")
        return 1 if FAIL else 0

    check("render_fable موجودة", hasattr(fable_ui, "render_fable"))

    # 3) Source inspection: الأجزاء الجديدة موجودة نصّياً في الملف
    src = Path(fable_ui.__file__).read_text(encoding="utf-8")
    markers = [
        ("قصصي المحفوظة", "قصصي المحفوظة"),
        ("استئناف جلسة", "fable_resume_"),
        ("حذف جلسة", "fable_del_"),
        ("محرر اللقطات", "shorts_apply_edit"),
        ("استعادة السيناريو", "shorts_reset_edit"),
        ("معاينة صوت اللقطة الأولى", "shorts_voice_preview"),
        ("بطاقة النشر", "shorts_gen_card"),
        ("shorts_card_title", "shorts_card_title"),
    ]
    for label, key in markers:
        check(f"ui: {label}", key in src)

    # 4) لا تكرار keys في الواجهة (أخطاء streamlit المحتملة)
    import re
    keys = re.findall(r'key="([^"]+)"', src)
    dups = [k for k in set(keys) if keys.count(k) > 1]
    check("لا مفاتيح مكررة بالواجهة", not dups, f"مكررة: {dups}")

    print(f"\n{'='*50}\nالنتيجة: {PASS} ناجحة / {FAIL} فاشلة")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
