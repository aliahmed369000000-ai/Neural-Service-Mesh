#!/usr/bin/env python3
"""
فحص تباين آلي (contrast check) لملف streamlit_app.py.

يفحص نمطين من الأخطاء الشائعة اللي تسبب اختفاء نص فوق خلفية:

1) عنصر HTML له style فيه background/background-color بلون hex ثابت
   (غير var(--...))، بدون تحديد color صريح بنفس الـ style — النص وقتها
   يرث لون الثيم العام، وقد يتطابق تقريباً مع الخلفية الثابتة في أحد
   الوضعين (فاتح/داكن) فيختفي. هذا بالضبط نمط الخطأ اللي انصلح سابقاً.

2) أزواج الألوان الثابتة في THEMES (bg/text) — يتأكد أن نسبة التباين
   (WCAG) بين النص والخلفية >= 4.5:1 (الحد الأدنى لنص عادي).

يخرج بكود 1 لو لقى أي مشكلة (لتفعيل فشل في CI)، و0 لو كله سليم.
"""

import ast
import re
import sys
from pathlib import Path

APP_FILE = Path(__file__).resolve().parent.parent / "streamlit_app.py"

# style="...background:#xxxxxx...  (بدون color: بنفس السلسلة)
STYLE_ATTR_RE = re.compile(r'style="([^"]*)"', re.DOTALL)
FIXED_BG_RE = re.compile(r'background(?:-color)?\s*:\s*#[0-9a-fA-F]{3,8}\b')
EXPLICIT_COLOR_RE = re.compile(r'(?<![-\w])color\s*:\s*(?!var\()')


def find_unpaired_fixed_backgrounds(text: str):
    """يرجّع قائمة (رقم السطر، مقتطف) لأي style فيه خلفية hex ثابتة
    وغير شفافة (opaque) بدون لون نص صريح بنفس الـ style attribute.
    يدعم style متعدد الأسطر (شائع في هذا الملف: background على سطر
    وcolor/إغلاق الاقتباس على سطر تالٍ). الخلفيات شبه الشفافة (8 خانات
    hex بقناة alpha منخفضة، مثل نمط var(--x-soft) الموجود أصلاً في
    المشروع) مستثناة عمداً لأنها تمتزج مع خلفية الثيم ولا تفرض تباينها
    الخاص."""
    issues = []
    for m in STYLE_ATTR_RE.finditer(text):
        style = m.group(1)
        bg_match = FIXED_BG_RE.search(style)
        if not bg_match or EXPLICIT_COLOR_RE.search(style):
            continue
        hex_val = bg_match.group(0).split("#", 1)[1]
        if len(hex_val) == 8:
            alpha = int(hex_val[6:8], 16)
            if alpha < 128:  # أقل من 50% تعتيم — شبه شفاف، نستثنيه
                continue
        lineno = text.count("\n", 0, m.start()) + 1
        snippet = " ".join(style.split())[:100]
        issues.append((lineno, snippet))
    return issues


def hex_to_rgb(h: str):
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def relative_luminance(rgb):
    def chan(c):
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (chan(c) for c in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(hex1: str, hex2: str) -> float:
    l1 = relative_luminance(hex_to_rgb(hex1))
    l2 = relative_luminance(hex_to_rgb(hex2))
    lighter, darker = max(l1, l2), min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def extract_themes_dict(text: str):
    """يستخرج THEMES = {...} كـ dict فعلي عبر ast، بدل regex هش."""
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "THEMES":
                    return ast.literal_eval(node.value)
    return None


def check_theme_contrast(themes: dict, min_ratio: float = 4.5):
    issues = []
    for theme_name, palette in themes.items():
        bg = palette.get("bg")
        text = palette.get("text")
        if not bg or not text:
            continue
        ratio = contrast_ratio(bg, text)
        if ratio < min_ratio:
            issues.append((theme_name, bg, text, ratio))
    return issues


def main() -> int:
    if not APP_FILE.exists():
        print(f"⚠️  الملف غير موجود: {APP_FILE}")
        return 1

    src = APP_FILE.read_text(encoding="utf-8")
    exit_code = 0

    style_issues = find_unpaired_fixed_backgrounds(src)
    if style_issues:
        exit_code = 1
        print(f"❌ {len(style_issues)} عنصر بخلفية ثابتة بدون لون نص صريح:\n")
        for lineno, snippet in style_issues:
            print(f"  سطر {lineno}: {snippet}")
        print()
    else:
        print("✅ لا توجد خلفيات ثابتة بدون لون نص صريح.")

    themes = extract_themes_dict(src)
    if themes:
        theme_issues = check_theme_contrast(themes)
        if theme_issues:
            exit_code = 1
            print(f"\n❌ {len(theme_issues)} ثيم بتباين أقل من 4.5:1:\n")
            for name, bg, text, ratio in theme_issues:
                print(f"  {name}: bg={bg} text={text} ratio={ratio:.2f}")
        else:
            print("✅ كل أزواج bg/text في THEMES تحقق تباين >= 4.5:1.")
    else:
        print("⚠️  تعذّر استخراج THEMES — تخطّي فحص التباين الأساسي.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
