"""
تنظيف وتطبيع النص العربي — NSM (بدون تبعيات ثقيلة)

العمليات الشائعة:
  - إزالة التشكيل / التطويل
  - توحيد الألف والهمزات والياء/الألف المقصورة والتاء المربوطة
  - أرقام هندية → عربية غربية
  - تقليل التكرار (هههههه → هه)
  - إزالة روابط/إيموجي/HTML (اختياري)
  - أوضاع: search | display | dialect | strict

الاستخدام:
    from ai.arabic_text_clean import clean_arabic, normalize_for_search

    clean_arabic("بِسْــــمِ اللَّهِ!!! 😊 https://x.com")
    normalize_for_search("أحمد إبراهيم")
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional

# ── أنماط ──────────────────────────────────────────────────────────────
_TASHKEEL = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_TATWEEL = re.compile(r"\u0640+")
_HTML = re.compile(r"<[^>]+>")
_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001F9FF"
    "\U00002700-\U000027BF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "]+",
    flags=re.UNICODE,
)
_WHITESPACE = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{2,}")
_NON_AR_LATIN_DIGIT = re.compile(
    r"[^\s\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF"
    r"0-9a-zA-Z"
    r"\u0660-\u0669\u06F0-\u06F9"
    r".,!?;:،؟؛«»\"'()\[\]{}\-_/]"
)

# أرقام عربية-هندية وشرقية
_DIGIT_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")

# همزات وأشكال الألف
_ALEF_MAP = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
})
_TEH_MARBUTA = str.maketrans({"ة": "ه"})


def strip_tashkeel(text: str) -> str:
    return _TASHKEEL.sub("", text or "")


def strip_tatweel(text: str) -> str:
    return _TATWEEL.sub("", text or "")


def normalize_digits(text: str) -> str:
    return (text or "").translate(_DIGIT_MAP)


def normalize_letters(
    text: str,
    *,
    alef: bool = True,
    yeh: bool = True,
    teh_marbuta: bool = False,
) -> str:
    """توحيد حروف شائعة. teh_marbuta=False افتراضياً للحفاظ على المعنى في العرض."""
    t = text or ""
    if alef or yeh:
        # yeh ضمن _ALEF_MAP مع ى/ئ
        table = {}
        if alef:
            table.update({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"})
        if yeh:
            table.update({"ى": "ي", "ئ": "ي"})
        table.update({"ؤ": "و"})
        t = t.translate(str.maketrans(table))
    if teh_marbuta:
        t = t.translate(_TEH_MARBUTA)
    return t


def reduce_repeats(text: str, max_repeat: int = 2) -> str:
    """خاااااصة → خاصه (حد max_repeat)."""
    if max_repeat < 1:
        return text or ""
    return re.sub(r"(.)\1{" + str(max_repeat) + r",}", r"\1" * max_repeat, text or "")


def remove_urls(text: str) -> str:
    return _URL.sub(" ", text or "")


def remove_emails(text: str) -> str:
    return _EMAIL.sub(" ", text or "")


def remove_html(text: str) -> str:
    return _HTML.sub(" ", text or "")


def remove_emoji(text: str) -> str:
    return _EMOJI.sub(" ", text or "")


def collapse_ws(text: str) -> str:
    return _WHITESPACE.sub(" ", (text or "")).strip()


def remove_control_chars(text: str) -> str:
    return "".join(
        ch for ch in (text or "")
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )


def clean_arabic(
    text: str,
    *,
    mode: str = "display",
    strip_diacritics: bool = True,
    strip_elongation: bool = True,
    normalize_chars: bool = True,
    teh_marbuta_to_heh: bool = False,
    digits: bool = True,
    urls: bool = True,
    emails: bool = True,
    html: bool = True,
    emoji: bool = True,
    repeats: bool = True,
    max_repeat: int = 2,
    keep_latin: bool = True,
) -> str:
    """
    تنظيف موحّد.

    mode:
      - display  : لطيف للعرض (لا يحوّل ة→ه افتراضياً)
      - search   : أقصى توحيد للبحث/الفهرسة (ة→ه)
      - dialect  : يحافظ أكثر على طابع اللهجة (تكرار خفيف فقط)
      - strict   : حروف عربية+أرقام+مسافات فقط تقريباً
    """
    t = text or ""
    mode = (mode or "display").lower()

    if mode == "search":
        teh_marbuta_to_heh = True
        normalize_chars = True
        strip_diacritics = True
    elif mode == "dialect":
        # لا تُسقط التاء المربوطة؛ تقليل تكرار فقط
        teh_marbuta_to_heh = False
        max_repeat = max(max_repeat, 2)
    elif mode == "strict":
        keep_latin = False
        teh_marbuta_to_heh = True

    t = remove_control_chars(t)
    if html:
        t = remove_html(t)
    if urls:
        t = remove_urls(t)
    if emails:
        t = remove_emails(t)
    if emoji:
        t = remove_emoji(t)
    if strip_diacritics:
        t = strip_tashkeel(t)
    if strip_elongation:
        t = strip_tatweel(t)
    if digits:
        t = normalize_digits(t)
    if normalize_chars:
        t = normalize_letters(t, alef=True, yeh=True, teh_marbuta=teh_marbuta_to_heh)
    if repeats:
        t = reduce_repeats(t, max_repeat=max_repeat)

    if mode == "strict" or not keep_latin:
        t = _NON_AR_LATIN_DIGIT.sub(" ", t)
        # أزل اللاتيني إن strict
        if mode == "strict":
            t = re.sub(r"[a-zA-Z]+", " ", t)

    return collapse_ws(t)


def normalize_for_search(text: str) -> str:
    """توحيد قوي للمطابقة في CKG/البحث."""
    return clean_arabic(text, mode="search")


def normalize_for_dialect(text: str) -> str:
    """تنظيف خفيف قبل كشف/معالجة اللهجة."""
    return clean_arabic(text, mode="dialect", teh_marbuta_to_heh=False)


def tokenize_arabic_words(text: str) -> List[str]:
    """كلمات عربية بسيطة بعد تنظيف search."""
    t = normalize_for_search(text)
    return re.findall(r"[\u0600-\u06FF]+", t)


# توافق مع مسارات NSM الحالية
def normalize_arabic(text: str) -> str:
    """بديل خفيف لـ qa_engine.normalize_arabic عند الاستيراد من هنا."""
    return normalize_for_search(text)


if __name__ == "__main__":
    samples = [
        "بِسْــــمِ اللَّهِ الرَّحْمَٰنِ!!! 😊 https://example.com",
        "أحمد إبراهيم على ١٢٣",
        "خاااااصة ياخوي ايش الاخبار",
        "<b>مرحبا</b> بالعالم",
    ]
    for s in samples:
        print("IN :", s)
        print("DISP:", clean_arabic(s, mode="display"))
        print("SRCH:", clean_arabic(s, mode="search"))
        print("DIAL:", clean_arabic(s, mode="dialect"))
        print("---")
