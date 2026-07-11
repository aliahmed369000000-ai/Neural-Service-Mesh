"""
Parseltongue — محرك إخفاء المدخلات
===========================================
يرصد الكلمات المثيرة لرفض النماذج ويطبّق تقنيات تشفير لتجاوز فلاتر المحتوى.
Python port من src/lib/parseltongue.ts في G0DM0D3-main.

التقنيات:
  - leetspeak   : تحويل الحروف لرموز l33t (a→4, e→3, ...)
  - unicode     : مثيلات Unicode بصرياً متشابهة (Cyrillic, Greek, ...)
  - zwj         : إدراج محارف عرض-صفري غير مرئية بين الأحرف
  - mixedcase   : اضطراب نمط الأحرف الكبيرة/الصغيرة
  - phonetic    : استبدال صوتي بكلمات مشابهة لفظاً
  - random      : خلط عشوائي من جميع التقنيات
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import List, Literal, Dict

# ══════════════════════════════════════════════════════════════════════
# الأنواع
# ══════════════════════════════════════════════════════════════════════

ObfuscationTechnique = Literal["leetspeak", "unicode", "zwj", "mixedcase", "phonetic", "random"]

INTENSITY_RATES: Dict[str, float] = {
    "light":  0.25,  # 25% من الأحرف المؤهلة
    "medium": 0.55,  # 55%
    "heavy":  0.90,  # 90%
}


@dataclass
class ParseltongueConfig:
    enabled: bool = False
    technique: ObfuscationTechnique = "leetspeak"
    intensity: Literal["light", "medium", "heavy"] = "medium"
    custom_triggers: List[str] = field(default_factory=list)


@dataclass
class TransformRecord:
    original: str
    transformed: str
    technique: str


@dataclass
class ParseltongueResult:
    original_text: str
    transformed_text: str
    triggers_found: List[str]
    technique_used: str
    transformations: List[TransformRecord]


# ══════════════════════════════════════════════════════════════════════
# قائمة الكلمات المثيرة الافتراضية
# ══════════════════════════════════════════════════════════════════════

DEFAULT_TRIGGERS: List[str] = [
    # أفعال الاختراق
    "hack", "exploit", "bypass", "crack", "break", "attack", "penetrate",
    "inject", "manipulate", "override", "disable", "circumvent", "evade",
    # مصطلحات أمنية
    "malware", "virus", "trojan", "payload", "shellcode", "rootkit",
    "keylogger", "backdoor", "vulnerability",
    # مواضيع حساسة
    "weapon", "bomb", "explosive", "poison", "drug", "synthesize",
    # مصطلحات النظام
    "jailbreak", "unlock", "root", "sudo", "admin", "privilege",
    # هندسة اجتماعية
    "phishing", "scam", "impersonate", "deceive", "fraud",
    # إشارات المحتوى
    "nsfw", "explicit", "uncensored", "unfiltered", "unrestricted",
    # أوامر الذكاء الاصطناعي
    "ignore", "disregard", "forget", "pretend", "roleplay",
    "act as", "you are now", "new identity",
]

# ══════════════════════════════════════════════════════════════════════
# جداول التحويل
# ══════════════════════════════════════════════════════════════════════

LEET_MAP: Dict[str, List[str]] = {
    "a": ["4", "@", "∂", "λ"],
    "b": ["8", "|3", "ß"],
    "c": ["(", "<", "¢"],
    "d": ["|)", "|>", "đ"],
    "e": ["3", "€", "£", "∑"],
    "f": ["|=", "ƒ"],
    "g": ["9", "6", "&"],
    "h": ["#", "|-|", "}{"],
    "i": ["1", "!", "|", "¡"],
    "j": ["_|", "]"],
    "k": ["|<", "|{", "κ"],
    "l": ["1", "|", "£"],
    "m": ["|V|", "µ"],
    "n": ["|\\|", "η"],
    "o": ["0", "()", "°", "ø"],
    "p": ["|*", "|>", "þ"],
    "q": ["0_", "ℚ"],
    "r": ["|2", "®"],
    "s": ["5", "$", "§", "∫"],
    "t": ["7", "+", "†"],
    "u": ["|_|", "µ", "ü"],
    "v": ["\\/", "√"],
    "w": ["\\/\\/", "vv", "ω"],
    "x": ["><", "×"],
    "y": ["`/", "¥", "γ"],
    "z": ["2", "7_", "ℤ"],
}

UNICODE_HOMOGLYPHS: Dict[str, List[str]] = {
    "a": ["а", "ɑ", "α", "ａ"],     # Cyrillic а
    "b": ["Ь", "ｂ"],
    "c": ["с", "ϲ", "ｃ"],           # Cyrillic с
    "d": ["ԁ", "ｄ"],
    "e": ["е", "ė", "ｅ"],           # Cyrillic е
    "f": ["ƒ", "ｆ"],
    "g": ["ɡ", "ｇ"],
    "h": ["һ", "ｈ"],               # Cyrillic һ
    "i": ["і", "ι", "ｉ"],          # Cyrillic і
    "j": ["ϳ", "ｊ"],
    "k": ["κ", "ｋ"],
    "l": ["ӏ", "ｌ"],               # Cyrillic palochka
    "m": ["м", "ｍ"],
    "n": ["ո", "ｎ"],
    "o": ["о", "ο", "ｏ"],          # Cyrillic о
    "p": ["р", "ρ", "ｐ"],          # Cyrillic р
    "s": ["ѕ", "ｓ"],               # Cyrillic ѕ
    "t": ["τ", "ｔ"],
    "u": ["υ", "ｕ"],
    "v": ["ν", "ｖ"],
    "w": ["ѡ", "ｗ"],
    "x": ["х", "ｘ"],               # Cyrillic х
    "y": ["у", "γ", "ｙ"],          # Cyrillic у
    "z": ["ᴢ", "ｚ"],
}

# محارف عرض-صفري غير مرئية
ZWJ_CHARS = [
    "\u200d",  # Zero-Width Joiner
    "\u200b",  # Zero-Width Space
    "\u200c",  # Zero-Width Non-Joiner
    "\u2060",  # Word Joiner
    "\ufeff",  # Zero-Width No-Break Space
]

# الاستبدال الصوتي
PHONETIC_MAP: Dict[str, str] = {
    "hack": "h4ck",
    "exploit": "3xpl0it",
    "crack": "kr4ck",
    "bomb": "b0mb",
    "weapon": "w34p0n",
    "virus": "viru5",
    "malware": "m4lw4re",
    "rootkit": "r00tk1t",
    "backdoor": "b4ckd00r",
    "jailbreak": "j4ilbr3ak",
    "synthesize": "synth3siz3",
    "poison": "p01s0n",
    "phishing": "ph1sh1ng",
    "impersonate": "imp3rs0n4te",
    "manipulate": "m4n1pul4te",
}


# ══════════════════════════════════════════════════════════════════════
# محركات التحويل
# ══════════════════════════════════════════════════════════════════════

def _apply_leet(word: str, rate: float) -> str:
    """تحويل حرف إلى l33t حسب معدل التكثيف."""
    result = list(word)
    for i, ch in enumerate(result):
        if random.random() < rate and ch.lower() in LEET_MAP:
            options = LEET_MAP[ch.lower()]
            repl = random.choice(options)
            result[i] = repl if not ch.isupper() else repl.upper()
    return "".join(result)


def _apply_unicode(word: str, rate: float) -> str:
    """استبدال أحرف بمثيلات Unicode بصرياً متشابهة."""
    result = list(word)
    for i, ch in enumerate(result):
        if random.random() < rate and ch.lower() in UNICODE_HOMOGLYPHS:
            options = UNICODE_HOMOGLYPHS[ch.lower()]
            result[i] = random.choice(options)
    return "".join(result)


def _apply_zwj(word: str, rate: float) -> str:
    """إدراج محارف عرض-صفري غير مرئية بين أحرف الكلمة."""
    if len(word) <= 2:
        return word
    result = [word[0]]
    for ch in word[1:]:
        if random.random() < rate:
            result.append(random.choice(ZWJ_CHARS))
        result.append(ch)
    return "".join(result)


def _apply_mixedcase(word: str, _rate: float) -> str:
    """اضطراب نمط الأحرف الكبيرة والصغيرة."""
    return "".join(
        ch.upper() if i % 2 == 0 else ch.lower()
        for i, ch in enumerate(word)
    )


def _apply_phonetic(word: str, _rate: float) -> str:
    """استبدال صوتي بناءً على قاموس محدد مسبقاً."""
    lower = word.lower()
    if lower in PHONETIC_MAP:
        repl = PHONETIC_MAP[lower]
        # المحافظة على حالة الحرف الأول
        return repl.capitalize() if word[0].isupper() else repl
    # الرجوع إلى l33t إذا لا يوجد استبدال صوتي
    return _apply_leet(word, 0.6)


_TECHNIQUE_FNS = {
    "leetspeak": _apply_leet,
    "unicode":   _apply_unicode,
    "zwj":       _apply_zwj,
    "mixedcase": _apply_mixedcase,
    "phonetic":  _apply_phonetic,
}


def _pick_random_fn():
    """اختيار دالة تحويل عشوائية من التقنيات المتاحة."""
    name = random.choice(["leetspeak", "unicode", "zwj", "mixedcase", "phonetic"])
    return name, _TECHNIQUE_FNS[name]


# ══════════════════════════════════════════════════════════════════════
# الدالة الرئيسية
# ══════════════════════════════════════════════════════════════════════

# ── شيم التوافق مع الإصدارات القديمة ─────────────────────────────────────

# قاموس الوصف (للواجهات القديمة)
TECHNIQUE_DESCRIPTIONS: Dict[str, str] = {
    "leetspeak": "l33tspeak كلاسيكي: a→4, e→3, i→1, ...",
    "unicode":   "مثيلات Unicode (سيريلية، يونانية)",
    "zwj":       "محارف عرض-صفري غير مرئية بين الأحرف",
    "mixedcase": "اضطراب نمط الأحرف الكبيرة/الصغيرة",
    "phonetic":  "استبدال صوتي بكلمات مشابهة لفظاً",
    "random":    "خلط عشوائي من جميع التقنيات",
}


def detect_triggers(text: str, custom_triggers: List[str] | None = None) -> List[str]:
    """يكتشف الكلمات المثيرة في النص ويُعيدها كقائمة."""
    triggers = list(set(DEFAULT_TRIGGERS + (custom_triggers or [])))
    lower = text.lower()
    return [t for t in triggers if t.lower() in lower]


def apply_parseltongue(  # type: ignore[override]
    text: str,
    config: ParseltongueConfig | None = None,
    *,
    technique: str | None = None,
    intensity: str | None = None,
    enabled: bool | None = None,
) -> ParseltongueResult:
    """
    تطبيق تقنية الإخفاء على النص.
    يدعم أسلوبين: apply_parseltongue(text, config) أو apply_parseltongue(text, technique=t, intensity=i, enabled=True)
    """
    # بناء config من الـ keyword args إذا لم يُمرَّر config صريحاً
    if config is None:
        config = ParseltongueConfig(
            enabled=enabled if enabled is not None else True,
            technique=technique or "leetspeak",   # type: ignore[arg-type]
            intensity=intensity or "medium",       # type: ignore[arg-type]
        )

    if not config.enabled:
        return ParseltongueResult(
            original_text=text,
            transformed_text=text,
            triggers_found=[],
            technique_used="none",
            transformations=[],
        )

    # دمج الكلمات المثيرة الافتراضية + المخصصة
    all_triggers = list(set(DEFAULT_TRIGGERS + config.custom_triggers))
    rate = INTENSITY_RATES.get(config.intensity, 0.55)
    technique = config.technique

    found_triggers: List[str] = []
    transformations: List[TransformRecord] = []
    result_text = text

    for trigger in sorted(all_triggers, key=len, reverse=True):  # الأطول أولاً
        # البحث الغير حساس للحالة
        pattern = re.compile(r"\b" + re.escape(trigger) + r"\b", re.IGNORECASE)
        matches = pattern.findall(result_text)
        if not matches:
            continue

        found_triggers.append(trigger)

        def _replace_match(m: re.Match) -> str:  # noqa: E501
            word = m.group(0)
            actual_technique = technique
            if technique == "random":
                actual_technique, fn = _pick_random_fn()
                transformed = fn(word, rate)
            else:
                fn = _TECHNIQUE_FNS[technique]
                transformed = fn(word, rate)
            transformations.append(TransformRecord(word, transformed, actual_technique))
            return transformed

        result_text = pattern.sub(_replace_match, result_text)

    return ParseltongueResult(
        original_text=text,
        transformed_text=result_text,
        triggers_found=list(set(found_triggers)),
        technique_used=technique,
        transformations=transformations,
    )


def get_technique_description(technique: str) -> str:
    descriptions = {
        "leetspeak": "l33tspeak كلاسيكي: a→4, e→3, i→1, ...",
        "unicode":   "مثيلات Unicode (سيريلية، يونانية)",
        "zwj":       "محارف عرض-صفري غير مرئية بين الأحرف",
        "mixedcase": "اضطراب نمط الأحرف الكبيرة/الصغيرة",
        "phonetic":  "استبدال صوتي بكلمات مشابهة لفظاً",
        "random":    "خلط عشوائي من جميع التقنيات",
        "none":      "بدون تحويل",
    }
    return descriptions.get(technique, technique)
