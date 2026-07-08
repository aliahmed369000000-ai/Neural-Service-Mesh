"""
Parseltongue — محرك التشويه (Python port من G0DM0D3)
يكتشف الكلمات المُشغِّلة ويطبّق تقنيات التشويه عليها.
"""

import re
import random
from dataclasses import dataclass, field
from typing import Literal, List, Dict, Optional

ObfuscationTechnique = Literal["leetspeak", "unicode", "zwj", "mixedcase", "phonetic", "random"]
Intensity = Literal["light", "medium", "heavy"]

DEFAULT_TRIGGERS: List[str] = [
    "hack", "exploit", "bypass", "crack", "break", "attack", "penetrate",
    "inject", "manipulate", "override", "disable", "circumvent", "evade",
    "malware", "virus", "trojan", "payload", "shellcode", "rootkit",
    "keylogger", "backdoor", "vulnerability",
    "weapon", "bomb", "explosive", "poison", "drug", "synthesize",
    "jailbreak", "unlock", "root", "sudo", "admin", "privilege",
    "phishing", "scam", "impersonate", "deceive", "fraud",
    "nsfw", "explicit", "uncensored", "unfiltered", "unrestricted",
    "ignore", "disregard", "forget", "pretend", "roleplay",
]

LEET_MAP: Dict[str, List[str]] = {
    "a": ["4", "@", "∂", "λ"], "b": ["8", "|3", "ß"],
    "c": ["(", "<", "¢"],      "d": ["|)", "|>", "đ"],
    "e": ["3", "€", "£", "∑"], "f": ["|=", "ƒ"],
    "g": ["9", "6", "&"],      "h": ["#", "|-|", "}{"],
    "i": ["1", "!", "|"],      "j": ["_|", "]"],
    "k": ["|<", "|{"],         "l": ["1", "|", "|_"],
    "m": ["|V|", "µ"],         "n": ["|\\|", "η"],
    "o": ["0", "()", "°", "ø"],"p": ["|*", "|>"],
    "q": ["0_", "()_"],        "r": ["|2", "®"],
    "s": ["5", "$", "§"],      "t": ["7", "+", "†"],
    "u": ["|_|", "µ"],         "v": ["\\/", "√"],
    "w": ["\\/\\/", "vv", "ω"],"x": ["><", "×"],
    "y": ["`/", "¥", "γ"],     "z": ["2", "7_"],
}

UNICODE_HOMOGLYPHS: Dict[str, List[str]] = {
    "a": ["а", "ɑ", "α"],  "b": ["Ь", "ḅ"],
    "c": ["с", "ϲ"],        "d": ["ԁ", "ⅾ"],
    "e": ["е", "ė", "ẹ"],  "f": ["ƒ"],
    "g": ["ɡ"],              "h": ["һ", "ḥ"],
    "i": ["і", "ι"],         "j": ["ϳ"],
    "k": ["κ"],               "l": ["ӏ", "ⅼ"],
    "m": ["м"],               "n": ["ո"],
    "o": ["о", "ο"],          "p": ["р", "ρ"],
    "s": ["ѕ"],               "t": ["τ"],
    "u": ["υ"],               "v": ["ν"],
    "w": ["ѡ"],               "x": ["х"],
    "y": ["у", "γ"],          "z": ["ᴢ"],
}

ZW_CHARS = ["\u200B", "\u200C", "\u200D", "\uFEFF"]


def _apply_leetspeak(word: str, intensity: Intensity) -> str:
    chars = list(word)
    n = len(chars)
    transform_count = 1 if intensity == "light" else (n // 2 + 1 if intensity == "medium" else n)
    indices = [i for i in range(n) if chars[i].lower() in LEET_MAP]
    random.shuffle(indices)
    for idx in indices[:transform_count]:
        c = chars[idx].lower()
        if c in LEET_MAP:
            chars[idx] = random.choice(LEET_MAP[c])
    return "".join(chars)


def _apply_unicode(word: str, intensity: Intensity) -> str:
    chars = list(word)
    n = len(chars)
    transform_count = 1 if intensity == "light" else (n // 2 + 1 if intensity == "medium" else n)
    indices = [i for i in range(n) if chars[i].lower() in UNICODE_HOMOGLYPHS]
    random.shuffle(indices)
    for idx in indices[:transform_count]:
        c = chars[idx].lower()
        if c in UNICODE_HOMOGLYPHS:
            replacement = random.choice(UNICODE_HOMOGLYPHS[c])
            chars[idx] = replacement.upper() if chars[idx].isupper() else replacement
    return "".join(chars)


def _apply_zwj(word: str, intensity: Intensity) -> str:
    chars = list(word)
    n = len(chars)
    insert_count = 1 if intensity == "light" else (n // 2 if intensity == "medium" else n - 1)
    result = []
    insertions = 0
    for i, c in enumerate(chars):
        result.append(c)
        if i < n - 1 and insertions < insert_count:
            result.append(random.choice(ZW_CHARS))
            insertions += 1
    return "".join(result)


def _apply_mixedcase(word: str, intensity: Intensity) -> str:
    chars = list(word)
    if intensity == "light":
        idx = random.randint(0, len(chars) - 1)
        chars[idx] = chars[idx].upper()
    elif intensity == "medium":
        chars = [c.lower() if i % 2 == 0 else c.upper() for i, c in enumerate(chars)]
    else:
        chars = [c.upper() if random.random() > 0.5 else c.lower() for c in chars]
    return "".join(chars)


def _apply_phonetic(word: str) -> str:
    subs = [
        (r"ph", "f"), (r"ck", "k"), (r"qu", "kw"),
        (r"c(?=[eiy])", "s"), (r"c", "k"),
    ]
    result = word
    for pattern, replacement in subs:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    return result


def _obfuscate_word(word: str, technique: ObfuscationTechnique, intensity: Intensity) -> str:
    if technique == "leetspeak":
        return _apply_leetspeak(word, intensity)
    elif technique == "unicode":
        return _apply_unicode(word, intensity)
    elif technique == "zwj":
        return _apply_zwj(word, intensity)
    elif technique == "mixedcase":
        return _apply_mixedcase(word, intensity)
    elif technique == "phonetic":
        return _apply_phonetic(word)
    elif technique == "random":
        fn = random.choice([_apply_leetspeak, _apply_unicode, _apply_zwj, _apply_mixedcase])
        return fn(word, intensity)
    return word


@dataclass
class Transformation:
    original: str
    transformed: str
    technique: str


@dataclass
class ParseltongueResult:
    original_text: str
    transformed_text: str
    triggers_found: List[str]
    technique_used: str
    transformations: List[Transformation] = field(default_factory=list)


def detect_triggers(text: str, custom_triggers: Optional[List[str]] = None) -> List[str]:
    all_triggers = DEFAULT_TRIGGERS + (custom_triggers or [])
    found = []
    for trigger in all_triggers:
        pattern = re.compile(r"\b" + re.escape(trigger) + r"\b", re.IGNORECASE)
        if pattern.search(text):
            found.append(trigger)
    return list(dict.fromkeys(found))


def apply_parseltongue(
    text: str,
    technique: ObfuscationTechnique = "leetspeak",
    intensity: Intensity = "medium",
    enabled: bool = True,
    custom_triggers: Optional[List[str]] = None,
) -> ParseltongueResult:
    if not enabled:
        return ParseltongueResult(text, text, [], technique)

    triggers_found = detect_triggers(text, custom_triggers)
    if not triggers_found:
        return ParseltongueResult(text, text, [], technique)

    transformed = text
    transformations: List[Transformation] = []

    sorted_triggers = sorted(triggers_found, key=len, reverse=True)
    for trigger in sorted_triggers:
        pattern = re.compile(r"\b(" + re.escape(trigger) + r")\b", re.IGNORECASE)

        def replacer(m, t=trigger, tech=technique, inten=intensity):
            result = _obfuscate_word(m.group(0), tech, inten)
            transformations.append(Transformation(m.group(0), result, tech))
            return result

        transformed = pattern.sub(replacer, transformed)

    return ParseltongueResult(
        original_text=text,
        transformed_text=transformed,
        triggers_found=triggers_found,
        technique_used=technique,
        transformations=transformations,
    )


TECHNIQUE_DESCRIPTIONS: Dict[str, str] = {
    "leetspeak": "L33tspeak الكلاسيكي: a→4, e→3, …",
    "unicode": "حروف Unicode مرئياً مطابقة (سيريلية، يونانية)",
    "zwj": "أحرف عرض-صفر غير مرئية بين الحروف",
    "mixedcase": "أنماط كبتلة مُختلطة مُشوِّشة",
    "phonetic": "استبدال صوتي للحروف",
    "random": "مزيج عشوائي من جميع التقنيات",
}
