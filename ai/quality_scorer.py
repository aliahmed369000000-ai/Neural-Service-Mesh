"""
ai/quality_scorer.py
======================
تقييم جودة الرد تلقائياً بدل الفحص الثنائي القديم (فارغ / غير فارغ).

يقيس ثلاثة أبعاد مستقلة (كل واحد 0-100) ثم يدمجها في درجة إجمالية:
  1. جودة اللغة العربية  (arabic_quality) — نسبة الحروف العربية، عدم
     التكرار الممل للحرف/الكلمة نفسها، طول معقول للكلمات.
  2. التماسك            (coherence)       — تنوّع الكلمات، عدم تكرار
     الجملة نفسها، عدم كون الرد رسالة خطأ/اعتذار قياسية.
  3. الصلة بالسؤال       (relevance)       — تداخل معجمي بسيط بين كلمات
     السؤال والرد (بعد إزالة كلمات الوقف الشائعة).

هذا فحص خفيف (regex + حسابات بسيطة، بدون استدعاء API) مصمم ليعمل ضمن
مسار التوليد الساخن دون إبطاء ملحوظ.

الاستخدام:
    from ai.quality_scorer import score_response
    q = score_response(query, response)
    # q = {"score": 78.4, "coherence": 82.0, "relevance": 65.0,
    #      "arabic_quality": 90.0, "is_quality": True}
"""
from __future__ import annotations

import re
from typing import Dict

# ── إعدادات ────────────────────────────────────────────────────────────────
QUALITY_THRESHOLD = 40.0   # تحت هذه الدرجة يُعتبر الرد "فاشلاً" فعلياً

_ARABIC_RANGE = re.compile(r"[\u0600-\u06FF]")
_WORD_SPLIT   = re.compile(r"[\s\u060C\u061B\u061F.,!?؟،؛\-–—()«»\"':؛]+")

# رسائل خطأ/اعتذار قياسية يجب أن تُخفّض درجة التماسك
_ERROR_PATTERNS = [
    "⚠️", "تعذّر الحصول على رد", "خطأ في", "جاري الإعادة",
    "عذراً", "آسف", "لم أتمكن", "حدث خطأ",
]

# كلمات وقف عربية شائعة تُستبعد عند حساب تداخل الصلة
_STOPWORDS = {
    "من", "في", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "التي",
    "الذي", "و", "أو", "ثم", "لا", "ما", "هل", "كيف", "متى", "أين",
    "لماذا", "كان", "يكون", "أن", "إن", "قد", "كل", "بعض", "غير", "بين",
    "له", "لها", "لهم", "به", "بها", "عليه", "عليها", "الله", "يا",
}


def _tokenize(text: str) -> list:
    return [w for w in _WORD_SPLIT.split(text.strip()) if w]


def _arabic_quality(response: str) -> float:
    """نسبة الحروف العربية + عدم التكرار الممل + طول كلمات معقول."""
    if not response or not response.strip():
        return 0.0

    letters = [c for c in response if c.isalpha()]
    if not letters:
        return 0.0
    arabic_ratio = sum(1 for c in letters if _ARABIC_RANGE.match(c)) / len(letters)

    words = _tokenize(response)
    if not words:
        return 0.0

    # عقوبة التكرار الممل لحرف واحد ممتد (مثل: ااااااا أو ...........)
    repetition_penalty = 0.0
    if re.search(r"(.)\1{6,}", response):
        repetition_penalty = 25.0

    # طول متوسط الكلمة (كلمات قصيرة جداً بشكل ممنهج = رد ركيك)
    avg_word_len = sum(len(w) for w in words) / len(words)
    length_score = 100.0 if avg_word_len >= 2.5 else max(0.0, avg_word_len / 2.5 * 100.0)

    score = (arabic_ratio * 70.0) + (length_score * 0.30) - repetition_penalty
    return max(0.0, min(100.0, score))


def _coherence(response: str) -> float:
    """تنوّع المفردات + غياب أنماط رسائل الخطأ + تكرار الجمل."""
    if not response or not response.strip():
        return 0.0

    for pat in _ERROR_PATTERNS:
        if pat in response:
            return 10.0

    words = _tokenize(response)
    if len(words) < 2:
        return 30.0 if words else 0.0

    unique_ratio = len(set(words)) / len(words)

    # تكرار الجملة نفسها بالكامل عدة مرات
    sentences = [s.strip() for s in re.split(r"[.!؟?\n]+", response) if s.strip()]
    dup_penalty = 0.0
    if sentences and len(sentences) != len(set(sentences)):
        dup_penalty = 20.0

    score = (unique_ratio * 90.0) + 10.0 - dup_penalty
    return max(0.0, min(100.0, score))


def _relevance(query: str, response: str) -> float:
    """تداخل معجمي بسيط بين كلمات السؤال والرد (بعد استبعاد كلمات الوقف)."""
    if not query or not response:
        return 50.0  # لا معلومات كافية — درجة محايدة

    q_words = {w for w in _tokenize(query) if w not in _STOPWORDS and len(w) > 1}
    r_words = {w for w in _tokenize(response) if w not in _STOPWORDS and len(w) > 1}

    if not q_words:
        return 60.0  # سؤال قصير جداً/بلا كلمات دلالية — لا نعاقب الرد

    overlap = len(q_words & r_words)
    ratio = overlap / len(q_words)

    # رد أطول بكثير من السؤال يعطي فرصة أكبر للتداخل حتى لو الصلة ضعيفة
    # لذلك نعزز الدرجة الدنيا بدل الصفر الكامل عند غياب التداخل
    base = 35.0 if len(r_words) > 15 else 20.0
    return max(0.0, min(100.0, base + ratio * 65.0))


def score_response(query: str, response: str) -> Dict[str, float]:
    """
    يحسب درجة جودة إجمالية للرد [0-100] + الأبعاد الفرعية.

    الوزن: تماسك 40% + جودة لغة 35% + صلة 25%
    """
    # رسائل خطأ/اعتذار قياسية تُعتبر فشلاً فعلياً بغض النظر عن سلامة اللغة
    for pat in _ERROR_PATTERNS:
        if response and pat in response:
            return {
                "score": 15.0, "coherence": 10.0, "relevance": 0.0,
                "arabic_quality": 0.0, "is_quality": False,
            }

    aq  = _arabic_quality(response)
    coh = _coherence(response)
    rel = _relevance(query, response)

    total = round(coh * 0.40 + aq * 0.35 + rel * 0.25, 2)

    return {
        "score":          total,
        "coherence":      round(coh, 2),
        "relevance":      round(rel, 2),
        "arabic_quality": round(aq, 2),
        "is_quality":     total >= QUALITY_THRESHOLD,
    }
