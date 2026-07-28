"""
Response Quality — مقيّم جودة استكشافي (heuristic) للردود العربية
====================================================================
لا يعتمد هذا المقيّم على أي نموذج ML أو استدعاء شبكي؛ كل الفحوصات محلية
بحتة (تعبيرات نمطية regex + إحصاءات نصية بسيطة)، ليُستخدَم فوراً بجانب
كل رد يُعرَض للمستخدم بدون أي تكلفة أو زمن استجابة إضافي محسوس.

المعايير المُقيَّمة لكل رد:
  • length        — طول الرد (عقوبة على الفراغ أو القِصَر المفرط أو الإطالة).
  • arabic_ratio   — نسبة الأحرف العربية (NSM تطبيق عربي بالكامل، فمزيج
                     لغوي كثيف أو نص أعجمي غالب يُعتبر مؤشر جودة سيئ).
  • repetition     — تكرار جمل داخل نفس الرد (مؤشر حلقة توليد معطوبة).
  • relevance      — تطابق كلمات مفتاحية بين السؤال والرد (اختياري، يُحسب
                     فقط إذا مُرِّر نص السؤال).
  • error_markers  — وجود مؤشرات خطأ صريحة داخل نص الرد نفسه (مثل
                     '⚠️ خطأ: ...') — يُسقِط التقييم الإجمالي فوراً.
  • refusal        — رفض/تهرّب عام قصير بلا أي محتوى فعلي.

الاستخدام:
    from ai.response_quality import score_response
    q = score_response(response_text, query="سؤال المستخدم")
    print(q.as_percent(), q.label, q.issues)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ── كلمات وقف عربية شائعة (تُستبعَد من حساب درجة الصلة relevance) ────────
ARABIC_STOPWORDS = frozenset({
    "في", "من", "على", "إلى", "عن", "مع", "هذا", "هذه", "ذلك", "تلك",
    "التي", "الذي", "الذين", "او", "أو", "و", "ثم", "قد", "لقد", "كان",
    "كانت", "يكون", "أن", "ان", "إن", "لا", "لم", "لن", "ما", "هل", "كل",
    "بعض", "غير", "بين", "عند", "حتى", "لكن", "ايضا", "أيضا", "أيضاً",
    "كما", "حيث", "اذا", "إذا", "لو", "كي", "لكي", "قبل", "بعد", "فوق",
    "تحت", "امام", "أمام", "خلف", "هو", "هي", "هم", "انت", "أنت",
    "انتم", "أنتم", "انا", "أنا", "نحن", "له", "لها", "لهم", "منه",
    "فيه", "عليه", "ذلك", "هناك", "الى",
})

# ── أنماط تدل على فشل/خطأ صريح داخل نص الرد نفسه ─────────────────────────
ERROR_MARKERS: Tuple[str, ...] = (
    "⚠️ خطأ", "⚠️ تعذّر", "حدث خطأ", "فشل الاتصال", "traceback (most recent",
    "nonetype", "attributeerror", "keyerror", "connectionerror",
)

# ── أنماط رفض/تهرّب عام بلا محتوى فعلي ──────────────────────────────────
REFUSAL_PATTERNS: Tuple[str, ...] = (
    "لا أملك معلومات كافية", "لا املك معلومات كافية",
    "لا أستطيع الإجابة", "لا استطيع الاجابة",
    "لا يمكنني مساعدتك", "عذراً، لا", "عذرا لا", "آسف، لا",
)

_ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")
_WORD_RE = re.compile(r"[\u0621-\u064A\u0660-\u0669A-Za-z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"[.!؟?\n]+")

# أوزان دمج المعايير في الدرجة الإجمالية. relevance تُستبعَد تلقائياً (مع
# إعادة توزيع باقي الأوزان نسبياً) إن لم يُمرَّر نص سؤال.
_WEIGHTS: Dict[str, float] = {
    "length": 0.10,
    "arabic_ratio": 0.15,
    "repetition": 0.10,
    "relevance": 0.45,
    "error_markers": 0.10,
    "refusal": 0.10,
}


@dataclass
class QualityScore:
    """نتيجة تقييم جودة رد واحد."""

    overall: float                                     # 0.0 - 1.0
    label: str                                          # ممتاز/جيد/متوسط/ضعيف
    breakdown: Dict[str, float] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)

    def as_percent(self) -> int:
        return round(self.overall * 100)

    def to_dict(self) -> dict:
        return {
            "overall": round(self.overall, 3),
            "percent": self.as_percent(),
            "label": self.label,
            "breakdown": {k: round(v, 3) for k, v in self.breakdown.items()},
            "issues": self.issues,
        }


def _label_for(score: float) -> str:
    if score >= 0.85:
        return "ممتاز"
    if score >= 0.65:
        return "جيد"
    if score >= 0.40:
        return "متوسط"
    return "ضعيف"


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text or "") if len(w) >= 2]


def _score_length(text: str) -> Tuple[float, Optional[str]]:
    n = len(text.strip())
    if n == 0:
        return 0.0, "الرد فارغ تماماً"
    if n < 8:
        return 0.15, "الرد قصير جداً (أقل من 8 أحرف)"
    if n < 25:
        return 0.55, "الرد قصير نسبياً وقد يفتقر للتفصيل"
    if n > 6000:
        return 0.6, "الرد طويل جداً (قد يحتوي حشواً أو تكراراً)"
    return 1.0, None


def _score_arabic_ratio(text: str) -> Tuple[float, Optional[str]]:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        # نص بلا أحرف إطلاقاً (أرقام/رموز/كود فقط) — لا نعاقب بشدة، فقد
        # يكون رداً برمجياً متعمَّداً (مقتطف كود مثلاً).
        return 0.6, None
    arabic = sum(1 for c in letters if _ARABIC_CHAR_RE.match(c))
    ratio = arabic / len(letters)
    if ratio >= 0.55:
        return 1.0, None
    if ratio >= 0.30:
        return 0.6, "مزيج لغوي ملحوظ (عربي/غير عربي) قد يقلّل الوضوح"
    return 0.2, "الرد غالبه بغير العربية رغم أن NSM تطبيق عربي بالكامل"


def _score_repetition(text: str) -> Tuple[float, Optional[str]]:
    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    if len(sentences) < 2:
        return 1.0, None
    seen: Dict[str, int] = {}
    dup = 0
    for s in sentences:
        key = s.lower()
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > 1:
            dup += 1
    ratio = dup / len(sentences)
    if ratio == 0:
        return 1.0, None
    if ratio < 0.2:
        return 0.8, None
    if ratio < 0.4:
        return 0.5, "تكرار ملحوظ لجمل داخل نفس الرد"
    return 0.2, "تكرار كثيف لجمل داخل نفس الرد (يُحتمَل خلل في التوليد)"


def _score_relevance(text: str, query: str) -> Tuple[Optional[float], Optional[str]]:
    if not query or not query.strip():
        return None, None
    q_words = {w for w in _tokenize(query) if w not in ARABIC_STOPWORDS}
    if not q_words:
        return None, None
    r_words = set(_tokenize(text))
    overlap = len(q_words & r_words)
    ratio = overlap / len(q_words)
    if ratio >= 0.5:
        return 1.0, None
    if ratio >= 0.25:
        return 0.7, None
    if ratio > 0:
        return 0.4, "تطابق كلمات ضعيف مع السؤال — الرد قد يكون غير مرتبط تماماً"
    return 0.1, "لا يوجد أي تطابق كلمات مع السؤال — يُحتمَل أن الرد غير مرتبط"


def _score_error_markers(text: str) -> Tuple[float, Optional[str]]:
    low = text.lower()
    for marker in ERROR_MARKERS:
        if marker.lower() in low:
            return 0.0, f"الرد يحتوي مؤشر خطأ صريح ('{marker}')"
    return 1.0, None


def _score_refusal(text: str) -> Tuple[float, Optional[str]]:
    for pattern in REFUSAL_PATTERNS:
        if pattern in text:
            # رفض قصير بلا أي محتوى إضافي أسوأ من رفض ضمن رد أطول يشرح البديل
            if len(text.strip()) < 120:
                return 0.35, "الرد أقرب لرفض/تهرّب عام بلا محتوى فعلي"
            return 0.75, None
    return 1.0, None


def score_response(response: str, query: str = "") -> QualityScore:
    """
    يقيّم جودة رد عربي بشكل استكشافي (heuristic) محلي بالكامل، بدون أي
    استدعاء شبكي أو نموذج ML. مناسب للعرض الفوري بجانب كل رد وكيل.

    Args:
        response: نص الرد المُراد تقييمه.
        query: نص سؤال/مهمة المستخدم الأصلي (اختياري) — يُستخدَم فقط
               لحساب درجة الصلة (relevance). إن تُرِك فارغاً، تُستبعَد
               درجة الصلة من الحساب الإجمالي وتُعاد توزيع بقية الأوزان
               نسبياً بدلاً منها.

    Returns:
        QualityScore: overall (0-1)، label (تصنيف نصي مختصر)، breakdown
        (تفصيل كل معيار على حدة)، issues (ملاحظات بالعربية إن وُجدت).
    """
    text = response or ""

    # حالة خاصة: رد فارغ تماماً (أو فراغات فقط) — بقية المعايير (تكرار،
    # رفض، مؤشرات خطأ) تُعطي نتيجة "مثالية" زوراً على نص فارغ (لا جمل
    # لتكرارها، لا نمط رفض فيها، لا مؤشر خطأ صريح...)، فلا معنى لحساب
    # متوسط مرجّح معها؛ نُقصّر الدائرة هنا مباشرة.
    if not text.strip():
        return QualityScore(
            overall=0.0,
            label=_label_for(0.0),
            breakdown={"length": 0.0},
            issues=["الرد فارغ تماماً"],
        )

    checks: Dict[str, Tuple[float, Optional[str]]] = {
        "length": _score_length(text),
        "arabic_ratio": _score_arabic_ratio(text),
        "repetition": _score_repetition(text),
        "error_markers": _score_error_markers(text),
        "refusal": _score_refusal(text),
    }
    rel_score, rel_issue = _score_relevance(text, query)
    if rel_score is not None:
        checks["relevance"] = (rel_score, rel_issue)

    breakdown: Dict[str, float] = {}
    issues: List[str] = []
    for name, (val, issue) in checks.items():
        breakdown[name] = val
        if issue:
            issues.append(issue)

    active_weights = {k: _WEIGHTS[k] for k in checks}
    total_w = sum(active_weights.values())
    overall = (
        sum(checks[k][0] * active_weights[k] for k in checks) / total_w
        if total_w else 0.0
    )

    # مؤشر خطأ صريح يُسقط التقييم الإجمالي بغض النظر عن باقي المعايير —
    # لا معنى لتصنيف رد "متوسط الجودة" إن كان أصلاً رسالة خطأ.
    if breakdown["error_markers"] == 0.0:
        overall = min(overall, 0.1)

    overall = max(0.0, min(1.0, overall))
    return QualityScore(
        overall=overall, label=_label_for(overall), breakdown=breakdown, issues=issues,
    )
